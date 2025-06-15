import os
import numpy as np
from abc import ABC
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from marft.utils.util import get_gard_norm, huber_loss, mse_loss, to_cuda
from marft.buffers import TokenBuffer
from marft.mas import MAS


class TPPOTrainer(ABC):


    def __init__(self, args, mas: MAS):
        self.mas = mas
        self.num_agent = mas.num_agents
        self.warmup_steps = args.warmup_steps
        self.agent_iteration_interval = args.agent_iteration_interval
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.entropy_coef = args.entropy_coef
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.gradient_cp_steps = args.gradient_cp_steps

        self.policy_optimizer = {}
        for agent in self.mas.agents:
            self.policy_optimizer[agent.role] = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, agent.parameters()),
                lr=self.lr,
                eps=1e-5,
                weight_decay=0,
            )
        self.critic_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.mas.critic.parameters()), lr=self.critic_lr, eps=1e-5)
        
        if args.load_path is not None:
            self.load_optimizers(os.path.join(args.load_path, "optimizers.pt"), map_location="cpu")
 
    def cal_token_mask(self, action_tokens_batch):
        pad_token = self.mas.tokenizer.pad_token_id
        token_mask = (action_tokens_batch != pad_token).int()
        return token_mask

    def cal_policy_loss(self, log_prob_infer, log_prob_batch, advantages_batch, entropy, token_mask):

        log_ratio = log_prob_infer - log_prob_batch
        imp_weights = torch.exp(log_ratio)
        approx_kl = (imp_weights - 1) - log_ratio
        surr1 = -torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        surr2 = -imp_weights * advantages_batch
        surr = torch.max(surr1, surr2)
        policy_loss = surr - self.entropy_coef * entropy

        policy_loss = (policy_loss * token_mask).sum() / token_mask.sum()
        approx_kl = (approx_kl * token_mask).sum() / token_mask.sum()
        entropy_value = (entropy * token_mask).sum() / token_mask.sum()

        return policy_loss, approx_kl, entropy_value

    def cal_value_loss(self, values_infer, value_preds_batch, return_batch, token_mask):

        value_pred_clipped = value_preds_batch + (values_infer - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        error_clipped = return_batch - value_pred_clipped
        error_unclipped = return_batch - values_infer
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_unclipped = huber_loss(error_unclipped, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_unclipped = mse_loss(error_unclipped)
        value_loss = torch.max(value_loss_clipped, value_loss_unclipped)
        value_loss = (value_loss * token_mask).sum() / token_mask.sum()

        return value_loss * self.value_loss_coef

    def ppo_update(self, sample, global_steps: int):

        agent_to_train = None
        if self.agent_iteration_interval > 0:
            time_slice = global_steps // self.agent_iteration_interval
            agent_to_train = (time_slice + 1) % self.num_agent

        observations, actions, rollout_observations, log_probs, value_preds, \
            returns, advantages, action_tokens = sample
            
        advantages_copy = advantages.copy()
        advantages_copy[advantages_copy == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-8)

        actions, rollout_observations, log_probs, value_preds, returns, advantages, action_tokens = \
            to_cuda((actions, rollout_observations, log_probs, value_preds, returns, advantages, action_tokens))
        token_mask = self.cal_token_mask(action_tokens)

        batch_size = rollout_observations.shape[0]
        cp_batch_size = int(batch_size // self.gradient_cp_steps)
        if cp_batch_size == 0:
            print(f"gradient_cp_steps > batch_size, set cp_batch_size = 1")
            cp_batch_size = 1

        # critic update
        # torch.cuda.empty_cache()
        self.critic_optimizer.zero_grad()
        value_loss = 0
        for start in range(0, batch_size, cp_batch_size):
            end = start + cp_batch_size
            if end > batch_size:
                end = batch_size
            cp_weight = (end - start) / batch_size
            cp_obs_batch, cp_action_tokens_batch, cp_value_preds_batch, cp_returns_batch, cp_token_mask = \
                rollout_observations[start:end], action_tokens[start:end], value_preds[start:end], returns[start:end], token_mask[start:end]
            values_infer = self.mas.get_token_values(cp_obs_batch, cp_action_tokens_batch, train=True).squeeze(-1)
            cp_value_loss = self.cal_value_loss(values_infer, cp_value_preds_batch, cp_returns_batch, cp_token_mask)
            cp_value_loss *= cp_weight
            cp_value_loss.backward()
            value_loss += cp_value_loss.item()
            # torch.cuda.empty_cache()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.mas.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.mas.critic.parameters())
        self.critic_optimizer.step()
        critic_grad_norm = critic_grad_norm.item()

        if global_steps < self.warmup_steps:
            return value_loss, critic_grad_norm, 0, 0, 0, 0
        
        # policy update
        # torch.cuda.empty_cache()
        for optimizer in self.policy_optimizer.values(): optimizer.zero_grad()
        total_approx_kl = 0.
        total_entropy = 0.
        policy_loss = 0.
        total_policy_grad_norm = 0.
        for start in range(0, batch_size, cp_batch_size):
            end = start + cp_batch_size
            if end > batch_size:
                end = batch_size
            cp_weight = (end - start) / batch_size
            cp_obs_batch, cp_action_tokens_batch, cp_adv_batch, cp_log_prob_batch, cp_token_mask = \
                rollout_observations[start:end], action_tokens[start:end], advantages[start:end], log_probs[start:end], token_mask[start:end]
            logits_infer, _ = self.mas.get_token_logits(cp_obs_batch, cp_action_tokens_batch, agent_to_train) # (cp_batch_size, num_agents, vocab_size)
            pi_log_prob = torch.log_softmax(logits_infer, dim=-1)
            if agent_to_train is not None:
                cp_action_tokens_batch = cp_action_tokens_batch[:, agent_to_train: agent_to_train + 1]
                cp_log_prob_batch = cp_log_prob_batch[:, agent_to_train: agent_to_train + 1]
                cp_adv_batch = cp_adv_batch[:, agent_to_train: agent_to_train + 1]
                cp_token_mask = cp_token_mask[:, agent_to_train: agent_to_train + 1]
            log_prob_infer = torch.gather(pi_log_prob, -1, cp_action_tokens_batch.unsqueeze(-1)).squeeze(-1)
            entropy = Categorical(logits=logits_infer).entropy()
            cp_policy_loss, approx_kl, cp_entropy = self.cal_policy_loss(log_prob_infer, cp_log_prob_batch, cp_adv_batch, entropy, cp_token_mask)
            total_approx_kl += approx_kl.item() * cp_weight
            total_entropy += cp_entropy.item() * cp_weight
            cp_policy_loss *= cp_weight
            cp_policy_loss.backward()
            policy_loss += cp_policy_loss.item()
            # torch.cuda.empty_cache()
        if total_approx_kl > 1.7e-6: # adjust to the real situation
            return value_loss, critic_grad_norm, 0, 0, total_approx_kl, total_entropy

        if agent_to_train is not None:
            agent = self.mas.agents[agent_to_train]
            policy_grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
            self.policy_optimizer[agent.role].step()
            total_policy_grad_norm = policy_grad_norm.item()
        else:
            for agent in self.mas.agents:
                policy_grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
                self.policy_optimizer[agent.role].step()
                total_policy_grad_norm += policy_grad_norm.item()

        return value_loss, critic_grad_norm, policy_loss, policy_grad_norm, total_approx_kl, total_entropy

    def train(self, buffer: TokenBuffer, global_steps: int):
        """
        Perform a training update using minibatch GD.
        :param buffer: (TokenBuffer) buffer containing training data.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {
            "value_loss": 0.,
            "value_grad_norm": 0.,
            "policy_loss": 0.,
            "policy_grad_norm": 0.,
            "entropy": 0.,
            "approx_kl": 0.,
        }

        update_time = 0
        for _ in range(self.ppo_epoch):
            data_generator = buffer.sample(self.num_mini_batch)
            for sample in data_generator:
                value_loss, value_grad_norm, policy_loss, policy_grad_norm, approx_kl, entropy = (
                    self.ppo_update(sample, global_steps)
                )
                train_info["value_loss"] += value_loss
                train_info["value_grad_norm"] += value_grad_norm
                train_info["policy_loss"] += policy_loss
                train_info["policy_grad_norm"] += policy_grad_norm
                train_info["entropy"] += entropy
                train_info["approx_kl"] += approx_kl
                update_time += 1

        for k in train_info.keys():
            train_info[k] /= update_time

        return train_info
    
    def save_optimizers(self, save_dir: str, steps: int) -> None:
        exp_path = os.path.join(save_dir, "steps_{:04d}".format(steps))
        os.makedirs(exp_path, exist_ok=True)
        torch.save(
            {
                "policy_opt_states": {
                    role: opt.state_dict()
                    for role, opt in self.policy_optimizer.items()
                },
                "critic_opt_state": self.critic_optimizer.state_dict(),
            },
            os.path.join(exp_path, f"optimizers.pt"),
        )
        print(f"[TPPOTrainer] optimizer states saved -> {exp_path}")

    def load_optimizers(self, path: str, map_location: str | torch.device = "cpu"):
        ckpt = torch.load(path, map_location=map_location)
        for role, opt_state in ckpt["policy_opt_states"].items():
            # The trainer’s __init__ already created the corresponding optimizer.
            self.policy_optimizer[role].load_state_dict(opt_state)
        self.critic_optimizer.load_state_dict(ckpt["critic_opt_state"])
        print(f"[TPPOTrainer] optimizer states loaded <- {path}")

    def prep_training(self):
        for agent in self.mas.agents:
            agent.train()
        self.mas.critic.train()

    def prep_rollout(self):
        for agent in self.mas.agents:
            agent.eval()
        self.mas.critic.eval()