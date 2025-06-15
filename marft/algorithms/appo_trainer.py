import os
import torch
import torch.nn as nn
import numpy as np
from abc import ABC
from marft.mas import MAS
from marft.buffers import ActionBuffer
from marft.utils.util import get_gard_norm, huber_loss, mse_loss, to_cuda


class APPOTrainer(ABC):

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

    def cal_policy_loss(self, log_prob_infer: torch.Tensor, log_prob_batch: torch.Tensor, advantages_batch: torch.Tensor, entropy: torch.Tensor):

        log_ratio = log_prob_infer - log_prob_batch
        imp_weights = torch.exp(log_ratio)
        approx_kl = ((imp_weights - 1) - log_ratio).mean()
        surr1 = -torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        surr2 = -imp_weights * advantages_batch
        surr = torch.max(surr1, surr2)
        policy_loss = surr.mean() - self.entropy_coef * entropy.mean()
        return policy_loss, approx_kl

    def cal_policy_loss(self, log_prob_infer: torch.Tensor, log_prob_batch: torch.Tensor, advantages_batch: torch.Tensor, entropy: torch.Tensor):

        log_ratio = log_prob_infer - log_prob_batch
        imp_weights = torch.exp(log_ratio)
        approx_kl = ((imp_weights - 1) - log_ratio).mean()
        surr1 = -torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        surr2 = -imp_weights * advantages_batch
        surr = torch.max(surr1, surr2)
        policy_loss = surr.mean() - self.entropy_coef * entropy.mean()
        return policy_loss, approx_kl

    def cal_value_loss(self, values_infer: torch.Tensor, value_preds_batch: torch.Tensor, return_batch: torch.Tensor):

        value_pred_clipped = value_preds_batch + (values_infer - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        error_clipped = return_batch - value_pred_clipped
        error_unclipped = return_batch - values_infer
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_unclipped = huber_loss(error_unclipped, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_unclipped = mse_loss(error_unclipped)
        value_loss = torch.max(value_loss_clipped, value_loss_unclipped).mean()
        return value_loss * self.value_loss_coef

    def ppo_update(self, sample, global_steps: int):

        agent_to_train = None
        if self.agent_iteration_interval > 0:
            time_slice = global_steps // self.agent_iteration_interval
            agent_to_train = time_slice % self.num_agent

        observations, actions, rollout_observations, log_probs, value_preds, \
            returns, advantages, action_tokens = sample
        
        advantages_copy = advantages.copy()
        advantages_copy[advantages_copy == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-8)

        actions, rollout_observations, log_probs, value_preds, returns, advantages, action_tokens = \
            to_cuda((actions, rollout_observations, log_probs, value_preds, returns, advantages, action_tokens))
        
        batch_size = rollout_observations.shape[0]
        cp_batch_size = int(batch_size // self.gradient_cp_steps)
        if cp_batch_size == 0:
            print(f"gradient_cp_steps > batch_size, set cp_batch_size = 1")
            cp_batch_size = 1

        # critic update with checkpoint gradient accumulation
        self.critic_optimizer.zero_grad()
        value_loss = 0
        for start in range(0, batch_size, cp_batch_size):
            end = start + cp_batch_size
            if end > batch_size:
                end = batch_size
            cp_weight = (end - start) / batch_size  # Weight for the chunk loss
            cp_obs_batch, cp_value_preds_batch, cp_returns_batch = \
                rollout_observations[start:end], value_preds[start:end], returns[start:end]
            values_infer = self.mas.get_action_values(cp_obs_batch)
            cp_value_loss = self.cal_value_loss(values_infer, cp_value_preds_batch, cp_returns_batch)
            cp_value_loss *= cp_weight  # Scale the loss by the chunk weight
            cp_value_loss.backward()
            value_loss += cp_value_loss.item()
        # Gradient clipping
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.mas.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.mas.critic.parameters())
        self.critic_optimizer.step()
        critic_grad_norm = critic_grad_norm.item()

        if global_steps < self.warmup_steps:
            return value_loss, critic_grad_norm, 0, 0, 0, 0
        
        # policy update
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
            cp_obs_batch, cp_act_batch, cp_adv_batch, cp_log_probs_batch = \
                rollout_observations[start:end], action_tokens[start:end], advantages[start:end], log_probs[start:end]
            log_prob_infer, cp_entropy = self.mas.get_joint_action_log_probs(cp_obs_batch, cp_act_batch, agent_to_train)
            if agent_to_train is not None:
                cp_log_probs_batch = cp_log_probs_batch[:, agent_to_train: agent_to_train + 1]
                cp_adv_batch = cp_adv_batch[:, agent_to_train: agent_to_train + 1]
            cp_policy_loss, approx_kl = self.cal_policy_loss(log_prob_infer, cp_log_probs_batch, cp_adv_batch, cp_entropy)
            total_approx_kl += approx_kl * cp_weight
            total_entropy += cp_entropy.mean().item() * cp_weight
            cp_policy_loss = cp_policy_loss * cp_weight
            cp_policy_loss.backward()
            policy_loss += cp_policy_loss.item()
        if total_approx_kl > 1e-3: # adjust to the real situation
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

        return value_loss, critic_grad_norm, policy_loss, total_policy_grad_norm, total_approx_kl, total_entropy

    def train(self, buffer: ActionBuffer, global_steps: int):
        """
        Perform a training update using minibatch GD.
        :param buffer: (ActionBuffer) buffer containing training data.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {
            "value_loss": 0,
            "value_grad_norm": 0,
            "policy_loss": 0,
            "policy_grad_norm": 0,
            "approx_kl": 0,
            "entropy": 0,
        }

        update_time = 0
        for _ in range(self.ppo_epoch):
            data_generator = buffer.sample(self.num_mini_batch)
            for sample in data_generator:
                value_loss, value_grad_norm, policy_loss, policy_grad_norm, approx_kl, entropy = self.ppo_update(sample, global_steps)
                train_info["value_loss"] += value_loss
                train_info["value_grad_norm"] += value_grad_norm
                train_info["policy_loss"] += policy_loss
                train_info["policy_grad_norm"] += policy_grad_norm
                train_info["approx_kl"] += approx_kl
                train_info["entropy"] += entropy
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
        print(f"[APPOTrainer] optimizer states saved -> {exp_path}")

    def load_optimizers(self, path: str, map_location: str | torch.device = "cpu"):
        ckpt = torch.load(path, map_location=map_location)
        for role, opt_state in ckpt["policy_opt_states"].items():
            # The trainer’s __init__ already created the corresponding optimizer.
            self.policy_optimizer[role].load_state_dict(opt_state)
        self.critic_optimizer.load_state_dict(ckpt["critic_opt_state"])
        print(f"[APPOTrainer] optimizer states loaded <- {path}")

    def prep_training(self):
        for agent in self.mas.agents:
            agent.train()
        self.mas.critic.train()

    def prep_rollout(self):
        for agent in self.mas.agents:
            agent.eval()
        self.mas.critic.eval()