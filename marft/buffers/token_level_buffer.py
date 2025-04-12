import numpy as np
from .base_buffer import BaseBuffer

class TokenBuffer(BaseBuffer):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param pad_token_id: (int) padding token id.
    """

    def __init__(self, args, num_agents, pad_token_id):
        super().__init__(args, num_agents)
        # for token-level preservations
        self.tppo_values = np.zeros((self.max_batch, self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.max_new_tokens), dtype=np.float32)
        self.tppo_returns = np.zeros((self.max_batch, self.episode_length, self.n_rollout_threads, self.num_agents, self.max_new_tokens), dtype=np.float32)
        self.tppo_advantages = np.zeros_like(self.tppo_returns)
        self.tppo_log_probs = np.zeros_like(self.tppo_returns)
        self.pad_token_id = pad_token_id

    def insert(self, next_obs, actions, rollout_obs, value_preds, rewards, masks, action_tokens, log_probs):
        self.obs[self.cur_batch_index, self.step + 1] = next_obs.copy()
        self.actions[self.cur_batch_index, self.step] = actions.copy()
        self.rollout_obs[self.cur_batch_index, self.step] = rollout_obs.copy()
        self.rewards[self.cur_batch_index, self.step] = rewards.copy()
        self.masks[self.cur_batch_index, self.step + 1] = masks.copy()
        self.action_tokens[self.cur_batch_index, self.step] = action_tokens.copy()
        self.tppo_values[self.cur_batch_index, self.step] = value_preds.copy()
        self.tppo_log_probs[self.cur_batch_index, self.step] = log_probs.copy()
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.pre_batch_index = self.cur_batch_index
        self.cur_batch_index = (self.cur_batch_index + 1) % self.max_batch
        self.obs[self.cur_batch_index, 0] = self.obs[self.pre_batch_index, -1].copy()

    def get_last_token_position(self, action_tokens) -> int:
        """
        Given the action tokens, return the last token position.

        Args:
            action_tokens: (torch.Tensor): (max_new_tokens)

        Return:
            last_token_position: (torch.Tensor): int
        """
        pos = len(action_tokens) - 1
        while action_tokens[pos] == self.pad_token_id: pos -= 1
        return pos

    def compute_gae_and_returns(self, next_value):
        self.tppo_values[self.cur_batch_index, -1, :, :, 0] = next_value
        for thread in range(self.n_rollout_threads):
            gae = 0
            for step in reversed(range(self.episode_length)):
                for agent in reversed(range(self.num_agents)):
                    last_token = self.get_last_token_position(self.action_tokens[self.cur_batch_index, step, thread, agent, :])
                    for token in reversed(range(last_token + 1)):
                        rew = self.rewards[self.cur_batch_index, step, thread, agent]
                        v = self.tppo_values[self.cur_batch_index, step, thread, agent, token]
                        if agent == self.num_agents - 1:
                            if token == last_token:
                                v_next = self.tppo_values[self.cur_batch_index, step + 1, thread, 0, 0]
                                mask_next = self.masks[self.cur_batch_index, step + 1, thread, 0]
                                delta = rew + self.gamma * v_next * mask_next - v
                                gae = delta + self.gamma * self.gae_lambda * mask_next * gae
                            else:
                                v_next = self.tppo_values[self.cur_batch_index, step, thread, agent, token + 1]
                                delta = self.gamma * v_next - v
                                gae = delta + self.gamma * self.gae_lambda * gae
                        else:
                            if token == last_token:
                                v_next = self.tppo_values[self.cur_batch_index, step, thread, agent + 1, 0]
                                mask_next = self.masks[self.cur_batch_index, step, thread, agent + 1]
                                delta = rew + self.gamma * v_next * mask_next - v
                                gae = delta + self.gamma * self.gae_lambda * mask_next * gae
                            else:
                                v_next = self.tppo_values[self.cur_batch_index, step, thread, agent, token + 1]
                                delta = self.gamma * v_next - v
                                gae = delta + self.gamma * self.gae_lambda * gae
                        self.tppo_returns[self.cur_batch_index, step, thread, agent, token] = gae + v
                        self.tppo_advantages[self.cur_batch_index, step, thread, agent, token] = gae
        self.cur_num_batch = self.cur_num_batch + 1 if self.cur_num_batch < self.max_batch else self.max_batch

    def sample(self, num_mini_batch: int = None, mini_batch_size: int = None):
        """
        Yield training data for TPPO.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        batch_size = self.n_rollout_threads * self.episode_length * self.cur_num_batch
        # num_mini_batch is the number of mini batches to split per single batch into thus should multiply cur_num_batch
        num_mini_batch *= self.cur_num_batch

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch
            mini_batch_size = batch_size // num_mini_batch

        rand = np.arange(batch_size)
        np.random.shuffle(rand)
        sampler = [rand[i * mini_batch_size : (i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        # keep (num_agent, (max_new_tokens))
        obs = self.obs[:, :-1].reshape(-1, *self.obs.shape[3:])
        actions = self.actions.reshape(-1, *self.actions.shape[3:])
        rollout_obs = self.rollout_obs[:, :-1].reshape(-1, *self.rollout_obs.shape[3:])
        value_preds = self.tppo_values[:, :-1].reshape(-1, *self.tppo_values.shape[3:])
        returns = self.tppo_returns.reshape(-1, *self.tppo_returns.shape[3:])
        advantages = self.tppo_advantages.reshape(-1, *self.tppo_advantages.shape[3:])
        log_prob = self.tppo_log_probs.reshape(-1, *self.tppo_log_probs.shape[3:])
        action_tokens = self.action_tokens.reshape(-1, *self.action_tokens.shape[3:])

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            # value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            # return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            # o_a_embd_batch = o_a_embds[indices].reshape(-1, *o_a_embds.shape[2:])
            obs_batch = obs[indices]
            action_batch = actions[indices]
            rollout_obs_batch = rollout_obs[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            advantages_batch = advantages[indices]
            log_prob_batch = log_prob[indices]
            action_tokens_batch = action_tokens[indices]
            yield obs_batch, action_batch, rollout_obs_batch, log_prob_batch, value_preds_batch, return_batch, advantages_batch, action_tokens_batch
