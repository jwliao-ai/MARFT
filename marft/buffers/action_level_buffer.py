import numpy as np
from .base_buffer import BaseBuffer

class ActionBuffer(BaseBuffer):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    """

    def __init__(self, args, num_agents):
        super().__init__(args, num_agents)
        # action-level preservations
        self.action_level_v_values = np.zeros((self.max_batch, self.episode_length + 1, self.n_rollout_threads, self.num_agents), dtype=np.float32)
        self.action_level_returns = np.zeros((self.max_batch, self.episode_length, self.n_rollout_threads, self.num_agents), dtype=np.float32)
        self.action_level_advantages = np.zeros_like(self.action_level_returns)
        self.action_level_log_probs = np.zeros_like(self.action_level_returns)

    def insert(self, next_obs, actions, rollout_obs, value_preds, rewards, masks, action_tokens, log_probs):
        self.obs[self.cur_batch_index, self.step + 1] = next_obs.copy()
        self.actions[self.cur_batch_index, self.step] = actions.copy()
        self.rollout_obs[self.cur_batch_index, self.step] = rollout_obs.copy()
        self.rewards[self.cur_batch_index, self.step] = rewards.copy()
        self.masks[self.cur_batch_index, self.step + 1] = masks.copy()
        self.action_tokens[self.cur_batch_index, self.step] = action_tokens.copy()
        self.action_level_v_values[self.cur_batch_index, self.step] = value_preds.copy()
        self.action_level_log_probs[self.cur_batch_index, self.step] = log_probs.copy()
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.pre_batch_index = self.cur_batch_index
        self.cur_batch_index = (self.cur_batch_index + 1) % self.max_batch
        self.obs[self.cur_batch_index, 0] = self.obs[self.pre_batch_index, -1].copy()

    def compute_gae_and_returns(self, next_value):
        self.action_level_v_values[self.cur_batch_index, -1] = next_value
        gae = 0
        for step in reversed(range(self.episode_length)):
            for agent in reversed(range(self.num_agents)):
                if agent == self.num_agents - 1:
                    delta = self.rewards[self.cur_batch_index, step, :, agent] \
                        + self.gamma * self.action_level_v_values[self.cur_batch_index, step + 1, :, 0] * self.masks[self.cur_batch_index, step + 1, :, 0] \
                        - self.action_level_v_values[self.cur_batch_index, step, :, agent]
                    gae = delta + self.gamma * self.gae_lambda * self.masks[self.cur_batch_index, step + 1, :, 0] * gae
                else:
                    delta = self.rewards[self.cur_batch_index, step, :, agent] \
                        + self.gamma * self.action_level_v_values[self.cur_batch_index, step, :, agent + 1] * self.masks[self.cur_batch_index, step, :, agent + 1] \
                        - self.action_level_v_values[self.cur_batch_index, step, :, agent]
                    gae = delta + self.gamma * self.gae_lambda * self.masks[self.cur_batch_index, step, :, agent + 1] * gae
                self.action_level_returns[self.cur_batch_index, step, :, agent] = self.action_level_v_values[self.cur_batch_index, step, :, agent] + gae
                self.action_level_advantages[self.cur_batch_index, step, :, agent] = gae
        self.cur_num_batch = self.cur_num_batch + 1 if self.cur_num_batch < self.max_batch else self.max_batch


    def sample(self, num_mini_batch: int = None, mini_batch_size: int = None):
        """
        Yield training data for APPO.
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

        # keep (num_agent, dim)
        obs = self.obs[:, :-1].reshape(-1, *self.obs.shape[3:])
        actions = self.actions.reshape(-1, *self.actions.shape[3:])
        rollout_obs = self.rollout_obs[:, :-1].reshape(-1, *self.rollout_obs.shape[3:])
        value_preds = self.action_level_v_values[:, :-1].reshape(-1, *self.action_level_v_values.shape[3:])
        returns = self.action_level_returns.reshape(-1, *self.action_level_returns.shape[3:])
        advantages = self.action_level_advantages.reshape(-1, *self.action_level_advantages.shape[3:])
        log_prob = self.action_level_log_probs.reshape(-1, *self.action_level_log_probs.shape[3:])
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