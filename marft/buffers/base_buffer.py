from abc import ABC, abstractmethod
import numpy as np

class BaseBuffer(ABC):
    def __init__(self, args, num_agents):
        super().__init__()
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.num_agents = num_agents
        self.max_new_tokens = args.max_new_tokens
        self.max_batch = 1  # when max_batch = 1, this is an on-policy buffer, otherwise it is a replaybuffer
        self.cur_num_batch = 0
        self.cur_batch_index = 0
        self.pre_batch_index = None
        self.step = 0

        self.obs = np.empty((self.max_batch, self.episode_length + 1, self.n_rollout_threads, self.num_agents), dtype=np.object_)
        self.actions = np.empty((self.max_batch, self.episode_length, self.n_rollout_threads, self.num_agents),dtype=np.object_)
        self.rollout_obs = np.empty((self.max_batch, self.episode_length + 1, self.n_rollout_threads, self.num_agents), dtype=np.object_)
        self.action_tokens = np.empty((self.max_batch, self.episode_length, self.n_rollout_threads, self.num_agents, self.max_new_tokens), dtype=np.int64)
        self.rewards = np.zeros((self.max_batch, self.episode_length, self.n_rollout_threads, self.num_agents), dtype=np.float32)
        self.masks = np.ones((self.max_batch, self.episode_length + 1, self.n_rollout_threads, self.num_agents), dtype=np.float32)

    @abstractmethod
    def insert(self, next_obs, actions, rollout_obs, value_preds, rewards, masks, action_tokens, log_probs):
        """
        Insert data into the buffer.
        :param next_obs: (np.ndarray) next observation
        :param actions: (np.ndarray) actions taken
        :param rollout_obs: (np.ndarray) rollout observations
        :param value_preds: (np.ndarray) value predictions
        :param rewards: (np.ndarray) rewards received
        :param masks: (np.ndarray) masks for the observations
        :param action_tokens: (np.ndarray) action tokens
        :param log_probs: (np.ndarray) log probabilities of the actions/tokens
        """
        pass

    @abstractmethod
    def sample(self, num_mini_batch: int = None, mini_batch_size: int = None):
        """
        Sample data from the buffer.
        :param num_mini_batch: (int) number of mini batches to sample
        :param mini_batch_size: (int) size of each mini batch
        :return: (dict) sampled data
        """
        pass

    @abstractmethod
    def compute_gae_and_returns(self, next_value):
        """
        Compute the Generalized Advantage Estimation (GAE) and returns for the collected data.
        :param next_value: (np.ndarray) next value prediction
        """
        pass

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.pre_batch_index = self.cur_batch_index
        self.cur_batch_index = (self.cur_batch_index + 1) % self.max_batch
        self.obs[self.cur_batch_index, 0] = self.obs[self.pre_batch_index, -1].copy()