"""Flat numpy replay buffer.

Stores the compact feature vector rather than the raw 456-dim observation, which keeps a
10^6-transition buffer near 250 MB instead of 3.6 GB.
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    :param capacity: maximum stored transitions
    :param feature_dim: width of the feature vector
    :param action_dim: width of the residual action
    """

    def __init__(self, capacity, feature_dim, action_dim):
        self.capacity = int(capacity)
        self.size = 0
        self.pos = 0
        self.obs = np.zeros((self.capacity, feature_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, feature_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        """
        :param done: True only for a genuine terminal, never for the 600-step time limit
        """
        i = self.pos
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        """
        :return: tensors (obs, actions, rewards, next_obs, dones) on ``device``
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        return tuple(torch.as_tensor(a[idx], device=device)
                     for a in (self.obs, self.actions, self.rewards,
                               self.next_obs, self.dones))

    def __len__(self):
        return self.size
