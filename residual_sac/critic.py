"""Twin Q networks for SAC.

Never part of the saved agent.
"""

import torch
import torch.nn.functional as F
from torch import nn


class QNetwork(nn.Module):
    """
    State-action value network.

    :param feature_dim: width of the feature vector
    :param action_dim: width of the residual action
    :param hidden: width of both hidden layers
    """

    def __init__(self, feature_dim, action_dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.fc_out(F.relu(self.fc2(F.relu(self.fc1(x)))))
