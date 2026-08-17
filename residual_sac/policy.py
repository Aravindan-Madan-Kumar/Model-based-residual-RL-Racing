"""Squashed-Gaussian actor over the residual channels.

Kept out of ``agent_interface`` deliberately: :class:`agent_interface.ResidualAgent`
stores this network as a state dict rather than as a submodule, so pickle never records
this class and the saved artifact depends on ``agent_interface`` alone.
"""

import torch
import torch.nn.functional as F
from torch import nn

from residual_sac.features import FEATURE_DIM, RESIDUAL_DIM

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class ResidualActor(nn.Module):
    """
    Two-layer MLP producing the mean and log-std of a squashed Gaussian.

    The mean head is zero-initialised so an untrained actor emits a zero residual and
    reproduces the geometric controller exactly.

    :param feature_dim: input width
    :param hidden: width of both hidden layers
    """

    def __init__(self, feature_dim=FEATURE_DIM, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_mean = nn.Linear(hidden, RESIDUAL_DIM)
        self.fc_logstd = nn.Linear(hidden, RESIDUAL_DIM)
        nn.init.zeros_(self.fc_mean.weight)
        nn.init.zeros_(self.fc_mean.bias)

    def forward(self, x):
        h = F.relu(self.fc2(F.relu(self.fc1(x))))
        log_std = torch.tanh(self.fc_logstd(h))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return self.fc_mean(h), log_std

    def sample(self, x):
        """
        Reparameterised sample with the tanh correction on the log-probability.

        :return: (action in [-1, 1], log_prob, deterministic action)
        """
        mean, log_std = self(x)
        normal = torch.distributions.Normal(mean, log_std.exp())
        pre = normal.rsample()
        action = torch.tanh(pre)
        log_prob = normal.log_prob(pre) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True), torch.tanh(mean)
