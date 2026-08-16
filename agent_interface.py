"""
###################################################
######### Agent Interface for Evaluation ##########
###################################################

To hand in an agent for evaluation implement the functions "convert_obs", "convert_action" and the "Agent" class.
For a clarification on the purpose of the different functions please read the function documentations and see the
task description info sheet.

This file will be used during the evaluation of your code. For that reason make sure that you do not use any imports that
are not part of the evaluation environment. Also, please refrain from using this file for your training code. For that
create and use separate scripts.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
import numpy as np

def convert_obs(obs: np.ndarray):
    """
    Pre-computation steps to convert the observations received from the environment into the format that can be
    fed to your agent.

    E.g. if your agent is a nn.Module and works on tensors, you would need to convert the observation to a tensor here.

    :param obs: 456-dimensional numpy ndarray containing the current observations of vehicle state, sensed cones and middle line
    :return: the converted obs that can be handled by the agent
    """

    # remove unnecessary information
    indices_to_remove = [7, 8, 9] # remove rgb_arrays

    # remove unecessary information and make tensor
    converted_obs = np.delete(obs, indices_to_remove)
    converted_obs = torch.tensor(converted_obs)

    return converted_obs


def convert_action(action):
    """
    any potentially needed computation steps to convert the actions provided by your agent
    into the format that can be fed into the environment.

    E.g. if your agent is a nn.Module and returns a tensor as action, you would need to convert the action tensor
    to a numpy array here.

    :param action: the action returned by your agent
    :return: the converted action that can be used as input to the environment's step function
    """

    # not used here
    converted_action = action.detach().cpu().numpy()
    return converted_action


class Agent(nn.Module):
    """
    The Agent Class

    To be able to evaluate your Code, the agent needs to implement the function "get_action". Feel free to implement
    additional functions but make sure to not change the signature of "get_action" and the name of the class.

    The querying process of your model looks schematically like that:
    OBS -> convert_obs -> get_action -> convert_action -> ACTION
    where OBS is the observation the environment provides and ACTION is the action that is fed into the environment.
    """

    def __init__(self, env):
        super().__init__()

        action_shape = env.action_space.shape
        obs_shape = convert_obs(env.reset()[0]).shape[0]

        self.fc1 = nn.Linear(obs_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mu = nn.Linear(64, np.prod(action_shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low)
                                         / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low)
                                        / 2.0, dtype=torch.float32)
        )
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias
        
    def get_action(self, obs):
        return self.forward(obs)