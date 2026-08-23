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

---

Policy mathematics: ``pure_pursuit/controller.py``. Tuning: ``pure_pursuit/tune_es.py``.
This module defines only the ``Agent`` class and the two conversion functions.

``Agent`` and ``ResidualAgent`` are defined here rather than in the ``pure_pursuit`` or
``residual_sac`` packages because pickle records the defining module of every class it
stores, and the evaluation harness imports ``agent_interface``.
"""

import os
import sys

import numpy as np
from torch import nn

# Required by a torch-based policy. Both packages are pinned in pixi.toml.
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# The harness imports this module from the repository root, which normally puts that
# root on sys.path. Setting it explicitly keeps the controller import working when the
# module is loaded from another working directory.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pure_pursuit.controller import DEFAULT_PARAMS, pure_pursuit_action  # noqa: E402


def convert_obs(obs: np.ndarray):
    """
    Pre-computation steps to convert the observations received from the environment into the format that can be
    fed to your agent.

    The controller is numpy-based and reads documented positions of the raw observation
    directly (velocities at indices 0-1, centerline at 416-455), so the layout is left
    intact and only the dtype is fixed. Unused dimensions - including the three RGB
    values at indices 7-9 - are simply never read.

    :param obs: 456-dimensional numpy ndarray containing the current observations of vehicle state, sensed cones and middle line
    :return: the converted obs that can be handled by the agent
    """
    return np.asarray(obs, dtype=np.float64)

    # Tensor variant for a torch-based policy. Deleting indices 7-9 shifts every higher
    # index down by three, so the centerline slice becomes 413:453 instead of 416:456.
    # indices_to_remove = [7, 8, 9] # remove rgb_arrays
    #
    # # remove unecessary information and make tensor
    # converted_obs = np.delete(obs, indices_to_remove)
    # converted_obs = torch.tensor(converted_obs)
    #
    # return converted_obs


def convert_action(action):
    """
    any potentially needed computation steps to convert the actions provided by agent
    into the format that can be fed into the environment.

    The agent already returns a plain numpy array, so this only enforces the dtype and
    the action-space bounds.

    :param action: the action returned by your agent
    :return: the converted action that can be used as input to the environment's step function
    """
    return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    # Variant for a policy whose get_action returns a tensor.
    # converted_action = action.detach().cpu().numpy()
    # return converted_action


class Agent(nn.Module):
    """
    The Agent Class

    To be able to evaluate, the agent needs to implement the function "get_action". Feel free to implement
    additional functions but make sure to not change the signature of "get_action" and the name of the class.
    
    OBS -> convert_obs -> get_action -> convert_action -> ACTION
    where OBS is the observation the environment provides and ACTION is the action that is fed into the environment.

    Geometric pure-pursuit controller. Stateless - ``get_action`` is a pure function of
    the observation, so no per-episode reset is required - and carries only its 10 gains.

    :param params: 10 controller gains in the order of
        ``pure_pursuit.controller.PARAM_NAMES``; defaults to the tuned set
    """

    def __init__(self, params=None):
        super().__init__()
        self.params = np.asarray(DEFAULT_PARAMS if params is None else params,
                                 dtype=np.float64)

        # MLP policy head. Takes `env` in place of `params`.
        #
        # action_shape = env.action_space.shape
        # obs_shape = convert_obs(env.reset()[0]).shape[0]
        #
        # self.fc1 = nn.Linear(obs_shape, 64)
        # self.fc2 = nn.Linear(64, 64)
        # self.fc_mu = nn.Linear(64, np.prod(action_shape))
        # # action rescaling
        # self.register_buffer(
        #     "action_scale", torch.tensor((env.action_space.high - env.action_space.low)
        #                                  / 2.0, dtype=torch.float32)
        # )
        # self.register_buffer(
        #     "action_bias", torch.tensor((env.action_space.high + env.action_space.low)
        #                                 / 2.0, dtype=torch.float32)
        # )

    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = torch.tanh(self.fc_mu(x))
    #     return x * self.action_scale + self.action_bias

    def get_action(self, obs):
        """
        :param obs: the output of :func:`convert_obs`
        :return: 3-element float32 ndarray, the input to :func:`convert_action`
        """
        action, _ = pure_pursuit_action(obs, self.params)
        return action

        # Variant for the MLP policy head.
        # return self.forward(obs)


class ResidualAgent(nn.Module):
    """
    Pure-pursuit controller with a learned residual correction.

    The residual acts on three interpretable quantities rather than on the raw pedals,
    so a zero residual reproduces :class:`Agent` exactly:

    ==============  =====================================================================
    ``apex_bias``   lateral shift of the aim point [m], giving a racing line
    ``speed_bias``  offset on the reference speed [m/s]
    ``steer_bias``  direct steering correction, covering slip the geometry cannot model
    ==============  =====================================================================

    The actor is held as a state dict and rebuilt on first use, so pickle records only
    this class and a dict of CPU tensors. The network class itself stays in
    ``residual_sac.policy`` and never appears in the saved artifact.

    Stateless with respect to the episode: ``get_action`` is a pure function of the
    observation.

    :param params: the 10 base controller gains
    :param actor: a trained ``residual_sac.policy.ResidualActor``
    :param hidden: hidden width of that actor, needed to rebuild it
    :param residual_scale: multiplier on residual authority, in [0, 1]
    """

    def __init__(self, params=None, actor=None, hidden=128, residual_scale=1.0):
        super().__init__()
        self.params = np.asarray(DEFAULT_PARAMS if params is None else params,
                                 dtype=np.float64)
        self.hidden = int(hidden)
        self.residual_scale = float(residual_scale)
        self.actor_state = ({} if actor is None else
                            {k: v.detach().cpu().clone()
                             for k, v in actor.state_dict().items()})
        self._actor = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_actor'] = None
        return state

    def actor(self):
        """
        Rebuild the actor from the stored state dict on first use.

        :return: an evaluation-mode ``ResidualActor`` on CPU
        """
        if self._actor is None:
            from residual_sac.policy import ResidualActor
            net = ResidualActor(hidden=self.hidden)
            if self.actor_state:
                net.load_state_dict(self.actor_state)
            net.eval()
            self._actor = net
        return self._actor

    def residual(self, obs):
        """
        Deterministic residual for one observation.

        :param obs: raw 456-dim observation
        :return: (3,) ndarray in [-1, 1]
        """
        import torch

        from residual_sac.features import residual_features

        features = residual_features(obs, self.params)
        with torch.no_grad():
            mean, _ = self.actor()(torch.from_numpy(features).unsqueeze(0))
            return torch.tanh(mean).squeeze(0).numpy()

    def act_from_residual(self, obs, residual):
        """
        Apply a residual vector to the geometric controller.

        Shared by :meth:`get_action` and the training rollout so both follow the same
        path from residual to environment action.

        :param obs: raw 456-dim observation
        :param residual: (3,) array in [-1, 1]
        :return: 3-element float32 action
        """
        from residual_sac.features import APEX_SCALE, SPEED_SCALE, STEER_SCALE

        scale = self.residual_scale
        action, _ = pure_pursuit_action(
            obs, self.params,
            apex_bias=float(residual[0]) * APEX_SCALE * scale,
            speed_bias=float(residual[1]) * SPEED_SCALE * scale,
            steer_bias=float(residual[2]) * STEER_SCALE * scale,
        )
        return action

    def get_action(self, obs):
        """
        :param obs: the output of :func:`convert_obs`
        :return: 3-element float32 ndarray, the input to :func:`convert_action`
        """
        return self.act_from_residual(obs, self.residual(obs))
