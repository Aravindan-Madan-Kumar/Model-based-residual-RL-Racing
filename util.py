import gymnasium as gym
import pickle
import numpy as np
import torch
from copy import deepcopy

from CarEnv.Configs import RACING_FAST
from CarEnv.Track.Generator import make_full_environment
from agent_interface import Agent


def save_model(model: Agent, save_path: str):
    """
    saves the provided actor to a file according to the given path
    :param model: the model to save
    :param save_path: where to save the actor to
    """
    model_copy = deepcopy(model)
    if isinstance(model_copy, torch.nn.Module):
        model_copy.to('cpu')  # for some models this might fail, in that case remove the line, but make sure the model is on cpu
    with open(save_path, 'wb') as filehandler:
        pickle.dump(model_copy, filehandler)
    print(f"Saving actor as {save_path}")


def load_model(load_path: str):
    """
    loads and returns a previously saved actor
    :param load_path: where to load the actor from
    :return: the loaded model
    """
    with open(load_path, 'rb') as filehandler:
        model = pickle.load(filehandler)         
    print(f"Loading actor from {load_path}")
    return model

def create_env(seed: int, render_env: bool = False, limit_speed_factor=None, render_width: int = 1280):
    """
    creates a racing environment
    (hint: it might make sense to use rendering for evaluation, but be aware that rendering slows the environment down)
    :param seed: (random) seed for the environment creation
    :param render_env: Can be set to True to render the environment steps with Pygame
    :param limit_speed_factor: Factor by which the rendering is at most sped up (if None: render as fast as possible)
    :param render_width: width of the Pygame rendering (in pixels)
    :return: the racing environment
    """

    env = gym.make('CarEnv:gym_envs/CarEnv-v1', config=RACING_FAST, render_env=render_env,
                   limit_speed_factor=limit_speed_factor, render_width=render_width)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=6000)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    return env
