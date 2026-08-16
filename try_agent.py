from CarEnv.Configs import RACING_FAST
from agent_interface import convert_obs, convert_action, Agent
from util import *
import torch

"""
Example evaluation script to test (and visualize) the performance of a previously saved agent on random tracks
"""

load_path_model = "./models/model.obj"  # replace this by the actual path to your previously saved actor
num_episodes = 10
seed = 42


model = load_model(load_path_model)
env = create_env(seed = seed, render_env=False, limit_speed_factor=None, render_width=1280)

total_reward = 0
total_eval_steps = 0

print(f"Starting evaluation of actor saved in {load_path_model}.")
print(f"Performing {num_episodes} evaluation runs...")

returns = [0]*num_episodes

for i in range(num_episodes):

    obs, _ = env.reset(seed=seed+i)

    done = False
    while not done:
        state = convert_obs(obs)
        with torch.no_grad():
            output = model.get_action(state)
        action = convert_action(output)

        obs, reward, terminated, truncated, info = env.step(action)
        returns[i] += reward
        done = terminated or truncated

env.close()

print(f"Mean return: {np.mean(returns)}")
print(f"Std. deviation: {np.std(returns)}")