"""
############################################
######### Example Training Script ##########
############################################
This code is part of a sample solution that can be a good starting point for how to structure the training of an agent and do logging. However, the performance that this solution achieves is not good.
"""

# implementation based on https://docs.cleanrl.dev/rl-algorithms/ddpg/

import torch
import numpy as np
import os
import random
import time
import gymnasium as gym
from dataclasses import dataclass
from gymnasium.spaces import Box
from stable_baselines3.common.buffers import ReplayBuffer
from torch import nn, optim
import torch.nn.functional as F

from agent_interface import convert_action, convert_obs, Agent
from util import save_model, create_env

@dataclass
class Args:
    exp_name: str = "ddpg_benchmark_final"
    # the name of this experiment

    seed: int = 1
    # seed of the experiment

    torch_deterministic: bool = True
    # if toggled, `torch.backends.cudnn.deterministic=False`

    track: bool = True
    # if toggled, this experiment will be tracked with Weights and Biases

    track_frequency: int = 100
    # frequency for the tracking of actor loss and qf_loss

    wandb_project_name: str = "RLLBC_BPA3"
    # the wandb's project name

    wandb_entity: str = None
    # the entity (team) of wandb's project

    save_model: bool = True
    # whether to save model into the `runs/{run_name}` folder

    env_id: str = "Racing-Env"
    # the environment id

    total_timesteps: int = 4000000
    # total timesteps of the experiments

    learning_rate: float = 3e-4
    # the learning rate of the optimizer

    buffer_size: int = int(4000000)
    # the replay memory buffer size

    gamma: float = 0.99
    # the discount factor gamma

    tau: float = 0.005
    # target smoothing coefficient

    batch_size: int = 32
    # the batch size of sample from the reply memory

    exploration_noise: float = 0.5
    # the scale of exploration noise

    learning_starts: int = 1e4
    # timestep to start learning

    policy_frequency: int = 2
    # the frequency of training policy (delayed)

    noise_clip: float = 0.5
    # noise clip parameter of the Target Policy Smoothing Regularization

    eval_freq: int = 10000
    # frequency of evaluation

    render_eval: bool = False
    # whether to render the evaluation episodes

    num_eval_episodes: int = 1
    # how many episodes to run at each evaluation time

    best_model_save_path: str = os.path.join("models", exp_name + "_best.obj")
    # where to save the model with the best evaluation performance

    last_model_save_path: str = os.path.join("models", exp_name + "_last.obj")
    # where to save the last model state

class QNetwork(nn.Module):
    def __init__(self, env, obs_shape):
        super().__init__()
        action_shape = env.action_space.shape
        self.fc1 = nn.Linear(obs_shape + np.prod(action_shape), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":

    args = Args()
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # use cuda if possible (much faster but only possible with nvidia gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    env = create_env(args.seed, render_env=False, limit_speed_factor=None, render_width=int(1280))
    obs_shape = convert_obs(env.reset()[0]).shape[0]
    env_eval = create_env(args.seed, render_env=args.render_eval, limit_speed_factor=None, render_width=1280)
    assert isinstance(env_eval.action_space, Box), "only continuous action space is supported"
    actor = Agent(env).to(device)
    qf1 = QNetwork(env, obs_shape).to(device)
    qf1_target = QNetwork(env, obs_shape).to(device)
    target_actor = Agent(env).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    # define Box that fits the shape of the observations obtained from convert_obs() to initialize replay buffer
    new_shape = (obs_shape, )
    new_box = Box(low=env.observation_space.low[:new_shape[0]], high=env.observation_space.high[:new_shape[0]])
    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        new_box,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )

    # start the game
    obs, _ = env.reset(seed=args.seed)
    obs = convert_obs(obs).to(device)
    best_eval_reward = -np.inf

    for global_step in range(args.total_timesteps):
        
        # action logic
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = convert_action(actor(obs))
                action += torch.normal(0, actor.action_scale[0] * args.exploration_noise).cpu().numpy()
                action = action.clip(env.action_space.low, env.action_space.high)

        # execute the game and log data.
        next_obs, reward, termination, truncation, info = env.step(action)
        next_obs = convert_obs(next_obs).to(device)
        stopped = False
        if termination or truncation == True:
            stopped = True

        # termination instead of stopped!
        rb.add(obs.cpu(), next_obs.cpu(), action, reward, termination, info)

        if termination or truncation:
            print(f"global_step={global_step}, episode_reward={info['episode']['r']}, episode_length={info['episode']['l']}")
            if args.track:
                wandb.log(data = {'episode_cumulative_reward': info['episode']['r'], 'episode_length': info['episode']['l']}, commit=False)
            next_obs, _ = env.reset()
            next_obs = convert_obs(next_obs).to(device)
                
        # CRUCIAL step, easy to overlook
        obs = next_obs

        # training logic
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                if args.track and global_step % args.track_frequency == 0:
                    wandb.log(data = {'actor_loss': actor_loss.item()}, commit = False)
            if args.track and global_step % args.track_frequency == 0:
                wandb.log(data = {'qf1_loss': qf1_loss.item()}, commit = False)

        # evaluation
        eval_episodes_performed = 0
        if global_step > args.learning_starts and global_step % args.eval_freq == 0:  # do an eval episode

            total_reward = 0
            total_eval_steps = 0

            for i in range(args.num_eval_episodes):

                print(f"Eval episode {i+1} started...")

                done = False
                truncation_eval = False
                eval_step = 0

                obs_eval, _ = env_eval.reset(seed=True)
                obs_eval = convert_obs(obs_eval).to(device)

                while not done and not truncation_eval:
                    eval_step += 1
                    with torch.no_grad():
                        action = convert_action(actor.get_action(obs_eval))
                        action = action.clip(env.action_space.low, env.action_space.high)
                    obs_eval, reward_eval, done, truncation_eval, info_eval = env_eval.step(action)
                    obs_eval = convert_obs(obs_eval).to(device)
                    total_reward += reward_eval

                total_eval_steps += eval_step
                eval_episodes_performed += 1

            # logging evaluation performance
            avg_eval_reward = total_reward / eval_episodes_performed
            avg_eval_steps = total_eval_steps / eval_episodes_performed
            print(f"Evaluation result: {avg_eval_steps} avg. steps, avg. reward: {avg_eval_reward}")
            if args.track:
                wandb.log(data = {'eval_cumulative_reward': avg_eval_reward, 'eval_episode_steps': avg_eval_steps}, step = global_step, commit = False)

            # save model state
            if avg_eval_reward > best_eval_reward:
                print("NEW BEST EVALUATION PERFORMANCE")
                best_eval_reward = avg_eval_reward
                save_model(actor, args.best_model_save_path)
            save_model(actor, args.last_model_save_path)
            
        if args.track:
            wandb.log(data = {}, step = global_step, commit=False)

    env.close()
    env_eval.close()