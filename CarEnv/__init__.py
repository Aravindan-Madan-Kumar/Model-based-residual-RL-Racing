from gymnasium.envs.registration import register
from CarEnv.Env import CarEnv

register(id='gym_envs/CarEnv-v1', entry_point='CarEnv.Env:CarEnv')
