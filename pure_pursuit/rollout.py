"""Rollout helpers shared by pure-pursuit tuning and evaluation.

The controller comes from :mod:`pure_pursuit.controller`, which is the single definition
of the policy mathematics.
"""

import os
import sys
import warnings

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

warnings.filterwarnings('ignore')

import gymnasium as gym  # noqa: E402

from CarEnv.Configs import RACING_FAST  # noqa: E402
from pure_pursuit.controller import TOP_SPEED, pure_pursuit_action  # noqa: E402


def make_env():
    """
    Build a bare CarEnv instance for tuning.

    Bypasses ``util.create_env`` because its ``RecordEpisodeStatistics`` wrapper and
    eager reset cost time across tens of thousands of rollouts. Environment, physics and
    reward are identical; final scoring in :mod:`pure_pursuit.evaluate` goes through
    ``util.create_env``.

    :return: the racing environment
    """
    return gym.make('CarEnv:gym_envs/CarEnv-v1', config=RACING_FAST, render_env=False,
                    limit_speed_factor=None, render_width=1280, disable_env_checker=True)


def rollout(env, params, seed=0, collect_speeds=False):
    """
    Run one full episode with the pure-pursuit controller.

    :param env: environment from :func:`make_env`
    :param params: 10 controller gains
    :param seed: reset seed. CarEnv uses a fixed track and ignores it; passed through so
        the call site mirrors the evaluation harness
    :param collect_speeds: also return summary statistics of the speed trace
    :return: dict with the episodic return, step count, termination flags and diagnostics
    """
    obs, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    speeds = []

    while True:
        if collect_speeds:
            speeds.append(obs[0] * TOP_SPEED)
        action, _ = pure_pursuit_action(obs, params)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        steps += 1
        if terminated or truncated:
            break

    unwrapped = env.unwrapped
    result = {
        'return': total,
        'steps': steps,
        'left_track': bool(terminated),
        'cones_hit': int(unwrapped.metrics.get('cones_hit', 0)),
        'traveled': float(unwrapped.traveled_distance),
        'track_progress': float(unwrapped.problem.track_progress),
        # Read from the live environment so a track or vehicle config change cannot
        # leave a stale constant in a generated report.
        'track_length': float(unwrapped.problem.lr.length),
        'top_speed': float(unwrapped.vehicle_model.top_speed),
        'max_grip': float(unwrapped.vehicle_model.max_grip),
    }
    if collect_speeds:
        result['mean_speed'] = float(np.mean(speeds))
        result['max_speed'] = float(np.max(speeds))
    return result


def score(env, params):
    """
    Episodic return only - the quantity the grader reports.

    :return: float episodic return
    """
    return rollout(env, params)['return']
