"""Score a racing-line agent through the grading path.

    pixi run python raceline/evaluate.py
    pixi run python raceline/evaluate.py --a-max 10.6 --episodes 3
"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('MPLBACKEND', 'Agg')

from agent_interface import RacelineAgent, convert_action, convert_obs  # noqa: E402
from raceline.controller import DEFAULT_PARAMS, TOP_SPEED  # noqa: E402
from raceline.optimize import DEFAULT_CORRIDOR, build  # noqa: E402
from raceline.track_data import A_MAX  # noqa: E402
from util import create_env  # noqa: E402


def make_agent(params=DEFAULT_PARAMS, corridor=DEFAULT_CORRIDOR, a_max=A_MAX,
               v_max=None):
    """
    :return: a :class:`agent_interface.RacelineAgent` over a freshly solved line
    """
    r = build(corridor=corridor, a_max=a_max, v_max=v_max)
    return RacelineAgent(r['line'], r['s'], r['speed'], r['kappa'], params), r


def score(agent, env, episodes=1, detail=False):
    """
    :return: mean return, and a diagnostics dict when ``detail`` is set
    """
    returns, diag = [], None
    for _ in range(episodes):
        obs, _ = env.reset(seed=42)
        total, done = 0.0, False
        rec = {'speed': [], 'alat': [], 'pos': []}
        while not done:
            rec['speed'].append(obs[0] * TOP_SPEED)
            rec['alat'].append(abs(obs[14]))
            rec['pos'].append(np.asarray(obs[4:6]).copy())
            obs, reward, terminated, truncated, _ = env.step(
                convert_action(agent.get_action(convert_obs(obs))))
            total += reward
            done = terminated or truncated
        returns.append(total)
        if diag is None:
            diag = {k: np.asarray(v) for k, v in rec.items()}
            diag['terminated'] = terminated
            diag['steps'] = len(rec['speed'])
    return (float(np.mean(returns)), diag) if detail else float(np.mean(returns))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corridor', type=float, default=DEFAULT_CORRIDOR)
    p.add_argument('--a-max', type=float, default=A_MAX)
    p.add_argument('--v-max', type=float, default=None)
    p.add_argument('--episodes', type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    agent, r = make_agent(corridor=args.corridor, a_max=args.a_max, v_max=args.v_max)
    env = create_env(42)
    value, diag = score(agent, env, args.episodes, detail=True)

    print(f"a_max {args.a_max:.2f}  corridor +-{args.corridor:.2f} m")
    print(f"solver predicts {60.0 / r['lap_time'] * r['progress_per_lap'] * 0.1:.1f}")
    print(f"measured return {value:.3f}  steps {diag['steps']}  "
          f"terminated early: {bool(diag['terminated'])}")
    print(f"speed mean {diag['speed'].mean():.2f}  max {diag['speed'].max():.2f} m/s")
    print(f"|a_lat| p95 {np.percentile(diag['alat'], 95):.2f}  max {diag['alat'].max():.2f}")


if __name__ == '__main__':
    main()
