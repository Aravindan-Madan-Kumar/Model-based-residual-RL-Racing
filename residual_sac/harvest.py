"""Install the best trained residual agent found across the run directories.

Scores every ``best.obj`` through the grading path, keeps the highest, and writes it to
``models/model.obj`` only if it beats what is already installed. Runs the same isolated
unpickle check as ``raceline/install.py``: the artifact must resolve through
``agent_interface`` alone and carry no CUDA storages.

    pixi run python residual_sac/harvest.py --save
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('MPLBACKEND', 'Agg')

from agent_interface import convert_action, convert_obs  # noqa: E402
from raceline.install import check_isolated_unpickle  # noqa: E402
from util import create_env, load_model, save_model  # noqa: E402

MODEL_PATH = os.path.join(REPO_ROOT, 'models', 'model.obj')


def grader_score(model, env, episodes=1):
    """
    :return: mean episodic return through the harness loop
    """
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset(seed=42)
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(
                convert_action(model.get_action(convert_obs(obs))))
            total += reward
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--save', action='store_true', help='write models/model.obj')
    p.add_argument('--pattern', default='runs_sector*/seed*/best.obj')
    return p.parse_args()


def main():
    args = parse_args()
    env = create_env(42)

    scored = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, args.pattern))):
        model = load_model(path)
        value = grader_score(model, env)
        scored.append((value, path, model))
        print(f"{os.path.relpath(path, REPO_ROOT):40s} {value:8.3f}")

    if not scored:
        raise SystemExit(f"no artifacts matched {args.pattern}")

    scored.sort(key=lambda row: row[0], reverse=True)
    value, path, model = scored[0]
    print(f"\nbest: {os.path.relpath(path, REPO_ROOT)} at {value:.3f}")

    if not args.save:
        print("dry run, pass --save to write models/model.obj")
        return

    if os.path.exists(MODEL_PATH):
        previous = grader_score(load_model(MODEL_PATH), env)
        print(f"currently installed: {previous:.3f}")
        if value <= previous:
            print("not installing, the new agent is not better")
            return

    save_model(model, MODEL_PATH)
    result = check_isolated_unpickle(MODEL_PATH)
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode != 0:
        raise SystemExit("isolated unpickle failed, the artifact is not submittable")

    print(f"reloaded artifact scores {grader_score(load_model(MODEL_PATH), env):.3f}")


if __name__ == '__main__':
    main()
