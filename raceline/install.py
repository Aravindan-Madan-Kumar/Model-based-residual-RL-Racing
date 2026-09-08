"""Build the racing-line agent and write it to the graded artifact path.

Also runs the two checks a local ``try_agent.py`` cannot catch: that the pickle resolves
through ``agent_interface`` alone, and that it carries no CUDA storages.

    pixi run python raceline/install.py --save
"""

import argparse
import os
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('MPLBACKEND', 'Agg')

from agent_interface import (RacelineAgent, SectorAgent, convert_action,  # noqa: E402
                             convert_obs)
from raceline.controller import (DEFAULT_PARAMS, PROFILE_A_ACCEL,  # noqa: E402
                                 PROFILE_A_BRAKE, PROFILE_A_MAX, PROFILE_SMOOTH)
from raceline.optimize import build, velocity_profile  # noqa: E402
from util import create_env, load_model, save_model  # noqa: E402

MODEL_PATH = os.path.join(REPO_ROOT, 'models', 'model.obj')


def make_agent():
    """
    :return: a :class:`agent_interface.RacelineAgent` with the tuned line and gains
    """
    line = build(smooth=PROFILE_SMOOTH)
    speed = velocity_profile(line['kappa'], line['ds'], a_max=PROFILE_A_MAX,
                             a_accel=PROFILE_A_ACCEL, a_brake=PROFILE_A_BRAKE)
    return RacelineAgent(line['line'], line['s'], speed, line['kappa'],
                         np.asarray(DEFAULT_PARAMS))


def make_sector_agent():
    """
    :return: a :class:`agent_interface.SectorAgent` on the tuned sector parameters
    """
    from raceline.sector import DEFAULT_PARAMS as SECTOR_DEFAULTS, build_profile

    params = np.asarray(SECTOR_DEFAULTS)
    profile = build_profile(params)
    return SectorAgent(profile['line'], profile['s'], profile['speed'],
                       profile['kappa'], params)


BUILDERS = {'raceline': make_agent, 'sector': make_sector_agent}


def grader_score(model, env, episodes=3):
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


def check_isolated_unpickle(path):
    """
    Unpickle in a subprocess whose only import root is the repository.

    ``pickle`` records the defining module of every stored class, and the grader imports
    ``agent_interface`` alone, so a class reachable from the artifact but defined
    elsewhere loads fine from the repository root and fails on the grader.

    :return: the subprocess result
    """
    script = (
        "import pickle, sys, numpy as np\n"
        "import torch\n"
        f"sys.path.insert(0, {REPO_ROOT!r})\n"
        f"m = pickle.load(open({path!r}, 'rb'))\n"
        "print('class', type(m).__module__ + '.' + type(m).__name__)\n"
        "bad = [k for k, v in vars(m).items() "
        "if torch.is_tensor(v) and v.is_cuda]\n"
        "assert not bad, f'cuda storages: {bad}'\n"
        "obs = np.zeros(456); obs[0] = 0.3\n"
        "a = m.get_action(obs)\n"
        "print('action', np.asarray(a).round(3), 'dtype', np.asarray(a).dtype)\n"
    )
    return subprocess.run([sys.executable, '-c', script], cwd='/',
                          capture_output=True, text=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--save', action='store_true', help='write models/model.obj')
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--agent', choices=sorted(BUILDERS), default='sector')
    return p.parse_args()


def main():
    args = parse_args()
    agent = BUILDERS[args.agent]()
    env = create_env(42)

    value = grader_score(agent, env, args.episodes)
    print(f"{args.agent} agent: {value:.3f}")

    if not args.save:
        print("dry run, pass --save to write models/model.obj")
        return

    previous = None
    if os.path.exists(MODEL_PATH):
        previous = grader_score(load_model(MODEL_PATH), env, 1)
        print(f"currently installed: {previous:.3f}")
        if value <= previous:
            print("not installing, the new agent is not better")
            return

    save_model(agent, MODEL_PATH)
    result = check_isolated_unpickle(MODEL_PATH)
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode != 0:
        raise SystemExit("isolated unpickle failed, the artifact is not submittable")

    reloaded = grader_score(load_model(MODEL_PATH), env, 1)
    print(f"reloaded artifact scores {reloaded:.3f}")


if __name__ == '__main__':
    main()
