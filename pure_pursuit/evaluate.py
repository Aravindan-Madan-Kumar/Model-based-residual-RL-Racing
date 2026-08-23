"""Score the tuned pure-pursuit agent and write pure_pursuit/RESULTS.md.

Saves the agent, reloads it through the pickle round-trip, runs the same loop as the
grading harness, and records the measurements. The report holds measured values only.

The artifact is ``models/model_pp.obj``. The graded filename ``models/model.obj`` is
left untouched, so running this cannot change what would be submitted.

Usage::

    pixi run python pure_pursuit/evaluate.py --episodes 10 --save
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import date

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pure_pursuit.controller import DEFAULT_PARAMS, PARAM_NAMES  # noqa: E402
from pure_pursuit.rollout import make_env, rollout  # noqa: E402

from agent_interface import Agent, convert_action, convert_obs  # noqa: E402
from util import create_env, load_model, save_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BEST_PATH = os.path.join(HERE, 'best_params.json')
RESULTS_PATH = os.path.join(HERE, 'RESULTS.md')
MODEL_PATH = os.path.join(REPO_ROOT, 'models', 'model_pp.obj')

GRAVITY = 9.81  # m/s^2


def load_params():
    """
    Use tuned gains when present, otherwise the controller defaults.

    :return: (gains, human-readable source)
    """
    if os.path.exists(BEST_PATH):
        with open(BEST_PATH) as fh:
            return np.asarray(json.load(fh)['params']), 'pure_pursuit/best_params.json'
    return np.asarray(DEFAULT_PARAMS), 'pure_pursuit.controller.DEFAULT_PARAMS'


def grader_loop(model, episodes, seed=42):
    """
    The ``try_agent.py`` evaluation loop, run against ``models/model_pp.obj``.

    :return: (list of episodic returns, list of step counts)
    """
    env = create_env(seed=seed, render_env=False, limit_speed_factor=None,
                     render_width=1280)
    returns, steps = [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total, n, done = 0.0, 0, False
        while not done:
            action = convert_action(model.get_action(convert_obs(obs)))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            n += 1
            done = terminated or truncated
        returns.append(total)
        steps.append(n)
    env.close()
    return returns, steps


def check_pickle_module(path):
    """
    Confirm the saved object's class resolves the way the grader resolves it.

    Runs in a subprocess whose working directory sits outside the repository, with only
    the repository root on ``sys.path``. A class defined in a training script unpickles
    fine from the repository root but fails here, and on the grader.

    :return: (ok, qualified class name or error message)
    """
    code = ("import pickle, sys;"
            f"sys.path.insert(0, {REPO_ROOT!r});"
            f"m = pickle.load(open({path!r}, 'rb'));"
            "print(type(m).__module__ + '.' + type(m).__name__)")
    proc = subprocess.run([sys.executable, '-c', code], cwd='/', capture_output=True,
                          text=True)
    if proc.returncode != 0:
        return False, proc.stderr.strip().splitlines()[-1]
    qualname = proc.stdout.strip()
    return qualname.startswith('agent_interface.'), qualname


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', action='store_true',
                        help='write models/model_pp.obj')
    args = parser.parse_args()

    params, source = load_params()
    print(f"gains from {source}")

    env = make_env()
    diag = rollout(env, params, seed=args.seed, collect_speeds=True)
    env.close()

    if args.save:
        save_model(Agent(params), MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"{MODEL_PATH} does not exist; rerun with --save")
    model = load_model(MODEL_PATH)

    returns, steps = grader_loop(model, args.episodes, args.seed)
    mean, std = float(np.mean(returns)), float(np.std(returns))
    print(f"\nMean return: {mean}\nStd. deviation: {std}")

    pickle_ok, pickle_msg = check_pickle_module(MODEL_PATH)
    print(f"pickle module check: {'OK' if pickle_ok else 'FAILED'} ({pickle_msg})")

    write_results(params, source, diag, steps, mean, std, pickle_ok, pickle_msg, args)
    print(f"wrote {RESULTS_PATH}")


def _table(rows):
    """
    Render a two-column markdown table.

    :param rows: iterable of (label, value) pairs
    :return: markdown string
    """
    body = '\n'.join(f"| {label} | {value} |" for label, value in rows)
    return f"| Metric | Value |\n| --- | --- |\n{body}\n"


def build_sections(params, source, diag, steps, mean, std, pickle_msg, args):
    """
    Assemble the report as data: a list of (heading, rows) pairs.

    Values are measured during the run, read from the live environment, or read from
    ``best_params.json``.

    :return: list of (str, list[tuple]) sections
    """
    friction_limit = diag['max_grip'] * GRAVITY
    sections = [
        ('Score', [
            ('**Mean return**', f"**{mean:.5f}**"),
            ('Std. deviation', f"{std:.5f}"),
            ('Episodes', args.episodes),
            ('Steps per episode', sorted(set(steps))),
        ]),
        ('Rollout diagnostics', [
            ('Steps', diag['steps']),
            ('Left track', diag['left_track']),
            ('Cones hit', diag['cones_hit']),
            ('Distance travelled', f"{diag['traveled']:.1f} m"),
            ('Track progress', f"{diag['track_progress']:.1f} m"),
            ('Laps completed', f"{diag['track_progress'] / diag['track_length']:.3f}"),
            ('Mean speed', f"{diag['mean_speed']:.2f} m/s"),
            ('Max speed', f"{diag['max_speed']:.2f} m/s"),
        ]),
        ('Environment constants', [
            ('Track centerline length', f"{diag['track_length']:.2f} m"),
            ('Vehicle top speed', f"{diag['top_speed']:.3f} m/s"),
            ('Vehicle `max_grip`', f"{diag['max_grip']:.2f}"),
            ('Friction limit', f"{friction_limit:.2f} m/s²"),
        ]),
        ('Tuned gains', [(f"`{n}`", f"{v:.6f}") for n, v in zip(PARAM_NAMES, params)]),
        ('Derived', [
            ('Max speed / top speed', f"{100 * diag['max_speed'] / diag['top_speed']:.0f} %"),
            ('Max speed / `vcap`', f"{100 * diag['max_speed'] / params[8]:.0f} %"),
            ('`alat` / friction limit', f"{100 * params[4] / friction_limit:.0f} %"),
            ('Mean speed x 600 x 0.01', f"{0.01 * diag['mean_speed'] * 600:.2f}"),
            ('Return per lap', f"{mean / (diag['track_progress'] / diag['track_length']):.2f}"),
        ]),
    ]

    if os.path.exists(BEST_PATH):
        with open(BEST_PATH) as fh:
            meta = json.load(fh)
        sections.append(('Tuning', [
            ('Random search samples', meta['random_samples']),
            ('Best return after random search', f"{meta['random_search_best']:.3f}"),
            ('ES generations', meta['generations']),
            ('ES seed', meta['seed']),
            ('Worker processes', meta.get('workers', 4)),
            ('Wall clock', f"{meta['wall_seconds']:.0f} s"),
        ]))

    sections.append(('Checks', [
        ('Pickle round-trip', 'pass'),
        ('Foreign-cwd unpickle', pickle_msg),
        ('Gains source', f"`{source}`"),
        ('Artifact', f"`{os.path.relpath(MODEL_PATH, REPO_ROOT)}`"),
    ]))
    return sections


def write_results(params, source, diag, steps, mean, std, pickle_ok, pickle_msg, args):
    """Write RESULTS.md: measured values"""
    sections = build_sections(params, source, diag, steps, mean, std, pickle_msg, args)
    parts = [f"# Pure-pursuit baseline - results",
             f"\nGenerated by `pure_pursuit/evaluate.py` on {date.today().isoformat()}.\n"]
    for heading, rows in sections:
        parts.append(f"\n## {heading}\n\n{_table(rows)}")
    with open(RESULTS_PATH, 'w') as fh:
        fh.write(''.join(parts))


if __name__ == '__main__':
    main()
