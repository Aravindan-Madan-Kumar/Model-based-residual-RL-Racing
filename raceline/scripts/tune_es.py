"""Tune the racing-line tracker on the fixed track by black-box search.

Same two-phase scheme that tuned the pure-pursuit gains: uniform random search to find a
basin, then a ``(mu, lambda)`` evolution strategy with per-dimension adaptive step size.

Eleven parameters are searched: the ten tracker gains plus ``a_max``, the lateral
acceleration the speed profile is allowed to plan for. ``a_max`` is included because the
right value is a property of the tyre model that cannot be derived from the config - a
plain ``mu * g`` friction circle underestimates it by about a third - so it is measured
here by what actually goes fastest without sliding off.

The minimum-curvature line itself does not depend on ``a_max``, only the speed profile
does, so each candidate recomputes the cheap forward-backward pass over a cached line.

Usage::

    pixi run python raceline/scripts/tune_es.py --random 300 --generations 20 --workers 4
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('MPLBACKEND', 'Agg')

from raceline.controller import PARAM_NAMES, RacelineTracker  # noqa: E402
from raceline.optimize import DEFAULT_CORRIDOR, build, velocity_profile  # noqa: E402

MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_PATH = os.path.join(MODULE_DIR, 'best_params.json')

SEARCH_NAMES = PARAM_NAMES + ('a_max', 'a_accel', 'a_brake', 'smooth')

# The last three shape the speed profile. They are searched rather than derived because
# the point-mass model behind the profile is only approximate: the tyre model saturates
# above a plain friction circle, and the front-wheel drivetrain makes acceleration and
# braking strongly asymmetric. Measured references are a_max 10.3, a_accel 5.2 peak,
# a_brake 10.7.
# ``smooth`` trades the theoretically fastest line for one the tracker can actually
# follow. The point-mass model says smoothing only costs time, but the model is 11%
# optimistic on the sharp line and 0.4% accurate on the centerline, so the trade is
# settled by measured return rather than by the model.
#            k_ld  ld0  ldmin ldmax ksteer kdamp  kp  v_scale t_prev k_lat a_max a_acc a_brk smooth
LOW = np.array([0.05, 0.0, 2.0, 4.0, 0.50, 0.0, 0.15, 0.60, 0.0, 0.00, 8.5, 3.0, 6.0, 0.0])
HIGH = np.array([1.20, 12.0, 15.0, 45.0, 2.50, 3.0, 3.00, 1.40, 1.5, 0.40, 12.5, 7.0, 13.0, 0.60])

# Lines are cached per rounded smoothing value so the least-squares solve runs a handful
# of times instead of once per candidate.
SMOOTH_STEP = 0.05

_STATE = None


def _worker_state():
    """One environment and a cache of racing lines per worker process."""
    global _STATE
    if _STATE is None:
        from pure_pursuit.rollout import make_env
        _STATE = {'env': make_env(), 'lines': {}}
    return _STATE


def _line_for(state, smooth):
    """
    :param smooth: smoothing weight, rounded to :data:`SMOOTH_STEP` before caching
    :return: the cached output of :func:`raceline.optimize.build`
    """
    key = round(float(smooth) / SMOOTH_STEP) * SMOOTH_STEP
    if key not in state['lines']:
        state['lines'][key] = build(corridor=DEFAULT_CORRIDOR, smooth=key)
    return state['lines'][key]


def rollout_score(env, line, candidate):
    """
    Run one full episode with the given parameters.

    :param line: the cached output of :func:`raceline.optimize.build`
    :param candidate: 14 values, the tracker gains, the profile limits, and the line
        smoothing weight
    :return: episodic return
    """
    from agent_interface import convert_action, convert_obs

    params = np.asarray(candidate[:10])
    a_max, a_accel, a_brake = (float(candidate[10]), float(candidate[11]),
                               float(candidate[12]))
    speed = velocity_profile(line['kappa'], line['ds'], a_max=a_max,
                             a_accel=a_accel, a_brake=a_brake)
    tracker = RacelineTracker(line['line'], line['s'], speed, line['kappa'])

    obs, _ = env.reset(seed=42)
    total, done = 0.0, False
    while not done:
        action, _ = tracker.action(convert_obs(obs), params)
        obs, reward, terminated, truncated, _ = env.step(convert_action(action))
        total += reward
        done = terminated or truncated
    return total


def _evaluate_batch(candidates):
    """
    :param candidates: parameter vectors as plain lists, which pickle cheaply
    :return: list of episodic returns
    """
    state = _worker_state()
    return [rollout_score(state['env'], _line_for(state, c[13]), c)
            for c in candidates]


def _map_population(pool, population, workers):
    """Evaluate a population across the worker pool, preserving order."""
    chunks = [[c.tolist() for c in population[i::workers]] for i in range(workers)]
    results = pool.map(_evaluate_batch, chunks)
    fitness = np.empty(len(population))
    for i in range(workers):
        fitness[i::workers] = results[i]
    return fitness


def random_search(pool, n, workers, rng):
    """
    :return: (best parameter vector, best return)
    """
    population = [LOW + rng.random(len(LOW)) * (HIGH - LOW) for _ in range(n)]
    fitness = _map_population(pool, population, workers)
    best = int(np.argmax(fitness))
    return population[best], float(fitness[best])


def evolution_strategy(pool, mean, generations, workers, rng, lam=12, mu=4):
    """
    :return: (best parameter vector, its return, per-generation history)
    """
    mean = np.asarray(mean, dtype=np.float64)
    sigma = (HIGH - LOW) * 0.15
    sigma_floor = (HIGH - LOW) * 0.02
    best_x, best_f = mean.copy(), -np.inf
    history = []

    for gen in range(generations):
        population = [np.clip(mean + sigma * rng.standard_normal(len(mean)), LOW, HIGH)
                      for _ in range(lam)]
        population[0] = mean.copy()

        fitness = _map_population(pool, population, workers)
        order = np.argsort(-fitness)
        elite = np.array([population[i] for i in order[:mu]])

        if fitness[order[0]] > best_f:
            best_f = float(fitness[order[0]])
            best_x = population[order[0]].copy()

        mean = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), sigma_floor)

        history.append({'generation': gen,
                        'best': float(fitness[order[0]]),
                        'elite_mean': float(fitness[order[:mu]].mean()),
                        'incumbent': best_f})
        print(f"gen {gen:2d}  best={fitness[order[0]]:7.3f}  "
              f"elite_mean={fitness[order[:mu]].mean():7.3f}  "
              f"incumbent={best_f:7.3f}", flush=True)

    return best_x, best_f, history


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--random', type=int, default=300)
    p.add_argument('--generations', type=int, default=20)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--out', default=BEST_PATH)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    start = time.time()

    with mp.Pool(args.workers) as pool:
        print(f"phase 1: random search, {args.random} samples")
        x0, f0 = random_search(pool, args.random, args.workers, rng)
        print(f"  best random return = {f0:.3f}  ({time.time() - start:.0f}s)\n")

        print(f"phase 2: (mu, lambda) ES, {args.generations} generations")
        best_x, best_f, history = evolution_strategy(
            pool, x0, args.generations, args.workers, rng)

    elapsed = time.time() - start
    print(f"\nbest return = {best_f:.4f}  wall = {elapsed:.0f}s")
    for name, value in zip(SEARCH_NAMES, best_x):
        print(f"  {name:10s} = {value:.6f}")

    with open(args.out, 'w') as fh:
        json.dump({'params': best_x[:10].tolist(),
                   'a_max': float(best_x[10]),
                   'a_accel': float(best_x[11]),
                   'a_brake': float(best_x[12]),
                   'smooth': round(float(best_x[13]) / SMOOTH_STEP) * SMOOTH_STEP,
                   'names': list(SEARCH_NAMES),
                   'return': best_f,
                   'random_search_best': f0,
                   'random_samples': args.random,
                   'generations': args.generations,
                   'seed': args.seed,
                   'wall_seconds': elapsed,
                   'history': history}, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
