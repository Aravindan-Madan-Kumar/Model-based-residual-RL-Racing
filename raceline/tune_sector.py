"""Tune the sector controller on the fixed track by black-box search.

Same two-phase scheme as :mod:`raceline.tune_es` - random search for a basin, then a
``(mu, lambda)`` evolution strategy - over the 34 parameters of
:mod:`raceline.sector`.

The search is seeded from the tuned racing-line agent so it starts at that agent's score
rather than from scratch: with 34 dimensions, random search alone rarely finds a
surviving lap, and there is no reason to rediscover what is already known.

Runs on CPU only. Episodes are Box2D physics with no network in the loop, so this
parallelises across processes and gains nothing from a GPU.

Usage::

    pixi run python raceline/tune_sector.py --random 600 --generations 60 --workers 4
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('MPLBACKEND', 'Agg')

from raceline.sector import (N_SECTORS, PARAM_NAMES, SectorTracker,  # noqa: E402
                             build_profile)

HERE = os.path.dirname(os.path.abspath(__file__))
BEST_PATH = os.path.join(HERE, 'best_sector.json')
SEED_PATH = os.path.join(HERE, 'best_params.json')

# Only the smoothing weight is quantised, because it is the one parameter that changes
# the line and so triggers a least-squares solve. The index is looked up rather than
# written down: hard-coding it once already pointed at the wrong parameter, which
# silently disabled the line cache.
SMOOTH_STEP = 0.05
SMOOTH_INDEX = PARAM_NAMES.index('smooth')


def _index(name):
    """
    :param name: entry of :data:`raceline.sector.PARAM_NAMES`
    :return: its position in the flat parameter vector
    """
    return PARAM_NAMES.index(name)


# Each ablation pins one feature to its neutral setting and lets everything else tune, so
# the drop against the full search is that feature's contribution. Features that exist to
# vary a quantity per sector are neutralised by tying the sectors together rather than by
# freezing them, which removes the per-sector freedom without also removing the global
# quantity.
def _off_launch(c):
    c[_index('launch_v')] = 0.0        # below this speed never triggers
    c[_index('launch_thr')] = 1.0
    c[_index('launch_slip')] = 1.0
    return c


def _off_filter(c):
    c[_index('steer_alpha')] = 1.0     # command passes straight through
    return c


def _off_sector_speed(c):
    # v_post still scales speed globally, so only the per-sector freedom is removed.
    i = _index('v_scale0')
    c[i:i + N_SECTORS] = 1.0
    return c


def _off_sector_brake(c):
    i = _index('a_brake0')
    c[i:i + N_SECTORS] = c[i]          # one global braking limit, still tuned
    return c


def _off_ld_curve(c):
    c[_index('k_ld_curve')] = 0.0      # lookahead no longer shortens in corners
    return c


ABLATIONS = {
    'launch': _off_launch,
    'filter': _off_filter,
    'sector_speed': _off_sector_speed,
    'sector_brake': _off_sector_brake,
    'ld_curve': _off_ld_curve,
}

#                 k_ld  ld0 ldmin ldmax kstr kdmp  kp  tprv klat kldc salpha vpost
_TRACKER_LOW = [0.05, 0.0, 2.0, 4.0, 0.50, 0.0, 0.05, 0.0, 0.00, 0.0, 0.15, 0.80]
_TRACKER_HIGH = [1.20, 12.0, 15.0, 45.0, 2.50, 3.0, 3.00, 1.5, 0.40, 0.9, 1.00, 1.45]
#                launch_v launch_thr launch_slip
_LAUNCH_LOW = [0.0, 0.30, 0.20]
_LAUNCH_HIGH = [14.0, 1.00, 1.00]
#                a_max a_accel smooth
_PROFILE_LOW = [8.5, 3.0, 0.0]
_PROFILE_HIGH = [12.5, 7.0, 0.60]

LOW = np.array(_TRACKER_LOW + _LAUNCH_LOW + _PROFILE_LOW
               + [0.60] * N_SECTORS + [4.0] * N_SECTORS)
HIGH = np.array(_TRACKER_HIGH + _LAUNCH_HIGH + _PROFILE_HIGH
                + [1.45] * N_SECTORS + [13.0] * N_SECTORS)

_STATE = None


def seed_vector():
    """
    Build a starting point from the tuned racing-line agent.

    :return: (34,) parameter vector, or ``None`` if that agent has not been tuned
    """
    if not os.path.exists(SEED_PATH):
        return None
    b = json.load(open(SEED_PATH))
    p = b['params']
    tracker = [p[0], p[1], p[2], p[3], p[4], p[5], p[6],
               p[8], p[9],
               0.0,                 # no curvature scheduling on the lookahead yet
               1.0,                 # unfiltered steering
               p[7]]                # the old global v_scale, applied at lookup
    launch = [0.0, 1.0, 1.0]        # launch mode off
    profile = [b['a_max'], b['a_accel'], b['smooth']]
    # Per-sector scaling starts neutral so the seed reproduces the racing-line agent
    # exactly rather than approximately.
    return np.clip(np.array(tracker + launch + profile
                            + [1.0] * N_SECTORS
                            + [b['a_brake']] * N_SECTORS), LOW, HIGH)


def project(candidate, disabled=()):
    """
    Round the smoothing weight and apply any ablation constraints.

    Applied everywhere a candidate is used, so a disabled feature cannot creep back in
    through the seed, the reported best, or the saved parameters.

    :param disabled: names of features to neutralise, keys of :data:`ABLATIONS`
    :return: a projected copy
    """
    c = np.array(candidate, dtype=float)
    c[SMOOTH_INDEX] = round(c[SMOOTH_INDEX] / SMOOTH_STEP) * SMOOTH_STEP
    for name in disabled:
        c = ABLATIONS[name](c)
    return c


def _worker_state():
    """One environment and a cache of solved lines per worker process."""
    global _STATE
    if _STATE is None:
        from pure_pursuit.rollout import make_env
        _STATE = {'env': make_env(), 'lines': {}}
    return _STATE


def rollout_score(env, candidate, disabled=(), lines=None):
    """
    :param candidate: a full parameter vector
    :param disabled: names of features to neutralise
    :param lines: optional cache of solved lines, keyed by smoothing weight
    :return: episodic return
    """
    from agent_interface import convert_action, convert_obs

    params = project(candidate, disabled)
    profile = build_profile(params, lines=lines)
    tracker = SectorTracker(profile['line'], profile['s'], profile['speed'],
                            profile['kappa'])

    obs, _ = env.reset(seed=42)
    total, done = 0.0, False
    while not done:
        action, _ = tracker.action(convert_obs(obs), params)
        obs, reward, terminated, truncated, _ = env.step(convert_action(action))
        total += reward
        done = terminated or truncated
    return total


def _evaluate_batch(payload):
    """
    :param payload: ``(candidates, disabled)``, plain types so it pickles cheaply
    :return: list of episodic returns
    """
    candidates, disabled = payload
    state = _worker_state()
    return [rollout_score(state['env'], c, disabled, state['lines'])
            for c in candidates]


def _map_population(pool, population, workers, disabled=()):
    """Evaluate a population across the worker pool, preserving order."""
    chunks = [([c.tolist() for c in population[i::workers]], tuple(disabled))
              for i in range(workers)]
    results = pool.map(_evaluate_batch, chunks)
    fitness = np.empty(len(population))
    for i in range(workers):
        fitness[i::workers] = results[i]
    return fitness


def random_search(pool, n, workers, rng, seed_point, disabled=()):
    """
    Sample around the seed rather than uniformly over the box.

    In 34 dimensions a uniform sample almost never survives a lap, so most of the budget
    would be spent scoring crashes. Perturbing the seed keeps the samples informative.

    :return: (best parameter vector, best return)
    """
    span = (HIGH - LOW) * 0.25
    population = [seed_point.copy()]
    population += [np.clip(seed_point + span * rng.standard_normal(len(LOW)), LOW, HIGH)
                   for _ in range(n - 1)]
    fitness = _map_population(pool, population, workers, disabled)
    best = int(np.argmax(fitness))
    return population[best], float(fitness[best])


def evolution_strategy(pool, mean, generations, workers, rng, lam=20, mu=6,
                       disabled=()):
    """
    :return: (best parameter vector, its return, per-generation history)
    """
    mean = np.asarray(mean, dtype=np.float64)
    sigma = (HIGH - LOW) * 0.12
    sigma_floor = (HIGH - LOW) * 0.015
    best_x, best_f = mean.copy(), -np.inf
    history = []

    for gen in range(generations):
        population = [np.clip(mean + sigma * rng.standard_normal(len(mean)), LOW, HIGH)
                      for _ in range(lam)]
        population[0] = mean.copy()

        fitness = _map_population(pool, population, workers, disabled)
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
        print(f"gen {gen:3d}  best={fitness[order[0]]:7.3f}  "
              f"elite_mean={fitness[order[:mu]].mean():7.3f}  "
              f"incumbent={best_f:7.3f}", flush=True)

    return best_x, best_f, history


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--random', type=int, default=600)
    p.add_argument('--generations', type=int, default=60)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=5)
    p.add_argument('--out', default=None)
    p.add_argument('--disable', default='',
                   help='comma-separated features to neutralise, one of '
                        + ', '.join(ABLATIONS) + '; empty means the full search')
    return p.parse_args()


def main():
    args = parse_args()
    disabled = tuple(n for n in args.disable.split(',') if n)
    unknown = [n for n in disabled if n not in ABLATIONS]
    if unknown:
        raise SystemExit(f"unknown features {unknown}, expected any of {list(ABLATIONS)}")

    tag = 'full' if not disabled else 'no_' + '_'.join(disabled)
    out = args.out or os.path.join(HERE, f'best_sector_{tag}.json')

    rng = np.random.default_rng(args.seed)
    start = time.time()

    seed_point = seed_vector()
    if seed_point is None:
        raise SystemExit(f"no seed at {SEED_PATH}, run raceline/tune_es.py first")
    seed_point = project(seed_point, disabled)

    print(f"variant: {tag}"
          + (f"  (disabled: {', '.join(disabled)})" if disabled else ''))

    with mp.Pool(args.workers) as pool:
        print(f"phase 1: {args.random} samples around the seed")
        x0, f0 = random_search(pool, args.random, args.workers, rng, seed_point,
                               disabled)
        print(f"  best sampled return = {f0:.3f}  ({time.time() - start:.0f}s)\n")

        print(f"phase 2: (mu, lambda) ES, {args.generations} generations")
        best_x, best_f, history = evolution_strategy(
            pool, x0, args.generations, args.workers, rng, disabled=disabled)

    best_x = project(best_x, disabled)
    elapsed = time.time() - start
    print(f"\nbest return = {best_f:.4f}  wall = {elapsed:.0f}s")
    for name, value in zip(PARAM_NAMES, best_x):
        print(f"  {name:12s} = {value:.6f}")

    with open(out, 'w') as fh:
        json.dump({'params': best_x.tolist(),
                   'names': list(PARAM_NAMES),
                   'variant': tag,
                   'disabled': list(disabled),
                   'return': best_f,
                   'seed_return': f0,
                   'random_samples': args.random,
                   'generations': args.generations,
                   'seed': args.seed,
                   'wall_seconds': elapsed,
                   'history': history}, fh, indent=2)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
