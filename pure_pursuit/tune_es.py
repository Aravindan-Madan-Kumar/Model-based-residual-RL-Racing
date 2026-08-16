"""Tune the 10 pure-pursuit gains on the fixed training track by black-box search.

No gradients are available - episodic return is a step-terminating, non-differentiable
function of the gains - so the search treats the simulator as a black box and uses only
the scalar return of a full 600-step episode.


1. Uniform random search over a hand-set box, to locate a basin.

2. A (mu, lambda) evolution strategy seeded from the best random sample. Each generation
   draws ``lambda`` perturbations of the current mean, keeps the ``mu`` best, and sets
   the next mean and per-dimension step size from that elite set.


Usage::

    pixi run python pure_pursuit/tune_es.py --random 220 --generations 18 --workers 4
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

from pure_pursuit.controller import PARAM_NAMES  # noqa: E402
from pure_pursuit.rollout import make_env, score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BEST_PATH = os.path.join(HERE, 'best_params.json')

# Search box. Bounds are physical: alat/abrake around the 0.7 g friction limit, vcap up
# to just under top speed, lookahead on the scale of the 60 m centerline horizon.
LOW = np.array([0.10, 0.0, 3.0, 0.5, 3.0, 2.0, 0.15, 0.0, 15.0, -0.5])
HIGH = np.array([1.20, 12.0, 14.0, 2.5, 10.0, 14.0, 2.0, 3.0, 41.0, 3.0])

_ENV = None


def _worker_env():
    """One environment per worker process, built lazily and reused."""
    global _ENV
    if _ENV is None:
        _ENV = make_env()
    return _ENV


def _evaluate_batch(candidates):
    """
    :param candidates: gain vectors as plain lists, which pickle cheaply across workers
    :return: list of episodic returns
    """
    env = _worker_env()
    return [score(env, np.asarray(c)) for c in candidates]


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
    Phase 1: uniform sampling of the search box.

    :return: (best gain vector, best return)
    """
    population = [LOW + rng.random(len(LOW)) * (HIGH - LOW) for _ in range(n)]
    fitness = _map_population(pool, population, workers)
    best = int(np.argmax(fitness))
    return population[best], float(fitness[best])


def evolution_strategy(pool, mean, generations, workers, rng, lam=12, mu=4, verbose=True):
    """
    Phase 2: (mu, lambda) evolution strategy with per-dimension adaptive step size.

    :param mean: starting gain vector
    :param generations: number of generations to run
    :param lam: population size per generation
    :param mu: number of elites retained per generation
    :return: (best gain vector seen, its return, per-generation history)
    """
    mean = np.asarray(mean, dtype=np.float64)
    sigma = (HIGH - LOW) * 0.15
    sigma_floor = (HIGH - LOW) * 0.03
    best_x, best_f = mean.copy(), -np.inf
    history = []

    for gen in range(generations):
        population = [np.clip(mean + sigma * rng.standard_normal(len(mean)), LOW, HIGH)
                      for _ in range(lam)]
        population[0] = mean.copy()  # always re-evaluate the incumbent

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
        if verbose:
            print(f"gen {gen:2d}  best={fitness[order[0]]:7.3f}  "
                  f"elite_mean={fitness[order[:mu]].mean():7.3f}  "
                  f"incumbent={best_f:7.3f}", flush=True)

    return best_x, best_f, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random', type=int, default=220, help='random search samples')
    parser.add_argument('--generations', type=int, default=18, help='ES generations')
    parser.add_argument('--workers', type=int, default=4, help='parallel processes')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', default=BEST_PATH)
    args = parser.parse_args()

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
    for name, value in zip(PARAM_NAMES, best_x):
        print(f"  {name:8s} = {value:.6f}")

    with open(args.out, 'w') as fh:
        json.dump({'params': best_x.tolist(),
                   'names': list(PARAM_NAMES),
                   'return': best_f,
                   'random_search_best': f0,
                   'random_samples': args.random,
                   'generations': args.generations,
                   'workers': args.workers,
                   'seed': args.seed,
                   'wall_seconds': elapsed,
                   'history': history}, fh, indent=2)
    print(f"wrote {args.out}")
    print("\nNow update DEFAULT_PARAMS in pure_pursuit/controller.py, then run:")
    print("  pixi run python pure_pursuit/evaluate.py --save")


if __name__ == '__main__':
    main()
