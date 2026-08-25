"""Minimum-curvature racing line and its velocity profile.

Two stages, both standard motorsport trajectory practice:

1. **Line.** The path is parameterised as a lateral offset from the centerline,
   ``p_i = c_i + alpha_i * n_i``, and ``alpha`` is chosen to minimise discrete curvature
   subject to a corridor bound. That is a bound-constrained linear least-squares problem,
   solved exactly rather than by search. Minimum curvature is used instead of shortest
   path because cornering speed goes as ``sqrt(1 / kappa)``, so flattening the corners
   buys more time than cutting distance.

2. **Speed.** A forward-backward pass over the resulting curvature, limited by the
   traction circle ``a_lat^2 + a_long^2 <= a_max^2``. The backward pass makes the car
   brake early enough for the next corner, the forward pass respects what the engine can
   actually deliver. This is what produces a late apex and a strong exit without either
   being asked for explicitly.

    pixi run python raceline/optimize.py
"""

import argparse
import os
import sys

import numpy as np
from scipy.optimize import lsq_linear

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from raceline.track_data import (A_ACCEL, A_BRAKE, A_MAX, CAR_HALF_WIDTH,  # noqa: E402
                                 CONE_OFFSET, ENGINE_POWER, MASS, closed_normals, load)

HERE = os.path.dirname(os.path.abspath(__file__))
LINE_PATH = os.path.join(HERE, 'raceline.npz')

# Staying inside the cones costs a little corner radius but avoids the -0.2 per contact.
# CAR_HALF_WIDTH is subtracted because the penalty triggers on the body, not the centre.
DEFAULT_CORRIDOR = CONE_OFFSET - CAR_HALF_WIDTH - 0.15   # 2.61 m


def second_difference_matrix(n):
    """
    Circulant second-difference operator for a closed loop.

    :param n: number of points
    :return: (n, n) matrix D with ``(D x)_i = x_{i-1} - 2 x_i + x_{i+1}``
    """
    d = np.zeros((n, n))
    i = np.arange(n)
    d[i, i] = -2.0
    d[i, (i - 1) % n] = 1.0
    d[i, (i + 1) % n] = 1.0
    return d


def solve_line(centerline, normals, corridor, smooth=0.0):
    """
    Minimum-curvature lateral offsets within a corridor.

    :param centerline: (n, 2) closed centerline
    :param normals: (n, 2) unit left normals
    :param corridor: scalar or (n,) maximum absolute offset [m]
    :param smooth: weight on an offset-rate penalty, damping high-frequency wiggle
    :return: (n,) offsets in metres
    """
    n = len(centerline)
    d = second_difference_matrix(n)

    # p = c + diag(alpha) n, so D p is affine in alpha with one block per coordinate.
    a = np.vstack([d * normals[:, 0], d * normals[:, 1]])
    b = -np.concatenate([d @ centerline[:, 0], d @ centerline[:, 1]])

    if smooth > 0.0:
        first = np.eye(n) - np.roll(np.eye(n), 1, axis=1)
        a = np.vstack([a, smooth * first])
        b = np.concatenate([b, np.zeros(n)])

    bound = np.broadcast_to(np.asarray(corridor, dtype=float), (n,))
    result = lsq_linear(a, b, bounds=(-bound, bound), tol=1e-10, max_iter=500)
    return result.x


def path_curvature(points):
    """
    Menger curvature of a closed polyline, signed by turn direction.

    :param points: (n, 2) closed polyline
    :return: (n,) signed curvature in 1/m
    """
    prev, nxt = np.roll(points, 1, axis=0), np.roll(points, -1, axis=0)
    ab = np.linalg.norm(points - prev, axis=1)
    bc = np.linalg.norm(nxt - points, axis=1)
    ca = np.linalg.norm(nxt - prev, axis=1)
    cross = ((points[:, 0] - prev[:, 0]) * (nxt[:, 1] - prev[:, 1])
             - (points[:, 1] - prev[:, 1]) * (nxt[:, 0] - prev[:, 0]))
    return 2.0 * cross / np.maximum(ab * bc * ca, 1e-12)


def velocity_profile(kappa, ds, a_max=A_MAX, v_max=None, grip_margin=1.0,
                     power_limited=True, iterations=3, a_accel=A_ACCEL,
                     a_brake=A_BRAKE):
    """
    Speed profile for a closed path, limited separately in each direction.

    Acceleration and braking are not symmetric on this car. Braking acts on both axles
    and reaches ``BRAKE_FORCE / MASS``, while acceleration is front-axle limited because
    the drivetrain is front-wheel drive, so it is roughly a third of that. Using one
    traction circle for both makes the profile ask for acceleration the car cannot
    deliver, and the car then trails its own reference for most of the lap.

    Both limits are still derated by how much grip the corner is already using.

    :param kappa: (n,) signed curvature [1/m]
    :param ds: (n,) arc length from point i to i+1 [m]
    :param a_max: lateral limit, the traction-circle radius [m/s^2]
    :param v_max: optional hard speed cap [m/s]
    :param grip_margin: fraction of ``a_max`` the profile is allowed to use, in (0, 1]
    :param power_limited: also cap acceleration by engine power
    :param iterations: wrap-around passes, needed because the loop is closed
    :param a_accel: longitudinal acceleration limit at zero lateral load [m/s^2]
    :param a_brake: braking limit at zero lateral load [m/s^2]
    :return: (n,) speed [m/s]
    """
    a = a_max * grip_margin
    n = len(kappa)

    # Pure cornering limit: all available grip spent on lateral acceleration.
    v = np.sqrt(a / np.maximum(np.abs(kappa), 1e-6))
    if v_max is not None:
        v = np.minimum(v, v_max)

    for _ in range(iterations):
        # Backward pass: brake early enough to make the next corner.
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            a_lat = v[j] ** 2 * abs(kappa[j])
            free = np.sqrt(max(1.0 - (a_lat / a) ** 2, 0.0))
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2.0 * a_brake * free * ds[i]))

        # Forward pass: accelerate as hard as the front axle and engine allow.
        for i in range(n):
            j = (i + 1) % n
            a_lat = v[i] ** 2 * abs(kappa[i])
            free = np.sqrt(max(1.0 - (a_lat / a) ** 2, 0.0))
            a_long = a_accel * free
            if power_limited:
                a_long = min(a_long, ENGINE_POWER / (MASS * max(v[i], 1.0)))
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2.0 * a_long * ds[i]))

    return v


def lap_time(v, ds):
    """
    :return: time to complete the closed path [s]
    """
    v_mid = 0.5 * (v + np.roll(v, -1))
    return float(np.sum(ds / np.maximum(v_mid, 1e-3)))


def build(corridor=DEFAULT_CORRIDOR, smooth=0.0, grip_margin=1.0, v_max=None,
          a_max=A_MAX, a_accel=A_ACCEL, a_brake=A_BRAKE):
    """
    Solve the line and its speed profile for the cached track.

    :return: dict with the line, its curvature, speeds and diagnostics
    """
    track = load()
    centerline, normals = track['centerline'], track['normals']

    alpha = solve_line(centerline, normals, corridor, smooth)
    line = centerline + alpha[:, None] * normals
    ds = np.linalg.norm(np.roll(line, -1, axis=0) - line, axis=1)
    kappa = path_curvature(line)
    v = velocity_profile(kappa, ds, a_max=a_max, v_max=v_max, grip_margin=grip_margin,
                         a_accel=a_accel, a_brake=a_brake)

    kappa_c = path_curvature(centerline)
    ds_c = np.linalg.norm(np.roll(centerline, -1, axis=0) - centerline, axis=1)
    v_c = velocity_profile(kappa_c, ds_c, a_max=a_max, v_max=v_max,
                           grip_margin=grip_margin, a_accel=a_accel, a_brake=a_brake)

    return {
        'line': line,
        'offset': alpha,
        'normals': closed_normals(line),
        'ds': ds,
        's': np.concatenate([[0.0], np.cumsum(ds)[:-1]]),
        'kappa': kappa,
        'speed': v,
        'lap_time': lap_time(v, ds),
        'length': float(np.sum(ds)),
        'centerline_lap_time': lap_time(v_c, ds_c),
        'centerline_length': float(np.sum(ds_c)),
        'progress_per_lap': float(np.sum(ds_c)),
        'corridor': float(np.max(corridor)),
    }


def save(result, path=LINE_PATH):
    """Store the arrays the run-time controller needs."""
    np.savez_compressed(path, line=result['line'], normals=result['normals'],
                        s=result['s'], ds=result['ds'], kappa=result['kappa'],
                        speed=result['speed'])
    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corridor', type=float, default=DEFAULT_CORRIDOR)
    p.add_argument('--smooth', type=float, default=0.0)
    p.add_argument('--grip-margin', type=float, default=1.0)
    p.add_argument('--v-max', type=float, default=None)
    p.add_argument('--a-max', type=float, default=A_MAX)
    return p.parse_args()


def main():
    args = parse_args()
    r = build(args.corridor, args.smooth, args.grip_margin, args.v_max, args.a_max)

    print(f"corridor +-{r['corridor']:.2f} m, offsets used "
          f"[{r['offset'].min():+.2f}, {r['offset'].max():+.2f}] m, "
          f"mean |offset| {np.abs(r['offset']).mean():.2f} m")
    print(f"length      raceline {r['length']:.1f} m   centerline {r['centerline_length']:.1f} m")
    print(f"peak kappa  raceline {np.abs(r['kappa']).max():.4f}      "
          f"centerline {np.abs(path_curvature(load()['centerline'])).max():.4f} 1/m")
    print(f"speed       mean {r['speed'].mean():.2f}  min {r['speed'].min():.2f}  "
          f"max {r['speed'].max():.2f} m/s")
    print(f"lap time    raceline {r['lap_time']:.2f} s   "
          f"centerline {r['centerline_lap_time']:.2f} s")
    # The reward integrates velocity projected on the track direction, so one lap is
    # worth the centerline length regardless of the path actually driven.
    laps = 60.0 / r['lap_time']
    print(f"projected {laps:.3f} laps in 60 s -> progress "
          f"{laps * r['progress_per_lap']:.0f} m -> return "
          f"{laps * r['progress_per_lap'] * 0.1:.1f}")
    laps_c = 60.0 / r['centerline_lap_time']
    print(f"  centerline for comparison: {laps_c * r['progress_per_lap'] * 0.1:.1f}")
    print(f"wrote {save(r)}")


if __name__ == '__main__':
    main()
