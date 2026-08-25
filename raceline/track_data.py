"""Track geometry for the fixed evaluation track, cached to disk.

The track never changes: ``CarEnv/Env.py`` loads ``SavedTracks/test_track/track00.obj``
and ignores the reset seed. So the geometry is extracted once here and stored as an
``.npz``, which lets the raceline solver and the offline tools work without constructing
an environment.

    pixi run python raceline/track_data.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK_PATH = os.path.join(HERE, 'track00.npz')

# Physical constants read from CarEnv/Configs.py (RACING_FAST + VEH_2CV). They are not
# observable at run time, so they are recorded here alongside the geometry.
MAX_GRIP = 0.7              # friction coefficient in the config
GRAVITY = 9.81

# The tyre model saturates above a plain mu*g friction circle, so the usable limit is
# measured from a rollout rather than derived: the tuned agent sustains 10.3 m/s^2
# lateral and 11.2 m/s^2 combined. mu*g would be 6.87, which underestimates by a third.
A_MAX = 10.3
MASS = 700.0
ENGINE_POWER = 60_000.0     # W
BRAKE_FORCE = 7_500.0       # N
CAR_HALF_WIDTH = 1.48 / 2   # collision_bb
CONE_OFFSET = 3.5           # cone_width 7.0, so cones sit at +-3.5 m
TRACK_HALF_WIDTH = 4.0      # track_width 8.0


def resample_closed(points, n):
    """
    Resample a closed polyline to ``n`` points evenly spaced in arc length.

    :param points: (m, 2) polyline, not repeating the first point at the end
    :param n: number of output points
    :return: (n, 2) resampled polyline
    """
    loop = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(target, s, loop[:, 0]),
                     np.interp(target, s, loop[:, 1])], axis=1)


def closed_normals(points):
    """
    Unit left-hand normals of a closed polyline, via central differences.

    :param points: (n, 2) closed polyline
    :return: (n, 2) unit normals, positive to the left of travel
    """
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    return normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)


def extract(n_points=400):
    """
    Build the track geometry by constructing the environment once.

    :param n_points: centerline resolution after resampling
    :return: dict of arrays, also written to :data:`TRACK_PATH`
    """
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from util import create_env

    env = create_env(42)
    env.reset(seed=42)
    track = env.unwrapped.problem.track_dict

    centerline = resample_closed(np.asarray(track['centerline'], dtype=np.float64),
                                 n_points)

    # The stored centerline is wound opposite to the driving direction, which would put
    # every aim point behind the car. Orient it by the start pose so that increasing
    # index always means forward.
    start_xy = np.asarray(track['start_xy'], dtype=np.float64)
    forward = np.array([np.cos(track['start_theta']), np.sin(track['start_theta'])])
    i0 = int(np.argmin(np.linalg.norm(centerline - start_xy, axis=1)))
    tangent = centerline[(i0 + 1) % n_points] - centerline[i0 - 1]
    if float(forward @ tangent) < 0.0:
        centerline = centerline[::-1].copy()

    normals = closed_normals(centerline)
    seg = np.linalg.norm(np.roll(centerline, -1, axis=0) - centerline, axis=1)

    data = {
        'centerline': centerline,
        'normals': normals,
        'seg_length': seg,
        'length': np.float64(track['length']),
        'width': np.float64(track['width']),
        'cone_pos': np.asarray(track['cone_pos'], dtype=np.float64),
        'cone_type': np.asarray(track['cone_type']),
        'start_xy': np.asarray(track['start_xy'], dtype=np.float64),
        'start_theta': np.float64(track['start_theta']),
    }
    np.savez_compressed(TRACK_PATH, **data)
    return data


def load():
    """
    :return: dict of track arrays, extracting them first if the cache is missing
    """
    if not os.path.exists(TRACK_PATH):
        return extract()
    with np.load(TRACK_PATH) as fh:
        return {k: fh[k] for k in fh.files}


if __name__ == '__main__':
    d = extract()
    print(f"centerline {d['centerline'].shape}, length {float(d['length']):.2f} m, "
          f"width {float(d['width']):.1f} m")
    print(f"cones {d['cone_pos'].shape[0]}, spacing "
          f"{float(np.median(np.linalg.norm(np.diff(d['cone_pos'], axis=0), axis=1))):.2f} m")
    print(f"a_max {A_MAX:.3f} m/s^2, corridor half-width for cone clearance "
          f"{CONE_OFFSET - CAR_HALF_WIDTH:.2f} m")
    print(f"wrote {TRACK_PATH}")
