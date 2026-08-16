"""Geometric pure-pursuit racing controller.

All of the policy's mathematics lives here so that ``agent_interface.py`` stays a thin
adapter. The controller is a pure function of the current observation - it holds no
state between steps - and depends only on ``numpy``.

It reads 42 of the 456 observation dimensions:

===================  ==========================================================
``obs[0]``           longitudinal velocity, normalised by ``TOP_SPEED``
``obs[1]``           lateral velocity (slip), normalised by ``TOP_SPEED``
``obs[416:456]``     20 centerline points, normalised by ``CENTERLINE_SCALE``,
                     already expressed in the car's frame (x forward, y left)
===================  ==========================================================

The 400 cone dimensions are unused: the track is fixed and the centerline carries the
same information more compactly.
"""

import numpy as np

# --- environment constants, read from CarEnv ----------------------------------------
TOP_SPEED = 40.454584370675605      # m/s, RacingProblem normalises obs[0:2] by this
WHEELBASE = 2.4                     # m, VEH_2CV
MAX_STEER = 0.6108652381980153      # rad, steering_controller.max_angle
CENTERLINE_SCALE = 60.0             # m, the centerline horizon
N_CENTERLINE = 20                   # centerline points in the observation
CENTERLINE_SLICE = slice(416, 456)  # observation indices holding the centerline
MAX_APEX_SHIFT = 3.0                # m, keeps the aim point inside the 8 m track

PARAM_NAMES = (
    'k_ld',     # lookahead gain per m/s
    'ld0',      # base lookahead [m]
    'ldmin',    # lookahead floor [m]
    'ksteer',   # steering gain
    'alat',     # assumed usable lateral acceleration [m/s^2]
    'abrake',   # assumed usable braking deceleration [m/s^2]
    'kp',       # speed P-gain
    'kdamp',    # lateral-slip damping gain
    'vcap',     # hard speed cap [m/s]
    'k_in',     # apex-shift strength
)

# Tuned on track00 by pure_pursuit/tune_es.py; see pure_pursuit/RESULTS.md.
DEFAULT_PARAMS = (
    0.4948805956840358,
    5.106832735968532,
    11.080564530323118,
    1.5704779704472525,
    8.802213734739626,
    4.949815975208273,
    1.6462906523241021,
    0.6237921528600951,
    30.179722478100213,
    0.04547113095916495,
)


def centerline_features(obs):
    """
    Extract track geometry from the raw observation.

    :param obs: raw 456-dim observation
    :return: ``(cl, s, kappa, sgn)`` - ``cl`` (20, 2) points in metres, ``s`` (20,)
        cumulative arc length in metres, ``kappa`` (18,) unsigned Menger curvature in
        1/m, ``sgn`` (18,) turn direction with +1 meaning a left-hand turn
    """
    cl = np.asarray(obs[CENTERLINE_SLICE], dtype=np.float64).reshape(
        N_CENTERLINE, 2) * CENTERLINE_SCALE

    seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])

    # Menger curvature of each consecutive triple: kappa = 4 * area / (|ab||bc||ca|)
    a, b, c = cl[:-2], cl[1:-1], cl[2:]
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(c - a, axis=1)
    cross = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
             - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    kappa = 2.0 * np.abs(cross) / np.maximum(ab * bc * ca, 1e-9)

    return cl, s, kappa, np.sign(cross)


def pure_pursuit_action(obs, params=DEFAULT_PARAMS):
    """
    Compute one control action.

    Steering comes from the pure-pursuit geometry of a bicycle model aimed at a point a
    speed-dependent distance ahead on the centerline, shifted toward the inside of the
    corner. The reference speed is the curvature-limited cornering speed, propagated
    backwards through the available braking deceleration so the car slows *before* a
    corner rather than in it.

    :param obs: raw 456-dim observation
    :param params: 10 gains in the order of :data:`PARAM_NAMES`
    :return: ``(action, v_ref)`` with ``action`` = [steering, throttle, brake], each in
        [-1, 1], and ``v_ref`` the reference speed in m/s
    """
    k_ld, ld0, ldmin, ksteer, alat, abrake, kp, kdamp, vcap, k_in = params

    v = obs[0] * TOP_SPEED
    v_lat = obs[1] * TOP_SPEED

    cl, s, kappa, sgn = centerline_features(obs)

    # --- aim point at a speed-dependent lookahead distance --------------------------
    lookahead = np.clip(k_ld * v + ld0, ldmin, 58.0)
    i = min(max(int(np.searchsorted(s, lookahead)), 1), N_CENTERLINE - 1)
    w = np.clip((lookahead - s[i - 1]) / max(s[i] - s[i - 1], 1e-6), 0.0, 1.0)
    target = cl[i - 1] * (1.0 - w) + cl[i] * w

    # --- shift the aim point toward the inside of the corner (racing line) ----------
    j = min(max(i - 1, 0), len(kappa) - 1)
    tangent = cl[min(i, N_CENTERLINE - 1)] - cl[i - 1]
    normal = np.array([-tangent[1], tangent[0]])
    n_norm = np.linalg.norm(normal)
    if n_norm > 1e-6:
        shift = np.clip(k_in * sgn[j] * kappa[j] * 300.0, -MAX_APEX_SHIFT, MAX_APEX_SHIFT)
        target = target + (normal / n_norm) * shift

    # --- pure-pursuit steering on the bicycle model ---------------------------------
    dist = max(float(np.linalg.norm(target)), 1e-3)
    alpha = np.arctan2(target[1], target[0])
    delta = np.arctan2(2.0 * WHEELBASE * np.sin(alpha), dist)
    steer = (delta / MAX_STEER) * ksteer - kdamp * v_lat / max(v, 3.0)
    steer = float(np.clip(steer, -1.0, 1.0))

    # --- speed profile: cornering limit propagated backwards through braking --------
    v_corner = np.sqrt(alat / np.maximum(kappa, 1e-4))
    v_allowed = np.minimum(
        np.sqrt(np.maximum(v_corner ** 2 + 2.0 * abrake * s[1:-1], 0.0)), TOP_SPEED)
    v_ref = float(min(float(v_allowed.min()), vcap))

    # --- pedals ---------------------------------------------------------------------
    # Each pedal maps [-1, 1] -> [0, 1] in the environment, so [0, 0, 0] means 50%
    # throttle AND 50% brake at once. Mapping through -0.9 keeps exactly one engaged.
    u = float(np.clip(kp * (v_ref - v), -1.0, 1.0))
    if u >= 0.0:
        throttle, brake = 1.8 * u - 0.9, -1.0
    else:
        throttle, brake = -1.0, 1.8 * (-u) - 0.9

    return np.array([steer, throttle, brake], dtype=np.float32), v_ref
