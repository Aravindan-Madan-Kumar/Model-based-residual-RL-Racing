"""Racing-line tracking controller.

The line and its speed profile are solved offline by :mod:`raceline.optimize`, so at run
time this only has to follow them. Because the evaluation track is fixed, the car's
global pose is enough to locate it on the line - no online planning is needed.

It reads 7 of the 456 observation dimensions:

===================  ==========================================================
``obs[0:2]``         longitudinal and lateral velocity, normalised by ``TOP_SPEED``
``obs[4:6]``         global position
``obs[6]``           global yaw
``obs[13:15]``       longitudinal and lateral acceleration
===================  ==========================================================

Steering is pure pursuit against the racing line rather than the centerline. The speed
reference is read from the precomputed profile a short preview distance ahead, so the car
begins braking on the profile's schedule instead of reacting to curvature it can already
see coming.

Stateless: every output is a pure function of one observation.
"""

import numpy as np

TOP_SPEED = 40.454584370675605
WHEELBASE = 2.4
MAX_STEER = 0.6108652381980153

PARAM_NAMES = (
    'k_ld',       # lookahead gain per m/s
    'ld0',        # base lookahead [m]
    'ldmin',      # lookahead floor [m]
    'ldmax',      # lookahead ceiling [m]
    'ksteer',     # steering gain
    'kdamp',      # lateral-slip damping gain
    'kp',         # speed P-gain
    'v_scale',    # multiplier on the profiled reference speed
    't_preview',  # speed preview horizon [s]
    'k_lat',      # cross-track correction gain, feeding back offset from the line
)

# Starting point for the tuner. Deliberately conservative on v_scale: the profile is
# solved for a point mass, so the first evaluation should undershoot rather than slide.
DEFAULT_PARAMS = (
    0.55,   # k_ld
    4.0,    # ld0
    6.0,    # ldmin
    30.0,   # ldmax
    1.0,    # ksteer
    0.5,    # kdamp
    1.5,    # kp
    0.95,   # v_scale
    0.5,    # t_preview
    0.15,   # k_lat
)


class RacelineTracker:
    """
    Follows a precomputed racing line.

    Holds only plain arrays so it pickles cleanly and depends on numpy alone.

    :param line: (n, 2) racing line points in world coordinates
    :param s: (n,) cumulative arc length along the line [m]
    :param speed: (n,) reference speed at each point [m/s]
    :param kappa: (n,) signed curvature at each point [1/m]
    """

    def __init__(self, line, s, speed, kappa):
        self.line = np.asarray(line, dtype=np.float64)
        self.s = np.asarray(s, dtype=np.float64)
        self.speed = np.asarray(speed, dtype=np.float64)
        self.kappa = np.asarray(kappa, dtype=np.float64)
        self.total_s = float(self.s[-1] + np.linalg.norm(self.line[0] - self.line[-1]))

    def nearest_index(self, position):
        """
        :param position: (2,) world position
        :return: index of the closest racing-line point
        """
        return int(np.argmin(np.sum((self.line - position) ** 2, axis=1)))

    def point_at(self, s_query):
        """
        Interpolate the line at an arc length, wrapping around the closed loop.

        :param s_query: arc length [m]
        :return: (2,) world position
        """
        s_wrapped = np.mod(s_query, self.total_s)
        i = int(np.searchsorted(self.s, s_wrapped))
        i0, i1 = (i - 1) % len(self.s), i % len(self.s)
        span = self.s[i1] - self.s[i0] if i1 > i0 else self.total_s - self.s[i0]
        w = 0.0 if span <= 1e-9 else np.clip((s_wrapped - self.s[i0]) / span, 0.0, 1.0)
        return self.line[i0] * (1.0 - w) + self.line[i1] * w

    def speed_at(self, s_query):
        """
        :param s_query: arc length [m]
        :return: reference speed [m/s]
        """
        s_wrapped = np.mod(s_query, self.total_s)
        i = int(np.searchsorted(self.s, s_wrapped)) % len(self.s)
        return float(self.speed[i])

    def action(self, obs, params=DEFAULT_PARAMS, apex_bias=0.0, speed_bias=0.0,
               steer_bias=0.0):
        """
        Compute one control action.

        :param obs: raw 456-dim observation
        :param params: 10 gains in the order of :data:`PARAM_NAMES`
        :param apex_bias: lateral shift of the aim point [m], positive to the left
        :param speed_bias: offset on the reference speed [m/s]
        :param steer_bias: additional steering command, before clipping
        :return: ``(action, v_ref)``
        """
        (k_ld, ld0, ldmin, ldmax, ksteer, kdamp, kp, v_scale, t_preview,
         k_lat) = params

        v = obs[0] * TOP_SPEED
        v_lat = obs[1] * TOP_SPEED
        position = np.asarray(obs[4:6], dtype=np.float64)
        theta = float(obs[6])

        i = self.nearest_index(position)
        s_here = self.s[i]

        # --- aim point on the racing line ------------------------------------------
        lookahead = float(np.clip(k_ld * v + ld0, ldmin, ldmax))
        target = self.point_at(s_here + lookahead)

        # --- into the car frame: x forward, y left ---------------------------------
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        delta_world = target - position
        tx = cos_t * delta_world[0] + sin_t * delta_world[1]
        ty = -sin_t * delta_world[0] + cos_t * delta_world[1]

        # Cross-track error, signed, so the controller closes on the line rather than
        # merely pointing at it. Pure pursuit alone leaves a steady-state offset in long
        # constant-radius corners.
        to_car = position - self.line[i]
        tangent = self.line[(i + 1) % len(self.line)] - self.line[i - 1]
        tangent = tangent / max(np.linalg.norm(tangent), 1e-9)
        cross_track = -tangent[1] * to_car[0] + tangent[0] * to_car[1]

        ty = ty + apex_bias

        # --- pure-pursuit steering on the bicycle model ----------------------------
        dist = max(float(np.hypot(tx, ty)), 1e-3)
        alpha = np.arctan2(ty, tx)
        steer_angle = np.arctan2(2.0 * WHEELBASE * np.sin(alpha), dist)
        steer = ((steer_angle / MAX_STEER) * ksteer
                 - kdamp * v_lat / max(v, 3.0)
                 - k_lat * cross_track
                 + steer_bias)
        steer = float(np.clip(steer, -1.0, 1.0))

        # --- speed reference from the profile, previewed ---------------------------
        preview = s_here + max(v * t_preview, 2.0)
        v_ref = self.speed_at(preview) * v_scale + speed_bias
        v_ref = float(np.clip(v_ref, 0.0, TOP_SPEED))

        # --- pedals -----------------------------------------------------------------
        # Each pedal maps [-1, 1] -> [0, 1], so [0, 0, 0] is 50% throttle AND 50% brake.
        # Mapping through -0.9 keeps exactly one engaged.
        u = float(np.clip(kp * (v_ref - v), -1.0, 1.0))
        if u >= 0.0:
            throttle, brake = 1.8 * u - 0.9, -1.0
        else:
            throttle, brake = -1.0, 1.8 * (-u) - 0.9

        return np.array([steer, throttle, brake], dtype=np.float32), v_ref
