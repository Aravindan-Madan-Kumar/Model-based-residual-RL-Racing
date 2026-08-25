"""Sector-tuned racing-line tracker.

Three additions over :mod:`raceline.controller`, each aimed at a loss measured on the
tuned tracker rather than guessed at:

**Per-sector profile.** A single ``a_brake`` has to be safe in the hardest braking zone
on the track, so it gets dragged down everywhere else: tuned globally it landed at 6.17
against a physical 10.7. Splitting the lap into sectors lets the profile brake hard where
that is safe and gently where it is not. The same argument applies to ``v_scale``.

**Launch.** The standing start costs 6.40 points, and front slip is active for 86.7% of
the first six seconds - the car is simply spinning its front wheels. Since the drivetrain
is front-wheel drive, that also costs steering authority. A capped launch throttle, cut
further while the slip flag is set, trades a little peak torque for traction.

**Steering filter.** The environment applies no meaningful steering rate limit
(``linear(60)`` is 6 rad per step against a 0.61 rad lock), so a first-order filter has
to come from the controller. The car's current steering angle is observable at
``obs[3]``, which lets the filter be applied without carrying state between steps.

Sector values are interpolated smoothly around the loop rather than applied as steps: an
abrupt change in reference speed at a sector boundary produces exactly the pedal chatter
the filter is meant to remove.
"""

import numpy as np

TOP_SPEED = 40.454584370675605
WHEELBASE = 2.4
MAX_STEER = 0.6108652381980153
STEER_NORM = 0.61          # obs[3] is the steering angle divided by this
KAPPA_REF = 0.15           # roughly the sharpest corner, used to normalise curvature

N_SECTORS = 8

TRACKER_NAMES = (
    'k_ld',         # lookahead gain per m/s
    'ld0',          # base lookahead [m]
    'ldmin',        # lookahead floor [m]
    'ldmax',        # lookahead ceiling [m]
    'ksteer',       # steering gain
    'kdamp',        # lateral-slip damping gain
    'kp_accel',     # speed gain when below the reference
    'kp_brake',     # speed gain when above it, separate because the car brakes at
                    # 10.7 m/s^2 but accelerates at 3.4-5.2
    't_preview',    # speed preview horizon [s]
    'k_lat',        # cross-track correction gain
    'k_ld_curve',   # how much curvature shortens the lookahead, in [0, 1]
    'steer_alpha',  # first-order filter coefficient, 1 means unfiltered
    'v_post',       # multiplier applied to the profile at lookup, on top of the
                    # per-sector scaling folded into the profile itself. The two are not
                    # equivalent: per-sector scaling is propagated through the braking
                    # passes, while this one is not, so it lets the reference sit above
                    # what the braking schedule planned for. That is physically
                    # inconsistent but it is what the tuned racing-line agent does, and
                    # it scores well, so the search keeps both and decides.
)
LAUNCH_NAMES = (
    'launch_v',     # speed below which launch handling applies [m/s]
    'launch_thr',   # throttle pedal during launch, in [0, 1]
    'launch_slip',  # extra multiplier on that pedal while the front slip flag is set
)
PROFILE_NAMES = ('a_max', 'a_accel', 'smooth')
SECTOR_NAMES = (tuple(f'v_scale{i}' for i in range(N_SECTORS))
                + tuple(f'a_brake{i}' for i in range(N_SECTORS)))
PARAM_NAMES = TRACKER_NAMES + LAUNCH_NAMES + PROFILE_NAMES + SECTOR_NAMES

N_TRACKER = len(TRACKER_NAMES)
N_LAUNCH = len(LAUNCH_NAMES)
N_PROFILE = len(PROFILE_NAMES)


def sector_curve(values, s, total_s):
    """
    Spread per-sector values over the lap as a smooth periodic curve.

    Each value is anchored at its sector's midpoint and linearly interpolated around the
    loop, so neighbouring sectors blend instead of stepping.

    :param values: (k,) one value per sector
    :param s: (n,) arc length of each point [m]
    :param total_s: lap length [m]
    :return: (n,) per-point values
    """
    values = np.asarray(values, dtype=float)
    k = len(values)
    centres = (np.arange(k) + 0.5) * (total_s / k)

    # Wrap by padding with the neighbouring sector on each side.
    padded_c = np.concatenate([[centres[-1] - total_s], centres,
                               [centres[0] + total_s]])
    padded_v = np.concatenate([[values[-1]], values, [values[0]]])
    return np.interp(np.mod(s, total_s), padded_c, padded_v)


def split_params(params):
    """
    :param params: the full flat parameter vector
    :return: ``(tracker, launch, profile, v_scale, a_brake)``
    """
    params = np.asarray(params, dtype=float)
    i = N_TRACKER
    j = i + N_LAUNCH
    k = j + N_PROFILE
    return (params[:i], params[i:j], params[j:k],
            params[k:k + N_SECTORS], params[k + N_SECTORS:k + 2 * N_SECTORS])


class SectorTracker:
    """
    Racing-line tracker with per-sector speed and braking.

    Holds plain arrays only, so it pickles cleanly and needs numpy alone.

    :param line: (n, 2) racing line in world coordinates
    :param s: (n,) cumulative arc length [m]
    :param speed: (n,) reference speed [m/s]
    :param kappa: (n,) signed curvature [1/m]
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

    def _index_at(self, s_query):
        """
        :return: index of the point at an arc length, wrapping around the loop
        """
        return int(np.searchsorted(self.s, np.mod(s_query, self.total_s))) % len(self.s)

    def point_at(self, s_query):
        """
        :param s_query: arc length [m]
        :return: (2,) interpolated world position
        """
        s_wrapped = np.mod(s_query, self.total_s)
        i = int(np.searchsorted(self.s, s_wrapped))
        i0, i1 = (i - 1) % len(self.s), i % len(self.s)
        span = self.s[i1] - self.s[i0] if i1 > i0 else self.total_s - self.s[i0]
        w = 0.0 if span <= 1e-9 else np.clip((s_wrapped - self.s[i0]) / span, 0.0, 1.0)
        return self.line[i0] * (1.0 - w) + self.line[i1] * w

    def action(self, obs, params, apex_bias=0.0, speed_bias=0.0, steer_bias=0.0):
        """
        Compute one control action.

        :param obs: raw 456-dim observation
        :param params: the full flat parameter vector
        :param apex_bias: lateral shift of the aim point [m], positive to the left
        :param speed_bias: offset on the reference speed [m/s]
        :param steer_bias: additional steering command, before filtering and clipping
        :return: ``(action, v_ref)``
        """
        tracker, launch, _, _, _ = split_params(params)
        (k_ld, ld0, ldmin, ldmax, ksteer, kdamp, kp_accel, kp_brake, t_preview,
         k_lat, k_ld_curve, steer_alpha, v_post) = tracker
        launch_v, launch_thr, launch_slip = launch

        v = obs[0] * TOP_SPEED
        v_lat = obs[1] * TOP_SPEED
        steer_now = float(obs[3])
        position = np.asarray(obs[4:6], dtype=np.float64)
        theta = float(obs[6])
        front_slip = float(obs[11])

        i = self.nearest_index(position)
        s_here = self.s[i]

        # --- aim point, pulled in through corners ----------------------------------
        # A lookahead long enough to be stable on a straight cuts the corner on a sharp
        # one, which is the dominant tracking error on this line.
        curve = min(abs(self.kappa[i]) / KAPPA_REF, 1.0)
        lookahead = (k_ld * v + ld0) * (1.0 - k_ld_curve * curve)
        lookahead = float(np.clip(lookahead, ldmin, max(ldmax, ldmin)))
        target = self.point_at(s_here + lookahead)

        # --- into the car frame: x forward, y left ---------------------------------
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        delta_world = target - position
        tx = cos_t * delta_world[0] + sin_t * delta_world[1]
        ty = -sin_t * delta_world[0] + cos_t * delta_world[1] + apex_bias

        to_car = position - self.line[i]
        tangent = self.line[(i + 1) % len(self.line)] - self.line[i - 1]
        tangent = tangent / max(np.linalg.norm(tangent), 1e-9)
        cross_track = -tangent[1] * to_car[0] + tangent[0] * to_car[1]

        # --- pure-pursuit steering on the bicycle model ----------------------------
        dist = max(float(np.hypot(tx, ty)), 1e-3)
        alpha = np.arctan2(ty, tx)
        steer_angle = np.arctan2(2.0 * WHEELBASE * np.sin(alpha), dist)
        steer_raw = ((steer_angle / MAX_STEER) * ksteer
                     - kdamp * v_lat / max(v, 3.0)
                     - k_lat * cross_track
                     + steer_bias)

        # First-order filter toward the command, using the wheel's current angle as the
        # state so the controller stays a pure function of the observation.
        steer = steer_now + steer_alpha * (steer_raw - steer_now)
        steer = float(np.clip(steer, -1.0, 1.0))

        # --- speed reference from the profile, previewed ---------------------------
        v_ref = float(self.speed[self._index_at(s_here + max(v * t_preview, 2.0))])
        v_ref = float(np.clip(v_ref * v_post + speed_bias, 0.0, TOP_SPEED))

        # --- pedals -----------------------------------------------------------------
        # Each pedal maps [-1, 1] -> [0, 1], so [0, 0, 0] is 50% throttle AND 50% brake.
        # Mapping through -0.9 keeps exactly one engaged.
        if v < launch_v:
            # Front-wheel drive: spinning the fronts costs cornering grip as well as
            # traction, so the launch gives away peak torque to keep them gripping.
            pedal = launch_thr * (launch_slip if front_slip > 0.5 else 1.0)
            throttle, brake = 1.8 * float(np.clip(pedal, 0.0, 1.0)) - 0.9, -1.0
        else:
            error = v_ref - v
            gain = kp_accel if error >= 0.0 else kp_brake
            u = float(np.clip(gain * error, -1.0, 1.0))
            if u >= 0.0:
                throttle, brake = 1.8 * u - 0.9, -1.0
            else:
                throttle, brake = -1.0, 1.8 * (-u) - 0.9

        return np.array([steer, throttle, brake], dtype=np.float32), v_ref


def build_profile(params, corridor=None, lines=None):
    """
    Solve the line and its per-sector speed profile for one parameter vector.

    :param params: the full flat parameter vector
    :param corridor: corridor half-width [m], defaulting to the cone-clearing value
    :param lines: optional dict used to cache solved lines by smoothing weight. Only the
        smoothing weight changes the line, and solving it is a least-squares problem over
        every track point, so reusing it across candidates is most of the run time.
    :return: dict with ``line``, ``s``, ``speed`` and ``kappa``
    """
    from raceline.optimize import DEFAULT_CORRIDOR, build, velocity_profile

    _, _, profile, v_scale, a_brake = split_params(params)
    a_max, a_accel, smooth = profile

    corridor = DEFAULT_CORRIDOR if corridor is None else corridor
    key = (round(float(smooth), 4), round(float(corridor), 4))
    if lines is None:
        line = build(corridor=corridor, smooth=float(smooth))
    else:
        if key not in lines:
            lines[key] = build(corridor=corridor, smooth=float(smooth))
        line = lines[key]
    total_s = float(line['s'][-1] + line['ds'][-1])
    speed = velocity_profile(
        line['kappa'], line['ds'], a_max=float(a_max), a_accel=float(a_accel),
        a_brake=sector_curve(a_brake, line['s'], total_s),
        v_scale=sector_curve(v_scale, line['s'], total_s))
    return {'line': line['line'], 's': line['s'], 'speed': speed,
            'kappa': line['kappa']}
