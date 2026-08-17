"""Compact state representation for the residual policy.

Reduces the 456-dimensional observation to :data:`FEATURE_DIM` values: signed track
curvature and lateral offset resampled at fixed arc lengths ahead, the vehicle's dynamic
state, and the base controller's own output. The 400 cone dimensions are dropped, since
on a fixed track the centerline carries the same information.
"""

import numpy as np

from pure_pursuit.controller import centerline_features, pure_pursuit_action

# Arc lengths [m] at which track geometry is sampled.
STATIONS = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 58.0)
FEATURE_DIM = 2 * len(STATIONS) + 12
RESIDUAL_DIM = 3

# Residual authority. Bounded so the correction cannot override the geometric
# controller. At full authority a saturated residual leaves the track within a few
# steps, which fills early replay with terminal transitions.
APEX_SCALE = 1.5     # m,   against an 8 m track width
SPEED_SCALE = 4.0    # m/s, against a ~14 m/s mean speed
STEER_SCALE = 0.15   # action units

_CURVATURE_GAIN = 20.0   # scales kappa into roughly unit range
_OFFSET_GAIN = 20.0      # metres per unit for the lateral offset
_YAW_GAIN = 5.0          # rad/s per unit
_ACCEL_GAIN = 10.0       # m/s^2 per unit
_TOP_SPEED = 40.454584370675605


def residual_features(obs, params):
    """
    Build the residual policy's input vector.

    :param obs: raw 456-dim observation
    :param params: the 10 base controller gains
    :return: (:data:`FEATURE_DIM`,) float32 ndarray
    """
    cl, s, kappa, sgn = centerline_features(obs)
    signed_kappa = kappa * sgn

    stations = np.asarray(STATIONS)
    # Curvature is defined on interior triples, so it lives at s[1:-1].
    kap = np.interp(stations, s[1:-1], signed_kappa,
                    left=signed_kappa[0], right=signed_kappa[-1])
    lat = np.interp(stations, s, cl[:, 1], left=cl[0, 1], right=cl[-1, 1])

    base_action, v_ref = pure_pursuit_action(obs, params)

    return np.concatenate([
        kap * _CURVATURE_GAIN,
        lat / _OFFSET_GAIN,
        [obs[0], obs[1], obs[2] / _YAW_GAIN, obs[3]],   # speed, slip, yaw rate, steering
        [obs[11], obs[12]],                             # front slip, rear slip flags
        [obs[13] / _ACCEL_GAIN, obs[14] / _ACCEL_GAIN],  # longitudinal, lateral accel
        [v_ref / _TOP_SPEED],
        base_action,
    ]).astype(np.float32)
