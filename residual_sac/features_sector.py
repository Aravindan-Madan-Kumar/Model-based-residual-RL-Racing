"""Compact state representation for the residual on the sector controller.

Reduces the 456-dimensional observation to :data:`FEATURE_DIM` values describing the
track ahead and how much grip the car has left, and deliberately omits the cross-track
error.

Omitting it is the point. The reward pays only for distance covered along the track, so
nothing in the objective rewards staying on the precomputed line - but the residual is
initialised as a copy of a controller whose whole purpose is to drive cross-track error
to zero, and cancelling that error faster is a smooth, immediately rewarded pattern
while carrying more speed pays late and risks a terminal. Handed the error signal
directly, the policy has an easy gradient toward becoming a better tracker, which is the
one thing the controller is already good at. What it is shown instead is the car's
physical margin - slip flags, accelerations against the measured limits, curvature
ahead, and its own speed error against the profile - so the available question is "is
there grip left here", not "am I on the line".

The 400 cone dimensions are dropped: on a fixed track the racing line carries the same
information.
"""

import numpy as np

TOP_SPEED = 40.454584370675605

# Arc lengths [m] ahead at which the line's curvature is sampled.
STATIONS = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 58.0)

# Residual channels, in order. Speed carries the larger share of the authority: exit
# speed dominates lap time on this car, whose acceleration is a third of its braking.
RESIDUAL_DIM = 2
SPEED_SCALE = 2.5    # m/s, against a ~16 m/s mean speed
STEER_SCALE = 0.08   # action units, against the controller's own filtered command

# Third channel, used only when the agent is built with ``channels=3``. Shifts the aim
# point sideways, which is what gives the residual authority over the line's shape rather
# than only over how it is driven. Bounded well inside the solved line's own excursion,
# whose widest point is 2.10 m against a 2.42 m zero-margin corridor.
APEX_SCALE = 1.0     # m

# Normalisation constants. The acceleration figures are the limits measured from
# rollouts, so a feature near 1.0 means the car is at the edge of adhesion.
_KAPPA_GAIN = 20.0        # scales 1/m curvature into roughly unit range
_HEADING_GAIN = 0.5       # rad per unit
_SPEED_ERR_GAIN = 5.0     # m/s per unit
_A_LAT_REF = 10.3         # m/s^2, measured lateral limit
_A_LONG_REF = 10.7        # m/s^2, measured braking limit
_A_COMBINED_REF = 11.2    # m/s^2, measured combined peak

FEATURE_DIM = len(STATIONS) + 15


def residual_features(obs, tracker, params):
    """
    Build the residual policy's input vector.

    :param obs: raw 456-dim observation
    :param tracker: a :class:`raceline.sector.SectorTracker` over the racing line
    :param params: the flat sector parameter vector
    :return: (:data:`FEATURE_DIM`,) float32 ndarray
    """
    position = np.asarray(obs[4:6], dtype=np.float64)
    theta = float(obs[6])
    v = float(obs[0]) * TOP_SPEED

    i = tracker.nearest_index(position)
    s_here = tracker.s[i]

    # Curvature of the line ahead, signed, at fixed arc lengths.
    kappa = np.array([tracker.kappa[tracker._index_at(s_here + d)] for d in STATIONS])

    # Heading of the car relative to the line's tangent. Says whether the car is settled
    # enough to use the grip it has, without revealing where the line is.
    tangent = tracker.line[(i + 1) % len(tracker.line)] - tracker.line[i - 1]
    heading_error = np.arctan2(
        np.sin(np.arctan2(tangent[1], tangent[0]) - theta),
        np.cos(np.arctan2(tangent[1], tangent[0]) - theta))

    base_action, v_ref = tracker.action(obs, params)

    a_long, a_lat = float(obs[13]), float(obs[14])
    combined = float(np.hypot(a_long, a_lat))

    features = np.concatenate([
        kappa * _KAPPA_GAIN,
        [heading_error / _HEADING_GAIN],
        [(v - v_ref) / _SPEED_ERR_GAIN],
        [obs[0], obs[1], obs[3]],           # speed, lateral speed, steering angle
        [obs[2]],                           # wheel angular velocity, already near unit range
        [obs[11], obs[12]],                 # front slip, rear slip flags
        [a_long / _A_LONG_REF, a_lat / _A_LAT_REF],
        [combined / _A_COMBINED_REF],       # how much of the friction circle is used
        [v_ref / TOP_SPEED],
        base_action,
    ]).astype(np.float32)

    assert features.shape == (FEATURE_DIM,), features.shape
    return features
