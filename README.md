# BPA III - Car Racing Challenge

## Agent Description

My agent is a geometric pure-pursuit controller with a learned residual on top, trained
with SAC. It scores **90.69** over 10 evaluation episodes on the fixed track, against
83.76 for the controller on its own and 62-64 for the `DSME-DDPG-tuned` baseline.

### Why I split it this way

Plain DDPG from scratch spends most of its budget learning things the track geometry
already tells you: aim at the road, slow down before a corner. That is expensive, and it
is also where the variance comes from, since an early policy that leaves the track gets a
terminal `-1` and learns nothing about the rest of the lap.

So I split the problem. A hand-written controller supplies the part that is plain
geometry, and RL only supplies the part geometry cannot express, mainly tyre slip and how
far the racing line can be pushed before grip runs out. Training then starts at 83.76
instead of at a crash, and every intermediate checkpoint is a usable submission.

### Part 1: the geometric controller

`pure_pursuit/controller.py`. Stateless, numpy only, a pure function of one observation.

Steering is pure pursuit on a bicycle model. I pick an aim point a speed-dependent
distance ahead on the centerline, shift it toward the inside of the corner so the car
takes a racing line rather than crawling down the middle, then solve for the steering
angle. For speed I take the cornering limit, `v = sqrt(a_lat / kappa)` from the Menger
curvature of the centerline, and propagate it backwards through the available braking
deceleration, so the car brakes *before* a corner instead of in it.

That leaves ten gains. I tuned them directly on the track with random search followed by
a `(mu, lambda)` evolution strategy, about four minutes on four laptop cores. Two tuning
runs from different seeds landed on different gains and the same score, so I read 83.76
as a structural limit of this controller form, not a tuning failure.

### Part 2: the residual policy

`residual_sac/`, with the agent classes in `agent_interface.py`.

I used SAC rather than DDPG for the entropy term and the twin critics. The residual is
small, bounded, and highly correlated with the controller's own output, which is exactly
the regime where a deterministic policy stops exploring. The entropy term keeps it from
collapsing.

The policy does not output pedals. It outputs three interpretable corrections that go
back into the controller:

| Channel | Meaning | Authority |
| --- | --- | --- |
| `apex_bias` | lateral shift of the aim point | 1.5 m, against an 8 m track |
| `speed_bias` | offset on the reference speed | 4 m/s, against a ~14 m/s mean |
| `steer_bias` | direct steering correction | 0.15 action units |

Two things fall out of this. A zero residual reproduces the controller exactly, and I
zero-initialise the actor's mean head, so an untrained agent already scores 83.76. And
because the channels are bounded, even a saturated residual is still a legal racing line,
so the policy cannot destroy itself in the first few thousand steps.

### Effective state space

The raw observation is 456-dimensional. My controller reads 42 of those, and the residual
policy reads a 28-dimensional engineered vector:

- **8 curvatures and 8 lateral offsets**, resampled at fixed arc lengths ahead
  (5, 10, 15, 20, 30, 40, 50, 58 m). I resample by arc length instead of using the 20 raw
  centerline points so the input is independent of the car's speed, and the same corner
  looks the same whether it is entered fast or slow.
- **8 vehicle-state values**: longitudinal and lateral velocity, yaw rate, steering angle,
  the two slip flags, longitudinal and lateral acceleration.
- **4 controller values**: the reference speed and the three base actions, so the policy
  can see what it is correcting.

I drop all 400 cone dimensions. The track is fixed, so the centerline carries the same
information in a twentieth of the width, and cones have no physics effect anyway. The
three RGB dimensions are dropped for the obvious reason.

### The one thing that made it work

Residual SAC does not train out of the box here. The actor starts at zero and the critic
starts random, and within about 4k steps the untrained critic drags the actor off the
controller and the run collapses from 83.76 down to roughly 15. Shrinking the residual
authority does not fix it, it only slows the collapse down.

What fixed it is a quadratic penalty on the residual in the actor loss, decaying linearly
to zero over 300k steps. Early on it holds the policy near the controller while the critic
becomes worth listening to, and by the time it has decayed the critic is accurate enough
to trust. With it the run climbs monotonically from the first evaluation onwards.
