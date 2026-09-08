"""Compare the driven paths and per-sector pace of the three tuned agents.

Rolls each agent through the grading path, records position, speed and reward per step,
and attributes every step to a sector by the nearest point on the sector controller's
racing line, so all three are partitioned by the same geometry.

    pixi run python raceline/scripts/plot_compare.py
"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from agent_interface import convert_action, convert_obs  # noqa: E402
from raceline.scripts.install import make_agent, make_sector_agent  # noqa: E402
from raceline.sector import N_SECTORS, TOP_SPEED  # noqa: E402
from raceline.track_data import load as load_track  # noqa: E402
from util import create_env, load_model  # noqa: E402

OUT = os.path.join(REPO_ROOT, 'notes', 'sector_compare.png')
DT = 0.1   # environment step, 600 steps to the 60 s episode limit

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_MUTED = '#52514e'
GRID = '#e2e1dd'
# Fixed categorical order, validated for CVD separation against the light surface.
SERIES = ('#2a78d6', '#eb6834', '#1baf7a')
# Single-hue sequential ramp, light to dark, for shading a sector by time gained. Kept
# neutral: a hue here would compete with the series colours drawn on top of it.
GAIN_RAMP = LinearSegmentedColormap.from_list(
    'gain', ['#f4f3f0', '#dcdbd6', '#c1c0ba', '#a3a29b', '#84837c'])


def rollout(agent, env):
    """
    :return: dict of per-step arrays over one episode
    """
    obs, _ = env.reset(seed=42)
    xy, speed, reward = [], [], []
    done = False
    while not done:
        xy.append(obs[4:6].copy())
        speed.append(obs[0] * TOP_SPEED)
        obs, r, terminated, truncated, _ = env.step(
            convert_action(agent.get_action(convert_obs(obs))))
        reward.append(r)
        done = terminated or truncated
    return {'xy': np.asarray(xy), 'speed': np.asarray(speed),
            'reward': np.asarray(reward)}


def assign_sectors(xy, line, s, total_s):
    """
    Attribute each step to a sector by the nearest point on the reference line.

    :return: ``(sector, s_here)``, the sector index and arc length at each step
    """
    d = np.sum((xy[:, None, :] - line[None, :, :]) ** 2, axis=2)
    s_here = s[np.argmin(d, axis=1)]
    sector = np.minimum((s_here / total_s * N_SECTORS).astype(int), N_SECTORS - 1)
    return sector, s_here


def unwrap_progress(s_here, total_s):
    """
    Turn the wrapping arc length into monotone distance along the lap.

    Gives every agent the same x-axis, so a sector boundary sits at one place for all of
    them. Distance actually driven does not: a wider line covers more ground for the same
    progress and slides its own boundaries to the right.

    :return: (n,) lap progress [m]
    """
    step = (np.diff(s_here) + total_s / 2) % total_s - total_s / 2
    return np.cumsum(np.concatenate([[0.0], step]))


def first_lap_end(progress, total_s):
    """
    :param progress: (n,) unwrapped lap progress [m]
    :return: index one past the last step of the first lap
    """
    done = np.flatnonzero(progress >= total_s)
    return int(done[0]) + 1 if len(done) else len(progress)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default=OUT)
    return p.parse_args()


def main():
    args = parse_args()
    env = create_env(42)

    agents = [('sector', make_sector_agent()),
              ('raceline', make_agent()),
              ('residual SAC',
               load_model(os.path.join(REPO_ROOT, 'runs', 'seed1', 'best.obj')))]

    ref = agents[0][1]
    line, s = ref.line, ref.s
    total_s = float(s[-1] + np.linalg.norm(line[0] - line[-1]))

    runs = []
    for (name, agent), colour in zip(agents, SERIES):
        r = rollout(agent, env)
        r['sector'], s_here = assign_sectors(r['xy'], line, s, total_s)
        r['progress'] = unwrap_progress(s_here, total_s)
        r['lap_end'] = first_lap_end(r['progress'], total_s)
        r['name'], r['colour'] = name, colour
        r['return'] = float(r['reward'].sum())
        runs.append(r)
        print(f"{name:14s} return {r['return']:7.3f}   lap {r['lap_end'] * DT:5.2f} s"
              f"   mean speed {r['speed'].mean():5.2f} m/s")

    # Per-sector pace on the first lap only. Total return over the episode is not a
    # per-sector pace measure: the run is cut off mid-lap by the 600-step limit, so a
    # faster agent simply visits the late sectors more often and the extra visits land
    # in whichever sectors the cutoff happens to fall in.
    table = np.zeros((len(runs), N_SECTORS))
    for i, r in enumerate(runs):
        sec = r['sector'][:r['lap_end']]
        for k in range(N_SECTORS):
            table[i, k] = np.count_nonzero(sec == k) * DT
    delta = table[1] - table[0]      # time the sector agent gains on the racing line

    print("\nfirst-lap time per sector [s]")
    print("sector " + ''.join(f"{k:>9d}" for k in range(N_SECTORS)) + "     total")
    for i, r in enumerate(runs):
        print(f"{r['name']:14s}" + ''.join(f"{v:9.2f}" for v in table[i])
              + f"{table[i].sum():10.2f}")
    print(f"{'gain vs racel':14s}" + ''.join(f"{v:9.2f}" for v in delta)
          + f"{delta.sum():10.2f}")

    fig = plt.figure(figsize=(16, 8.4), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], hspace=0.62, wspace=0.16,
                          left=0.04, right=0.975, top=0.90, bottom=0.07)
    ax = fig.add_subplot(gs[:, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_v = fig.add_subplot(gs[2, 1])
    for a in (ax, ax_t, ax_d, ax_v):
        a.set_facecolor(SURFACE)
        for side in ('top', 'right'):
            a.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=INK_MUTED, labelsize=9)

    # --- track, sectors shaded by the time the sector agent gains there --------------
    norm = plt.Normalize(0.0, max(delta.max(), 1e-6))
    for k in range(N_SECTORS):
        lo, hi = k / N_SECTORS * total_s, (k + 1) / N_SECTORS * total_s
        seg = line[(s >= lo) & (s <= hi)]
        ax.plot(seg[:, 0], seg[:, 1], lw=16, alpha=0.85, solid_capstyle='butt',
                color=GAIN_RAMP(norm(max(delta[k], 0.0))), zorder=0)
        mid = line[np.argmin(np.abs(s - 0.5 * (lo + hi)))]
        ax.annotate(f"S{k}\n{delta[k]:+.2f}s", mid, fontsize=8.5, color=INK,
                    ha='center', va='center', zorder=4, linespacing=1.3,
                    bbox=dict(boxstyle='round,pad=0.3', fc=SURFACE, ec=GRID, lw=0.8))

    ax.plot(*load_track()['centerline'].T, color='#c9c8c4', lw=1.0, ls=(0, (5, 4)),
            zorder=1)
    for r in runs:
        ax.plot(r['xy'][:, 0], r['xy'][:, 1], color=r['colour'], lw=2.0, alpha=0.95,
                zorder=3, solid_capstyle='round')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]', color=INK_MUTED, fontsize=9)
    ax.set_ylabel('y [m]', color=INK_MUTED, fontsize=9)
    ax.set_title('driven paths, sectors shaded by time gained over the racing line',
                 color=INK, fontsize=11.5, loc='left', pad=10)

    # Direct labels rather than a legend box: the aqua slot sits under 3:1 against the
    # surface, so identity cannot rest on the swatch alone.
    for r, dy in zip(runs, (26, 14, 2)):
        ax.annotate(f"{r['name']}   {r['return']:.2f}", (0.035, 0.0), fontsize=10,
                    color=r['colour'], weight='bold', xycoords='axes fraction',
                    textcoords='offset points', xytext=(0, dy + 6))

    # --- first-lap time per sector ---------------------------------------------------
    w, idx = 0.26, np.arange(N_SECTORS)
    for i, r in enumerate(runs):
        ax_t.bar(idx + (i - 1) * w, table[i], w * 0.92, color=r['colour'],
                 label=r['name'], zorder=2)
    ax_t.set_xticks(idx, [f"S{k}" for k in idx])
    ax_t.set_ylabel('time [s]', color=INK_MUTED, fontsize=9)
    ax_t.set_title('first-lap time by sector, lower is faster', color=INK,
                   fontsize=11.5, loc='left', pad=8)
    ax_t.legend(fontsize=8.5, frameon=False, ncol=3, loc='upper right',
                labelcolor=INK_MUTED)
    ax_t.grid(axis='y', color=GRID, lw=0.8, zorder=0)
    ax_t.set_axisbelow(True)

    # --- where the gain comes from ---------------------------------------------------
    ax_d.bar(idx, delta, 0.6, zorder=2,
             color=[SERIES[0] if v >= 0 else SERIES[1] for v in delta])
    for k, v in enumerate(delta):
        ax_d.annotate(f"{v:+.2f}", (k, v), fontsize=8.5, color=INK_MUTED, ha='center',
                      va='bottom' if v >= 0 else 'top', textcoords='offset points',
                      xytext=(0, 3 if v >= 0 else -3))
    ax_d.axhline(0, color=INK_MUTED, lw=1.0, zorder=3)
    ax_d.set_xticks(idx, [f"S{k}" for k in idx])
    ax_d.set_ylabel('gain [s]', color=INK_MUTED, fontsize=9)
    ax_d.set_title(f"time the sector agent gains, {delta.sum():+.2f} s over the lap",
                   color=INK, fontsize=11.5, loc='left', pad=8)
    ax_d.margins(y=0.28)
    ax_d.grid(axis='y', color=GRID, lw=0.8, zorder=0)
    ax_d.set_axisbelow(True)

    # --- speed against lap progress --------------------------------------------------
    for k in range(1, N_SECTORS):
        ax_v.axvline(k / N_SECTORS * total_s, color=GRID, lw=0.9, zorder=0)
    for r in runs:
        n = r['lap_end']
        ax_v.plot(r['progress'][:n], r['speed'][:n], color=r['colour'], lw=1.5,
                  zorder=2)
    ax_v.set_xlabel('lap progress [m]', color=INK_MUTED, fontsize=9)
    ax_v.set_ylabel('speed [m/s]', color=INK_MUTED, fontsize=9)
    ax_v.set_title('speed over the first lap', color=INK, fontsize=11.5, loc='left',
                   pad=8)
    ax_v.set_xlim(0, total_s)
    ax_v.grid(axis='y', color=GRID, lw=0.8, zorder=0)
    ax_v.set_axisbelow(True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=145, facecolor=SURFACE)
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
