"""Summarise the sector-controller ablation.

Reads every ``best_sector_*.json`` produced by ``raceline/job_claix_ablation.sh`` and
reports what each feature contributes: the shortfall of the run without it against the
full search. A negative contribution means the feature costs points and should be
dropped.

    pixi run python raceline/ablation_report.py
"""

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = 92.235   # the tuned racing-line agent every variant is seeded from

LABELS = {
    'launch': 'launch handling',
    'filter': 'steering filter',
    'sector_speed': 'per-sector speed scale',
    'sector_brake': 'per-sector braking limit',
    'kp_split': 'split accel/brake speed gain',
    'ld_curve': 'curvature-scheduled lookahead',
}


def load_runs(directory):
    """
    :return: dict mapping variant tag to its saved result
    """
    runs = {}
    for path in sorted(glob.glob(os.path.join(directory, 'best_sector_*.json'))):
        with open(path) as fh:
            data = json.load(fh)
        runs[data.get('variant', os.path.basename(path))] = data
    return runs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', default=HERE)
    p.add_argument('--baseline', type=float, default=BASELINE)
    return p.parse_args()


def main():
    args = parse_args()
    runs = load_runs(args.dir)
    if not runs:
        raise SystemExit(f"no best_sector_*.json under {args.dir}")

    full = runs.get('full')
    if full is None:
        raise SystemExit("the full search result is missing, cannot attribute anything")

    print(f"racing-line baseline (all features off) {args.baseline:8.3f}")
    print(f"full search, every feature on           {full['return']:8.3f}"
          f"   {full['return'] - args.baseline:+.3f} over baseline\n")

    rows = []
    for tag, data in runs.items():
        if tag == 'full':
            continue
        for name in data['disabled']:
            rows.append((name, data['return'], full['return'] - data['return']))
    rows.sort(key=lambda r: -r[2])

    print(f"{'feature':32s} {'without it':>11s} {'contributes':>12s}")
    for name, without, contribution in rows:
        note = '  costs points, drop it' if contribution < 0 else ''
        print(f"{LABELS.get(name, name):32s} {without:11.3f} {contribution:+12.3f}{note}")

    missing = set(LABELS) - {n for n, _, _ in rows}
    if missing:
        print(f"\nnot yet run: {', '.join(sorted(missing))}")


if __name__ == '__main__':
    main()
