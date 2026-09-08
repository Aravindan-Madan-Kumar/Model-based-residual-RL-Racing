"""Summarise the multi-seed line search.

Reads every ``best_line_seed*.json`` from ``raceline/slurm/job_claix_line.sh`` and reports what
each seed found, with the line parameters that produced it.

    pixi run python raceline/scripts/line_report.py
"""

import argparse
import glob
import json
import os

MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_runs(directory):
    """
    :return: list of saved results, ordered by return, best first
    """
    runs = []
    for path in sorted(glob.glob(os.path.join(directory, 'best_line_seed*.json'))):
        with open(path) as fh:
            data = json.load(fh)
        data['file'] = os.path.basename(path)
        runs.append(data)
    return sorted(runs, key=lambda d: -d['return'])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', default=MODULE_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    runs = load_runs(args.dir)
    if not runs:
        raise SystemExit(f"no best_line_seed*.json under {args.dir}")

    fields = ('corridor', 'smooth', 'a_max', 'a_accel', 'steer_alpha')
    header = f"{'file':<24}{'seed':>6}{'return':>10}{'vs seed':>10}"
    print(header + ''.join(f"{f:>11}" for f in fields))

    for r in runs:
        idx = {n: i for i, n in enumerate(r['names'])}
        row = (f"{r['file']:<24}{r['seed']:>6}{r['return']:>10.3f}"
               f"{r['return'] - r['seed_return']:>+10.3f}")
        print(row + ''.join(f"{r['params'][idx[f]]:>11.3f}" for f in fields))

    values = [r['return'] for r in runs]
    print(f"\n{len(runs)} seeds   best {max(values):.3f}   median "
          f"{sorted(values)[len(values) // 2]:.3f}   worst {min(values):.3f}"
          f"   spread {max(values) - min(values):.3f}")
    print(f"seed point {runs[0]['seed_return']:.3f}")


if __name__ == '__main__':
    main()
