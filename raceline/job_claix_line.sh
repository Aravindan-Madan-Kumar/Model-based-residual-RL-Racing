#!/usr/bin/env bash
#SBATCH --job-name=bpa3-line-tune
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=c23mm
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --array=0-7
#
# Tune the sector controller with the racing line's shape as a search dimension.
#
#   sbatch raceline/job_claix_line.sh
#   pixi run python raceline/line_report.py
#
# Adds the corridor half-width to the parameter vector and refines the smoothing grid.
# Those two set the line's shape between them, and neither was searched properly before:
# the corridor was pinned at 2.61 m, where it turns out never to bind, so the smoothing
# weight alone decided how wide the line ran.
#
# Eight seeds because the two new dimensions trade off against the tracker gains, and a
# single ES run settles into whichever basin it first descends. All eight start from the
# installed 96.891 agent, so none of them can report worse than that.
#
# A CPU partition on purpose: each evaluation is a Box2D episode with no network in the
# loop, so a GPU has nothing to do and c23g would bill 24 core-h per GPU-h to run it no
# faster.

set -euo pipefail

RANDOM_SAMPLES="${RANDOM_SAMPLES:-1200}"
GENERATIONS="${GENERATIONS:-140}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-48}}"
SEED="$((11 + SLURM_ARRAY_TASK_ID))"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Each rollout is single-threaded; the parallelism comes from the worker pool.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# pygame and matplotlib must not look for a display on a compute node.
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg

export PATH="$HOME/.pixi/bin:$PATH"

printf '\n=== BPA3 line tuning, task %s, seed %s ===\n' \
    "${SLURM_ARRAY_TASK_ID}" "${SEED}"
echo "Job ${SLURM_ARRAY_JOB_ID}  Host $(hostname)  Workers ${WORKERS}"

printf '\n=== Pre-flight checks ===\n'
pixi run python -m py_compile agent_interface.py raceline/sector.py \
    raceline/tune_sector.py
pixi run python -c "
from raceline.tune_sector import seed_vector, project, rollout_score
from pure_pursuit.rollout import make_env
from raceline.sector import PARAM_NAMES
v = project(seed_vector())
print('parameters:', len(v))
print('corridor  :', round(v[PARAM_NAMES.index('corridor')], 3))
print('smooth    :', round(v[PARAM_NAMES.index('smooth')], 3))
print('seed reproduces:', round(rollout_score(make_env(), v), 3))
"

printf '\n=== Tuning ===\n'
pixi run python raceline/tune_sector.py \
    --random "${RANDOM_SAMPLES}" \
    --generations "${GENERATIONS}" \
    --workers "${WORKERS}" \
    --seed "${SEED}" \
    --out "raceline/best_line_seed${SLURM_ARRAY_TASK_ID}.json"

printf '\n=== Result ===\n'
pixi run python -c "
import json
b = json.load(open('raceline/best_line_seed${SLURM_ARRAY_TASK_ID}.json'))
print('best return', round(b['return'], 4), 'from seed', round(b['seed_return'], 4))
"
