#!/usr/bin/env bash
#SBATCH --job-name=bpa3-sector-tune
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=c23mm
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=48G
#
# Black-box tuning of the sector controller.
#
#   sbatch raceline/slurm/job_claix_tune.sh
#
# A CPU partition on purpose. Each evaluation is a Box2D episode with no network in the
# loop, so there is nothing for a GPU to do, and c23g would bill 24 core-h per GPU-h to
# run it no faster. Billing is wall time times cores, so spreading the same ~1800
# episodes over 96 cores costs the same few core-hours it would on 4 - it just finishes
# in minutes instead of half an hour.

set -euo pipefail

RANDOM_SAMPLES="${RANDOM_SAMPLES:-1200}"
GENERATIONS="${GENERATIONS:-120}"
SEED="${SEED:-5}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-96}}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Each rollout is single-threaded; the parallelism comes from the worker pool.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# pygame and matplotlib must not look for a display on a compute node.
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg

export PATH="$HOME/.pixi/bin:$PATH"

printf '\n=== BPA3 sector tuning: %s samples, %s generations, %s workers ===\n' \
    "${RANDOM_SAMPLES}" "${GENERATIONS}" "${WORKERS}"
echo "Job ID: ${SLURM_JOB_ID}  Host: $(hostname)"

printf '\n=== Pixi environment ===\n'
which pixi
pixi --version
pixi run python --version

printf '\n=== Pre-flight checks ===\n'
pixi run python -m py_compile agent_interface.py raceline/sector.py \
    raceline/scripts/tune_sector.py
pixi run python -c "
from raceline.tune_sector import seed_vector, rollout_score
from pure_pursuit.rollout import make_env
print('seed reproduces:', round(rollout_score(make_env(), seed_vector()), 3))
"

printf '\n=== Tuning ===\n'
pixi run python raceline/scripts/tune_sector.py \
    --random "${RANDOM_SAMPLES}" \
    --generations "${GENERATIONS}" \
    --workers "${WORKERS}" \
    --seed "${SEED}"

printf '\n=== Result ===\n'
pixi run python -c "
import json
b = json.load(open('raceline/best_sector.json'))
print('best return', round(b['return'], 4), 'from seed', round(b['seed_return'], 4))
"
