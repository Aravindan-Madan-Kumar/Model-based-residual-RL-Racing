#!/usr/bin/env bash
#SBATCH --job-name=bpa3-sector-ablation
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=c23mm
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --array=0-6
#
# Ablation over the features the sector controller adds on top of the racing-line agent.
#
#   sbatch raceline/job_claix_ablation.sh
#   pixi run python raceline/ablation_report.py
#
# Task 0 tunes everything. Tasks 1-6 each neutralise one feature and tune the rest, so
# the shortfall against task 0 is that feature's contribution. Every variant is seeded
# from the same 92.235 racing-line agent and uses the same search budget and seed, so the
# only difference between runs is the feature under test.
#
# A CPU partition on purpose: each evaluation is a Box2D episode with no network in the
# loop, so a GPU has nothing to do and c23g would bill 24 core-h per GPU-h to run it no
# faster.

set -euo pipefail

VARIANTS=(
    ""                # 0: full search, the reference
    "launch"          # 1: no launch handling
    "filter"          # 2: no steering filter
    "sector_speed"    # 3: one global speed scale instead of per sector
    "sector_brake"    # 4: one global braking limit instead of per sector
    "kp_split"        # 5: one speed gain for accelerating and braking
    "ld_curve"        # 6: lookahead no longer shortens through corners
)
DISABLE="${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"

RANDOM_SAMPLES="${RANDOM_SAMPLES:-1200}"
GENERATIONS="${GENERATIONS:-120}"
SEED="${SEED:-5}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-48}}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Each rollout is single-threaded; the parallelism comes from the worker pool.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# pygame and matplotlib must not look for a display on a compute node.
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg

export PATH="$HOME/.pixi/bin:$PATH"

printf '\n=== BPA3 sector ablation, task %s: %s ===\n' \
    "${SLURM_ARRAY_TASK_ID}" "${DISABLE:-full search}"
echo "Job ${SLURM_ARRAY_JOB_ID}  Host $(hostname)  Workers ${WORKERS}"

printf '\n=== Pre-flight checks ===\n'
pixi run python -m py_compile agent_interface.py raceline/sector.py \
    raceline/tune_sector.py
pixi run python -c "
from raceline.tune_sector import seed_vector, rollout_score
from pure_pursuit.rollout import make_env
print('seed reproduces:', round(rollout_score(make_env(), seed_vector()), 3))
"

printf '\n=== Tuning ===\n'
pixi run python raceline/tune_sector.py \
    --random "${RANDOM_SAMPLES}" \
    --generations "${GENERATIONS}" \
    --workers "${WORKERS}" \
    --seed "${SEED}" \
    --disable "${DISABLE}"
