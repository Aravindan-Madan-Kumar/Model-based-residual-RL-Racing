#!/usr/bin/env bash
#SBATCH --job-name=bpa3-sector-run
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=c23mm
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#
# One residual SAC run, parameterised through the environment:
#
#   sbatch --export=ALL,SEED=1,DECAY=200000,CHANNELS=2 residual_sac/job_claix_sector_run.sh
#
# A CPU partition on purpose. The environment is a numba vehicle model with no network in
# its loop, and the policy is two 128-wide layers, so a GPU node measured 144 SPS against
# 105 SPS on two CPU cores - not worth 24 core-h of billing per GPU-h, and the GPU queue
# was the slower path to a started job.
#
# --resume is always passed: a run whose checkpoint exists continues from it, and a fresh
# seed starts from scratch. That makes re-submission after a wall-clock kill idempotent.

set -euo pipefail

SEED="${SEED:?set SEED}"
DECAY="${DECAY:?set DECAY}"
CHANNELS="${CHANNELS:-2}"
STEPS="${STEPS:-900000}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs runs_sector_cpu models

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg
export WANDB_MODE=disabled
export PATH="$HOME/.pixi/bin:$PATH"

printf '\n=== BPA3 sector residual SAC ===\n'
echo "Job ${SLURM_JOB_ID}  Host $(hostname)"
echo "seed ${SEED}  decay ${DECAY}  channels ${CHANNELS}  steps ${STEPS}"

pixi run python -m py_compile agent_interface.py residual_sac/train_sector.py \
    residual_sac/features_sector.py

printf '\n=== Training ===\n'
pixi run python residual_sac/train_sector.py \
    --seed "${SEED}" \
    --total-timesteps "${STEPS}" \
    --penalty-decay-steps "${DECAY}" \
    --channels "${CHANNELS}" \
    --hidden 128 \
    --batch-size 256 \
    --checkpoint-every 25000 \
    --device cpu \
    --resume \
    --out-dir runs_sector_cpu

printf '\n=== Result ===\n'
cat "runs_sector_cpu/seed${SEED}/summary.json" | head -8
