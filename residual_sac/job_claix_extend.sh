#!/usr/bin/env bash
#SBATCH --job-name=bpa3-residual-sac-extend
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=c23g
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#
# Single-seed continuation run for the winning seed of job 3028953.
#
#   sbatch residual_sac/job_claix_extend.sh
#
# --resume picks up runs_extend/seed1/ckpt.pt if one exists, so the walltime cap is a
# pause rather than a loss: resubmitting the same script continues the run.
# The 8-seed campaign predates checkpointing and left no ckpt.pt, so the first
# submission starts from scratch.

set -euo pipefail

SEED="${SEED:-1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-400000}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs runs_extend models

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# pygame and matplotlib must not look for a display on a compute node.
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg

export WANDB_MODE=disabled
export PATH="$HOME/.pixi/bin:$PATH"

printf '\n=== BPA3 residual SAC extension, seed %s, %s steps ===\n' \
    "${SEED}" "${TOTAL_TIMESTEPS}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"

printf '\n=== Pixi environment ===\n'
which pixi
pixi --version
pixi run python --version

printf '\n=== GPU visibility ===\n'
nvidia-smi || true
pixi run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

printf '\n=== Pre-flight checks ===\n'
pixi run python -m py_compile agent_interface.py try_agent.py residual_sac/train.py

printf '\n=== Training ===\n'
pixi run python residual_sac/train.py \
    --seed "${SEED}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --hidden 128 \
    --batch-size 256 \
    --checkpoint-every 25000 \
    --resume \
    --device cuda \
    --out-dir runs_extend

printf '\n=== Result ===\n'
cat "runs_extend/seed${SEED}/summary.json"
