#!/usr/bin/env bash
#SBATCH --job-name=bpa3-sector-sac
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=c23g
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#
# Residual SAC on top of the 97.667 sector controller, five seeds.
#
#   sbatch residual_sac/slurm/job_claix_sector.sh
#
# Tasks 0 and 1 release the leash over 200k steps, tasks 2-4 over 300k. The 300k figure
# is the one already measured not to collapse; 200k gives the policy more of its budget
# off the leash, at the risk of straying while the critic is still untrained.
#
# Every task resumes from its own checkpoint if one exists, so re-submitting the same
# array continues the runs rather than restarting them.
#
# A GPU node bills 24 core-h per GPU-h and holds 24 cores per GPU, so the full
# --cpus-per-task=24 is requested whether or not they are all used.

set -euo pipefail

STEPS="${STEPS:-700000}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs runs_sector models

# One process per seed; the array provides the parallelism.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# pygame and matplotlib must not look for a display on a compute node.
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg

export WANDB_MODE=disabled
export PATH="$HOME/.pixi/bin:$PATH"

# Tasks 0-1 on the fast release, 2-4 on the proven one.
if [ "${SLURM_ARRAY_TASK_ID}" -le 1 ]; then
    DECAY=200000
else
    DECAY=300000
fi

printf '\n=== BPA3 sector residual SAC, seed %s ===\n' "${SLURM_ARRAY_TASK_ID}"
echo "Job ${SLURM_ARRAY_JOB_ID}  Task ${SLURM_ARRAY_TASK_ID}  Host $(hostname)"
echo "steps ${STEPS}  penalty-decay ${DECAY}"

printf '\n=== GPU visibility ===\n'
nvidia-smi || true
pixi run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

printf '\n=== Pre-flight checks ===\n'
pixi run python -m py_compile agent_interface.py residual_sac/scripts/train_sector.py \
    residual_sac/features_sector.py

printf '\n=== Training ===\n'
pixi run python residual_sac/scripts/train_sector.py \
    --seed "${SLURM_ARRAY_TASK_ID}" \
    --total-timesteps "${STEPS}" \
    --penalty-decay-steps "${DECAY}" \
    --hidden 128 \
    --batch-size 256 \
    --checkpoint-every 25000 \
    --device cuda \
    --resume \
    --out-dir runs_sector

printf '\n=== Result ===\n'
pixi run python -c "
import json
b = json.load(open('runs_sector/seed${SLURM_ARRAY_TASK_ID}/summary.json'))
print('best', round(b['best_return'], 4), 'baseline', round(b['baseline'], 4),
      'decay', b['penalty_decay_steps'], 'hours', round(b['wall_seconds'] / 3600, 2))
"
