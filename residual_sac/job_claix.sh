#!/usr/bin/env bash
#SBATCH --job-name=bpa3-residual-sac
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --array=0-15
#
# Environment setup follows hpc_bpa3_guide/slurm/submit_train_gpu.sh.
#
# One array task per seed. Confirm single-seed throughput against the 8 h limit before
# submitting the full array.
#
#   sbatch residual_sac/job_claix.sh
#   pixi run python residual_sac/select_best.py --runs runs --install

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs runs models

# One process per seed, so the intra-op thread pools stay at 1 and the job array
# provides the parallelism.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# pygame and matplotlib must not look for a display on a compute node.
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg

export WANDB_MODE=disabled
export PATH="$HOME/.pixi/bin:$PATH"

printf '\n=== BPA3 residual SAC, seed %s ===\n' "${SLURM_ARRAY_TASK_ID}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}  Task: ${SLURM_ARRAY_TASK_ID}"
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
    --seed "${SLURM_ARRAY_TASK_ID}" \
    --total-timesteps 1000000 \
    --hidden 128 \
    --batch-size 256 \
    --device cuda \
    --out-dir runs

printf '\n=== Result ===\n'
cat "runs/seed${SLURM_ARRAY_TASK_ID}/summary.json"
