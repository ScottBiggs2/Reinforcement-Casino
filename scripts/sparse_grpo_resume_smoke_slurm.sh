#!/usr/bin/env bash
# Submit sparse GRPO resume smoke on a GPU node (do not run heavy training on login nodes).
#
#   sbatch scripts/sparse_grpo_resume_smoke_slurm.sh
#   MASK_PATH=/path/to/mask.pt sbatch scripts/sparse_grpo_resume_smoke_slurm.sh
#
#SBATCH --job-name=sparse_resume_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=logs/sparse_resume_smoke_%j.out
#SBATCH --error=logs/sparse_resume_smoke_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

export SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER}}"
export RL_CASINO_SCRATCH_ROOT="${RL_CASINO_SCRATCH_ROOT:-$SCRATCH_USER_ROOT}"
export TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"

exec bash "${REPO_ROOT}/scripts/sparse_grpo_resume_smoke.sh"
