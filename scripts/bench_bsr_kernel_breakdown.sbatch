#!/bin/bash
# Kernel-level breakdown of BSR backward at rho=0.025 (user's training sparsity).
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --job-name=bsr_kbreak
#SBATCH --output=logs/bsr_kbreak_%j.out
#SBATCH --error=logs/bsr_kbreak_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
TRAIN_ENV="${TRAIN_ENV:-/home/${USER}/.conda/envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH_USER_ROOT}/.triton_cache}"
mkdir -p "${TRITON_CACHE_DIR}"

OUT_BASE="${SCRATCH_USER_ROOT}/rl_casino_bench/bsr_kbreak/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_BASE"

echo "OUT_BASE=${OUT_BASE}"

exec "$TRAIN_PY" scripts/bench_bsr_kernel_breakdown.py \
  --output_dir "$OUT_BASE" \
  --rho "${KBREAK_RHO:-0.025}" \
  --batch_tokens "${KBREAK_BATCH_TOKENS:-2048}" \
  --n_warmup "${KBREAK_WARMUP:-10}" \
  --n_iters "${KBREAK_ITERS:-200}"
