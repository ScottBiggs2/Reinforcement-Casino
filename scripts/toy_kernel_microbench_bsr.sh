#!/bin/bash
# Toy job: kernel-only microbench on GPU (no TRL/dataset).
# Runs sandbox microtests that repeatedly launch BSR kernels to separate compile noise from steady-state.
#
# Submit from repo root:
#   sbatch scripts/toy_kernel_microbench_bsr.sh
#
# Outputs under OUT_BASE:
# - bsr_recompile_microtest.txt
# - bsr_grad_input_parity.txt

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=toy_bsr_microbench
#SBATCH --output=logs/toy_bsr_microbench_%j.out
#SBATCH --error=logs/toy_bsr_microbench_%j.err

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
OUT_BASE="${OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_toys}/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_BASE"

TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found: ${TRAIN_PY}" >&2
  exit 1
fi
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Keep compile cache stable across runs.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH_USER_ROOT}/.triton_cache}"
mkdir -p "${TRITON_CACHE_DIR}"

echo "OUT_BASE=${OUT_BASE}"
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"

if [ -f "src/sandbox/bsr_recompile_microtest.py" ]; then
  "$TRAIN_PY" -u src/sandbox/bsr_recompile_microtest.py 2>&1 | tee "${OUT_BASE}/bsr_recompile_microtest.txt"
else
  echo "Missing src/sandbox/bsr_recompile_microtest.py" | tee "${OUT_BASE}/bsr_recompile_microtest.txt"
fi

if [ -f "src/sandbox/bsr_grad_input_parity.py" ]; then
  "$TRAIN_PY" -u src/sandbox/bsr_grad_input_parity.py 2>&1 | tee "${OUT_BASE}/bsr_grad_input_parity.txt"
else
  echo "Missing src/sandbox/bsr_grad_input_parity.py" | tee "${OUT_BASE}/bsr_grad_input_parity.txt"
fi

echo "DONE. OUT_BASE=${OUT_BASE}"
