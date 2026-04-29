#!/bin/bash
# A2: bench torchao 2:4 vs our BSR-16 vs dense on H200, isolated linear layer.
# Submit from repo root:  sbatch scripts/bench_2to4_vs_bsr.sbatch
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --job-name=bench_2to4
#SBATCH --output=logs/bench_2to4_%j.out
#SBATCH --error=logs/bench_2to4_%j.err

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

# rl_casino env (per project memory: lives at /home/$USER/.conda/envs/rl_casino)
TRAIN_ENV="${TRAIN_ENV:-/home/${USER}/.conda/envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found: ${TRAIN_PY}" >&2
  echo "Override with: TRAIN_ENV=/path/to/env sbatch ..." >&2
  exit 1
fi
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Triton cache on scratch (warm across runs)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH_USER_ROOT}/.triton_cache}"
mkdir -p "${TRITON_CACHE_DIR}"

OUT_BASE="${A2_OUT:-${SCRATCH_USER_ROOT}/rl_casino_bench/2to4_vs_bsr}/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_BASE"

echo "REPO_ROOT=${REPO_ROOT}"
echo "OUT_BASE=${OUT_BASE}"
echo "TRAIN_PY=${TRAIN_PY}"

# Quick env probe
"$TRAIN_PY" - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
try:
    import torchao
    print(f"torchao={torchao.__version__}")
except ImportError as e:
    print(f"torchao NOT INSTALLED: {e}")
    print("→ install with:  pip install torchao")
PY

exec "$TRAIN_PY" scripts/bench_2to4_vs_bsr.py \
  --output_dir "$OUT_BASE" \
  --batch_tokens "${A2_BATCH_TOKENS:-2048}" \
  --rho "${A2_RHO:-0.5}" \
  --n_warmup "${A2_WARMUP:-5}" \
  --n_iters "${A2_ITERS:-100}" \
  --dtype "${A2_DTYPE:-bf16}"
