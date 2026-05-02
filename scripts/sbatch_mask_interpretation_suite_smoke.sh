#!/usr/bin/env bash
# Slurm: full interpretation suite smoke test on TWO random masks (different seeds).
# Uses a reference .pt only for topology; generates smoke_random_{a,b}_seed*.pt under OUT_DIR/smoke_masks/.
#
# Default: GPU + CKA + effective rank + extended Jaccard + heatmap + plot_layer_metrics (MC band).
# Use a *small* reference mask for a quick run, or increase --time / reduce CKA samples.
#
# Drop-in (Explorer-style):
#   export MASK_SMOKE_REFERENCE=/scratch/${USER}/path/to/any_same_arch_mask.pt
#   export MASK_SUITE_OUT_DIR=/scratch/${USER}/mask_suite_smoke_${SLURM_JOB_ID:-local}
#   sbatch scripts/sbatch_mask_interpretation_suite_smoke.sh
#
# CPU-only / no CKA:
#   sbatch --partition=short --gres=none --mem=64G --time=01:00:00 \
#     --export=ALL,MASK_SMOKE_REFERENCE=/path/to/ref.pt,MASK_SUITE_OUT_DIR=/scratch/${USER}/smoke1,MASK_SMOKE_NO_CKA=1 \
#     scripts/sbatch_mask_interpretation_suite_smoke.sh
#
# Optional env:
#   MASK_SMOKE_SEED_A=10001  MASK_SMOKE_SEED_B=20002  (must differ; defaults shown)
#   MASK_SMOKE_SPARSITY=90
#   MASK_SUITE_CKA_MODEL=google/gemma-3-270m-it
#   MASK_SUITE_CKA_DATASET=tulu3
#   MASK_SMOKE_NO_CKA=1
#   MASK_SMOKE_CKA_NSAMPLES=32
#   MASK_SMOKE_CKA_BATCH=2
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=mask_interp_smoke
#SBATCH --output=logs/mask_interpretation_smoke_%j.out
#SBATCH --error=logs/mask_interpretation_smoke_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_SCRIPT_HOME}/.." && pwd)"
fi
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/pipeline_common.sh"
pipeline_setup

mkdir -p logs

MASK_SMOKE_REFERENCE="${MASK_SMOKE_REFERENCE:-}"
if [ -z "$MASK_SMOKE_REFERENCE" ] || [ ! -f "$MASK_SMOKE_REFERENCE" ]; then
  echo "ERROR: set MASK_SMOKE_REFERENCE to an existing .pt mask (same arch as --cka-model)." >&2
  exit 2
fi

MASK_SUITE_OUT_DIR="${MASK_SUITE_OUT_DIR:-/scratch/${USER}/mask_interpretation_smoke/run_${SLURM_JOB_ID:-local}}"
SEED_A="${MASK_SMOKE_SEED_A:-10001}"
SEED_B="${MASK_SMOKE_SEED_B:-20002}"
SPARSITY="${MASK_SMOKE_SPARSITY:-90}"
CKA_MODEL="${MASK_SUITE_CKA_MODEL:-google/gemma-3-270m-it}"
CKA_DS="${MASK_SUITE_CKA_DATASET:-tulu3}"
CKA_NS="${MASK_SMOKE_CKA_NSAMPLES:-64}"
CKA_BS="${MASK_SMOKE_CKA_BATCH:-4}"

if [ "$SEED_A" = "$SEED_B" ]; then
  echo "ERROR: MASK_SMOKE_SEED_A and MASK_SMOKE_SEED_B must differ." >&2
  exit 2
fi

cmd=(
  "$TRAIN_PY" "${REPO_ROOT}/src/cold_start/mask_interpretation_suite.py"
  --smoke-debug
  --smoke-reference "$MASK_SMOKE_REFERENCE"
  --smoke-seed-a "$SEED_A"
  --smoke-seed-b "$SEED_B"
  --smoke-sparsity "$SPARSITY"
  --out-dir "$MASK_SUITE_OUT_DIR"
  --device cpu
  --cka-model "$CKA_MODEL"
  --cka-dataset "$CKA_DS"
  --cka-device cuda
  --cka-n-samples "$CKA_NS"
  --cka-batch-size "$CKA_BS"
)

if [ "${MASK_SMOKE_NO_CKA:-0}" = "1" ]; then
  cmd+=( --smoke-no-cka )
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "OUT_DIR=$MASK_SUITE_OUT_DIR"
echo "REFERENCE=$MASK_SMOKE_REFERENCE"
echo "SEEDS=$SEED_A $SEED_B  SPARSITY=$SPARSITY  NO_CKA=${MASK_SMOKE_NO_CKA:-0}"
echo "RUN: ${cmd[*]}"
exec "${cmd[@]}"
