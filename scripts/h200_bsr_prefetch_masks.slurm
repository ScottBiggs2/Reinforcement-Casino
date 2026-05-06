#!/usr/bin/env bash
# CPU-oriented job: generate random BSR masks into H200_BSR_MASK_CACHE/masks/
# (see --prefetch_masks_only in src/full_training/h200_sparse_dpo_bsr_benchmark.py).
# Submit from repo root via scripts/h200_bsr_orchestrate.sh or manually:
#
#   export H200_BSR_MASK_CACHE=/scratch/$USER/rl_casino_h200_bsr/my_run_id
#   export BENCHMARK_SPARSITIES=50,75,90,99.75
#   sbatch scripts/h200_bsr_prefetch_masks.slurm
#
# Override partition/mem if Short nodes are tight; meta-skeleton Llama loads are mostly CPU-bound.
#
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --job-name=h200_bsr_masks
#SBATCH --output=logs/h200_bsr_prefetch_masks_%j.out
#SBATCH --error=logs/h200_bsr_prefetch_masks_%j.err
#SBATCH --export=ALL

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
DEFAULT_TRAIN_ENV="${SCRATCH_USER_ROOT}/conda_envs/rl_casino"
if [ -n "${TRAIN_PY:-}" ] && [ -x "$TRAIN_PY" ]; then
  TRAIN_ENV="$(dirname "$(dirname "$TRAIN_PY")")"
elif [ -n "${TRAIN_ENV:-}" ] && [ -x "${TRAIN_ENV}/bin/python" ]; then
  TRAIN_PY="${TRAIN_ENV}/bin/python"
elif [ -x "${DEFAULT_TRAIN_ENV}/bin/python" ]; then
  TRAIN_ENV="${DEFAULT_TRAIN_ENV}"
  TRAIN_PY="${TRAIN_ENV}/bin/python"
else
  echo "ERROR: TRAIN_PY not found for prefetch job." >&2
  exit 1
fi
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export HF_DATASETS_CACHE_ROOT="${HF_DATASETS_CACHE_ROOT:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_DATASETS_CACHE_ROOT}"

if [ -z "${H200_BSR_MASK_CACHE:-}" ]; then
  echo "ERROR: H200_BSR_MASK_CACHE must be set (parent dir; masks go under <parent>/masks/)." >&2
  exit 1
fi

PREFETCH_OUT="${H200_BSR_PREFETCH_OUT:-${H200_BSR_MASK_CACHE}/prefetch_log}"
mkdir -p "${H200_BSR_MASK_CACHE}/masks" "$PREFETCH_OUT"

echo "REPO_ROOT=${REPO_ROOT}  TRAIN_PY=${TRAIN_PY}"
echo "H200_BSR_MASK_CACHE=${H200_BSR_MASK_CACHE}"
echo "H200_BSR_PREFETCH_OUT=${PREFETCH_OUT}"
echo "BENCHMARK_SPARSITIES=${BENCHMARK_SPARSITIES:-<unset>}"

CHK_ARGS=()
if [ -n "${CHECKPOINT:-}" ] && [ "${CHECKPOINT,,}" != "none" ]; then
  CHK_ARGS+=(--checkpoint "${CHECKPOINT}")
fi

BLOCK_ARGS=()
if [ -n "${H200_BSR_BLOCK_SIZE:-}" ]; then
  BLOCK_ARGS+=(--block_size_bsr "${H200_BSR_BLOCK_SIZE}")
fi

MLP_ARGS=()
if [ "${H200_BSR_MLP_ONLY:-0}" != "0" ]; then
  MLP_ARGS+=(--mlp_only)
fi

FLOOR_ARGS=()
if [ -n "${H200_BSR_MIN_LAYER_KEEP_RATIO:-}" ]; then
  FLOOR_ARGS+=(--min_layer_keep_ratio "${H200_BSR_MIN_LAYER_KEEP_RATIO}")
fi

export WANDB_MODE="disabled"
export WANDB_DISABLED="true"
export WANDB_CONSOLE="off"
export WANDB_SILENT="true"

exec "$TRAIN_PY" src/full_training/h200_sparse_dpo_bsr_benchmark.py \
  --prefetch_masks_only \
  --mask_cache_dir "${H200_BSR_MASK_CACHE}" \
  --output_dir "${PREFETCH_OUT}" \
  --model_name "${MODEL}" \
  "${CHK_ARGS[@]}" \
  --benchmark_sparsities "${BENCHMARK_SPARSITIES}" \
  --phase_mask_types "${H200_BSR_PHASE_MASK_TYPES:-element,block}" \
  "${BLOCK_ARGS[@]}" \
  "${MLP_ARGS[@]}" \
  "${FLOOR_ARGS[@]}"
