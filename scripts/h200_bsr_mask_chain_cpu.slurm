#!/usr/bin/env bash
# One **independent** chain: CPU generates mask(s) for a **single** sparsity, then submits **one**
# H200 microbench job with `--dependency=afterok:this_cpu_job`. Submit many of these in parallel
# from `submit_independent_mask_gpu_chains.sh` — no single prefetch blocks all GPUs.
#
# Required env (via sbatch --export=ALL,...):
#   CHAIN_SPARSITY       e.g. 50  or 99.75
#   H200_BSR_MASK_CACHE  parent dir; masks go to <dir>/masks/*.pt (shared across chains is OK)
#
# Optional:
#   CHAIN_MASK_TYPES     default element,block  (microbench needs element; include block for block/ column)
#   MODEL, CHAIN_GPU_PARTITION, H200_BSR_BLOCK_SIZE, STEPS, TRIM_FRAC, LR, BLOCK_SIZE (microbench)
#
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:15:00
#SBATCH --job-name=h200_mask_cpu
#SBATCH --output=logs/h200_bsr_mask_chain_cpu_%j.out
#SBATCH --error=logs/h200_bsr_mask_chain_cpu_%j.err
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

CHAIN_SPARSITY="${CHAIN_SPARSITY:?set CHAIN_SPARSITY}"
H200_BSR_MASK_CACHE="${H200_BSR_MASK_CACHE:?set H200_BSR_MASK_CACHE}"
CHAIN_MASK_TYPES="${CHAIN_MASK_TYPES:-element,block}"

if [[ "${CHAIN_MASK_TYPES}" != *element* ]]; then
  echo "ERROR: Optimizer microbench requires an element mask; include 'element' in CHAIN_MASK_TYPES (got: ${CHAIN_MASK_TYPES})" >&2
  exit 1
fi

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
DEFAULT_TRAIN_ENV="${SCRATCH_USER_ROOT}/conda_envs/rl_casino"
if [ -n "${TRAIN_PY:-}" ] && [ -x "$TRAIN_PY" ]; then
  :
elif [ -n "${TRAIN_ENV:-}" ] && [ -x "${TRAIN_ENV}/bin/python" ]; then
  TRAIN_PY="${TRAIN_ENV}/bin/python"
elif [ -x "${DEFAULT_TRAIN_ENV}/bin/python" ]; then
  TRAIN_ENV="${DEFAULT_TRAIN_ENV}"
  TRAIN_PY="${TRAIN_ENV}/bin/python"
else
  TRAIN_PY="python3"
fi
export PATH="${TRAIN_ENV:-}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export HF_DATASETS_CACHE_ROOT="${HF_DATASETS_CACHE_ROOT:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_DATASETS_CACHE_ROOT}"

PREFETCH_OUT="${H200_BSR_PREFETCH_OUT:-${H200_BSR_MASK_CACHE}/prefetch_cpu_${SLURM_JOB_ID}}"
mkdir -p "${H200_BSR_MASK_CACHE}/masks" "${PREFETCH_OUT}"

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

echo "CHAIN_CPU_JOB=${SLURM_JOB_ID}  CHAIN_SPARSITY=${CHAIN_SPARSITY}  H200_BSR_MASK_CACHE=${H200_BSR_MASK_CACHE}"
echo "CHAIN_MASK_TYPES=${CHAIN_MASK_TYPES}"

"$TRAIN_PY" src/full_training/h200_sparse_dpo_bsr_benchmark.py \
  --prefetch_masks_only \
  --mask_cache_dir "${H200_BSR_MASK_CACHE}" \
  --output_dir "${PREFETCH_OUT}" \
  --model_name "${MODEL}" \
  "${CHK_ARGS[@]}" \
  --benchmark_sparsities "${CHAIN_SPARSITY}" \
  --phase_mask_types "${CHAIN_MASK_TYPES}" \
  "${BLOCK_ARGS[@]}" \
  "${MLP_ARGS[@]}" \
  "${FLOOR_ARGS[@]}"

PATH_ARGS=( --mask-cache-dir "${H200_BSR_MASK_CACHE}" --sparsity "${CHAIN_SPARSITY}" )
[[ -n "${H200_BSR_BLOCK_SIZE:-}" ]] && PATH_ARGS+=( --block-size-bsr "${H200_BSR_BLOCK_SIZE}" )
[[ "${H200_BSR_MLP_ONLY:-0}" != "0" ]] && PATH_ARGS+=( --mlp-only )
[[ -n "${H200_BSR_MIN_LAYER_KEEP_RATIO:-}" ]] && PATH_ARGS+=( --min-layer-keep-ratio "${H200_BSR_MIN_LAYER_KEEP_RATIO}" )

ELEM_PT="$("$TRAIN_PY" "${REPO_ROOT}/scripts/print_bsr_mask_path.py" "${PATH_ARGS[@]}" --mask-type element)"
export ELEM_MASK="${ELEM_PT}"
if [ ! -f "${ELEM_MASK}" ]; then
  echo "ERROR: missing element mask ${ELEM_MASK}" >&2
  exit 1
fi

RUN_BLOCK=0
export BLOCK_MASK=""
if [[ "${CHAIN_MASK_TYPES}" == *block* ]]; then
  BLOCK_PT="$("$TRAIN_PY" "${REPO_ROOT}/scripts/print_bsr_mask_path.py" "${PATH_ARGS[@]}" --mask-type block)"
  export BLOCK_MASK="${BLOCK_PT}"
  if [ ! -f "${BLOCK_MASK}" ]; then
    echo "ERROR: missing block mask ${BLOCK_MASK}" >&2
    exit 1
  fi
  RUN_BLOCK=1
fi
export RUN_BLOCK="${RUN_BLOCK}"

export STEPS="${STEPS:-50}"
export TRIM_FRAC="${TRIM_FRAC:-0.10}"
export LR="${LR:-5e-7}"
export BLOCK_SIZE="${BLOCK_SIZE:-32}"

SP_TAG="${CHAIN_SPARSITY//./p}"
CHAIN_GPU_PARTITION="${CHAIN_GPU_PARTITION:-multigpu}"

GPU_JID=$(sbatch --parsable \
  --dependency=afterok:"${SLURM_JOB_ID}" \
  --partition="${CHAIN_GPU_PARTITION}" \
  --job-name="h200_mb_s${SP_TAG}" \
  --export=ALL \
  "${REPO_ROOT}/scripts/h200_sparse_adamw_optstep_microbench.slurm")

echo "CHAIN_GPU_JOB=${GPU_JID}  (depends on CPU ${SLURM_JOB_ID})"
echo "ELEM_MASK=${ELEM_MASK}"
echo "BLOCK_MASK=${BLOCK_MASK:-}  RUN_BLOCK=${RUN_BLOCK}"
