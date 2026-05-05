#!/bin/bash
# H200 BSR–DPO speed ablation v2 (rebuilt driver): 8h-friendly defaults, BENCH_JSON timing,
# post-run ``report_h200_speed_ablation.py``, and optional Slurm-array phase sharding.
#
# Submit from repo root:
#   sbatch scripts/h200_speed_ablation_v2.sh
#
# Default sparsity sweep: **99.75, 97.5, 95, 90** (element masks, block_1d Adam, dense grad-input)
# → 1 dense + 4 sparse = **5 phases** unless you override ``BENCHMARK_SPARSITIES``.
#
# Full legacy-style grid (element+block masks × block_1d+block_2d; may need ``H200_BSR_ARRAY_PHASE=1``):
#   export H200_BSR_FULL_GRID=1
#   export BENCHMARK_SPARSITIES=99.75,97.5,95,90   # optional explicit list
#   sbatch scripts/h200_speed_ablation_v2.sh
#
# One phase per array task (match ``--array`` upper bound to expanded phase count minus one):
#   export H200_BSR_ARRAY_PHASE=1
#   sbatch --array=0-16 scripts/h200_speed_ablation_v2.sh   # example for 17-phase full grid
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=h200_speed_ablat
#SBATCH --output=logs/h200_speed_ablat_%j.out
#SBATCH --error=logs/h200_speed_ablat_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

echo "GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-n/a}"

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
case "${H200_BSR_OUT:-}" in
  /rl_casino_h200_bsr/*)
    echo "WARNING: Ignoring invalid H200_BSR_OUT=${H200_BSR_OUT} (export SCRATCH_USER_ROOT before H200_BSR_OUT on the login node)." >&2
    unset H200_BSR_OUT
    ;;
esac
DEFAULT_TRAIN_ENV="${SCRATCH_USER_ROOT}/conda_envs/rl_casino"
if [ -n "${TRAIN_PY:-}" ] && [ -x "$TRAIN_PY" ]; then
  TRAIN_ENV="$(dirname "$(dirname "$TRAIN_PY")")"
elif [ -n "${TRAIN_ENV:-}" ] && [ -x "${TRAIN_ENV}/bin/python" ]; then
  TRAIN_PY="${TRAIN_ENV}/bin/python"
elif [ -x "${DEFAULT_TRAIN_ENV}/bin/python" ]; then
  if [ -n "${TRAIN_ENV:-}" ]; then
    echo "WARNING: TRAIN_ENV=${TRAIN_ENV} has no usable bin/python; using ${DEFAULT_TRAIN_ENV}." >&2
  fi
  TRAIN_ENV="${DEFAULT_TRAIN_ENV}"
  TRAIN_PY="${TRAIN_ENV}/bin/python"
else
  echo "ERROR: TRAIN_PY not found. Checked TRAIN_PY=${TRAIN_PY:-<unset>}, TRAIN_ENV=${TRAIN_ENV:-<unset>}/bin/python, and ${DEFAULT_TRAIN_ENV}/bin/python" >&2
  exit 1
fi
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH_USER_ROOT}/.triton_cache}"
mkdir -p "${TRITON_CACHE_DIR}"

export RL_CASINO_BSR_QUIET_INJECTION="${RL_CASINO_BSR_QUIET_INJECTION:-1}"
export WANDB_MODE="disabled"
export WANDB_DISABLED="true"
export WANDB_CONSOLE="off"
export WANDB_SILENT="true"
export RL_CASINO_DISABLE_TQDM="${RL_CASINO_DISABLE_TQDM:-1}"

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export HF_DATASETS_CACHE_ROOT="${HF_DATASETS_CACHE_ROOT:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_DATASETS_CACHE_ROOT}"

export H200_BSR_STEPS_PER_PHASE="${H200_BSR_STEPS_PER_PHASE:-25}"
export DPO_LEARNING_RATE="${DPO_LEARNING_RATE:-5e-7}"
export DPO_WARMUP_RATIO="${DPO_WARMUP_RATIO:-0.1}"
export DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-1024}"
export DPO_MAX_PROMPT_LENGTH="${DPO_MAX_PROMPT_LENGTH:-1024}"
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE="${DPO_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export DPO_GRADIENT_ACCUMULATION_STEPS="${DPO_GRADIENT_ACCUMULATION_STEPS:-64}"
export DPO_OPTIM="${DPO_OPTIM:-adamw_8bit}"

export BSR_USE_ATOMIC="${BSR_USE_ATOMIC:-0}"
export BSR_BATCH_CHUNKS="${BSR_BATCH_CHUNKS:-1}"
export RL_CASINO_BSR_GRAD_INPUT_MODE="${RL_CASINO_BSR_GRAD_INPUT_MODE:-dense}"

export RL_CASINO_BSR_DETAILED_TIMING="${RL_CASINO_BSR_DETAILED_TIMING:-0}"
export RL_CASINO_LOGGING_STEPS="${RL_CASINO_LOGGING_STEPS:-1}"
if [ "${RL_CASINO_LOGGING_STEPS}" -gt "${H200_BSR_STEPS_PER_PHASE}" ]; then
  echo "WARNING: RL_CASINO_LOGGING_STEPS=${RL_CASINO_LOGGING_STEPS} > H200_BSR_STEPS_PER_PHASE=${H200_BSR_STEPS_PER_PHASE}; clamping to 1." >&2
  export RL_CASINO_LOGGING_STEPS=1
fi

export H200_ABLATION_MODE="${H200_ABLATION_MODE:-optimizer}"
if [ "${H200_ABLATION_MODE}" != "optimizer" ]; then
  echo "WARNING: H200_ABLATION_MODE=${H200_ABLATION_MODE} unsupported in this deadline-safe mode; forcing optimizer-only benchmark." >&2
  export H200_ABLATION_MODE="optimizer"
fi

if [ "${H200_BSR_FULL_GRID:-0}" = "1" ]; then
  export BENCHMARK_SPARSITIES="${BENCHMARK_SPARSITIES:-99.75,97.5,95,90}"
  PY_GRID_ARGS=(
    --phase_mask_types "${H200_BSR_MASK_TYPES:-element,block}"
    --phase_adam_kernels "${H200_BSR_ADAM_KERNELS:-block_1d,block_2d}"
  )
else
  export BENCHMARK_SPARSITIES="${BENCHMARK_SPARSITIES:-99.75,97.5,95,90}"
  PY_GRID_ARGS=(
    --phase_mask_types "${H200_BSR_MASK_TYPES:-element}"
    --phase_adam_kernels "${H200_BSR_ADAM_KERNELS:-block_1d}"
  )
fi

export H200_BSR_SKIP_DENSE="${H200_BSR_SKIP_DENSE:-0}"
SKIP_DENSE_ARGS=()
if [ "${H200_BSR_SKIP_DENSE}" != "0" ]; then
  SKIP_DENSE_ARGS+=(--no_dense_baseline)
fi

PHASE_SLICE_ARGS=()
if [ "${H200_BSR_ARRAY_PHASE:-0}" = "1" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  _i="${SLURM_ARRAY_TASK_ID}"
  PHASE_SLICE_ARGS+=(--phase_start "${_i}" --phase_end "$((_i + 1))")
fi

OUT_BASE="${H200_BSR_OUT:-${SCRATCH_USER_ROOT}/rl_casino_h200_bsr}/${RUN_ID:-${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT_BASE"

echo "REPO_ROOT=${REPO_ROOT}"
echo "OUT_BASE=${OUT_BASE}"
echo "H200_BSR_FULL_GRID=${H200_BSR_FULL_GRID:-0}  BENCHMARK_SPARSITIES=${BENCHMARK_SPARSITIES}"
echo "H200_BSR_ARRAY_PHASE=${H200_BSR_ARRAY_PHASE:-0}  phase_slice=${PHASE_SLICE_ARGS[*]}"
echo "BSR_BATCH_CHUNKS=${BSR_BATCH_CHUNKS} (floor; kernel autoscales total programs)"

GC_ARGS=()
if [ "${DPO_GRADIENT_CHECKPOINTING:-1}" = "0" ]; then
  GC_ARGS+=(--no_gradient_checkpointing)
fi

set +e
BENCH_SCRIPT="src/full_training/h200_sparse_dpo_optimizer_benchmark.py"
"$TRAIN_PY" "${BENCH_SCRIPT}" \
  --model_name "$MODEL" \
  --dataset tulu3 \
  --n_steps "$H200_BSR_STEPS_PER_PHASE" \
  --batch_size "$DPO_PER_DEVICE_TRAIN_BATCH_SIZE" \
  --grad_accum "$DPO_GRADIENT_ACCUMULATION_STEPS" \
  --lr "$DPO_LEARNING_RATE" \
  --warmup_ratio "$DPO_WARMUP_RATIO" \
  --max_length "$DPO_MAX_LENGTH" \
  --max_prompt_length "$DPO_MAX_PROMPT_LENGTH" \
  --output_dir "$OUT_BASE" \
  --dataset_cache_dir "$HF_DATASETS_CACHE" \
  --run_label "h200_ablat_${SLURM_JOB_ID:-local}" \
  --benchmark_sparsities "$BENCHMARK_SPARSITIES" \
  "${PY_GRID_ARGS[@]}" \
  "${PHASE_SLICE_ARGS[@]}" \
  "${SKIP_DENSE_ARGS[@]}" \
  "${GC_ARGS[@]}"
_RC=$?
set -e

_SLURM_OUT="${REPO_ROOT}/logs/h200_speed_ablat_${SLURM_JOB_ID:-local}.out"
if [ -f "${_SLURM_OUT}" ]; then
  "$TRAIN_PY" "${REPO_ROOT}/scripts/report_h200_speed_ablation.py" \
    --run-dir "$OUT_BASE" \
    --slurm-out "${_SLURM_OUT}" || true
fi

exit "${_RC}"
