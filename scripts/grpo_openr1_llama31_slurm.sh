#!/usr/bin/env bash
# Open-R1 Math GRPO (dense or sparse BSR) with scratch under RL_CASINO_SCRATCH_ROOT.
# Hyperparameter defaults and rationale: docs/hyperparams/open_r1_llama31.yaml
# Runbook: docs/GRPO_OPEN_R1_RUNBOOK.md
#
# Submit from repo root: sbatch scripts/grpo_openr1_llama31_slurm.sh
#
# Examples:
#   GRPO_MODE=dense sbatch scripts/grpo_openr1_llama31_slurm.sh
#   GRPO_MODE=sparse GRPO_MASK=/path/to/mask.pt sbatch scripts/grpo_openr1_llama31_slurm.sh
#
# Sparse + SNIP mask (after generating a .pt with src/cold_start/inference_mask_finder.py --method snip):
#   export GRPO_MASK="${GRPO_MASK_DIR}/snip_grpo.pt"
#   export GRPO_MODE=sparse
#   sbatch scripts/grpo_openr1_llama31_slurm.sh
# For SNIP objectives (`--snip-objective lm` vs `dpo_preference`) and common env knobs, see docs/GRPO_OPEN_R1_RUNBOOK.md.
# Scratch layout (override any): GRPO_MASK_DIR, GRPO_DENSE_OUTPUT_BASE, GRPO_SPARSE_OUTPUT_BASE.
#   GRPO_RESUME=auto GRPO_TARGET_STEPS=1000 sbatch ...   # same run_slug / run_name as prior job
#
# Explorer / many sites: the gpu partition requires an explicit --gres; omitting it can yield
# "sbatch: error: Batch job submission failed: Access/permission denied". Match other H200 jobs:
#   scripts/h200_sparse_dpo_bsr_benchmark.sh, scripts/pipeline_stage_01_dense.sh
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=grpo_openr1
#SBATCH --output=logs/grpo_openr1_%j.out
#SBATCH --error=logs/grpo_openr1_%j.err
# Orchestrators submit sparse GRPO with sbatch --gres=gpu:a100:1 (overrides this header). If the scheduler
# rejects gpu:h200:1 (or you need a different GPU type), replace the --gres line above, e.g.:
# #SBATCH --gres=gpu:1

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

# Optional: scripts/train_with_auto_resume.sh — export USE_TRAIN_WITH_AUTO_RESUME=1 AUTO_RESUME_SOFT_SECONDS=...
# AUTO_RESUME_MODE defaults from GRPO_MODE (dense_grpo vs sparse_grpo).
if [ "${USE_TRAIN_WITH_AUTO_RESUME:-0}" = "1" ]; then
  if [ "${GRPO_MODE:-dense}" = "sparse" ]; then
    export AUTO_RESUME_MODE="${AUTO_RESUME_MODE:-sparse_grpo}"
  else
    export AUTO_RESUME_MODE="${AUTO_RESUME_MODE:-dense_grpo}"
  fi
  exec bash "${REPO_ROOT}/scripts/train_with_auto_resume.sh" "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh"
fi

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
export RL_CASINO_SCRATCH_ROOT="${RL_CASINO_SCRATCH_ROOT:-$SCRATCH_USER_ROOT}"
# GRPO artifacts (separate from DPO pipeline's rl_casino_masks / rl_casino_train).
export GRPO_MASK_DIR="${GRPO_MASK_DIR:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/masks}"
export GRPO_DENSE_OUTPUT_BASE="${GRPO_DENSE_OUTPUT_BASE:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/dense}"
export GRPO_SPARSE_OUTPUT_BASE="${GRPO_SPARSE_OUTPUT_BASE:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/sparse}"
mkdir -p "${GRPO_MASK_DIR}"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/grpo_training_env_defaults.sh"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found: ${TRAIN_PY}" >&2
  exit 1
fi
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRL_SKIP_VLLM_IMPORT="${TRL_SKIP_VLLM_IMPORT:-1}"
# Fail fast on login/CPU nodes (torch loads fp32 → huge RAM → OOM kill with no Python traceback).
cuda_preflight() {
  "${TRAIN_PY}" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print(
        "ERROR: CUDA is not available. Use a GPU allocation (sbatch/salloc with --gres). "
        "On Explorer, `python -c \"import torch; print(torch.cuda.is_available())\"` must print True.",
        file=sys.stderr,
    )
    sys.exit(1)
name = torch.cuda.get_device_name(0)
print(f"CUDA preflight OK: device_count={torch.cuda.device_count()} device0={name!r}")
PY
}

# GRPO_train.py / sparse_grpo_bsr.py always log to W&B; scrub login-shell disables that survive sbatch --export=ALL.
unset WANDB_DISABLED 2>/dev/null || true
unset WANDB_SILENT 2>/dev/null || true
export WANDB_MODE="online"

echo "REPO_ROOT=${REPO_ROOT}"
echo "RL_CASINO_SCRATCH_ROOT=${RL_CASINO_SCRATCH_ROOT}"
echo "GRPO_MASK_DIR=${GRPO_MASK_DIR}"
echo "GRPO_DENSE_OUTPUT_BASE=${GRPO_DENSE_OUTPUT_BASE}"
echo "GRPO_SPARSE_OUTPUT_BASE=${GRPO_SPARSE_OUTPUT_BASE}"
echo "GRPO_MODE=${GRPO_MODE} GRPO_NGPUS=${GRPO_NGPUS}"
echo "GRPO_HPARAM_MODE=${GRPO_HPARAM_MODE:-unknown}  (override ablations: export GRPO_HPARAM_OVERRIDE=1 before sbatch)"
echo "MODEL=${MODEL} DATASET=${GRPO_DATASET} STEPS=${GRPO_TARGET_STEPS} RESUME=${GRPO_RESUME:-<none>}"
echo "LR=${GRPO_LR} beta=${GRPO_BETA} bs=${GRPO_PER_DEVICE_BS} accum=${GRPO_GRAD_ACCUM} gen=${GRPO_NUM_GEN}/${GRPO_GEN_BATCH}"
echo "seq caps: prompt=${GRPO_MAX_PROMPT_LENGTH} completion=${GRPO_MAX_COMPLETION_LENGTH} reward=${GRPO_REWARD_PROFILE}"
echo "warmup_ratio=${GRPO_WARMUP_RATIO} max_grad_norm=${GRPO_MAX_GRAD_NORM}"

# Sparse only: integer warmup steps ≈ dense (ratio × total steps). Uses frozen env only (no silent fallbacks).
GRPO_SPARSE_WARMUP_STEPS="$("${TRAIN_PY}" -c "import os; s=int(os.environ['GRPO_TARGET_STEPS']); r=float(os.environ['GRPO_WARMUP_RATIO']); print(max(0, int(s*r)))")"
export GRPO_SPARSE_WARMUP_STEPS
echo "sparse_warmup_steps=${GRPO_SPARSE_WARMUP_STEPS} (matches dense warmup_ratio×steps)"

cuda_preflight || exit 1

LAUNCHER=( "${TRAIN_PY}" )
if [ "${GRPO_NGPUS}" -gt 1 ]; then
  LAUNCHER=( "${TRAIN_ENV}/bin/torchrun" --standalone --nproc_per_node="${GRPO_NGPUS}" )
fi

RESUME_ARGS=()
if [ -n "${GRPO_RESUME:-}" ]; then
  RESUME_ARGS+=( --resume_from_checkpoint "${GRPO_RESUME}" )
fi

RUN_SLUG_ARGS=()
if [ -n "${GRPO_RUN_SLUG:-}" ]; then
  RUN_SLUG_ARGS+=( --run_slug "${GRPO_RUN_SLUG}" )
fi

RUN_NAME_ARGS=()
if [ -n "${GRPO_RUN_NAME:-}" ]; then
  RUN_NAME_ARGS+=( --run_name "${GRPO_RUN_NAME}" )
fi

SPARSE_LAZY_ARGS=()
if [ "${GRPO_SPARSE_ADAMW_LAZY:-0}" = "1" ]; then
  SPARSE_LAZY_ARGS+=( --sparse_adamw_lazy_state )
fi

# Dense-only: forward optional weight-delta logging into GRPO_train.py.
DELTA_ARGS=()
if [ -n "${GRPO_DELTA_LOG_INTERVAL:-}" ]; then
  DELTA_ARGS+=( --delta_log_interval "${GRPO_DELTA_LOG_INTERVAL}" )
fi
if [ -n "${GRPO_DELTA_LOG_END_STEP:-}" ]; then
  DELTA_ARGS+=( --delta_log_end_step "${GRPO_DELTA_LOG_END_STEP}" )
fi

if [ "${GRPO_MODE}" = "dense" ]; then
  "${LAUNCHER[@]}" src/full_training/GRPO_train.py \
    --model_name "${MODEL}" \
    --dataset "${GRPO_DATASET}" \
    --num_steps "${GRPO_TARGET_STEPS}" \
    --learning_rate "${GRPO_LR}" \
    --beta "${GRPO_BETA}" \
    --per_device_train_batch_size "${GRPO_PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${GRPO_GRAD_ACCUM}" \
    --num_generations "${GRPO_NUM_GEN}" \
    --generation_batch_size "${GRPO_GEN_BATCH}" \
    --save_steps "${GRPO_SAVE_STEPS}" \
    --save_total_limit "${GRPO_SAVE_TOTAL_LIMIT}" \
    --max_prompt_length "${GRPO_MAX_PROMPT_LENGTH}" \
    --max_completion_length "${GRPO_MAX_COMPLETION_LENGTH}" \
    --precision "${GRPO_PRECISION}" \
    --optim "${GRPO_OPTIM}" \
    --grpo_reward_profile "${GRPO_REWARD_PROFILE}" \
    --warmup_ratio "${GRPO_WARMUP_RATIO}" \
    --max_grad_norm "${GRPO_MAX_GRAD_NORM}" \
    --output_base_dir "${GRPO_DENSE_OUTPUT_BASE}" \
    --dataset_cache_dir "${HF_DATASETS_CACHE}" \
    "${RUN_SLUG_ARGS[@]}" \
    "${RUN_NAME_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "${DELTA_ARGS[@]}"
elif [ "${GRPO_MODE}" = "sparse" ]; then
  if [ -z "${GRPO_MASK:-}" ] || [ ! -f "${GRPO_MASK}" ]; then
    echo "ERROR: set GRPO_MASK to a .pt mask file for sparse mode." >&2
    exit 1
  fi
  "${LAUNCHER[@]}" src/full_training/sparse_grpo_bsr.py \
    --model_name "${MODEL}" \
    --checkpoint "${MODEL}" \
    --mask "${GRPO_MASK}" \
    --dataset "${GRPO_DATASET}" \
    --n_steps "${GRPO_TARGET_STEPS}" \
    --lr "${GRPO_LR}" \
    --grpo_beta "${GRPO_BETA}" \
    --batch_size "${GRPO_PER_DEVICE_BS}" \
    --grad_accum "${GRPO_GRAD_ACCUM}" \
    --num_generations "${GRPO_NUM_GEN}" \
    --generation_batch_size "${GRPO_GEN_BATCH}" \
    --save_steps "${GRPO_SAVE_STEPS}" \
    --save_total_limit "${GRPO_SAVE_TOTAL_LIMIT}" \
    --max_prompt_length "${GRPO_MAX_PROMPT_LENGTH}" \
    --max_completion_length "${GRPO_MAX_COMPLETION_LENGTH}" \
    --precision "${GRPO_PRECISION}" \
    --grpo_reward_profile "${GRPO_REWARD_PROFILE}" \
    --warmup_steps "${GRPO_SPARSE_WARMUP_STEPS}" \
    --max_grad_norm "${GRPO_MAX_GRAD_NORM}" \
    --output_base_dir "${GRPO_SPARSE_OUTPUT_BASE}" \
    --dataset_cache_dir "${HF_DATASETS_CACHE}" \
    "${RUN_NAME_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "${SPARSE_LAZY_ARGS[@]}"
else
  echo "ERROR: GRPO_MODE must be 'dense' or 'sparse'." >&2
  exit 1
fi

echo ""
echo "To resume this run after requeue, re-submit with the same GRPO_RUN_SLUG (dense) or"
echo "GRPO_RUN_NAME (sparse) and: export GRPO_RESUME=auto"
echo "W&B id is stored under the run directory as wandb_run_id.txt"
echo "Hyperparameter record: docs/hyperparams/open_r1_llama31.yaml"
