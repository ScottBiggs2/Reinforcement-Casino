#!/usr/bin/env bash
# GPU job: element-wise random mask at SPARSITY_PERCENT + submit sparse GRPO training with THE SAME
# hyperparameter env as dense (see scripts/grpo_training_env_defaults.sh + grpo_openr1_llama31_slurm.sh).
#
# Submit from repo root:
#   sbatch --export=ALL,HF_TOKEN="$HF_TOKEN" scripts/run_grpo_random_mask_and_sparse.slurm
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=grpo_rand_tr
#SBATCH --output=logs/run_grpo_random_mask_sparse_%j.out
#SBATCH --error=logs/run_grpo_random_mask_sparse_%j.err

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
export RL_CASINO_SCRATCH_ROOT="${RL_CASINO_SCRATCH_ROOT:-$SCRATCH_USER_ROOT}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export HF_TOKEN="${HF_TOKEN:-}"

export GRPO_MASK_DIR="${GRPO_MASK_DIR:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/masks}"
export GRPO_SPARSE_OUTPUT_BASE="${GRPO_SPARSE_OUTPUT_BASE:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/sparse}"
mkdir -p "${GRPO_MASK_DIR}"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/grpo_training_env_defaults.sh"

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export GRPO_DATASET="${GRPO_DATASET:-math-220k}"

export SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
export RANDOM_SEED="${RANDOM_SEED:-42}"

export ORCH_TRAIN_PARTITION_SPARSE="${ORCH_TRAIN_PARTITION_SPARSE:-gpu}"
export ORCH_TRAIN_GRES_SPARSE="${ORCH_TRAIN_GRES_SPARSE:-gpu:a100:1}"
export ORCH_TRAIN_TIME_GRPO="${ORCH_TRAIN_TIME_GRPO:-08:00:00}"

RUN_SUFFIX="${SLURM_JOB_ID:-manual}"
export GRPO_SPARSE_RANDOM_RUN_NAME="${GRPO_SPARSE_RANDOM_RUN_NAME:-grpo_sparse_random_sp${SPARSITY_PERCENT/./_}_steps${GRPO_TARGET_STEPS}_${RUN_SUFFIX}}"

if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found: ${TRAIN_PY}" >&2
  exit 1
fi

sanitize_model_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s//-/_}"
  s="${s,,}"
  s="$(echo "$s" | tr -c '[:alnum:]_' '_' )"
  while [[ "$s" == *"__"* ]]; do s="${s//__/_}"; done
  s="${s##_}"
  s="${s%%_}"
  echo "$s"
}

MODEL_SANITIZED="$(sanitize_model_name "$MODEL")"
MASK_RANDOM="${MASK_RANDOM:-${GRPO_MASK_DIR}/grpo_random_elem_sp${SPARSITY_PERCENT/./_}_${MODEL_SANITIZED}_math220k_seed${RANDOM_SEED}.pt}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "Random mask → ${MASK_RANDOM}"
echo "Sparse run name: ${GRPO_SPARSE_RANDOM_RUN_NAME}"
echo "Training hyperparams (must match dense): LR=${GRPO_LR} beta=${GRPO_BETA} seq=${GRPO_MAX_PROMPT_LENGTH}/${GRPO_MAX_COMPLETION_LENGTH} warmup_ratio=${GRPO_WARMUP_RATIO} max_grad_norm=${GRPO_MAX_GRAD_NORM}"

"$TRAIN_PY" - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("ERROR: need CUDA", file=sys.stderr)
    sys.exit(1)
print(f"CUDA OK {torch.cuda.get_device_name(0)!r}")
PY

echo ""
echo "=== Generating random element mask ==="
"$TRAIN_PY" src/utils/generate_random_mask.py \
  --model_name "${MODEL}" \
  --sparsity_percent "${SPARSITY_PERCENT}" \
  --min-layer-keep-ratio "${MIN_LAYER_KEEP_RATIO}" \
  --seed "${RANDOM_SEED}" \
  --mask-granularity element \
  --output_file "${MASK_RANDOM}"

ls -la "${MASK_RANDOM}"

orch_sanitize_job_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  s="$(echo "$s" | tr -cd '[:alnum:]_.-')"
  printf '%.64s' "$s"
}

echo ""
echo "=== Submitting sparse GRPO (random mask) ==="
SPARSE_EXPORT="ALL"
SPARSE_EXPORT+=",GRPO_MODE=sparse"
SPARSE_EXPORT+=",GRPO_MASK=${MASK_RANDOM}"
SPARSE_EXPORT+=",GRPO_RUN_NAME=${GRPO_SPARSE_RANDOM_RUN_NAME}"
SPARSE_EXPORT+=",GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS}"
SPARSE_EXPORT+=",GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH}"
SPARSE_EXPORT+=",GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH}"
SPARSE_EXPORT+=",GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}"
SPARSE_EXPORT+=",WANDB_MODE=online"
SPARSE_EXPORT+=",GRPO_SAVE_STEPS=${GRPO_SAVE_STEPS}"
SPARSE_EXPORT+=",GRPO_SAVE_TOTAL_LIMIT=${GRPO_SAVE_TOTAL_LIMIT}"
SPARSE_EXPORT+=",GRPO_SPARSE_OUTPUT_BASE=${GRPO_SPARSE_OUTPUT_BASE}"
SPARSE_EXPORT+=",MODEL=${MODEL}"
SPARSE_EXPORT+=",GRPO_DATASET=${GRPO_DATASET}"
SPARSE_EXPORT+=",GRPO_LR=${GRPO_LR}"
SPARSE_EXPORT+=",GRPO_BETA=${GRPO_BETA}"
SPARSE_EXPORT+=",GRPO_PER_DEVICE_BS=${GRPO_PER_DEVICE_BS}"
SPARSE_EXPORT+=",GRPO_GRAD_ACCUM=${GRPO_GRAD_ACCUM}"
SPARSE_EXPORT+=",GRPO_NUM_GEN=${GRPO_NUM_GEN}"
SPARSE_EXPORT+=",GRPO_GEN_BATCH=${GRPO_GEN_BATCH}"
SPARSE_EXPORT+=",GRPO_OPTIM=${GRPO_OPTIM}"
SPARSE_EXPORT+=",GRPO_PRECISION=${GRPO_PRECISION}"
SPARSE_EXPORT+=",GRPO_WARMUP_RATIO=${GRPO_WARMUP_RATIO}"
SPARSE_EXPORT+=",GRPO_MAX_GRAD_NORM=${GRPO_MAX_GRAD_NORM}"
SPARSE_EXPORT+=",HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
SPARSE_EXPORT+=",RL_CASINO_SCRATCH_ROOT=${RL_CASINO_SCRATCH_ROOT}"

jname="$(orch_sanitize_job_name "grpo_sp_random_${GRPO_SPARSE_RANDOM_RUN_NAME}")"
jid=$(sbatch --parsable \
  --partition="${ORCH_TRAIN_PARTITION_SPARSE}" \
  --nodes=1 \
  --ntasks=1 \
  --gres="${ORCH_TRAIN_GRES_SPARSE}" \
  --mem=128G \
  --time="${ORCH_TRAIN_TIME_GRPO}" \
  --cpus-per-task=8 \
  --job-name="${jname}" \
  --export="${SPARSE_EXPORT}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh")

echo "Submitted sparse GRPO random: ${jid}"
echo "  tail -f ${REPO_ROOT}/logs/grpo_openr1_${jid}.out"
