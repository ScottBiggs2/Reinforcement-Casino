#!/usr/bin/env bash
# One GPU job: build element-wise checkpoint-diff oracle mask (|θ_final − θ_init|),
# then submit sparse GRPO with the same locked training env as other GRPO children.
#
# Defaults match the May 2026 Open-R1 500-step dense run under rl_casino_grpo; override via env.
#
# Submit from repo root:
#   sbatch --export=ALL,HF_TOKEN="$HF_TOKEN" scripts/run_grpo_oracle_checkpoint_diff_and_sparse.slurm
#
# Optional overrides (export before sbatch):
#   GRPO_ORACLE_INITIAL_MODEL, GRPO_ORACLE_FINAL_CKPT, ORACLE_MASK_OUT,
#   GRPO_MASK_DIR, GRPO_SPARSE_OUTPUT_BASE, GRPO_SPARSE_ORACLE_RUN_NAME,
#   SPARSITY_PERCENT, MIN_LAYER_KEEP_RATIO,
#   ORCH_TRAIN_PARTITION, ORCH_TRAIN_GRES, ORCH_TRAIN_TIME, ORCH_TRAIN_MEM, ORCH_TRAIN_CPUS
#
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=grpo_oracle_tr
#SBATCH --output=logs/run_grpo_oracle_mask_sparse_%j.out
#SBATCH --error=logs/run_grpo_oracle_mask_sparse_%j.err

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

# HF init vs local trainer checkpoint (directory with config.json + model shards).
export GRPO_ORACLE_INITIAL_MODEL="${GRPO_ORACLE_INITIAL_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export GRPO_ORACLE_FINAL_CKPT="${GRPO_ORACLE_FINAL_CKPT:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/dense/grpo_dense_openr1_steps500_with_deltas_6512509/checkpoints/checkpoint-500}"

SP_TAG="$(echo "${SPARSITY_PERCENT}" | tr '.' '_')"
RUN_SUFFIX="${SLURM_JOB_ID:-manual}"
ORACLE_SUBDIR="${ORACLE_MASK_SUBDIR:-oracle_${RUN_SUFFIX}}"
# Default filename matches paper_snippets.md (checkpoint-500 + sparsityXpct).
MASK_ORACLE="${ORACLE_MASK_OUT:-${GRPO_MASK_DIR}/${ORACLE_SUBDIR}/checkpoint_diff_ground_truth_checkpoint-500_sparsity${SPARSITY_PERCENT}pct.pt}"
mkdir -p "$(dirname "${MASK_ORACLE}")"

export GRPO_SPARSE_ORACLE_RUN_NAME="${GRPO_SPARSE_ORACLE_RUN_NAME:-grpo_sparse_oracle_ckpt500_sp${SP_TAG}_steps${GRPO_TARGET_STEPS}_${RUN_SUFFIX}}"

ORCH_TRAIN_PARTITION="${ORCH_TRAIN_PARTITION:-multigpu}"
ORCH_TRAIN_GRES="${ORCH_TRAIN_GRES:-gpu:h200:1}"
ORCH_TRAIN_TIME="${ORCH_TRAIN_TIME:-08:00:00}"
ORCH_TRAIN_MEM="${ORCH_TRAIN_MEM:-128G}"
ORCH_TRAIN_CPUS="${ORCH_TRAIN_CPUS:-8}"

if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found: ${TRAIN_PY}" >&2
  exit 1
fi

echo "================================================================="
echo " GRPO oracle: checkpoint-diff mask + sparse train submit"
echo "================================================================="
echo "REPO_ROOT=${REPO_ROOT}"
echo "initial_model=${GRPO_ORACLE_INITIAL_MODEL}"
echo "final_model  =${GRPO_ORACLE_FINAL_CKPT}"
echo "mask out     =${MASK_ORACLE}"
echo "sparse name  =${GRPO_SPARSE_ORACLE_RUN_NAME}"
echo "Verify checkpoint dir (mtime / layout):"
ls -ld "${GRPO_ORACLE_FINAL_CKPT}" 2>/dev/null || { echo "ERROR: final checkpoint path missing." >&2; exit 1; }
if command -v stat >/dev/null 2>&1; then
  stat "${GRPO_ORACLE_FINAL_CKPT}" || true
  if [ -f "${GRPO_ORACLE_FINAL_CKPT}/trainer_state.json" ]; then
    stat "${GRPO_ORACLE_FINAL_CKPT}/trainer_state.json" || true
  fi
fi
ls -la "${GRPO_ORACLE_FINAL_CKPT}" | head -n 15
echo "================================================================="

"$TRAIN_PY" - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("ERROR: need CUDA for this job template", file=sys.stderr)
    sys.exit(1)
print(f"CUDA OK {torch.cuda.get_device_name(0)!r}")
PY

echo ""
echo "=== [1/2] Building checkpoint-diff oracle mask (element-wise) ==="
"$TRAIN_PY" src/warm_start/checkpoint_diff_mask_finder.py \
  --initial_model "${GRPO_ORACLE_INITIAL_MODEL}" \
  --final_model "${GRPO_ORACLE_FINAL_CKPT}" \
  --sparsity_percent "${SPARSITY_PERCENT}" \
  --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
  --mask-granularity element \
  --output_file "${MASK_ORACLE}"

if [ ! -f "${MASK_ORACLE}" ]; then
  echo "ERROR: oracle mask not written: ${MASK_ORACLE}" >&2
  exit 1
fi
ls -la "${MASK_ORACLE}"

orch_sanitize_job_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  s="$(echo "$s" | tr -cd '[:alnum:]_.-')"
  printf '%.64s' "$s"
}

echo ""
echo "=== [2/2] Submitting sparse GRPO (oracle mask) ==="
SPARSE_EXPORT="ALL"
SPARSE_EXPORT+=",GRPO_MODE=sparse"
SPARSE_EXPORT+=",GRPO_MASK=${MASK_ORACLE}"
SPARSE_EXPORT+=",GRPO_RUN_NAME=${GRPO_SPARSE_ORACLE_RUN_NAME}"
SPARSE_EXPORT+=",GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS}"
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
SPARSE_EXPORT+=",GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH}"
SPARSE_EXPORT+=",GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH}"
SPARSE_EXPORT+=",GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}"
SPARSE_EXPORT+=",GRPO_WARMUP_RATIO=${GRPO_WARMUP_RATIO}"
SPARSE_EXPORT+=",GRPO_MAX_GRAD_NORM=${GRPO_MAX_GRAD_NORM}"
SPARSE_EXPORT+=",GRPO_SAVE_STEPS=${GRPO_SAVE_STEPS}"
SPARSE_EXPORT+=",GRPO_SAVE_TOTAL_LIMIT=${GRPO_SAVE_TOTAL_LIMIT}"
SPARSE_EXPORT+=",GRPO_SPARSE_OUTPUT_BASE=${GRPO_SPARSE_OUTPUT_BASE}"
SPARSE_EXPORT+=",HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
SPARSE_EXPORT+=",RL_CASINO_SCRATCH_ROOT=${RL_CASINO_SCRATCH_ROOT}"
SPARSE_EXPORT+=",WANDB_MODE=online"

jname="$(orch_sanitize_job_name "grpo_sp_oracle_${GRPO_SPARSE_ORACLE_RUN_NAME}")"
jid=$(sbatch --parsable \
  --partition="${ORCH_TRAIN_PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --gres="${ORCH_TRAIN_GRES}" \
  --mem="${ORCH_TRAIN_MEM}" \
  --time="${ORCH_TRAIN_TIME}" \
  --cpus-per-task="${ORCH_TRAIN_CPUS}" \
  --job-name="${jname}" \
  --export="${SPARSE_EXPORT}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh")

echo "Submitted sparse GRPO (oracle): ${jid}"
echo "  tail -f ${REPO_ROOT}/logs/grpo_openr1_${jid}.out"
echo "Oracle mask: ${MASK_ORACLE}"
