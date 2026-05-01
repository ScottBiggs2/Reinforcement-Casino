#!/usr/bin/env bash
# GPU job — single submission:
#   1) Generate Open-R1 GRPO GraSP element/SNR mask (.pt).
#   2) Immediately sbatch sparse GRPO training with that mask (same exports as orchestrate_grpo_3way_focused).
#
# Dense / magnitude flows are intentionally NOT started here — use orchestrate_grpo_3way_focused.slurm for that.
#
# Mask only (skip training submit):
#   sbatch --export=ALL,...,SKIP_GRASP_SPARSE_LAUNCH=1 scripts/run_grpo_grasp_mask_only.slurm
#
# Submit from repo root:
#   sbatch --export=ALL,HF_TOKEN="$HF_TOKEN" scripts/run_grpo_grasp_mask_only.slurm
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=grpo_grsp_ms
#SBATCH --output=logs/run_grpo_grasp_mask_sparse_%j.out
#SBATCH --error=logs/run_grpo_grasp_mask_sparse_%j.err

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
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"

export GRPO_MASK_DIR="${GRPO_MASK_DIR:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/masks}"
export GRPO_SPARSE_OUTPUT_BASE="${GRPO_SPARSE_OUTPUT_BASE:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/sparse}"
mkdir -p "${GRPO_MASK_DIR}"

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export GRPO_DATASET="${GRPO_DATASET:-math-220k}"
export GRPO_DATASET_HF="${GRPO_DATASET_HF:-open-r1/OpenR1-Math-220k}"

export SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
export GRPO_GRASP_OBJECTIVE="${GRPO_GRASP_OBJECTIVE:-lm}"
export GRPO_GRASP_PREFERENCE_BETA="${GRPO_GRASP_PREFERENCE_BETA:-1.0}"
export GRPO_N_SAMPLES="${GRPO_N_SAMPLES:-256}"
export ORCH_MASK_BATCH_SIZE="${ORCH_MASK_BATCH_SIZE:-1}"
export GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-512}"

export ORCH_SCORE_SNR_MODE="${ORCH_SCORE_SNR_MODE:-per_weight}"
export ORCH_SCORE_SNR_TRANSFORM="${ORCH_SCORE_SNR_TRANSFORM:-log1p}"
export ORCH_SCORE_SNR_EPS="${ORCH_SCORE_SNR_EPS:-1e-8}"
export ORCH_SCORE_SNR_RAM_BUDGET_GB="${ORCH_SCORE_SNR_RAM_BUDGET_GB:-96}"
export ORCH_SCORE_SNR_ALLOW_LARGE_RAM="${ORCH_SCORE_SNR_ALLOW_LARGE_RAM:-1}"

export GRPO_TARGET_STEPS="${GRPO_TARGET_STEPS:-800}"
export GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-2048}"
export GRPO_REWARD_PROFILE="${GRPO_REWARD_PROFILE:-llama_cot}"
export GRPO_SAVE_STEPS="${GRPO_SAVE_STEPS:-50}"
export GRPO_SAVE_TOTAL_LIMIT="${GRPO_SAVE_TOTAL_LIMIT:-3}"
export GRPO_SPARSE_GRASP_RUN_NAME="${GRPO_SPARSE_GRASP_RUN_NAME:-grpo_sparse_grasp_${GRPO_GRASP_OBJECTIVE}_elem_snr${ORCH_SCORE_SNR_MODE}_${ORCH_SCORE_SNR_TRANSFORM}_steps${GRPO_TARGET_STEPS}_v1}"

SKIP_GRASP_SPARSE_LAUNCH="${SKIP_GRASP_SPARSE_LAUNCH:-0}"

EXTRA_GRASP_FLAGS="${EXTRA_GRASP_FLAGS:-}"

orch_sanitize_job_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  s="$(echo "$s" | tr -cd '[:alnum:]_.-')"
  printf '%.64s' "$s"
}

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
MASK_GRPO_GRASP_ELEM_SNR="${MASK_GRPO_GRASP_ELEM_SNR:-${GRPO_MASK_DIR}/grpo_grasp_${GRPO_GRASP_OBJECTIVE}_sp${SPARSITY_PERCENT/./_}_${MODEL_SANITIZED}_${GRPO_DATASET_HF##*/}_elem_snr${ORCH_SCORE_SNR_MODE}_${ORCH_SCORE_SNR_TRANSFORM}.pt}"

SNR_ARGS=( --score-snr "${ORCH_SCORE_SNR_MODE}"
           --score-snr-transform "${ORCH_SCORE_SNR_TRANSFORM}"
           --score-snr-eps "${ORCH_SCORE_SNR_EPS}"
           --score-snr-ram-budget-gb "${ORCH_SCORE_SNR_RAM_BUDGET_GB}" )
if [ "${ORCH_SCORE_SNR_ALLOW_LARGE_RAM:-0}" = "1" ]; then
  SNR_ARGS+=( --score-snr-allow-large-ram )
fi

echo "Sparse output base: ${GRPO_SPARSE_OUTPUT_BASE}"
echo "Output mask: ${MASK_GRPO_GRASP_ELEM_SNR}"
echo "MODEL=${MODEL}  DATASET_HF=${GRPO_DATASET_HF}  n_samples=${GRPO_N_SAMPLES}"
echo "SKIP_GRASP_SPARSE_LAUNCH=${SKIP_GRASP_SPARSE_LAUNCH}"

"$TRAIN_PY" - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("ERROR: need CUDA", file=sys.stderr)
    sys.exit(1)
print(f"CUDA OK {torch.cuda.get_device_name(0)!r}")
PY

"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
  --model_name "${MODEL}" \
  --mode grpo \
  --method grasp \
  --grasp-objective "${GRPO_GRASP_OBJECTIVE}" \
  --dataset_name "${GRPO_DATASET_HF}" \
  --n_samples "${GRPO_N_SAMPLES}" \
  --sparsity "${SPARSITY_PERCENT}" \
  --max_length "${GRPO_MAX_PROMPT_LENGTH}" \
  --batch_size "${ORCH_MASK_BATCH_SIZE}" \
  --seed 42 \
  --grasp-preference-beta "${GRPO_GRASP_PREFERENCE_BETA}" \
  --min-layer-keep-ratio "${MIN_LAYER_KEEP_RATIO}" \
  "${SNR_ARGS[@]}" \
  --mask-granularity element \
  --output "${MASK_GRPO_GRASP_ELEM_SNR}" \
  ${EXTRA_GRASP_FLAGS}

echo ""
echo "OK — mask written:"
ls -la "${MASK_GRPO_GRASP_ELEM_SNR}"

if [ "${SKIP_GRASP_SPARSE_LAUNCH}" != "0" ]; then
  echo "SKIP_GRASP_SPARSE_LAUNCH set — done (no sparse training sbatch)."
  exit 0
fi

echo ""
echo "=== Submitting sparse GRPO with GraSP-elem-SNR mask (child job; separate GPU allocation) ==="
SPARSE_GRASP_EXPORT="ALL"
SPARSE_GRASP_EXPORT+=",GRPO_MODE=sparse"
SPARSE_GRASP_EXPORT+=",GRPO_MASK=${MASK_GRPO_GRASP_ELEM_SNR}"
SPARSE_GRASP_EXPORT+=",GRPO_RUN_NAME=${GRPO_SPARSE_GRASP_RUN_NAME}"
SPARSE_GRASP_EXPORT+=",GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS}"
SPARSE_GRASP_EXPORT+=",GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH}"
SPARSE_GRASP_EXPORT+=",GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH}"
SPARSE_GRASP_EXPORT+=",GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}"
SPARSE_GRASP_EXPORT+=",WANDB_MODE=online"
SPARSE_GRASP_EXPORT+=",GRPO_SAVE_STEPS=${GRPO_SAVE_STEPS}"
SPARSE_GRASP_EXPORT+=",GRPO_SAVE_TOTAL_LIMIT=${GRPO_SAVE_TOTAL_LIMIT}"
SPARSE_GRASP_EXPORT+=",GRPO_SPARSE_OUTPUT_BASE=${GRPO_SPARSE_OUTPUT_BASE}"
SPARSE_GRASP_EXPORT+=",MODEL=${MODEL}"
SPARSE_GRASP_EXPORT+=",GRPO_DATASET=${GRPO_DATASET}"

grasp_jname="$(orch_sanitize_job_name "grpo_sp_grasp_${GRPO_SPARSE_GRASP_RUN_NAME}")"
grasp_jid=$(sbatch --parsable \
  --job-name="${grasp_jname}" \
  --export="${SPARSE_GRASP_EXPORT}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh")
echo "Submitted GRPO sparse GraSP-elem-SNR job id: ${grasp_jid}"
echo "Tail: tail -f ${REPO_ROOT}/logs/grpo_openr1_${grasp_jid}.out"
