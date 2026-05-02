#!/usr/bin/env bash
# GPU job — end-to-end (masks + sparse training submits):
#   1) Vanilla GraSP-ABS  (--score-snr off, --grasp-rank-direction abs) → mask .pt
#   2) SNR + GraSP-ABS    (--score-snr per_weight + log1p, --grasp-rank-direction abs) → mask .pt
#   3) sbatch TWO sparse GRPO children (separate GPU allocations), one per mask — same pattern as
#      scripts/run_grpo_grasp_mask_only.slurm
#
# Submit from repo root (non-interactive):
#   sbatch --export=ALL,HF_TOKEN="$HF_TOKEN" scripts/run_grpo_grasp_vanilla_vs_snr_abs.slurm
#
# Mask-only (no training children):  SKIP_GRASP_SPARSE_LAUNCH=1
#
# Optional overrides (same env names as run_grpo_grasp_mask_only.slurm):
#   MODEL, GRPO_DATASET_HF, GRPO_DATASET, SPARSITY_PERCENT, GRPO_N_SAMPLES, GRASP_EXTRA_FLAGS,
#   GRPO_TARGET_STEPS, GRPO_SPARSE_OUTPUT_BASE, SKIP_GRASP_SPARSE_LAUNCH, GRPO_MASK_SUBDIR, etc.
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=grpo_gr2mask
#SBATCH --output=logs/run_grpo_grasp_vanilla_snr_abs_%j.out
#SBATCH --error=logs/run_grpo_grasp_vanilla_snr_abs_%j.err

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
export GRPO_DATASET_HF="${GRPO_DATASET_HF:-open-r1/OpenR1-Math-220k}"

export SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
export GRPO_GRASP_OBJECTIVE="${GRPO_GRASP_OBJECTIVE:-lm}"
export GRPO_GRASP_PREFERENCE_BETA="${GRPO_GRASP_PREFERENCE_BETA:-1.0}"
export GRPO_N_SAMPLES="${GRPO_N_SAMPLES:-256}"
export ORCH_MASK_BATCH_SIZE="${ORCH_MASK_BATCH_SIZE:-1}"
export GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-512}"

# SNR branch only (vanilla run ignores these except for output naming clarity).
export ORCH_SCORE_SNR_TRANSFORM="${ORCH_SCORE_SNR_TRANSFORM:-log1p}"
export ORCH_SCORE_SNR_EPS="${ORCH_SCORE_SNR_EPS:-1e-8}"
export ORCH_SCORE_SNR_RAM_BUDGET_GB="${ORCH_SCORE_SNR_RAM_BUDGET_GB:-96}"
export ORCH_SCORE_SNR_ALLOW_LARGE_RAM="${ORCH_SCORE_SNR_ALLOW_LARGE_RAM:-1}"

# Sparse GRPO training (children): scheduling only — hyperparams from grpo_training_env_defaults.sh
export ORCH_TRAIN_PARTITION_SPARSE="${ORCH_TRAIN_PARTITION_SPARSE:-gpu}"
export ORCH_TRAIN_GRES_SPARSE="${ORCH_TRAIN_GRES_SPARSE:-gpu:a100:1}"
export ORCH_TRAIN_TIME_GRPO="${ORCH_TRAIN_TIME_GRPO:-08:00:00}"

SKIP_GRASP_SPARSE_LAUNCH="${SKIP_GRASP_SPARSE_LAUNCH:-0}"

# Optional extra flags for inference_mask_finder (e.g. --also-emit-rank-variants)
GRASP_EXTRA_FLAGS="${GRASP_EXTRA_FLAGS:-}"

# Optional subdirectory under GRPO_MASK_DIR for this batch (default: job-specific).
GRPO_MASK_SUBDIR="${GRPO_MASK_SUBDIR:-grasp_cmp_${SLURM_JOB_ID:-manual}}"
MASK_SUBDIR_FULL="${GRPO_MASK_DIR}/${GRPO_MASK_SUBDIR}"
mkdir -p "${MASK_SUBDIR_FULL}"

if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found: ${TRAIN_PY}" >&2
  exit 1
fi

orch_sanitize_job_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  s="$(echo "$s" | tr -cd '[:alnum:]_.-')"
  printf '%.64s' "$s"
}

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
DS_TAIL="${GRPO_DATASET_HF##*/}"
SP_TAG="$(echo "${SPARSITY_PERCENT}" | tr '.' '_')"

BASE_STEM="grpo_grasp_${GRPO_GRASP_OBJECTIVE}_sp${SP_TAG}_${MODEL_SANITIZED}_${DS_TAIL}_elem_rankabs"
OUT_VANILLA="${MASK_SUBDIR_FULL}/${BASE_STEM}_snroff.pt"
OUT_SNR_ABS="${MASK_SUBDIR_FULL}/${BASE_STEM}_snrperweight_${ORCH_SCORE_SNR_TRANSFORM}.pt"

echo "REPO_ROOT=${REPO_ROOT}"
echo "MASK_SUBDIR_FULL=${MASK_SUBDIR_FULL}"
echo "Vanilla ABS → ${OUT_VANILLA}"
echo "SNR+ABS     → ${OUT_SNR_ABS}"

"$TRAIN_PY" - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("ERROR: need CUDA", file=sys.stderr)
    sys.exit(1)
print(f"CUDA OK {torch.cuda.get_device_name(0)!r}")
PY

common_grasp_args=(
  --model_name "${MODEL}"
  --mode grpo
  --method grasp
  --grasp-objective "${GRPO_GRASP_OBJECTIVE}"
  --dataset_name "${GRPO_DATASET_HF}"
  --n_samples "${GRPO_N_SAMPLES}"
  --sparsity "${SPARSITY_PERCENT}"
  --max_length "${GRPO_MAX_PROMPT_LENGTH}"
  --batch_size "${ORCH_MASK_BATCH_SIZE}"
  --seed 42
  --grasp-preference-beta "${GRPO_GRASP_PREFERENCE_BETA}"
  --min-layer-keep-ratio "${MIN_LAYER_KEEP_RATIO}"
  --grasp-rank-direction abs
  --mask-granularity element
)

echo ""
echo "=== [1/2] Vanilla GraSP (SNR off) + ABS ranking ==="
"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
  "${common_grasp_args[@]}" \
  --output "${OUT_VANILLA}" \
  ${GRASP_EXTRA_FLAGS}

echo ""
echo "=== [2/2] GraSP + per-weight SNR + ABS ranking ==="
SNR_ARGS=(
  --score-snr per_weight
  --score-snr-transform "${ORCH_SCORE_SNR_TRANSFORM}"
  --score-snr-eps "${ORCH_SCORE_SNR_EPS}"
  --score-snr-ram-budget-gb "${ORCH_SCORE_SNR_RAM_BUDGET_GB}"
)
if [ "${ORCH_SCORE_SNR_ALLOW_LARGE_RAM:-0}" = "1" ]; then
  SNR_ARGS+=( --score-snr-allow-large-ram )
fi

"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
  "${common_grasp_args[@]}" \
  "${SNR_ARGS[@]}" \
  --output "${OUT_SNR_ABS}" \
  ${GRASP_EXTRA_FLAGS}

echo ""
echo "Done. Outputs:"
ls -la "${OUT_VANILLA}" "${OUT_SNR_ABS}"

if [ "${SKIP_GRASP_SPARSE_LAUNCH}" != "0" ]; then
  echo ""
  echo "SKIP_GRASP_SPARSE_LAUNCH=${SKIP_GRASP_SPARSE_LAUNCH} — not submitting sparse GRPO children."
  exit 0
fi

RUN_SUFFIX="${SLURM_JOB_ID:-manual}"
GRPO_RUN_NAME_VANILLA="${GRPO_RUN_NAME_VANILLA:-grpo_sparse_grasp_${GRPO_GRASP_OBJECTIVE}_vanilla_abs_steps${GRPO_TARGET_STEPS}_${RUN_SUFFIX}}"
GRPO_RUN_NAME_SNR="${GRPO_RUN_NAME_SNR:-grpo_sparse_grasp_${GRPO_GRASP_OBJECTIVE}_snrperweight_${ORCH_SCORE_SNR_TRANSFORM}_abs_steps${GRPO_TARGET_STEPS}_${RUN_SUFFIX}}"

echo ""
echo "=== Submitting sparse GRPO (vanilla GraSP-ABS mask) → job 1 ==="
SPARSE_V_EXPORT="ALL"
SPARSE_V_EXPORT+=",GRPO_MODE=sparse"
SPARSE_V_EXPORT+=",GRPO_MASK=${OUT_VANILLA}"
SPARSE_V_EXPORT+=",GRPO_RUN_NAME=${GRPO_RUN_NAME_VANILLA}"
SPARSE_V_EXPORT+=",GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS}"
SPARSE_V_EXPORT+=",GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH}"
SPARSE_V_EXPORT+=",GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH}"
SPARSE_V_EXPORT+=",GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}"
SPARSE_V_EXPORT+=",WANDB_MODE=online"
SPARSE_V_EXPORT+=",GRPO_SAVE_STEPS=${GRPO_SAVE_STEPS}"
SPARSE_V_EXPORT+=",GRPO_SAVE_TOTAL_LIMIT=${GRPO_SAVE_TOTAL_LIMIT}"
SPARSE_V_EXPORT+=",GRPO_SPARSE_OUTPUT_BASE=${GRPO_SPARSE_OUTPUT_BASE}"
SPARSE_V_EXPORT+=",MODEL=${MODEL}"
SPARSE_V_EXPORT+=",GRPO_DATASET=${GRPO_DATASET}"
SPARSE_V_EXPORT+=",GRPO_LR=${GRPO_LR}"
SPARSE_V_EXPORT+=",GRPO_BETA=${GRPO_BETA}"
SPARSE_V_EXPORT+=",GRPO_PER_DEVICE_BS=${GRPO_PER_DEVICE_BS}"
SPARSE_V_EXPORT+=",GRPO_GRAD_ACCUM=${GRPO_GRAD_ACCUM}"
SPARSE_V_EXPORT+=",GRPO_NUM_GEN=${GRPO_NUM_GEN}"
SPARSE_V_EXPORT+=",GRPO_GEN_BATCH=${GRPO_GEN_BATCH}"
SPARSE_V_EXPORT+=",GRPO_OPTIM=${GRPO_OPTIM}"
SPARSE_V_EXPORT+=",GRPO_PRECISION=${GRPO_PRECISION}"
SPARSE_V_EXPORT+=",GRPO_WARMUP_RATIO=${GRPO_WARMUP_RATIO}"
SPARSE_V_EXPORT+=",GRPO_MAX_GRAD_NORM=${GRPO_MAX_GRAD_NORM}"
SPARSE_V_EXPORT+=",HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
SPARSE_V_EXPORT+=",RL_CASINO_SCRATCH_ROOT=${RL_CASINO_SCRATCH_ROOT}"

jname_v="$(orch_sanitize_job_name "grpo_sp_grasp_vanilla_${GRPO_RUN_NAME_VANILLA}")"
jid_v=$(sbatch --parsable \
  --partition="${ORCH_TRAIN_PARTITION_SPARSE}" \
  --nodes=1 \
  --ntasks=1 \
  --gres="${ORCH_TRAIN_GRES_SPARSE}" \
  --mem=128G \
  --time="${ORCH_TRAIN_TIME_GRPO}" \
  --cpus-per-task=8 \
  --job-name="${jname_v}" \
  --export="${SPARSE_V_EXPORT}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh")
echo "Submitted sparse GRPO (vanilla ABS mask): ${jid_v}"
echo "  tail -f ${REPO_ROOT}/logs/grpo_openr1_${jid_v}.out"

echo ""
echo "=== Submitting sparse GRPO (SNR+ABS mask) → job 2 ==="
SPARSE_S_EXPORT="ALL"
SPARSE_S_EXPORT+=",GRPO_MODE=sparse"
SPARSE_S_EXPORT+=",GRPO_MASK=${OUT_SNR_ABS}"
SPARSE_S_EXPORT+=",GRPO_RUN_NAME=${GRPO_RUN_NAME_SNR}"
SPARSE_S_EXPORT+=",GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS}"
SPARSE_S_EXPORT+=",GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH}"
SPARSE_S_EXPORT+=",GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH}"
SPARSE_S_EXPORT+=",GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}"
SPARSE_S_EXPORT+=",WANDB_MODE=online"
SPARSE_S_EXPORT+=",GRPO_SAVE_STEPS=${GRPO_SAVE_STEPS}"
SPARSE_S_EXPORT+=",GRPO_SAVE_TOTAL_LIMIT=${GRPO_SAVE_TOTAL_LIMIT}"
SPARSE_S_EXPORT+=",GRPO_SPARSE_OUTPUT_BASE=${GRPO_SPARSE_OUTPUT_BASE}"
SPARSE_S_EXPORT+=",MODEL=${MODEL}"
SPARSE_S_EXPORT+=",GRPO_DATASET=${GRPO_DATASET}"
SPARSE_S_EXPORT+=",GRPO_LR=${GRPO_LR}"
SPARSE_S_EXPORT+=",GRPO_BETA=${GRPO_BETA}"
SPARSE_S_EXPORT+=",GRPO_PER_DEVICE_BS=${GRPO_PER_DEVICE_BS}"
SPARSE_S_EXPORT+=",GRPO_GRAD_ACCUM=${GRPO_GRAD_ACCUM}"
SPARSE_S_EXPORT+=",GRPO_NUM_GEN=${GRPO_NUM_GEN}"
SPARSE_S_EXPORT+=",GRPO_GEN_BATCH=${GRPO_GEN_BATCH}"
SPARSE_S_EXPORT+=",GRPO_OPTIM=${GRPO_OPTIM}"
SPARSE_S_EXPORT+=",GRPO_PRECISION=${GRPO_PRECISION}"
SPARSE_S_EXPORT+=",GRPO_WARMUP_RATIO=${GRPO_WARMUP_RATIO}"
SPARSE_S_EXPORT+=",GRPO_MAX_GRAD_NORM=${GRPO_MAX_GRAD_NORM}"
SPARSE_S_EXPORT+=",HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
SPARSE_S_EXPORT+=",RL_CASINO_SCRATCH_ROOT=${RL_CASINO_SCRATCH_ROOT}"

jname_s="$(orch_sanitize_job_name "grpo_sp_grasp_snrabs_${GRPO_RUN_NAME_SNR}")"
jid_s=$(sbatch --parsable \
  --partition="${ORCH_TRAIN_PARTITION_SPARSE}" \
  --nodes=1 \
  --ntasks=1 \
  --gres="${ORCH_TRAIN_GRES_SPARSE}" \
  --mem=128G \
  --time="${ORCH_TRAIN_TIME_GRPO}" \
  --cpus-per-task=8 \
  --job-name="${jname_s}" \
  --export="${SPARSE_S_EXPORT}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh")
echo "Submitted sparse GRPO (SNR+ABS mask): ${jid_s}"
echo "  tail -f ${REPO_ROOT}/logs/grpo_openr1_${jid_s}.out"
