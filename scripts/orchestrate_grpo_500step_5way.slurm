#!/usr/bin/env bash
# One-shot GRPO 5-way 500-step sweep:
#   1. Dense GRPO (with delta logging through step 200) — also serves as magnitude source.
#   2. Random sparse GRPO at 97.5% sparsity, element-wise.
#   3. GraSP-ABS sparse GRPO at 97.5% sparsity, element-wise (no SNR).
#   4. GraSP-ABS+SNR sparse GRPO at 97.5% sparsity, element-wise (per_weight log1p SNR).
#   5. Magnitude (warm-start, target_step=200) sparse GRPO — submitted by the magnitude poller.
#
# Hyperparameter contract: docs/hyperparams/grpo_500step_5way_sweep.yaml
# Single source of truth for runtime defaults: scripts/grpo_training_env_defaults.sh
#
# Training partition defaults to multigpu (shorter queue); still single gpu:h200:1 per job.
# All 5 training children use identical hyperparameters.
#
# Submit from repo root:
#   sbatch --export=ALL,HF_TOKEN="$HF_TOKEN" scripts/orchestrate_grpo_500step_5way.slurm
#
# Optional overrides (set in the submitting shell BEFORE sbatch):
#   GRPO_TARGET_STEPS, GRPO_DELTA_LOG_END_STEP, GRPO_MAGNITUDE_TARGET_STEP,
#   SPARSITY_PERCENT, MIN_LAYER_KEEP_RATIO, GRPO_N_SAMPLES, RANDOM_SEED, MODEL,
#   GRPO_DATASET, GRPO_DATASET_HF, GRPO_MASK_DIR, GRPO_DENSE_OUTPUT_BASE,
#   GRPO_SPARSE_OUTPUT_BASE, ORCH_MASK_BATCH_SIZE,
#   ORCH_TRAIN_PARTITION (default multigpu), ORCH_TRAIN_GRES (default gpu:h200:1).
#
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=grpo_5way
#SBATCH --output=logs/orch_grpo_500step_5way_%j.out
#SBATCH --error=logs/orch_grpo_500step_5way_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

########################################
# Environment + scratch layout
########################################

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
export RL_CASINO_SCRATCH_ROOT="${RL_CASINO_SCRATCH_ROOT:-$SCRATCH_USER_ROOT}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export HF_TOKEN="${HF_TOKEN:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"

# GRPO scratch layout (must agree with grpo_openr1_llama31_slurm.sh defaults).
export GRPO_MASK_DIR="${GRPO_MASK_DIR:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/masks}"
export GRPO_DENSE_OUTPUT_BASE="${GRPO_DENSE_OUTPUT_BASE:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/dense}"
export GRPO_SPARSE_OUTPUT_BASE="${GRPO_SPARSE_OUTPUT_BASE:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/sparse}"
mkdir -p "${GRPO_MASK_DIR}"

# Single source of truth for training hyperparameters.
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/grpo_training_env_defaults.sh"

########################################
# Sweep-level locks (override before sbatch if you must, but defaults are the contract)
########################################

# Sweep target steps.
export GRPO_TARGET_STEPS="${GRPO_TARGET_STEPS:-500}"

# Mask construction.
export SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
export RANDOM_SEED="${RANDOM_SEED:-42}"

# GraSP calibration.
export GRPO_DATASET_HF="${GRPO_DATASET_HF:-open-r1/OpenR1-Math-220k}"
export GRPO_GRASP_OBJECTIVE="${GRPO_GRASP_OBJECTIVE:-lm}"
export GRPO_GRASP_PREFERENCE_BETA="${GRPO_GRASP_PREFERENCE_BETA:-1.0}"
export GRPO_N_SAMPLES="${GRPO_N_SAMPLES:-256}"
export ORCH_MASK_BATCH_SIZE="${ORCH_MASK_BATCH_SIZE:-1}"

# SNR variant knobs (only the GraSP-ABS+SNR mask uses these; vanilla GraSP-ABS keeps SNR off).
export ORCH_SCORE_SNR_TRANSFORM="${ORCH_SCORE_SNR_TRANSFORM:-log1p}"
export ORCH_SCORE_SNR_EPS="${ORCH_SCORE_SNR_EPS:-1e-8}"
export ORCH_SCORE_SNR_RAM_BUDGET_GB="${ORCH_SCORE_SNR_RAM_BUDGET_GB:-96}"
export ORCH_SCORE_SNR_ALLOW_LARGE_RAM="${ORCH_SCORE_SNR_ALLOW_LARGE_RAM:-1}"

# Delta logging schedule (drives the magnitude mask). Magnitude target step MUST <= dense end step.
export GRPO_DELTA_LOG_INTERVAL="${GRPO_DELTA_LOG_INTERVAL:-50}"
export GRPO_DELTA_LOG_END_STEP="${GRPO_DELTA_LOG_END_STEP:-200}"
export GRPO_MAGNITUDE_TARGET_STEP="${GRPO_MAGNITUDE_TARGET_STEP:-200}"

# Training children allocation (uniform: gpu:h200:1 for ALL 5 training jobs).
ORCH_TRAIN_PARTITION="${ORCH_TRAIN_PARTITION:-multigpu}"
ORCH_TRAIN_GRES="${ORCH_TRAIN_GRES:-gpu:h200:1}"
ORCH_TRAIN_TIME="${ORCH_TRAIN_TIME:-08:00:00}"
ORCH_TRAIN_MEM="${ORCH_TRAIN_MEM:-128G}"
ORCH_TRAIN_CPUS="${ORCH_TRAIN_CPUS:-8}"

# Magnitude poller allocation (CPU; can be re-submitted standalone if it dies).
ORCH_POLL_PARTITION="${ORCH_POLL_PARTITION:-short}"
ORCH_POLL_TIME="${ORCH_POLL_TIME:-08:00:00}"
ORCH_POLL_MEM="${ORCH_POLL_MEM:-64G}"
ORCH_POLL_CPUS="${ORCH_POLL_CPUS:-4}"

# Identifiers tied to this sweep submission.
RUN_SUFFIX="${RUN_SUFFIX:-${SLURM_JOB_ID:-manual}}"
GRPO_DENSE_RUN_SLUG="${GRPO_DENSE_RUN_SLUG:-grpo_dense_openr1_steps${GRPO_TARGET_STEPS}_with_deltas_${RUN_SUFFIX}}"
GRPO_RUN_NAME_RANDOM="${GRPO_RUN_NAME_RANDOM:-grpo_sparse_random_sp${SPARSITY_PERCENT/./_}_steps${GRPO_TARGET_STEPS}_${RUN_SUFFIX}}"
GRPO_RUN_NAME_GRASP_ABS="${GRPO_RUN_NAME_GRASP_ABS:-grpo_sparse_grasp_abs_${GRPO_GRASP_OBJECTIVE}_steps${GRPO_TARGET_STEPS}_${RUN_SUFFIX}}"
GRPO_RUN_NAME_GRASP_ABS_SNR="${GRPO_RUN_NAME_GRASP_ABS_SNR:-grpo_sparse_grasp_abs_snr_${ORCH_SCORE_SNR_TRANSFORM}_${GRPO_GRASP_OBJECTIVE}_steps${GRPO_TARGET_STEPS}_${RUN_SUFFIX}}"
GRPO_RUN_NAME_MAGNITUDE="${GRPO_RUN_NAME_MAGNITUDE:-grpo_sparse_magnitude_step${GRPO_MAGNITUDE_TARGET_STEP}_steps${GRPO_TARGET_STEPS}_${RUN_SUFFIX}}"

# Mask file naming (deterministic; lives under a per-sweep subdir to avoid collisions across reruns).
GRPO_MASK_SUBDIR="${GRPO_MASK_SUBDIR:-5way_${RUN_SUFFIX}}"
MASK_SUBDIR_FULL="${GRPO_MASK_DIR}/${GRPO_MASK_SUBDIR}"
mkdir -p "${MASK_SUBDIR_FULL}"

########################################
# Helpers
########################################

if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found/executable: ${TRAIN_PY}" >&2
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

orch_sanitize_job_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  s="$(echo "$s" | tr -cd '[:alnum:]_.-')"
  printf '%.64s' "$s"
}

MODEL_SANITIZED="$(sanitize_model_name "$MODEL")"
DS_TAIL="${GRPO_DATASET_HF##*/}"
SP_TAG="$(echo "${SPARSITY_PERCENT}" | tr '.' '_')"

# Mask paths.
MASK_RANDOM="${MASK_SUBDIR_FULL}/grpo_random_elem_sp${SP_TAG}_${MODEL_SANITIZED}_${DS_TAIL}_seed${RANDOM_SEED}.pt"
MASK_GRASP_ABS="${MASK_SUBDIR_FULL}/grpo_grasp_abs_${GRPO_GRASP_OBJECTIVE}_elem_sp${SP_TAG}_${MODEL_SANITIZED}_${DS_TAIL}_snroff.pt"
MASK_GRASP_ABS_SNR="${MASK_SUBDIR_FULL}/grpo_grasp_abs_${GRPO_GRASP_OBJECTIVE}_elem_sp${SP_TAG}_${MODEL_SANITIZED}_${DS_TAIL}_snrperweight_${ORCH_SCORE_SNR_TRANSFORM}.pt"

# Dense run dir (deltas land here; magnitude poller reads from here).
GRPO_DENSE_RUN_DIR="${GRPO_DENSE_OUTPUT_BASE}/${GRPO_DENSE_RUN_SLUG}"
GRPO_DENSE_DELTA_DIR="${GRPO_DENSE_RUN_DIR}/deltas"

# Magnitude mask path written by the poller.
MASK_MAGNITUDE="${MASK_SUBDIR_FULL}/grpo_magnitude_${MODEL_SANITIZED}_${DS_TAIL}_sp${SP_TAG}_step${GRPO_MAGNITUDE_TARGET_STEP}.pt"

########################################
# Preflight
########################################

echo "================================================================="
echo " GRPO 5-way 500-step sweep — orchestrator preflight"
echo "================================================================="
echo "REPO_ROOT          = ${REPO_ROOT}"
echo "TRAIN_PY           = ${TRAIN_PY}"
echo "MODEL              = ${MODEL} (sanitized=${MODEL_SANITIZED})"
echo "GRPO_DATASET       = ${GRPO_DATASET}  (HF: ${GRPO_DATASET_HF})"
echo "GRPO_TARGET_STEPS  = ${GRPO_TARGET_STEPS}"
echo "Sparsity           = ${SPARSITY_PERCENT}%  min_layer_keep_ratio=${MIN_LAYER_KEEP_RATIO}"
echo "GraSP n_samples    = ${GRPO_N_SAMPLES}  batch_size=${ORCH_MASK_BATCH_SIZE}  rank_dir=abs"
echo "Delta logging      = every ${GRPO_DELTA_LOG_INTERVAL} steps through step ${GRPO_DELTA_LOG_END_STEP}"
echo "Magnitude target   = step ${GRPO_MAGNITUDE_TARGET_STEP}"
echo "Mask dir           = ${MASK_SUBDIR_FULL}"
echo "Dense run slug     = ${GRPO_DENSE_RUN_SLUG}"
echo "Dense run dir      = ${GRPO_DENSE_RUN_DIR}"
echo "Hyperparam locks   :"
echo "  LR=${GRPO_LR} beta=${GRPO_BETA} per_dev_bs=${GRPO_PER_DEVICE_BS} grad_accum=${GRPO_GRAD_ACCUM}"
echo "  num_gen=${GRPO_NUM_GEN} gen_bs=${GRPO_GEN_BATCH} optim=${GRPO_OPTIM} prec=${GRPO_PRECISION}"
echo "  prompt_len=${GRPO_MAX_PROMPT_LENGTH} completion_len=${GRPO_MAX_COMPLETION_LENGTH}"
echo "  reward=${GRPO_REWARD_PROFILE} warmup_ratio=${GRPO_WARMUP_RATIO} max_grad_norm=${GRPO_MAX_GRAD_NORM}"
echo "Training children  : ${ORCH_TRAIN_PARTITION} ${ORCH_TRAIN_GRES} ${ORCH_TRAIN_TIME} mem=${ORCH_TRAIN_MEM}"
echo "Magnitude poller   : ${ORCH_POLL_PARTITION} CPU ${ORCH_POLL_TIME} mem=${ORCH_POLL_MEM}"
echo "================================================================="

"$TRAIN_PY" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available; this orchestrator must run on a GPU node.", file=sys.stderr)
    sys.exit(1)
print(f"CUDA OK: device_count={torch.cuda.device_count()} device0={torch.cuda.get_device_name(0)!r}")
PY

########################################
# Build masks (random, GraSP-ABS, GraSP-ABS+SNR) on the parent GPU
########################################

echo ""
echo "================================================================="
echo " [1/3] Generating random elementwise mask"
echo " → ${MASK_RANDOM}"
echo "================================================================="
"$TRAIN_PY" src/utils/generate_random_mask.py \
  --model_name "${MODEL}" \
  --sparsity_percent "${SPARSITY_PERCENT}" \
  --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
  --seed "${RANDOM_SEED}" \
  --mask-granularity element \
  --output_file "${MASK_RANDOM}"

if [ ! -f "${MASK_RANDOM}" ]; then
  echo "ERROR: random mask not written: ${MASK_RANDOM}" >&2
  exit 1
fi
ls -la "${MASK_RANDOM}"

GRASP_COMMON_ARGS=(
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
echo "================================================================="
echo " [2/3] Generating GraSP-ABS mask (SNR off)"
echo " → ${MASK_GRASP_ABS}"
echo "================================================================="
"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
  "${GRASP_COMMON_ARGS[@]}" \
  --output "${MASK_GRASP_ABS}"

if [ ! -f "${MASK_GRASP_ABS}" ]; then
  echo "ERROR: GraSP-ABS mask not written: ${MASK_GRASP_ABS}" >&2
  exit 1
fi
ls -la "${MASK_GRASP_ABS}"

echo ""
echo "================================================================="
echo " [3/3] Generating GraSP-ABS + per-weight SNR (${ORCH_SCORE_SNR_TRANSFORM}) mask"
echo " → ${MASK_GRASP_ABS_SNR}"
echo "================================================================="
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
  "${GRASP_COMMON_ARGS[@]}" \
  "${SNR_ARGS[@]}" \
  --output "${MASK_GRASP_ABS_SNR}"

if [ ! -f "${MASK_GRASP_ABS_SNR}" ]; then
  echo "ERROR: GraSP-ABS+SNR mask not written: ${MASK_GRASP_ABS_SNR}" >&2
  exit 1
fi
ls -la "${MASK_GRASP_ABS_SNR}"

echo ""
echo "All 3 immediately-buildable masks present. Proceeding to job submission."

########################################
# Single helper: build the consolidated --export string for a sparse training child.
# Every sparse child uses the same locked hyperparameter set (sourced from
# scripts/grpo_training_env_defaults.sh) so the comparison is hermetic.
# Args: $1 = mask path, $2 = run name.
########################################

build_sparse_export() {
  local mask_path="$1"
  local run_name="$2"
  local exp="ALL"
  exp+=",GRPO_MODE=sparse"
  exp+=",GRPO_MASK=${mask_path}"
  exp+=",GRPO_RUN_NAME=${run_name}"
  exp+=",GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS}"
  exp+=",MODEL=${MODEL}"
  exp+=",GRPO_DATASET=${GRPO_DATASET}"
  exp+=",GRPO_LR=${GRPO_LR}"
  exp+=",GRPO_BETA=${GRPO_BETA}"
  exp+=",GRPO_PER_DEVICE_BS=${GRPO_PER_DEVICE_BS}"
  exp+=",GRPO_GRAD_ACCUM=${GRPO_GRAD_ACCUM}"
  exp+=",GRPO_NUM_GEN=${GRPO_NUM_GEN}"
  exp+=",GRPO_GEN_BATCH=${GRPO_GEN_BATCH}"
  exp+=",GRPO_OPTIM=${GRPO_OPTIM}"
  exp+=",GRPO_PRECISION=${GRPO_PRECISION}"
  exp+=",GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH}"
  exp+=",GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH}"
  exp+=",GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}"
  exp+=",GRPO_WARMUP_RATIO=${GRPO_WARMUP_RATIO}"
  exp+=",GRPO_MAX_GRAD_NORM=${GRPO_MAX_GRAD_NORM}"
  exp+=",GRPO_SAVE_STEPS=${GRPO_SAVE_STEPS}"
  exp+=",GRPO_SAVE_TOTAL_LIMIT=${GRPO_SAVE_TOTAL_LIMIT}"
  exp+=",GRPO_SPARSE_OUTPUT_BASE=${GRPO_SPARSE_OUTPUT_BASE}"
  exp+=",HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
  exp+=",RL_CASINO_SCRATCH_ROOT=${RL_CASINO_SCRATCH_ROOT}"
  exp+=",WANDB_MODE=online"
  printf '%s' "${exp}"
}

submit_sparse_train() {
  # Args: $1=mask path, $2=run name, $3=tag for job name
  local mask_path="$1"
  local run_name="$2"
  local tag="$3"
  local export_str
  export_str="$(build_sparse_export "${mask_path}" "${run_name}")"
  local jname
  jname="$(orch_sanitize_job_name "grpo_sp_${tag}_${run_name}")"
  local jid
  jid=$(sbatch --parsable \
    --partition="${ORCH_TRAIN_PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --gres="${ORCH_TRAIN_GRES}" \
    --mem="${ORCH_TRAIN_MEM}" \
    --time="${ORCH_TRAIN_TIME}" \
    --cpus-per-task="${ORCH_TRAIN_CPUS}" \
    --job-name="${jname}" \
    --export="${export_str}" \
    "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh")
  printf '%s' "${jid}"
}

########################################
# 1) Submit the dense GRPO child (with delta logging through step 200).
########################################

echo ""
echo "================================================================="
echo " Submitting [1/5] dense GRPO (${GRPO_TARGET_STEPS} steps; deltas every ${GRPO_DELTA_LOG_INTERVAL} thru step ${GRPO_DELTA_LOG_END_STEP})"
echo "================================================================="
DENSE_EXPORT="ALL"
DENSE_EXPORT+=",GRPO_MODE=dense"
DENSE_EXPORT+=",GRPO_RUN_SLUG=${GRPO_DENSE_RUN_SLUG}"
DENSE_EXPORT+=",GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS}"
DENSE_EXPORT+=",GRPO_DELTA_LOG_INTERVAL=${GRPO_DELTA_LOG_INTERVAL}"
DENSE_EXPORT+=",GRPO_DELTA_LOG_END_STEP=${GRPO_DELTA_LOG_END_STEP}"
DENSE_EXPORT+=",MODEL=${MODEL}"
DENSE_EXPORT+=",GRPO_DATASET=${GRPO_DATASET}"
DENSE_EXPORT+=",GRPO_LR=${GRPO_LR}"
DENSE_EXPORT+=",GRPO_BETA=${GRPO_BETA}"
DENSE_EXPORT+=",GRPO_PER_DEVICE_BS=${GRPO_PER_DEVICE_BS}"
DENSE_EXPORT+=",GRPO_GRAD_ACCUM=${GRPO_GRAD_ACCUM}"
DENSE_EXPORT+=",GRPO_NUM_GEN=${GRPO_NUM_GEN}"
DENSE_EXPORT+=",GRPO_GEN_BATCH=${GRPO_GEN_BATCH}"
DENSE_EXPORT+=",GRPO_OPTIM=${GRPO_OPTIM}"
DENSE_EXPORT+=",GRPO_PRECISION=${GRPO_PRECISION}"
DENSE_EXPORT+=",GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH}"
DENSE_EXPORT+=",GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH}"
DENSE_EXPORT+=",GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}"
DENSE_EXPORT+=",GRPO_WARMUP_RATIO=${GRPO_WARMUP_RATIO}"
DENSE_EXPORT+=",GRPO_MAX_GRAD_NORM=${GRPO_MAX_GRAD_NORM}"
DENSE_EXPORT+=",GRPO_SAVE_STEPS=${GRPO_SAVE_STEPS}"
DENSE_EXPORT+=",GRPO_SAVE_TOTAL_LIMIT=${GRPO_SAVE_TOTAL_LIMIT}"
DENSE_EXPORT+=",GRPO_DENSE_OUTPUT_BASE=${GRPO_DENSE_OUTPUT_BASE}"
DENSE_EXPORT+=",HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
DENSE_EXPORT+=",RL_CASINO_SCRATCH_ROOT=${RL_CASINO_SCRATCH_ROOT}"
DENSE_EXPORT+=",WANDB_MODE=online"

DENSE_JNAME="$(orch_sanitize_job_name "grpo_dense_${GRPO_DENSE_RUN_SLUG}")"
DENSE_JID=$(sbatch --parsable \
  --partition="${ORCH_TRAIN_PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --gres="${ORCH_TRAIN_GRES}" \
  --mem="${ORCH_TRAIN_MEM}" \
  --time="${ORCH_TRAIN_TIME}" \
  --cpus-per-task="${ORCH_TRAIN_CPUS}" \
  --job-name="${DENSE_JNAME}" \
  --export="${DENSE_EXPORT}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh")
echo "  dense GRPO job id: ${DENSE_JID}"

########################################
# 2,3,4) Submit sparse GRPO children (random, GraSP-ABS, GraSP-ABS+SNR).
########################################

echo ""
echo "================================================================="
echo " Submitting [2/5] sparse GRPO — random mask"
echo "================================================================="
RANDOM_JID=$(submit_sparse_train "${MASK_RANDOM}" "${GRPO_RUN_NAME_RANDOM}" "random")
echo "  sparse random job id: ${RANDOM_JID}"

echo ""
echo "================================================================="
echo " Submitting [3/5] sparse GRPO — GraSP-ABS (SNR off)"
echo "================================================================="
GRASP_ABS_JID=$(submit_sparse_train "${MASK_GRASP_ABS}" "${GRPO_RUN_NAME_GRASP_ABS}" "graspabs")
echo "  sparse GraSP-ABS job id: ${GRASP_ABS_JID}"

echo ""
echo "================================================================="
echo " Submitting [4/5] sparse GRPO — GraSP-ABS + per-weight SNR (${ORCH_SCORE_SNR_TRANSFORM})"
echo "================================================================="
GRASP_ABS_SNR_JID=$(submit_sparse_train "${MASK_GRASP_ABS_SNR}" "${GRPO_RUN_NAME_GRASP_ABS_SNR}" "graspabssnr")
echo "  sparse GraSP-ABS+SNR job id: ${GRASP_ABS_SNR_JID}"

########################################
# 5) Submit the magnitude poller. It waits for deltas_step_${GRPO_MAGNITUDE_TARGET_STEP}.pt,
#    then builds the magnitude mask and sbatches a sparse GRPO child with the same locked
#    hyperparameters (the poller itself sources grpo_training_env_defaults.sh and forwards
#    them explicitly).
########################################

echo ""
echo "================================================================="
echo " Submitting [5/5] magnitude poller (waits for ${GRPO_DENSE_DELTA_DIR}/deltas_step_${GRPO_MAGNITUDE_TARGET_STEP}.pt)"
echo "================================================================="
POLL_EXPORT="ALL"
POLL_EXPORT+=",MODEL=${MODEL}"
POLL_EXPORT+=",GRPO_DATASET=${GRPO_DATASET}"
POLL_EXPORT+=",GRPO_DATASET_HF=${GRPO_DATASET_HF}"
POLL_EXPORT+=",GRPO_DENSE_DELTA_DIR=${GRPO_DENSE_DELTA_DIR}"
POLL_EXPORT+=",GRPO_DENSE_OUTPUT_BASE=${GRPO_DENSE_OUTPUT_BASE}"
POLL_EXPORT+=",GRPO_DENSE_RUN_SLUG=${GRPO_DENSE_RUN_SLUG}"
POLL_EXPORT+=",GRPO_SPARSE_OUTPUT_BASE=${GRPO_SPARSE_OUTPUT_BASE}"
POLL_EXPORT+=",GRPO_MASK_DIR=${GRPO_MASK_DIR}"
POLL_EXPORT+=",MASK_GRPO_MAGNITUDE_STEP200=${MASK_MAGNITUDE}"
POLL_EXPORT+=",GRPO_MAGNITUDE_TARGET_STEP=${GRPO_MAGNITUDE_TARGET_STEP}"
POLL_EXPORT+=",GRPO_SPARSE_MAGNITUDE_RUN_NAME=${GRPO_RUN_NAME_MAGNITUDE}"
POLL_EXPORT+=",GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS}"
POLL_EXPORT+=",GRPO_LR=${GRPO_LR}"
POLL_EXPORT+=",GRPO_BETA=${GRPO_BETA}"
POLL_EXPORT+=",GRPO_PER_DEVICE_BS=${GRPO_PER_DEVICE_BS}"
POLL_EXPORT+=",GRPO_GRAD_ACCUM=${GRPO_GRAD_ACCUM}"
POLL_EXPORT+=",GRPO_NUM_GEN=${GRPO_NUM_GEN}"
POLL_EXPORT+=",GRPO_GEN_BATCH=${GRPO_GEN_BATCH}"
POLL_EXPORT+=",GRPO_OPTIM=${GRPO_OPTIM}"
POLL_EXPORT+=",GRPO_PRECISION=${GRPO_PRECISION}"
POLL_EXPORT+=",GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH}"
POLL_EXPORT+=",GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH}"
POLL_EXPORT+=",GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE}"
POLL_EXPORT+=",GRPO_WARMUP_RATIO=${GRPO_WARMUP_RATIO}"
POLL_EXPORT+=",GRPO_MAX_GRAD_NORM=${GRPO_MAX_GRAD_NORM}"
POLL_EXPORT+=",GRPO_SAVE_STEPS=${GRPO_SAVE_STEPS}"
POLL_EXPORT+=",GRPO_SAVE_TOTAL_LIMIT=${GRPO_SAVE_TOTAL_LIMIT}"
POLL_EXPORT+=",SPARSITY_PERCENT=${SPARSITY_PERCENT}"
POLL_EXPORT+=",MIN_LAYER_KEEP_RATIO=${MIN_LAYER_KEEP_RATIO}"
POLL_EXPORT+=",HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
POLL_EXPORT+=",RL_CASINO_SCRATCH_ROOT=${RL_CASINO_SCRATCH_ROOT}"
POLL_EXPORT+=",ORCH_TRAIN_PARTITION=${ORCH_TRAIN_PARTITION}"
POLL_EXPORT+=",ORCH_TRAIN_GRES=${ORCH_TRAIN_GRES}"
POLL_EXPORT+=",ORCH_TRAIN_TIME=${ORCH_TRAIN_TIME}"
POLL_EXPORT+=",ORCH_TRAIN_MEM=${ORCH_TRAIN_MEM}"
POLL_EXPORT+=",ORCH_TRAIN_CPUS=${ORCH_TRAIN_CPUS}"

POLL_JID=$(sbatch --parsable \
  --partition="${ORCH_POLL_PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${ORCH_POLL_CPUS}" \
  --mem="${ORCH_POLL_MEM}" \
  --time="${ORCH_POLL_TIME}" \
  --job-name="grpo_mag_poll_${RUN_SUFFIX}" \
  --export="${POLL_EXPORT}" \
  "${REPO_ROOT}/scripts/grpo_magnitude_after_step200.slurm")
echo "  magnitude poller job id: ${POLL_JID}"

########################################
# Summary
########################################

echo ""
echo "================================================================="
echo " GRPO 5-way 500-step sweep — submitted"
echo "================================================================="
echo "Hyperparam manifest : docs/hyperparams/grpo_500step_5way_sweep.yaml"
echo ""
echo "Mask paths:"
echo "  random           : ${MASK_RANDOM}"
echo "  GraSP-ABS        : ${MASK_GRASP_ABS}"
echo "  GraSP-ABS+SNR    : ${MASK_GRASP_ABS_SNR}"
echo "  magnitude (TBD)  : ${MASK_MAGNITUDE}    (poller will write this after dense reaches step ${GRPO_MAGNITUDE_TARGET_STEP})"
echo ""
echo "Run dirs:"
echo "  dense run dir    : ${GRPO_DENSE_RUN_DIR}"
echo "  dense delta dir  : ${GRPO_DENSE_DELTA_DIR}"
echo "  sparse base      : ${GRPO_SPARSE_OUTPUT_BASE}"
echo ""
echo "Slurm job ids:"
echo "  [1/5] dense              : ${DENSE_JID}     (${GRPO_DENSE_RUN_SLUG})"
echo "  [2/5] sparse random      : ${RANDOM_JID}    (${GRPO_RUN_NAME_RANDOM})"
echo "  [3/5] sparse GraSP-ABS   : ${GRASP_ABS_JID} (${GRPO_RUN_NAME_GRASP_ABS})"
echo "  [4/5] sparse GraSP-ABS+SNR: ${GRASP_ABS_SNR_JID} (${GRPO_RUN_NAME_GRASP_ABS_SNR})"
echo "  [5/5] magnitude poller   : ${POLL_JID}     (will sbatch a sparse child once deltas exist)"
echo ""
echo "Monitoring:"
echo "  squeue -u ${USER:-<user>}"
echo "  tail -f ${REPO_ROOT}/logs/grpo_openr1_${DENSE_JID}.out"
echo "  tail -f ${REPO_ROOT}/logs/grpo_openr1_${RANDOM_JID}.out"
echo "  tail -f ${REPO_ROOT}/logs/grpo_openr1_${GRASP_ABS_JID}.out"
echo "  tail -f ${REPO_ROOT}/logs/grpo_openr1_${GRASP_ABS_SNR_JID}.out"
echo "  tail -f ${REPO_ROOT}/logs/grpo_mag_poll_${POLL_JID}.out"
echo ""
echo "To resubmit just this orchestrator (e.g. after editing defaults):"
echo "  cd ${REPO_ROOT}"
echo "  sbatch --export=ALL,HF_TOKEN=\"\$HF_TOKEN\" scripts/orchestrate_grpo_500step_5way.slurm"
echo "================================================================="
