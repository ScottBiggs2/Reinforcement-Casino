#!/usr/bin/env bash
# One-off orchestrator: Light-R1 DPO sparse sweep (6 masks):
#   - GRaSP: element/base
#   - GRaSP: element/SNR
#   - GRaSP: block/base
#   - GRaSP: block/SNR
#   - Random: element/base
#   - Random: block/base
#
# Purpose: accelerate queueing for this specific GRaSP focus run:
#   - NUM_STEPS_DPO=200
#   - child walltime forced to 04:00:00 (does not change global defaults)
#
# Submit from repo root:
#   sbatch scripts/orchestrate_light_r1_grasp_sweep.slurm
#
# Override knobs by exporting env vars before sbatch (see "Config" section).
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=orch_lr1_grasp6
#SBATCH --output=logs/orch_lr1_grasp6_%j.out
#SBATCH --error=logs/orch_lr1_grasp6_%j.err

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
# Config (override before sbatch)
########################################

SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"

# Output bases (dense runs live elsewhere; this sweep only touches masks + sparse outputs)
export MASK_OUT_BASE="${MASK_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_masks}"
export SPARSE_OUT_BASE="${SPARSE_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_sparse_train}"

# Model + dataset
export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export DPO_DS_KEY_LIGHT_R1="${DPO_DS_KEY_LIGHT_R1:-light-r1}"

# Sparsity + floors
export SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
export RANDOM_MASK_SEED="${RANDOM_MASK_SEED:-42}"

# Fast training for this sweep only
export NUM_STEPS_DPO="${NUM_STEPS_DPO:-200}"
export DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-25}"
export DPO_SAVE_TOTAL_LIMIT="${DPO_SAVE_TOTAL_LIMIT:-2}"

# Mask generation calibration / objectives
export COLD_CAV_SUBSET="${COLD_CAV_SUBSET:-512}"
export COLD_CAV_NUM_BATCHES="${COLD_CAV_NUM_BATCHES:-32}"
export ORCH_MASK_BATCH_SIZE="${ORCH_MASK_BATCH_SIZE:-4}"
export DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-1024}"
export DPO_BETA="${DPO_BETA:-0.1}"
export GRASP_OBJECTIVE="${GRASP_OBJECTIVE:-dpo_preference}"

# Block-mask knobs
export ORCH_MASK_BLOCK_SIZE="${ORCH_MASK_BLOCK_SIZE:-256}"
export ORCH_MASK_BLOCK_REDUCTION="${ORCH_MASK_BLOCK_REDUCTION:-mean}" # mean|max

# SNR knobs (only used for the SNR variants)
export ORCH_SCORE_SNR_MODE="${ORCH_SCORE_SNR_MODE:-per_weight}" # per_tensor|per_weight
export ORCH_SCORE_SNR_TRANSFORM="${ORCH_SCORE_SNR_TRANSFORM:-log1p}"
export ORCH_SCORE_SNR_EPS="${ORCH_SCORE_SNR_EPS:-1e-8}"
export ORCH_SCORE_SNR_RAM_BUDGET_GB="${ORCH_SCORE_SNR_RAM_BUDGET_GB:-48}"
export ORCH_SCORE_SNR_ALLOW_LARGE_RAM="${ORCH_SCORE_SNR_ALLOW_LARGE_RAM:-1}"

# Random masks: disable by default for relaunches (random already queued successfully).
# Set ORCH_DO_RANDOM_MASKS=1 to re-enable.
export ORCH_DO_RANDOM_MASKS="${ORCH_DO_RANDOM_MASKS:-0}"

# W&B: keep enabled by default; these names come from mask filename + RUN_ID in pipeline_common.sh
export ORCH_DISABLE_WANDB="${ORCH_DISABLE_WANDB:-0}"
if [ "${ORCH_DISABLE_WANDB}" = "1" ]; then
  export WANDB_MODE="disabled"
  export WANDB_DISABLED="true"
  export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
  export WANDB_SILENT="${WANDB_SILENT:-true}"
else
  unset WANDB_DISABLED || true
fi

########################################
# Derived paths + preflight
########################################

if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found/executable: ${TRAIN_PY}" >&2
  exit 1
fi

"$TRAIN_PY" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available in this job; masks require a GPU allocation.", file=sys.stderr)
    sys.exit(1)
print(f"CUDA OK: device_count={torch.cuda.device_count()} device0={torch.cuda.get_device_name(0)!r}")
PY

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

export MASK_RUN_ID="${MASK_RUN_ID:-orch_lr1_grasp6_${SLURM_JOB_ID:-manual}}"
export MASK_DIR="${MASK_OUT_BASE}/${MASK_RUN_ID}"
mkdir -p "$MASK_DIR"

# Make RUN_ID stable and descriptive: this controls sparse output dir + W&B run_name suffix.
export RUN_ID="${RUN_ID:-dpo200_sparse_lr1_grasp6_${MASK_RUN_ID}}"
export PIPELINE_RUN_ID="${PIPELINE_RUN_ID:-${RUN_ID}}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "TRAIN_PY=${TRAIN_PY}"
echo "SCRATCH_USER_ROOT=${SCRATCH_USER_ROOT}"
echo "MASK_RUN_ID=${MASK_RUN_ID}"
echo "MASK_DIR=${MASK_DIR}"
echo "MODEL=${MODEL} (sanitized=${MODEL_SANITIZED})"
echo "DPO_DS_KEY_LIGHT_R1=${DPO_DS_KEY_LIGHT_R1}"
echo "SPARSITY_PERCENT=${SPARSITY_PERCENT} MIN_LAYER_KEEP_RATIO=${MIN_LAYER_KEEP_RATIO}"
echo "NUM_STEPS_DPO=${NUM_STEPS_DPO} (forced for this sweep)"
echo "Child walltime override: 04:00:00"
echo "COLD_CAV_SUBSET=${COLD_CAV_SUBSET} COLD_CAV_NUM_BATCHES=${COLD_CAV_NUM_BATCHES} ORCH_MASK_BATCH_SIZE=${ORCH_MASK_BATCH_SIZE}"
echo "Block: ORCH_MASK_BLOCK_SIZE=${ORCH_MASK_BLOCK_SIZE} ORCH_MASK_BLOCK_REDUCTION=${ORCH_MASK_BLOCK_REDUCTION}"
echo "SNR: ORCH_SCORE_SNR_MODE=${ORCH_SCORE_SNR_MODE} ORCH_SCORE_SNR_TRANSFORM=${ORCH_SCORE_SNR_TRANSFORM}"
echo "SNR RAM: ORCH_SCORE_SNR_RAM_BUDGET_GB=${ORCH_SCORE_SNR_RAM_BUDGET_GB} ORCH_SCORE_SNR_ALLOW_LARGE_RAM=${ORCH_SCORE_SNR_ALLOW_LARGE_RAM}"
echo "ORCH_DO_RANDOM_MASKS=${ORCH_DO_RANDOM_MASKS}"

########################################
# Submission helper
########################################

orch_sanitize_job_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  s="$(echo "$s" | tr -cd '[:alnum:]_.-')"
  printf '%.64s' "$s"
}

submit_sparse_dpo_one_mask() {
  local label="$1"
  local mask_path="$2"
  local job_tag="$3"

  if [ ! -f "$mask_path" ]; then
    echo "WARNING: missing mask for ${label}: ${mask_path} (skipping submit)"
    return 0
  fi

  local jname
  jname="$(orch_sanitize_job_name "lr1_${job_tag}")"

  # Force 4h wall for this sweep (overrides the #SBATCH time in pipeline_sparse_one_mask.sh).
  local jid
  jid=$(sbatch --parsable \
    --time="04:00:00" \
    --job-name="${jname}" \
    --export=ALL,"PIPELINE_RUN_ID=${PIPELINE_RUN_ID},RUN_ID=${RUN_ID},PIPELINE_MASK_FILE=${mask_path},DPO_DATASET_KEY=${DPO_DS_KEY_LIGHT_R1},NUM_STEPS_DPO=${NUM_STEPS_DPO},DPO_SAVE_STEPS=${DPO_SAVE_STEPS},DPO_SAVE_TOTAL_LIMIT=${DPO_SAVE_TOTAL_LIMIT}" \
    "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh")
  echo "${label}: ${jid}"
}

########################################
# 1) Random masks (element + block)
########################################

MASK_RANDOM_ELEM="${MASK_DIR}/random_elem_${MODEL//\//_}_light_r1_sp${SPARSITY_PERCENT}pct_seed${RANDOM_MASK_SEED}.pt"
MASK_RANDOM_BLOCK="${MASK_DIR}/random_block${ORCH_MASK_BLOCK_SIZE}_${ORCH_MASK_BLOCK_REDUCTION}_${MODEL//\//_}_light_r1_sp${SPARSITY_PERCENT}pct_seed${RANDOM_MASK_SEED}.pt"

if [ "${ORCH_DO_RANDOM_MASKS}" = "1" ]; then
  echo ""
  echo "=== Generating RANDOM element mask → ${MASK_RANDOM_ELEM}"
  set +e
  "$TRAIN_PY" src/utils/generate_random_mask.py \
    --model_name "${MODEL}" \
    --sparsity_percent "${SPARSITY_PERCENT}" \
    --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
    --seed "${RANDOM_MASK_SEED}" \
    --mask-granularity element \
    --output_file "${MASK_RANDOM_ELEM}"
  set -e

  echo ""
  echo "=== Generating RANDOM block mask → ${MASK_RANDOM_BLOCK}"
  set +e
  "$TRAIN_PY" src/utils/generate_random_mask.py \
    --model_name "${MODEL}" \
    --sparsity_percent "${SPARSITY_PERCENT}" \
    --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
    --seed "${RANDOM_MASK_SEED}" \
    --mask-granularity block \
    --mask-block-size "${ORCH_MASK_BLOCK_SIZE}" \
    --mask-block-reduction "${ORCH_MASK_BLOCK_REDUCTION}" \
    --output_file "${MASK_RANDOM_BLOCK}"
  set -e
else
  echo ""
  echo "=== Skipping random mask generation/submission (ORCH_DO_RANDOM_MASKS=0) ==="
fi

########################################
# 2) GRaSP masks (element/base, element/SNR, block/base, block/SNR)
########################################

GRASP_BASE_PREFIX="grasp_${MODEL_SANITIZED}_light_r1_sp${SPARSITY_PERCENT}pct_obj${GRASP_OBJECTIVE}"
MASK_GRASP_ELEM_BASE="${MASK_DIR}/${GRASP_BASE_PREFIX}_elem_base.pt"
MASK_GRASP_ELEM_SNR="${MASK_DIR}/${GRASP_BASE_PREFIX}_elem_snr${ORCH_SCORE_SNR_MODE}_${ORCH_SCORE_SNR_TRANSFORM}.pt"
MASK_GRASP_BLOCK_BASE="${MASK_DIR}/${GRASP_BASE_PREFIX}_block${ORCH_MASK_BLOCK_SIZE}_${ORCH_MASK_BLOCK_REDUCTION}_base.pt"
MASK_GRASP_BLOCK_SNR="${MASK_DIR}/${GRASP_BASE_PREFIX}_block${ORCH_MASK_BLOCK_SIZE}_${ORCH_MASK_BLOCK_REDUCTION}_snr${ORCH_SCORE_SNR_MODE}_${ORCH_SCORE_SNR_TRANSFORM}.pt"

gen_grasp_mask() {
  local label="$1"
  local out_path="$2"
  local granularity="$3" # element|block
  local use_snr="$4"     # 0|1

  local -a SNRS=()
  if [ "$use_snr" = "1" ]; then
    SNRS+=( --score-snr "${ORCH_SCORE_SNR_MODE}" )
    SNRS+=( --score-snr-transform "${ORCH_SCORE_SNR_TRANSFORM}" )
    SNRS+=( --score-snr-eps "${ORCH_SCORE_SNR_EPS}" )
    SNRS+=( --score-snr-ram-budget-gb "${ORCH_SCORE_SNR_RAM_BUDGET_GB}" )
    if [ "${ORCH_SCORE_SNR_ALLOW_LARGE_RAM}" = "1" ]; then
      SNRS+=( --score-snr-allow-large-ram )
    fi
  fi

  local -a MG=()
  if [ "$granularity" != "element" ]; then
    MG+=( --mask-granularity "$granularity" )
    MG+=( --mask-block-size "${ORCH_MASK_BLOCK_SIZE}" )
    MG+=( --mask-block-reduction "${ORCH_MASK_BLOCK_REDUCTION}" )
  fi

  echo ""
  echo "=== Generating ${label} → ${out_path}"
  set +e
  "$TRAIN_PY" src/cold_start/inference_mask_finder.py \
    --model_name "${MODEL}" \
    --method grasp \
    --mode dpo \
    --dataset_name "${DPO_DS_KEY_LIGHT_R1}" \
    --grasp-objective "${GRASP_OBJECTIVE}" \
    --n_samples "${COLD_CAV_SUBSET}" \
    --sparsity "${SPARSITY_PERCENT}" \
    --batch_size 1 \
    --max_length "${DPO_MAX_LENGTH}" \
    --min-layer-keep-ratio "${MIN_LAYER_KEEP_RATIO}" \
    --grasp-preference-beta "${DPO_BETA}" \
    --snip-num-batches "${COLD_CAV_NUM_BATCHES}" \
    "${SNRS[@]}" \
    "${MG[@]}" \
    --output "${out_path}"
  set -e
}

gen_grasp_mask "GRaSP element/base" "${MASK_GRASP_ELEM_BASE}" "element" "0"
gen_grasp_mask "GRaSP element/SNR"  "${MASK_GRASP_ELEM_SNR}"  "element" "1"
gen_grasp_mask "GRaSP block/base"   "${MASK_GRASP_BLOCK_BASE}" "block" "0"
gen_grasp_mask "GRaSP block/SNR"    "${MASK_GRASP_BLOCK_SNR}"  "block" "1"

########################################
# 3) Queue sparse DPO children (Light-R1 only)
########################################

echo ""
echo "=== Submitting 6 Light-R1 sparse DPO jobs (NUM_STEPS_DPO=${NUM_STEPS_DPO}) ==="

if [ "${ORCH_DO_RANDOM_MASKS}" = "1" ]; then
  submit_sparse_dpo_one_mask "DPO sparse random element (Light-R1)" "${MASK_RANDOM_ELEM}"  "rand_elem"
  submit_sparse_dpo_one_mask "DPO sparse random block (Light-R1)"   "${MASK_RANDOM_BLOCK}" "rand_block"
else
  echo "Skipping random sparse submits (ORCH_DO_RANDOM_MASKS=0)"
fi
submit_sparse_dpo_one_mask "DPO sparse GRaSP elem/base (Light-R1)" "${MASK_GRASP_ELEM_BASE}" "grasp_elem_base"
submit_sparse_dpo_one_mask "DPO sparse GRaSP elem/SNR (Light-R1)"  "${MASK_GRASP_ELEM_SNR}"  "grasp_elem_snr"
submit_sparse_dpo_one_mask "DPO sparse GRaSP block/base (Light-R1)" "${MASK_GRASP_BLOCK_BASE}" "grasp_block_base"
submit_sparse_dpo_one_mask "DPO sparse GRaSP block/SNR (Light-R1)"  "${MASK_GRASP_BLOCK_SNR}"  "grasp_block_snr"

echo ""
echo "=== Done ==="
echo "Mask dir: ${MASK_DIR}"
echo "Sparse out base: ${SPARSE_OUT_BASE} (RUN_ID=${RUN_ID})"
echo "Monitor: squeue -u ${USER:-<user>}"
