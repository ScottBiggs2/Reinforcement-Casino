#!/usr/bin/env bash
# DPO GraSP+Magnitude focused sweep.
#
# For each dataset (Tulu3 + Light-R1):
#   1. Verify dense-run deltas (base_state.pt + deltas_step_<TARGET_STEP>.pt) exist (fail-fast).
#   2. Build a magnitude mask from |Δθ| accumulated up to MAGNITUDE_TARGET_STEP via
#      src/warm_start/even_better_mask_finder.py (--method magnitude --target_step <N>).
#   3. Generate a GraSP element/SNR DPO mask via src/cold_start/inference_mask_finder.py
#      (--method grasp --mode dpo --grasp-objective dpo_preference --score-snr per_weight
#       --score-snr-transform log1p --mask-granularity element).
#   4. Queue two scripts/pipeline_sparse_one_mask.sh sparse-DPO children — one per mask, both
#      at NUM_STEPS_DPO=500.
#
# Submit from repo root:
#   sbatch scripts/orchestrate_dpo_gm_sweep.slurm
#
# This deliberately re-makes GraSP masks rather than relying on the older orch_lr1_grasp6_*
# outputs, which never finished writing (Pass 1 silently died inside torch.autograd.grad due to
# reentrant gradient checkpointing on older Transformers; fixed on cav_fixes commit dbbebbc).
# GRaSP Pass 2 (HVP) can also OOM with checkpointing enabled — see grasp_scorer.py (disable GC for Pass 2).
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=orch_dpo_gm
#SBATCH --output=logs/orch_dpo_gm_%j.out
#SBATCH --error=logs/orch_dpo_gm_%j.err

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
export RL_CASINO_SCRATCH_ROOT="${RL_CASINO_SCRATCH_ROOT:-$SCRATCH_USER_ROOT}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export HF_TOKEN="${HF_TOKEN:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"

export MASK_OUT_BASE="${MASK_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_masks}"
export TRAIN_OUT_BASE="${TRAIN_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_train}"
export SPARSE_OUT_BASE="${SPARSE_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_sparse_train}"

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

# Datasets: registry keys understood by src/utils/dataset_registry.py.
# Order matters only for log readability.
DPO_DS_KEY_TULU3="${DPO_DS_KEY_TULU3:-tulu3}"
DPO_DS_KEY_LIGHT_R1="${DPO_DS_KEY_LIGHT_R1:-light-r1}"
DPO_DS_KEY_TULU3_SANITIZED="${DPO_DS_KEY_TULU3_SANITIZED:-tulu3}"
DPO_DS_KEY_LIGHT_R1_SANITIZED="${DPO_DS_KEY_LIGHT_R1_SANITIZED:-light_r1}"

# Where the existing dense DPO runs live (must match the dense runs the user already has).
TULU3_DENSE_RUN_ID="${TULU3_DENSE_RUN_ID:-dpo5k_dense_tulu3}"
LIGHT_R1_DENSE_RUN_ID="${LIGHT_R1_DENSE_RUN_ID:-dpo5k_dense_light-r1}"

# Sparsity + global pool floor.
export SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"

# Magnitude target step (must already exist as deltas_step_<N>.pt under each dense run).
export MAGNITUDE_TARGET_STEP="${MAGNITUDE_TARGET_STEP:-200}"

# DPO GraSP knobs (mask calibration; uses dpo_preference objective).
export DPO_GRASP_OBJECTIVE="${DPO_GRASP_OBJECTIVE:-dpo_preference}"
export DPO_BETA="${DPO_BETA:-0.1}"
export COLD_CAV_SUBSET="${COLD_CAV_SUBSET:-256}"
export COLD_CAV_NUM_BATCHES="${COLD_CAV_NUM_BATCHES:-32}"
export DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-1024}"
export ORCH_MASK_BATCH_SIZE="${ORCH_MASK_BATCH_SIZE:-1}"

# SNR knobs (per_weight + log1p — the focused setting).
export ORCH_SCORE_SNR_MODE="${ORCH_SCORE_SNR_MODE:-off}"
export ORCH_SCORE_SNR_TRANSFORM="${ORCH_SCORE_SNR_TRANSFORM:-log1p}"
export ORCH_SCORE_SNR_EPS="${ORCH_SCORE_SNR_EPS:-1e-8}"
export ORCH_SCORE_SNR_RAM_BUDGET_GB="${ORCH_SCORE_SNR_RAM_BUDGET_GB:-48}"
export ORCH_SCORE_SNR_ALLOW_LARGE_RAM="${ORCH_SCORE_SNR_ALLOW_LARGE_RAM:-1}"

# Sparse training knobs (per child sbatch via pipeline_sparse_one_mask.sh).
export NUM_STEPS_DPO="${NUM_STEPS_DPO:-500}"
export DPO_LEARNING_RATE="${DPO_LEARNING_RATE:-5e-7}"
export DPO_WARMUP_RATIO="${DPO_WARMUP_RATIO:-0.1}"
export DPO_MAX_PROMPT_LENGTH="${DPO_MAX_PROMPT_LENGTH:-1024}"
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE="${DPO_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export DPO_GRADIENT_ACCUMULATION_STEPS="${DPO_GRADIENT_ACCUMULATION_STEPS:-64}"
export DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-50}"
export DPO_SAVE_TOTAL_LIMIT="${DPO_SAVE_TOTAL_LIMIT:-3}"
export DPO_CHILD_WALL="${DPO_CHILD_WALL:-07:45:00}"
# Mask run id (single dir for this sweep so all 4 masks are colocated).
export MASK_RUN_ID="${MASK_RUN_ID:-orch_dpo_gm_${SLURM_JOB_ID:-manual}}"
export MASK_DIR="${MASK_OUT_BASE}/${MASK_RUN_ID}"
mkdir -p "${MASK_DIR}"

########################################
# Preflight
########################################

if [ ! -x "$TRAIN_PY" ]; then
  echo "ERROR: TRAIN_PY not found/executable: ${TRAIN_PY}" >&2
  exit 1
fi

"$TRAIN_PY" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available; this orchestrator must run on a GPU node.", file=sys.stderr)
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

orch_sanitize_job_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  s="$(echo "$s" | tr -cd '[:alnum:]_.-')"
  printf '%.64s' "$s"
}

MODEL_SANITIZED="$(sanitize_model_name "$MODEL")"

echo "REPO_ROOT=${REPO_ROOT}"
echo "MODEL=${MODEL} (sanitized=${MODEL_SANITIZED})"
echo "MASK_DIR=${MASK_DIR}"
echo "Sparsity=${SPARSITY_PERCENT}%  MAGNITUDE_TARGET_STEP=${MAGNITUDE_TARGET_STEP}"
echo "DPO_GRASP_OBJECTIVE=${DPO_GRASP_OBJECTIVE}  COLD_CAV_SUBSET=${COLD_CAV_SUBSET}  N_BATCHES=${COLD_CAV_NUM_BATCHES}"
echo "SNR mode=${ORCH_SCORE_SNR_MODE} transform=${ORCH_SCORE_SNR_TRANSFORM}"
echo "Sparse children: NUM_STEPS_DPO=${NUM_STEPS_DPO} wall=${DPO_CHILD_WALL}"

########################################
# Helpers
########################################

# Verify the dense run delta dir for a dataset, and echo it.
verify_delta_dir() {
  local ds_run_id="$1"
  local ds_sanitized="$2"
  local delta_dir="${TRAIN_OUT_BASE}/${ds_run_id}/deltas/${MODEL_SANITIZED}_${ds_sanitized}"
  if [ ! -d "${delta_dir}" ]; then
    echo "ERROR: delta dir missing: ${delta_dir}" >&2
    echo "  Magnitude mask requires a dense DPO run with delta logging at step ${MAGNITUDE_TARGET_STEP}." >&2
    return 1
  fi
  for f in "base_state.pt" "deltas_step_${MAGNITUDE_TARGET_STEP}.pt"; do
    if [ ! -f "${delta_dir}/${f}" ]; then
      echo "ERROR: ${delta_dir}/${f} missing — re-run dense DPO with DELTA_LOG_INTERVAL=50 DELTA_LOG_END_STEP=${MAGNITUDE_TARGET_STEP}." >&2
      return 1
    fi
  done
  echo "${delta_dir}"
}

# Build magnitude mask from delta dir; writes to provided output path.
build_magnitude_mask() {
  local delta_dir="$1"
  local out_path="$2"
  local label="$3"

  echo ""
  echo "=== Building magnitude mask (${label}) → ${out_path}"
  set +e
  "$TRAIN_PY" src/warm_start/even_better_mask_finder.py \
    --delta_log_dir "${delta_dir}" \
    --method magnitude \
    --sparsity_percent "${SPARSITY_PERCENT}" \
    --target_step "${MAGNITUDE_TARGET_STEP}" \
    --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
    --output_file "${out_path}"
  local rc=$?
  set -e
  if [ "${rc}" -ne 0 ] || [ ! -f "${out_path}" ]; then
    echo "ERROR: magnitude mask gen failed for ${label} (exit ${rc})" >&2
    return 1
  fi
}

# Build GraSP element/SNR DPO mask; writes to provided output path.
build_grasp_dpo_mask() {
  local ds_key="$1"
  local out_path="$2"
  local label="$3"

  local -a SNR_ARGS=()
  if [ "${ORCH_SCORE_SNR_MODE}" != "off" ]; then
    SNR_ARGS=( --score-snr "${ORCH_SCORE_SNR_MODE}"
               --score-snr-transform "${ORCH_SCORE_SNR_TRANSFORM}"
               --score-snr-eps "${ORCH_SCORE_SNR_EPS}"
               --score-snr-ram-budget-gb "${ORCH_SCORE_SNR_RAM_BUDGET_GB}" )
    if [ "${ORCH_SCORE_SNR_ALLOW_LARGE_RAM:-0}" = "1" ]; then
      SNR_ARGS+=( --score-snr-allow-large-ram )
    fi
  fi

  echo ""
  echo "=== Building GraSP element/SNR mask (${label}) → ${out_path}"
  set +e
  "$TRAIN_PY" src/cold_start/inference_mask_finder.py \
    --model_name "${MODEL}" \
    --method grasp \
    --mode dpo \
    --dataset_name "${ds_key}" \
    --grasp-objective "${DPO_GRASP_OBJECTIVE}" \
    --n_samples "${COLD_CAV_SUBSET}" \
    --sparsity "${SPARSITY_PERCENT}" \
    --batch_size "${ORCH_MASK_BATCH_SIZE}" \
    --max_length "${DPO_MAX_LENGTH}" \
    --min-layer-keep-ratio "${MIN_LAYER_KEEP_RATIO}" \
    --grasp-preference-beta "${DPO_BETA}" \
    --snip-num-batches "${COLD_CAV_NUM_BATCHES}" \
    "${SNR_ARGS[@]}" \
    --mask-granularity element \
    --output "${out_path}"
  local rc=$?
  set -e
  if [ "${rc}" -ne 0 ] || [ ! -f "${out_path}" ]; then
    echo "ERROR: GraSP mask gen failed for ${label} (exit ${rc})" >&2
    return 1
  fi
}

# Submit one sparse DPO child via pipeline_sparse_one_mask.sh.
submit_sparse_dpo_child() {
  local label="$1"
  local mask_path="$2"
  local ds_key="$3"
  local run_id="$4"
  local job_tag="$5"

  if [ ! -f "${mask_path}" ]; then
    echo "WARNING: skipping ${label} submit — mask not on disk: ${mask_path}"
    return 0
  fi

  local jname
  jname="$(orch_sanitize_job_name "dpo_${job_tag}")"

  local export_str="ALL"
  export_str+=",PIPELINE_RUN_ID=${run_id}"
  export_str+=",RUN_ID=${run_id}"
  export_str+=",PIPELINE_MASK_FILE=${mask_path}"
  export_str+=",DPO_DATASET_KEY=${ds_key}"
  export_str+=",NUM_STEPS_DPO=${NUM_STEPS_DPO}"
  export_str+=",DPO_LEARNING_RATE=${DPO_LEARNING_RATE}"
  export_str+=",DPO_WARMUP_RATIO=${DPO_WARMUP_RATIO}"
  export_str+=",DPO_MAX_LENGTH=${DPO_MAX_LENGTH}"
  export_str+=",DPO_MAX_PROMPT_LENGTH=${DPO_MAX_PROMPT_LENGTH}"
  export_str+=",DPO_PER_DEVICE_TRAIN_BATCH_SIZE=${DPO_PER_DEVICE_TRAIN_BATCH_SIZE}"
  export_str+=",DPO_GRADIENT_ACCUMULATION_STEPS=${DPO_GRADIENT_ACCUMULATION_STEPS}"
  export_str+=",DPO_SAVE_STEPS=${DPO_SAVE_STEPS}"
  export_str+=",DPO_SAVE_TOTAL_LIMIT=${DPO_SAVE_TOTAL_LIMIT}"
  export_str+=",DPO_BETA=${DPO_BETA}"
  export_str+=",SPARSE_OUT_BASE=${SPARSE_OUT_BASE}"
  export_str+=",WANDB_MODE=online"

  local jid
  jid=$(sbatch --parsable \
    --time="${DPO_CHILD_WALL}" \
    --job-name="${jname}" \
    --export="${export_str}" \
    "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh")
  echo "${label}: ${jid}"
}

########################################
# 1) Build masks per dataset
########################################

declare -A MASK_MAGNITUDE
declare -A MASK_GRASP

build_masks_for_dataset() {
  local ds_key="$1"
  local ds_sanitized="$2"
  local ds_dense_run_id="$3"

  echo ""
  echo "########################################"
  echo "# Dataset: ${ds_key} (sanitized=${ds_sanitized}, dense=${ds_dense_run_id})"
  echo "########################################"

  local delta_dir
  delta_dir=$(verify_delta_dir "${ds_dense_run_id}" "${ds_sanitized}") || return 1
  echo "Verified delta dir: ${delta_dir}"

  local mag_out
  mag_out="${MASK_DIR}/warm_magnitude_${MODEL_SANITIZED}_${ds_sanitized}_sp${SPARSITY_PERCENT}pct_step${MAGNITUDE_TARGET_STEP}.pt"
  if build_magnitude_mask "${delta_dir}" "${mag_out}" "magnitude/${ds_key}"; then
    MASK_MAGNITUDE["${ds_key}"]="${mag_out}"
  fi

  local grasp_out
  grasp_out="${MASK_DIR}/grasp_${MODEL_SANITIZED}_${ds_sanitized}_sp${SPARSITY_PERCENT}pct_obj${DPO_GRASP_OBJECTIVE}_elem_snr${ORCH_SCORE_SNR_MODE}_${ORCH_SCORE_SNR_TRANSFORM}.pt"
  if build_grasp_dpo_mask "${ds_key}" "${grasp_out}" "grasp/${ds_key}"; then
    MASK_GRASP["${ds_key}"]="${grasp_out}"
  fi
}

# Build masks for both datasets. We swallow per-dataset failures so a Tulu3 issue does not
# block Light-R1 mask gen (and vice versa).
set +e
build_masks_for_dataset "${DPO_DS_KEY_TULU3}"    "${DPO_DS_KEY_TULU3_SANITIZED}"    "${TULU3_DENSE_RUN_ID}"
build_masks_for_dataset "${DPO_DS_KEY_LIGHT_R1}" "${DPO_DS_KEY_LIGHT_R1_SANITIZED}" "${LIGHT_R1_DENSE_RUN_ID}"
set -e

########################################
# 2) Submit sparse DPO children (4 jobs total when all masks land)
########################################

echo ""
echo "########################################"
echo "# Submitting sparse DPO children (NUM_STEPS_DPO=${NUM_STEPS_DPO})"
echo "########################################"

# Per-method run id stems so output dirs are stable for resume / wandb naming.
TULU3_MAG_RUN_ID="${TULU3_MAG_RUN_ID:-dpo${NUM_STEPS_DPO}_sparse_magnitude_tulu3_step${MAGNITUDE_TARGET_STEP}}"
TULU3_GRASP_RUN_ID="${TULU3_GRASP_RUN_ID:-dpo${NUM_STEPS_DPO}_sparse_grasp_elem_snr_tulu3}"
LR1_MAG_RUN_ID="${LR1_MAG_RUN_ID:-dpo${NUM_STEPS_DPO}_sparse_magnitude_lr1_step${MAGNITUDE_TARGET_STEP}}"
LR1_GRASP_RUN_ID="${LR1_GRASP_RUN_ID:-dpo${NUM_STEPS_DPO}_sparse_grasp_elem_snr_lr1}"

if [ -n "${MASK_MAGNITUDE[${DPO_DS_KEY_TULU3}]:-}" ]; then
  submit_sparse_dpo_child "DPO sparse magnitude (Tulu3)" \
    "${MASK_MAGNITUDE[${DPO_DS_KEY_TULU3}]}" \
    "${DPO_DS_KEY_TULU3}" \
    "${TULU3_MAG_RUN_ID}" \
    "tulu3_mag"
fi
if [ -n "${MASK_GRASP[${DPO_DS_KEY_TULU3}]:-}" ]; then
  submit_sparse_dpo_child "DPO sparse GraSP-elem-SNR (Tulu3)" \
    "${MASK_GRASP[${DPO_DS_KEY_TULU3}]}" \
    "${DPO_DS_KEY_TULU3}" \
    "${TULU3_GRASP_RUN_ID}" \
    "tulu3_grasp_elem_snr"
fi
if [ -n "${MASK_MAGNITUDE[${DPO_DS_KEY_LIGHT_R1}]:-}" ]; then
  submit_sparse_dpo_child "DPO sparse magnitude (Light-R1)" \
    "${MASK_MAGNITUDE[${DPO_DS_KEY_LIGHT_R1}]}" \
    "${DPO_DS_KEY_LIGHT_R1}" \
    "${LR1_MAG_RUN_ID}" \
    "lr1_mag"
fi
if [ -n "${MASK_GRASP[${DPO_DS_KEY_LIGHT_R1}]:-}" ]; then
  submit_sparse_dpo_child "DPO sparse GraSP-elem-SNR (Light-R1)" \
    "${MASK_GRASP[${DPO_DS_KEY_LIGHT_R1}]}" \
    "${DPO_DS_KEY_LIGHT_R1}" \
    "${LR1_GRASP_RUN_ID}" \
    "lr1_grasp_elem_snr"
fi

echo ""
echo "=== Done ==="
echo "Mask dir: ${MASK_DIR}"
echo "Sparse out base: ${SPARSE_OUT_BASE}"
ls -lh "${MASK_DIR}" || true
echo "Monitor: squeue -u ${USER:-<user>}"
