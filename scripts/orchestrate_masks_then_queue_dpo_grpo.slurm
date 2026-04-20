#!/usr/bin/env bash
# One short GPU job that:
#   1) generates the required DPO+GRPO mask .pt files
#   2) submits downstream DPO+GRPO training jobs with Slurm afterok dependencies
#
# Submit from repo root:
#   cd /path/to/rl_casino && sbatch scripts/orchestrate_masks_then_queue_dpo_grpo.slurm
#
# Override any knobs by exporting env vars before sbatch (see "Config" section below).
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=orch_masks_queue
#SBATCH --output=logs/orch_masks_queue_%j.out
#SBATCH --error=logs/orch_masks_queue_%j.err

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

# Scratch / env
SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER:-unknown}}"
RL_CASINO_SCRATCH_ROOT="${RL_CASINO_SCRATCH_ROOT:-$SCRATCH_USER_ROOT}"
TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
TRAIN_PY="${TRAIN_ENV}/bin/python"
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${RL_CASINO_SCRATCH_ROOT}/hf_cache/datasets}"

# Model + datasets
export HF_TOKEN="${HF_TOKEN:-}" # required for gated models (e.g. Llama). If unset, HF may still work if token is cached.
export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

# DPO mask generation config (tulu3 defaults)
export DPO_DS_KEY="${DPO_DS_KEY:-tulu3}"  # dataset registry key for DPO mode
export COLD_DATASET_HF="${COLD_DATASET_HF:-allenai/llama-3.1-tulu-3-8b-preference-mixture}"

# GRPO mask generation config (OpenR1 math defaults)
export GRPO_DATASET_HF="${GRPO_DATASET_HF:-open-r1/OpenR1-Math-220k}"

# Mask knobs
export SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"

# Calibration sizes
export COLD_CAV_SUBSET="${COLD_CAV_SUBSET:-256}"
export COLD_CAV_NUM_BATCHES="${COLD_CAV_NUM_BATCHES:-16}"
export GRPO_N_SAMPLES="${GRPO_N_SAMPLES:-64}"

# SNIP objectives
export DPO_SNIP_OBJECTIVE="${DPO_SNIP_OBJECTIVE:-dpo_preference}"
export GRPO_SNIP_OBJECTIVE="${GRPO_SNIP_OBJECTIVE:-lm}"

# DPO preference beta for SNIP (objective=dpo_preference)
export DPO_BETA="${DPO_BETA:-0.1}"
export GRPO_SNIP_PREFERENCE_BETA="${GRPO_SNIP_PREFERENCE_BETA:-1.0}"

# Output locations for masks
export MASK_RUN_ID="${MASK_RUN_ID:-orch_masks_${SLURM_JOB_ID:-manual}}"
export MASK_DIR="${MASK_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_masks}/${MASK_RUN_ID}"
export GRPO_MASK_DIR="${GRPO_MASK_DIR:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/masks}"

# Downstream run ids (stable names recommended for resume)
export DPO_DENSE_RUN_ID="${DPO_DENSE_RUN_ID:-dpo5k_dense_${DPO_DS_KEY}}"
export DPO_SPARSE_RANDOM_RUN_ID="${DPO_SPARSE_RANDOM_RUN_ID:-dpo5k_sparse_random_${DPO_DS_KEY}}"
export DPO_SPARSE_CAV_RUN_ID="${DPO_SPARSE_CAV_RUN_ID:-dpo5k_sparse_cav_${DPO_DS_KEY}}"
export DPO_SPARSE_SNIP_RUN_ID="${DPO_SPARSE_SNIP_RUN_ID:-dpo5k_sparse_snip_${DPO_DS_KEY}}"

export GRPO_DENSE_RUN_SLUG="${GRPO_DENSE_RUN_SLUG:-llama31_${GRPO_DATASET_HF##*/}_grpo_dense_v1}"
export GRPO_SPARSE_RANDOM_RUN_NAME="${GRPO_SPARSE_RANDOM_RUN_NAME:-grpo_sparse_random_sp975_v1}"
export GRPO_SPARSE_CAV_RUN_NAME="${GRPO_SPARSE_CAV_RUN_NAME:-grpo_sparse_cav_sp975_v1}"
export GRPO_SPARSE_SNIP_RUN_NAME="${GRPO_SPARSE_SNIP_RUN_NAME:-grpo_sparse_snip_${GRPO_SNIP_OBJECTIVE}_sp975_v1}"

# Downstream training knobs (DPO doc defaults)
export NUM_STEPS_DPO="${NUM_STEPS_DPO:-5000}"
export DPO_LEARNING_RATE="${DPO_LEARNING_RATE:-5e-7}"
export DPO_WARMUP_RATIO="${DPO_WARMUP_RATIO:-0.1}"
export DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-1024}"
export DPO_MAX_PROMPT_LENGTH="${DPO_MAX_PROMPT_LENGTH:-1024}"
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE="${DPO_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export DPO_GRADIENT_ACCUMULATION_STEPS="${DPO_GRADIENT_ACCUMULATION_STEPS:-64}"
export DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-50}"
export DPO_SAVE_TOTAL_LIMIT="${DPO_SAVE_TOTAL_LIMIT:-3}"

# Downstream training knobs (GRPO doc defaults)
export GRPO_DATASET="${GRPO_DATASET:-math-220k}"
export GRPO_TARGET_STEPS="${GRPO_TARGET_STEPS:-5000}"
export GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-2048}"
export GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-512}"
export GRPO_REWARD_PROFILE="${GRPO_REWARD_PROFILE:-llama_cot}"

# W&B off by default for orchestrated cluster jobs unless you explicitly opt in
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export GRPO_USE_WANDB="${GRPO_USE_WANDB:-0}"

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
    print("ERROR: CUDA is not available in this job; masks require a GPU allocation.", file=sys.stderr)
    sys.exit(1)
print(f"CUDA OK: device_count={torch.cuda.device_count()} device0={torch.cuda.get_device_name(0)!r}")
PY

sanitize_model_name() {
  # Must match the repo’s sanitize conventions (see inference_mask_finder.py / pipeline_common.sh).
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

mkdir -p "$MASK_DIR" "$GRPO_MASK_DIR"
echo "REPO_ROOT=${REPO_ROOT}"
echo "TRAIN_PY=${TRAIN_PY}"
echo "SCRATCH_USER_ROOT=${SCRATCH_USER_ROOT}"
echo "MASK_DIR=${MASK_DIR}"
echo "GRPO_MASK_DIR=${GRPO_MASK_DIR}"
echo "MODEL=${MODEL} (sanitized=${MODEL_SANITIZED})"
echo "DPO_DS_KEY=${DPO_DS_KEY}  COLD_DATASET_HF=${COLD_DATASET_HF}"
echo "GRPO_DATASET_HF=${GRPO_DATASET_HF}  GRPO_DATASET=${GRPO_DATASET}"
echo "SPARSITY_PERCENT=${SPARSITY_PERCENT}  MIN_LAYER_KEEP_RATIO=${MIN_LAYER_KEEP_RATIO}"
echo "DPO_SNIP_OBJECTIVE=${DPO_SNIP_OBJECTIVE}  GRPO_SNIP_OBJECTIVE=${GRPO_SNIP_OBJECTIVE}"

########################################
# 1) Mask generation (GPU)
########################################

# DPO masks (three variants expected by scripts/dpo_5k_hpc_copypaste.md)
export MASK_RANDOM="${MASK_DIR}/random_${MODEL//\//_}_sparsity${SPARSITY_PERCENT}pct.pt"
export MASK_CAV="${MASK_DIR}/cold_cav_${MODEL_SANITIZED}_sparsity${SPARSITY_PERCENT}pct.pt"
export MASK_SNIP="${MASK_DIR}/cold_snip_${MODEL_SANITIZED}_${DPO_DS_KEY}_sparsity${SPARSITY_PERCENT}pct_${DPO_SNIP_OBJECTIVE}.pt"

echo ""
echo "=== Generating DPO random mask → ${MASK_RANDOM}"
"$TRAIN_PY" src/utils/generate_random_mask.py \
  --model_name "${MODEL}" \
  --sparsity_percent "${SPARSITY_PERCENT}" \
  --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
  --output_file "${MASK_RANDOM}"

echo ""
echo "=== Generating DPO CAV cold mask → ${MASK_CAV}"
"$TRAIN_PY" src/cold_start/cav_cold_mask_finder.py \
  --model_name "${MODEL}" \
  --dataset_name "${COLD_DATASET_HF}" \
  --method cav \
  --sparsity_percent "${SPARSITY_PERCENT}" \
  --subset_size "${COLD_CAV_SUBSET}" \
  --num_batches "${COLD_CAV_NUM_BATCHES}" \
  --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
  --output_file "${MASK_CAV}"

echo ""
echo "=== Generating DPO SNIP mask → ${MASK_SNIP}"
"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
  --model_name "${MODEL}" \
  --method snip \
  --mode dpo \
  --dataset_name "${DPO_DS_KEY}" \
  --snip-objective "${DPO_SNIP_OBJECTIVE}" \
  --n_samples "${COLD_CAV_SUBSET}" \
  --sparsity "${SPARSITY_PERCENT}" \
  --batch_size 4 \
  --max_length "${DPO_MAX_LENGTH}" \
  --min-layer-keep-ratio "${MIN_LAYER_KEEP_RATIO}" \
  --snip-num-batches "${COLD_CAV_NUM_BATCHES}" \
  --snip-preference-beta "${DPO_BETA}" \
  --output "${MASK_SNIP}"

# GRPO masks (three variants expected by docs/GRPO_HPC_COPYPASTE.md)
export MASK_GRPO_RANDOM="${GRPO_MASK_DIR}/grpo_random_sp${SPARSITY_PERCENT/./}_$(sanitize_model_name "${MODEL}")_${GRPO_DATASET_HF##*/}_seed42.pt"
export MASK_GRPO_CAV="${GRPO_MASK_DIR}/grpo_cav_sp${SPARSITY_PERCENT/./}_$(sanitize_model_name "${MODEL}")_${GRPO_DATASET_HF##*/}.pt"
export MASK_GRPO_SNIP="${GRPO_MASK_DIR}/grpo_snip_${GRPO_SNIP_OBJECTIVE}_sp${SPARSITY_PERCENT/./}_$(sanitize_model_name "${MODEL}")_${GRPO_DATASET_HF##*/}.pt"

echo ""
echo "=== Generating GRPO random mask → ${MASK_GRPO_RANDOM}"
"$TRAIN_PY" src/utils/generate_random_mask.py \
  --model_name "${MODEL}" \
  --sparsity_percent "${SPARSITY_PERCENT}" \
  --seed 42 \
  --output_file "${MASK_GRPO_RANDOM}"

echo ""
echo "=== Generating GRPO CAV mask → ${MASK_GRPO_CAV}"
"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
  --model_name "${MODEL}" \
  --mode grpo \
  --method cav \
  --dataset_name "${GRPO_DATASET_HF}" \
  --n_samples "${GRPO_N_SAMPLES}" \
  --sparsity "${SPARSITY_PERCENT}" \
  --max_length "${GRPO_MAX_PROMPT_LENGTH}" \
  --batch_size 4 \
  --seed 42 \
  --output "${MASK_GRPO_CAV}"

echo ""
echo "=== Generating GRPO SNIP mask (${GRPO_SNIP_OBJECTIVE}) → ${MASK_GRPO_SNIP}"
"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
  --model_name "${MODEL}" \
  --mode grpo \
  --method snip \
  --snip-objective "${GRPO_SNIP_OBJECTIVE}" \
  --dataset_name "${GRPO_DATASET_HF}" \
  --n_samples "${GRPO_N_SAMPLES}" \
  --sparsity "${SPARSITY_PERCENT}" \
  --max_length "${GRPO_MAX_PROMPT_LENGTH}" \
  --batch_size 4 \
  --seed 42 \
  --snip-num-batches "${COLD_CAV_NUM_BATCHES}" \
  --snip-preference-beta "${GRPO_SNIP_PREFERENCE_BETA}" \
  --output "${MASK_GRPO_SNIP}"

echo ""
echo "=== Mask outputs written ==="
ls -lh "${MASK_RANDOM}" "${MASK_CAV}" "${MASK_SNIP}" "${MASK_GRPO_RANDOM}" "${MASK_GRPO_CAV}" "${MASK_GRPO_SNIP}"

########################################
# 2) Queue downstream jobs (afterok)
########################################

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: expected SLURM_JOB_ID to be set (this script must be run via sbatch)." >&2
  exit 1
fi

dep="afterok:${SLURM_JOB_ID}"
echo ""
echo "=== Submitting downstream jobs with dependency: ${dep}"

submit_job() {
  local label="$1"
  shift
  local jid
  jid=$(sbatch --parsable --dependency="${dep}" "$@")
  echo "${label}: ${jid}"
}

echo ""
echo "--- DPO dense ---"
submit_job "DPO dense (stage 1 only) RUN_ID=${DPO_DENSE_RUN_ID}" \
  --export=ALL,PIPELINE_CHAIN_NEXT_STAGE=0,PIPELINE_RUN_ID="${DPO_DENSE_RUN_ID}",RUN_ID="${DPO_DENSE_RUN_ID}" \
  "${REPO_ROOT}/scripts/pipeline_stage_01_dense.sh"

echo ""
echo "--- DPO sparse (3 masks) ---"
submit_job "DPO sparse random RUN_ID=${DPO_SPARSE_RANDOM_RUN_ID}" \
  --export=ALL,PIPELINE_RUN_ID="${DPO_SPARSE_RANDOM_RUN_ID}",RUN_ID="${DPO_SPARSE_RANDOM_RUN_ID}",PIPELINE_MASK_FILE="${MASK_RANDOM}" \
  "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh"

submit_job "DPO sparse CAV RUN_ID=${DPO_SPARSE_CAV_RUN_ID}" \
  --export=ALL,PIPELINE_RUN_ID="${DPO_SPARSE_CAV_RUN_ID}",RUN_ID="${DPO_SPARSE_CAV_RUN_ID}",PIPELINE_MASK_FILE="${MASK_CAV}" \
  "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh"

submit_job "DPO sparse SNIP RUN_ID=${DPO_SPARSE_SNIP_RUN_ID}" \
  --export=ALL,PIPELINE_RUN_ID="${DPO_SPARSE_SNIP_RUN_ID}",RUN_ID="${DPO_SPARSE_SNIP_RUN_ID}",PIPELINE_MASK_FILE="${MASK_SNIP}" \
  "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh"

echo ""
echo "--- GRPO dense ---"
submit_job "GRPO dense slug=${GRPO_DENSE_RUN_SLUG}" \
  --export=ALL,GRPO_MODE=dense,GRPO_RUN_SLUG="${GRPO_DENSE_RUN_SLUG}",GRPO_TARGET_STEPS="${GRPO_TARGET_STEPS}",GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH}",GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH}",GRPO_REWARD_PROFILE="${GRPO_REWARD_PROFILE}",GRPO_USE_WANDB="${GRPO_USE_WANDB}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh"

echo ""
echo "--- GRPO sparse (3 masks) ---"
submit_job "GRPO sparse random name=${GRPO_SPARSE_RANDOM_RUN_NAME}" \
  --export=ALL,GRPO_MODE=sparse,GRPO_MASK="${MASK_GRPO_RANDOM}",GRPO_RUN_NAME="${GRPO_SPARSE_RANDOM_RUN_NAME}",GRPO_TARGET_STEPS="${GRPO_TARGET_STEPS}",GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH}",GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH}",GRPO_REWARD_PROFILE="${GRPO_REWARD_PROFILE}",GRPO_USE_WANDB="${GRPO_USE_WANDB}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh"

submit_job "GRPO sparse CAV name=${GRPO_SPARSE_CAV_RUN_NAME}" \
  --export=ALL,GRPO_MODE=sparse,GRPO_MASK="${MASK_GRPO_CAV}",GRPO_RUN_NAME="${GRPO_SPARSE_CAV_RUN_NAME}",GRPO_TARGET_STEPS="${GRPO_TARGET_STEPS}",GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH}",GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH}",GRPO_REWARD_PROFILE="${GRPO_REWARD_PROFILE}",GRPO_USE_WANDB="${GRPO_USE_WANDB}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh"

submit_job "GRPO sparse SNIP name=${GRPO_SPARSE_SNIP_RUN_NAME}" \
  --export=ALL,GRPO_MODE=sparse,GRPO_MASK="${MASK_GRPO_SNIP}",GRPO_RUN_NAME="${GRPO_SPARSE_SNIP_RUN_NAME}",GRPO_TARGET_STEPS="${GRPO_TARGET_STEPS}",GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH}",GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH}",GRPO_REWARD_PROFILE="${GRPO_REWARD_PROFILE}",GRPO_USE_WANDB="${GRPO_USE_WANDB}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh"

echo ""
echo "=== Done. Monitor with: squeue -u ${USER:-<user>} ==="

