#!/usr/bin/env bash
# One short GPU job that:
#   1) generates masks (shared random baseline + per-task CAV/SNIP for Tulu3, Light-R1, GRPO)
#   2) submits downstream training jobs with Slurm afterok dependencies:
#      Tulu3 DPO (dense + 3 sparse), Light-R1 DPO (dense + 3 sparse), GRPO (dense + 3 sparse)
#
# The random global mask is generated once and reused for DPO (Tulu3) sparse-random, DPO (Light-R1)
# sparse-random, and GRPO sparse-random.
#
# Submit from repo root:
#   cd /path/to/rl_casino && sbatch scripts/orchestrate_masks_then_queue_dpo_grpo.slurm
#
# If masks already exist and you only need to (re)submit the 12 training jobs (no GPU, no afterok):
#   export MASK_RUN_ID=orch_masks_<jobid>
#   bash scripts/orchestrate_queue_training_only.sh
#
# Override any knobs by exporting env vars before sbatch (see "Config" section below).
# Orchestrator-only: ORCH_USE_TRAIN_AUTO_RESUME, ORCH_TRAIN_SOFT_SECONDS, ORCH_TRAIN_* resource overrides.
#
# Long training jobs: optional automatic resume (soft timeout + chained sbatch) is documented in
# scripts/README.md ("Automatic wall-time resume"). For this orchestrator, export ORCH_USE_TRAIN_AUTO_RESUME=1
# (and optionally ORCH_TRAIN_SOFT_SECONDS, ORCH_TRAIN_PARTITION, …) so queued children use
# scripts/orchestrate_training_child_entry.sh → train_with_auto_resume.sh with Slurm resources on the CLI.
# Or submit pipeline_stage_01_dense.sh, pipeline_sparse_one_mask.sh, or grpo_openr1_llama31_slurm.sh
# directly with USE_TRAIN_WITH_AUTO_RESUME=1 as in the README.
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# (Host RAM for the Python process — not GPU VRAM. Mask OOM is typically CUDA OOM, not this limit.)
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

# Training output roots (pipeline_common.sh / grpo_openr1_llama31_slurm.sh). Export before queueing
# children so training_resume_probe in train_with_auto_resume.sh can resolve checkpoint paths.
export TRAIN_OUT_BASE="${TRAIN_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_train}"
export SPARSE_OUT_BASE="${SPARSE_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_sparse_train}"
export GRPO_DENSE_OUTPUT_BASE="${GRPO_DENSE_OUTPUT_BASE:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/dense}"
export GRPO_SPARSE_OUTPUT_BASE="${GRPO_SPARSE_OUTPUT_BASE:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/sparse}"

# Model + datasets
export HF_TOKEN="${HF_TOKEN:-}" # required for gated models (e.g. Llama). If unset, HF may still work if token is cached.
export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

# DPO mask generation config (tulu3 defaults)
export DPO_DS_KEY="${DPO_DS_KEY:-tulu3}"  # dataset registry key for DPO mode
export COLD_DATASET_HF="${COLD_DATASET_HF:-allenai/llama-3.1-tulu-3-8b-preference-mixture}"

# Light-R1 DPO (same model + hyperparameters; separate CAV/SNIP calibration — random mask shared)
export DPO_DS_KEY_LIGHT_R1="${DPO_DS_KEY_LIGHT_R1:-light-r1}"
export COLD_DATASET_HF_LIGHT_R1="${COLD_DATASET_HF_LIGHT_R1:-qihoo360/Light-R1-DPOData}"

# GRPO mask generation config (OpenR1 math defaults)
export GRPO_DATASET_HF="${GRPO_DATASET_HF:-open-r1/OpenR1-Math-220k}"

# Mask knobs
export SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"

# CAV implementation + safety knobs
# - cold_v2_all_params: unified CAV v2 (all params suspect; global selector; coverage gated)
# - legacy: use existing scripts (DPO via cav_cold_mask_finder, GRPO via inference_mask_finder)
export ORCH_CAV_IMPL="${ORCH_CAV_IMPL:-cold_v2_all_params}"
# Opt-in rank-collapse protection (applies to broadcast-based scoring via * |W|).
export ORCH_CAV_WEIGHT_ABS="${ORCH_CAV_WEIGHT_ABS:-0}"
# Also score embedding matrices.
# We default this ON for CAV v2 so the coverage gate includes `model.embed_tokens.weight`
# (aligns with warm-start "all params suspect" masks).
export ORCH_CAV_INCLUDE_EMBEDS="${ORCH_CAV_INCLUDE_EMBEDS:-1}"

# inference_mask_finder batch size (all mask methods that pass --batch_size). Lower if SNIP/CAV OOMs.
# Sequence caps for masks follow DPO_MAX_LENGTH / GRPO_MAX_PROMPT_LENGTH (exported below for downstream;
# mask steps use the same vars in the commands in section "1) Mask generation").
export ORCH_MASK_BATCH_SIZE="${ORCH_MASK_BATCH_SIZE:-4}"

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

########################################
# Oracle checkpoint directory for Light-R1
export ORACLE_FINAL_CKPT_DIR="${ORACLE_FINAL_CKPT_DIR:-/scratch/biggs.s/rl_casino_train/dpo5k_dense_light-r1/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1/checkpoint-500}"

# Oracle checkpoint directory for Tulu3
export ORACLE_FINAL_CKPT_DIR_TULU3="${ORACLE_FINAL_CKPT_DIR_TULU3:-/scratch/biggs.s/rl_casino_train/dpo5k_dense_tulu3/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500}"

# Downstream run ids (stable names recommended for resume)
export DPO_DENSE_RUN_ID="${DPO_DENSE_RUN_ID:-dpo5k_dense_${DPO_DS_KEY}}"
export DPO_SPARSE_RANDOM_RUN_ID="${DPO_SPARSE_RANDOM_RUN_ID:-dpo5k_sparse_random_${DPO_DS_KEY}}"
export DPO_SPARSE_CAV_RUN_ID="${DPO_SPARSE_CAV_RUN_ID:-dpo5k_sparse_cav_${DPO_DS_KEY}}"
export DPO_SPARSE_SNIP_RUN_ID="${DPO_SPARSE_SNIP_RUN_ID:-dpo5k_sparse_snip_${DPO_DS_KEY}}"

export DPO_DENSE_RUN_ID_LIGHT_R1="${DPO_DENSE_RUN_ID_LIGHT_R1:-dpo5k_dense_${DPO_DS_KEY_LIGHT_R1}}"
export DPO_SPARSE_RANDOM_RUN_ID_LIGHT_R1="${DPO_SPARSE_RANDOM_RUN_ID_LIGHT_R1:-dpo5k_sparse_random_${DPO_DS_KEY_LIGHT_R1}}"
export DPO_SPARSE_CAV_RUN_ID_LIGHT_R1="${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1:-dpo5k_sparse_cav_${DPO_DS_KEY_LIGHT_R1}}"
export DPO_SPARSE_SNIP_RUN_ID_LIGHT_R1="${DPO_SPARSE_SNIP_RUN_ID_LIGHT_R1:-dpo5k_sparse_snip_${DPO_DS_KEY_LIGHT_R1}}"

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
export GRPO_SAVE_STEPS="${GRPO_SAVE_STEPS:-50}"
export GRPO_SAVE_TOTAL_LIMIT="${GRPO_SAVE_TOTAL_LIMIT:-3}"

# Queued training jobs: opt-in soft timeout + chained sbatch (see scripts/README.md).
export ORCH_USE_TRAIN_AUTO_RESUME="${ORCH_USE_TRAIN_AUTO_RESUME:-0}"
export MAX_AUTO_RESUME="${MAX_AUTO_RESUME:-8}"
export ORCH_TRAIN_PARTITION="${ORCH_TRAIN_PARTITION:-gpu}"
export ORCH_TRAIN_GRES="${ORCH_TRAIN_GRES:-gpu:h200:1}"
export ORCH_TRAIN_MEM="${ORCH_TRAIN_MEM:-128G}"
export ORCH_TRAIN_CPUS="${ORCH_TRAIN_CPUS:-8}"
export ORCH_TRAIN_TIME_DPO="${ORCH_TRAIN_TIME_DPO:-07:45:00}"
export ORCH_TRAIN_TIME_GRPO="${ORCH_TRAIN_TIME_GRPO:-08:00:00}"
if [ "${ORCH_USE_TRAIN_AUTO_RESUME}" = "1" ] && [ -n "${ORCH_TRAIN_SOFT_SECONDS:-}" ]; then
  export AUTO_RESUME_SOFT_SECONDS="${ORCH_TRAIN_SOFT_SECONDS}"
fi
# When ORCH_USE_TRAIN_AUTO_RESUME=1, downstream sbatch uses orchestrate_training_child_entry.sh with
# ORCH_TRAIN_PARTITION (default gpu), ORCH_TRAIN_GRES (default gpu:h200:1), ORCH_TRAIN_MEM (128G),
# ORCH_TRAIN_TIME_DPO (07:45:00), ORCH_TRAIN_TIME_GRPO (08:00:00), ORCH_TRAIN_CPUS (8 for GRPO jobs).

# W&B: default ON for training jobs.
# Disable explicitly for speed ablations / benchmarks by setting ORCH_DISABLE_WANDB=1.
export ORCH_DISABLE_WANDB="${ORCH_DISABLE_WANDB:-0}"
if [ "${ORCH_DISABLE_WANDB}" = "1" ]; then
  export WANDB_MODE="disabled"
  export WANDB_DISABLED="true"
  export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
  export WANDB_SILENT="${WANDB_SILENT:-true}"
else
  unset WANDB_DISABLED || true
fi
export GRPO_USE_WANDB="${GRPO_USE_WANDB:-1}"

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
echo "ORCH_CAV_IMPL=${ORCH_CAV_IMPL}  ORCH_CAV_WEIGHT_ABS=${ORCH_CAV_WEIGHT_ABS}  ORCH_CAV_INCLUDE_EMBEDS=${ORCH_CAV_INCLUDE_EMBEDS}"

########################################
# 1) Mask generation (GPU)
########################################
# Random baseline: one file reused for DPO (Tulu3), DPO (Light-R1), and GRPO sparse-random.
export RANDOM_MASK_SEED="${RANDOM_MASK_SEED:-42}"

# DPO masks — Tulu3 calibration (scripts/dpo_5k_hpc_copypaste.md)
export MASK_RANDOM="${MASK_DIR}/random_${MODEL//\//_}_sparsity${SPARSITY_PERCENT}pct_seed${RANDOM_MASK_SEED}.pt"
export MASK_WANDA="${MASK_DIR}/wanda_owl_${MODEL_SANITIZED}_tulu3_sparsity${SPARSITY_PERCENT}pct.pt"
export MASK_ORACLE_TULU3="${MASK_DIR}/ground_truth_ckpt_diff_${MODEL_SANITIZED}_tulu3_sparsity${SPARSITY_PERCENT}pct.pt"
export MASK_SNIP="${MASK_DIR}/cold_snip_${MODEL_SANITIZED}_${DPO_DS_KEY}_sparsity${SPARSITY_PERCENT}pct_${DPO_SNIP_OBJECTIVE}.pt"

# DPO masks — Light-R1 (same model/sparsity; Wanda+OWL calibrated on Light-R1)
export MASK_WANDA_LIGHT_R1="${MASK_DIR}/wanda_owl_${MODEL_SANITIZED}_light_r1_sparsity${SPARSITY_PERCENT}pct.pt"
export MASK_SNIP_LIGHT_R1="${MASK_DIR}/cold_snip_${MODEL_SANITIZED}_light_r1_sparsity${SPARSITY_PERCENT}pct_${DPO_SNIP_OBJECTIVE}.pt"
export MASK_ORACLE_LIGHT_R1="${MASK_DIR}/ground_truth_ckpt_diff_${MODEL_SANITIZED}_light_r1_sparsity${SPARSITY_PERCENT}pct.pt"

# GRPO: reuse MASK_RANDOM for sparse-random; Wanda+OWL + SNIP are GRPO-specific paths
export MASK_GRPO_RANDOM="${MASK_RANDOM}"
export MASK_GRPO_WANDA="${GRPO_MASK_DIR}/grpo_wanda_owl_sp${SPARSITY_PERCENT/./}_$(sanitize_model_name "${MODEL}")_${GRPO_DATASET_HF##*/}.pt"
export MASK_GRPO_SNIP="${GRPO_MASK_DIR}/grpo_snip_${GRPO_SNIP_OBJECTIVE}_sp${SPARSITY_PERCENT/./}_$(sanitize_model_name "${MODEL}")_${GRPO_DATASET_HF##*/}.pt"

echo ""
echo "=== Generating shared random mask (DPO + GRPO) → ${MASK_RANDOM}"
"$TRAIN_PY" src/utils/generate_random_mask.py \
  --model_name "${MODEL}" \
  --sparsity_percent "${SPARSITY_PERCENT}" \
  --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
  --seed "${RANDOM_MASK_SEED}" \
  --output_file "${MASK_RANDOM}"

echo ""
echo "=== Generating DPO Wanda+OWL mask (Tulu3) → ${MASK_WANDA}"
"$TRAIN_PY" src/cold_start/wanda_owl_mask_finder.py \
  --model_name "${MODEL}" \
  --method wanda_owl \
  --n_samples "${COLD_CAV_SUBSET}" \
  --max_length "${DPO_MAX_LENGTH}" \
  --batch_size "${ORCH_MASK_BATCH_SIZE}" \
  --sparsity_percent "${SPARSITY_PERCENT}" \
  --output_file "${MASK_WANDA}"

echo ""
echo "=== Generating ORACLE (checkpoint-diff) mask (Tulu3) → ${MASK_ORACLE_TULU3}"
echo "ORACLE_FINAL_CKPT_DIR_TULU3=${ORACLE_FINAL_CKPT_DIR_TULU3}"
"$TRAIN_PY" src/warm_start/checkpoint_diff_mask_finder.py \
  --initial_model "${MODEL}" \
  --final_model "${ORACLE_FINAL_CKPT_DIR_TULU3}" \
  --sparsity_percent "${SPARSITY_PERCENT}" \
  --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
  --output_file "${MASK_ORACLE_TULU3}"

echo ""
echo "=== Generating DPO SNIP mask (Tulu3) → ${MASK_SNIP}"
"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
  --model_name "${MODEL}" \
  --method snip \
  --mode dpo \
  --dataset_name "${DPO_DS_KEY}" \
  --snip-objective "${DPO_SNIP_OBJECTIVE}" \
  --n_samples "${COLD_CAV_SUBSET}" \
  --sparsity "${SPARSITY_PERCENT}" \
  --batch_size "${ORCH_MASK_BATCH_SIZE}" \
  --max_length "${DPO_MAX_LENGTH}" \
  --min-layer-keep-ratio "${MIN_LAYER_KEEP_RATIO}" \
  --snip-num-batches "${COLD_CAV_NUM_BATCHES}" \
  --snip-preference-beta "${DPO_BETA}" \
  --output "${MASK_SNIP}"

# TEMP SCOPE REDUCTION: Light-R1 Wanda+OWL disabled (focusing on Tulu3)
#echo ""
#echo "=== Generating DPO Wanda+OWL mask (Light-R1) → ${MASK_WANDA_LIGHT_R1}"
#"$TRAIN_PY" src/cold_start/wanda_owl_mask_finder.py \
#  --model_name "${MODEL}" \
#  --method wanda_owl \
#  --n_samples "${COLD_CAV_SUBSET}" \
#  --max_length "${DPO_MAX_LENGTH}" \
#  --batch_size "${ORCH_MASK_BATCH_SIZE}" \
#  --sparsity_percent "${SPARSITY_PERCENT}" \
#  --output_file "${MASK_WANDA_LIGHT_R1}"

echo ""
echo "=== Generating DPO SNIP mask (Light-R1) → ${MASK_SNIP_LIGHT_R1}"
# TEMP SCOPE REDUCTION: Light-R1 SNIP disabled
#"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
#  --model_name "${MODEL}" \
#  --method snip \
#  --mode dpo \
#  --dataset_name "${DPO_DS_KEY_LIGHT_R1}" \
#  --snip-objective "${DPO_SNIP_OBJECTIVE}" \
#  --n_samples "${COLD_CAV_SUBSET}" \
#  --sparsity "${SPARSITY_PERCENT}" \
#  --batch_size "${ORCH_MASK_BATCH_SIZE}" \
#  --max_length "${DPO_MAX_LENGTH}" \
#  --min-layer-keep-ratio "${MIN_LAYER_KEEP_RATIO}" \
#  --snip-num-batches "${COLD_CAV_NUM_BATCHES}" \
#  --snip-preference-beta "${DPO_BETA}" \
#  --output "${MASK_SNIP_LIGHT_R1}"

echo ""
#echo "=== Generating ORACLE (checkpoint-diff) mask (Light-R1) → ${MASK_ORACLE_LIGHT_R1}"
#echo "ORACLE_FINAL_CKPT_DIR=${ORACLE_FINAL_CKPT_DIR}"
#"$TRAIN_PY" src/warm_start/checkpoint_diff_mask_finder.py \
#  --initial_model "${MODEL}" \
#  --final_model "${ORACLE_FINAL_CKPT_DIR}" \
#  --sparsity_percent "${SPARSITY_PERCENT}" \
#  --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
#  --output_file "${MASK_ORACLE_LIGHT_R1}"
echo ""
# TEMP SCOPE REDUCTION: GRPO Wanda+OWL mask disabled (focusing on Tulu3; GRPO gets dense + random-sparse only)
#echo "=== Generating GRPO Wanda+OWL mask → ${MASK_GRPO_WANDA}"
#"$TRAIN_PY" src/cold_start/wanda_owl_mask_finder.py \
#  --model_name "${MODEL}" \
#  --method wanda_owl \
#  --n_samples "${GRPO_N_SAMPLES}" \
#  --max_length "${GRPO_MAX_PROMPT_LENGTH}" \
#  --batch_size "${ORCH_MASK_BATCH_SIZE}" \
#  --sparsity_percent "${SPARSITY_PERCENT}" \
#  --output_file "${MASK_GRPO_WANDA}"

echo ""
echo "=== Generating GRPO SNIP mask (${GRPO_SNIP_OBJECTIVE}) → ${MASK_GRPO_SNIP}"
# TEMP SCOPE REDUCTION: GRPO SNIP disabled
#"$TRAIN_PY" src/cold_start/inference_mask_finder.py \
#  --model_name "${MODEL}" \
#  --mode grpo \
#  --method snip \
#  --snip-objective "${GRPO_SNIP_OBJECTIVE}" \
#  --dataset_name "${GRPO_DATASET_HF}" \
#  --n_samples "${GRPO_N_SAMPLES}" \
#  --sparsity "${SPARSITY_PERCENT}" \
#  --max_length "${GRPO_MAX_PROMPT_LENGTH}" \
#  --batch_size "${ORCH_MASK_BATCH_SIZE}" \
#  --seed 42 \
#  --snip-num-batches "${COLD_CAV_NUM_BATCHES}" \
#  --snip-preference-beta "${GRPO_SNIP_PREFERENCE_BETA}" \
#  --output "${MASK_GRPO_SNIP}"

echo ""
echo "=== Mask outputs written ==="
ls -lh "${MASK_DIR}"
echo "(Listing all generated masks in ${MASK_DIR})"

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
echo "ORCH_USE_TRAIN_AUTO_RESUME=${ORCH_USE_TRAIN_AUTO_RESUME:-0}  TRAIN_OUT_BASE=${TRAIN_OUT_BASE}  SPARSE_OUT_BASE=${SPARSE_OUT_BASE}"
echo "GRPO_DENSE_OUTPUT_BASE=${GRPO_DENSE_OUTPUT_BASE}  GRPO_SPARSE_OUTPUT_BASE=${GRPO_SPARSE_OUTPUT_BASE}"

orch_sanitize_job_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  s="$(echo "$s" | tr -cd '[:alnum:]_.-')"
  printf '%.64s' "$s"
}

ORCH_TRAIN_ENTRY="${REPO_ROOT}/scripts/orchestrate_training_child_entry.sh"

_orch_train_bases_export() {
  echo "NUM_STEPS_DPO=${NUM_STEPS_DPO},GRPO_TARGET_STEPS=${GRPO_TARGET_STEPS},TRAIN_OUT_BASE=${TRAIN_OUT_BASE},SPARSE_OUT_BASE=${SPARSE_OUT_BASE},GRPO_DENSE_OUTPUT_BASE=${GRPO_DENSE_OUTPUT_BASE},GRPO_SPARSE_OUTPUT_BASE=${GRPO_SPARSE_OUTPUT_BASE},GRPO_SAVE_STEPS=${GRPO_SAVE_STEPS},GRPO_SAVE_TOTAL_LIMIT=${GRPO_SAVE_TOTAL_LIMIT}"
}

submit_training_job() {
  local label="$1"
  local inner_script="$2"
  local auto_mode="$3"
  local export_extra="$4"
  local job_tag="${5:-$auto_mode}"

  local bases_steps jid
  bases_steps="$(_orch_train_bases_export)"

  if [ "${ORCH_USE_TRAIN_AUTO_RESUME:-0}" != "1" ]; then
    jid=$(sbatch --parsable --dependency="${dep}" \
      --export=ALL,"${export_extra},${bases_steps}" \
      "${inner_script}")
    echo "${label}: ${jid}"
    return
  fi

  local part gres mem time_wall out_log err_log jname full_export
  local -a cpus_args=()
  part="${ORCH_TRAIN_PARTITION:-gpu}"
  gres="${ORCH_TRAIN_GRES:-gpu:h200:1}"
  mem="${ORCH_TRAIN_MEM:-128G}"

  case "${auto_mode}" in
    dense_dpo|sparse_dpo)
      time_wall="${ORCH_TRAIN_TIME_DPO:-07:45:00}"
      ;;
    dense_grpo|sparse_grpo)
      time_wall="${ORCH_TRAIN_TIME_GRPO:-08:00:00}"
      cpus_args=( --cpus-per-task="${ORCH_TRAIN_CPUS:-8}" )
      ;;
    *)
      echo "ERROR: unknown auto-resume mode tag: ${auto_mode}" >&2
      exit 1
      ;;
  esac

  case "${auto_mode}" in
    dense_dpo)
      out_log='logs/pipeline_%j_p1_dense.out'
      err_log='logs/pipeline_%j_p1_dense.err'
      ;;
    sparse_dpo)
      out_log='logs/pipeline_sparse_one_mask_%j.out'
      err_log='logs/pipeline_sparse_one_mask_%j.err'
      ;;
    dense_grpo|sparse_grpo)
      out_log='logs/grpo_openr1_%j.out'
      err_log='logs/grpo_openr1_%j.err'
      ;;
  esac

  full_export="ALL,${export_extra},${bases_steps},ORCH_TRAIN_INNER=${inner_script},AUTO_RESUME_MODE=${auto_mode},ORCH_USE_TRAIN_AUTO_RESUME=1"
  if [ -n "${AUTO_RESUME_SOFT_SECONDS:-}" ]; then
    full_export+=",AUTO_RESUME_SOFT_SECONDS=${AUTO_RESUME_SOFT_SECONDS}"
  fi
  full_export+=",MAX_AUTO_RESUME=${MAX_AUTO_RESUME}"

  jname="$(orch_sanitize_job_name "orch_${job_tag}")"

  jid=$(sbatch --parsable --dependency="${dep}" \
    --partition="${part}" --nodes=1 --ntasks=1 \
    --gres="${gres}" --mem="${mem}" --time="${time_wall}" \
    "${cpus_args[@]}" \
    --job-name="${jname}" \
    --output="${out_log}" --error="${err_log}" \
    --export="${full_export}" \
    "${ORCH_TRAIN_ENTRY}")
  echo "${label}: ${jid}"
}

echo ""
echo "--- DPO dense (Tulu3) ---"
# TEMP SCOPE REDUCTION: all Tulu3 jobs disabled
#submit_training_job "DPO dense (stage 1 only) RUN_ID=${DPO_DENSE_RUN_ID}" \
#  "${REPO_ROOT}/scripts/pipeline_stage_01_dense.sh" \
#  dense_dpo \
#  "PIPELINE_CHAIN_NEXT_STAGE=0,PIPELINE_RUN_ID=${DPO_DENSE_RUN_ID},RUN_ID=${DPO_DENSE_RUN_ID},DPO_DATASET_KEY=tulu3" \
#  "dpo_dense_${DPO_DENSE_RUN_ID}"

echo ""
echo "--- DPO sparse Tulu3 (3 masks; random = shared baseline file) ---"
submit_training_job "DPO sparse Wanda+OWL RUN_ID=${DPO_SPARSE_CAV_RUN_ID}" \
  "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh" \
  sparse_dpo \
  "PIPELINE_RUN_ID=${DPO_SPARSE_CAV_RUN_ID},RUN_ID=${DPO_SPARSE_CAV_RUN_ID},PIPELINE_MASK_FILE=${MASK_WANDA},DPO_DATASET_KEY=tulu3" \
  "dpo_sp_wanda_${DPO_SPARSE_CAV_RUN_ID}"

submit_training_job "DPO sparse ORACLE RUN_ID=${DPO_SPARSE_CAV_RUN_ID}_oracle" \
  "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh" \
  sparse_dpo \
  "PIPELINE_RUN_ID=${DPO_SPARSE_CAV_RUN_ID}_oracle,RUN_ID=${DPO_SPARSE_CAV_RUN_ID}_oracle,PIPELINE_MASK_FILE=${MASK_ORACLE_TULU3},DPO_DATASET_KEY=tulu3" \
  "dpo_sp_oracle_${DPO_SPARSE_CAV_RUN_ID}_oracle"

echo ""
echo "--- DPO dense + sparse (Light-R1, same hyperparameters; shared random mask) ---"
## TEMP SCOPE REDUCTION: only Light-R1 sparse CAV + oracle
#submit_training_job "DPO dense Light-R1 RUN_ID=${DPO_DENSE_RUN_ID_LIGHT_R1}" \
#  "${REPO_ROOT}/scripts/pipeline_stage_01_dense.sh" \
#  dense_dpo \
#  "PIPELINE_CHAIN_NEXT_STAGE=0,PIPELINE_RUN_ID=${DPO_DENSE_RUN_ID_LIGHT_R1},RUN_ID=${DPO_DENSE_RUN_ID_LIGHT_R1},DPO_DATASET_KEY=${DPO_DS_KEY_LIGHT_R1}" \
#  "dpo_dense_lr1_${DPO_DENSE_RUN_ID_LIGHT_R1}"
#
#submit_training_job "DPO sparse random Light-R1 RUN_ID=${DPO_SPARSE_RANDOM_RUN_ID_LIGHT_R1}" \
#  "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh" \
#  sparse_dpo \
#  "PIPELINE_RUN_ID=${DPO_SPARSE_RANDOM_RUN_ID_LIGHT_R1},RUN_ID=${DPO_SPARSE_RANDOM_RUN_ID_LIGHT_R1},PIPELINE_MASK_FILE=${MASK_RANDOM},DPO_DATASET_KEY=${DPO_DS_KEY_LIGHT_R1}" \
#  "dpo_sp_rand_lr1_${DPO_SPARSE_RANDOM_RUN_ID_LIGHT_R1}"
#
# TEMP SCOPE REDUCTION: Light-R1 Wanda+OWL disabled (no mask generated this run)
#submit_training_job "DPO sparse Wanda+OWL Light-R1 RUN_ID=${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1}" \
#  "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh" \
#  sparse_dpo \
#  "PIPELINE_RUN_ID=${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1},RUN_ID=${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1},PIPELINE_MASK_FILE=${MASK_WANDA_LIGHT_R1},DPO_DATASET_KEY=${DPO_DS_KEY_LIGHT_R1}" \
#  "dpo_sp_wanda_lr1_${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1}"

#submit_training_job "DPO sparse ORACLE Light-R1 RUN_ID=${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1}_oracle" \
#  "${REPO_ROOT}/scripts/pipeline_sparse_one_mask.sh" \
#  sparse_dpo \
#  "PIPELINE_RUN_ID=${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1}_oracle,RUN_ID=${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1}_oracle,PIPELINE_MASK_FILE=${MASK_ORACLE_LIGHT_R1},DPO_DATASET_KEY=${DPO_DS_KEY_LIGHT_R1}" \
#  "dpo_sp_oracle_lr1_${DPO_SPARSE_CAV_RUN_ID_LIGHT_R1}_oracle"
echo ""
echo "--- GRPO dense ---"
submit_training_job "GRPO dense slug=${GRPO_DENSE_RUN_SLUG}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh" \
  dense_grpo \
  "GRPO_MODE=dense,GRPO_RUN_SLUG=${GRPO_DENSE_RUN_SLUG},GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH},GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH},GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE},GRPO_USE_WANDB=${GRPO_USE_WANDB}" \
  "grpo_dense_${GRPO_DENSE_RUN_SLUG}"

echo ""
echo "--- GRPO sparse (3 masks) ---"
submit_training_job "GRPO sparse random name=${GRPO_SPARSE_RANDOM_RUN_NAME}" \
  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh" \
  sparse_grpo \
  "GRPO_MODE=sparse,GRPO_MASK=${MASK_GRPO_RANDOM},GRPO_RUN_NAME=${GRPO_SPARSE_RANDOM_RUN_NAME},GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH},GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH},GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE},GRPO_USE_WANDB=${GRPO_USE_WANDB}" \
  "grpo_sp_rand_${GRPO_SPARSE_RANDOM_RUN_NAME}"

# TEMP SCOPE REDUCTION: GRPO Wanda+OWL sparse disabled (no mask generated this run)
#submit_training_job "GRPO sparse Wanda+OWL name=${GRPO_SPARSE_CAV_RUN_NAME}" \
#  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh" \
#  sparse_grpo \
#  "GRPO_MODE=sparse,GRPO_MASK=${MASK_GRPO_WANDA},GRPO_RUN_NAME=${GRPO_SPARSE_CAV_RUN_NAME},GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH},GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH},GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE},GRPO_USE_WANDB=${GRPO_USE_WANDB}" \
#  "grpo_sp_wanda_${GRPO_SPARSE_CAV_RUN_NAME}"
#
#submit_training_job "GRPO sparse SNIP name=${GRPO_SPARSE_SNIP_RUN_NAME}" \
#  "${REPO_ROOT}/scripts/grpo_openr1_llama31_slurm.sh" \
#  sparse_grpo \
#  "GRPO_MODE=sparse,GRPO_MASK=${MASK_GRPO_SNIP},GRPO_RUN_NAME=${GRPO_SPARSE_SNIP_RUN_NAME},GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH},GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH},GRPO_REWARD_PROFILE=${GRPO_REWARD_PROFILE},GRPO_USE_WANDB=${GRPO_USE_WANDB}" \
#  "grpo_sp_snip_${GRPO_SPARSE_SNIP_RUN_NAME}"

echo ""
echo "=== Done. Monitor with: squeue -u ${USER:-<user>} ==="

