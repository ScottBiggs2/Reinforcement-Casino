#!/bin/bash
# Diagnostic H1: warm magnitude mask (float32 delta files) vs bf16 checkpoint-diff oracle.
#
# Hypothesis 1: the checkpoint_diff oracle uses bf16-quantized weights, which may
# lose signal for Qwen3's small DPO changes.  The delta files saved by
# FlexibleCheckpointCallback store exact float32 differences computed during
# training BEFORE any bf16 re-quantisation.
#
# This job:
#   Phase 1 — generate a magnitude mask from the step-250 float32 delta files
#   Phase 2 — run 500-step sparse DPO with that mask
#
# Compare the resulting loss curve against the checkpoint-diff oracle run.
# If this mask shows markedly better learning, bf16 quantisation is the culprit.
#
# Required env (set before sbatch or via export):
#   DENSE_RUN_BASE  — base dir of the dense DPO run that produced the delta files
#                     e.g. /scratch/biggs.s/rl_casino_train/20260601_175443_7366081
#
# Submit from repo root:
#   export DENSE_RUN_BASE=/scratch/biggs.s/rl_casino_train/20260601_175443_7366081
#   sbatch scripts/diag_h1_warm_mask_qwen3_8b.slurm
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=07:45:00
#SBATCH --job-name=diag_h1_warm_q38b
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/diag_h1_warm_mask_qwen3_%j.out
#SBATCH --error=logs/diag_h1_warm_mask_qwen3_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_SCRIPT_HOME}/.." && pwd)"
fi
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
export MODEL="${MODEL:-Qwen/Qwen3-8B}"
export DPO_DATASET_KEY="${DPO_DATASET_KEY:-light-r1}"
export NUM_STEPS_DPO="${NUM_STEPS_DPO:-500}"
export SPARSITY_PCT="${SPARSITY_PCT:-97.5}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"

# Path to the dense DPO run that produced the delta files.
# The deltas live under: ${DENSE_RUN_BASE}/deltas/qwen_qwen3_8b_light_r1/
export DENSE_RUN_BASE="${DENSE_RUN_BASE:?ERROR: set DENSE_RUN_BASE to the dense DPO run directory}"

# Training knobs matching the dense baseline
export DPO_LEARNING_RATE="${DPO_LEARNING_RATE:-5e-7}"
export DPO_WARMUP_RATIO="${DPO_WARMUP_RATIO:-0.1}"
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE="${DPO_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export DPO_GRADIENT_ACCUMULATION_STEPS="${DPO_GRADIENT_ACCUMULATION_STEPS:-64}"
export DPO_BETA="${DPO_BETA:-0.1}"
export DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-1024}"
export DPO_MAX_PROMPT_LENGTH="${DPO_MAX_PROMPT_LENGTH:-1024}"
export DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-500}"
export DPO_SAVE_TOTAL_LIMIT="${DPO_SAVE_TOTAL_LIMIT:-2}"

# ---------------------------------------------------------------------------
# Pipeline boilerplate
# ---------------------------------------------------------------------------
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/pipeline_common.sh"
pipeline_setup

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
DELTA_DIR="${DENSE_RUN_BASE}/deltas/qwen_qwen3_8b_light_r1"
MASK_DIR="${MASK_OUT_BASE}/diag_h1_${RUN_ID}"
SPARSE_OUT="${SPARSE_OUT_BASE}/diag_h1_${RUN_ID}"
MASK_FILE="${MASK_DIR}/warm_magnitude_qwen3_8b_light_r1_sp${SPARSITY_PCT}pct.pt"
RUN_NAME="${RUN_NAME:-diag_h1_warm_mask_qwen3_8b_${RUN_ID}}"

mkdir -p "$MASK_DIR" "$SPARSE_OUT" logs

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if [ ! -d "${DELTA_DIR}" ]; then
  echo "ERROR: delta dir not found: ${DELTA_DIR}" >&2
  echo "       Set DENSE_RUN_BASE to the directory containing the dense DPO run." >&2
  exit 1
fi
echo "Delta files in: ${DELTA_DIR}"
ls "${DELTA_DIR}/"

# ---------------------------------------------------------------------------
# Phase 1: Generate warm magnitude mask from float32 delta files
# ---------------------------------------------------------------------------
echo "===== Phase 1: warm magnitude mask from float32 deltas (step ≤250) ====="
"$TRAIN_PY" src/warm_start/even_better_mask_finder.py \
  --delta_log_dir    "$DELTA_DIR" \
  --method           magnitude \
  --sparsity_percent "$SPARSITY_PCT" \
  --target_step      250 \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --output_file      "$MASK_FILE"
echo "  mask → ${MASK_FILE}"

# ---------------------------------------------------------------------------
# Phase 2: Sparse DPO with warm magnitude mask
# ---------------------------------------------------------------------------
EFF_BATCH=$(( DPO_PER_DEVICE_TRAIN_BATCH_SIZE * DPO_GRADIENT_ACCUMULATION_STEPS ))
echo "===== Phase 2: sparse DPO — warm mask (float32), ${NUM_STEPS_DPO} steps, eff_batch=${EFF_BATCH} ====="

timeout --signal=TERM --kill-after=60 $((7 * 60 * 60 + 30 * 60)) \
  "$TRAIN_PY" src/full_training/sparse_dpo_efficiency.py \
    --model_name    "$MODEL" \
    --checkpoint    None \
    --mask          "$MASK_FILE" \
    --dataset       "$DPO_DATASET_KEY" \
    --n_steps       "$NUM_STEPS_DPO" \
    --batch_size    "$DPO_PER_DEVICE_TRAIN_BATCH_SIZE" \
    --grad_accum    "$DPO_GRADIENT_ACCUMULATION_STEPS" \
    --lr            "$DPO_LEARNING_RATE" \
    --warmup_ratio  "$DPO_WARMUP_RATIO" \
    --dpo_beta      "$DPO_BETA" \
    --max_length    "$DPO_MAX_LENGTH" \
    --max_prompt_length "$DPO_MAX_PROMPT_LENGTH" \
    --save_steps    "$DPO_SAVE_STEPS" \
    --save_total_limit "$DPO_SAVE_TOTAL_LIMIT" \
    --optimizer     sparse_adamw \
    --block_size    32 \
    --save_csv \
    --output_base_dir  "$SPARSE_OUT" \
    --dataset_cache_dir "$HF_DATASETS_CACHE_ROOT" \
    --run_name      "$RUN_NAME" \
  2>&1 | tee "logs/diag_h1_warm_mask_qwen3_8b_${RUN_ID}.log"

echo "===== Done. RUN_ID=${RUN_ID} ====="
echo "  mask      : ${MASK_FILE}"
echo "  sparse out: ${SPARSE_OUT}"
echo ""
echo "Compare the loss curve from this run against the checkpoint-diff oracle run."
echo "If this mask shows markedly better learning → H1 (bf16 quantisation) confirmed."
