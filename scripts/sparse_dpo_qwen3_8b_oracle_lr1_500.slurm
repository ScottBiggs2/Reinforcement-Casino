#!/bin/bash
# Sparse DPO: Qwen3-8B on Light-R1, 500 steps, oracle (checkpoint-diff) 97.5% mask.
#
# The oracle mask is built from the diff between the pre-trained Qwen3-8B weights
# and the final step-500 checkpoint from the dense DPO run.  Parameters that moved
# most are marked as "active" (top 2.5% by |Δw|); the rest are frozen.
#
# Phases (sequential, same GPU allocation):
#   1. Oracle mask generation — loads both models to CPU RAM via --force_cpu.
#      8B × 2 × bf16 = ~32 GB RAM; fine with 128G allocation.
#   2. Sparse DPO training — loads model in bf16 on GPU (~16 GB).
#
# Dense DPO checkpoint used as "final" for oracle mask:
#   /scratch/biggs.s/rl_casino_train/20260601_175443_7366081/
#     checkpoints/qwen_qwen3_8b_light_r1/checkpoint-500
#
# Submit from repo root:
#   sbatch scripts/sparse_dpo_qwen3_8b_oracle_lr1_500.slurm
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=07:45:00
#SBATCH --job-name=sdpo_q38b_oracle_lr1
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/sparse_dpo_qwen3_8b_oracle_lr1_%j.out
#SBATCH --error=logs/sparse_dpo_qwen3_8b_oracle_lr1_%j.err

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

# Final checkpoint from the dense Qwen3-8B run (override via env if path changes)
export ORACLE_FINAL_CKPT="${ORACLE_FINAL_CKPT:-/scratch/biggs.s/rl_casino_train/20260601_175443_7366081/checkpoints/qwen_qwen3_8b_light_r1/checkpoint-500}"

# Training knobs — match dense Qwen3-8B baseline
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
MASK_DIR="${MASK_OUT_BASE}/${RUN_ID}"
SPARSE_OUT="${SPARSE_OUT_BASE}/${RUN_ID}"
MODEL_SANITIZED=$(echo "$MODEL" | tr '/' '_' | tr '-' '_' | tr '[:upper:]' '[:lower:]')
MASK_FILE="${MASK_DIR}/oracle_diff_${MODEL_SANITIZED}_light_r1_sparsity${SPARSITY_PCT}pct.pt"
RUN_NAME="${RUN_NAME:-sdpo_qwen3_8b_oracle_lr1_${RUN_ID}}"

mkdir -p "$MASK_DIR" "$SPARSE_OUT" logs

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if [ ! -d "${ORACLE_FINAL_CKPT}" ]; then
  echo "ERROR: ORACLE_FINAL_CKPT not found: ${ORACLE_FINAL_CKPT}" >&2
  echo "       Set ORACLE_FINAL_CKPT to the step-500 checkpoint directory." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Phase 1: Oracle mask (CPU only — both models loaded via --force_cpu)
# ---------------------------------------------------------------------------
echo "===== Phase 1: oracle mask (|Δw| from checkpoint diff, ${SPARSITY_PCT}% sparsity) ====="
echo "  initial : ${MODEL}"
echo "  final   : ${ORACLE_FINAL_CKPT}"

"$TRAIN_PY" src/warm_start/checkpoint_diff_mask_finder.py \
  --initial_model        "$MODEL" \
  --final_model          "$ORACLE_FINAL_CKPT" \
  --sparsity_percent     "$SPARSITY_PCT" \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --output_file          "$MASK_FILE" \
  --force_cpu
echo "  mask → ${MASK_FILE}"

# ---------------------------------------------------------------------------
# Phase 2: Sparse DPO training
# ---------------------------------------------------------------------------
EFF_BATCH=$(( DPO_PER_DEVICE_TRAIN_BATCH_SIZE * DPO_GRADIENT_ACCUMULATION_STEPS ))
echo "===== Phase 2: sparse DPO (${NUM_STEPS_DPO} steps, eff_batch=${EFF_BATCH}) ====="

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
  2>&1 | tee "logs/sparse_dpo_qwen3_8b_oracle_lr1_${RUN_ID}.log"

echo "===== Done. RUN_ID=${RUN_ID} ====="
echo "  mask      : ${MASK_FILE}"
echo "  checkpoint: ${SPARSE_OUT}/$(basename "${MASK_FILE%.pt}")/checkpoints/"
