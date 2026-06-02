#!/bin/bash
# Sparse DPO: Qwen3-32B on Light-R1, 250 steps, random 97.5% mask.
#
# Memory strategy (single H200, 80 GB VRAM):
#   - --load_in_8bit: weights stored as int8 (~32 GB for 32B model vs 64 GB in bf16)
#   - --precompute_ref_log_probs: ref model runs one forward pass to cache logprobs,
#     then freed from VRAM before training begins (eliminates the frozen 32 GB copy)
#   - SparseAdamW states: only for 2.5% active weights (~0.8B params × 2 states ≈ 6 GB)
#   - Mask: torch.bool, negligible
#   Budget: precompute phase ~64 GB (policy + ref, both int8); training ~40 GB
#
#   Note: bitsandbytes int8 + SparseAdamW is best-effort.  int8 params have
#   requires_grad=True only for the unmasked 2.5%; bitsandbytes' autograd handles
#   the forward/backward pass.  If the optimizer step raises a dtype error, resubmit
#   with MULTIGPU_NGPUS=2 on the dense 32B script as fallback.
#
# Phases (sequential):
#   1. Random mask generation — loads Qwen3-32B to CPU RAM (device_map="cpu"),
#      ~64 GB RAM, no VRAM. Requires 192G --mem allocation.
#   2. Sparse DPO training — loads model in int8 on GPU.
#
# Submit from repo root:
#   sbatch scripts/sparse_dpo_qwen3_32b_random_lr1_250.slurm
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=07:45:00
#SBATCH --job-name=sdpo_q332b_rand_lr1
#SBATCH --mem=192G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/sparse_dpo_qwen3_32b_random_lr1_%j.out
#SBATCH --error=logs/sparse_dpo_qwen3_32b_random_lr1_%j.err

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
export MODEL="${MODEL:-Qwen/Qwen3-32B}"
export DPO_DATASET_KEY="${DPO_DATASET_KEY:-light-r1}"
export NUM_STEPS_DPO="${NUM_STEPS_DPO:-250}"        # reduced from 500 per user request
export SPARSITY_PCT="${SPARSITY_PCT:-97.5}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
export RANDOM_MASK_SEED="${RANDOM_MASK_SEED:-42}"

# Training knobs — match 32B dense baseline where possible
# batch=1 (vs 2 for 8B) to give VRAM headroom
export DPO_LEARNING_RATE="${DPO_LEARNING_RATE:-5e-7}"
export DPO_WARMUP_RATIO="${DPO_WARMUP_RATIO:-0.1}"
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE="${DPO_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export DPO_GRADIENT_ACCUMULATION_STEPS="${DPO_GRADIENT_ACCUMULATION_STEPS:-128}"
export DPO_BETA="${DPO_BETA:-0.1}"
export DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-1024}"
export DPO_MAX_PROMPT_LENGTH="${DPO_MAX_PROMPT_LENGTH:-1024}"
export DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-250}"
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
MASK_FILE="${MASK_DIR}/random_${MODEL_SANITIZED}_sparsity${SPARSITY_PCT}pct_seed${RANDOM_MASK_SEED}.pt"
RUN_NAME="${RUN_NAME:-sdpo_qwen3_32b_rand_lr1_${RUN_ID}}"

mkdir -p "$MASK_DIR" "$SPARSE_OUT" logs

# ---------------------------------------------------------------------------
# Phase 1: Random mask (CPU only — loads 32B weights to RAM, not VRAM)
# ---------------------------------------------------------------------------
echo "===== Phase 1: random mask (${SPARSITY_PCT}% sparsity, seed=${RANDOM_MASK_SEED}) ====="
echo "  Model loads to CPU RAM (~64 GB), no VRAM used"
"$TRAIN_PY" src/utils/generate_random_mask.py \
  --model_name          "$MODEL" \
  --sparsity_percent    "$SPARSITY_PCT" \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --seed                "$RANDOM_MASK_SEED" \
  --output_file         "$MASK_FILE"
echo "  mask → ${MASK_FILE}"

# ---------------------------------------------------------------------------
# Phase 2: Sparse DPO training (int8 weights + precomputed ref logprobs)
# ---------------------------------------------------------------------------
EFF_BATCH=$(( DPO_PER_DEVICE_TRAIN_BATCH_SIZE * DPO_GRADIENT_ACCUMULATION_STEPS ))
echo "===== Phase 2: sparse DPO (${NUM_STEPS_DPO} steps, int8 weights, eff_batch=${EFF_BATCH}) ====="
echo "  VRAM budget: ~32 GB (int8 policy) + ~32 GB (int8 ref, freed after precompute)"

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
    --load_in_8bit \
    --precompute_ref_log_probs \
    --output_base_dir  "$SPARSE_OUT" \
    --dataset_cache_dir "$HF_DATASETS_CACHE_ROOT" \
    --run_name      "$RUN_NAME" \
  2>&1 | tee "logs/sparse_dpo_qwen3_32b_random_lr1_${RUN_ID}.log"

echo "===== Done. RUN_ID=${RUN_ID} ====="
echo "  mask      : ${MASK_FILE}"
echo "  checkpoint: ${SPARSE_OUT}/$(basename "${MASK_FILE%.pt}")/checkpoints/"
