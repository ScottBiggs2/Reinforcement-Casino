#!/bin/bash
# Sparse DPO: Qwen3-32B on Light-R1, 250 steps, random 97.5% mask.
#
# Memory strategy — why bf16 on 4 GPUs (device_map, NOT FSDP):
#   The sparse path uses a custom SparseAdamW + Triton kernels that are not FSDP/DDP
#   compatible, so the model is spread with device_map="auto" (naive model parallelism).
#   device_map does NOT shard gradients — each GPU holds full grads for its layers — and
#   a random 2.5% mask leaves almost every weight *tensor* with requires_grad=True, so
#   gradients are effectively dense (~64 GB) on top of the bf16 weights (~64 GB).
#     - bf16 on 2 GPUs OOMs at accelerator.prepare: weights + dense grads + the precompute
#       reference-model copy fill GPU 0 to ~140 GB (observed: job 7605119).
#     - int8 (32 GB weights) fit on 2 GPUs but dequant is too slow to finish 250 steps in
#       the 18h wall (~6 min/step ≈ 27h).
#     - bf16 on 4 GPUs: ~16 GB weights + ~16 GB grads per GPU + activations ≈ comfortable;
#       bf16 matmuls keep ~3-4 min/step so 250 steps fit the wall.
#   - --precompute_ref_log_probs: ref model runs one forward pass to cache logprobs,
#     then freed before training begins
#   - SparseAdamW states: only for 2.5% active weights (sparse, ~1 GB total)
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True reduces allocator fragmentation
#     (precompute frees the ref model, leaving fragmented free space).
#   This bf16 path matches the working Qwen3-8B oracle run (sparse_adamw, bf16).
#
# Phases (sequential):
#   1. Random mask generation — loads Qwen3-32B to CPU RAM (device_map="cpu"),
#      ~64 GB RAM, no VRAM.
#   2. Sparse DPO training — loads model in bf16, spread across 4 GPUs via device_map.
#
# Submit from repo root:
#   sbatch scripts/sparse_dpo_qwen3_32b_random_lr1_250.slurm
#
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
# device_map MP is sequential (more GPUs add memory headroom, not speed); ~3-4 min/step
# at eff-batch 128 → 250 steps + precompute ≈ 14-15h. multigpu MaxTime is 24h.
#SBATCH --time=18:00:00
#SBATCH --job-name=sdpo_q332b_rand_lr1
#SBATCH --mem=400G
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
# Reduce CUDA allocator fragmentation after the precompute ref model is freed.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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
# Rolling HF checkpoints every 50 steps (limit 2): on a long ~15h run this leaves a
# resumable checkpoint if the job is interrupted (resume via --resume_from_checkpoint auto).
export DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-50}"
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
# Phase 2: Sparse DPO training (bf16 weights across 4 GPUs + precomputed ref logprobs)
# ---------------------------------------------------------------------------
EFF_BATCH=$(( DPO_PER_DEVICE_TRAIN_BATCH_SIZE * DPO_GRADIENT_ACCUMULATION_STEPS ))
echo "===== Phase 2: sparse DPO (${NUM_STEPS_DPO} steps, bf16 weights, eff_batch=${EFF_BATCH}) ====="
echo "  VRAM budget: ~64 GB bf16 policy + ~64 GB grads split across 4 GPUs + ref (freed after precompute)"

# Internal soft timeout 17h30m — sends SIGTERM before the 18h SLURM wall so partial
# work + the latest rolling checkpoint are flushed cleanly.
timeout --signal=TERM --kill-after=60 $((17 * 60 * 60 + 30 * 60)) \
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
    --precompute_ref_log_probs \
    --output_base_dir  "$SPARSE_OUT" \
    --dataset_cache_dir "$HF_DATASETS_CACHE_ROOT" \
    --run_name      "$RUN_NAME" \
  2>&1 | tee "logs/sparse_dpo_qwen3_32b_random_lr1_${RUN_ID}.log"

echo "===== Done. RUN_ID=${RUN_ID} ====="
echo "  mask      : ${MASK_FILE}"
echo "  checkpoint: ${SPARSE_OUT}/$(basename "${MASK_FILE%.pt}")/checkpoints/"
