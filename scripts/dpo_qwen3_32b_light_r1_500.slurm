#!/bin/bash
# Dense DPO training: Qwen3-32B on Light-R1, 250 steps, 8× H200 (FSDP).
#
# Memory strategy — why 8 GPUs:
#   FSDP full_shard with bf16 mixed precision keeps everything the optimizer needs in
#   fp32: master weights (128 GB) + grads (128 GB) + adamw_torch_fused m,v (256 GB)
#   = ~512 GB of fp32 state, sharded evenly across N GPUs = 512/N GB per GPU.
#     - N=4 → 128 GB/GPU + activations ≈ 139 GB → OOM at step 2 (observed: job 7605118,
#       fp32 Adam states allocated after step 1's optimizer.step()).
#     - N=8 → 64 GB/GPU + activations ≈ ~80 GB → comfortable on the 141 GB H200.
#   The H200 nodes (d4052-d4055) have 8 GPUs + 500 GB RAM each, so one node suffices.
#   (adamw_8bit would fit on 4 GPUs but is a silent no-op under FSDP full_shard — loss
#   frozen at ln2=0.6931, rewards exactly 0 — so it is NOT an option here.)
#   - precompute_ref_log_probs: ref model runs once to cache logprobs, then freed
#     before training begins.
#     NOTE: fsdp_cpu_ram_efficient_loading is intentionally omitted — it leaves
#     non-individually-wrapped modules (embed_tokens, final norm) on CPU, which
#     causes a device mismatch during the precompute forward pass on rank > 0.
#     Each rank loads the model to CPU RAM (mmap-shared safetensors) then shards;
#     the H200 node's 500 GB RAM (request --mem=256G) absorbs 8 ranks comfortably.
#   - activation_checkpointing in fsdp_config: activations recomputed, not stored
#     (replaces gradient_checkpointing in TrainingArguments, which adds a redundant
#     AllGather in the FSDP backward pass)
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True reduces allocator fragmentation.
#
# Chat template note: Qwen3-32B uses ChatML (<|im_start|>/<|im_end|>).
# DPO_train.py never calls apply_chat_template — raw text tokenization only —
# so this does not affect training. Downstream eval must use Qwen3 ChatML and
# should pass enable_thinking=False for standard (non-CoT) inference.
#
# Submit from repo root:
#   sbatch scripts/dpo_qwen3_32b_light_r1_500.slurm
#
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
# 250 steps at eff-batch 128 (8-way DP, grad_accum 16) ≈ 2-3h + precompute; 6h is
# safe headroom under the 24h multigpu cap and is backfill-friendly.
#SBATCH --time=06:00:00
#SBATCH --job-name=dpo_qwen3_32b_lr1_500
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/dpo_qwen3_32b_light_r1_%j.out
#SBATCH --error=logs/dpo_qwen3_32b_light_r1_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_SCRIPT_HOME}/.." && pwd)"
fi
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Model / dataset / parallelism
# ---------------------------------------------------------------------------
export MODEL="${MODEL:-Qwen/Qwen3-32B}"
export DPO_DATASET_KEY="${DPO_DATASET_KEY:-light-r1}"
export NUM_STEPS_DPO="${NUM_STEPS_DPO:-250}"
export MULTIGPU_NGPUS="${MULTIGPU_NGPUS:-8}"
# Reduce CUDA allocator fragmentation (the OOM hint from torch).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Training knobs — match Qwen3-8B baseline where possible
# ---------------------------------------------------------------------------
export DPO_LEARNING_RATE="${DPO_LEARNING_RATE:-5e-7}"
export DPO_WARMUP_RATIO="${DPO_WARMUP_RATIO:-0.1}"
# batch=1; grad_accum=16 → effective = 1×16×8GPUs = 128 (same as 8B baseline)
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE="${DPO_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export DPO_GRADIENT_ACCUMULATION_STEPS="${DPO_GRADIENT_ACCUMULATION_STEPS:-16}"
export DPO_BETA="${DPO_BETA:-0.1}"
export DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-1024}"
export DPO_MAX_PROMPT_LENGTH="${DPO_MAX_PROMPT_LENGTH:-1024}"
# Optimizer: adamw_torch_fused (full precision). Do NOT use adamw_8bit here —
# bitsandbytes 8-bit optimizers do not apply updates under FSDP full_shard (silent
# no-op → loss stuck at ln2=0.6931, rewards exactly 0). fsdp_use_orig_params=true is
# kept below (harmless for fused AdamW; was originally added for the bnb path).
export MULTIGPU_OPTIM="${MULTIGPU_OPTIM:-adamw_torch_fused}"

# HF checkpoint: save at end of training only
export DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-250}"
export DPO_SAVE_TOTAL_LIMIT="${DPO_SAVE_TOTAL_LIMIT:-2}"

# ---------------------------------------------------------------------------
# FSDP config — one FSDP unit per Qwen3 decoder layer
# fsdp_use_orig_params: preserves unflattened param views for bitsandbytes compat
# activation_checkpointing: avoids redundant AllGather that gradient_checkpointing
#   in TrainingArguments would cause; see memory strategy note above
# ---------------------------------------------------------------------------
FSDP_CONFIG='{"fsdp_transformer_layer_cls_to_wrap":"Qwen3DecoderLayer","fsdp_use_orig_params":true,"fsdp_state_dict_type":"FULL_STATE_DICT","fsdp_backward_prefetch_policy":"BACKWARD_PRE","activation_checkpointing":true}'

# ---------------------------------------------------------------------------
# Pipeline boilerplate (paths, env check, W&B, etc.)
# ---------------------------------------------------------------------------
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/pipeline_common.sh"
pipeline_setup

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
OUT_BASE="${TRAIN_OUT_BASE}/${RUN_ID}"
RUN_NAME="${RUN_NAME:-dpo_qwen3_32b_light_r1_${RUN_ID}}"
EFF_BATCH=$(( DPO_PER_DEVICE_TRAIN_BATCH_SIZE * DPO_GRADIENT_ACCUMULATION_STEPS * MULTIGPU_NGPUS ))

mkdir -p "$OUT_BASE" logs

echo "===== DPO: Qwen3-32B / light-r1 / ${NUM_STEPS_DPO} steps (RUN_ID=${RUN_ID}) ====="
echo "  model            : ${MODEL}"
echo "  dataset          : ${DPO_DATASET_KEY}"
echo "  steps            : ${NUM_STEPS_DPO}"
echo "  gpus             : ${MULTIGPU_NGPUS}  (FSDP full_shard + precompute_ref_log_probs + activation_checkpointing)"
echo "  lr               : ${DPO_LEARNING_RATE}  warmup: ${DPO_WARMUP_RATIO}"
echo "  batch/device     : ${DPO_PER_DEVICE_TRAIN_BATCH_SIZE}  grad_accum: ${DPO_GRADIENT_ACCUMULATION_STEPS}  effective: ${EFF_BATCH}"
echo "  dpo_beta         : ${DPO_BETA}  max_length: ${DPO_MAX_LENGTH}"
echo "  optim            : ${MULTIGPU_OPTIM}"
echo "  HF ckpt          : every ${DPO_SAVE_STEPS} steps (limit ${DPO_SAVE_TOTAL_LIMIT})"
echo "  output base      : ${OUT_BASE}"

torchrun --standalone --nproc_per_node="${MULTIGPU_NGPUS}" \
  src/full_training/DPO_train.py \
    --model_name                   "$MODEL" \
    --dataset                      "$DPO_DATASET_KEY" \
    --num_steps                    "$NUM_STEPS_DPO" \
    --learning_rate                "$DPO_LEARNING_RATE" \
    --warmup_ratio                 "$DPO_WARMUP_RATIO" \
    --per_device_train_batch_size  "$DPO_PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps  "$DPO_GRADIENT_ACCUMULATION_STEPS" \
    --dpo_beta                     "$DPO_BETA" \
    --max_length                   "$DPO_MAX_LENGTH" \
    --max_prompt_length            "$DPO_MAX_PROMPT_LENGTH" \
    --save_steps                   "$DPO_SAVE_STEPS" \
    --save_total_limit             "$DPO_SAVE_TOTAL_LIMIT" \
    --optim                        "$MULTIGPU_OPTIM" \
    --fsdp                         "full_shard auto_wrap" \
    --fsdp_config                  "$FSDP_CONFIG" \
    --precompute_ref_log_probs \
    --output_base_dir              "$OUT_BASE" \
    --dataset_cache_dir            "$HF_DATASETS_CACHE_ROOT" \
    --run_name                     "$RUN_NAME" \
  2>&1 | tee "logs/dpo_qwen3_32b_light_r1_${RUN_ID}.log"

echo "===== Done. RUN_ID=${RUN_ID} ====="
echo "  checkpoint : ${OUT_BASE}/checkpoints/qwen_qwen3_32b_light_r1/checkpoint-500/"
