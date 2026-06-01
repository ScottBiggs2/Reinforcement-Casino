#!/bin/bash
# Dense DPO training: Qwen3-8B on Light-R1, 500 steps.
#
# Chat template note: Qwen3-8B uses ChatML (<|im_start|>/<|im_end|>) while
# Llama 3.1 uses the Llama-3 template (<|start_header_id|>/<|eot_id|>).
# DPO_train.py never calls apply_chat_template — it tokenizes raw text directly —
# so this difference does NOT affect training. Downstream eval/generation scripts
# that call apply_chat_template must switch to Qwen3's ChatML template and should
# pass enable_thinking=False for standard (non-CoT) inference.
#
# Hyperparameters match the Llama 3.1 base case (dpo_5k_hpc_copypaste.md):
#   LR=5e-7, warmup=0.1, batch=2, grad_accum=64, beta=0.1, max_length=1024
# Delta logs: steps 50,100,150,200,250 (same schedule as Llama 3.1 base case)
# HF checkpoint: step 500 only (end of training)
#
# Submit from repo root:
#   sbatch scripts/dpo_qwen3_8b_light_r1_500.slurm
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=07:45:00
#SBATCH --job-name=dpo_qwen3_8b_lr1_500
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/dpo_qwen3_8b_light_r1_%j.out
#SBATCH --error=logs/dpo_qwen3_8b_light_r1_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_SCRIPT_HOME}/.." && pwd)"
fi
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Model / dataset / hyperparameters
# ---------------------------------------------------------------------------
export MODEL="${MODEL:-Qwen/Qwen3-8B}"
export DPO_DATASET_KEY="${DPO_DATASET_KEY:-light-r1}"
export NUM_STEPS_DPO="${NUM_STEPS_DPO:-500}"

# Training knobs — match Llama 3.1 base case from dpo_5k_hpc_copypaste.md
export DPO_LEARNING_RATE="${DPO_LEARNING_RATE:-5e-7}"
export DPO_WARMUP_RATIO="${DPO_WARMUP_RATIO:-0.1}"
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE="${DPO_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export DPO_GRADIENT_ACCUMULATION_STEPS="${DPO_GRADIENT_ACCUMULATION_STEPS:-64}"
export DPO_BETA="${DPO_BETA:-0.1}"
export DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-1024}"
export DPO_MAX_PROMPT_LENGTH="${DPO_MAX_PROMPT_LENGTH:-1024}"

# Delta log schedule: same as Llama 3.1 base case → [50, 100, 150, 200, 250]
export DELTA_LOG_INTERVAL="${DELTA_LOG_INTERVAL:-50}"
export DELTA_LOG_END_STEP="${DELTA_LOG_END_STEP:-250}"

# HF checkpoint: save only at step 500 (end of training)
export DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-500}"
export DPO_SAVE_TOTAL_LIMIT="${DPO_SAVE_TOTAL_LIMIT:-2}"

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
RUN_NAME="${RUN_NAME:-dpo_qwen3_8b_light_r1_${RUN_ID}}"

mkdir -p "$OUT_BASE" logs

echo "===== DPO: Qwen3-8B / light-r1 / ${NUM_STEPS_DPO} steps (RUN_ID=${RUN_ID}) ====="
echo "  model            : ${MODEL}"
echo "  dataset          : ${DPO_DATASET_KEY}"
echo "  steps            : ${NUM_STEPS_DPO}"
echo "  lr               : ${DPO_LEARNING_RATE}  warmup: ${DPO_WARMUP_RATIO}"
echo "  batch/device     : ${DPO_PER_DEVICE_TRAIN_BATCH_SIZE}  grad_accum: ${DPO_GRADIENT_ACCUMULATION_STEPS}"
echo "  dpo_beta         : ${DPO_BETA}  max_length: ${DPO_MAX_LENGTH}"
echo "  delta schedule   : every ${DELTA_LOG_INTERVAL} steps through step ${DELTA_LOG_END_STEP}"
echo "  HF ckpt          : every ${DPO_SAVE_STEPS} steps (limit ${DPO_SAVE_TOTAL_LIMIT})"
echo "  output base      : ${OUT_BASE}"

timeout --signal=TERM --kill-after=60 $((7 * 60 * 60 + 30 * 60)) \
  "$TRAIN_PY" src/full_training/DPO_train.py \
    --model_name          "$MODEL" \
    --dataset             "$DPO_DATASET_KEY" \
    --num_steps           "$NUM_STEPS_DPO" \
    --learning_rate       "$DPO_LEARNING_RATE" \
    --warmup_ratio        "$DPO_WARMUP_RATIO" \
    --per_device_train_batch_size  "$DPO_PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps  "$DPO_GRADIENT_ACCUMULATION_STEPS" \
    --dpo_beta            "$DPO_BETA" \
    --max_length          "$DPO_MAX_LENGTH" \
    --max_prompt_length   "$DPO_MAX_PROMPT_LENGTH" \
    --delta_log_interval  "$DELTA_LOG_INTERVAL" \
    --delta_log_end_step  "$DELTA_LOG_END_STEP" \
    --save_steps          "$DPO_SAVE_STEPS" \
    --save_total_limit    "$DPO_SAVE_TOTAL_LIMIT" \
    --gradient_checkpointing \
    --output_base_dir     "$OUT_BASE" \
    --dataset_cache_dir   "$HF_DATASETS_CACHE_ROOT" \
    --run_name            "$RUN_NAME" \
  2>&1 | tee "logs/dpo_qwen3_8b_light_r1_${RUN_ID}.log"

echo "===== Done. RUN_ID=${RUN_ID} ====="
echo "  checkpoints : ${OUT_BASE}/checkpoints/qwen_qwen3_8b_light_r1/"
echo "  deltas      : ${OUT_BASE}/deltas/qwen_qwen3_8b_light_r1/"
