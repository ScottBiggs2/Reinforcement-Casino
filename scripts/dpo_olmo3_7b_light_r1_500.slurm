#!/bin/bash
# Dense DPO training: Olmo-3-7B-Instruct on Light-R1, 500 steps.
#
# Chat template note: Olmo 3 ships its own chat template (distinct from Llama-3
# and Qwen3 ChatML). DPO_train.py never calls apply_chat_template — it tokenizes
# raw text directly — so this difference does NOT affect training. Downstream
# eval/generation scripts that call apply_chat_template must use Olmo 3's template.
#
# Env note: Olmo 3 registers as model_type=olmo3 / Olmo3ForCausalLM and needs
# transformers >= 4.57. The rl_casino env has transformers 4.57.1 (olmo3 in the
# model mapping), torch 2.9.0+cu128, trl 0.24.0 — loads out of the box.
#
# Hyperparameters match the Qwen3-8B / Llama 3.1 base case:
#   LR=5e-7, warmup=0.1, batch=2, grad_accum=64, beta=0.1, max_length=1024
#
# Delta logging (bf16) runs every 50 steps THROUGH step 500 (not 250 as Qwen3).
# This yields both:
#   - deltas_step_250.pt  → warm magnitude / "rewinding" mask (250-step)
#   - deltas_step_500.pt  → end-to-end "ground_truth" oracle mask (delta-log method)
# HF checkpoint: step 500 only (end of training) — also feeds the checkpoint-diff oracle.
#
# Submit from repo root:
#   sbatch scripts/dpo_olmo3_7b_light_r1_500.slurm
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=07:45:00
#SBATCH --job-name=dpo_olmo3_7b_lr1_500
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/dpo_olmo3_7b_light_r1_%j.out
#SBATCH --error=logs/dpo_olmo3_7b_light_r1_%j.err

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
export MODEL="${MODEL:-allenai/Olmo-3-7B-Instruct}"
export DPO_DATASET_KEY="${DPO_DATASET_KEY:-light-r1}"
export NUM_STEPS_DPO="${NUM_STEPS_DPO:-500}"

# Training knobs — match Qwen3-8B / Llama 3.1 base case
export DPO_LEARNING_RATE="${DPO_LEARNING_RATE:-5e-7}"
export DPO_WARMUP_RATIO="${DPO_WARMUP_RATIO:-0.1}"
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE="${DPO_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export DPO_GRADIENT_ACCUMULATION_STEPS="${DPO_GRADIENT_ACCUMULATION_STEPS:-64}"
export DPO_BETA="${DPO_BETA:-0.1}"
export DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-1024}"
export DPO_MAX_PROMPT_LENGTH="${DPO_MAX_PROMPT_LENGTH:-1024}"

# Delta log schedule: every 50 steps through step 500 → [50,100,...,500].
# End at 500 (vs Qwen3's 250) so the final-step delta feeds the delta-log oracle.
export DELTA_LOG_INTERVAL="${DELTA_LOG_INTERVAL:-50}"
export DELTA_LOG_END_STEP="${DELTA_LOG_END_STEP:-500}"

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
RUN_NAME="${RUN_NAME:-dpo_olmo3_7b_light_r1_${RUN_ID}}"

mkdir -p "$OUT_BASE" logs

echo "===== DPO: Olmo-3-7B / light-r1 / ${NUM_STEPS_DPO} steps (RUN_ID=${RUN_ID}) ====="
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
  2>&1 | tee "logs/dpo_olmo3_7b_light_r1_${RUN_ID}.log"

echo "===== Done. RUN_ID=${RUN_ID} ====="
echo "  checkpoints : ${OUT_BASE}/checkpoints/allenai_olmo_3_7b_instruct_light_r1/"
echo "  deltas      : ${OUT_BASE}/deltas/allenai_olmo_3_7b_instruct_light_r1/"
echo ""
echo "  Next: build masks from this run's outputs (fill RUN_ID / paths):"
echo "    sbatch scripts/gen_masks_olmo3_7b_lr1.slurm    # edit DENSE_RUN_ID first"
