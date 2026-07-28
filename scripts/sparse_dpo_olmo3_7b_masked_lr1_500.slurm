#!/bin/bash
# Sparse DPO: Olmo-3-7B-Instruct on Light-R1, 500 steps, on a PRE-COMPUTED mask.
#
# Parametrized by MASK_FILE (required) so this one script covers all three
# non-random masks built by gen_masks_olmo3_7b_lr1.slurm:
#   - magnitude@250      (warm "rewinding" mask)
#   - oracle (delta-log ground_truth @ step 500)
#   - oracle (checkpoint diff, base vs step-500 ckpt)
# The random-mask condition has its own generator+train script:
#   scripts/sparse_dpo_olmo3_7b_random_lr1_500.slurm
#
# Submit from repo root (MASK_LABEL is a short tag for run/output names):
#   sbatch --export=ALL,MASK_FILE=/scratch/.../oracle_deltalog_...step500.pt,MASK_LABEL=oracle_deltalog \
#     scripts/sparse_dpo_olmo3_7b_masked_lr1_500.slurm
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=07:45:00
#SBATCH --job-name=sdpo_o37b_masked_lr1
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/sparse_dpo_olmo3_7b_masked_lr1_%j.out
#SBATCH --error=logs/sparse_dpo_olmo3_7b_masked_lr1_%j.err

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
export MODEL="${MODEL:-allenai/Olmo-3-7B-Instruct}"
export DPO_DATASET_KEY="${DPO_DATASET_KEY:-light-r1}"
export NUM_STEPS_DPO="${NUM_STEPS_DPO:-500}"

if [ -z "${MASK_FILE:-}" ]; then
  echo "ERROR: MASK_FILE is required." >&2
  echo "       sbatch --export=ALL,MASK_FILE=<path>,MASK_LABEL=<tag> scripts/sparse_dpo_olmo3_7b_masked_lr1_500.slurm" >&2
  exit 1
fi
MASK_LABEL="${MASK_LABEL:-$(basename "${MASK_FILE%.pt}")}"

# Training knobs — match dense Olmo-3-7B baseline
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
SPARSE_OUT="${SPARSE_OUT_BASE}/${RUN_ID}_${MASK_LABEL}"
RUN_NAME="${RUN_NAME:-sdpo_olmo3_7b_${MASK_LABEL}_lr1_${RUN_ID}}"

mkdir -p "$SPARSE_OUT" logs

if [ ! -f "${MASK_FILE}" ]; then
  echo "ERROR: mask not found: ${MASK_FILE}" >&2
  echo "       Run: sbatch --export=ALL,DENSE_RUN_ID=<run_id> scripts/gen_masks_olmo3_7b_lr1.slurm" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Sparse DPO training
# ---------------------------------------------------------------------------
EFF_BATCH=$(( DPO_PER_DEVICE_TRAIN_BATCH_SIZE * DPO_GRADIENT_ACCUMULATION_STEPS ))
echo "===== Sparse DPO: Olmo-3-7B / light-r1 / ${NUM_STEPS_DPO} steps (RUN_ID=${RUN_ID}) ====="
echo "  mask label : ${MASK_LABEL}"
echo "  mask file  : ${MASK_FILE}"
echo "  eff_batch  : ${EFF_BATCH}"
echo "  output     : ${SPARSE_OUT}"

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
  2>&1 | tee "logs/sparse_dpo_olmo3_7b_${MASK_LABEL}_lr1_${RUN_ID}.log"

echo "===== Done. RUN_ID=${RUN_ID} (label=${MASK_LABEL}) ====="
echo "  checkpoint: ${SPARSE_OUT}/"
