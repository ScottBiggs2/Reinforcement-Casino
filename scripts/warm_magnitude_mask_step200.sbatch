#!/bin/bash
# Warm-start magnitude mask at target_step=200 (Scott spec).
# Uses src/warm_start/even_better_mask_finder.py --method magnitude.
# Score = sum_{t logged, t<=200} |W_t - W_0|, then top-K by global magnitude
# with min_layer_keep_ratio floor.
#
# Required env vars:
#   DELTA_LOG_DIR : dir containing base_state.pt + deltas_step_*.pt
#   DATASET_TAG   : "lightr1" or "tulu3" (used in output filename only)
#
#SBATCH --job-name=warm_mag_mask_step200
#SBATCH --output=logs/warm_mag_mask_step200_%j.out
#SBATCH --error=logs/warm_mag_mask_step200_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --requeue

set -euo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="/home/xie.yiyi/.conda/envs/rl_casino"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

cd /home/xie.yiyi/rc-sparse-speed
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="/scratch/xie.yiyi/hf_cache"
export RL_CASINO_WARM_MASK_SCORE_DEVICE="${RL_CASINO_WARM_MASK_SCORE_DEVICE:-cuda}"

: "${DELTA_LOG_DIR:?Set DELTA_LOG_DIR=/scratch/.../deltas/<subdir>/}"
: "${DATASET_TAG:?Set DATASET_TAG=lightr1 | tulu3}"

SPARSITY=97.5
MIN_LAYER_KEEP_RATIO=0.0025
TARGET_STEP=200

OUT_DIR="/scratch/xie.yiyi/transfer_v1/warm_masks_llama8b"
mkdir -p "$OUT_DIR"
OUT_FILE="$OUT_DIR/warm_magnitude_dpo_${DATASET_TAG}_step${TARGET_STEP}_sp${SPARSITY}.pt"

echo "=========================================="
echo "Warm magnitude mask  |  Llama-3.1-8B-Instruct  |  $DATASET_TAG"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="
echo "DELTA_LOG_DIR        = $DELTA_LOG_DIR"
echo "TARGET_STEP          = $TARGET_STEP"
echo "SPARSITY             = $SPARSITY%"
echo "MIN_LAYER_KEEP_RATIO = $MIN_LAYER_KEEP_RATIO"
echo "OUT_FILE             = $OUT_FILE"
echo ""
echo "Available delta files (only steps <= $TARGET_STEP will be used):"
ls -lh "$DELTA_LOG_DIR/"deltas_step_*.pt 2>/dev/null || echo "  (none found)"
echo "=========================================="

python src/warm_start/even_better_mask_finder.py \
    --delta_log_dir "$DELTA_LOG_DIR" \
    --method magnitude \
    --sparsity_percent "$SPARSITY" \
    --target_step "$TARGET_STEP" \
    --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
    --output_file "$OUT_FILE"

echo ""
echo "=========================================="
echo "Finished: $(date)"
ls -lh "$OUT_FILE"
echo "=========================================="
