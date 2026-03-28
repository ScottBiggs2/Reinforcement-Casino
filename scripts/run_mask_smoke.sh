#!/bin/bash
# Lightweight smoke path for hybrid global masking.
# Usage:
#   DELTA_LOG_DIR=/path/to/deltas sbatch scripts/run_mask_smoke.sh
# Optional env vars:
#   MODEL, DATASET, SPARSITY, TARGET_STEP, MIN_LAYER_KEEP_RATIO

#SBATCH --job-name=mask_smoke
#SBATCH --output=logs/mask_smoke_%j.out
#SBATCH --error=logs/mask_smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs masks/smoke

ENV_PATH="/scratch/biggs.s/conda_envs/rl_casino"
PYTHON_BIN="${ENV_PATH}/bin/python"
export PATH="${ENV_PATH}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL="${MODEL:-google/gemma-3-270m-it}"
DATASET="${DATASET:-qihoo360/Light-R1-DPOData}"
SPARSITY="${SPARSITY:-97.5}"
TARGET_STEP="${TARGET_STEP:-50}"
MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
DELTA_LOG_DIR="${DELTA_LOG_DIR:-}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: missing training env python at $PYTHON_BIN" >&2
  exit 1
fi

echo "=== Hybrid selector smoke ==="
"$PYTHON_BIN" src/analysis/verify_hybrid_mask_selector.py

if [ -z "$DELTA_LOG_DIR" ] || [ ! -d "$DELTA_LOG_DIR" ]; then
  echo "ERROR: set DELTA_LOG_DIR to an existing delta-log directory before running this smoke job." >&2
  exit 1
fi

echo "=== Warm magnitude smoke (hybrid default) ==="
"$PYTHON_BIN" src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "$DELTA_LOG_DIR" \
  --method magnitude \
  --sparsity_percent "$SPARSITY" \
  --target_step "$TARGET_STEP" \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --output_file "masks/smoke/warm_hybrid_step${TARGET_STEP}.pt"

echo "=== Warm magnitude smoke (pure global control) ==="
"$PYTHON_BIN" src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "$DELTA_LOG_DIR" \
  --method magnitude \
  --sparsity_percent "$SPARSITY" \
  --target_step "$TARGET_STEP" \
  --min_layer_keep_ratio 0.0 \
  --output_file "masks/smoke/warm_pure_global_step${TARGET_STEP}.pt"

echo "=== Cold Fisher smoke (hybrid default) ==="
"$PYTHON_BIN" src/cold_start/cold_mask_finder.py \
  --model_name "$MODEL" \
  --dataset_name "$DATASET" \
  --sparsity_percent "$SPARSITY" \
  --n_calibration_samples 8 \
  --mini_batch_size 2 \
  --max_length 256 \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --output_file "masks/smoke/cold_fisher_hybrid.pt"

echo "=== Cold Fisher smoke (pure global control) ==="
"$PYTHON_BIN" src/cold_start/cold_mask_finder.py \
  --model_name "$MODEL" \
  --dataset_name "$DATASET" \
  --sparsity_percent "$SPARSITY" \
  --n_calibration_samples 8 \
  --mini_batch_size 2 \
  --max_length 256 \
  --min_layer_keep_ratio 0.0 \
  --output_file "masks/smoke/cold_fisher_pure_global.pt"

echo "Hybrid mask smoke complete. Inspect metadata in masks/smoke/*.pt"
