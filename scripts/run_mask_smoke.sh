#!/bin/bash
# HPC smoke path for exact chunked global masking.
# Usage:
#   DELTA_LOG_DIR=/path/to/deltas sbatch scripts/run_mask_smoke.sh
# Optional env vars:
#   MODEL, DATASET, SPARSITY, TARGET_STEP, MIN_LAYER_KEEP_RATIO
#   RL_CASINO_CHUNKED_SELECTOR_MIN_NUMEL, RL_CASINO_WARM_MASK_SCORE_DEVICE

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
export RL_CASINO_CHUNKED_SELECTOR_MIN_NUMEL="${RL_CASINO_CHUNKED_SELECTOR_MIN_NUMEL:-1}"
export RL_CASINO_WARM_MASK_SCORE_DEVICE="${RL_CASINO_WARM_MASK_SCORE_DEVICE:-cpu}"

MODEL="${MODEL:-google/gemma-3-270m-it}"
DATASET="${DATASET:-qihoo360/Light-R1-DPOData}"
SPARSITY="${SPARSITY:-97.5}"
TARGET_STEP="${TARGET_STEP:-50}"
MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
DELTA_LOG_DIR="${DELTA_LOG_DIR:-}"
SMOKE_CALIBRATION_SAMPLES="${SMOKE_CALIBRATION_SAMPLES:-8}"
SMOKE_MINI_BATCH_SIZE="${SMOKE_MINI_BATCH_SIZE:-2}"
SMOKE_MAX_LENGTH="${SMOKE_MAX_LENGTH:-256}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="${OUTPUT_DIR:-masks/smoke/${JOB_TAG}}"
export MIN_LAYER_KEEP_RATIO

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: missing training env python at $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
WARM_HYBRID_OUT="${OUTPUT_DIR}/warm_hybrid_step${TARGET_STEP}.pt"
WARM_PURE_OUT="${OUTPUT_DIR}/warm_pure_global_step${TARGET_STEP}.pt"
COLD_HYBRID_OUT="${OUTPUT_DIR}/cold_fisher_hybrid.pt"
COLD_PURE_OUT="${OUTPUT_DIR}/cold_fisher_pure_global.pt"
export OUTPUT_DIR WARM_HYBRID_OUT WARM_PURE_OUT COLD_HYBRID_OUT COLD_PURE_OUT

echo "=== Smoke configuration ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "DELTA_LOG_DIR=$DELTA_LOG_DIR"
echo "MODEL=$MODEL"
echo "DATASET=$DATASET"
echo "SPARSITY=$SPARSITY"
echo "TARGET_STEP=$TARGET_STEP"
echo "MIN_LAYER_KEEP_RATIO=$MIN_LAYER_KEEP_RATIO"
echo "RL_CASINO_CHUNKED_SELECTOR_MIN_NUMEL=$RL_CASINO_CHUNKED_SELECTOR_MIN_NUMEL"
echo "RL_CASINO_WARM_MASK_SCORE_DEVICE=$RL_CASINO_WARM_MASK_SCORE_DEVICE"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU inventory ==="
  nvidia-smi
fi

echo "=== Exact chunked selector smoke ==="
"$PYTHON_BIN" src/analysis/verify_hybrid_mask_selector.py

if [ -z "$DELTA_LOG_DIR" ] || [ ! -d "$DELTA_LOG_DIR" ]; then
  echo "ERROR: set DELTA_LOG_DIR to an existing delta-log directory before running this smoke job." >&2
  exit 1
fi

echo "=== Warm magnitude smoke (hybrid default, forced chunked selector) ==="
"$PYTHON_BIN" src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "$DELTA_LOG_DIR" \
  --method magnitude \
  --sparsity_percent "$SPARSITY" \
  --target_step "$TARGET_STEP" \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --output_file "$WARM_HYBRID_OUT"

echo "=== Warm magnitude smoke (pure global control, forced chunked selector) ==="
"$PYTHON_BIN" src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "$DELTA_LOG_DIR" \
  --method magnitude \
  --sparsity_percent "$SPARSITY" \
  --target_step "$TARGET_STEP" \
  --min_layer_keep_ratio 0.0 \
  --output_file "$WARM_PURE_OUT"

echo "=== Cold Fisher smoke (hybrid default, forced chunked selector) ==="
"$PYTHON_BIN" src/cold_start/cold_mask_finder.py \
  --model_name "$MODEL" \
  --dataset_name "$DATASET" \
  --sparsity_percent "$SPARSITY" \
  --n_calibration_samples "$SMOKE_CALIBRATION_SAMPLES" \
  --mini_batch_size "$SMOKE_MINI_BATCH_SIZE" \
  --max_length "$SMOKE_MAX_LENGTH" \
  --min_layer_keep_ratio "$MIN_LAYER_KEEP_RATIO" \
  --output_file "$COLD_HYBRID_OUT"

echo "=== Cold Fisher smoke (pure global control, forced chunked selector) ==="
"$PYTHON_BIN" src/cold_start/cold_mask_finder.py \
  --model_name "$MODEL" \
  --dataset_name "$DATASET" \
  --sparsity_percent "$SPARSITY" \
  --n_calibration_samples "$SMOKE_CALIBRATION_SAMPLES" \
  --mini_batch_size "$SMOKE_MINI_BATCH_SIZE" \
  --max_length "$SMOKE_MAX_LENGTH" \
  --min_layer_keep_ratio 0.0 \
  --output_file "$COLD_PURE_OUT"

echo "=== Verifying saved mask artifacts ==="
"$PYTHON_BIN" - <<'PY'
import math
import os
import torch


def expected_keep_count(total_params: int, sparsity_percent: float) -> int:
    keep_percent = 100.0 - sparsity_percent
    return max(1, min(total_params, int(keep_percent / 100.0 * total_params)))


def summarize(path: str, expected_pooling_mode: str, expected_floor, expected_score_device=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing expected smoke artifact: {path}")

    payload = torch.load(path, map_location="cpu")
    if "masks" not in payload:
        raise KeyError(f"{path} did not contain a 'masks' entry")

    masks = payload["masks"]
    metadata = payload.get("metadata", {})
    if not metadata:
        raise ValueError(f"{path} did not contain metadata")

    total = sum(mask.numel() for mask in masks.values())
    kept = sum(int(mask.sum().item()) for mask in masks.values())
    expected_keep = expected_keep_count(total, float(metadata["sparsity_percent"]))
    if kept != expected_keep:
        raise ValueError(
            f"{path} kept {kept} params but expected {expected_keep} from metadata sparsity "
            f"{metadata['sparsity_percent']}"
        )
    if kept <= 0 or kept >= total:
        raise ValueError(f"{path} produced a degenerate mask (kept={kept}, total={total})")

    for name, mask in masks.items():
        if mask.dtype != torch.float32:
            raise TypeError(f"{path}:{name} expected float32 on disk, found {mask.dtype}")

    pooling_mode = metadata.get("pooling_mode")
    if pooling_mode != expected_pooling_mode:
        raise ValueError(f"{path} pooling_mode={pooling_mode}, expected {expected_pooling_mode}")

    if expected_floor is None:
        if "min_layer_keep_ratio" in metadata:
            raise ValueError(f"{path} unexpectedly saved min_layer_keep_ratio for pure-global run")
    else:
        value = float(metadata.get("min_layer_keep_ratio", float("nan")))
        if not math.isclose(value, expected_floor, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{path} min_layer_keep_ratio={value}, expected {expected_floor}")

    if expected_score_device is not None:
        score_device = metadata.get("score_device")
        if score_device != expected_score_device:
            raise ValueError(f"{path} score_device={score_device}, expected {expected_score_device}")

    print(
        f"OK {path}: pooling_mode={pooling_mode}, kept={kept}/{total}, "
        f"sparsity={metadata['sparsity_percent']}, score_device={metadata.get('score_device')}"
    )


min_layer_keep_ratio = float(os.environ["MIN_LAYER_KEEP_RATIO"])
warm_score_device = os.environ["RL_CASINO_WARM_MASK_SCORE_DEVICE"]

summarize(
    os.environ["WARM_HYBRID_OUT"],
    expected_pooling_mode="global_with_layer_floor",
    expected_floor=min_layer_keep_ratio,
    expected_score_device=warm_score_device,
)
summarize(
    os.environ["WARM_PURE_OUT"],
    expected_pooling_mode="global",
    expected_floor=None,
    expected_score_device=warm_score_device,
)
summarize(
    os.environ["COLD_HYBRID_OUT"],
    expected_pooling_mode="global_with_layer_floor",
    expected_floor=min_layer_keep_ratio,
)
summarize(
    os.environ["COLD_PURE_OUT"],
    expected_pooling_mode="global",
    expected_floor=None,
)
PY

echo "=== Smoke outputs ==="
printf '%s\n' "$WARM_HYBRID_OUT" "$WARM_PURE_OUT" "$COLD_HYBRID_OUT" "$COLD_PURE_OUT"
echo "Exact chunked mask smoke complete."
