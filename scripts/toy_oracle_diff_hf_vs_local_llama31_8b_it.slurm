#!/usr/bin/env bash
#
# 1-hour GPU helper: build "oracle" checkpoint-diff masks by comparing
# HF base (`meta-llama/Llama-3.1-8B-Instruct`) vs a local run's:
#   - final_model/
#   - checkpoints/checkpoint-175/
#   - checkpoints/checkpoint-200/
#
# Usage (from repo root):
#   sbatch scripts/toy_oracle_diff_hf_vs_local_llama31_8b_it.slurm
#
# Override defaults:
#   sbatch --export=ALL,BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct,LOCAL_RUN_DIR=/scratch/... scripts/toy_oracle_diff_hf_vs_local_llama31_8b_it.slurm
#
# Notes:
# - Requires HF token for gated models if not already cached: export HF_TOKEN=...
# - Writes masks under $OUT_DIR (default: ./masks/toy_oracle_ckpt_diff/)
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=toy_oracle_diff
#SBATCH --output=logs/toy_oracle_diff_%j.out
#SBATCH --error=logs/toy_oracle_diff_%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_HERE}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

# Optionally point at a conda env like the other scripts do.
# If TRAIN_ENV is not set, we fall back to `python` on PATH.
TRAIN_ENV="${TRAIN_ENV:-}"
if [ -n "${TRAIN_ENV}" ] && [ -x "${TRAIN_ENV}/bin/python" ]; then
  PY="${TRAIN_ENV}/bin/python"
  export PATH="${TRAIN_ENV}/bin:${PATH}"
else
  PY="${PYTHON:-python}"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_TOKEN="${HF_TOKEN:-}"

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-/scratch/biggs.s/rl_casino_sparse_train/dpo200_sparse_lr1_random_elem_rerun_orch_lr1_grasp6_6376972_0428_1356/random_elem_meta-llama_Llama-3.1-8B-Instruct_light_r1_sp97.5pct_seed42/fullpipe_sparse_random_elem_meta-llama_Llama-3.1-8B-Instruct_light_r1_sp97.5pct_seed42_dpo200_sparse_lr1_random_elem_rerun_orch_lr1_grasp6_6376972_0428_1356}"

SPARSITY_PERCENT="${SPARSITY_PERCENT:-97.5}"
MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/masks/toy_oracle_ckpt_diff}"
mkdir -p "${OUT_DIR}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "PY=${PY}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "LOCAL_RUN_DIR=${LOCAL_RUN_DIR}"
echo "SPARSITY_PERCENT=${SPARSITY_PERCENT}"
echo "MIN_LAYER_KEEP_RATIO=${MIN_LAYER_KEEP_RATIO}"
echo "OUT_DIR=${OUT_DIR}"

"${PY}" - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

FINAL_MODEL_DIR="${LOCAL_RUN_DIR}/final_model"
CKPT_175_DIR="${LOCAL_RUN_DIR}/checkpoints/checkpoint-175"
CKPT_200_DIR="${LOCAL_RUN_DIR}/checkpoints/checkpoint-200"

run_one () {
  local label="$1"
  local final_path="$2"
  local out_path="$3"

  echo ""
  echo "=== Oracle diff: ${label} ==="
  echo "  initial: ${BASE_MODEL}"
  echo "  final:   ${final_path}"
  echo "  out:     ${out_path}"

  "${PY}" src/warm_start/checkpoint_diff_mask_finder.py \
    --initial_model "${BASE_MODEL}" \
    --final_model "${final_path}" \
    --sparsity_percent "${SPARSITY_PERCENT}" \
    --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
    --output_file "${out_path}"
}

run_one "HF → final_model" "${FINAL_MODEL_DIR}" "${OUT_DIR}/oracle_diff_final_model_sparsity${SPARSITY_PERCENT}pct.pt"
run_one "HF → checkpoint-175" "${CKPT_175_DIR}" "${OUT_DIR}/oracle_diff_checkpoint175_sparsity${SPARSITY_PERCENT}pct.pt"
run_one "HF → checkpoint-200" "${CKPT_200_DIR}" "${OUT_DIR}/oracle_diff_checkpoint200_sparsity${SPARSITY_PERCENT}pct.pt"

echo ""
echo "Done. Outputs:"
ls -lh "${OUT_DIR}" || true

