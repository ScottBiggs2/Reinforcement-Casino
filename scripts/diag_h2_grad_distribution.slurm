#!/bin/bash
# Diagnostic H2: gradient concentration — single DPO backward pass, no training.
#
# Hypothesis 2: Qwen3's DPO gradients are more uniformly distributed across
# parameters than Llama's, making any fixed 2.5% mask (oracle or random)
# insufficient to capture the useful signal.
#
# Metric: coefficient of variation (CV = std/mean) of per-layer mean |∇w|.
#   High CV → concentrated gradients → sparse training viable
#   Low  CV → uniform gradients      → sparse training may be fundamentally limited
#
# This job runs the diagnostic for BOTH Qwen3-8B and Llama 3.1 8B IT so you
# can compare the two distributions directly.
#
# Submit from repo root:
#   sbatch scripts/diag_h2_grad_distribution.slurm
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:30:00
#SBATCH --job-name=diag_h2_grad_dist
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/diag_h2_grad_dist_%j.out
#SBATCH --error=logs/diag_h2_grad_dist_%j.err

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
export MODEL_QWEN="${MODEL_QWEN:-Qwen/Qwen3-8B}"
export MODEL_LLAMA="${MODEL_LLAMA:-meta-llama/Llama-3.1-8B-Instruct}"
export DPO_DATASET_KEY="${DPO_DATASET_KEY:-light-r1}"
export N_SAMPLES="${N_SAMPLES:-32}"   # DPO examples for the backward pass

# ---------------------------------------------------------------------------
# Pipeline boilerplate (paths, env, HF cache)
# ---------------------------------------------------------------------------
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/pipeline_common.sh"
pipeline_setup

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
mkdir -p logs

echo "===== H2 gradient distribution diagnostic (RUN_ID=${RUN_ID}) ====="
echo "  Qwen3 model   : ${MODEL_QWEN}"
echo "  Llama model   : ${MODEL_LLAMA}"
echo "  Dataset       : ${DPO_DATASET_KEY}  (${N_SAMPLES} samples)"
echo ""
echo "Runs single DPO backward pass for each model — no training."
echo "High CV = concentrated gradients (sparse DPO viable)."
echo "Low  CV = uniform gradients      (sparse DPO may be insufficient)."
echo ""

"$TRAIN_PY" src/diagnostics/grad_distribution_diag.py \
  --model_name         "$MODEL_QWEN" \
  --model_name2        "$MODEL_LLAMA" \
  --dataset            "$DPO_DATASET_KEY" \
  --n_samples          "$N_SAMPLES" \
  --output_prefix      "logs/grad_diag_${RUN_ID}" \
  --dataset_cache_dir  "$HF_DATASETS_CACHE_ROOT" \
  2>&1 | tee "logs/diag_h2_grad_dist_${RUN_ID}.log"

echo "===== Done. JSON summaries in logs/grad_diag_${RUN_ID}_*.json ====="
echo ""
echo "Interpretation guide:"
echo "  Qwen3  CV >> Llama CV  →  gradient concentration is NOT the issue"
echo "  Qwen3  CV ~= Llama CV  →  H2 inconclusive"
echo "  Qwen3  CV << Llama CV  →  H2 confirmed: Qwen3 gradients too uniform for sparse DPO"
