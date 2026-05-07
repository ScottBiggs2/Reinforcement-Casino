#!/bin/bash
# Eval one Tulu3-trained checkpoint (or base) on the full 7-benchmark suite.
# Mirrors scripts/run_evals_slurm.sh but parameterized by MODEL_PATH + TAG so
# we can fan out cleanly over the 5 sparse + 1 dense + 1 base candidates.
#
# Submit (from repo root on cluster):
#   MODEL_PATH=/scratch/.../final_model TAG=warm_mag \
#     sbatch scripts/eval_tulu3_suite.sbatch
#
# Output dir: results/eval_tulu3_${TAG}_${SLURM_JOB_ID}/
#
#SBATCH --job-name=eval_tulu3
#SBATCH --output=logs/eval_tulu3_%j.out
#SBATCH --error=logs/eval_tulu3_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100@80g
#SBATCH --mem=128G
#SBATCH --time=06:00:00
# constraint: A100-40GB OOMs on Llama-8B vLLM during MMLU loglikelihood;
# d1028/d1029 are 80GB (feature a100@80g); d1026 is 40GB.

set -euo pipefail

if [ -z "${MODEL_PATH:-}" ]; then
    echo "ERROR: MODEL_PATH env var is required" >&2
    exit 2
fi
TAG="${TAG:-untagged}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs results

OUTPUT_DIR="results/eval_tulu3_${TAG}_${SLURM_JOB_ID}"

echo "==================================================="
echo "Eval start  : $(date)"
echo "Node        : $(hostname)"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "MODEL_PATH  : ${MODEL_PATH}"
echo "TAG         : ${TAG}"
echo "OUTPUT_DIR  : ${OUTPUT_DIR}"
echo "==================================================="

# HF token from on-disk cache (gated Llama-3.1)
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    export HF_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set; gated Llama-3.1 download may fail" >&2
fi

ENV_PATH="${EVAL_ENV:-/scratch/biggs.s/conda_envs/rl_casino_eval}"
PYTHON_BIN="$ENV_PATH/bin/python"
export PATH="$ENV_PATH/bin:$PATH"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "ERROR: Python not found at $PYTHON_BIN" >&2
    exit 1
fi

echo "--- env check ---"
which python
python --version
python -c "import torch, transformers; print('torch', torch.__version__, '/ transformers', transformers.__version__)"
python -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || echo "vLLM NOT available — will fall back to HF"

bash scripts/install_lm_eval.sh
python -c "import lm_eval; print('lm-eval', lm_eval.__version__)"

VLLM_ARG=""
if [ "${FORCE_HF_BACKEND:-0}" = "1" ]; then
    echo "FORCE_HF_BACKEND=1 -- using HF backend"
elif python -c "import vllm" 2>/dev/null; then
    VLLM_ARG="--use_vllm"
fi

export HF_ALLOW_CODE_EVAL=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_MODULE_LOADING=LAZY
export TRANSFORMERS_VERBOSITY=error

echo "--- launching run_all_benchmarks.py ---"
PYTHONUNBUFFERED=1 python src/evaluation/run_all_benchmarks.py \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size auto \
    --trust_remote_code \
    --apply_chat_template \
    --verbose \
    ${VLLM_ARG} \
    "$@"
RC=$?

echo "==================================================="
echo "Eval end    : $(date)"
echo "Exit code   : ${RC}"
echo "Results     : ${OUTPUT_DIR}"
echo "==================================================="
exit ${RC}
