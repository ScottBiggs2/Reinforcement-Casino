#!/bin/bash
# Slurm job script to run the full evaluation suite
# Usage: sbatch scripts/run_evals_slurm.sh --model_path <HUGGINGFACE_ID_OR_PATH>

#SBATCH --job-name=llm_eval_suite
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00

# Exit on any error
set -e

# Slurm copies batch scripts to spool — use submit directory as repo root.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs results

echo "Evaluation started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Evaluation settings
# Set LIMIT to a number (e.g. 100) for faster testing, or "" for full evaluation
LIMIT="" 
# Set BATCH_SIZE to "auto" (recommended) or a specific integer
BATCH_SIZE="auto"

# Check for HF Token (required for gated Llama 3.1 models)
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN is not set. Accessing gated models (Llama 3.1) may fail."
    # If you have your token in a file or want to set it here, uncomment:
    # export HF_TOKEN="your_token"
fi

# Conda environment activation
# if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
#   source "$HOME/miniconda3/etc/profile.d/conda.sh"
# elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
#   source "$HOME/anaconda3/etc/profile.d/conda.sh"
# fi

# Activate environment using the full path to the environment's python
# This is much more robust on Slurm than 'conda activate'
# Dedicated env for eval harness (lm-eval + benchmark runners)
ENV_PATH="/scratch/biggs.s/conda_envs/rl_casino_eval"
PYTHON_BIN="$ENV_PATH/bin/python"

# Export PATH to ensure sub-scripts use the environment's binaries
export PATH="$ENV_PATH/bin:$PATH"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "ERROR: Python binary not found at $PYTHON_BIN"
    exit 1
fi

echo "================================"
echo "Environment Check"
echo "================================"
which python
python --version
python -c "import torch; import vllm; import transformers; print(f'PyTorch: {torch.__version__}, vLLM: {vllm.__version__}, Transformers: {transformers.__version__}')"

# Install/Verify lm-eval inside the env using the dedicated eval requirements
# (The main requirements.txt is for training systems; eval_requirements.txt handles the harness)
# $PYTHON_BIN -m pip install -r requirements.txt -q
bash scripts/install_lm_eval.sh
$PYTHON_BIN -c "import lm_eval; print(f'lm-eval: {lm_eval.__version__}')"

# Try using vLLM for 5-10x speedup (highly recommended for A100/H100)
# To use this, you must have installed it manually: pip install vllm
echo "Checking for vLLM..."
VLLM_ARG=""
if [ "${FORCE_HF_BACKEND:-0}" = "1" ]; then
    echo "FORCE_HF_BACKEND=1 set: using Transformers backend (no vLLM)."
elif $PYTHON_BIN -c "import vllm" 2>/dev/null; then
    echo "✓ Using vLLM backend"
    VLLM_ARG="--use_vllm"
else
    echo "⚠ vLLM not found or incompatible. Using standard Transformers backend."
    echo "  (To enable vLLM, ensure 'vllm' is installed in your env and compatible with your CUDA version)"
fi

echo "================================"
echo "Running All Benchmarks"
echo "================================"

# IMPORTANT: For gated models like Llama 3.1, ensure HF_TOKEN is set
# export HF_TOKEN="your_token_here"

# Required for HumanEval/MBPP coding benchmarks
export HF_ALLOW_CODE_EVAL=1

# ENSURE CUDA MULTIPROCESSING USES SPAWN
# This is critical to prevent "Cannot re-initialize CUDA in forked subprocess"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_MODULE_LOADING=LAZY

# We use the environment's python directly
echo "DEBUG: Launching run_all_benchmarks.py via $PYTHON_BIN"
PYTHONUNBUFFERED=1 $PYTHON_BIN src/evaluation/run_all_benchmarks.py \
  --output_dir "results/eval_${SLURM_JOB_ID}" \
  --batch_size "$BATCH_SIZE" \
  ${LIMIT:+--limit "$LIMIT"} \
  --verbose \
  $VLLM_ARG \
  "$@"
echo "DEBUG: run_all_benchmarks.py command completed with exit code $?"

echo "================================"
echo "Evaluation finished at: $(date)"
echo "Results saved to: results/eval_${SLURM_JOB_ID}"
echo "================================"
