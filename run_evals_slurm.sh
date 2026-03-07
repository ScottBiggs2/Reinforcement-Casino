#!/bin/bash
# Slurm job script to run the full evaluation suite
# Usage: sbatch run_evals_slurm.sh --model_path <HUGGINGFACE_ID_OR_PATH>

#SBATCH --job-name=llm_eval_suite
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Exit on any error
set -e

# Job runs in the directory you submitted from
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results

echo "Evaluation started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

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
ENV_PATH="/scratch/biggs.s/conda_envs/rl_casino"
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
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Install/Verify lm-eval inside the env
$PYTHON_BIN -m pip install -r requirements.txt -q
bash install_lm_eval.sh
$PYTHON_BIN -c "import lm_eval; print(f'lm-eval: {lm_eval.__version__}')"

echo "================================"
echo "Running All Benchmarks"
echo "================================"

# IMPORTANT: For gated models like Llama 3.1, ensure HF_TOKEN is set
# If not already in your environment, uncomment and add it here:
# export HF_TOKEN="your_token_here"

# We use the environment's python directly
$PYTHON_BIN src/evaluation/run_all_benchmarks.py \
  --output_dir "results/eval_${SLURM_JOB_ID}" \
  --verbose \
  "$@"

echo "================================"
echo "Evaluation finished at: $(date)"
echo "Results saved to: results/eval_${SLURM_JOB_ID}"
echo "================================"
