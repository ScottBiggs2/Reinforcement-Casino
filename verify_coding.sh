#!/bin/bash
# Slurm job script to verify the coding evaluation fix
# Usage: sbatch verify_coding.sh --model_path <HUGGINGFACE_ID_OR_PATH> [other options]

#SBATCH --job-name=verify_coding
#SBATCH --output=logs/verify_coding_%j.out
#SBATCH --error=logs/verify_coding_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# Exit on any error
set -e

# Job runs in the directory you submitted from
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results/verify_coding

echo "Verification started at: $(date)"
echo "Model Path: $1"

# Check for arguments
if [ -z "$1" ]; then
    echo "ERROR: Arguments are required (e.g. --model_path <PATH>)."
    exit 1
fi

ENV_PATH="/scratch/biggs.s/conda_envs/rl_casino"
PYTHON_BIN="$ENV_PATH/bin/python"
export PATH="$ENV_PATH/bin:$PATH"

# Required for HumanEval/MBPP coding benchmarks
export HF_ALLOW_CODE_EVAL=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_MODULE_LOADING=LAZY

echo "Running coding verification with limit=10..."
$PYTHON_BIN src/evaluation/run_all_benchmarks.py \
  --benchmarks coding \
  --limit 10 \
  --use_vllm \
  --verbose \
  --output_dir "results/verify_coding_${SLURM_JOB_ID}" \
  "$@"

echo "Verification finished at: $(date)"
echo "Results available in: results/verify_coding_${SLURM_JOB_ID}"
