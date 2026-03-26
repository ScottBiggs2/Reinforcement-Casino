#!/bin/bash
# Slurm job to run the attention masking diagnostic
# Usage: sbatch scripts/run_mask_diagnostics.sh

#SBATCH --job-name=mask_diag
#SBATCH --output=logs/mask_diag_%j.out
#SBATCH --error=logs/mask_diag_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00

ENV_PATH="/scratch/biggs.s/conda_envs/rl_casino"
PYTHON_BIN="$ENV_PATH/bin/python"
export PATH="$ENV_PATH/bin:$PATH"

# Slurm copies batch scripts to spool — use submit directory as repo root.
# Submit from repo:  cd .../rl_casino && sbatch scripts/run_mask_diagnostics.sh
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working dir: $(pwd)"

export PYTHONPATH=.
echo "Installing/verifying training requirements..."
if "$PYTHON_BIN" -c "import trl" 2>/dev/null; then
    echo "Training requirements already satisfied; skipping pip install."
else
    "$PYTHON_BIN" -m pip install -r requirements.txt -q
fi

echo "Running Attention Masking Diagnostic..."
$PYTHON_BIN diagnose_attention_masking.py
