#!/bin/bash
# Phase 8.5: Run baseline and entropy-MCTS with execution-lite reward. WandB enabled.
# Run from project dir: sbatch run_experiment_2.sh
#SBATCH --job-name=entropy_grpo_8.5
#SBATCH --output=logs/run_experiment_2_%j.out
#SBATCH --error=logs/run_experiment_2_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Job runs in the directory you submitted from (SLURM_SUBMIT_DIR)
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs checkpoints

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working dir: $(pwd)"

# Conda in batch: source conda.sh so activate works
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate EntropyTreeGRPO_env

echo "================================"
echo "Installing requirements..."
echo "================================"
pip install -r requirements.txt -q
if [ $? -ne 0 ]; then echo "ERROR: pip install failed"; exit 1; fi

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

RUN_NAME="grpo_${SLURM_JOB_ID:-local}_8.5"
CHECKPOINT_DIR="$(pwd)/checkpoints"
MAX_TREE_NODES=48
BRANCH_WIDTH=3
STEPS_PER_EXPANSION=16
MAX_NEW_TOKENS=256
NUM_BASELINE_SAMPLES=4
NUM_EPOCHS=25
SAVE_EVERY_STEPS=100
EXEC_TIMEOUT=2.25

echo "================================"
echo "Run 1/2: Baseline GRPO (execution-lite)"
echo "================================"
python scripts/run_experiment_2.py \
  --method baseline \
  --num_epochs "$NUM_EPOCHS" \
  --run_name "${RUN_NAME}_baseline" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --wandb_project entropy-tree-grpo \
  --save_every_steps "$SAVE_EVERY_STEPS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_baseline_samples "$NUM_BASELINE_SAMPLES" \
  --exec_timeout "$EXEC_TIMEOUT" \
  --device cuda

echo "================================"
echo "Run 2/2: Entropy-MCTS GRPO (execution-lite)"
echo "================================"
python scripts/run_experiment_2.py \
  --method entropy_mcts \
  --num_epochs "$NUM_EPOCHS" \
  --run_name "${RUN_NAME}_mcts" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --wandb_project entropy-tree-grpo \
  --save_every_steps "$SAVE_EVERY_STEPS" \
  --max_tree_nodes "$MAX_TREE_NODES" \
  --branch_width "$BRANCH_WIDTH" \
  --steps_per_expansion "$STEPS_PER_EXPANSION" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_baseline_samples "$NUM_BASELINE_SAMPLES" \
  --exec_timeout "$EXEC_TIMEOUT" \
  --device cuda

echo "================================"
echo "Job finished at: $(date)"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "================================"
