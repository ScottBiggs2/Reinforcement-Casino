#!/bin/bash
#SBATCH --job-name=cold_start_exp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/cold_start_%j.out
#SBATCH --error=logs/cold_start_%j.err

# -------------------------------------------------------
# Cold-Start Mask Experiment
# Generates CAV + SNIP masks (no training),
# then trains DPO for 10 steps with each mask vs dense baseline.
# -------------------------------------------------------

echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "========================================"

# --- Activate conda env ---
source /shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/xie.yiyi/conda_envs/rl_casino

echo "Python   : $(python --version)"
echo "Torch    : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA     : $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"
echo ""

# --- Working directory ---
cd /home/xie.yiyi/Reinforcement-Casino
mkdir -p logs

# --- Run experiment ---
python run_cold_start_experiment.py

echo ""
echo "========================================"
echo "Finished : $(date)"
echo "========================================"
