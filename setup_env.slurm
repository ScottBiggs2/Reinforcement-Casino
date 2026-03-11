#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup_env_%j.out
#SBATCH --error=logs/setup_env_%j.err

echo "========================================"
echo "ENV SETUP Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

CONDA_BASE=/shared/EL9/explorer/miniconda3/24.11.1/miniconda3
ENV_PREFIX=/scratch/xie.yiyi/conda_envs/rl_casino
REPO=/home/xie.yiyi/Reinforcement-Casino

source $CONDA_BASE/etc/profile.d/conda.sh

# --- Remove broken env ---
echo ""
echo "=== Removing broken env ==="
conda deactivate 2>/dev/null || true
rm -rf $ENV_PREFIX
echo "Done."

# --- Create fresh env with Python 3.11.14 ---
echo ""
echo "=== Creating fresh conda env (Python 3.11.14) ==="
conda create --prefix $ENV_PREFIX python=3.11.14 -y
echo "Done."

# --- Activate ---
conda activate $ENV_PREFIX
echo "Python: $(python --version)"

# --- Install requirements ---
echo ""
echo "=== Installing requirements ==="
cd $REPO
pip install -r requirements.txt
pip install scikit-learn   # needed for CAV cold-start probes

echo ""
echo "=== Verifying installs ==="
python -c "
import torch, transformers, trl, sklearn, triton
print('torch       :', torch.__version__)
print('transformers:', transformers.__version__)
print('trl         :', trl.__version__)
print('sklearn     :', sklearn.__version__)
print('triton      :', triton.__version__)
print('CUDA avail  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU         :', torch.cuda.get_device_name(0))
"

echo ""
echo "========================================"
echo "Setup complete: $(date)"
echo "========================================"
