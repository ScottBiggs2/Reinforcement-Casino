#!/bin/bash
# One-off diagnostic: humaneval under chat template, 8 docs, dump raw+filtered
# responses. Same env stack as aicr_eval_tulu3_suite.sbatch.
#SBATCH --job-name=diag_humaneval
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:b200:1
#SBATCH --time=00:30:00
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/diag_humaneval_%j.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh
W=/work/neu/p2026_0038_neu/xie_yiyi
conda deactivate || true
conda activate "$W/envs/rl_casino_eval"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
CUDA_ROOT=/apps/aicr/packages/cuda/13.1.1/extrybz
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$LD_LIBRARY_PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1 VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_MODULE_LOADING=LAZY

cd "$W/rc-sparse-speed"
PYTHONUNBUFFERED=1 python scripts/diag_humaneval_chat.py
