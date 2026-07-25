#!/bin/bash
# Pre-download Qwen/Qwen3-32B to the shared HF cache on a CPU node (no GPU, so no
# idle-GPU flag) BEFORE the GPU training jobs, so they load from cache in ~1-2 min
# instead of spending 15-20 min downloading on an idle GPU (which trips RC's
# 15-min idle auto-cancel). See AGENT_PLAYBOOK.md §7b.
#SBATCH --job-name=qwen3_32b_precache
#SBATCH --output=logs/qwen3_32b_precache_%j.out
#SBATCH --error=logs/qwen3_32b_precache_%j.err
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -euo pipefail
CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="/home/xie.yiyi/.conda/envs/rl_casino"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi
export HF_HOME="/scratch/xie.yiyi/hf_cache"
echo "Pre-caching Qwen/Qwen3-32B to $HF_HOME  | $(date)"
huggingface-cli download Qwen/Qwen3-32B --exclude "*.gguf" "original/*"
echo "Done: $(date)"
du -sh /scratch/xie.yiyi/hf_cache/hub/models--Qwen--Qwen3-32B 2>/dev/null
