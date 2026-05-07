#!/bin/bash
#SBATCH --job-name=transfer_p3_random_mask
#SBATCH --output=logs/transfer_p3_random_mask_%j.out
#SBATCH --error=logs/transfer_p3_random_mask_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00

set -euo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="/home/xie.yiyi/.conda/envs/rl_casino"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

cd /home/xie.yiyi/rc-sparse-speed
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="/scratch/xie.yiyi/hf_cache"

OUT="/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/random_baseline_lightr1_sp97.5_seed42.pt"
mkdir -p "$(dirname "$OUT")"

if [ -f "$OUT" ]; then
  echo "Random mask already exists: $OUT — skipping"
  ls -lh "$OUT"
  exit 0
fi

echo "Generating random mask: $OUT"
python src/utils/generate_random_mask.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --sparsity_percent 97.5 \
    --min_layer_keep_ratio 0.0025 \
    --seed 42 \
    --output_file "$OUT"

ls -lh "$OUT"
