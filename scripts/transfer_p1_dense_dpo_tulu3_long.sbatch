#!/bin/bash
#SBATCH --job-name=transfer_p1_dpo_tulu3_long
#SBATCH --output=logs/transfer_p1_dpo_tulu3_long_%j.out
#SBATCH --error=logs/transfer_p1_dpo_tulu3_long_%j.err
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --requeue

set -euo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="/home/xie.yiyi/.conda/envs/rl_casino"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

cd /home/xie.yiyi/rc-sparse-speed
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="/scratch/xie.yiyi/hf_cache"
export HF_DATASETS_CACHE="/scratch/xie.yiyi/hf_cache/datasets"
export WANDB_PROJECT="rl_casino_transfer_v1"

OUT_BASE="/scratch/xie.yiyi/transfer_v1/dense_dpo_tulu3_long_llama8b"
mkdir -p "$OUT_BASE"

NGPUS=$(nvidia-smi -L | wc -l)
echo "Visible GPUs: $NGPUS"
echo "Tulu3 long (5000 steps) for Mukherjee persistence check + better-trained oracle"

torchrun --standalone --nproc_per_node="${NGPUS}" \
  src/full_training/DPO_train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset tulu3 \
    --num_steps 5000 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.1 \
    --dpo_beta 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --max_length 1024 \
    --max_prompt_length 1024 \
    --save_steps 100 \
    --save_total_limit 30 \
    --output_base_dir "$OUT_BASE" \
    --resume_from_checkpoint auto \
    --use_wandb \
    --run_name dense_dpo_llama8b_tulu3_long_5000steps
