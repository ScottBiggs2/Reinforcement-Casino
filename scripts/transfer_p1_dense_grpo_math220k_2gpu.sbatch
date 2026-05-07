#!/bin/bash
#SBATCH --job-name=transfer_p1_grpo_math_2g
#SBATCH --output=logs/transfer_p1_grpo_math_2g_%j.out
#SBATCH --error=logs/transfer_p1_grpo_math_2g_%j.err
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
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

OUT_BASE="/scratch/xie.yiyi/transfer_v1/dense_grpo_math220k_llama8b"
mkdir -p "$OUT_BASE"

NGPUS=$(nvidia-smi -L | wc -l)
echo "Visible GPUs: $NGPUS  (variant: 2-GPU multigpu)"
GLOBAL_BATCH=$((8 * NGPUS))
GEN_BATCH=$GLOBAL_BATCH
echo "global_batch=$GLOBAL_BATCH  generation_batch_size=$GEN_BATCH"

torchrun --standalone --nproc_per_node="${NGPUS}" \
  src/full_training/GRPO_train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset math-220k \
    --num_steps 5000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_generations 8 \
    --generation_batch_size 8 \
    --learning_rate 5e-6 \
    --beta 0.025 \
    --warmup_ratio 0.1 \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
    --precision bf16 \
    --save_steps 50 \
    --save_total_limit 8 \
    --grpo_reward_profile llama_cot \
    --output_base_dir "$OUT_BASE" \
    --resume_from_checkpoint auto \
    --use_wandb \
    --run_slug llama8b_math220k_dense_scott_full
