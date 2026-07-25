#!/bin/bash
# SMOKE: validate the FSDP dense path (改动 A) on 8B before committing 32B compute.
# FSDP full_shard via torchrun across 2 H200, 20 steps. Success = no FSDP/ref
# wrap error, GPUs busy (no idle), loss curve sane.
#SBATCH --job-name=qwen3_smoke_fsdp
#SBATCH --output=logs/qwen3_smoke_fsdp_%j.out
#SBATCH --error=logs/qwen3_smoke_fsdp_%j.err
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
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
export WANDB_MODE=disabled WANDB_DISABLED=true
export PYTORCH_ALLOC_CONF=expandable_segments:True

NGPUS=$(nvidia-smi -L | wc -l)
echo "FSDP smoke | Qwen3-8B | light-r1 | ${NGPUS}x H200 | $(date)"

torchrun --standalone --nproc_per_node="${NGPUS}" \
  src/full_training/DPO_train.py \
    --model_name Qwen/Qwen3-8B \
    --dataset light-r1 \
    --fsdp \
    --fsdp_layer_cls Qwen3DecoderLayer \
    --num_steps 20 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.1 \
    --dpo_beta 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --max_length 1024 \
    --max_prompt_length 1024 \
    --optim adamw_8bit \
    --gradient_checkpointing \
    --save_steps 10 \
    --save_total_limit 2 \
    --output_base_dir /scratch/xie.yiyi/transfer_v1/smoke_fsdp_8b \
    --run_name smoke_fsdp_8b_lightr1_20steps
echo "Finished: $(date)"
