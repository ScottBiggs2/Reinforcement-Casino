#!/bin/bash
# SMOKE: validate the --lora path before spending real compute on the rebuttal runs.
# 20 steps, Llama-3.1-8B-Instruct + light-r1, LoRA r=16 @ lr 1e-4.
#
# This exists to catch the two silent failure modes, not to produce a number:
#   (1) adapter gets no gradient under gradient checkpointing -> loss flat, margin ~0.
#       Guarded by enable_input_require_grads(); this run proves the guard works.
#   (2) trainable-param count is not what we think it is -> the whole matched-budget
#       comparison is wrong. The run prints it; check it before launching r=64.
#
# PASS if: trainable params ≈ 41.9M (0.52%) for r=16, loss moves off its step-1 value,
#          and rewards/margins is non-zero by step 20.
#SBATCH --job-name=lora_smoke_8b
#SBATCH --output=logs/lora_smoke_8b_%j.out
#SBATCH --error=logs/lora_smoke_8b_%j.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=12
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

MODEL_BASE="meta-llama/Llama-3.1-8B-Instruct"
OUT_BASE="/scratch/xie.yiyi/rebuttal_lora/smoke"
mkdir -p "$OUT_BASE"

python -c "import peft, trl; print('peft', peft.__version__, '| trl', trl.__version__)"

echo "=========================================="
echo "SMOKE: LoRA DPO  |  $MODEL_BASE  |  light-r1  |  r=16  |  lr 1e-4  |  20 steps"
echo "Job ID : ${SLURM_JOB_ID:-local}   GPUs: $(nvidia-smi -L | wc -l)   Started: $(date)"
echo "=========================================="

python src/full_training/DPO_train.py \
    --model_name "$MODEL_BASE" \
    --dataset light-r1 \
    --lora \
    --lora_r 16 \
    --learning_rate 1e-4 \
    --num_steps 20 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --max_length 1024 \
    --max_prompt_length 1024 \
    --dpo_beta 0.1 \
    --optim adamw_8bit \
    --gradient_checkpointing \
    --output_base_dir "$OUT_BASE" \
    --run_name "lora_smoke_8b_lightr1_r16_20steps"

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "Check: trainable params line, then efficiency_summary.json under $OUT_BASE"
echo "=========================================="
