#!/bin/bash
# LoRA PEFT baseline for NeurIPS #29841 rebuttal (AC item 2 / reviewer bf6h W3).
#
# Answers: "the abstract motivates this as an alternative to low-rank PEFT but there
# is no PEFT baseline." Everything except the parameterisation is held identical to
# the dense/sparse DPO runs (Table 7 of the submission): Llama-3.1-8B-Instruct,
# light-r1, 500 steps, eff. batch 128 (bs 2 x ga 64), beta 0.1, warmup 0.1,
# max_len 1024/1024, bf16, adamw_8bit, 1x H200.
#
# The ONE deliberate difference: LoRA gets a tuned LR (1e-4 / 5e-5) because a
# zero-initialised adapter does not move at the full-FT 5e-7. This must be stated in
# the rebuttal — it makes the comparison favourable to LoRA, not to us.
#
# Array arms:
#   0  r=64, lr 1e-4  -> ~168M trainable (2.1%), the matched-budget arm vs 97.5% sparse (2.5%)
#   1  r=16, lr 1e-4  -> ~42M  trainable (0.5%), standard-practice point
#   2  r=16, lr 5e-5  -> LR sensitivity control for arm 1
#
# Reports (efficiency_summary.json per arm): trainable params, s/step, peak VRAM,
# final loss. Reward margin / preference accuracy come from the trainer logs.
#SBATCH --job-name=lora_baseline_8b
#SBATCH --output=logs/lora_baseline_8b_%A_%a.out
#SBATCH --error=logs/lora_baseline_8b_%A_%a.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=0-2

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

MODEL_BASE="meta-llama/Llama-3.1-8B-Instruct"
OUT_BASE="/scratch/xie.yiyi/rebuttal_lora"
mkdir -p "$OUT_BASE"

RANKS=(64 16 16)
LRS=(1e-4 1e-4 5e-5)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
R="${RANKS[$IDX]}"
LR="${LRS[$IDX]}"
TAG="lora_r${R}_lr${LR}"

echo "=========================================="
echo "LoRA baseline arm $IDX  |  r=$R  |  lr=$LR  |  500 steps  |  light-r1"
echo "Job ID : ${SLURM_JOB_ID:-local}   Started: $(date)"
echo "=========================================="

# --resume_from_checkpoint auto + --requeue: the school hard-caps walltime at 8h.
python src/full_training/DPO_train.py \
    --model_name "$MODEL_BASE" \
    --dataset light-r1 \
    --lora \
    --lora_r "$R" \
    --learning_rate "$LR" \
    --num_steps 500 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --max_length 1024 \
    --max_prompt_length 1024 \
    --dpo_beta 0.1 \
    --optim adamw_8bit \
    --gradient_checkpointing \
    --save_steps 100 \
    --save_total_limit 3 \
    --resume_from_checkpoint auto \
    --output_base_dir "$OUT_BASE" \
    --use_wandb \
    --run_name "rebuttal_${TAG}_lightr1_500steps"

echo ""
echo "=========================================="
echo "Finished arm $IDX ($TAG): $(date)"
echo "=========================================="
