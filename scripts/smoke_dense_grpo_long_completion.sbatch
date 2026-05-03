#!/bin/bash
# Single-knob diagnostic: does lifting max_completion_length 1024 -> 3072 unblock
# accuracy_reward in dense GRPO 8B on math-220k? All other hparams match
# dense_scott_full (lr=5e-6, beta=0.025, batch=2, accum=4, num_gen=8, gen_batch=8,
# llama_cot reward). 100 steps. New OUT_BASE -- does not touch long run.
#SBATCH --job-name=smoke_grpo_lc3072
#SBATCH --output=logs/smoke_grpo_lc3072_%j.out
#SBATCH --error=logs/smoke_grpo_lc3072_%j.err
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
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

OUT_BASE="/scratch/xie.yiyi/transfer_v1/dense_grpo_math220k_smoke_lc3072"
mkdir -p "$OUT_BASE"

NGPUS=$(nvidia-smi -L | wc -l)
echo "=========================================="
echo "SMOKE: dense GRPO  |  Llama-3.1-8B-Instruct  |  math-220k"
echo "Single knob: max_completion_length 1024 -> 3072"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "Visible GPUs: $NGPUS"
echo "=========================================="

torchrun --standalone --nproc_per_node="${NGPUS}" \
  src/full_training/GRPO_train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset math-220k \
    --num_steps 100 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_generations 8 \
    --generation_batch_size 8 \
    --learning_rate 5e-6 \
    --beta 0.025 \
    --warmup_ratio 0.1 \
    --max_prompt_length 512 \
    --max_completion_length 3072 \
    --precision bf16 \
    --save_steps 25 \
    --save_total_limit 4 \
    --grpo_reward_profile llama_cot \
    --output_base_dir "$OUT_BASE" \
    --resume_from_checkpoint auto \
    --use_wandb \
    --run_slug llama8b_math220k_dense_smoke_lc3072

echo ""
echo "=========================================="
echo "Finished: $(date)"
ls -la "$OUT_BASE" 2>&1 | tail -5
echo "=========================================="
