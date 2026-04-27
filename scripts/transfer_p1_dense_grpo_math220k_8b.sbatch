#!/bin/bash
#SBATCH --job-name=transfer_p1_grpo_math
#SBATCH --output=logs/transfer_p1_grpo_math_%j.out
#SBATCH --error=logs/transfer_p1_grpo_math_%j.err
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
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
echo "Visible GPUs: $NGPUS"

echo "=========================================="
echo "Phase 1C: Dense GRPO  |  Llama-3.1-8B-Instruct  |  math-220k (open-r1/OpenR1-Math-220k)"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="
echo "Scott full spec: 5000 steps; LLAMA_COT_PROMPT_SUFFIX fix in place (smoke verified PASS)."
echo "  Per-device 8 × ${NGPUS} GPU = 8×${NGPUS} completions per step (with num_generations=8)."
echo "  beta=0.025 (script default + grpo_openr1_llama31_slurm spec, NOT orchestrator's 1.0)."
echo ""

torchrun --standalone --nproc_per_node="${NGPUS}" \
  src/full_training/GRPO_train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset math-220k \
    --num_steps 500 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_generations 8 \
    --generation_batch_size 8 \
    --learning_rate 5e-6 \
    --beta 0.025 \
    --warmup_ratio 0.1 \
    --max_prompt_length 512 \
    --max_completion_length 2048 \
    --precision bf16 \
    --save_steps 50 \
    --save_total_limit 8 \
    --grpo_reward_profile llama_cot \
    --output_base_dir "$OUT_BASE" \
    --resume_from_checkpoint auto \
    --use_wandb \
    --run_slug llama8b_math220k_dense_scott_full

echo ""
echo "=========================================="
echo "Finished: $(date)"
ls -la "$OUT_BASE" 2>&1 | tail -5
echo "=========================================="
