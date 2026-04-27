#!/bin/bash
#SBATCH --job-name=transfer_p1_dpo_lightr1
#SBATCH --output=logs/transfer_p1_dpo_lightr1_%j.out
#SBATCH --error=logs/transfer_p1_dpo_lightr1_%j.err
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

OUT_BASE="/scratch/xie.yiyi/transfer_v1/dense_dpo_lightr1_llama8b"
mkdir -p "$OUT_BASE"

NGPUS=$(nvidia-smi -L | wc -l)
echo "Visible GPUs: $NGPUS"

echo "=========================================="
echo "Phase 1A: Dense DPO  |  Llama-3.1-8B-Instruct  |  Light-R1"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="
echo "Scott full spec: 5000 steps × eff_batch 128 (per_device 2 × grad_accum 8 × ${NGPUS} GPU)"
echo "Light-R1 has 2970 pairs → 1 epoch ≈ 23 steps. 5000 steps = ~217 epochs."
echo "  Save every 25 steps, total_limit=20 to capture early ckpts (step 25-500) as oracle source."
echo "  Pick best ckpt by inspecting trainer_state.json (NOT necessarily final)."
echo ""

torchrun --standalone --nproc_per_node="${NGPUS}" \
  src/full_training/DPO_train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset light-r1 \
    --num_steps 500 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.1 \
    --dpo_beta 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --max_length 1024 \
    --max_prompt_length 1024 \
    --save_steps 25 \
    --save_total_limit 30 \
    --output_base_dir "$OUT_BASE" \
    --resume_from_checkpoint auto \
    --use_wandb \
    --run_name dense_dpo_llama8b_lightr1_scott_full

echo ""
echo "=========================================="
echo "Finished: $(date)"
ls -la "$OUT_BASE" 2>&1 | tail -5
echo "=========================================="
