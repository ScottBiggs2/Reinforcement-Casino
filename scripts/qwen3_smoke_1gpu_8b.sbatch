#!/bin/bash
# SMOKE: validate the device_map="auto" multi-GPU sparse path (改动 B+C) on 8B
# BEFORE committing any 32B compute. Runs 8B sparse DPO with device_map="auto"
# across 2 H200 for 20 steps, identical config to the completed single-GPU run
# (job 7362097) so the per-step loss/margin curve is directly comparable.
# Success = (1) no cross-device kernel error, (2) loss curve matches 7362097's
# first ~20 steps within fp tolerance.
#SBATCH --job-name=qwen3_smoke_1gpu
#SBATCH --output=logs/qwen3_smoke_1gpu_%j.out
#SBATCH --error=logs/qwen3_smoke_1gpu_%j.err
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

MODEL_BASE="Qwen/Qwen3-8B"
MASK="/scratch/xie.yiyi/transfer_v1/oracle_masks_qwen3_8b/oracle_dpo_light_r1_step500_sp97.5.pt"
OUT_BASE="/scratch/xie.yiyi/transfer_v1/smoke_1gpu_8b"
mkdir -p "$OUT_BASE"

if [ ! -f "$MASK" ]; then echo "FATAL: mask not found: $MASK"; exit 1; fi

echo "=========================================="
echo "SMOKE: device_map=auto sparse DPO  |  $MODEL_BASE  |  light-r1  |  1x H200 baseline"
echo "Job ID : ${SLURM_JOB_ID:-local}   GPUs: $(nvidia-smi -L | wc -l)   Started: $(date)"
echo "Config matches single-GPU job 7362097: bs2 x ga64, lr5e-7, beta0.1, 20 steps"
echo "Compare loss/margin vs: logs/qwen3_p3_sparse_dpo_7362097.out (first 20 steps)"
echo "=========================================="

python src/full_training/sparse_dpo_efficiency.py \
    --model_name "$MODEL_BASE" \
    --dataset light-r1 \
    --mask "$MASK" \
    --n_steps 20 \
    --batch_size 2 \
    --grad_accum 64 \
    --lr 5e-7 \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --max_length 1024 \
    --max_prompt_length 1024 \
    --dpo_beta 0.1 \
    --optimizer sparse_adamw \
    --gradient_checkpointing \
    --save_model false \
    --output_base_dir "$OUT_BASE" \
    --run_name "smoke_1gpu_8b_lightr1_20steps"

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
