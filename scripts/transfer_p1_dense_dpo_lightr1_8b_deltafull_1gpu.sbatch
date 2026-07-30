#!/bin/bash
# Dense DPO on Light-R1 (Llama-3.1-8B-Instruct), Scott spec, deltas through step 500,
# on a SINGLE GPU.
#
# Why 1 GPU instead of the 4-GPU DDP version: on 2026-07-27 all 32 h200 GPUs in
# multigpu were allocated except 2 on d4055, so a 4-GPU single-node request could not
# be placed. The four jobs ahead in the queue were each blocked on something other
# than capacity (QOSMaxGRESPerUser, Dependency, drained nodes), so the wait was on
# hardware, not priority.
#
# The optimisation is IDENTICAL to the 4-GPU run, not an approximation:
#   4 GPU: per_device 2 x grad_accum 16 x 4 = effective batch 128
#   1 GPU: per_device 2 x grad_accum 64 x 1 = effective batch 128
# lr, beta, warmup_ratio, num_steps, max_length are unchanged, so this is still
# Scott spec. Same shape as the sparse arms in transfer_p3_sparse_dpo_*.sbatch
# (batch 2 x accum 64 on one GPU), which completed 500 steps in 5:34 and 5:54.
#
# Cost is wall-clock only: ~1:45 on 4 GPUs (job 6366610) -> ~6 h here. Hence 8 h
# walltime (the cluster hard cap) plus --requeue and auto-resume.
#SBATCH --job-name=p1_dpo_lightr1_deltafull_1gpu
#SBATCH --output=logs/p1_dpo_lightr1_deltafull_1gpu_%j.out
#SBATCH --error=logs/p1_dpo_lightr1_deltafull_1gpu_%j.err
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
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

OUT_BASE="/scratch/xie.yiyi/transfer_v1/dense_dpo_lightr1_llama8b"
mkdir -p "$OUT_BASE"

NGPUS=$(nvidia-smi -L | wc -l)

echo "=========================================="
echo "Phase 1A (rebuild, 1 GPU): Dense DPO | Llama-3.1-8B-Instruct | Light-R1"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : ${SLURMD_NODENAME:-$(hostname)}"
echo "Started: $(date)"
echo "=========================================="
echo "Visible GPUs : $NGPUS  (expected 1)"
echo "Spec         : 500 steps, eff_batch 128 (per_device 2 x grad_accum 64 x 1 GPU)"
echo "               lr 5e-7, beta 0.1, warmup_ratio 0.1, max_length 1024"
echo "Deltas       : every 50 steps through step 500 (10 files x ~30 GB)"
echo "Expect       : ~6 h (4-GPU reference job 6366610 was 01:45:26)"
echo "=========================================="

torchrun --standalone --nproc_per_node=1 \
  src/full_training/DPO_train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset light-r1 \
    --num_steps 500 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.1 \
    --dpo_beta 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --max_length 1024 \
    --max_prompt_length 1024 \
    --save_steps 25 \
    --save_total_limit 30 \
    --delta_log_interval 50 \
    --delta_log_end_step 500 \
    --output_base_dir "$OUT_BASE" \
    --resume_from_checkpoint auto \
    --use_wandb \
    --run_name dense_dpo_llama8b_lightr1_scott_full_deltafull_1gpu

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "--- checkpoints ---"
ls -d "$OUT_BASE"/checkpoints/*/checkpoint-* 2>/dev/null | sort -V | tail -5
echo "--- deltas (these are what make future rho sweeps free) ---"
ls -lh "$OUT_BASE"/deltas/*/ 2>/dev/null | tail -12
echo "=========================================="
