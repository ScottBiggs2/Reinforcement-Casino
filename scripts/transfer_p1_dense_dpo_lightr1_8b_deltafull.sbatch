#!/bin/bash
# Dense DPO on Light-R1 (Llama-3.1-8B-Instruct), Scott spec, WITH deltas logged
# through step 500.
#
# Why this exists instead of transfer_p1_dense_dpo_lightr1_8b.sbatch:
# that run used the default --delta_log_end_step, which resolves to
# min(num_steps, max(interval, num_steps//10)) = 50. So it saved deltas_step_50
# and nothing else. When the checkpoints were later deleted (between 2026-05-27,
# when job 7050891 still loaded checkpoint-500, and 2026-07-25) the |dtheta| score
# tensor at step 500 went with them, and the surviving oracle masks are bool --
# they cannot be re-thresholded to any other sparsity. Logging deltas out to 500
# makes every future rho a CPU job on a 30 GB file instead of a retrain.
#SBATCH --job-name=transfer_p1_dpo_lightr1_deltafull
#SBATCH --output=logs/transfer_p1_dpo_lightr1_deltafull_%j.out
#SBATCH --error=logs/transfer_p1_dpo_lightr1_deltafull_%j.err
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
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
echo "Phase 1A (rebuild): Dense DPO | Llama-3.1-8B-Instruct | Light-R1"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=========================================="
echo "Visible GPUs : $NGPUS"
echo "Spec         : 500 steps, eff_batch 128 (per_device 2 x grad_accum 16 x ${NGPUS} GPU)"
echo "               lr 5e-7, beta 0.1, warmup_ratio 0.1, max_length 1024"
echo "Reference run: job 6366610, 2026-04-27, elapsed 01:45:26 (same spec, deltas to 50 only)"
echo "Deltas       : every 50 steps through step 500 (10 files x ~30 GB)"
echo "=========================================="

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
    --delta_log_interval 50 \
    --delta_log_end_step 500 \
    --output_base_dir "$OUT_BASE" \
    --resume_from_checkpoint auto \
    --use_wandb \
    --run_name dense_dpo_llama8b_lightr1_scott_full_deltafull

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "--- checkpoints ---"
ls -d "$OUT_BASE"/checkpoints/*/checkpoint-* 2>/dev/null | sort -V | tail -5
echo "--- deltas (these are what make future rho sweeps free) ---"
ls -lh "$OUT_BASE"/deltas/*/ 2>/dev/null | tail -12
echo "=========================================="
