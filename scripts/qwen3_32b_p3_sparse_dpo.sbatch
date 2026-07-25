#!/bin/bash
# Phase 3 (Qwen3-32B): Sparse DPO (high-sparsity treatment) via device_map="auto"
# across 8 H200. Uses the validated multi-GPU sparse path (改动 B+C): per-param
# mask colocation + explicit ref model. Single process, NOT torchrun.
# Usage: DATASET=light-r1 MASK_PATH=<...pt> RUN_TAG=oracle_step500 sbatch qwen3_32b_p3_sparse_dpo.sbatch
#SBATCH --job-name=qwen3_32b_p3_sparse
#SBATCH --output=logs/qwen3_32b_p3_sparse_%j.out
#SBATCH --error=logs/qwen3_32b_p3_sparse_%j.err
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:3
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
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

MODEL_BASE="Qwen/Qwen3-32B"
DATASET="${DATASET:?Set DATASET=light-r1 | tulu3}"
DS_TAG=$(echo "$DATASET" | tr '-' '_')
: "${MASK_PATH:?Set MASK_PATH=/scratch/.../oracle_dpo_${DS_TAG}_step500_sp97.5.pt}"
: "${RUN_TAG:?Set RUN_TAG=oracle_step500 | oracle_step150}"

OUT_BASE="/scratch/xie.yiyi/transfer_v1/sparse_dpo_${DS_TAG}_qwen3_32b_${RUN_TAG}"
mkdir -p "$OUT_BASE"
if [ ! -f "$MASK_PATH" ]; then echo "FATAL: MASK_PATH not found: $MASK_PATH"; exit 1; fi

echo "=========================================="
echo "Phase 3 (Qwen3-32B): Sparse DPO  |  $MODEL_BASE  |  $DATASET  |  device_map=auto x3 H200"
echo "Job ID : ${SLURM_JOB_ID:-local}   GPUs: $(nvidia-smi -L | wc -l)   Started: $(date)"
echo "RUN_TAG=$RUN_TAG  MASK_PATH=$MASK_PATH"
echo "Hparams: 500 steps, eff_batch 128 (bs2 x ga64), lr=5e-7, beta=0.1, optimizer=sparse_adamw"
echo "=========================================="

# GPU keep-alive sidecar: naive MP (device_map) keeps per-GPU util low → RC's
# idle monitor can false-positive cancel a genuinely-running job (§7b). Tiny
# periodic matmul on each GPU keeps it visibly active. Killed on exit.
python -c "import torch
n=torch.cuda.device_count()
gs=[torch.randn(2048,2048,device=f'cuda:{i}') for i in range(n)]
while True:
    for _ in range(20):
        for g in gs: g.matmul(g)
    torch.cuda.synchronize()" &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null' EXIT
echo "keep-alive sidecar PID $KEEPALIVE on $(nvidia-smi -L | wc -l) GPUs"

# Self-resubmitting requeue chain (see p1 for rationale): timeout ~30m before the
# 8h wall so the tail runs and advance.sh resubmits this phase (resume auto) until
# checkpoint-500 exists. save_steps=10 keeps progress visible to the stall guard.
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-27000}"   # 7.5h of the 8h wall
# plain python single process — device_map=auto handles multi-GPU. Do NOT use torchrun.
set +e
timeout --signal=TERM "${TRAIN_TIMEOUT}s" \
python src/full_training/sparse_dpo_efficiency.py \
    --model_name "$MODEL_BASE" \
    --dataset "$DATASET" \
    --mask "$MASK_PATH" \
    --device_map auto \
    --n_steps 500 \
    --batch_size 2 \
    --grad_accum 64 \
    --lr 5e-7 \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --max_length 1024 \
    --max_prompt_length 1024 \
    --dpo_beta 0.1 \
    --optimizer sparse_adamw \
    --save_steps 10 \
    --save_total_limit 3 \
    --gradient_checkpointing \
    --output_base_dir "$OUT_BASE" \
    --run_name "sparse_dpo_qwen3_32b_${DS_TAG}_${RUN_TAG}_500steps" \
    --use_wandb \
    --resume_from_checkpoint auto
TRAIN_RC=$?
set -e
echo "training exited rc=$TRAIN_RC (124 = hit TRAIN_TIMEOUT mid-run, expected)"

echo ""
echo "Finished: $(date)"
ls -la "$OUT_BASE" 2>&1 | tail -10
echo "=========================================="

# Advance the chain: resubmit to resume, or mark the pipeline done once at step 500.
kill $KEEPALIVE 2>/dev/null || true
bash "$PWD/scripts/qwen3_32b_advance.sh" || echo "advance.sh rc=$? (nonzero => STALL halt or done)"
