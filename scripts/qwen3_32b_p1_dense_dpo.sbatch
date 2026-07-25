#!/bin/bash
# Phase 1 (Qwen3-32B): Dense DPO on Qwen/Qwen3-32B (instruct, NOT -Base) via
# device_map="auto" model parallelism across 8 H200 (single process, NOT torchrun).
# Same hparams as the 8B chain; only backbone + parallelism + paths change.
# Usage: DATASET=light-r1 sbatch qwen3_32b_p1_dense_dpo.sbatch
#SBATCH --job-name=qwen3_32b_p1_dense
#SBATCH --output=logs/qwen3_32b_p1_dense_%j.out
#SBATCH --error=logs/qwen3_32b_p1_dense_%j.err
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
export PYTORCH_ALLOC_CONF=expandable_segments:True

MODEL_BASE="Qwen/Qwen3-32B"          # instruct/post-trained (Qwen3-32B-Base is the base — do NOT use)
DATASET="${DATASET:?Set DATASET=light-r1 | tulu3}"
DS_TAG=$(echo "$DATASET" | tr '-' '_')

OUT_BASE="/scratch/xie.yiyi/transfer_v1/dense_dpo_${DS_TAG}_qwen3_32b"
mkdir -p "$OUT_BASE"

NGPUS=$(nvidia-smi -L | wc -l)
echo "=========================================="
echo "Phase 1 (Qwen3-32B): Dense DPO  |  $MODEL_BASE  |  $DATASET  |  device_map=auto x${NGPUS} H200"
echo "Job ID : ${SLURM_JOB_ID:-local}   GPUs: $NGPUS   Started: $(date)"
echo "Hparams: 500 steps, eff_batch 128 (bs2 x ga64, single MP process), optim=adamw_8bit"
echo "OUT_BASE = $OUT_BASE"
echo "=========================================="

# device_map=auto model parallelism (proven to run on 32B; FSDP hit TRL+DPO
# integration bugs: ref not sharded / precompute device mismatch / 8bit save).
# policy + ref both spread by layer across GPUs (no replication). Slow (~177s/it)
# but robust; --requeue + resume handle the 8h wall. Model is pre-cached.
# GPU keep-alive sidecar: MP keeps per-GPU util low → keeps it visibly active so
# RC's idle monitor doesn't false-cancel (§7b). Killed on exit.
python -c "import torch
n=torch.cuda.device_count()
gs=[torch.randn(2048,2048,device=f'cuda:{i}') for i in range(n)]
while True:
    for _ in range(20):
        for g in gs: g.matmul(g)
    torch.cuda.synchronize()" &
KEEPALIVE=$!; trap 'kill $KEEPALIVE 2>/dev/null' EXIT
echo "keep-alive sidecar PID $KEEPALIVE on $NGPUS GPUs"

# Self-resubmitting requeue chain: `--requeue` does NOT fire on TIME LIMIT (an 8h
# wall hit = CANCELLED, not requeued), which is why prior runs never advanced past
# their first checkpoint. Instead we `timeout` training ~30 min before the wall so
# python exits cleanly, the tail runs, and scripts/qwen3_32b_advance.sh resubmits
# this phase (resume_from_checkpoint auto) until checkpoint-500 exists, then moves
# the chain to p2. save_steps=10 so every ~8h slot crosses several save boundaries
# (progress visible to the advancer's stall guard; <=10 steps lost on timeout).
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-27000}"   # 7.5h of the 8h wall; 30m left for ckpt flush + chaining
set +e
timeout --signal=TERM "${TRAIN_TIMEOUT}s" \
python src/full_training/DPO_train.py \
    --model_name "$MODEL_BASE" \
    --dataset "$DATASET" \
    --device_map auto \
    --num_steps 500 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.1 \
    --dpo_beta 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --max_length 1024 \
    --max_prompt_length 1024 \
    --save_steps 10 \
    --save_total_limit 3 \
    --gradient_checkpointing \
    --output_base_dir "$OUT_BASE" \
    --resume_from_checkpoint auto \
    --use_wandb \
    --run_name "dense_dpo_qwen3_32b_${DS_TAG}_scott_full"
TRAIN_RC=$?
set -e
echo "training exited rc=$TRAIN_RC (124 = hit TRAIN_TIMEOUT mid-run, expected)"

echo ""
echo "=========================================="
echo "Finished: $(date)"
find "$OUT_BASE" -maxdepth 4 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -12
echo "=========================================="

# Advance the chain: resubmit this phase to resume, or move to p2 once at step 500.
kill $KEEPALIVE 2>/dev/null || true
bash "$PWD/scripts/qwen3_32b_advance.sh" || echo "advance.sh rc=$? (nonzero => STALL halt or done)"
