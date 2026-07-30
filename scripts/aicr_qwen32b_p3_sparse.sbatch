#!/bin/bash
# Phase 3 (Qwen3-32B, AICR): sparse DPO under the p2 oracle mask @97.5%.
# Same hparams as p1/the 8B chain (Scott spec); optimizer sparse_adamw.
#
# 4x B200 (not 3): SparseAdamW holds DENSE fp32 moment buffers for every 2-D
# param -> ~254 GB of optimizer state on top of policy+ref+grads (~197 GB bf16),
# ~480 GB total. 3x178 GiB = 534 is tight with uneven per-layer placement;
# 4x178 = 712 GiB is safe. Naive-MP step time is roughly stage-count invariant.
#
# Self-chain: same timeout+afterok pattern as p1; exits 0 instantly once
# checkpoint-500 exists.
#
# Usage: sbatch scripts/aicr_qwen32b_p3_sparse.sbatch   (or via aicr_qwen32b_chain.sh)
#SBATCH --job-name=q32b_p3_sparse
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --gres=gpu:b200:4
#SBATCH --time=24:00:00
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/q32b_p3_%j.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh

export WANDB_PROJECT="rl_casino_transfer_v1"
export PYTORCH_ALLOC_CONF=expandable_segments:True

MODEL_BASE="Qwen/Qwen3-32B"
RUN_TAG="${RUN_TAG:-oracle_step500}"
MASK_PATH="${MASK_PATH:-$S/transfer_v1/oracle_masks_qwen3_32b/oracle_dpo_light_r1_step500_sp97.5.pt}"

OUT_BASE=$S/transfer_v1/sparse_dpo_light_r1_qwen3_32b_${RUN_TAG}
mkdir -p "$OUT_BASE"
if [ ! -f "$MASK_PATH" ]; then echo "FATAL: MASK_PATH not found: $MASK_PATH"; exit 1; fi

DONE_CKPT=$(find "$OUT_BASE" -maxdepth 4 -type d -name "checkpoint-500" 2>/dev/null | head -1)
if [ -n "$DONE_CKPT" ]; then echo "p3 already complete ($DONE_CKPT)"; exit 0; fi

# Preflight: same gate as p1 — the base model must resolve from the offline cache.
python scripts/preflight_base_model.py "$MODEL_BASE" || {
  echo "FATAL: base model not resolvable offline — fix the cache before resubmitting"; exit 1; }

echo "=========================================="
echo "Phase 3 (Qwen3-32B @ AICR): sparse DPO | RUN_TAG=$RUN_TAG"
echo "Job ${SLURM_JOB_ID:-local} | GPUs: $(nvidia-smi -L | wc -l) x B200 | $(date)"
echo "MASK_PATH=$MASK_PATH"
echo "=========================================="

TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-82800}"   # 23h of the 24h wall
set +e
timeout --signal=TERM "${TRAIN_TIMEOUT}s" \
python src/full_training/sparse_dpo_efficiency.py \
    --model_name "$MODEL_BASE" \
    --dataset "light-r1" \
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
    --run_name "sparse_dpo_qwen3_32b_light_r1_${RUN_TAG}_500steps" \
    --use_wandb \
    --resume_from_checkpoint auto
RC=$?
set -e
echo "training exited rc=$RC (124 = hit TRAIN_TIMEOUT mid-run, expected)"
find "$OUT_BASE" -maxdepth 4 -type d -name "checkpoint-*" | sort -V | tail -5

if [ "$RC" -ne 0 ] && [ "$RC" -ne 124 ]; then exit "$RC"; fi
exit 0
