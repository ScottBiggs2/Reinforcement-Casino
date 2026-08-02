#!/bin/bash
# Sparse DPO on Qwen/Qwen3-8B @ AICR — same hparams as the June Explorer chain
# (Scott spec: 500 steps, eff. batch 128, lr 5e-7, beta 0.1, sparse_adamw).
# Single B200 (178 GiB): policy+ref bf16 ~33 GB + grads + dense fp32 moments
# ~66 GB fits with margin; the June run fit the same config on 1x H200 (141 GB).
#
# Usage: MASK_PATH=<...pt> RUN_TAG=random_sp97.5_seed42 \
#        sbatch --chdir=$W/rc-sparse-speed scripts/aicr_qwen8b_p3_sparse.sbatch
#SBATCH --job-name=q8b_p3_sparse
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gres=gpu:b200:1
#SBATCH --time=24:00:00
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/q8b_p3_%j.out

set -euo pipefail
cd /work/neu/p2026_0038_neu/xie_yiyi/rc-sparse-speed
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

export WANDB_PROJECT="${WANDB_PROJECT:-qwen3}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

MODEL_BASE="Qwen/Qwen3-8B"
RUN_TAG="${RUN_TAG:?Set RUN_TAG, e.g. random_sp97.5_seed42}"
MASK_PATH="${MASK_PATH:?Set MASK_PATH=/scratch/.../random_baseline_light_r1_sp97.5_seed42.pt}"

OUT_BASE=$S/transfer_v1/sparse_dpo_light_r1_qwen3_8b_${RUN_TAG}
mkdir -p "$OUT_BASE"
if [ ! -f "$MASK_PATH" ]; then echo "FATAL: MASK_PATH not found: $MASK_PATH"; exit 1; fi

DONE_CKPT=$(find "$OUT_BASE" -maxdepth 4 -type d -name "checkpoint-500" 2>/dev/null | head -1)
if [ -n "$DONE_CKPT" ]; then echo "p3 already complete ($DONE_CKPT)"; exit 0; fi

python scripts/preflight_base_model.py "$MODEL_BASE" || {
  echo "FATAL: base model not resolvable offline — fix the cache before resubmitting"; exit 1; }

echo "=========================================="
echo "Sparse DPO Qwen3-8B @ AICR | RUN_TAG=$RUN_TAG"
echo "Job ${SLURM_JOB_ID:-local} | $(date)"
echo "MASK_PATH=$MASK_PATH"
echo "=========================================="

TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-82800}"   # 23h of the 24h wall
set +e
timeout --signal=TERM "${TRAIN_TIMEOUT}s" \
python src/full_training/sparse_dpo_efficiency.py \
    --model_name "$MODEL_BASE" \
    --dataset "light-r1" \
    --mask "$MASK_PATH" \
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
    --save_steps 50 \
    --save_total_limit 3 \
    --gradient_checkpointing \
    --output_base_dir "$OUT_BASE" \
    --run_name "sparse_dpo_qwen3_8b_light_r1_${RUN_TAG}_500steps" \
    --use_wandb \
    --resume_from_checkpoint auto
RC=$?
set -e
echo "training exited rc=$RC (124 = hit TRAIN_TIMEOUT mid-run, expected)"
find "$OUT_BASE" -maxdepth 4 -type d -name "checkpoint-*" | sort -V | tail -5

if [ "$RC" -ne 0 ] && [ "$RC" -ne 124 ]; then exit "$RC"; fi
exit 0
