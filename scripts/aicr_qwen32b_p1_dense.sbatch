#!/bin/bash
# Phase 1 (Qwen3-32B, AICR): dense DPO on Qwen/Qwen3-32B, resuming the Explorer
# trajectory from transferred checkpoint-250 (job chain frozen 2026-06-14 at
# p1 250/500; see docs/run_logs/qwen3_32b_autopipeline_2026-06-12.md).
#
# Identical hparams to the Explorer run (Scott spec): 500 steps, eff batch 128
# (bs2 x ga64, single MP process), lr 5e-7 linear, warmup_ratio 0.1, beta 0.1,
# max_length/max_prompt 1024, adamw_8bit, device_map=auto. Only cluster, GPU
# type (H200->B200) and the 24h walltime differ. Disclose in any write-up that
# the trajectory spans two clusters at step 250 (bf16 numerics + trl 0.24 vs
# the June Explorer env).
#
# Self-chain: 24h wall, training wrapped in `timeout 23h` so it exits cleanly
# and the next afterok job resumes (--resume_from_checkpoint auto). If already
# at checkpoint-500 this exits 0 immediately, so extra chained slots are free.
#
# Usage: sbatch scripts/aicr_qwen32b_p1_dense.sbatch   (or via aicr_qwen32b_chain.sh)
#SBATCH --job-name=q32b_p1_dense
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --gres=gpu:b200:3
#SBATCH --time=24:00:00
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/q32b_p1_%j.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh

export WANDB_PROJECT="rl_casino_transfer_v1"
export PYTORCH_ALLOC_CONF=expandable_segments:True

OUT_BASE=$S/transfer_v1/dense_dpo_light_r1_qwen3_32b
CKDIR=$OUT_BASE/checkpoints/qwen_qwen3_32b_light_r1
mkdir -p "$OUT_BASE"

if [ -d "$CKDIR/checkpoint-500" ]; then
  echo "p1 already complete (checkpoint-500 exists) — nothing to do"; exit 0
fi

# Gate: never resume from a partially-transferred checkpoint. The sentinel is
# written by the operator only after the Explorer->AICR md5 verification passes
# (migration lesson: a "size stopped changing" check once submitted jobs against
# a 6.6 GB fragment of a 32.1 GB file).
LATEST=$(ls -d "$CKDIR"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
if [ ! -f "$OUT_BASE/CKPT250_VERIFIED" ] && [ "${LATEST:-0}" -le 250 ]; then
  echo "FATAL: checkpoint-250 transfer not verified (no CKPT250_VERIFIED) and no later checkpoint"; exit 1
fi

# Preflight: the base model must resolve from the offline cache. 235953 burned a
# submission (and cancelled six downstream jobs) discovering mid-run that
# Qwen/Qwen3-32B was never in the AICR cache.
python scripts/preflight_base_model.py "Qwen/Qwen3-32B" || {
  echo "FATAL: base model not resolvable offline — fix the cache before resubmitting"; exit 1; }

echo "=========================================="
echo "Phase 1 (Qwen3-32B @ AICR): dense DPO resume | latest ckpt: ${LATEST:-none}"
echo "Job ${SLURM_JOB_ID:-local} | GPUs: $(nvidia-smi -L | wc -l) x B200 | $(date)"
echo "=========================================="

TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-82800}"   # 23h of the 24h wall; 1h for ckpt flush
set +e
timeout --signal=TERM "${TRAIN_TIMEOUT}s" \
python src/full_training/DPO_train.py \
    --model_name "Qwen/Qwen3-32B" \
    --dataset "light-r1" \
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
    --run_name "dense_dpo_qwen3_32b_light_r1_scott_full"
RC=$?
set -e
echo "training exited rc=$RC (124 = hit TRAIN_TIMEOUT mid-run, expected)"
find "$CKDIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -5

# 0 or timeout -> let the afterok chain continue; a real crash halts the chain.
if [ "$RC" -ne 0 ] && [ "$RC" -ne 124 ]; then exit "$RC"; fi
exit 0
