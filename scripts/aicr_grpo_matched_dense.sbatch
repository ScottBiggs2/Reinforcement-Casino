#!/bin/bash
# Dense GRPO baseline for the matched-schedule GRPO set on AICR — the "does our
# framework train GRPO" demonstration for NeurIPS #29841 (reviewer bf6h W1).
#
# This arm shares ONE schedule with both sparse arms (aicr_grpo_matched_sparse.sbatch):
# cosine over 500 steps, 50-step warmup, lr 5e-6, cap 2048, global grad clip 0.1.
# Appendix D.4's dense-vs-sparse comparison was unreadable because none of that was
# matched: dense ran cosine over max_steps=5000 (99.8% of peak LR at step 500) against
# sparse linear over 500 (0.2%), a 449x gap, plus a 10x global-clip mismatch
# (docs/run_logs/grpo_schedule_confound_2026-07-25.md, DO_NOT_REPEAT.md 2026-07-27).
# Cap 2048 per Scott's locked grpo_500step_5way_sweep.yaml + Table 6 — at the as-run
# 1024 cap, 90% of completions truncate and the reward barely scores reasoning.
#
# Usage: sbatch aicr_grpo_matched_dense.sbatch
#SBATCH --job-name=grpo_dense
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:b200:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/grpo_dense_%j.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh

export WANDB_PROJECT="rl_casino_rebuttal"
OUT_BASE="$S/rebuttal_analysis/grpo_matched_aicr/dense"
mkdir -p "$OUT_BASE"

# bitsandbytes 0.50.0 has never run an optimizer step on sm_100 here; fail in
# seconds with a readable error instead of after the model loads.
python - <<'PY'
import torch, bitsandbytes as bnb
p = torch.nn.Parameter(torch.randn(64, 64, device="cuda"))
opt = bnb.optim.AdamW8bit([p], lr=1e-3)
p.grad = torch.randn_like(p)
opt.step()
print("preflight: bnb AdamW8bit OK on", torch.cuda.get_device_name(0))
PY

echo "=========================================="
echo "Matched-schedule GRPO — DENSE | Llama-3.1-8B-Instruct | OpenR1-Math-220k | AICR b200"
echo "  steps=500 cosine warmup_ratio=0.1 lr=5e-6 beta=0.025 clip=0.1 cap=2048"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "=========================================="

python src/full_training/GRPO_train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset math-220k \
    --num_steps 500 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --max_grad_norm 0.1 \
    --beta 0.025 \
    --optim adamw_8bit \
    --num_generations 8 \
    --generation_batch_size 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_prompt_length 512 \
    --max_completion_length 2048 \
    --grpo_reward_profile llama_cot \
    --save_steps 50 \
    --save_total_limit 3 \
    --output_base_dir "$OUT_BASE" \
    --run_slug grpo_matched_dense_500steps_cap2048 \
    --run_name grpo_matched_aicr_dense_500steps_cap2048 \
    --use_wandb \
    --resume_from_checkpoint auto

echo "Finished: $(date)"
