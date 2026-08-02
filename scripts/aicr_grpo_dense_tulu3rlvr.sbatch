#!/bin/bash
# Dense GRPO on Tülu-3 RLVR (GSM8K+MATH rows) — the missing GRPO⇄GRPO cell for
# Figure 2 (reviewer XD33: "the GRPO-OpenR1 ↔ GRPO-Tülu3 cell does not exist").
#
# Schedule-matched to aicr_grpo_matched_dense.sbatch: cosine over 500 steps,
# 50-step warmup, lr 5e-6, cap 2048, global grad clip 0.1, adamw_8bit, bf16.
# Only the dataset differs: tulu3-rlvr-math = allenai/RLVR-GSM-MATH-IF-Mixed-Constraints
# filtered to its verifiable GSM8K+MATH rows (IF rows need a constraint verifier we
# do not implement; see dataset_registry.py). The DPO-side Tülu3 is the preference
# mixture, so this cell is "GRPO on Tülu-3's RL data", to be disclosed in the caption.
#
# Downstream: checkpoint-500 → checkpoint-diff oracle mask (ρ=97.5) → Jaccard/CKA
# against oracle_grpo_math220k_step500_sp97.5 for the Figure 2 revision.
#
# Usage: sbatch aicr_grpo_dense_tulu3rlvr.sbatch
#SBATCH --job-name=grpo_dense_t3rlvr
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:b200:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/grpo_dense_t3rlvr_%j.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh

export WANDB_PROJECT="rl_casino_rebuttal"
OUT_BASE="$S/rebuttal_analysis/grpo_matched_aicr/dense_tulu3rlvr"
mkdir -p "$OUT_BASE"

# bitsandbytes AdamW8bit preflight on sm_100 (same guard as the math-220k dense arm).
python - <<'PY'
import torch, bitsandbytes as bnb
p = torch.nn.Parameter(torch.randn(64, 64, device="cuda"))
opt = bnb.optim.AdamW8bit([p], lr=1e-3)
p.grad = torch.randn_like(p)
opt.step()
print("preflight: bnb AdamW8bit OK on", torch.cuda.get_device_name(0))
PY

echo "=========================================="
echo "Matched-schedule GRPO — DENSE | Llama-3.1-8B-Instruct | Tulu3-RLVR (GSM+MATH) | AICR b200"
echo "  steps=500 cosine warmup_ratio=0.1 lr=5e-6 beta=0.025 clip=0.1 cap=2048"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "=========================================="

python src/full_training/GRPO_train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset tulu3-rlvr-math \
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
    --run_slug grpo_matched_dense_tulu3rlvr_500steps_cap2048 \
    --run_name grpo_matched_aicr_dense_tulu3rlvr_500steps_cap2048 \
    --use_wandb \
    --resume_from_checkpoint auto

echo "Finished: $(date)"
