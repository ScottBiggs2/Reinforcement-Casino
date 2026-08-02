#!/bin/bash
# Dense GRPO on ReasonMed (medical MCQ) — the second GRPO dataset, so that the
# GRPO⇄GRPO oracle-mask comparison in Figure 2 has a medical arm to sit against
# GRPO-OpenR1-Math.
#
# SCHEDULE IS COPIED VERBATIM from aicr_grpo_matched_dense.sbatch (the math-220k
# dense arm): cosine over 500 steps, warmup_ratio 0.1, lr 5e-6, beta 0.025, global
# clip 0.1, cap 2048, llama_cot rewards, adamw_8bit, bf16, 1×B200. **Dataset is the
# only variable**, which is the whole point — a mask difference must not be
# attributable to a schedule difference (grpo_schedule_confound_2026-07-25.md).
#
# THE DATASET IS NOT RLVR AND MUST NOT BE CALLED THAT.
# lingshu-medical-mllm/ReasonMed ships instruction/input/output and NO answer
# column. scripts/prep_reasonmed_mcq.py recovers a target by parsing the answer
# sentence out of the dataset's own CoT and matching it to one of the lettered
# options. So `solution` is the TEACHER CoT's conclusion, not the gold label from
# MedQA/MedMCQA/PubMedQA — ReasonMed does not carry those. Keep rate 12.5 %, and
# the survivors are the easier/cleaner questions. Disclose both wherever this runs.
#
# Answers are emitted as the 1-based OPTION NUMBER, not a letter, because
# format_number_reward pays 0.5 only on a numeric token; a letter target would
# silently zero that term and change the reward composition versus the math arm.
#
# Prereq: python scripts/prep_reasonmed_mcq.py --out $S/datasets/reasonmed_mcq
# Downstream: checkpoint-500 → checkpoint-diff oracle mask ρ=97.5 → Jaccard/CKA
# against oracle_grpo_math220k_step500_sp97.5 → Figure 2 replot.
#
# Usage: sbatch aicr_grpo_dense_reasonmed.sbatch
#SBATCH --job-name=grpo_reasonmed
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:b200:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --exclude=b0003
#SBATCH --output=/work/neu/p2026_0038_neu/xie_yiyi/logs/grpo_reasonmed_%j.out

set -euo pipefail
source /work/neu/p2026_0038_neu/xie_yiyi/aicr_env.sh

export WANDB_PROJECT="rl_casino_rebuttal"
OUT_BASE="$S/rebuttal_analysis/grpo_matched_aicr/dense_reasonmed"
DATA_DIR="$S/datasets/reasonmed_mcq"
mkdir -p "$OUT_BASE"

# Fail fast rather than 5 h in: the prep step is a separate manual stage.
if [ ! -f "$DATA_DIR/data.parquet" ]; then
  echo "FATAL: $DATA_DIR/data.parquet missing — run scripts/prep_reasonmed_mcq.py first" >&2
  exit 1
fi
python - <<PY
from datasets import load_dataset
ds = load_dataset("parquet", data_files="$DATA_DIR/data.parquet", split="train")
assert set(ds.column_names) == {"prompt", "solution"}, ds.column_names
assert len(ds) >= 4000, f"only {len(ds)} rows; 500 steps x 8 needs 4000"
print(f"dataset gate OK: {len(ds)} rows, columns {ds.column_names}")
PY

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
echo "Matched-schedule GRPO — DENSE | Llama-3.1-8B-Instruct | ReasonMed MCQ | AICR b200"
echo "  steps=500 cosine warmup_ratio=0.1 lr=5e-6 beta=0.025 clip=0.1 cap=2048"
echo "  job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
echo "=========================================="

python src/full_training/GRPO_train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset "$DATA_DIR" \
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
    --run_slug grpo_reasonmed_500steps_cap2048 \
    --run_name grpo_matched_aicr_reasonmed_500steps_cap2048 \
    --use_wandb \
    --resume_from_checkpoint auto

echo "done $(date)"
