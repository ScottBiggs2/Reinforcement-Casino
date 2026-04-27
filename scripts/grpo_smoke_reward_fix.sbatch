#!/bin/bash
#SBATCH --job-name=grpo_smoke_reward_fix
#SBATCH --output=logs/grpo_smoke_reward_fix_%j.out
#SBATCH --error=logs/grpo_smoke_reward_fix_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00

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

# WandB OFF for smoke (per playbook §3.6 env drift caveat).
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_CONSOLE=off
export WANDB_SILENT=true

OUT_DIR="/scratch/xie.yiyi/grpo_smoke_reward_fix_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "GRPO smoke after LLAMA_COT_PROMPT_SUFFIX fix"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "Out    : $OUT_DIR"
echo "=========================================="
echo ""
echo "Validation goal: rewards/format_reasoning_reward/mean should be > 0"
echo "  in the first few steps. Pre-fix: constant 0 across 200 steps."
echo ""

python src/full_training/GRPO_train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset math-220k \
    --subset_size 64 \
    --num_steps 20 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_generations 8 \
    --generation_batch_size 8 \
    --learning_rate 5e-6 \
    --beta 0.025 \
    --warmup_ratio 0.1 \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
    --precision bf16 \
    --save_steps 20 \
    --save_total_limit 1 \
    --grpo_reward_profile llama_cot \
    --output_base_dir "$OUT_DIR" \
    --run_slug llama8b_math220k_smoke

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo ""
echo "=== reward summary from trainer_state.json ==="
TS=$(find "$OUT_DIR" -name "trainer_state.json" -path "*checkpoint-*" 2>/dev/null | head -1)
if [ -n "$TS" ] && [ -f "$TS" ]; then
  python -c "
import json, statistics
with open('$TS') as f: ts = json.load(f)
hist = ts['log_history']
keys = ['reward', 'rewards/accuracy_reward/mean', 'rewards/format_number_reward/mean', 'rewards/format_reasoning_reward/mean']
for k in keys:
    vals = [h[k] for h in hist if k in h and h[k] is not None]
    if vals:
        print(f'{k:50s}  first={vals[0]:.3f}  last={vals[-1]:.3f}  mean={statistics.mean(vals):.3f}  max={max(vals):.3f}')
    else:
        print(f'{k:50s}  (no data)')
print()
fr = [h.get('rewards/format_reasoning_reward/mean') for h in hist if h.get('rewards/format_reasoning_reward/mean') is not None]
print('=== VERDICT ===')
if not fr:
    print('FAIL: no format_reasoning_reward in log_history')
elif max(fr) >= 0.1:
    print(f'PASS: format_reasoning_reward reached {max(fr):.3f} (>= 0.1) — fix works in trainer')
else:
    print(f'FAIL: format_reasoning_reward stayed at max={max(fr):.4f} — fix not flowing through trainer')
"
else
  echo "(trainer_state.json not found, smoke likely failed)"
fi
echo "=========================================="
