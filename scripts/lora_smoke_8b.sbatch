#!/bin/bash
# SMOKE: validate the --lora path before spending real compute on the rebuttal runs.
# Llama-3.1-8B-Instruct + light-r1, LoRA r=16 @ lr 3e-5.
#
# This exists to catch silent failures, not to produce a number. It runs training
# TWICE: 10 steps, then a resume to 20. The second pass is the point — the 500-step
# arms depend on PEFT checkpoint save/resume, and without this the first execution of
# that path would be a 6-hour job.
#
# PASS requires ALL of:
#   1. trainable params == 41,943,040 (0.522%) for r=16. If this is wrong the whole
#      matched-budget comparison is wrong. Verified arithmetic: 81920*r per layer x 32.
#   2. grad_norm > 0 in the step logs. This is the real test of
#      enable_input_require_grads(): under gradient checkpointing with frozen base
#      weights, nothing in the checkpointed segment requires grad and the adapter
#      receives ZERO gradient — the run completes and looks normal. Loss wobbling is
#      NOT sufficient evidence (warmup_ratio 0.1 on a 10-step run ends at step 1).
#   3. rewards/margins off 0.0 by step 10.
#   4. Pass 2 prints a resume and starts from global_step 10, not 0. Trainer resume
#      with PEFT has historically failed to reload adapter weights silently.
#   5. efficiency_summary.json exists and steps_this_segment == 10 on pass 2
#      (proves the resume-aware s/step fix works).
#SBATCH --job-name=lora_smoke_8b
#SBATCH --output=logs/lora_smoke_8b_%j.out
#SBATCH --error=logs/lora_smoke_8b_%j.err
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --requeue

set -uo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="${TRAIN_ENV:-/home/xie.yiyi/.conda/envs/rl_casino}"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

REPO="${REPO_DIR:-/home/xie.yiyi/rc-sparse-speed}"
cd "$REPO"
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
SCRATCH_ROOT="${SCRATCH_USER_ROOT:-/scratch/$USER}"
export HF_HOME="$SCRATCH_ROOT/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH_ROOT/hf_cache/datasets"
export WANDB_MODE=disabled WANDB_DISABLED=true

MODEL_BASE="meta-llama/Llama-3.1-8B-Instruct"
OUT_BASE="$SCRATCH_ROOT/rebuttal_lora/smoke"
rm -rf "$OUT_BASE"          # always a cold start, or pass 1 would itself resume
mkdir -p "$OUT_BASE"

python -c "import peft, trl; print('peft', peft.__version__, '| trl', trl.__version__)" || exit 1

run_pass () {
    local steps="$1" label="$2"
    echo ""
    echo "=========================================="
    echo "PASS $label — target global_step $steps"
    echo "=========================================="
    python src/full_training/DPO_train.py \
        --model_name "$MODEL_BASE" \
        --dataset light-r1 \
        --lora \
        --lora_r 16 \
        --learning_rate 3e-5 \
        --num_steps "$steps" \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 64 \
        --warmup_ratio 0.1 \
        --weight_decay 0.0 \
        --max_length 1024 \
        --max_prompt_length 1024 \
        --dpo_beta 0.1 \
        --optim adamw_8bit \
        --gradient_checkpointing \
        --save_steps 5 \
        --save_total_limit 2 \
        --resume_from_checkpoint auto \
        --output_base_dir "$OUT_BASE" \
        --run_name "lora_smoke_8b_r16_${label}"
    return $?
}

# Pass 1: cold start, 10 steps, writes checkpoints at 5 and 10.
run_pass 10 "1_cold" || { echo "SMOKE FAIL: pass 1 rc=$?"; exit 1; }

# Pass 2: same dir, target 20 -> must RESUME from checkpoint-10, not restart.
run_pass 20 "2_resume" || { echo "SMOKE FAIL: pass 2 rc=$?"; exit 1; }

echo ""
echo "=========================================="
echo "CHECKS"
echo "=========================================="
SUMMARY=$(ls "$OUT_BASE"/checkpoints/*/efficiency_summary.json 2>/dev/null | head -1)
if [ -z "$SUMMARY" ]; then
    echo "FAIL: no efficiency_summary.json written"; exit 1
fi
cat "$SUMMARY"
python - "$SUMMARY" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
ok = True
def chk(cond, msg):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + msg); ok = ok and cond
chk(s.get("trainable_params") == 41_943_040,
    f"trainable_params == 41,943,040 (got {s.get('trainable_params'):,})")
chk(s.get("resumed") is True, f"pass 2 resumed (got {s.get('resumed')})")
chk(s.get("steps_this_segment") == 10,
    f"steps_this_segment == 10 (got {s.get('steps_this_segment')}) — resume-aware s/step")
chk(s.get("steps_completed") == 20,
    f"steps_completed == 20 (got {s.get('steps_completed')}) — resume continued, did not restart")
m = s.get("train_batch_metrics") or {}
chk(m.get("rewards_margin") not in (None, 0.0),
    f"rewards/margins moved off zero (got {m.get('rewards_margin')})")
chk(m.get("logps_chosen") is not None and m.get("logps_rejected") is not None,
    "logps/chosen and logps/rejected both recorded (likelihood-displacement guard)")
print("\nSMOKE " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
RC=$?

echo ""
echo "Still check BY EYE in the step logs above (not machine-checkable here):"
echo "  * grad_norm > 0  <- the real enable_input_require_grads() test"
echo "  * pass 2 printed a resume line and did not restart at step 0"
echo "Finished: $(date)"
exit $RC
