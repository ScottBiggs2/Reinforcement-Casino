#!/bin/bash
#SBATCH --job-name=cross_task_jcka
#SBATCH --output=logs/cross_task_jcka_%j.out
#SBATCH --error=logs/cross_task_jcka_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --requeue

set -euo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="/home/xie.yiyi/.conda/envs/rl_casino"
[ -f "$CONDA_SH" ] && source "$CONDA_SH" && conda activate "$ENV_PATH"

cd /home/xie.yiyi/rc-sparse-speed
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="/scratch/xie.yiyi/hf_cache"

ORACLE_DIR=/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b
DPO_LIGHTR1_500=$ORACLE_DIR/oracle_dpo_lightr1_step500_sp97.5.pt
DPO_LIGHTR1_150=$ORACLE_DIR/oracle_dpo_lightr1_step150_sp97.5.pt
GRPO_MATH_500=$ORACLE_DIR/oracle_grpo_math220k_step500_sp97.5.pt
GRPO_MATH_850=$ORACLE_DIR/oracle_grpo_math220k_step850_sp97.5.pt

OUT_DIR=/scratch/xie.yiyi/transfer_v1/cross_task_analysis
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "Cross-task Jaccard + CKA: Open-R1(GRPO) vs Light-R1(DPO)"
echo "Started: $(date)"
echo "=========================================="

# ------- Part 1: Jaccard for several pairs -------
python << PYEOF
import torch
def jac(pa, pb, label):
    print(f"\n=== {label} ===")
    print(f"  A: {pa}")
    print(f"  B: {pb}")
    ma = torch.load(pa, map_location="cpu", weights_only=False)
    mb = torch.load(pb, map_location="cpu", weights_only=False)
    masks_a = ma.get("masks", ma)
    masks_b = mb.get("masks", mb)
    inter, union = 0, 0
    per = []
    for k in set(masks_a) & set(masks_b):
        a = masks_a[k].bool()
        b = masks_b[k].bool()
        i = (a & b).sum().item()
        u = (a | b).sum().item()
        inter += i; union += u
        per.append(i / max(u, 1))
    agg = inter / max(union, 1)
    print(f"  aggregate Jaccard: {agg:.4f}")
    print(f"  per-layer mean:    {sum(per)/len(per):.4f}")
    print(f"  per-layer min:     {min(per):.4f}")
    print(f"  per-layer max:     {max(per):.4f}")
    return agg

print("===========================================")
print("CROSS-TASK comparisons (DPO vs GRPO)")
print("===========================================")
j1 = jac("$DPO_LIGHTR1_500", "$GRPO_MATH_500", "Light-R1 step500 (DPO) vs Math-GRPO step500")
j2 = jac("$DPO_LIGHTR1_500", "$GRPO_MATH_850", "Light-R1 step500 (DPO) vs Math-GRPO step850")
j3 = jac("$DPO_LIGHTR1_150", "$GRPO_MATH_500", "Light-R1 step150 (DPO) vs Math-GRPO step500")

print("\n===========================================")
print("REFERENCE: WITHIN-DPO + WITHIN-GRPO")
print("===========================================")
ref1 = jac("$DPO_LIGHTR1_150", "$DPO_LIGHTR1_500", "Light-R1 step150 vs step500 (within-DPO)")
ref2 = jac("$GRPO_MATH_500", "$GRPO_MATH_850", "Math-GRPO step500 vs step850 (within-GRPO)")

print("\n===========================================")
print("SUMMARY TABLE")
print("===========================================")
print(f"  within-DPO (Light-R1 150 vs 500):       {ref1:.4f}")
print(f"  within-GRPO (Math 500 vs 850):          {ref2:.4f}")
print(f"  cross-task DPO-LR1-500 vs GRPO-MATH-500: {j1:.4f}")
print(f"  cross-task DPO-LR1-500 vs GRPO-MATH-850: {j2:.4f}")
print(f"  cross-task DPO-LR1-150 vs GRPO-MATH-500: {j3:.4f}")
PYEOF

# ------- Part 2: CKA via existing tool -------
echo ""
echo "=========================================="
echo "Part 2: per-layer CKA between masked models"
echo "=========================================="

# Build masks JSON for CKA tool
MASKS_JSON="$OUT_DIR/cross_task_masks.json"
cat > "$MASKS_JSON" << JSONEOF
[
  {"label": "Oracle-DPO-Light-R1-step500", "path": "$DPO_LIGHTR1_500"},
  {"label": "Oracle-GRPO-Math-step500",    "path": "$GRPO_MATH_500"},
  {"label": "Oracle-GRPO-Math-step850",    "path": "$GRPO_MATH_850"}
]
JSONEOF

python src/analysis/inspect_mask_erank_cka.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --masks_json "$MASKS_JSON" \
    --output_dir "$OUT_DIR" \
    --layer_substr down_proj \
    --n_samples 64 \
    --max_length 512 \
    --batch_size 4 \
    --seed 42

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "Outputs:"
ls -lh "$OUT_DIR" 2>&1 | head -10
echo "=========================================="
