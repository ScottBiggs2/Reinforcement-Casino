#!/bin/bash
#SBATCH --job-name=cross_dpo_grpo
#SBATCH --output=logs/cross_dpo_grpo_%j.out
#SBATCH --error=logs/cross_dpo_grpo_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --requeue

set -euo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] && source "$CONDA_SH" && conda activate /home/xie.yiyi/.conda/envs/rl_casino

cd /home/xie.yiyi/rc-sparse-speed
mkdir -p logs

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="/scratch/xie.yiyi/hf_cache"

ORACLE_DIR=/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b
DPO_LR1=$ORACLE_DIR/oracle_dpo_lightr1_step500_sp97.5.pt
GRPO_OPENR1=$ORACLE_DIR/oracle_grpo_math220k_step500_sp97.5.pt

OUT_DIR=/scratch/xie.yiyi/transfer_v1/cross_task_dpo_grpo
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "Cross-task: Open-R1 (GRPO) ↔ Light-R1 (DPO) oracle masks"
echo "Started: $(date)"
echo "  mask A (DPO Light-R1) = $DPO_LR1"
echo "  mask B (GRPO Open-R1) = $GRPO_OPENR1"
echo "=========================================="

# ---- Part 1: Jaccard ----
echo ""
echo "=== Part 1: Jaccard ==="
python << PYEOF
import torch
ma = torch.load("$DPO_LR1", map_location="cpu", weights_only=False)
mb = torch.load("$GRPO_OPENR1", map_location="cpu", weights_only=False)
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
print(f"Jaccard(Oracle-DPO-LightR1, Oracle-GRPO-OpenR1):")
print(f"  aggregate     = {agg:.4f}")
print(f"  per-layer mean= {sum(per)/len(per):.4f}")
print(f"  per-layer min = {min(per):.4f}")
print(f"  per-layer max = {max(per):.4f}")

# Save jaccard JSON for downstream use
import json
out = {
    "mask_a": "$DPO_LR1", "mask_b": "$GRPO_OPENR1",
    "aggregate_jaccard": agg,
    "per_layer_mean": sum(per)/len(per),
    "per_layer_min": min(per),
    "per_layer_max": max(per),
}
with open("$OUT_DIR/jaccard.json", "w") as f: json.dump(out, f, indent=2)
PYEOF

# ---- Part 2: Scott mask-vs-mask CKA ----
echo ""
echo "=== Part 2: per-layer CKA via Scott mask_to_cka.py (--compare mask_vs_mask) ==="
python src/cold_start/mask_to_cka.py \
    "$DPO_LR1" "$GRPO_OPENR1" \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --compare mask_vs_mask \
    --dataset_name tulu3 \
    --n_samples 64 \
    --max_length 512 \
    --batch_size 4 \
    --seed 42 \
    --device cuda \
    --output "$OUT_DIR/cka_mask_vs_mask.json"

echo ""
echo "=========================================="
echo "Finished: $(date)"
ls -lh "$OUT_DIR" 2>&1
echo "=========================================="
