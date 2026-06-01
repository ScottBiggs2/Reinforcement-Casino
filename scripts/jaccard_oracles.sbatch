#!/bin/bash
#SBATCH --job-name=jaccard_oracles
#SBATCH --output=logs/jaccard_oracles_%j.out
#SBATCH --error=logs/jaccard_oracles_%j.err
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="/home/xie.yiyi/.conda/envs/rl_casino"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; conda activate "$ENV_PATH"; fi

cd /home/xie.yiyi/rc-sparse-speed
mkdir -p logs

cat > /tmp/jaccard_oracles.py << 'PYEOF'
import os, sys, torch
ORACLE_DIR = "/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b"

def load_masks(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob.get("masks", blob)

def jaccard(masks_a, masks_b, label):
    inter, union = 0, 0
    per_layer = {}
    common = set(masks_a) & set(masks_b)
    only_a = set(masks_a) - set(masks_b)
    only_b = set(masks_b) - set(masks_a)
    if only_a or only_b:
        print(f"  WARNING: {len(only_a)} keys in A only, {len(only_b)} keys in B only")
    for k in common:
        a = masks_a[k].bool()
        b = masks_b[k].bool()
        i = (a & b).sum().item()
        u = (a | b).sum().item()
        inter += i
        union += u
        per_layer[k] = i / max(u, 1)
    agg = inter / max(union, 1)
    vals = list(per_layer.values())
    print(f"=== {label} ===")
    print(f"  aggregate Jaccard: {agg:.4f}")
    print(f"  per-layer mean:    {sum(vals)/len(vals):.4f}")
    print(f"  per-layer min:     {min(vals):.4f}")
    print(f"  per-layer max:     {max(vals):.4f}")
    print(f"  total ∩:           {inter:,}")
    print(f"  total ∪:           {union:,}")
    print()
    return agg

print("Tulu3 oracles step150 vs step500:")
print(f"  loading {ORACLE_DIR}/oracle_dpo_tulu3_step500_sp97.5.pt")
m500 = load_masks(f"{ORACLE_DIR}/oracle_dpo_tulu3_step500_sp97.5.pt")
print(f"  loading {ORACLE_DIR}/oracle_dpo_tulu3_step150_sp97.5.pt")
m150 = load_masks(f"{ORACLE_DIR}/oracle_dpo_tulu3_step150_sp97.5.pt")
j_tulu3 = jaccard(m500, m150, "Tulu3: oracle_step500 vs oracle_step150")

print("=" * 60)
print("Comparison:")
print(f"  Light-R1: 0.9897 (computed earlier)")
print(f"  Tulu3:    {j_tulu3:.4f}")
PYEOF

python /tmp/jaccard_oracles.py
