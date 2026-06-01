#!/bin/bash
#SBATCH --job-name=per_layer_jaccard
#SBATCH --output=logs/per_layer_jaccard_%j.out
#SBATCH --error=logs/per_layer_jaccard_%j.err
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00

set -euo pipefail

CONDA_SH="/shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] && source "$CONDA_SH" && conda activate /home/xie.yiyi/.conda/envs/rl_casino

cd /home/xie.yiyi/rc-sparse-speed
mkdir -p logs

python << 'PYEOF'
import torch, json, itertools, os

ORACLE_DIR = "/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b"
OUT = "/scratch/xie.yiyi/transfer_v1/multi_mask_analysis/per_layer_jaccard.json"

masks_meta = [
    ("DPO-LightR1-500", f"{ORACLE_DIR}/oracle_dpo_lightr1_step500_sp97.5.pt"),
    ("DPO-Tulu3-500",   f"{ORACLE_DIR}/oracle_dpo_tulu3_step500_sp97.5.pt"),
    ("GRPO-OpenR1-500", f"{ORACLE_DIR}/oracle_grpo_math220k_step500_sp97.5.pt"),
    ("Random",          f"{ORACLE_DIR}/random_baseline_lightr1_sp97.5_seed42.pt"),
]

loaded = {}
for label, path in masks_meta:
    print(f"Loading {label} ...")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    loaded[label] = blob.get("masks", blob)
print(f"All masks loaded. # keys in first: {len(next(iter(loaded.values())))}")

# Group by transformer layer index (e.g. model.layers.5.*)
def layer_idx(k):
    parts = k.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try: return int(parts[i + 1])
            except: return None
    return None

result = {"per_layer_all_params": {}, "per_layer_down_proj": {}, "per_layer_aggregated": {}}

pairs = list(itertools.combinations([m[0] for m in masks_meta], 2))
for la, lb in pairs:
    ma = loaded[la]
    mb = loaded[lb]
    pair_key = f"{la} ⇄ {lb}"

    # 1) per-tensor Jaccard (every param tensor)
    per_tensor = {}
    for k in set(ma) & set(mb):
        a = ma[k].bool()
        b = mb[k].bool()
        i = (a & b).sum().item()
        u = (a | b).sum().item()
        per_tensor[k] = {"intersect": i, "union": u, "jaccard": i / max(u, 1)}

    # 2) per-layer down_proj only (matches CKA layers)
    per_layer_dp = {}
    for k, v in per_tensor.items():
        if "down_proj" in k:
            li = layer_idx(k)
            if li is not None:
                per_layer_dp[li] = v["jaccard"]

    # 3) per-layer aggregated across all tensors in that layer
    per_layer_agg_data = {}  # layer_idx -> [intersect, union]
    for k, v in per_tensor.items():
        li = layer_idx(k)
        if li is None: continue
        if li not in per_layer_agg_data:
            per_layer_agg_data[li] = [0, 0]
        per_layer_agg_data[li][0] += v["intersect"]
        per_layer_agg_data[li][1] += v["union"]
    per_layer_agg = {li: i / max(u, 1) for li, (i, u) in per_layer_agg_data.items()}

    result["per_layer_all_params"][pair_key] = per_tensor
    result["per_layer_down_proj"][pair_key] = per_layer_dp
    result["per_layer_aggregated"][pair_key] = per_layer_agg

with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(f"Wrote {OUT}")

# Print summary
print()
print("=== per-layer aggregated Jaccard summary ===")
for pair_key, layer_dict in result["per_layer_aggregated"].items():
    vals = list(layer_dict.values())
    print(f"  {pair_key}: layers={len(vals)} mean={sum(vals)/len(vals):.4f} min={min(vals):.4f} max={max(vals):.4f}")
PYEOF
