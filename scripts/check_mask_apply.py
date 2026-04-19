#!/usr/bin/env python3
"""Sanity check that a mask file has the expected key naming + shapes for
LLaMA-3.1-8B. Pure file inspection — does NOT load the model, so it runs
on login nodes with no GPU and trivial memory.

Usage:
    python scripts/check_mask_apply.py \
        --mask /scratch/xie.yiyi/rl_casino_masks/llama8b/warm_momentum_step50_sp97.5.pt
"""
import argparse
import re
from collections import Counter

import torch


# LLaMA-3.1-8B architecture (from HF config):
#   32 layers, hidden_size=4096, intermediate_size=14336
# MLP params: gate_proj [14336, 4096], up_proj [14336, 4096], down_proj [4096, 14336]
# Attention params: q_proj [4096, 4096], k_proj [1024, 4096], v_proj [1024, 4096],
#                   o_proj [4096, 4096]
EXPECTED_MLP_SHAPES = {
    "gate_proj": (14336, 4096),
    "up_proj": (14336, 4096),
    "down_proj": (4096, 14336),
}
EXPECTED_ATTN_SHAPES = {
    "q_proj": (4096, 4096),
    "k_proj": (1024, 4096),
    "v_proj": (1024, 4096),
    "o_proj": (4096, 4096),
}
N_LAYERS = 32

KEY_RE = re.compile(
    r"^model\.layers\.(\d+)\.(mlp|self_attn)\."
    r"(gate_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)\.weight$"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mask", required=True)
    args = p.parse_args()

    print(f"[load] mask: {args.mask}")
    data = torch.load(args.mask, map_location="cpu", weights_only=False)
    mask_dict = data["masks"] if isinstance(data, dict) and "masks" in data else data

    print(f"[load] total keys in mask: {len(mask_dict)}\n")

    # ---------- Key name structure ----------
    bad_names = []
    layer_idx_seen = set()
    by_proj = Counter()
    by_kind = Counter()
    for k in mask_dict:
        m = KEY_RE.match(k)
        if not m:
            bad_names.append(k)
            continue
        li, kind, proj = int(m.group(1)), m.group(2), m.group(3)
        layer_idx_seen.add(li)
        by_proj[proj] += 1
        by_kind[kind] += 1

    if bad_names:
        print(f"[check] UNEXPECTED key names: {len(bad_names)} entries")
        for k in bad_names[:5]:
            print(f"  {k}")
        if len(bad_names) > 5:
            print(f"  ... and {len(bad_names)-5} more")
    else:
        print("[check] all keys match LLaMA naming pattern "
              "model.layers.{i}.(mlp|self_attn).{proj}.weight ✓")

    print(f"[check] layer indices covered: "
          f"{len(layer_idx_seen)}/{N_LAYERS}  "
          f"{'✓' if len(layer_idx_seen) == N_LAYERS else '⚠️'}")
    missing_layers = sorted(set(range(N_LAYERS)) - layer_idx_seen)
    if missing_layers:
        print(f"  missing layer idx: {missing_layers[:10]}"
              f"{'...' if len(missing_layers) > 10 else ''}")

    print(f"[check] projections present: {dict(by_proj)}")
    print(f"[check] kinds present: {dict(by_kind)}")

    # ---------- Shape check ----------
    shape_bad = []
    total_elems, total_kept = 0, 0
    nonbinary = []
    for k, m in mask_dict.items():
        match = KEY_RE.match(k)
        if match:
            proj = match.group(3)
            expected = EXPECTED_MLP_SHAPES.get(proj) or EXPECTED_ATTN_SHAPES.get(proj)
            if expected is not None and tuple(m.shape) != expected:
                shape_bad.append((k, tuple(m.shape), expected))

        # binary + sparsity stats
        uniq = torch.unique(m)
        if not torch.all((uniq == 0) | (uniq == 1)):
            nonbinary.append((k, uniq.tolist()[:5]))
        total_elems += m.numel()
        total_kept += int((m == 1).sum().item())

    print()
    if shape_bad:
        print(f"[check] SHAPE MISMATCH: {len(shape_bad)} entries")
        for k, got, exp in shape_bad[:5]:
            print(f"  {k}: mask shape {got}, expected {exp}")
    else:
        print("[check] all shapes match LLaMA-3.1-8B expected dims ✓")

    if nonbinary:
        print(f"[check] NON-BINARY mask values in {len(nonbinary)} entries")
        for k, vs in nonbinary[:3]:
            print(f"  {k}: sample unique values {vs}")
    else:
        print("[check] all mask values are strictly {0, 1} ✓")

    print(f"\n[check] overall keep_frac: {total_kept / max(total_elems, 1):.4f}  "
          f"(sparsity = {1 - total_kept / max(total_elems, 1):.4f})")
    print(f"[check] total masked elements: {total_elems:,}")

    # ---------- Per-layer keep_frac (spot outliers) ----------
    per_layer_kept = Counter()
    per_layer_total = Counter()
    for k, m in mask_dict.items():
        match = KEY_RE.match(k)
        if not match:
            continue
        li = int(match.group(1))
        per_layer_kept[li] += int((m == 1).sum().item())
        per_layer_total[li] += m.numel()

    per_layer_frac = {
        li: per_layer_kept[li] / max(per_layer_total[li], 1)
        for li in sorted(per_layer_total)
    }
    if per_layer_frac:
        fracs = list(per_layer_frac.values())
        print(f"[check] per-layer keep_frac: "
              f"min={min(fracs):.4f}, max={max(fracs):.4f}, "
              f"mean={sum(fracs)/len(fracs):.4f}")

    print("\n[check] done.")


if __name__ == "__main__":
    main()
