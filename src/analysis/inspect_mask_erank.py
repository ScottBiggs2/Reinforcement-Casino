#!/usr/bin/env python3
"""
Compute effective rank (erank) of each mask, per layer.

Two modes:
- --structural: erank of the binary mask itself. Fast; captures the
  geometric pattern of what positions are kept.
- --weighted:   erank of (base_weight * mask). Needs the base model
  loaded; captures the actual information bandwidth that survives
  pruning.

Effective rank (Roy & Vetterli, 2007):
    erank(A) = exp(H(p_i))  where  p_i = σ_i^2 / Σ σ_j^2
    H is Shannon entropy in nats.

Usage:
    # Structural (no model needed, but SVD on each mask)
    python src/analysis/inspect_mask_erank.py \\
        --masks_json scripts/probe_pair_masks_dpo.json \\
                     scripts/probe_pair_masks_grpo.json \\
        --mode structural \\
        --output_json /scratch/$USER/mask_erank.json

    # Weighted (with base model)
    python src/analysis/inspect_mask_erank.py \\
        --masks_json scripts/probe_pair_masks_dpo.json \\
                     scripts/probe_pair_masks_grpo.json \\
        --mode weighted \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --output_json /scratch/$USER/mask_erank_weighted.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch


def load_mask(path: str) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data.get("masks", data) if isinstance(data, dict) else data


def _layer_index(name: str) -> int:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


def erank_from_singular_values(sigma: torch.Tensor) -> float:
    """erank = exp(-Σ p_i log p_i),  p_i = σ_i^2 / Σ σ_j^2."""
    s2 = sigma.float().pow(2)
    total = s2.sum().clamp_min(1e-30)
    p = s2 / total
    # Avoid log(0); entries with p=0 contribute 0.
    nonzero = p > 0
    h = -(p[nonzero] * p[nonzero].log()).sum()
    return float(torch.exp(h).item())


def compute_erank(tensor_2d: torch.Tensor, device: torch.device) -> float:
    """Shift to device, compute SVD, return erank. Falls back to float32."""
    x = tensor_2d.to(device=device, dtype=torch.float32)
    try:
        sigma = torch.linalg.svdvals(x)
    except RuntimeError as e:
        # Very large / degenerate tensors can fail on GPU — try CPU.
        sigma = torch.linalg.svdvals(x.cpu())
    return erank_from_singular_values(sigma)


def inspect_mask_structural(label: str, path: str, device: torch.device,
                            layer_pattern: str) -> dict:
    """Compute erank of the binary mask itself, per matched layer."""
    mask = load_mask(path)
    per_layer = {}
    for name, t in mask.items():
        if layer_pattern not in name:
            continue
        if t.dim() != 2:
            continue
        t0 = time.time()
        er = compute_erank(t, device)
        dt = time.time() - t0
        per_layer[name] = {
            "erank": er,
            "n_rows": t.shape[0],
            "n_cols": t.shape[1],
            "kept_frac": float(t.float().mean().item()),
            "rank_upper_bound": int(min(t.shape)),
            "seconds": dt,
        }
    return {"label": label, "mode": "structural", "per_layer": per_layer}


def inspect_mask_weighted(label: str, mask_path: str,
                          base_params: dict, device: torch.device,
                          layer_pattern: str) -> dict:
    """erank of (base_weight * mask)."""
    mask = load_mask(mask_path)
    per_layer = {}
    for name, m in mask.items():
        if layer_pattern not in name:
            continue
        if name not in base_params:
            continue
        if m.dim() != 2:
            continue
        w = base_params[name]
        if w.shape != m.shape:
            continue
        t0 = time.time()
        masked = w.to(torch.float32) * m.to(torch.float32)
        er = compute_erank(masked, device)
        dt = time.time() - t0
        per_layer[name] = {
            "erank": er,
            "n_rows": w.shape[0],
            "n_cols": w.shape[1],
            "kept_frac": float(m.float().mean().item()),
            "rank_upper_bound": int(min(w.shape)),
            "seconds": dt,
        }
    return {"label": label, "mode": "weighted", "per_layer": per_layer}


def print_summary(results: list):
    """One row per mask, cols = layer indices (for down_proj), value = erank."""
    # Gather all layer indices seen
    all_idx = set()
    for r in results:
        for name in r["per_layer"]:
            if "down_proj" not in name:
                continue
            idx = _layer_index(name)
            if idx >= 0:
                all_idx.add(idx)
    sorted_idx = sorted(all_idx)

    if not sorted_idx:
        print("(no down_proj layers matched)")
        return

    header = f"{'mask':28s}  " + "  ".join(f"{i:>6d}" for i in sorted_idx)
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        idx_to_er = {}
        for name, stats in r["per_layer"].items():
            if "down_proj" not in name:
                continue
            idx = _layer_index(name)
            if idx >= 0:
                idx_to_er[idx] = stats["erank"]
        cells = "  ".join(
            f"{idx_to_er.get(i, float('nan')):6.1f}" for i in sorted_idx
        )
        print(f"{r['label']:28s}  {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks_json", nargs="+", required=True)
    ap.add_argument("--mode", choices=["structural", "weighted"], default="structural")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="Base model for --mode weighted")
    ap.add_argument("--layer_pattern", default="down_proj",
                    help="Only compute erank for layers whose name contains this substring")
    ap.add_argument("--output_json", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[erank] device={device} mode={args.mode}")

    specs = []
    for jp in args.masks_json:
        with open(jp) as f:
            specs.extend(json.load(f))

    base_params = None
    if args.mode == "weighted":
        from transformers import AutoModelForCausalLM
        print(f"[erank] Loading base model for weighted erank: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cpu",
        )
        base_params = {n: p.detach() for n, p in model.named_parameters()}
        print(f"[erank] Loaded {len(base_params)} parameter tensors")

    results = []
    for spec in specs:
        if not os.path.exists(spec["path"]):
            print(f"[skip] {spec['label']}: missing {spec['path']}")
            continue
        print(f"[load] {spec['label']}", flush=True)
        if args.mode == "structural":
            r = inspect_mask_structural(spec["label"], spec["path"],
                                        device, args.layer_pattern)
        else:
            r = inspect_mask_weighted(spec["label"], spec["path"],
                                      base_params, device, args.layer_pattern)
        results.append(r)

    print_summary(results)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[out] → {args.output_json}")


if __name__ == "__main__":
    main()
