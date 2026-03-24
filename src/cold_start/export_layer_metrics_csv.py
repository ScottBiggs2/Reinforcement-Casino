#!/usr/bin/env python3
"""Export per-layer mask comparison metrics to CSV.

This script is designed for charting and analysis workflows.

For each shared layer between `mask_a` and `mask_b`, it exports:
  - per-mask sparsity / density
  - per-mask effective rank (entropy-based, from singular values)
  - per-layer Jaccard similarity
  - optional per-layer CKA (if a CKA JSON is provided)
"""

import argparse
import csv
import json
import math
import os
import re
from typing import Dict, Optional, Tuple

import torch


LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def load_masks(path: str) -> Tuple[Dict[str, torch.Tensor], Optional[dict]]:
    """Load a mask dict from a `.pt` file."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict) and "masks" in data:
        return data["masks"], data.get("metadata")
    if isinstance(data, dict):
        return data, None
    raise ValueError(f"Unrecognized mask format: {path}")


def canonical_name(name: str) -> str:
    """Normalize parameter/layer names so JSON sources can be merged cleanly."""
    if name.endswith(".weight"):
        return name[:-7]
    return name


def parse_layer_index(name: str) -> Optional[int]:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def compute_basic_stats(mask_tensor: torch.Tensor) -> dict:
    """Compute count, density, and sparsity stats for a mask tensor."""
    m = mask_tensor.bool()
    total = int(m.numel())
    kept = int(m.sum().item())
    density = (kept / total) if total > 0 else 0.0
    sparsity = 1.0 - density
    return {
        "n_params": total,
        "n_kept": kept,
        "density": density,
        "sparsity": sparsity,
    }


def effective_rank(mask_tensor: torch.Tensor, eps: float = 1e-12) -> Tuple[Optional[float], Optional[float]]:
    """Compute entropy-based effective rank and normalized effective rank.

    Effective rank is defined as:
      erank(W) = exp(H(p)), where p_i = sigma_i / sum_j sigma_j
      and sigma_i are singular values of W.

    Returns (erank, erank_normalized). For tensors with <2 dims, returns (None, None).
    """
    if mask_tensor.ndim < 2:
        return None, None

    W = mask_tensor.float().reshape(mask_tensor.shape[0], -1)
    m, n = W.shape
    max_rank = min(m, n)
    if max_rank == 0:
        return 0.0, 0.0

    s = torch.linalg.svdvals(W)
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0, 0.0

    p = s / s.sum()
    entropy = -(p * torch.log(torch.clamp(p, min=eps))).sum().item()
    erank = float(math.exp(entropy))
    return erank, erank / float(max_rank)


def compute_per_layer_jaccard(masks_a: Dict[str, torch.Tensor], masks_b: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute per-layer Jaccard for shared layers."""
    out = {}
    for name in sorted(set(masks_a.keys()) & set(masks_b.keys())):
        a = masks_a[name].bool()
        b = masks_b[name].bool()
        union = int((a | b).sum().item())
        inter = int((a & b).sum().item())
        out[name] = (inter / union) if union > 0 else 0.0
    return out


def load_per_layer_json(path: Optional[str], key: str) -> Dict[str, float]:
    """Load a per-layer metric dict from JSON and canonicalize names."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get(key, {})
    return {canonical_name(k): float(v) for k, v in raw.items() if v is not None}


def default_output_path(mask_a: str, mask_b: str) -> str:
    base_a = os.path.splitext(os.path.basename(mask_a))[0]
    base_b = os.path.splitext(os.path.basename(mask_b))[0]
    try:
        out_dir = os.path.commonpath([os.path.abspath(mask_a), os.path.abspath(mask_b)])
    except ValueError:
        out_dir = "."
    return os.path.join(out_dir, f"layer_metrics_{base_a}_vs_{base_b}.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Export per-layer mask metrics (sparsity, effective rank, Jaccard, optional CKA) to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mask_a", type=str, help="First mask file (.pt)")
    parser.add_argument("mask_b", type=str, help="Second mask file (.pt)")
    parser.add_argument("--cka-json", type=str, default=None, help="Optional CKA JSON containing `per_layer_cka`.")
    parser.add_argument("--jaccard-json", type=str, default=None, help="Optional Jaccard JSON containing `per_layer_jaccard`.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV path.")
    args = parser.parse_args()

    for path in (args.mask_a, args.mask_b):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Mask file not found: {path}")

    masks_a, meta_a = load_masks(args.mask_a)
    masks_b, meta_b = load_masks(args.mask_b)

    common = sorted(set(masks_a.keys()) & set(masks_b.keys()))
    if not common:
        raise ValueError("The two masks share no common layer names.")

    if args.jaccard_json:
        per_layer_jaccard = load_per_layer_json(args.jaccard_json, "per_layer_jaccard")
    else:
        per_layer_jaccard = {
            canonical_name(k): v
            for k, v in compute_per_layer_jaccard(masks_a, masks_b).items()
        }

    per_layer_cka = load_per_layer_json(args.cka_json, "per_layer_cka")

    rows = []
    for name in common:
        ca = canonical_name(name)
        t_a = masks_a[name]
        t_b = masks_b[name]

        s_a = compute_basic_stats(t_a)
        s_b = compute_basic_stats(t_b)

        er_a, er_a_norm = effective_rank(t_a)
        er_b, er_b_norm = effective_rank(t_b)

        rows.append(
            {
                "layer": ca,
                "layer_index": parse_layer_index(ca),
                "shape_a": str(tuple(t_a.shape)),
                "shape_b": str(tuple(t_b.shape)),
                "n_params": s_a["n_params"],
                "kept_a": s_a["n_kept"],
                "density_a": round(s_a["density"], 8),
                "sparsity_a": round(s_a["sparsity"], 8),
                "effective_rank_a": None if er_a is None else round(er_a, 8),
                "effective_rank_a_norm": None if er_a_norm is None else round(er_a_norm, 8),
                "kept_b": s_b["n_kept"],
                "density_b": round(s_b["density"], 8),
                "sparsity_b": round(s_b["sparsity"], 8),
                "effective_rank_b": None if er_b is None else round(er_b, 8),
                "effective_rank_b_norm": None if er_b_norm is None else round(er_b_norm, 8),
                "jaccard": round(per_layer_jaccard.get(ca, float("nan")), 8),
                "cka": round(per_layer_cka.get(ca, float("nan")), 8),
            }
        )

    rows.sort(
        key=lambda r: (
            r["layer_index"] if r["layer_index"] is not None else 10**9,
            r["layer"],
        )
    )

    output_path = args.output or default_output_path(args.mask_a, args.mask_b)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    fieldnames = [
        "layer",
        "layer_index",
        "shape_a",
        "shape_b",
        "n_params",
        "kept_a",
        "density_a",
        "sparsity_a",
        "effective_rank_a",
        "effective_rank_a_norm",
        "kept_b",
        "density_b",
        "sparsity_b",
        "effective_rank_b",
        "effective_rank_b_norm",
        "jaccard",
        "cka",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} layer rows to: {output_path}")
    if meta_a:
        print(f"metadata_a: {meta_a}")
    if meta_b:
        print(f"metadata_b: {meta_b}")
    if args.cka_json:
        print(f"CKA source: {args.cka_json}")
    if args.jaccard_json:
        print(f"Jaccard source: {args.jaccard_json}")


if __name__ == "__main__":
    main()
