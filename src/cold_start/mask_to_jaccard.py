#!/usr/bin/env python3
"""Compare two cold-start mask files with per-layer and global Jaccard scores."""

import argparse
import json
import os
import sys

import torch

def load_masks(path):
    """Load a masks dict from a .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict) and "masks" in data:
        return data["masks"], data.get("metadata")
    if isinstance(data, dict):
        return data, None
    raise ValueError(f"Unrecognized mask format: {path}")

def compute_jaccard(masks_a, masks_b, device="cpu"):
    """Compute per-layer and aggregate Jaccard scores for two mask dicts."""
    per_layer = {}
    total_intersection = 0
    total_union = 0

    common_keys = set(masks_a.keys()) & set(masks_b.keys())
    if not common_keys:
        return {
            "aggregate_jaccard": 0.0,
            "mean_jaccard": 0.0,
            "min_jaccard": 0.0,
            "max_jaccard": 0.0,
            "per_layer": {},
            "note": "The two masks share no common layer names.",
        }

    for name in sorted(common_keys):
        a = masks_a[name].to(device).bool()
        b = masks_b[name].to(device).bool()
        inter = (a & b).sum().item()
        union = (a | b).sum().item()
        jaccard = inter / union if union > 0 else 0.0
        per_layer[name] = round(jaccard, 6)
        total_intersection += inter
        total_union += union

    aggregate = total_intersection / total_union if total_union > 0 else 0.0
    vals = list(per_layer.values())
    return {
        "aggregate_jaccard": round(aggregate, 6),
        "mean_jaccard":      round(sum(vals) / len(vals), 6),
        "min_jaccard":       round(min(vals), 6),
        "max_jaccard":       round(max(vals), 6),
        "per_layer":         per_layer,
        "n_layers":          len(per_layer),
        "total_intersection": int(total_intersection),
        "total_union":        int(total_union),
    }

def main():
    parser = argparse.ArgumentParser(
        description="Compute Jaccard similarity between two cold-start mask files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mask_a", type=str, help="First mask file (.pt)")
    parser.add_argument("mask_b", type=str, help="Second mask file (.pt)")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON path. Auto-generated from mask filenames if omitted.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Computation device.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    if not os.path.isfile(args.mask_a):
        print(f"Error: file not found: {args.mask_a}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.mask_b):
        print(f"Error: file not found: {args.mask_b}", file=sys.stderr)
        sys.exit(1)

    print("Loading mask A:", args.mask_a)
    masks_a, meta_a = load_masks(args.mask_a)
    print("Loading mask B:", args.mask_b)
    masks_b, meta_b = load_masks(args.mask_b)

    jaccard = compute_jaccard(masks_a, masks_b, device=args.device)

    report = {
        "mask_a": os.path.abspath(args.mask_a),
        "mask_b": os.path.abspath(args.mask_b),
        "jaccard": {
            "aggregate": jaccard["aggregate_jaccard"],
            "mean":      jaccard["mean_jaccard"],
            "min":       jaccard["min_jaccard"],
            "max":       jaccard["max_jaccard"],
            "n_layers":          jaccard.get("n_layers"),
            "total_intersection": jaccard.get("total_intersection"),
            "total_union":        jaccard.get("total_union"),
        },
        "per_layer_jaccard": jaccard["per_layer"],
    }
    if meta_a:
        report["metadata_a"] = meta_a
    if meta_b:
        report["metadata_b"] = meta_b

    if args.output is None:
        base_a = os.path.splitext(os.path.basename(args.mask_a))[0]
        base_b = os.path.splitext(os.path.basename(args.mask_b))[0]
        try:
            out_dir = os.path.commonpath(
                [os.path.abspath(args.mask_a), os.path.abspath(args.mask_b)]
            )
        except ValueError:
            out_dir = "."
        args.output = os.path.join(out_dir, f"jaccard_{base_a}_vs_{base_b}.json")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nJaccard summary:")
    print(f"  aggregate: {jaccard['aggregate_jaccard']:.4f}")
    print(f"  mean:      {jaccard['mean_jaccard']:.4f}")
    print(f"  min:       {jaccard['min_jaccard']:.4f}")
    print(f"  max:       {jaccard['max_jaccard']:.4f}")
    print(f"\nReport written to: {args.output}")


if __name__ == "__main__":
    main()
