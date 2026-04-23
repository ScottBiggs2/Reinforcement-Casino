#!/usr/bin/env python3
"""
Inspect sparsity (kept-fraction) of each mask file — overall and per layer.

Useful for confirming that warm/cold masks were actually generated at the
expected sparsity level and for spotting layers where a mask has collapsed
to near-zero density (which then shows up as near_constant in probe runs).

Usage:
    python src/analysis/inspect_mask_sparsity.py \\
        --masks_json scripts/probe_pair_masks_cav_random_oracle.json.example

Prints a summary table to stdout and optionally writes a JSON with per-layer
densities to --output_json.
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch


def load_mask(path: str) -> dict:
    """Mirrors probe_analysis.load_mask — handles both raw and {masks: ...} formats."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "masks" in data:
        return data["masks"]
    return data


def _layer_index(name: str) -> int:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


def inspect_one(label: str, path: str) -> dict:
    """Return {overall_kept, n_layers, per_layer: {layer_name: kept_frac}, min_kept, max_kept}."""
    mask = load_mask(path)
    total_elems = 0
    total_kept = 0
    per_layer = OrderedDict()

    for name, t in mask.items():
        t = t.float()
        n = t.numel()
        kept = float(t.sum().item())
        total_elems += n
        total_kept += kept
        per_layer[name] = kept / n if n > 0 else float("nan")

    overall = total_kept / total_elems if total_elems > 0 else float("nan")
    vals = list(per_layer.values())
    return {
        "label": label,
        "path": path,
        "overall_kept": overall,
        "n_layers": len(per_layer),
        "min_kept": min(vals) if vals else float("nan"),
        "max_kept": max(vals) if vals else float("nan"),
        "per_layer": per_layer,
    }


def print_summary(results: list):
    """One-line-per-mask table sorted by kept fraction (densest first)."""
    print(f"\n{'label':28s}  {'kept%':>8s}  {'min%':>8s}  {'max%':>8s}  {'layers':>6s}")
    print("-" * 66)
    for r in sorted(results, key=lambda x: -x["overall_kept"]):
        print(
            f"{r['label']:28s}  "
            f"{r['overall_kept']*100:7.2f}%  "
            f"{r['min_kept']*100:7.2f}%  "
            f"{r['max_kept']*100:7.2f}%  "
            f"{r['n_layers']:6d}"
        )


def print_per_layer_by_depth(results: list):
    """Group per-layer kept% by MLP layer index across all masks — shows if
    some masks over-prune early vs late layers."""
    print("\nPer-layer kept% (rows=mask, cols=layer-index):")
    # Build {label: {layer_idx: kept}} restricted to down_proj entries.
    # Mask keys look like "model.layers.N.mlp.down_proj.weight" — match
    # substring rather than endswith to handle the trailing ".weight".
    by_label = {}
    all_indices = set()
    for r in results:
        idx_map = {}
        for name, frac in r["per_layer"].items():
            if "down_proj" not in name:
                continue
            idx = _layer_index(name)
            if idx >= 0:
                idx_map[idx] = frac
                all_indices.add(idx)
        by_label[r["label"]] = idx_map

    sorted_idx = sorted(all_indices)
    header = f"{'mask':28s}  " + "  ".join(f"{i:5d}" for i in sorted_idx)
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: -x["overall_kept"]):
        row = by_label[r["label"]]
        cells = "  ".join(
            f"{row.get(i, float('nan'))*100:4.1f}" if i in row else "  -  "
            for i in sorted_idx
        )
        print(f"{r['label']:28s}  {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks_json", nargs="+", required=True,
                    help="One or more JSON files with [{label, path}, ...] entries")
    ap.add_argument("--output_json", default=None,
                    help="Optional: dump per-layer densities here")
    args = ap.parse_args()

    specs = []
    for jp in args.masks_json:
        with open(jp) as f:
            specs.extend(json.load(f))

    results = []
    for spec in specs:
        if not os.path.exists(spec["path"]):
            print(f"[skip] {spec['label']:28s}  missing file: {spec['path']}")
            continue
        print(f"[load] {spec['label']}", flush=True)
        results.append(inspect_one(spec["label"], spec["path"]))

    print_summary(results)
    print_per_layer_by_depth(results)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        serializable = [
            {
                **{k: v for k, v in r.items() if k != "per_layer"},
                "per_layer": dict(r["per_layer"]),
            }
            for r in results
        ]
        with open(args.output_json, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n[out] per-layer densities → {args.output_json}")


if __name__ == "__main__":
    main()
