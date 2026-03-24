#!/usr/bin/env python3
"""Batch-convert JSON analysis reports into per-layer CSV files.

Supports common report formats in this project:
  - CKA JSON with `per_layer_cka`
  - Jaccard JSON with `per_layer_jaccard`
  - Legacy Jaccard JSON with `per_layer`

If `mask_a` and `mask_b` paths are present and readable, this also computes:
  - layer-wise sparsity / density for both masks
  - layer-wise effective rank for both masks
  - per-layer Jaccard fallback when absent in JSON
"""

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")

STATS_CACHE: Dict[Tuple[str, str], dict] = {}
ERANK_CACHE: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]] = {}
SHAPE_CACHE: Dict[Tuple[str, str], str] = {}


def canonical_name(name: str) -> str:
    if name.endswith(".weight"):
        return name[:-7]
    return name


def parse_layer_index(name: str) -> Optional[int]:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def load_masks(path: str) -> Tuple[Dict[str, torch.Tensor], Optional[dict]]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict) and "masks" in data:
        return data["masks"], data.get("metadata")
    if isinstance(data, dict):
        return data, None
    raise ValueError(f"Unrecognized mask format: {path}")


def compute_basic_stats(mask_tensor: torch.Tensor) -> dict:
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


def effective_rank(
    mask_tensor: torch.Tensor,
    eps: float = 1e-12,
    sample_dim: int = 256,
) -> Tuple[Optional[float], Optional[float]]:
    if mask_tensor.ndim < 2:
        return None, None

    W = mask_tensor.float().reshape(mask_tensor.shape[0], -1)
    m, n = W.shape
    max_rank = min(m, n)
    if max_rank == 0:
        return 0.0, 0.0

    # For large matrices, approximate effective rank from a sampled submatrix.
    if m > sample_dim or n > sample_dim:
        rr = torch.randperm(m)[: min(sample_dim, m)]
        cc = torch.randperm(n)[: min(sample_dim, n)]
        W = W[rr][:, cc]
        max_rank = min(W.shape[0], W.shape[1])

    s = torch.linalg.svdvals(W)
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0, 0.0

    p = s / s.sum()
    entropy = -(p * torch.log(torch.clamp(p, min=eps))).sum().item()
    erank = float(math.exp(entropy))
    return erank, erank / float(max_rank)


def compute_jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.bool()
    bb = b.bool()
    union = int((aa | bb).sum().item())
    inter = int((aa & bb).sum().item())
    return (inter / union) if union > 0 else 0.0


def metric_dict(report: dict, *keys: str) -> Dict[str, float]:
    for key in keys:
        if key in report and isinstance(report[key], dict):
            return {canonical_name(k): float(v) for k, v in report[key].items() if v is not None}
    return {}


def summarize(report: dict, metric_name: str) -> dict:
    if metric_name in report and isinstance(report[metric_name], dict):
        return report[metric_name]

    prefixes = {
        "jaccard": ["aggregate_jaccard", "mean_jaccard", "min_jaccard", "max_jaccard"],
        "cka": ["mean", "min", "max", "n_layers", "n_skipped"],
    }
    out = {}
    for k in prefixes.get(metric_name, []):
        if k in report:
            out[k] = report[k]
    return out


def maybe_load_masks_from_report(report: dict) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
    mask_a = report.get("mask_a")
    mask_b = report.get("mask_b")
    if not mask_a or not mask_b:
        return None, None
    if not (os.path.isfile(mask_a) and os.path.isfile(mask_b)):
        return None, None

    try:
        a_raw, _ = load_masks(mask_a)
        b_raw, _ = load_masks(mask_b)
    except Exception:
        return None, None

    a = {canonical_name(k): v for k, v in a_raw.items()}
    b = {canonical_name(k): v for k, v in b_raw.items()}
    return a, b


def convert_one(json_path: Path, output_dir: Path) -> Optional[Path]:
    with json_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    per_layer_cka = metric_dict(report, "per_layer_cka")
    per_layer_jaccard = metric_dict(report, "per_layer_jaccard", "per_layer")

    masks_a, masks_b = maybe_load_masks_from_report(report)

    layer_names = set(per_layer_cka.keys()) | set(per_layer_jaccard.keys())
    if masks_a and masks_b:
        layer_names |= set(masks_a.keys()) & set(masks_b.keys())

    if not layer_names:
        return None

    rows = []
    mask_a_path = report.get("mask_a") or ""
    mask_b_path = report.get("mask_b") or ""

    for layer in sorted(layer_names):
        t_a = masks_a.get(layer) if masks_a else None
        t_b = masks_b.get(layer) if masks_b else None

        if t_a is not None:
            key_a = (mask_a_path, layer)
            if key_a not in STATS_CACHE:
                STATS_CACHE[key_a] = compute_basic_stats(t_a)
                ERANK_CACHE[key_a] = effective_rank(t_a)
                SHAPE_CACHE[key_a] = str(tuple(t_a.shape))
            s_a = STATS_CACHE[key_a]
            er_a, er_a_norm = ERANK_CACHE[key_a]
            shape_a = SHAPE_CACHE[key_a]
        else:
            s_a = {"n_params": None, "n_kept": None, "density": None, "sparsity": None}
            er_a, er_a_norm, shape_a = None, None, None

        if t_b is not None:
            key_b = (mask_b_path, layer)
            if key_b not in STATS_CACHE:
                STATS_CACHE[key_b] = compute_basic_stats(t_b)
                ERANK_CACHE[key_b] = effective_rank(t_b)
                SHAPE_CACHE[key_b] = str(tuple(t_b.shape))
            s_b = STATS_CACHE[key_b]
            er_b, er_b_norm = ERANK_CACHE[key_b]
            shape_b = SHAPE_CACHE[key_b]
        else:
            s_b = {"n_params": None, "n_kept": None, "density": None, "sparsity": None}
            er_b, er_b_norm, shape_b = None, None, None

        jaccard = per_layer_jaccard.get(layer)
        if jaccard is None and t_a is not None and t_b is not None:
            jaccard = compute_jaccard(t_a, t_b)

        cka = per_layer_cka.get(layer)

        rows.append(
            {
                "layer": layer,
                "layer_index": parse_layer_index(layer),
                "shape_a": shape_a,
                "shape_b": shape_b,
                "n_params": s_a["n_params"],
                "kept_a": s_a["n_kept"],
                "density_a": None if s_a["density"] is None else round(s_a["density"], 8),
                "sparsity_a": None if s_a["sparsity"] is None else round(s_a["sparsity"], 8),
                "effective_rank_a": None if er_a is None else round(er_a, 8),
                "effective_rank_a_norm": None if er_a_norm is None else round(er_a_norm, 8),
                "kept_b": s_b["n_kept"],
                "density_b": None if s_b["density"] is None else round(s_b["density"], 8),
                "sparsity_b": None if s_b["sparsity"] is None else round(s_b["sparsity"], 8),
                "effective_rank_b": None if er_b is None else round(er_b, 8),
                "effective_rank_b_norm": None if er_b_norm is None else round(er_b_norm, 8),
                "jaccard": None if jaccard is None else round(float(jaccard), 8),
                "cka": None if cka is None else round(float(cka), 8),
                "source_json": str(json_path),
            }
        )

    rows.sort(key=lambda r: (r["layer_index"] if r["layer_index"] is not None else 10**9, r["layer"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{json_path.stem}.csv"

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
        "source_json",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Optional compact summary file for quick filtering in BI tools.
    summary_path = output_dir / f"{json_path.stem}_summary.csv"
    cka_summary = summarize(report, "cka")
    jaccard_summary = summarize(report, "jaccard")

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_json",
                "mask_a",
                "mask_b",
                "compare",
                "label_a",
                "label_b",
                "n_samples",
                "seed",
                "cka_mean",
                "cka_min",
                "cka_max",
                "jaccard_aggregate",
                "jaccard_mean",
                "jaccard_min",
                "jaccard_max",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_json": str(json_path),
                "mask_a": report.get("mask_a"),
                "mask_b": report.get("mask_b"),
                "compare": report.get("compare"),
                "label_a": report.get("label_a"),
                "label_b": report.get("label_b"),
                "n_samples": report.get("n_samples"),
                "seed": report.get("seed"),
                "cka_mean": cka_summary.get("mean"),
                "cka_min": cka_summary.get("min"),
                "cka_max": cka_summary.get("max"),
                "jaccard_aggregate": jaccard_summary.get("aggregate", jaccard_summary.get("aggregate_jaccard")),
                "jaccard_mean": jaccard_summary.get("mean", jaccard_summary.get("mean_jaccard")),
                "jaccard_min": jaccard_summary.get("min", jaccard_summary.get("min_jaccard")),
                "jaccard_max": jaccard_summary.get("max", jaccard_summary.get("max_jaccard")),
            }
        )

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert all JSON reports in a directory to per-layer CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, default="masks", help="Directory containing JSON reports.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output CSV directory (default: same as input).")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subdirectories for JSON files.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    pattern = "**/*.json" if args.recursive else "*.json"
    json_files = sorted(input_dir.glob(pattern))

    converted = []
    skipped = []
    for jp in json_files:
        print(f"Converting: {jp}")
        out = convert_one(jp, output_dir)
        if out is None:
            skipped.append(jp)
        else:
            converted.append(out)

    print(f"Found JSON files: {len(json_files)}")
    print(f"Converted:        {len(converted)}")
    print(f"Skipped:          {len(skipped)}")
    for p in converted:
        print(f"  ✓ {p}")
    for p in skipped:
        print(f"  - (no per-layer metrics) {p}")


if __name__ == "__main__":
    main()
