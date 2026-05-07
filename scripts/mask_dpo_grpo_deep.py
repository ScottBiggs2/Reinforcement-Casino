#!/usr/bin/env python
"""Driver for Phase-1 deep DPO-vs-GRPO mask comparison.

Loads N masks from --masks_json, streams one (or two) at a time, emits CSVs + PNGs.
CPU-only by default; pass --base_model to additionally compute the magnitude histogram
of disjoint regions on a chosen DPO/GRPO pair.

Outputs (under --output_dir):
  density_by_layer_subbucket.csv     long: (mask, layer, subbucket, density, n_total, n_active)
  density_by_attn_head.csv           long: (mask, layer, head_kind, head_idx, density)
  density_by_mlp_neuron.csv          long: (mask, layer, neuron_idx, gate_d, up_d, down_d, agree_3)
  pairwise_subbucket_jaccard.csv     long: (mask_a, mask_b, layer, subbucket, jaccard, intersection, union)
  disjoint_magnitude.csv             long: (pair, region, bin_lo, bin_hi, count) + summary rows
  density_layer_subbucket.png        per-mask density: layer × subbucket heatmap (one panel per mask)
  density_attn_head_q.png            per-mask q-head density: layer × head_idx heatmap
  pairwise_subbucket_jaccard.png     subbucket × pair heatmap (collapsed across layers)
  disjoint_magnitude.png             3 overlaid histograms for selected pair
  summary.md                         narrative + key numbers
"""
import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).parent.parent
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from cold_start.mask_dpo_grpo_deep import (
    classify_subbucket,
    decoder_layer_index,
    per_mask_density_layer_subbucket,
    per_mask_attn_head_density,
    per_mask_mlp_neuron_density,
    pair_jaccard_by_subbucket,
    disjoint_magnitude_histogram,
)

SUBBUCKET_ORDER = [
    "q", "k", "v", "o",
    "gate", "up", "down",
    "attn_norm", "mlp_norm", "final_norm",
    "embed", "lm_head", "other",
]


def _load_mask_dict(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "masks" in obj:
        return obj["masks"]
    return obj


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _density_heatmap(
    rows: List[Dict[str, Any]],
    mask_labels: List[str],
    out_path: Path,
    title: str,
    target_subbuckets: List[str] = SUBBUCKET_ORDER[:7],
) -> None:
    """layer × subbucket heatmap, one column per mask. Mask global density baseline shown in title."""
    n_masks = len(mask_labels)
    n_subbuckets = len(target_subbuckets)
    fig, axes = plt.subplots(1, n_masks, figsize=(2.4 * n_masks + 1.8, 6.5), sharey=True)
    if n_masks == 1:
        axes = [axes]
    by_mask: Dict[str, Dict[Tuple[int, str], Tuple[int, int]]] = {l: {} for l in mask_labels}
    for r in rows:
        l = r["mask"]
        if l not in by_mask:
            continue
        if r["layer"] < 0:
            continue
        if r["subbucket"] not in target_subbuckets:
            continue
        key = (r["layer"], r["subbucket"])
        prev = by_mask[l].get(key, (0, 0))
        by_mask[l][key] = (prev[0] + r["n_active"], prev[1] + r["n_total"])
    n_layers = max((r["layer"] for r in rows if r["layer"] >= 0), default=31) + 1
    vmax = 0.0
    grids: Dict[str, np.ndarray] = {}
    for l in mask_labels:
        g = np.zeros((n_layers, n_subbuckets), dtype=np.float32)
        for (layer, sb), (a, t) in by_mask[l].items():
            j = target_subbuckets.index(sb)
            g[layer, j] = a / t if t > 0 else 0.0
        grids[l] = g
        vmax = max(vmax, float(g.max())) if g.size else vmax
    for ax, l in zip(axes, mask_labels):
        im = ax.imshow(grids[l], aspect="auto", origin="lower", vmin=0, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(n_subbuckets))
        ax.set_xticklabels(target_subbuckets, rotation=45, ha="right")
        ax.set_title(l, fontsize=8)
        ax.set_xlabel("subbucket")
    axes[0].set_ylabel("decoder layer")
    fig.suptitle(title, fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="density (frac active)")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _attn_head_heatmap(
    rows: List[Dict[str, Any]],
    mask_labels: List[str],
    out_path: Path,
    head_kind: str,
    title: str,
) -> None:
    n_masks = len(mask_labels)
    fig, axes = plt.subplots(1, n_masks, figsize=(2.6 * n_masks + 1.8, 6.0), sharey=True)
    if n_masks == 1:
        axes = [axes]
    n_heads = 32 if head_kind in ("q", "o") else 8
    n_layers = 32
    grids: Dict[str, np.ndarray] = {l: np.zeros((n_layers, n_heads), dtype=np.float32) for l in mask_labels}
    for r in rows:
        if r["head_kind"] != head_kind:
            continue
        if r["mask"] not in grids:
            continue
        grids[r["mask"]][r["layer"], r["head_idx"]] = r["density"]
    vmax = max((float(g.max()) for g in grids.values()), default=0.0)
    for ax, l in zip(axes, mask_labels):
        im = ax.imshow(grids[l], aspect="auto", origin="lower", vmin=0, vmax=vmax, cmap="magma")
        ax.set_title(l, fontsize=8)
        ax.set_xlabel(f"{head_kind}-head idx")
    axes[0].set_ylabel("decoder layer")
    fig.suptitle(title, fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="density")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _pairwise_subbucket_heatmap(
    rows: List[Dict[str, Any]],
    out_path: Path,
    target_subbuckets: List[str] = SUBBUCKET_ORDER[:7],
) -> None:
    pair_keys: List[Tuple[str, str]] = []
    pair_to_vals: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in rows:
        if r["layer"] != -2:
            continue
        if r["subbucket"] not in target_subbuckets:
            continue
        pk = (r["mask_a"], r["mask_b"])
        if pk not in pair_to_vals:
            pair_to_vals[pk] = {}
            pair_keys.append(pk)
        pair_to_vals[pk][r["subbucket"]] = r["jaccard"]
    if not pair_keys:
        return
    grid = np.zeros((len(pair_keys), len(target_subbuckets)), dtype=np.float32)
    for i, pk in enumerate(pair_keys):
        for j, sb in enumerate(target_subbuckets):
            grid[i, j] = pair_to_vals[pk].get(sb, 0.0)
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(target_subbuckets), 0.4 * len(pair_keys) + 1.2))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.001, float(grid.max())))
    ax.set_xticks(range(len(target_subbuckets)))
    ax.set_xticklabels(target_subbuckets, rotation=45, ha="right")
    ax.set_yticks(range(len(pair_keys)))
    ax.set_yticklabels([f"{a}\n  vs {b}" for a, b in pair_keys], fontsize=7)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if grid[i, j] < grid.max() * 0.6 else "black")
    ax.set_title("Pairwise Jaccard by subbucket (layer-collapsed)")
    fig.colorbar(im, ax=ax, fraction=0.03, label="Jaccard")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _disjoint_magnitude_plot(rows: List[Dict[str, Any]], out_path: Path, title: str) -> None:
    by_region: Dict[str, List[Tuple[float, float, int]]] = {"both": [], "a_only": [], "b_only": []}
    summaries: Dict[str, Tuple[int, float]] = {}
    for r in rows:
        if r.get("summary"):
            summaries[r["region"]] = (r["count"], r.get("mean_abs", 0.0))
            continue
        by_region[r["region"]].append((r["bin_lo"], r["bin_hi"], r["count"]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for region, color in [("both", "tab:purple"), ("a_only", "tab:blue"), ("b_only", "tab:red")]:
        bins = by_region[region]
        if not bins:
            continue
        x = [(lo + hi) / 2 for lo, hi, _ in bins]
        y = np.array([c for _, _, c in bins], dtype=np.float64)
        if y.sum() > 0:
            y = y / y.sum()
        n, m = summaries.get(region, (0, 0.0))
        ax.plot(x, y, label=f"{region}  n={n}  mean|W|={m:.4f}", color=color, linewidth=1.6)
    ax.set_xlabel("|W_base|")
    ax.set_ylabel("frac of region")
    ax.set_title(title)
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.legend(fontsize=8)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--masks_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--base_model",
        default=None,
        help="HF id or local path; only loaded for --magnitude_pairs.",
    )
    p.add_argument(
        "--magnitude_pairs",
        default="",
        help="Comma-sep list of `labelA::labelB` pairs to run disjoint-magnitude on.",
    )
    p.add_argument("--n_q_heads", type=int, default=32)
    p.add_argument("--n_kv_heads", type=int, default=8)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--mag_n_bins", type=int, default=80)
    p.add_argument("--mag_max", type=float, default=0.2)
    p.add_argument(
        "--skip_mlp_neuron",
        action="store_true",
        help="MLP-neuron density is the largest CSV (32*14336 rows per mask). Skip if not needed.",
    )
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    masks_meta = json.load(open(args.masks_json))
    labels = [m["label"] for m in masks_meta]
    print(f"[deep] {len(labels)} masks: {labels}", flush=True)

    density_rows: List[Dict[str, Any]] = []
    head_rows: List[Dict[str, Any]] = []
    neuron_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for entry in masks_meta:
        label = entry["label"]
        path = entry["path"]
        print(f"[deep] loading {label} ...", flush=True)
        ti = time.time()
        masks = _load_mask_dict(path)
        for r in per_mask_density_layer_subbucket(masks):
            r["mask"] = label
            density_rows.append(r)
        for r in per_mask_attn_head_density(
            masks, args.n_q_heads, args.n_kv_heads, args.head_dim
        ):
            r["mask"] = label
            head_rows.append(r)
        if not args.skip_mlp_neuron:
            for r in per_mask_mlp_neuron_density(masks):
                r["mask"] = label
                r["agree_3"] = float(
                    (r["gate_density"] > 0)
                    and (r["up_density"] > 0)
                    and (r["down_density"] > 0)
                )
                neuron_rows.append(r)
        del masks
        gc.collect()
        print(f"[deep]   {label} done ({time.time()-ti:.1f}s, total elapsed {time.time()-t0:.1f}s)", flush=True)

    _write_csv(out / "density_by_layer_subbucket.csv", density_rows)
    _write_csv(out / "density_by_attn_head.csv", head_rows)
    if not args.skip_mlp_neuron:
        _write_csv(out / "density_by_mlp_neuron.csv", neuron_rows)

    _density_heatmap(
        density_rows, labels, out / "density_layer_subbucket.png",
        "Mask density per (layer, subbucket)",
    )
    _attn_head_heatmap(head_rows, labels, out / "density_attn_head_q.png", "q", "Q-head density (layer × head)")
    _attn_head_heatmap(head_rows, labels, out / "density_attn_head_o.png", "o", "O-proj density per query head (layer × head)")
    _attn_head_heatmap(head_rows, labels, out / "density_attn_head_kv.png", "v", "V-head density (layer × kv-head)")

    pair_rows: List[Dict[str, Any]] = []
    for i in range(len(masks_meta)):
        for j in range(i + 1, len(masks_meta)):
            la, lb = masks_meta[i]["label"], masks_meta[j]["label"]
            print(f"[deep] pair Jaccard: {la}  ⇄  {lb}", flush=True)
            ma = _load_mask_dict(masks_meta[i]["path"])
            mb = _load_mask_dict(masks_meta[j]["path"])
            for r in pair_jaccard_by_subbucket(ma, mb):
                r["mask_a"] = la
                r["mask_b"] = lb
                pair_rows.append(r)
            del ma, mb
            gc.collect()
    _write_csv(out / "pairwise_subbucket_jaccard.csv", pair_rows)
    _pairwise_subbucket_heatmap(pair_rows, out / "pairwise_subbucket_jaccard.png")

    if args.base_model and args.magnitude_pairs:
        print(f"[deep] loading base model {args.base_model} for magnitude histogram ...", flush=True)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        sd = {k: v.detach() for k, v in model.state_dict().items()}
        del model
        gc.collect()
        bins = torch.linspace(0.0, args.mag_max, args.mag_n_bins + 1)
        all_mag_rows: List[Dict[str, Any]] = []
        for spec in args.magnitude_pairs.split(","):
            spec = spec.strip()
            if not spec:
                continue
            la, lb = spec.split("::")
            pa = next(m["path"] for m in masks_meta if m["label"] == la)
            pb = next(m["path"] for m in masks_meta if m["label"] == lb)
            print(f"[deep] magnitude hist: {la}  ⇄  {lb}", flush=True)
            ma = _load_mask_dict(pa)
            mb = _load_mask_dict(pb)
            for r in disjoint_magnitude_histogram(ma, mb, sd, bins=bins):
                r["pair"] = f"{la}__VS__{lb}"
                all_mag_rows.append(r)
            _disjoint_magnitude_plot(
                [r for r in all_mag_rows if r.get("pair") == f"{la}__VS__{lb}"],
                out / f"disjoint_magnitude__{la}__VS__{lb}.png",
                f"|W_base| in disjoint regions: {la} (a) vs {lb} (b)",
            )
            del ma, mb
            gc.collect()
        _write_csv(out / "disjoint_magnitude.csv", all_mag_rows)

    summary_lines: List[str] = []
    summary_lines.append("# DPO vs GRPO mask deep-comparison — Phase 1\n")
    summary_lines.append(f"Inputs: {args.masks_json}\n")
    summary_lines.append(f"Masks: {labels}\n")
    summary_lines.append("")
    summary_lines.append("## Global density per mask\n")
    summary_lines.append("| mask | density (frac active) | n_active | n_total |")
    summary_lines.append("|---|---|---|---|")
    by_mask_global: Dict[str, Tuple[int, int]] = {l: (0, 0) for l in labels}
    for r in density_rows:
        a, t = by_mask_global[r["mask"]]
        by_mask_global[r["mask"]] = (a + r["n_active"], t + r["n_total"])
    for l in labels:
        a, t = by_mask_global[l]
        d = a / t if t > 0 else 0.0
        summary_lines.append(f"| {l} | {d:.4f} | {a:,} | {t:,} |")
    summary_lines.append("")
    summary_lines.append("## Subbucket density (layer-collapsed) per mask\n")
    summary_lines.append("| mask | " + " | ".join(SUBBUCKET_ORDER[:7]) + " | embed | lm_head |")
    summary_lines.append("|" + "---|" * (len(SUBBUCKET_ORDER[:7]) + 3))
    by_mask_sb: Dict[str, Dict[str, Tuple[int, int]]] = {l: {} for l in labels}
    for r in density_rows:
        d = by_mask_sb[r["mask"]].setdefault(r["subbucket"], (0, 0))
        by_mask_sb[r["mask"]][r["subbucket"]] = (d[0] + r["n_active"], d[1] + r["n_total"])
    for l in labels:
        cells = []
        for sb in SUBBUCKET_ORDER[:7] + ["embed", "lm_head"]:
            a, t = by_mask_sb[l].get(sb, (0, 0))
            d = a / t if t > 0 else 0.0
            cells.append(f"{d:.3f}")
        summary_lines.append("| " + l + " | " + " | ".join(cells) + " |")
    summary_lines.append("")
    summary_lines.append("## Pairwise layer-collapsed Jaccard by subbucket\n")
    pairs_seen: List[Tuple[str, str]] = []
    pair_subbucket: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in pair_rows:
        if r["layer"] != -2:
            continue
        pk = (r["mask_a"], r["mask_b"])
        if pk not in pair_subbucket:
            pair_subbucket[pk] = {}
            pairs_seen.append(pk)
        pair_subbucket[pk][r["subbucket"]] = r["jaccard"]
    summary_lines.append("| pair | " + " | ".join(SUBBUCKET_ORDER[:7]) + " | embed | lm_head |")
    summary_lines.append("|" + "---|" * (len(SUBBUCKET_ORDER[:7]) + 3))
    for pk in pairs_seen:
        cells = []
        for sb in SUBBUCKET_ORDER[:7] + ["embed", "lm_head"]:
            cells.append(f"{pair_subbucket[pk].get(sb, 0.0):.3f}")
        summary_lines.append(f"| {pk[0]}  ⇄  {pk[1]} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(summary_lines))

    print(f"[deep] DONE in {time.time()-t0:.1f}s. Outputs at: {out}", flush=True)


if __name__ == "__main__":
    main()
