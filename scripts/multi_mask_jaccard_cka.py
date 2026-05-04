"""
Pairwise Jaccard + mask-vs-mask CKA across N masks (Llama-3.1-8B-Instruct).

Loads the base model + calibration data ONCE; for each mask, applies → collects
activations at `--layer_substr` layers → restores. Then computes all C(N, 2)
pairwise Jaccards (cheap, CPU set ops) + pairwise CKA from the cached activations.

Per-pair Jaccard uses `mask_jaccard_aggregates.extended_jaccard_report` so each
pair gets bucketed by parameter type (attn / mlp / norm / other) and decoder layer.

Outputs (all under --output_dir):
  multi_mask_jaccard_cka.json    full structured results (back-compat)
  jaccard_matrix.csv             N×N symmetric (label headers)
  cka_matrix.csv                 N×N symmetric
  family_matrix.csv              family×family mean Jaccard / CKA
  pairs_long.csv                 one row per pair: {family,dataset,step,sparsity}_{a,b}
                                  + jaccard{global,attn,mlp,norm}, cka_mean/min/max
  jaccard_heatmap.png            10×10 heatmap with family separators
  cka_heatmap.png                10×10 heatmap
  per_decoder_layer.png          line plot of Jaccard per decoder layer, one line per pair

masks.json schema (extra fields are optional; missing fields default to "unknown"):
  [
    {
      "label":    "DPO-LightR1-s500-sp97.5",
      "path":     "/scratch/.../oracle_dpo_lightr1_step500_sp97.5.pt",
      "family":   "DPO-oracle",
      "dataset":  "LightR1",
      "step":     500,
      "sparsity": 97.5
    },
    ...
  ]
"""
import argparse
import csv
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

_SRC = Path(__file__).parent.parent / "src"
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from cold_start.utils.activation_hooks import FeatureExtractor
from cold_start.mask_to_cka import (
    apply_mask,
    collect_activations,
    compute_layerwise_cka,
    load_calibration_samples,
    load_masks,
    restore_weights,
    set_seed,
    DEFAULT_CALIBRATION_DATASET,
)
from cold_start.mask_jaccard_aggregates import extended_jaccard_report


# ---------- helpers ----------

def _load_mask_dict(path):
    """Load a mask file and return the {param_name: tensor} dict (handles both wrapped + raw)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "masks" in obj:
        return obj["masks"]
    return obj


def jaccard_global(masks_a, masks_b):
    """Size-weighted global Jaccard + per-layer min/mean/max over common keys."""
    inter, union = 0, 0
    per = []
    for k in set(masks_a) & set(masks_b):
        a = masks_a[k].bool()
        b = masks_b[k].bool()
        i = (a & b).sum().item()
        u = (a | b).sum().item()
        inter += i
        union += u
        per.append(i / max(u, 1))
    agg = inter / max(union, 1)
    return {
        "aggregate": agg,
        "per_layer_mean": float(np.mean(per)) if per else 0.0,
        "per_layer_min": float(min(per)) if per else 0.0,
        "per_layer_max": float(max(per)) if per else 0.0,
        "intersect": inter,
        "union": union,
    }


def write_matrix_csv(path, labels, get_value):
    """Write a symmetric N×N matrix CSV with `labels` as header + leading column."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for la in labels:
            row = [la]
            for lb in labels:
                if la == lb:
                    row.append("1.0")
                else:
                    row.append(f"{get_value(la, lb):.6f}")
            w.writerow(row)


def write_family_matrix_csv(path, family_names, family_grid):
    """family×family CSV with mean values."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + family_names)
        for fa in family_names:
            row = [fa]
            for fb in family_names:
                v = family_grid.get((fa, fb))
                row.append(f"{v:.6f}" if v is not None else "")
            w.writerow(row)


def family_aggregate(entries, pair_value):
    """Mean of pair_value(la,lb) grouped by (family_a, family_b). Within-family includes self-loop = 1.0."""
    families = sorted({e["family"] for e in entries})
    family_to_labels = {f: [e["label"] for e in entries if e["family"] == f] for f in families}
    grid = {}
    for fa in families:
        for fb in families:
            vals = []
            for la in family_to_labels[fa]:
                for lb in family_to_labels[fb]:
                    if la == lb:
                        vals.append(1.0)
                    else:
                        v = pair_value(la, lb)
                        if v is not None:
                            vals.append(v)
            grid[(fa, fb)] = float(np.mean(vals)) if vals else None
    return families, grid


def heatmap(path, labels, get_value, title, cmap, vmin=0.0, vmax=1.0, family_separators=None):
    n = len(labels)
    M = np.eye(n)
    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if i != j:
                v = get_value(la, lb)
                M[i, j] = v if v is not None else np.nan
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * n + 4), max(6, 0.55 * n + 3)))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v < (vmin + vmax) / 2 else "black")
    if family_separators:
        for s in family_separators:
            ax.axhline(s - 0.5, color="black", linewidth=0.6)
            ax.axvline(s - 0.5, color="black", linewidth=0.6)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_decoder_layer_lines(path, pair_keys, pair_to_per_layer, title):
    """One line per pair, x = decoder layer index 0..31."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for key in pair_keys:
        per = pair_to_per_layer.get(key, {})
        # keys are decoder layer indices as strings (or "non_decoder")
        layer_indices = sorted(int(k) for k in per if k.isdigit())
        if not layer_indices:
            continue
        ys = [per[str(i)]["aggregate_jaccard"] for i in layer_indices]
        ax.plot(layer_indices, ys, marker="o", markersize=3, linewidth=1, label=key)
    ax.set_xlabel("decoder layer index")
    ax.set_ylabel("size-weighted Jaccard (within layer)")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2, loc="best")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--masks_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layer_substr", default="down_proj")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calibration_dataset", default=DEFAULT_CALIBRATION_DATASET)
    p.add_argument("--skip_cka", action="store_true",
                   help="Skip activation-CKA (Jaccard only). Useful for quick CPU-only runs.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    with open(args.masks_json) as f:
        masks_list = json.load(f)
    # Normalize entries: ensure family/dataset/step/sparsity exist
    for e in masks_list:
        e.setdefault("family", "unknown")
        e.setdefault("dataset", "unknown")
        e.setdefault("step", None)
        e.setdefault("sparsity", None)

    labels = [e["label"] for e in masks_list]
    pairs = list(itertools.combinations(labels, 2))
    label_to_entry = {e["label"]: e for e in masks_list}

    print(f"Will analyze {len(masks_list)} masks → {len(pairs)} pairs:")
    for m in masks_list:
        print(f"  {m['label']:<35s} ({m['family']}/{m['dataset']}/step={m['step']}/sp={m['sparsity']})")
    print()

    # ---- Pairwise Jaccard (cheap, do first) ----
    print("===== Part 1: Pairwise Jaccard =====")
    print("Loading mask dicts to CPU...")
    mask_dicts = {e["label"]: _load_mask_dict(e["path"]) for e in masks_list}

    jaccard_results = {}        # full structured (back-compat)
    jaccard_global_lookup = {}  # (la,lb) -> float (aggregate)
    jaccard_buckets = {}        # (la,lb) -> by_param_bucket dict
    jaccard_decoder = {}        # (la,lb) -> by_decoder_layer dict
    for la, lb in pairs:
        ma, mb = mask_dicts[la], mask_dicts[lb]
        glob = jaccard_global(ma, mb)
        ext = extended_jaccard_report(
            ma, mb,
            include_param_buckets=True,
            include_decoder_layers=True,
        )
        full = {**glob, **ext}
        key = f"{la} ⇄ {lb}"
        jaccard_results[key] = full
        jaccard_global_lookup[(la, lb)] = jaccard_global_lookup[(lb, la)] = glob["aggregate"]
        jaccard_buckets[(la, lb)] = ext.get("by_param_bucket", {})
        jaccard_decoder[(la, lb)] = ext.get("by_decoder_layer", {})
        attn_j = jaccard_buckets[(la, lb)].get("attn", {}).get("aggregate_jaccard")
        mlp_j = jaccard_buckets[(la, lb)].get("mlp", {}).get("aggregate_jaccard")
        attn_s = f"{attn_j:.4f}" if attn_j is not None else " n/a "
        mlp_s = f"{mlp_j:.4f}" if mlp_j is not None else " n/a "
        print(f"  J({la}, {lb}): glob={glob['aggregate']:.4f}  attn={attn_s}  mlp={mlp_s}")
    del mask_dicts
    print()

    # ---- Optional CKA ----
    cka_results = {}
    cka_global_lookup = {}
    if not args.skip_cka:
        print("===== Part 2: CKA setup =====")
        print(f"Loading model: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=None
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        print("Snapshotting base weights...")
        original = {n: p.detach().clone() for n, p in model.named_parameters()}

        print(f"Loading {args.n_samples} calibration samples from {args.calibration_dataset!r}")
        chosen, _rejected = load_calibration_samples(
            n_samples=args.n_samples, seed=args.seed, dataset_name=args.calibration_dataset
        )
        texts = chosen[: args.n_samples]

        extractor = FeatureExtractor()
        extractor.register(model)

        print()
        print("===== Part 3: Collect activations (one forward pass per mask) =====")
        acts = {}
        for entry in masks_list:
            label = entry["label"]
            print(f"\n--- mask: {label}")
            masks_tensor, _meta = load_masks(entry["path"])
            apply_mask(model, masks_tensor)
            t0 = time.time()
            a = collect_activations(model, extractor, tokenizer, texts, device,
                                    batch_size=args.batch_size, max_length=args.max_length)
            print(f"  collected in {time.time() - t0:.1f}s, layers cached={len(a)}")
            acts[label] = a
            restore_weights(model, original)
        del original

        print()
        print("===== Part 4: Pairwise CKA =====")
        for la, lb in pairs:
            cka = compute_layerwise_cka(acts[la], acts[lb], device="cpu")
            agg = {
                "per_layer_mean": float(cka.get("mean_cka", 0.0)),
                "per_layer_min": float(cka.get("min_cka", 0.0)),
                "per_layer_max": float(cka.get("max_cka", 0.0)),
                "n_layers": int(cka.get("n_layers", 0)),
                "n_skipped": int(cka.get("n_skipped", 0)),
                "per_layer": cka.get("per_layer", {}),
            }
            key = f"{la} ⇄ {lb}"
            cka_results[key] = agg
            cka_global_lookup[(la, lb)] = cka_global_lookup[(lb, la)] = agg["per_layer_mean"]
            print(f"  CKA({la}, {lb}): mean={agg['per_layer_mean']:.4f}  min={agg['per_layer_min']:.4f}")

    # ---- Save consolidated JSON (back-compat) ----
    out = {
        "model": args.model,
        "layer_substr": args.layer_substr,
        "n_samples": args.n_samples,
        "calibration_dataset": args.calibration_dataset,
        "masks": masks_list,
        "jaccard": jaccard_results,
        "cka": cka_results,
    }
    out_path = os.path.join(args.output_dir, "multi_mask_jaccard_cka.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    # ---- Matrix CSVs ----
    j_csv = os.path.join(args.output_dir, "jaccard_matrix.csv")
    write_matrix_csv(j_csv, labels, lambda a, b: jaccard_global_lookup.get((a, b), 0.0))
    print(f"Wrote {j_csv}")

    if cka_global_lookup:
        c_csv = os.path.join(args.output_dir, "cka_matrix.csv")
        write_matrix_csv(c_csv, labels, lambda a, b: cka_global_lookup.get((a, b), 0.0))
        print(f"Wrote {c_csv}")

    # ---- Family matrices ----
    fams_j, grid_j = family_aggregate(masks_list, lambda a, b: jaccard_global_lookup.get((a, b)))
    fj_csv = os.path.join(args.output_dir, "family_matrix_jaccard.csv")
    write_family_matrix_csv(fj_csv, fams_j, grid_j)
    print(f"Wrote {fj_csv}")

    if cka_global_lookup:
        fams_c, grid_c = family_aggregate(masks_list, lambda a, b: cka_global_lookup.get((a, b)))
        fc_csv = os.path.join(args.output_dir, "family_matrix_cka.csv")
        write_family_matrix_csv(fc_csv, fams_c, grid_c)
        print(f"Wrote {fc_csv}")

    # ---- pairs_long.csv ----
    long_csv = os.path.join(args.output_dir, "pairs_long.csv")
    with open(long_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "label_a", "label_b",
            "family_a", "family_b",
            "dataset_a", "dataset_b",
            "step_a", "step_b",
            "sparsity_a", "sparsity_b",
            "jaccard_global",
            "jaccard_attn", "jaccard_mlp", "jaccard_norm", "jaccard_other",
            "cka_mean", "cka_min", "cka_max",
        ])
        for la, lb in pairs:
            ea, eb = label_to_entry[la], label_to_entry[lb]
            buckets = jaccard_buckets[(la, lb)]
            cka = cka_results.get(f"{la} ⇄ {lb}", {})
            w.writerow([
                la, lb,
                ea["family"], eb["family"],
                ea["dataset"], eb["dataset"],
                ea["step"], eb["step"],
                ea["sparsity"], eb["sparsity"],
                f"{jaccard_global_lookup[(la, lb)]:.6f}",
                f"{buckets.get('attn', {}).get('aggregate_jaccard', ''):.6f}" if buckets.get("attn") else "",
                f"{buckets.get('mlp', {}).get('aggregate_jaccard', ''):.6f}" if buckets.get("mlp") else "",
                f"{buckets.get('norm', {}).get('aggregate_jaccard', ''):.6f}" if buckets.get("norm") else "",
                f"{buckets.get('other', {}).get('aggregate_jaccard', ''):.6f}" if buckets.get("other") else "",
                f"{cka.get('per_layer_mean', ''):.6f}" if cka else "",
                f"{cka.get('per_layer_min', ''):.6f}" if cka else "",
                f"{cka.get('per_layer_max', ''):.6f}" if cka else "",
            ])
    print(f"Wrote {long_csv}")

    # ---- Heatmaps ----
    # Sort labels by family for cleaner block structure
    sorted_entries = sorted(masks_list, key=lambda e: (e["family"], e["dataset"], e["step"] or 0, e["label"]))
    sorted_labels = [e["label"] for e in sorted_entries]
    # family separator indices
    family_separators = []
    for i in range(1, len(sorted_entries)):
        if sorted_entries[i]["family"] != sorted_entries[i - 1]["family"]:
            family_separators.append(i)

    j_png = os.path.join(args.output_dir, "jaccard_heatmap.png")
    heatmap(j_png, sorted_labels,
            lambda a, b: jaccard_global_lookup.get((a, b), 0.0),
            "Pairwise Jaccard (size-weighted global, sorted by family)",
            cmap="viridis", vmin=0.0, vmax=1.0,
            family_separators=family_separators)
    print(f"Wrote {j_png}")

    if cka_global_lookup:
        c_png = os.path.join(args.output_dir, "cka_heatmap.png")
        heatmap(c_png, sorted_labels,
                lambda a, b: cka_global_lookup.get((a, b), 0.0),
                "Pairwise CKA (per-layer mean, sorted by family)",
                cmap="magma", vmin=0.0, vmax=1.0,
                family_separators=family_separators)
        print(f"Wrote {c_png}")

    # ---- Per-decoder-layer Jaccard line plot ----
    pl_png = os.path.join(args.output_dir, "per_decoder_layer.png")
    pair_keys = [f"{la} ⇄ {lb}" for la, lb in pairs]
    pair_to_per_layer = {f"{la} ⇄ {lb}": jaccard_decoder[(la, lb)] for la, lb in pairs}
    per_decoder_layer_lines(pl_png, pair_keys, pair_to_per_layer,
                            "Per-decoder-layer Jaccard (one line per pair)")
    print(f"Wrote {pl_png}")

    # ---- Console summary ----
    print()
    print("===== SUMMARY (sorted by Jaccard, descending) =====")
    print(f"{'pair':<70s}  {'Jaccard':>8s}  {'CKA':>8s}")
    print("-" * 92)
    rows = []
    for la, lb in pairs:
        j = jaccard_global_lookup[(la, lb)]
        c = cka_global_lookup.get((la, lb), float("nan"))
        rows.append((j, c, la, lb))
    rows.sort(key=lambda r: -r[0])
    for j, c, la, lb in rows:
        c_s = f"{c:>8.4f}" if not (isinstance(c, float) and (c != c)) else "    n/a"
        print(f"  {(la + ' ⇄ ' + lb):<68s}  {j:>8.4f}  {c_s}")


if __name__ == "__main__":
    main()
