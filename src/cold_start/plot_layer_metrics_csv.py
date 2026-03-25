#!/usr/bin/env python3
"""Generate quick visualization panels from layer-metrics CSV files."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def to_float(x):
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def has_valid(vals):
    return any(not math.isnan(v) for v in vals)


def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expected_random_jaccard(density_a: float, density_b: float) -> float:
    """Expected Jaccard between two independent random masks with given densities."""
    denom = density_a + density_b - density_a * density_b
    if denom <= 0:
        return 0.0
    return (density_a * density_b) / denom


def plot_one(csv_path: Path, out_path: Path):
    rows = read_rows(csv_path)
    if not rows:
        return False

    x = list(range(len(rows)))
    layer_labels = [r.get("layer", str(i)) for i, r in enumerate(rows)]

    jaccard = [to_float(r.get("jaccard")) for r in rows]
    cka = [to_float(r.get("cka")) for r in rows]
    sparsity_a = [to_float(r.get("sparsity_a")) for r in rows]
    sparsity_b = [to_float(r.get("sparsity_b")) for r in rows]
    erank_a = [to_float(r.get("effective_rank_a_norm")) for r in rows]
    erank_b = [to_float(r.get("effective_rank_b_norm")) for r in rows]

    # Per-layer random baseline: E[Jaccard] = d_a*d_b / (d_a+d_b - d_a*d_b)
    random_baseline = []
    for sa, sb in zip(sparsity_a, sparsity_b):
        if math.isnan(sa) or math.isnan(sb):
            # Fall back to using whichever sparsity is available
            s = sa if not math.isnan(sa) else sb
            d = (1.0 - s) if not math.isnan(s) else float("nan")
            if math.isnan(d):
                random_baseline.append(float("nan"))
            else:
                random_baseline.append(expected_random_jaccard(d, d))
        else:
            random_baseline.append(expected_random_jaccard(1.0 - sa, 1.0 - sb))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(csv_path.name, fontsize=12)

    # Jaccard
    ax = axes[0, 0]
    if has_valid(jaccard):
        ax.plot(x, jaccard, marker="o", linewidth=1.3, markersize=3, label="Jaccard")
        if has_valid(random_baseline):
            ax.plot(
                x, random_baseline,
                linestyle="--", linewidth=1.1, color="gray", alpha=0.7,
                label="random baseline",
            )
        ax.set_ylim(0, max(1, max((v for v in jaccard if not math.isnan(v)), default=1) * 1.05))
        ax.legend(fontsize=8)
        ax.set_title("Per-layer Jaccard")
    else:
        ax.text(0.5, 0.5, "No Jaccard data", ha="center", va="center")
        ax.set_title("Per-layer Jaccard")
    ax.grid(alpha=0.3)

    # CKA
    ax = axes[0, 1]
    if has_valid(cka):
        ax.plot(x, cka, marker="o", linewidth=1.3, markersize=3, color="tab:purple")
        ax.set_ylim(0, 1)
        ax.set_title("Per-layer CKA")
    else:
        ax.text(0.5, 0.5, "No CKA data", ha="center", va="center")
        ax.set_title("Per-layer CKA")
    ax.grid(alpha=0.3)

    # Sparsity A/B
    ax = axes[1, 0]
    drawn = False
    if has_valid(sparsity_a):
        ax.plot(x, sparsity_a, marker="o", linewidth=1.3, markersize=3, label="sparsity_a")
        drawn = True
    if has_valid(sparsity_b):
        ax.plot(x, sparsity_b, marker="o", linewidth=1.3, markersize=3, label="sparsity_b")
        drawn = True
    if drawn:
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.set_title("Per-layer Sparsity")
    else:
        ax.text(0.5, 0.5, "No sparsity data", ha="center", va="center")
        ax.set_title("Per-layer Sparsity")
    ax.grid(alpha=0.3)

    # Effective rank normalized A/B
    ax = axes[1, 1]
    drawn = False
    if has_valid(erank_a):
        ax.plot(x, erank_a, marker="o", linewidth=1.3, markersize=3, label="erank_a_norm")
        drawn = True
    if has_valid(erank_b):
        ax.plot(x, erank_b, marker="o", linewidth=1.3, markersize=3, label="erank_b_norm")
        drawn = True
    if drawn:
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.set_title("Per-layer Effective Rank (normalized)")
    else:
        ax.text(0.5, 0.5, "No effective-rank data", ha="center", va="center")
        ax.set_title("Per-layer Effective Rank (normalized)")
    ax.grid(alpha=0.3)

    tick_step = max(1, len(x) // 12)
    tick_idx = list(range(0, len(x), tick_step))
    tick_text = [layer_labels[i].split(".")[-2] if "." in layer_labels[i] else str(i) for i in tick_idx]
    for ax in axes.ravel():
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_text, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Layer")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Create plot panels from layer-metrics CSV files.")
    parser.add_argument("--input-dir", type=str, default="masks")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--pattern", type=str, default="*.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    pattern = f"**/{args.pattern}" if args.recursive else args.pattern
    csv_files = sorted(input_dir.glob(pattern))

    # Ignore summary CSV artifacts.
    csv_files = [p for p in csv_files if not p.name.endswith("_summary.csv")]

    made = 0
    for p in csv_files:
        out = p.with_name(f"{p.stem}_plots.png")
        if plot_one(p, out):
            made += 1
            print(f"✓ {out}")

    print(f"Generated plot files: {made}")


if __name__ == "__main__":
    main()
