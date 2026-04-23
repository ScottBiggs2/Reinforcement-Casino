#!/usr/bin/env python3
"""
Summarize and visualize CKA results: cold masks vs ground truth (warm magnitude).

Reads all cka_*.json from the results directory and produces a single multi-panel
figure answering: why do cold CAV/SNIP best approximate the oracle?

Usage:
    python scripts/plot_cka_vs_ground_truth.py [--results_dir PATH] [--output PATH]

Defaults:
    --results_dir  /scratch/$USER/rl_casino_outputs/cka_results_llama8b
    --output       <results_dir>/cka_summary.png
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────

METHOD_COLORS = {
    "snip": "#2196F3",    # blue
    "cav": "#4CAF50",     # green
    "fisher": "#FF9800",  # orange
    "gt": "#9C27B0",      # purple  (ground truth)
}
METHOD_LABELS = {
    "snip": "SNIP",
    "cav": "CAV",
    "fisher": "Fisher",
    "gt": "Ground Truth",
}
MODE_STYLES = {"dpo": "-", "grpo": "--"}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def extract_layer_index(layer_name):
    """'model.layers.17.mlp' -> 17"""
    m = re.search(r"layers\.(\d+)", layer_name)
    return int(m.group(1)) if m else None


def sorted_per_layer(per_layer_cka):
    """Return (layer_indices, cka_scores) sorted by layer index."""
    items = []
    for name, score in per_layer_cka.items():
        idx = extract_layer_index(name)
        if idx is not None:
            items.append((idx, score))
    items.sort()
    return [i for i, _ in items], [s for _, s in items]


# ── Load results ──────────────────────────────────────────────────────────

def load_all_results(results_dir):
    """Load all cka_*.json files, keyed by filename stem."""
    results = {}
    for p in sorted(Path(results_dir).glob("cka_*.json")):
        results[p.stem] = load_json(p)
    return results


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_summary(results, output_path):
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "CKA Analysis: Cold Pruning Methods vs Ground Truth (Warm Magnitude)",
        fontsize=16, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(3, 2, hspace=0.38, wspace=0.28, top=0.93, bottom=0.06)

    # ── Panel 1: Mean CKA to Ground Truth (bar chart) ─────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_mean_cka_bars(ax1, results)

    # ── Panel 2: Layer-wise CKA to Ground Truth (DPO) ─────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_layerwise_vs_gt(ax2, results, mode="dpo")

    # ── Panel 3: Layer-wise CKA to Ground Truth (GRPO) ────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_layerwise_vs_gt(ax3, results, mode="grpo")

    # ── Panel 4: Fidelity — original vs masked (DPO) ─────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_fidelity(ax4, results)

    # ── Panel 5: SNIP vs CAV similarity ───────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    _plot_snip_vs_cav(ax5, results)

    # ── Panel 6: Summary table ────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    _plot_summary_table(ax6, results)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSummary figure saved to: {output_path}")
    plt.close()


def _plot_mean_cka_bars(ax, results):
    """Bar chart: mean CKA to ground truth for each cold method × mode."""
    methods = ["snip", "cav", "fisher"]
    modes = ["dpo", "grpo"]

    x = np.arange(len(methods))
    width = 0.35

    for i, mode in enumerate(modes):
        means = []
        for method in methods:
            key = f"cka_vs_gt_cold_{method}_{mode}"
            if key in results:
                means.append(results[key]["cka"]["mean"])
            else:
                means.append(0)
        bars = ax.bar(
            x + i * width, means, width,
            label=mode.upper(),
            color=[METHOD_COLORS[m] for m in methods],
            alpha=0.9 if i == 0 else 0.5,
            edgecolor="black" if i == 0 else "gray",
            linewidth=0.8,
        )
        for bar, val in zip(bars, means):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
                )

    ax.set_ylabel("Mean CKA to Ground Truth")
    ax.set_title("A. Cold Method → Ground Truth Alignment", fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.8, color="gray", linestyle=":", alpha=0.5, label="High similarity")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def _plot_layerwise_vs_gt(ax, results, mode="dpo"):
    """Line plot: per-layer CKA to ground truth for each cold method."""
    methods = ["snip", "cav", "fisher"]

    for method in methods:
        key = f"cka_vs_gt_cold_{method}_{mode}"
        if key not in results:
            continue
        layers, scores = sorted_per_layer(results[key]["per_layer_cka"])
        ax.plot(
            layers, scores,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            linewidth=2,
            marker="o", markersize=3,
        )

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("CKA to Ground Truth")
    title_mode = mode.upper()
    ax.set_title(f"B. Layer-wise CKA to Ground Truth ({title_mode})", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.8, color="gray", linestyle=":", alpha=0.4)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def _plot_fidelity(ax, results):
    """Bar chart: how much each method distorts the original model (DPO)."""
    entries = [
        ("snip", "cka_fidelity_cold_snip_dpo"),
        ("cav", "cka_fidelity_cold_cav_dpo"),
        ("fisher", "cka_fidelity_cold_fisher_dpo"),
        ("gt", "cka_fidelity_gt_dpo"),
    ]

    labels, means = [], []
    colors = []
    for method, key in entries:
        if key in results:
            labels.append(METHOD_LABELS[method])
            means.append(results[key]["cka"]["mean"])
            colors.append(METHOD_COLORS[method])

    if not labels:
        ax.text(0.5, 0.5, "No fidelity data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("D. Representational Fidelity", fontweight="bold")
        return

    bars = ax.bar(labels, means, color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Mean CKA (original vs masked)")
    ax.set_title("D. Representational Fidelity (DPO)", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.8, color="gray", linestyle=":", alpha=0.4)
    ax.grid(axis="y", alpha=0.3)


def _plot_snip_vs_cav(ax, results):
    """Layer-wise CKA between SNIP and CAV — are they finding the same circuits?"""
    for mode in ["dpo", "grpo"]:
        key = f"cka_cross_snip_vs_cav_{mode}"
        if key not in results:
            continue
        layers, scores = sorted_per_layer(results[key]["per_layer_cka"])
        ax.plot(
            layers, scores,
            color=METHOD_COLORS["snip"] if mode == "dpo" else METHOD_COLORS["cav"],
            linestyle=MODE_STYLES[mode],
            label=f"SNIP vs CAV ({mode.upper()})",
            linewidth=2, marker="o", markersize=3,
        )
        mean_val = results[key]["cka"]["mean"]
        ax.axhline(
            y=mean_val, color="gray", linestyle=":",
            alpha=0.4,
        )
        ax.text(
            0.02, mean_val + 0.02,
            f"mean={mean_val:.3f} ({mode.upper()})",
            transform=ax.get_yaxis_transform(),
            fontsize=8, color="gray",
        )

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("CKA (SNIP vs CAV)")
    ax.set_title("E. SNIP vs CAV: Same Circuits?", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def _plot_summary_table(ax, results):
    """Text summary table of all results."""
    ax.axis("off")
    ax.set_title("F. Summary", fontweight="bold")

    rows = []
    # Group A
    for mode in ["dpo", "grpo"]:
        for method in ["snip", "cav", "fisher"]:
            key = f"cka_vs_gt_cold_{method}_{mode}"
            if key in results:
                cka = results[key]["cka"]
                rows.append([
                    f"{METHOD_LABELS[method]} vs GT ({mode.upper()})",
                    f"{cka['mean']:.4f}",
                    f"{cka['min']:.4f}",
                    f"{cka['max']:.4f}",
                ])

    # Group B fidelity
    for method, key in [
        ("snip", "cka_fidelity_cold_snip_dpo"),
        ("cav", "cka_fidelity_cold_cav_dpo"),
        ("fisher", "cka_fidelity_cold_fisher_dpo"),
        ("gt", "cka_fidelity_gt_dpo"),
    ]:
        if key in results:
            cka = results[key]["cka"]
            rows.append([
                f"Fidelity: {METHOD_LABELS[method]} (DPO)",
                f"{cka['mean']:.4f}",
                f"{cka['min']:.4f}",
                f"{cka['max']:.4f}",
            ])

    # Group C
    for mode in ["dpo", "grpo"]:
        key = f"cka_cross_snip_vs_cav_{mode}"
        if key in results:
            cka = results[key]["cka"]
            rows.append([
                f"SNIP vs CAV ({mode.upper()})",
                f"{cka['mean']:.4f}",
                f"{cka['min']:.4f}",
                f"{cka['max']:.4f}",
            ])

    if not rows:
        ax.text(0.5, 0.5, "No results found", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        return

    col_labels = ["Comparison", "Mean", "Min", "Max"]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#E0E0E0")
        table[0, j].set_text_props(fontweight="bold")

    # Highlight best vs GT rows
    gt_rows_dpo = {}
    for i, row in enumerate(rows):
        label = row[0]
        for method in ["SNIP", "CAV", "Fisher"]:
            if label.startswith(f"{method} vs GT (DPO)"):
                gt_rows_dpo[method] = (i + 1, float(row[1]))  # +1 for header

    if gt_rows_dpo:
        best_method = max(gt_rows_dpo, key=lambda m: gt_rows_dpo[m][1])
        row_idx = gt_rows_dpo[best_method][0]
        for j in range(len(col_labels)):
            table[row_idx, j].set_facecolor("#C8E6C9")  # light green


def main():
    parser = argparse.ArgumentParser(
        description="Summarize CKA results: cold masks vs ground truth",
    )
    default_dir = f"/scratch/{os.environ.get('USER', 'xie.yiyi')}/rl_casino_outputs/cka_results_llama8b"
    parser.add_argument(
        "--results_dir", type=str, default=default_dir,
        help="Directory containing cka_*.json files",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output PNG path (default: <results_dir>/cka_summary.png)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.results_dir, "cka_summary.png")

    print(f"Loading results from: {args.results_dir}")
    results = load_all_results(args.results_dir)

    if not results:
        print(f"ERROR: No cka_*.json files found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(results)} result files:")
    for name in results:
        cka = results[name]["cka"]
        print(f"  {name}: mean={cka['mean']:.4f}  min={cka['min']:.4f}  max={cka['max']:.4f}")

    plot_summary(results, args.output)


if __name__ == "__main__":
    main()
