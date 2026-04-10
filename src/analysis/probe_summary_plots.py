#!/usr/bin/env python3
"""
Generate consolidated probe analysis visualizations.

Reads all probe_results.json files from a directory and produces:
  1. Summary bar chart: all masks × all properties (mean accuracy)
  2. Delta heatmap: all masks' Δ from baseline, grouped by category
  3. DPO vs GRPO scatter: direct comparison per property

Usage:
    python src/analysis/probe_summary_plots.py \
        --results_dir /scratch/$USER/rl_casino_outputs/probe_results_llama8b \
        --output_dir /scratch/$USER/rl_casino_outputs/probe_results_llama8b/summary
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_all_results(results_dir):
    """Load all probe_results.json files and extract per-mask mean accuracies."""
    all_data = {}
    for subdir in sorted(Path(results_dir).iterdir()):
        json_path = subdir / "probe_results.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)
        all_data[subdir.name] = data
    return all_data


def extract_mean_accuracies(all_data):
    """Extract mean accuracy per (mask_name, property) across layers.

    Returns:
        masks: dict {mask_name: {property: mean_accuracy}}
        baseline: {property: mean_accuracy}  (averaged across runs)
    """
    masks = {}
    baselines = defaultdict(list)

    for comparison_name, data in all_data.items():
        for config_label, prop_results in data.items():
            config_clean = config_label.replace("\n", " ").strip()

            accs = {}
            for prop, layer_dict in prop_results.items():
                vals = list(layer_dict.values())
                accs[prop] = np.mean(vals)

            if "baseline" in config_clean.lower() or "no mask" in config_clean.lower():
                for prop, val in accs.items():
                    baselines[prop].append(val)
            else:
                if config_clean not in masks:
                    masks[config_clean] = accs

    baseline = {prop: np.mean(vals) for prop, vals in baselines.items()}
    return masks, baseline


def plot_summary_bar(masks, baseline, properties, output_path):
    """Bar chart: all masks grouped by property, with baseline reference line."""
    import matplotlib.pyplot as plt

    n_masks = len(masks)
    n_props = len(properties)
    mask_names = list(masks.keys())

    fig, axes = plt.subplots(1, n_props, figsize=(5 * n_props, 7), sharey=True)
    if n_props == 1:
        axes = [axes]

    colors = plt.cm.tab20(np.linspace(0, 1, n_masks))

    for pi, prop in enumerate(properties):
        ax = axes[pi]
        vals = [masks[m].get(prop, 0) for m in mask_names]
        bars = ax.barh(range(n_masks), vals, color=colors, height=0.7)

        # Baseline reference
        bl = baseline.get(prop, 0)
        ax.axvline(bl, color="black", linestyle="--", linewidth=2, label=f"Baseline ({bl:.2f})")
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Chance")

        ax.set_xlim(0.4, 1.0)
        ax.set_title(prop.capitalize(), fontsize=14, fontweight="bold")
        ax.set_yticks(range(n_masks))
        if pi == 0:
            ax.set_yticklabels(mask_names, fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.legend(fontsize=8, loc="lower right")

        # Value labels
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8)

    fig.suptitle("Probe Accuracy Summary — All Masks (Llama 3.1 8B, 97.5% Sparsity)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[summary] Saved bar chart → {output_path}")
    plt.close()


def plot_delta_heatmap(masks, baseline, properties, output_path):
    """Single heatmap: rows = masks, columns = properties, values = Δ from baseline."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    mask_names = list(masks.keys())
    n_masks = len(mask_names)
    n_props = len(properties)

    mat = np.full((n_masks, n_props), np.nan)
    for mi, m in enumerate(mask_names):
        for pi, prop in enumerate(properties):
            bl = baseline.get(prop, 0.5)
            val = masks[m].get(prop, 0.5)
            mat[mi, pi] = val - bl

    abs_max = max(0.05, np.nanmax(np.abs(mat)))
    cmap = plt.cm.RdBu
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    fig, ax = plt.subplots(figsize=(8, max(6, n_masks * 0.5 + 2)))
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(n_props))
    ax.set_xticklabels([p.capitalize() for p in properties], fontsize=12, fontweight="bold")
    ax.set_yticks(range(n_masks))
    ax.set_yticklabels(mask_names, fontsize=9)

    for mi in range(n_masks):
        for pi in range(n_props):
            val = mat[mi, pi]
            if np.isnan(val):
                continue
            sign = "+" if val > 0 else ""
            txt_color = "white" if abs(val) > abs_max * 0.6 else "black"
            ax.text(pi, mi, f"{sign}{val:.3f}", ha="center", va="center",
                    fontsize=9, color=txt_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Δ accuracy (mask − baseline)", fontsize=11)

    ax.set_title("Knowledge Retention: Δ from Baseline\n(Llama 3.1 8B, 97.5% Sparsity)",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[summary] Saved delta heatmap → {output_path}")
    plt.close()


def plot_dpo_vs_grpo(masks, properties, output_path):
    """Scatter plot: DPO accuracy (x) vs GRPO accuracy (y) per property."""
    import matplotlib.pyplot as plt

    # Find DPO/GRPO pairs
    pairs = {}  # {(method, start_type): {"dpo": {...}, "grpo": {...}}}
    for mask_name, accs in masks.items():
        name_lower = mask_name.lower()
        if "dpo" in name_lower:
            algo = "dpo"
        elif "grpo" in name_lower:
            algo = "grpo"
        else:
            continue
        # Extract method identifier (remove DPO/GRPO)
        method_key = mask_name.replace("-DPO", "").replace("-GRPO", "").strip()
        if method_key not in pairs:
            pairs[method_key] = {}
        pairs[method_key][algo] = accs

    # Filter to pairs that have both
    complete_pairs = {k: v for k, v in pairs.items() if "dpo" in v and "grpo" in v}
    if not complete_pairs:
        print("[summary] No DPO/GRPO pairs found, skipping scatter plot.")
        return

    prop_colors = {"syntax": "#e74c3c", "semantics": "#3498db",
                   "factual": "#2ecc71", "math": "#f39c12"}
    prop_markers = {"syntax": "o", "semantics": "s", "factual": "D", "math": "^"}

    fig, ax = plt.subplots(figsize=(10, 10))

    for method_key, algos in complete_pairs.items():
        for prop in properties:
            dpo_val = algos["dpo"].get(prop, 0.5)
            grpo_val = algos["grpo"].get(prop, 0.5)
            ax.scatter(dpo_val, grpo_val,
                       c=prop_colors.get(prop, "gray"),
                       marker=prop_markers.get(prop, "o"),
                       s=120, edgecolors="black", linewidth=0.5, zorder=3)
            ax.annotate(method_key, (dpo_val, grpo_val),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points")

    # Diagonal line (DPO = GRPO)
    lims = [0.4, 1.0]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1, zorder=1)
    ax.fill_between(lims, lims, [1.0, 1.0], alpha=0.05, color="blue", label="GRPO better")
    ax.fill_between(lims, [0.4, 0.4], lims, alpha=0.05, color="red", label="DPO better")

    # Legend for properties
    for prop in properties:
        ax.scatter([], [], c=prop_colors.get(prop, "gray"),
                   marker=prop_markers.get(prop, "o"), s=80,
                   label=prop.capitalize(), edgecolors="black", linewidth=0.5)
    ax.legend(fontsize=10, loc="upper left")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("DPO Mask — Probe Accuracy", fontsize=12)
    ax.set_ylabel("GRPO Mask — Probe Accuracy", fontsize=12)
    ax.set_title("DPO vs GRPO: Knowledge Retention per Property\n(Llama 3.1 8B, 97.5% Sparsity)",
                 fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[summary] Saved DPO vs GRPO scatter → {output_path}")
    plt.close()


def plot_grouped_comparison(masks, baseline, properties, output_path):
    """Grouped bar chart: Cold vs Warm, DPO vs GRPO side by side."""
    import matplotlib.pyplot as plt

    # Categorize masks
    categories = {
        "Cold DPO": [], "Cold GRPO": [],
        "Warm DPO": [], "Warm GRPO": [],
    }
    for mask_name, accs in masks.items():
        name_lower = mask_name.lower()
        is_cold = "cold" in name_lower
        is_warm = "warm" in name_lower
        is_dpo = "dpo" in name_lower
        is_grpo = "grpo" in name_lower

        if is_cold and is_dpo:
            categories["Cold DPO"].append((mask_name, accs))
        elif is_cold and is_grpo:
            categories["Cold GRPO"].append((mask_name, accs))
        elif is_warm and is_dpo:
            categories["Warm DPO"].append((mask_name, accs))
        elif is_warm and is_grpo:
            categories["Warm GRPO"].append((mask_name, accs))

    cat_names = ["Cold DPO", "Cold GRPO", "Warm DPO", "Warm GRPO"]
    cat_colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

    fig, axes = plt.subplots(1, len(properties), figsize=(5 * len(properties), 6), sharey=True)
    if len(properties) == 1:
        axes = [axes]

    bar_width = 0.18
    x = np.arange(len(cat_names))

    for pi, prop in enumerate(properties):
        ax = axes[pi]
        bl = baseline.get(prop, 0.5)

        for ci, cat in enumerate(cat_names):
            if categories[cat]:
                vals = [accs.get(prop, 0.5) for _, accs in categories[cat]]
                mean_val = np.mean(vals)
                std_val = np.std(vals) if len(vals) > 1 else 0
                ax.bar(ci, mean_val, width=0.6, color=cat_colors[ci],
                       yerr=std_val, capsize=4, edgecolor="black", linewidth=0.5)
                ax.text(ci, mean_val + std_val + 0.01, f"{mean_val:.2f}",
                        ha="center", fontsize=9, fontweight="bold")
            else:
                ax.bar(ci, 0, width=0.6, color="lightgray")

        ax.axhline(bl, color="black", linestyle="--", linewidth=2, label=f"Baseline ({bl:.2f})")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.4)
        ax.set_ylim(0.4, 1.0)
        ax.set_xticks(range(len(cat_names)))
        ax.set_xticklabels(cat_names, fontsize=9, rotation=20, ha="right")
        ax.set_title(prop.capitalize(), fontsize=14, fontweight="bold")
        if pi == 0:
            ax.set_ylabel("Probe Accuracy", fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle("Cold vs Warm × DPO vs GRPO — Mean Probe Accuracy\n(Llama 3.1 8B, 97.5% Sparsity)",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[summary] Saved grouped comparison → {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate consolidated probe analysis plots")
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing probe result subdirectories")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for summary plots (default: results_dir/summary)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.results_dir, "summary")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[summary] Loading results from: {args.results_dir}")
    all_data = load_all_results(args.results_dir)
    print(f"[summary] Found {len(all_data)} comparisons")

    masks, baseline = extract_mean_accuracies(all_data)
    print(f"[summary] {len(masks)} unique mask configurations")
    print(f"[summary] Baseline: {baseline}")

    properties = ["syntax", "semantics", "factual", "math"]

    # 1. Summary bar chart
    plot_summary_bar(masks, baseline, properties,
                     os.path.join(output_dir, "summary_bar_chart.png"))

    # 2. Delta heatmap
    plot_delta_heatmap(masks, baseline, properties,
                       os.path.join(output_dir, "summary_delta_heatmap.png"))

    # 3. DPO vs GRPO scatter
    plot_dpo_vs_grpo(masks, properties,
                     os.path.join(output_dir, "summary_dpo_vs_grpo.png"))

    # 4. Grouped comparison (Cold/Warm × DPO/GRPO)
    plot_grouped_comparison(masks, baseline, properties,
                            os.path.join(output_dir, "summary_grouped_comparison.png"))

    print(f"\n[summary] All plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
