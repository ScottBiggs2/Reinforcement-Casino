#!/usr/bin/env python3
"""Generate standalone Panel A and Panel D from CKA results."""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHOD_COLORS = {
    "snip": "#2196F3",
    "cav": "#4CAF50",
    "fisher": "#FF9800",
    "gt": "#9C27B0",
}
METHOD_LABELS = {
    "snip": "SNIP",
    "cav": "CAV",
    "fisher": "Fisher",
    "gt": "Ground Truth\n(Warm Magnitude)",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_all_results(results_dir):
    results = {}
    for p in sorted(Path(results_dir).glob("cka_*.json")):
        results[p.stem] = load_json(p)
    return results


def plot_panel_a(results, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ["snip", "cav", "fisher"]
    modes = ["dpo", "grpo"]
    x = np.arange(len(methods))
    width = 0.35

    for i, mode in enumerate(modes):
        means = []
        for method in methods:
            key = f"cka_vs_gt_cold_{method}_{mode}"
            means.append(results[key]["cka"]["mean"] if key in results else 0)
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
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
                )

    ax.set_ylabel("Mean CKA to Ground Truth", fontsize=12)
    ax.set_title("Cold Method → Ground Truth Alignment", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.8, color="gray", linestyle=":", alpha=0.5, label="High similarity")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Panel A saved to: {output_path}")


def plot_panel_d(results, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    entries = [
        ("snip", "cka_fidelity_cold_snip_dpo"),
        ("cav", "cka_fidelity_cold_cav_dpo"),
        ("fisher", "cka_fidelity_cold_fisher_dpo"),
        ("gt", "cka_fidelity_gt_dpo"),
    ]

    labels, means, colors = [], [], []
    for method, key in entries:
        if key in results:
            labels.append(METHOD_LABELS[method])
            means.append(results[key]["cka"]["mean"])
            colors.append(METHOD_COLORS[method])

    bars = ax.bar(labels, means, color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Mean CKA (Original vs Masked)", fontsize=12)
    ax.set_title("Representational Fidelity (DPO)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.8, color="gray", linestyle=":", alpha=0.4)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Panel D saved to: {output_path}")


def main():
    default_dir = f"/scratch/{os.environ.get('USER', 'xie.yiyi')}/rl_casino_outputs/cka_results_llama8b"
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=default_dir)
    parser.add_argument("--output_dir", "-o", default=None,
                        help="Output directory (default: same as results_dir)")
    args = parser.parse_args()

    out = args.output_dir or args.results_dir
    os.makedirs(out, exist_ok=True)

    results = load_all_results(args.results_dir)
    if not results:
        print(f"No cka_*.json found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    plot_panel_a(results, os.path.join(out, "panel_a_gt_alignment.png"))
    plot_panel_d(results, os.path.join(out, "panel_d_fidelity.png"))


if __name__ == "__main__":
    main()
