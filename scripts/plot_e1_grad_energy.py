"""
Figures for E1: gradient energy captured by each mask.

Two panels, both from e1_grad_energy.json:

  1. phi per mask against its own chance baseline. The baseline is per mask on purpose -- it
     ranges 0.158-0.193 across these masks because per-tensor keep rates are far from uniform and
     gradient energy is unevenly distributed, so a single sqrt(1-rho) line would misstate every
     ratio.
  2. phi per transformer block. Free, since phi already needs per-tensor norms, and it shows
     *where* a mask sits relative to gradient mass -- which is the panel that connects to the
     layer-wise Jaccard/CKA story.

  python scripts/plot_e1_grad_energy.py --json .../e1_grad_energy.json --out_dir .../figures
"""

import argparse
import json
import os

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import numpy as np


def plot_bars(masks, out_dir, key, chance_key, tag):
    order = sorted(masks, key=lambda m: -(m[key] / m[chance_key]))
    labels = [m["label"] for m in order]
    phis = [m[key] for m in order]
    chances = [m[chance_key] for m in order]
    ratios = [p / c for p, c in zip(phis, chances)]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = ["C3" if "random" in l else ("C0" if "dpo" in l else "C1") for l in labels]
    ax.bar(x, phis, color=colors, width=0.62, zorder=2)
    # Chance is per mask, so draw it as a per-bar tick rather than one global line.
    for xi, c in zip(x, chances):
        ax.plot([xi - 0.34, xi + 0.34], [c, c], color="0.25", lw=1.8, zorder=3)
    for xi, p, r in zip(x, phis, ratios):
        ax.annotate(f"{r:.2f}×", (xi, p), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9)
    ax.set_ylabel(r"$\phi(M) = \|g \odot M\|_2 / \|g\|_2$")
    ax.set_title("DPO gradient energy captured at initialization\n"
                 "(black ticks: that mask's own chance baseline; labels: ratio to chance)",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="C0"),
        plt.Rectangle((0, 0), 1, 1, color="C1"),
        plt.Rectangle((0, 0), 1, 1, color="C3"),
    ]
    ax.legend(handles, ["DPO-derived mask", "GRPO-derived mask", "random (validity gate)"],
              fontsize=9, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"e1_phi_by_mask{tag}.{ext}"), dpi=175)
    plt.close(fig)


def plot_layers(masks, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for i, m in enumerate(masks):
        per = m["per_layer"]
        ks = sorted(int(k) for k in per if k != "embed_lm_head")
        ys = [per[str(k)]["phi"] for k in ks]
        style = "--" if "random" in m["label"] else "-"
        ax.plot(ks, ys, style, lw=1.6, label=m["label"], alpha=0.9)
    ax.set_xlabel("transformer block")
    ax.set_ylabel(r"$\phi$ within block")
    ax.set_title("Where each mask sits relative to gradient mass, by block")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"e1_phi_by_layer.{ext}"), dpi=175)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.json) as f:
        data = json.load(f)
    masks = data["masks"]

    plot_bars(masks, args.out_dir, "phi_2d", "chance_energy_weighted_2d", "")
    plot_layers(masks, args.out_dir)

    print(f"{'mask':<26} {'phi_2d':>9} {'chance':>9} {'ratio':>7}  keep_rate_2d spread")
    for m in masks:
        per = m["per_layer"]
        rates = [v["keep_rate"] for v in per.values()]
        print(f"{m['label']:<26} {m['phi_2d']:>9.6f} {m['chance_energy_weighted_2d']:>9.6f} "
              f"{m['phi_2d']/m['chance_energy_weighted_2d']:>7.3f}  "
              f"[{min(rates):.4f}, {max(rates):.4f}]")
    print(f"\nwrote {args.out_dir}/e1_phi_by_mask.png, e1_phi_by_layer.png")


if __name__ == "__main__":
    main()
