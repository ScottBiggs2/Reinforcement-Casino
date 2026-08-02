#!/usr/bin/env python
"""Accurate stable-rank figure for the LoRA-rank rebuttal paragraph.

Every number shown is computed from the delta JSONs at plot time — nothing is
hand-typed — so the figure cannot drift from the data the way the draft
paragraph did (it quoted the three per-run MEDIANS 132/178/229 as if they were
the min-max range of 48 matrices).

Curves: sorted per-matrix stable rank ||dW||_F^2 / ||dW||_2^2 vs percentile,
log2 y-axis, against the r=64 cap. The two Llama runs measured in July 2026
cover ALL 226 matched 2-D params; the 2026-07-25 runs cover the first 48.
Style matches scripts/plot_lora_geometry_rebuttal.py.

Usage: python scripts/plot_stable_rank_vs_lora_cap.py [--outdir docs/paper_drafts]
"""
import argparse
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, PLUM = "#2a78d6", "#eb6834", "#1baf7a", "#8e5bc0"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
LORA_R = 64

RL = "/Users/xxiellan/Reinforcement-Casino/docs/run_logs"
SOURCES = [  # path, label, color
    (f"{RL}/delta_spectrum_2026-07-31/delta_llama_dpo_lightr1_spec.json",
     "DPO · Light-R1", BLUE),
    (f"{RL}/delta_analysis_2026-07-25/delta_llama_dpo_tulu3.json",
     "DPO · Tülu-3", PLUM),
    (f"{RL}/delta_spectrum_2026-07-31/delta_llama_grpo_math220k_spec.json",
     "GRPO · Math-220k", ORANGE),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs/paper_drafts")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for path, label, color in SOURCES:
        srs = sorted(r["stable_rank"] for r in json.load(open(path))["stable_ranks"])
        n = len(srs)
        med = statistics.median(srs)
        above = sum(s > LORA_R for s in srs)
        pct = 100 * (np.arange(n) + 0.5) / n
        ax.plot(pct, srs, color=color, linewidth=2.4, label=label)

    ax.axhline(LORA_R, color=INK, linestyle="--", linewidth=2)
    ax.text(98, LORA_R * 0.88, f"LoRA r={LORA_R}",
            ha="right", va="top", fontsize=13, color=INK)

    ax.set_yscale("log", base=2)
    ax.set_yticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("matrix percentile",
                  fontsize=13, color=MUTED)
    ax.set_ylabel("stable rank  ‖ΔW‖²_F / ‖ΔW‖²₂", fontsize=13, color=MUTED)
    ax.set_title("Per-matrix stable rank of ΔW vs the LoRA r=64 cap  (Llama-3.1-8B)",
                 fontsize=16, fontweight="bold", color=INK)
    ax.grid(True, axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.legend(fontsize=11.5, loc="lower right", framealpha=0.95)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(args.outdir, f"stable_rank_vs_lora_cap.{ext}")
        fig.savefig(out, dpi=200)
        print("wrote", out)


if __name__ == "__main__":
    main()
