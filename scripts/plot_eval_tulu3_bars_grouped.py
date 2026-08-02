#!/usr/bin/env python3
"""Clustered vertical-bar version of the Table-1 eval figure: one cluster per
benchmark, 9 bars (base + 8 trained arms) per cluster, fixed order and colors.

Usage: python scripts/plot_eval_tulu3_bars_grouped.py <table1_summary.json> <out_prefix>
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e5e4e0"

# hue = arm group, lightness step = variant within group
ARMS = [
    ("base_llama31_8b_instruct", "Base Llama-3.1-8B-Instruct", "#8a8985"),
    ("dense_dpo_tulu3_ckpt500", "Dense DPO Tulu3", "#2a78d6"),
    ("sparse_oracle_dpo_tulu3", "Sparse oracle (DPO Tulu3)", "#0c6e4d"),
    ("sparse_oracle_dpo_lightr1", "Sparse oracle (DPO Light-R1)", "#1baf7a"),
    ("sparse_oracle_grpo_math", "Sparse oracle (GRPO math)", "#5fd3a9"),
    ("sparse_warm_mag_step200", "Sparse warm magnitude@200", "#008300"),
    ("sparse_random", "Sparse random (control)", "#4a3aa7"),
    ("lora_r64_lr1e-4_lightr1", "LoRA r64 lr1e-4 (Light-R1)", "#c94f1d"),
    ("lora_r64_lr5e-6_lightr1", "LoRA r64 lr5e-6 (Light-R1)", "#f0925c"),
]

PANELS = [
    ("mmlu", "MMLU"),
    ("math", "MATH"),
    ("gsm8k", "GSM8K"),
    ("humaneval", "HumanEval"),
    ("mbpp", "MBPP"),
    ("ifeval", "IFEval"),
    ("squad", "SQuAD\n(contains)"),
    ("gpqa_diamond", "GPQA\nDiamond"),
]


def main():
    summary_path, out_prefix = sys.argv[1], sys.argv[2]
    S = json.load(open(summary_path))

    plt.rcParams.update({
        "font.size": 9.5, "text.color": INK, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK, "ytick.color": INK2,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })
    fig, ax = plt.subplots(figsize=(13.8, 6.2))

    n = len(ARMS)
    width = 0.088
    xs = np.arange(len(PANELS))
    for i, (key, label, color) in enumerate(ARMS):
        offs = (i - (n - 1) / 2) * width
        for j, (bench, _) in enumerate(PANELS):
            v = S.get(key, {}).get(bench)
            if v is None:
                continue
            ax.bar(xs[j] + offs, v, width=width * 0.86, color=color, zorder=3,
                   label=label if j == 0 else None)
            ax.text(xs[j] + offs, v + 0.012, f"{v:.3f}", rotation=90,
                    va="bottom", ha="center", color=INK2, fontsize=6.4)
    # legend needs every arm even if its first cluster cell is missing
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for _, l, c in ARMS]

    for j, (bench, _) in enumerate(PANELS):
        if all(S.get(k, {}).get(bench) is None for k, _, _ in ARMS):
            ax.text(xs[j], 0.02, "running…", ha="center", va="bottom",
                    color=INK2, fontsize=8.5, style="italic")

    ax.set_xticks(xs, labels=[t for _, t in PANELS])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score (headline metric per benchmark)")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
              fontsize=8.5, bbox_to_anchor=(0.5, 1.24))
    ax.set_title("General-capability retention (Tulu-3 eval suite) — Table-1 arms, AICR 2026-08-01",
                 fontsize=11.5, color=INK, pad=86)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor=SURFACE)
    print(f"wrote {out_prefix}.png / .pdf")


if __name__ == "__main__":
    main()
