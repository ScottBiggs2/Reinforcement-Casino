#!/usr/bin/env python3
"""Working figure for the NeurIPS #29841 rebuttal: Table-1 eval suite as small
multiples of horizontal bars (one panel per benchmark, 8 trained arms as bars,
base model as a dashed reference line).

Usage: python scripts/plot_eval_tulu3_bars.py <table1_summary.json> <out_prefix>
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical palette validated with the dataviz six-checks validator
# (light surface #fcfcfb): dense blue / sparse aqua / random violet / LoRA orange.
C_DENSE, C_SPARSE, C_RANDOM, C_LORA = "#2a78d6", "#1baf7a", "#4a3aa7", "#eb6834"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e5e4e0"

ARMS = [  # bottom-to-top plotting order; base handled as a reference line
    ("lora_r64_lr5e-6_lightr1", "LoRA r64 lr5e-6 (Light-R1)", C_LORA),
    ("lora_r64_lr1e-4_lightr1", "LoRA r64 lr1e-4 (Light-R1)", C_LORA),
    ("sparse_random", "Sparse random mask", C_RANDOM),
    ("sparse_warm_mag_step200", "Sparse warm magnitude@200", C_SPARSE),
    ("sparse_oracle_grpo_math", "Sparse oracle (GRPO math)", C_SPARSE),
    ("sparse_oracle_dpo_lightr1", "Sparse oracle (DPO Light-R1)", C_SPARSE),
    ("sparse_oracle_dpo_tulu3", "Sparse oracle (DPO Tulu3)", C_SPARSE),
    ("dense_dpo_tulu3_ckpt500", "Dense DPO Tulu3", C_DENSE),
]
BASE = "base_llama31_8b_instruct"

PANELS = [
    ("mmlu", "MMLU (acc)"),
    ("math", "MATH (exact match, macro)"),
    ("gsm8k", "GSM8K (strict exact match)"),
    ("humaneval", "HumanEval (pass@1)"),
    ("mbpp", "MBPP (pass@1)"),
    ("ifeval", "IFEval (prompt strict acc)"),
    ("squad", "SQuAD (contains)"),
    ("gpqa_diamond", "GPQA Diamond (acc)"),
]


def main():
    summary_path, out_prefix = sys.argv[1], sys.argv[2]
    S = json.load(open(summary_path))

    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 5.6), sharey=True)

    ylabels = [label for _, label, _ in ARMS]
    for ax, (bench, title) in zip(axes.flat, PANELS):
        vals = [S.get(key, {}).get(bench) for key, _, _ in ARMS]
        colors = [c for _, _, c in ARMS]
        ys = range(len(ARMS))
        for y, v, c in zip(ys, vals, colors):
            if v is None:
                ax.text(0.02, y, "running…", va="center", ha="left",
                        color=INK2, fontsize=7, style="italic",
                        transform=ax.get_yaxis_transform())
            else:
                ax.barh(y, v, height=0.62, color=c, zorder=3)
        base_v = S.get(BASE, {}).get(bench)
        if base_v is not None:
            ax.axvline(base_v, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=4)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=4)
        ax.set_yticks(list(ys), labels=ylabels)
        present = [v for v in vals + [base_v] if v is not None]
        ax.set_xlim(0, (max(present) * 1.28) if present else 1)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.tick_params(length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        for y, v in zip(ys, vals):  # selective labels: value at each bar end
            if v is not None:
                ax.text(v, y, f" {v:.3f}", va="center", ha="left",
                        color=INK2, fontsize=7)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_DENSE, label="Dense DPO"),
        plt.Rectangle((0, 0), 1, 1, color=C_SPARSE, label="Sparse (oracle / warm mask)"),
        plt.Rectangle((0, 0), 1, 1, color=C_RANDOM, label="Sparse random (control)"),
        plt.Rectangle((0, 0), 1, 1, color=C_LORA, label="LoRA baseline"),
        plt.Line2D([0], [0], color=INK2, lw=1.1, ls=(0, (4, 3)),
                   label="Base Llama-3.1-8B-Instruct"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("General-capability retention (Tulu-3 eval suite) — Table-1 arms, AICR 2026-08-01",
                 fontsize=11, color=INK, y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor=SURFACE)
    print(f"wrote {out_prefix}.png / .pdf")


if __name__ == "__main__":
    main()
