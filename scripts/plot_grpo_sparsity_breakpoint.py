"""Where does GRPO masking break? Performance vs sparsity, against the dense baseline.

Renders ONE panel, chosen with --metric:

  kl      (default) KL fold change over training. THIS is the axis that resolves the
          breakpoint: plateau 5.0-8.1x (dense 5.87x) through rho=99, then 1.97x at
          rho=99.75 against a random-mask floor of 1.36x. No overlap.

  reward  final training reward. Included for completeness, but it CANNOT locate a
          breakpoint and is not annotated as if it could: rho=99.75 (1.0088) sits
          *inside* the dense +/-2 SE band [0.9870, 1.0780] (z = 0.68), rho=99 and
          rho=99.75 both sit inside the random band as well, and the dense and random
          bands overlap each other (z = 1.28). An earlier version of this figure drew a
          "breaks here" arrow on this axis; that arrow contradicted the figure's own
          decoding rule and has been removed.

This split is the repo's own instruction, not a stylistic choice --
docs/run_logs/DO_NOT_REPEAT.md: "reward saturates across 70-99, so KL displacement is the
metric that resolves sparsity tolerance."

Both panels locate a knee on the TRAINING OBJECTIVE only. Held-out GSM8K separates no
sparse arm from the untrained base at any rho (grpo_matched_aicr_2026-07-29.md section 4c).

Usage: python scripts/plot_grpo_sparsity_breakpoint.py breakpoint.json out_prefix [--metric kl|reward]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
RHOS = ["50", "60", "70", "80", "90", "95", "97.5", "99", "99.75"]


def band(ax, mean, se, colour, label, k=2.0):
    """Reference condition: mean line, plus its +/-k SE band when an SE exists."""
    if se:
        ax.axhspan(mean - k * se, mean + k * se, color=colour, alpha=0.13, linewidth=0)
    ax.axhline(mean, color=colour, linewidth=1.8, linestyle=(0, (5, 2)))
    ax.text(0.0023, mean, label, color=colour, fontsize=12.5, fontweight="600",
            va="bottom", ha="left")


def main() -> None:
    src, prefix = sys.argv[1], sys.argv[2]
    metric = sys.argv[4] if len(sys.argv) > 4 else "kl"
    d = json.load(open(src))
    keep = np.array([d[r]["keep"] for r in RHOS])

    fig, ax = plt.subplots(figsize=(11.0, 6.4), facecolor="white")

    if metric == "loss":
        # GRPO's surrogate loss has no meaningful value -- only its gradient does.
        # Advantages are group-normalised to zero mean, so the loss oscillates about 0.
        # Plotted only to show the reviewer WHY it cannot answer the question.
        y = np.array([d[r]["loss_last50"] for r in RHOS])
        e = np.array([2 * d[r]["loss_se"] for r in RHOS])
        band(ax, d["dense"]["loss_last50"], d["dense"]["loss_se"], ORANGE, "Dense")
        band(ax, d["random"]["loss_last50"], d["random"]["loss_se"], AQUA, "Random mask")
        ylab = "GRPO surrogate loss, last 50 of 500 steps"
        note = ("\nGRPO's loss is a policy-gradient surrogate: only its gradient carries "
                "meaning, not its value. Every arm overlaps every other;\nrandom — the "
                "worst arm — has among the LOWEST loss. This axis cannot locate a breakpoint.")
        brk = None
    elif metric == "kl":
        fold = lambda v: v["kl_last"]
        y, e = np.array([fold(d[r]) for r in RHOS]), None
        band(ax, fold(d["dense"]), None, ORANGE, "Dense")
        band(ax, fold(d["random"]), None, AQUA, "Random mask")
        ylab = "KL displacement from base, last 50 of 500 steps"
        note = ("\nPlateau at or above dense from 50% down to 5% of parameters; monotone "
                "decline from 2.5% (79% of dense) to 0.25% (19%, near the random floor).\n"
                "Absolute final KL — NOT normalised by the first-50 window: all arms start from "
                "the same base model (KL=0 at step 0), so that window is an outcome,\nnot a "
                "baseline, and dividing by it manufactures a false cliff. Single seed per level; "
                "random reference is ρ=97.5 only (not density-matched).")
        brk = 1
    else:
        y = np.array([d[r]["reward_last50"] for r in RHOS])
        e = np.array([2 * d[r]["reward_se"] for r in RHOS])
        band(ax, d["dense"]["reward_last50"], d["dense"]["reward_se"], ORANGE, "Dense")
        band(ax, d["random"]["reward_last50"], d["random"]["reward_se"], AQUA,
             "Random mask")
        ylab = "Mean training reward, last 50 of 500 steps"
        note = ("\nBars and bands are ±2 SE. No level separates from dense on this axis — "
                "including ρ=99.75 — so no break is marked; see the KL panel.")
        brk = None

    ax.errorbar(keep, y, yerr=e, color=BLUE, linewidth=2.6, marker="o", markersize=8,
                capsize=4, elinewidth=1.5, zorder=3)

    if brk is not None:
        ax.annotate("roll-off begins", xy=(0.025, y[RHOS.index("97.5")]),
                    xytext=(0.055, y[RHOS.index("97.5")] - 0.0007), color=INK,
                    fontsize=12, fontweight="600",
                    arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.4))
        ax.annotate("≈ random floor", xy=(0.0025, y[-1]),
                    xytext=(0.0075, y[-1] + 0.00042), color=INK,
                    fontsize=12, fontweight="600",
                    arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.4))

    ax.set_title("How sparse can GRPO go?", fontsize=17, fontweight="700",
                 color=INK, pad=34, loc="center")
    ax.set_ylabel(ylab, fontsize=13, color=MUTED)
    ax.set_xlabel("Parameters trained", fontsize=13, color=MUTED, labelpad=10)
    ax.set_xscale("log")
    ax.set_xlim(0.0022, 0.42)
    ax.invert_xaxis()
    ax.set_xticks(keep)
    ax.set_xticklabels([f"{k*100:g}%" for k in keep], fontsize=12)
    ax.grid(True, axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=12)

    sec = ax.secondary_xaxis("top")
    sec.set_xscale("log")
    sec.set_xticks(keep)
    sec.set_xticklabels([f"ρ={r}%" for r in RHOS], fontsize=10, rotation=45,
                        ha="left", rotation_mode="anchor")
    sec.tick_params(colors=MUTED)
    for s in sec.spines.values():
        s.set_visible(False)

    fig.text(0.075, 0.012,
             "Llama-3.1-8B · Open-R1-Math-220k · 500 steps, schedule-matched.\n"
             "Training objective only — held-out GSM8K separates no sparse arm from base at any ρ."
             + note,
             fontsize=9, color=MUTED, linespacing=1.7)
    fig.tight_layout(rect=[0.0, 0.155, 0.995, 1.0])
    for ext in ("png", "pdf"):
        fig.savefig(f"{prefix}.{ext}", dpi=200, facecolor="white")
    print(f"wrote {prefix}.{{png,pdf}}  (metric={metric})")


if __name__ == "__main__":
    main()
