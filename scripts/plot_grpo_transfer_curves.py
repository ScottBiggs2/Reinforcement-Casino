"""Figure 3's GRPO counterpart, in Figure 3's format.

The submitted Figure 3 is two panels, one per training dataset, each showing the
curves obtained under oracle masks from the *other* tasks plus a random mask —
the in-task mask is deliberately absent. Reviewer XD33 was right that none of it
is GRPO training: every curve there is DPO, and only the mask's origin varies.

This is the same figure for GRPO. Training is Open-R1 Math throughout, so the
cross-task masks are the two DPO-derived ones and the random baseline: three
curves, exactly the three the published panels carry. Title, axis labels, boxed
in-plot legend, tick placement, grid, raw (unsmoothed) thin lines and the colour
of each mask source are all matched to the published figure — the Tülu3-derived
mask keeps the brown it has in the left panel, the Light-R1-derived mask keeps
the red it has in the right panel, random keeps its grey.

Two deliberate departures, both forced:

  y is REWARD, not loss. GRPO's `loss` is the policy-gradient surrogate: it
  oscillates about zero (±0.4) and its first-50 → last-50 change has no
  consistent sign across the three arms (+0.031→+0.043, -0.001→+0.014,
  +0.034→+0.015). Plotting it would reproduce the format while showing nothing.
  Reward moves 0.900→1.023, 0.901→1.024, 0.903→0.986. Note this flips the
  reading direction: in the published figure lower is better, here higher is.

  One panel, not two. A second panel would need sparse GRPO arms on a second
  dataset; only a dense run exists there.

  Lines are a debiased EMA (0.99, the wandb workspace setting), and only that —
  no faint raw trace underneath. The published figure plots raw values, which
  works there because DPO's loss has a large trend against small noise. Here the
  per-step reward carries ±0.4 of noise around an arm-to-arm separation of ~0.1,
  so raw curves overlay into a single band and the figure would assert, falsely,
  that the arms coincide.

These curves are the TRAINING objective. Held-out GSM8K separates no sparse arm
from the untrained base at this rho (grpo_matched_aicr_2026-07-29.md 4c).

Usage:
  python scripts/plot_grpo_transfer_curves.py --out out_prefix \
      oracle_dpo_lightr1=path.json oracle_dpo_tulu3=path.json random_seed42=path.json
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Sampled from 600-dpi renders of the submitted PDF: Figure 1 (p6) for the
# dense/oracle pair, Figure 3 (p7) for everything else.
BLUE  = "#2b7eb8"   # Figure 1's "DPO Dense Training"
GREEN = "#2b9d32"   # Figure 1's "DPO Oracle Light R1"
RED   = "#d83233"   # Figure 3's Light-R1 oracle curve
BROWN = "#925f54"   # Figure 3's Tulu3 oracle curve
GREY  = "#858585"   # Figure 3's random-mask curve
BLACK = "#1a1a1a"   # ReasonMed keeps the black it has in Figure 2's new pair line
INK, GRIDC, SPINEC = "#000000", "#d9d9d9", "#cccccc"

ARMS = {
    # key -> (label in the published figures' naming convention, colour)
    "dense":              ("GRPO Dense Training",   BLUE),
    "oracle_grpo_math":   ("GRPO Oracle OpenR1",    GREEN),
    "random_seed42":      ("GRPO Random Mask",      GREY),
    "oracle_dpo_lightr1": ("GRPO Oracle Light R1",  RED),
    "oracle_dpo_tulu3":   ("GRPO Oracle Tulu3",     BROWN),
    "oracle_reasonmed":   ("GRPO Oracle ReasonMed", BLACK),
}


def load_reward(path):
    h = json.load(open(path))["log_history"]
    pts = sorted((e["step"], e["reward"]) for e in h if "reward" in e)
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def ema(y, w=0.99):
    """Debiased exponential moving average — the TensorBoard/wandb formulation."""
    out, last, num = [], 0.0, 0.0
    for p in y:
        last = last * w + (1 - w) * p
        num = num * w + (1 - w)
        out.append(last / num)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("arms", nargs="+", help="key=trainer_state.json")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(5.2, 3.1), facecolor="white")
    for spec in args.arms:
        key, path = spec.split("=", 1)
        if key not in ARMS:
            raise SystemExit(f"unknown arm key {key!r}; known: {list(ARMS)}")
        label, colour = ARMS[key]
        step, reward = load_reward(path)
        ax.plot(step, ema(reward), color=colour, linewidth=1.6, label=label)

    ax.set_title("GRPO Open-R1 Math Reward over Time",
                 fontsize=11, fontweight="bold", color=INK, pad=8)
    ax.set_xlabel("Step", fontsize=10, color=INK)
    ax.set_ylabel("Reward", fontsize=10, color=INK)
    ax.set_xlim(0, 500)
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.grid(True, color=GRIDC, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(SPINEC)
        s.set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=9)

    leg = ax.legend(loc="lower right", fontsize=8.5, frameon=True,
                    edgecolor=SPINEC, framealpha=1.0, borderpad=0.5)
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, facecolor="white")
    print(f"wrote {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
