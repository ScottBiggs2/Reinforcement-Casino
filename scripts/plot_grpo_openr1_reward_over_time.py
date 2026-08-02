"""The workspace panel "Open-R1 GRPO Reward over Time", rebuilt for export.

The wandb panel can't produce the figure we want to hand out: it stamps
"Showing first 10 runs" once the run set exceeds ten (this one holds eleven),
its legend order follows the runs-table sort rather than sparsity, and its line
weight isn't adjustable. So this pulls the same eleven histories over the HTTP
API — anchored by run ID per FIGURE_RUN_REGISTRY.md — and re-renders the panel:
bold lines, no run-limit stamp, legend reading Dense, then the oracle sweep
50 → 99.75 ascending, then Random.

Sparsity is an ordered quantity, so the nine oracle arms wear one green ramp
(light = 50%, dark = 99.75%) instead of nine unrelated hues — the plateau reads
as a single band and the eye only has to find where the dark end peels off.
Dense keeps Figure 1's blue, Random keeps Figure 3's grey (dashed). Lines are
the debiased EMA (0.99, the wandb workspace setting) over every logged point,
which is what the panel itself draws; y is clamped to [0.85, 1.1] to match the
workspace view.

Usage:
  python scripts/plot_grpo_openr1_reward_over_time.py \
      --out docs/paper_drafts/grpo_openr1_reward_over_time
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY, PROJECT = "xxiellan-northeastern-university", "g-rpo_group"

BLUE = "#2b7eb8"    # Figure 1's dense blue
GREY = "#858585"    # Figure 3's random-mask grey
INK, GRIDC, SPINEC = "#000000", "#d9d9d9", "#cccccc"

# Legend order is exactly this list. (name, run_id, rho or None)
ARMS = [
    ("Dense GRPO",    "19l4oj1g", None),
    ("Oracle 50%",    "iv5u2p50", 50.0),
    ("Oracle 60%",    "zt8m3fmw", 60.0),
    ("Oracle 70%",    "kmldaw7j", 70.0),
    ("Oracle 80%",    "zn6mlwro", 80.0),
    ("Oracle 90%",    "tnmd83vy", 90.0),
    ("Oracle 95%",    "grqecwgz", 95.0),
    ("Oracle 97.5%",  "0l98g0wm", 97.5),
    ("Oracle 99%",    "hgza08ki", 99.0),
    ("Oracle 99.75%", "1jbhb31m", 99.75),
    ("Random 97.5%",  "5o74vhmw", None),
]
ORACLE_CMAP = plt.get_cmap("Greens")
ORACLE_SHADES = ORACLE_CMAP(np.linspace(0.40, 1.0, 9))


def ema(y, w=0.99):
    """Debiased exponential moving average — the TensorBoard/wandb formulation."""
    out, last, num = [], 0.0, 0.0
    for p in y:
        last = last * w + (1 - w) * p
        num = num * w + (1 - w)
        out.append(last / num)
    return np.array(out)


def fetch(api, run_id):
    r = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    hist = [h for h in r.scan_history(keys=["train/global_step", "train/reward"])
            if h["train/reward"] is not None and h["train/global_step"] is not None]
    step = np.array([h["train/global_step"] for h in hist], dtype=float)
    reward = np.array([h["train/reward"] for h in hist], dtype=float)
    order = np.argsort(step, kind="stable")
    return step[order], reward[order]


def row_major(handles_labels, ncol):
    """matplotlib fills legends column-major; permute so rows read left-to-right."""
    h, l = handles_labels
    nrow = -(-len(h) // ncol)
    idx = [r * ncol + c for c in range(ncol) for r in range(nrow) if r * ncol + c < len(h)]
    return [h[i] for i in idx], [l[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/paper_drafts/grpo_openr1_reward_over_time")
    args = ap.parse_args()

    api = wandb.Api()
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor="white")

    oracle_i = 0
    for name, run_id, rho in ARMS:
        step, reward = fetch(api, run_id)
        if rho is not None:
            colour, ls, lw = ORACLE_SHADES[oracle_i], "-", 2.4
            oracle_i += 1
        elif name.startswith("Dense"):
            colour, ls, lw = BLUE, "-", 2.6
        else:
            colour, ls, lw = GREY, "--", 2.2
        ax.plot(step, ema(reward), color=colour, linestyle=ls, linewidth=lw,
                label=name, solid_capstyle="round")
        print(f"{name:16s} {len(step)} pts, final EMA {ema(reward)[-1]:.4f}")

    ax.set_title("Open-R1 GRPO Reward over Time",
                 fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_xlabel("Step", fontsize=10, color=INK)
    ax.set_ylabel("Reward", fontsize=10, color=INK)
    ax.set_xlim(0, 500)
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_ylim(0.85, 1.1)
    ax.grid(True, color=GRIDC, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(SPINEC)
        s.set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=9)

    ncol = 4
    leg = ax.legend(*row_major(ax.get_legend_handles_labels(), ncol),
                    loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=ncol,
                    fontsize=8.5, frameon=True, edgecolor=SPINEC,
                    framealpha=1.0, borderpad=0.6, columnspacing=1.2)
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, facecolor="white",
                    bbox_inches="tight")
    print(f"wrote {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
