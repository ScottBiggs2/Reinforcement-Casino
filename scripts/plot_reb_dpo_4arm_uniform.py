"""4-arm DPO loss figure, uniform colours across panels (Irene 2026-08-01).

Each panel shows the same four arms — random / dense / oracle / magnitude
rewinding step200 — and each arm keeps one colour in both panels: random
orange, dense blue, oracle green, magnitude red. All solid raw curves, in the
submitted figure's panel styling. Runs are the eight registry arms
(FIGURE_RUN_REGISTRY.md main table), pinned by run ID.

Usage: python scripts/plot_reb_dpo_4arm_uniform.py [--out prefix]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENT = "xxiellan-northeastern-university"
INK, GRIDC, SPINEC = "#000000", "#d9d9d9", "#cccccc"

ORANGE, BLUE, GREEN, RED = "#ff7f0e", "#2b7eb8", "#2b9d32", "#d83233"

# arm order and colours are identical in both panels
ARMS = [
    ("DPO Random Mask", ORANGE),
    ("DPO Dense Training", BLUE),
    ("DPO Oracle", GREEN),
    ("DPO Magnitude Rewinding Step200", RED),
]

# panel -> run paths in ARMS order (project/run_id)
PANELS = {
    "DPO Light R1": [
        "dpo_lightr1/q1qik0lm",           # random seed42 B200
        "dpo_lightr1/yr7ljjsr",           # dense B200
        "rl_casino_transfer_v1/qbclyo1k", # oracle sp97.5 (rho-sweep run, B200)
        "dpo_lightr1/krz2rao7",           # magnitude rewinding step200 e1-e2
    ],
    "DPO Tulu3": [
        "dpo_tulu3/cm4p962p",
        "dpo_tulu3/kvyy7w7u",
        "dpo_tulu3/y4wz3763",
        "dpo_tulu3/7plzu31r",
    ],
}

LEGEND_LOC = {"DPO Light R1": "center right", "DPO Tulu3": "lower left"}
YLIM = {"DPO Light R1": (-0.02, 0.75), "DPO Tulu3": (0.60, 0.74)}


def fetch(api, path):
    run = api.run(f"{ENT}/{path}")
    hist = run.history(keys=["train/global_step", "train/loss"], pandas=True,
                       samples=2000)
    hist = hist.dropna().sort_values("train/global_step")
    return hist["train/global_step"], hist["train/loss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/paper_drafts/reb_dpo_4arm_uniform")
    args = ap.parse_args()

    api = wandb.Api()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), facecolor="white")
    for ax, (panel, paths) in zip(axes, PANELS.items()):
        for (label, colour), path in zip(ARMS, paths):
            x, y = fetch(api, path)
            ax.plot(x, y, color=colour, linewidth=1.6, label=label)
        ax.set_title(f"{panel} Loss over Time", fontsize=11, fontweight="bold",
                     color=INK, pad=8)
        ax.set_xlabel("Step", fontsize=10, color=INK)
        ax.set_ylabel("Loss", fontsize=10, color=INK)
        ax.set_xlim(0, 500)
        ax.set_xticks([0, 100, 200, 300, 400, 500])
        ax.set_ylim(*YLIM[panel])
        ax.grid(True, color=GRIDC, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(SPINEC)
            s.set_linewidth(0.8)
        ax.tick_params(colors=INK, labelsize=9)
        leg = ax.legend(loc=LEGEND_LOC[panel], fontsize=8.5, frameon=True,
                        edgecolor=SPINEC, framealpha=1.0, borderpad=0.5)
        leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, facecolor="white")
    print(f"wrote {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
