"""Rebuild the submitted Figure 3 (DPO cross-task mask transfer, two panels)
with the rebuttal's new arms, plus a fourth curve: magnitude rewinding step200.

Faithful to the published panels: raw (unsmoothed) train/loss, tab10 colours as
they appear in the submission (left panel brown/pink/grey, right panel
red/green/purple — the original figure's per-panel default cycling is kept, not
harmonised), boxed in-plot legend, same titles/ticks/grid. The added magnitude
curve is tab:orange in both panels.

Run sources are pinned by run ID (FIGURE_RUN_REGISTRY.md is authoritative):
the two cross-task oracle arms per panel have no post-May rerun and remain the
May Explorer runs; random and magnitude are the 2026-07/08 AICR B200 e1-e2
reruns. Hardware is therefore mixed within a panel — disclosed in the registry.

Usage: python scripts/plot_reb_dpo_transfer_4arm.py [--out prefix]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENT = "xxiellan-northeastern-university"
INK, GRIDC, SPINEC = "#000000", "#d9d9d9", "#cccccc"

# tab10, as in the submitted figure's default cycling
BROWN, PINK, GREY = "#8c564b", "#e377c2", "#7f7f7f"
RED, GREEN, PURPLE = "#d62728", "#2ca02c", "#9467bd"
ORANGE = "#ff7f0e"  # the new magnitude arm, both panels

# panel -> [(project/run_id, label, colour)]
PANELS = {
    "DPO Light R1": [
        ("huggingface/gmar1wmy", "DPO Oracle Tulu3", BROWN),
        ("huggingface/h1pzy06y", "DPO Oracle OpenR1", PINK),
        ("dpo_lightr1/krz2rao7", "DPO Magnitude Rewinding Step200", ORANGE),
        ("dpo_lightr1/q1qik0lm", "DPO Random Mask", GREY),
    ],
    "DPO Tulu3": [
        ("huggingface/ih0xc9vx", "DPO Oracle Light R1", RED),
        ("huggingface/3z3ru6eg", "DPO Oracle OpenR1", GREEN),
        ("dpo_tulu3/7plzu31r", "DPO Magnitude Rewinding Step200", ORANGE),
        ("dpo_tulu3/cm4p962p", "DPO Random Mask", PURPLE),
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
    ap.add_argument("--out", default="docs/paper_drafts/reb_dpo_transfer_4arm")
    args = ap.parse_args()

    api = wandb.Api()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), facecolor="white")
    for ax, (panel, arms) in zip(axes, PANELS.items()):
        for path, label, colour in arms:
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
