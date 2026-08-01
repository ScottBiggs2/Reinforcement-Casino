"""
Figures for the corrected-builder E1 rerun (rebuttal_e1_e2/RESULTS_E1_corrected.md).

Two panels, both light-surface paper PNGs (+PDF):
  fig1  grouped bar — phi/chance for every mask, grouped by cell, with the chance line at 1.0.
  fig2  ordinal line — warm-mask phi/chance vs rewind step k, one line per cell, oracle appended.

Colour follows the paper's own convention (RESULTS §4.11): warm masks are a single-hue ordinal
blue ramp (light→dark as k grows), the oracle is a separate hue (orange), random is a neutral gray
carrying a hatch so identity never rests on "it's gray" alone. Palette hexes are the validated
data-viz defaults; the ramp passed --ordinal and the {oracle, warm, random} families passed the
categorical CVD/contrast gates (random is an intentional neutral baseline).

Numbers are transcribed from …/e1_fixed/<cell>/e1_grad_energy*.json (phi_2d / chance_ew_2d).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---- data (phi_2d / chance_energy_weighted_2d) ----
CELLS = [
    ("Olmo-3-7B\nDPO · Light-R1", "A"),
    ("Qwen3-8B\nDPO · Light-R1", "B"),
    ("Llama-3.1-8B GRPO\ncross-obj (DPO grad)", "C1"),
    ("Llama-3.1-8B GRPO\ntrue GRPO grad", "C2"),
]
MASKS = ["oracle", "k=50", "k=100", "k=250", "random"]
RATIO = {  # cell tag -> [oracle, k50, k100, k250, random]
    "A":  [1.099, 1.082, 1.107, 1.102, 0.996],
    "B":  [2.023, 1.890, 2.043, 2.024, 0.902],
    "C1": [2.295, 2.499, 2.387, 2.313, 1.004],
    "C2": [2.173, 2.347, 2.255, 2.208, 0.993],
}

# ---- palette (validated) ----
ORACLE   = "#eb6834"                          # distinct hue
WARM     = ["#86b6ef", "#3987e5", "#1c5cab"]  # ordinal ramp, k=50→250 light→dark
RANDOM   = "#b9b8b2"                           # neutral baseline (+hatch)
INK      = "#0b0b0b"
INK2     = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
AXIS     = "#c3c2b7"
BAR_COLORS = [ORACLE, WARM[0], WARM[1], WARM[2], RANDOM]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

OUT = os.path.join(os.path.dirname(__file__), "..", "rebuttal_e1_e2", "figures")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK2, length=0, labelsize=12.5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    ax.xaxis.grid(False)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


# ============================ Figure 1: grouped bar ============================
def fig_grouped_bar():
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    n_masks = len(MASKS)
    group_w = 0.82
    bw = group_w / n_masks
    for gi, (_, tag) in enumerate(CELLS):
        for mi, v in enumerate(RATIO[tag]):
            x = gi + (mi - (n_masks - 1) / 2) * bw
            hatch = "////" if MASKS[mi] == "random" else None
            ax.bar(x, v, width=bw * 0.9, color=BAR_COLORS[mi], edgecolor="white",
                   linewidth=0.8, hatch=hatch, zorder=3)

    ax.axhline(1.0, color=MUTED, linewidth=1.6, linestyle=(0, (4, 3)), zorder=2)

    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([c for c, _ in CELLS], fontsize=12, color=INK)
    ax.set_ylabel("φ / chance", color=INK, fontsize=14)
    ax.set_ylim(0, 2.75)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    _style_axes(ax)
    ax.set_title("Gradient-energy capture on corrected full-coverage masks (ρ = 97.5%)",
                 color=INK, fontsize=15.5, fontweight="bold", pad=14, loc="center")

    legend = [Patch(facecolor=ORACLE, label="oracle (ckpt-diff)"),
              Patch(facecolor=WARM[0], label="warm k=50"),
              Patch(facecolor=WARM[1], label="warm k=100"),
              Patch(facecolor=WARM[2], label="warm k=250"),
              Patch(facecolor=RANDOM, hatch="////", label="random")]
    ax.legend(handles=legend, ncol=5, frameon=False, fontsize=12,
              loc="upper center", bbox_to_anchor=(0.5, -0.16), handlelength=1.4,
              columnspacing=1.6, labelcolor=INK2)
    save(fig, "e1_phi_by_cell")


# ============================ Figure 2: k-trend line ============================
def fig_k_trend():
    LINE = {"A": "#2a78d6", "B": "#eb6834", "C1": "#1baf7a", "C2": "#e87ba4"}
    LABEL = {"A": "Olmo3 DPO", "B": "Qwen3 DPO",
             "C1": "Llama GRPO · cross-obj", "C2": "Llama GRPO · true GRPO"}
    kx = [0, 1, 2]           # k = 50, 100, 250 (evenly spaced)
    ox = 3.0                 # oracle position
    fig, ax = plt.subplots(figsize=(9.6, 5.6))

    for tag, col in LINE.items():
        oracle, k50, k100, k250, _ = RATIO[tag]
        ys = [k50, k100, k250]
        ax.plot(kx, ys, "-", color=col, linewidth=2.4, zorder=3)
        ax.plot(kx, ys, "o", color=col, markersize=7.5, markeredgecolor="white",
                markeredgewidth=1.1, zorder=4)
        ax.plot(ox, oracle, "D", color=col, markersize=8, markerfacecolor="white",
                markeredgecolor=col, markeredgewidth=2.0, zorder=4)
        ax.plot([2, ox], [k250, oracle], ":", color=col, linewidth=1.4, alpha=0.7, zorder=2)
        ax.annotate(LABEL[tag], (ox, oracle), textcoords="offset points", xytext=(11, 0),
                    va="center", ha="left", fontsize=12, color=col, fontweight="bold")

    ax.axhline(1.0, color=MUTED, linewidth=1.6, linestyle=(0, (4, 3)), zorder=1)

    ax.set_xticks(kx + [ox])
    ax.set_xticklabels(["k=50", "k=100", "k=250", "oracle\n(k=T)"], fontsize=12.5, color=INK)
    ax.set_xlim(-0.25, 4.55)
    ax.set_ylim(0.8, 2.7)
    ax.set_ylabel("φ / chance", color=INK, fontsize=14)
    ax.set_xlabel("warm-start rewind step k  →  oracle", color=INK2, fontsize=12.5)
    _style_axes(ax)
    ax.set_title("Warm-mask capture falls as k → oracle:\ncloser to the oracle, less aligned with the initial gradient",
                 color=INK, fontsize=15, fontweight="bold", pad=12, loc="center")
    save(fig, "e1_phi_vs_k")


if __name__ == "__main__":
    fig_grouped_bar()
    fig_k_trend()
    print("wrote figures to", OUT)
    for f in sorted(os.listdir(OUT)):
        if f.startswith("e1_"):
            print("  ", f)
