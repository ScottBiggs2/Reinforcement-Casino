"""Figure 2, reproduced exactly, plus the one line the submitted version could not have.

Everything is matched to the published figure: the suptitle, the boxed 3-column
legend above the panels, the panel titles, both axis labels, the 0..1 y-range,
and every line's colour and dash pattern. Those six colours were sampled off a
600-dpi render of page 7 of the submitted PDF rather than guessed, because the
palette is a hand-picked mix (Okabe-Ito pink/amber/vermillion, an IBM purple)
that no single named matplotlib palette reproduces.

The single addition is GRPO-OpenR1 ⇄ GRPO-Tülu3RLVR, which needed a second GRPO
training run on a different dataset to exist at all. It is solid, following the
figure's own grammar (solid = oracle⇄oracle, dashed = ⇄random), and black —
the one value that cannot be confused with any of the six under any colour-vision
deficiency, and which reads as "this is the new one".

Usage: python scripts/plot_fig2_jaccard_cka.py multi_mask_jaccard_cka.json out_prefix
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Sampled from the published figure (600-dpi render, modal colour of each line core).
BLUE       = "#0d79b6"   # DPO-LightR1 ⇄ DPO-Tulu3
VERMILLION = "#d7660d"   # DPO-LightR1 ⇄ GRPO-OpenR1
GREEN      = "#0da37a"   # DPO-Tulu3 ⇄ GRPO-OpenR1
PINK       = "#ce80ab"   # DPO-LightR1 ⇄ Random
AMBER      = "#e7a40d"   # DPO-Tulu3 ⇄ Random
PURPLE     = "#7f66f0"   # GRPO-OpenR1 ⇄ Random
NEW        = "#b3002d"   # GRPO-OpenR1 ⇄ GRPO-Tulu3RLVR  (the addition)
BLACK      = "#111111"   # GRPO-OpenR1 ⇄ GRPO-ReasonMed  (second GRPO⇄GRPO cell,
                         # 2026-07-31: cross-DOMAIN — math vs medical MCQ — where
                         # Tulu3RLVR was cross-dataset within math)

GRIDC, SPINEC, INK = "#d9d9d9", "#cccccc", "#000000"

# The three ⇄random pairs are numerically on top of each other: from layer 1 on they
# all sit in 0.0124-0.0134, about one pixel apart on a 0-1 axis, so no vertical
# styling can separate them. They are interleaved instead — each drawn on one third
# of every PHASE_LAYERS-wide slot, so the three tile the band evenly.
#
# The slots are cut in DATA coordinates, not with a dash pattern. A dash phase
# advances along path length, and pink/amber carry a 0.067->0.013 drop at layer 0
# that purple does not, so their phases desynchronise immediately and stay that way
# — which is exactly how one of the three ended up as a sliver behind the other two.
PHASE_LAYERS = 1.5
DASH_A, DASH_B, DASH_C = 0, 1, 2   # phase slot index, not a linestyle

# Order is the legend's column-major fill order, so the printed legend matches
# the published one entry for entry, with the new pair appended last.
SERIES = [
    ("DPO-LightR1 ⇄ DPO-Tulu3",      "DPO-LightR1 ⇄ DPO-Tulu3",       BLUE,       "-",    2.0),
    ("DPO-LightR1 ⇄ GRPO-OpenR1",    "DPO-LightR1 ⇄ GRPO-OpenR1",     VERMILLION, "-",    2.0),
    ("DPO-LightR1 ⇄ Random-s42",     "DPO-LightR1 ⇄ Random",          PINK,       DASH_A, 1.5),
    ("DPO-Tulu3 ⇄ GRPO-OpenR1",      "DPO-Tulu3 ⇄ GRPO-OpenR1",       GREEN,      "-",    2.0),
    ("DPO-Tulu3 ⇄ Random-s42",       "DPO-Tulu3 ⇄ Random",            AMBER,      DASH_B, 1.5),
    ("GRPO-OpenR1 ⇄ Random-s42",     "GRPO-OpenR1 ⇄ Random",          PURPLE,     DASH_C, 1.5),
    ("GRPO-OpenR1 ⇄ GRPO-Tulu3RLVR", "GRPO-OpenR1 ⇄ GRPO-Tulu3",      NEW,        "-",    2.2),
    # Densely dotted, not solid: this line sits within 0.01 of the crimson
    # GRPO⇄GRPO-Tulu3 line across all 32 layers (0.2966 vs 0.3051 aggregate) and
    # the two merge into one dark band in grayscale/CVD. Dotted is deliberately
    # NOT the dashed pattern — dashed means "⇄ random" in this figure's grammar.
    ("GRPO-OpenR1 ⇄ GRPO-ReasonMed", "GRPO-OpenR1 ⇄ GRPO-ReasonMed",  BLACK, (0, (1.6, 1.3)), 2.4),
]


def jaccard_series(entry):
    by = entry.get("by_decoder_layer", {})
    idx = sorted(int(k) for k in by if k.isdigit())
    return np.array(idx), np.array([by[str(i)]["aggregate_jaccard"] for i in idx])


def cka_series(entry):
    pts = []
    for name, v in entry.get("per_layer", {}).items():
        if v is None:
            continue
        parts = name.split(".")
        if "layers" in parts:
            pts.append((int(parts[parts.index("layers") + 1]), v))
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def phase_slot(x, y, slot, n_slots=3, period=PHASE_LAYERS, samples=1400):
    """Resample onto a fine grid and blank every point outside this series' slot.

    Returns arrays with NaN gaps, which matplotlib renders as a broken line — an
    evenly tiled dash whose period is fixed in layer units and therefore identical
    across curves regardless of how much each one wobbles.
    """
    xs = np.linspace(x.min(), x.max(), samples)
    ys = np.interp(xs, x, y)
    which = np.floor((xs / period) * n_slots).astype(int) % n_slots
    ys = np.where(which == slot, ys, np.nan)
    return xs, ys


def lookup(d, key):
    """Pair keys are unordered upstream; accept either orientation."""
    if key in d:
        return d[key]
    a, b = key.split(" ⇄ ")
    return d.get(f"{b} ⇄ {a}")


def panel(ax, data, extract, title, ylabel, interleave):
    """interleave: tile the three ⇄random curves into alternating slots.

    Only the Jaccard panel needs it — there the three are within ~0.001 of each
    other and hide one another. In CKA they range over 0.1-0.67 and are already
    separated, so interleaving would just chop readable curves into fragments.
    """
    for key, label, colour, ls, lw in SERIES:
        entry = lookup(data, key)
        if entry is None:
            continue
        x, y = extract(entry)
        if not len(x):
            continue
        if isinstance(ls, int):                       # a ⇄random pair
            if interleave:
                x, y = phase_slot(x, y, ls)
                style = "-"
            else:
                style = (0, (4.2, 2.1))
        else:
            style = ls
        ax.plot(x, y, color=colour, linestyle=style, linewidth=lw, label=label,
                solid_capstyle="butt")
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    ax.set_xlabel("Layer", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, fontsize=11, color=INK)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0, 31)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    # 4:6 height:width. Exactly square cells (5/6.2) read as too boxy against the
    # published figure's wide panels, so this sits between the two.
    ax.set_box_aspect(4 / 6)
    ax.grid(True, color=GRIDC, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(SPINEC)
        s.set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=10)


def main():
    src, prefix = sys.argv[1], sys.argv[2]
    d = json.load(open(src))
    jac, cka = d["jaccard"], d["cka"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.7), facecolor="white")
    panel(a, jac, jaccard_series, "Jaccard per layer", "Jaccard", interleave=True)
    panel(b, cka, cka_series, "CKA per layer", "CKA", interleave=False)

    # The interleaved ⇄random series are drawn as solid lines with NaN gaps, so their
    # auto handles would read as solid. Swap in dashed proxies to keep the published
    # figure's "dashed = compared against a random mask" cue.
    from matplotlib.lines import Line2D
    handles, labels = [], []
    for key, label, colour, ls, lw in SERIES:
        if isinstance(ls, int):
            handles.append(Line2D([], [], color=colour, linestyle=(0, (4.2, 2.1)), linewidth=lw))
        else:
            handles.append(Line2D([], [], color=colour, linestyle=ls, linewidth=lw))
        labels.append(label)
    # Four columns, not the published three: matplotlib fills column-major, so
    # 8 entries at ncol=4 stay two rows deep with the original six in their
    # original slots and the two new pairs appended on the right. At ncol=3 the
    # legend would grow to three rows and squash the panels — a layout change
    # the figure did not ask for.
    leg = fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8.5,
                     frameon=True, edgecolor=SPINEC, framealpha=1.0,
                     bbox_to_anchor=(0.5, 0.955), handlelength=2.4,
                     columnspacing=1.6, borderpad=0.55)
    leg.get_frame().set_linewidth(0.8)

    fig.suptitle("Oracle and Random Mask Comparisons",
                 fontsize=13, fontweight="bold", color=INK, y=0.99)
    # set_box_aspect fixes the panel height, so tight_layout can no longer shrink
    # the axes to make room and silently clips the x-label; reserve the strip.
    fig.tight_layout(rect=[0, 0.05, 1, 0.87])
    for ext in ("png", "pdf"):
        fig.savefig(f"{prefix}.{ext}", dpi=300, facecolor="white")
    print(f"wrote {prefix}.png/.pdf")

    for key, label, *_ in SERIES:
        ej, ec = lookup(jac, key), lookup(cka, key)
        if ej and ec:
            print(f"  {label:<28} Jaccard {ej['aggregate']:.4f}   CKA {ec['per_layer_mean']:.4f}")


if __name__ == "__main__":
    main()
