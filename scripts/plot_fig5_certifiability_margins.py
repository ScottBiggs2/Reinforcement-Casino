"""Regenerate the paper's Figure 5 (certifiability margins, PDF + ECDF panels)
from mask_score_gap_histograms.npz — with the reviewer-promised title and
larger labels (reply item 15: "Figure 5 gets a title").

Panel layout matches the submitted figure: left = PDF of margins
m_i(s) = |s_i - tau_rho(s)| on a log-x axis, right = ECDF; series are warm-mag
milestones (steps <=50/100/150/200), random seed 42, and the oracle.

Usage:
  python scripts/plot_fig5_certifiability_margins.py --npz mask_score_gap_histograms.npz --out prefix
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
# Milestone ramp (sequential, one hue light->dark) + distinct baselines.
STEP_COLOURS = {50: "#2a78d6", 100: "#eb6834", 150: "#1baf7a", 200: "#e34948"}
RANDOM_STYLE = dict(color="#52514e", linestyle=(0, (4, 2)))
ORACLE_STYLE = dict(color="#1d9a2f", linestyle=(0, (1, 1.5)))

LABEL_FS = 15
TICK_FS = 13
LEGEND_FS = 12.5
TITLE_FS = 16


def _ecdf_from_log_bins(counts, log_edges):
    total = float(counts.sum())
    xc = 10 ** ((log_edges[:-1] + log_edges[1:]) / 2.0)
    return xc, np.cumsum(counts.astype(np.float64)) / max(total, 1.0)


def _pdf_from_log_bins(counts, log_edges):
    total = float(counts.sum())
    lin_c = (10 ** log_edges[:-1] + 10 ** log_edges[1:]) / 2.0
    widths = (10 ** log_edges[1:] - 10 ** log_edges[:-1]).clip(min=1e-30)
    return lin_c, counts.astype(np.float64) / max(total, 1.0) / widths


def series_from_npz(z):
    """Yield (label, style, counts, log_edges) for margin histograms."""
    out = []
    for key in z.files:
        k = key.lower()
        if "margin" not in k or not k.endswith("_counts"):
            continue
        edges_key = key[: -len("_counts")] + "_log_edges"
        if edges_key not in z.files:
            continue
        if "magnitude" in k:
            import re
            m = re.search(r"step(\d+)", k)
            step = int(m.group(1)) if m else -1
            out.append((f"warm mag, step ≤ {step}",
                        dict(color=STEP_COLOURS.get(step, "#888"), linestyle="-"),
                        z[key], z[edges_key], (0, step)))
        elif "random" in k:
            out.append(("random (seed 42)", RANDOM_STYLE, z[key], z[edges_key], (1, 0)))
        elif "oracle" in k:
            out.append((r"oracle $|\theta^T_i-\theta^0_i|$", ORACLE_STYLE,
                        z[key], z[edges_key], (2, 0)))
    out.sort(key=lambda t: t[4])
    return [(a, b, c, d) for a, b, c, d, _ in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rho", default="97.5")
    args = ap.parse_args()

    z = np.load(args.npz)
    series = series_from_npz(z)
    if not series:
        raise SystemExit(f"no margin histograms found; keys = {list(z.files)[:20]} ...")

    fig, (a, b) = plt.subplots(1, 2, figsize=(15.5, 6.2), facecolor="white")

    for label, style, counts, log_edges in series:
        x, dens = _pdf_from_log_bins(counts, log_edges)
        a.plot(x, dens, linewidth=2.2, label=label, **style)
        x, cdf = _ecdf_from_log_bins(counts, log_edges)
        b.plot(x, cdf, linewidth=2.2, label=label, **style)

    for ax, ylab, sub in (
        (a, r"density $\hat f(m)$", "PDF"),
        (b, r"$\Pr[\,m_i(s) \leq x\,]$", "ECDF"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel(r"margin  $m_i(s) = |\,s_i - \hat{\tau}_\rho(s)\,|$", fontsize=LABEL_FS, color=INK)
        ax.set_ylabel(ylab, fontsize=LABEL_FS, color=INK)
        ax.set_title(sub, fontsize=TITLE_FS - 1, fontweight="600", color=INK, loc="left", pad=10)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=TICK_FS)
        leg = ax.legend(fontsize=LEGEND_FS, frameon=True, framealpha=0.95, edgecolor=GRID,
                        loc="upper left" if ax is a else "lower right")
        for t in leg.get_texts():
            t.set_color(INK)
    b.set_ylim(0, 1.02)

    fig.suptitle(
        rf"Certifiability margins of warm-start magnitude scoring at $\rho={args.rho}\%$ (DPO Light-R1)",
        fontsize=TITLE_FS + 1, fontweight="700", color=INK, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, facecolor="white")
    print(f"wrote {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
