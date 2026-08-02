"""Loss retention vs sparsity rho for the DPO oracle sweep (Nadim's request).

One panel: rho (fraction of weights zeroed) on x, and on y the absolute gap

    dL(rho) = L_sparse_final - L_dense_final

where the finals are last-50-step means and error bars are the last-50 SEs,
propagated as hypot(se_sparse, se_dense) for two independent runs.

The retention fraction (L_start - L_sparse) / (L_start - L_dense) is still
computed and PRINTED, but deliberately not plotted: L_start and L_dense are the
same constants for every arm, so retention = 1 - dL/D is an affine transform of
the y already shown -- a second panel would carry no extra information, and its
narrow y-range makes a ~1-sigma spread read as a trend. Print it, don't plot it.

Usage:
  python scripts/plot_dpo_retention_vs_rho.py --dense dense.json --out prefix \
      80=sp80.json 90=sp90.json 97.5=sp97.5.json 99=sp99.json
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BLUE = "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
TAIL = 50  # steps averaged for the "final" loss


def losses(path):
    h = json.load(open(path))["log_history"]
    pts = sorted((e["step"], e["loss"]) for e in h if "loss" in e)
    return np.array([p[1] for p in pts])


def style(ax, pos, labels):
    # ORDINAL x, matching scripts/plot_grpo_reward_vs_rho.py. The linear rho axis
    # this figure used worked at 4 levels (80/90/97.5/99); at 7 spanning 60..99.75
    # three of them fall inside the last 2.25 rho units and both the ticks and the
    # value labels collide. Equal spacing is the usual convention for a sparsity
    # ladder -- but horizontal distance is then NOT proportional to Δρ.
    # The stagger() helper the linear axis needed is gone with it: evenly spaced
    # points cannot crowd, so every label sits above its marker.
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlim(pos[0] - 0.45, pos[-1] + 0.45)
    ax.grid(True, axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset_label", default="Light-R1 · Llama-3.1-8B-Instruct")
    ap.add_argument("--abs_gap", action="store_true",
                    help="Plot |L_sparse - L_dense| instead of the signed gap "
                         "(matching plot_grpo_reward_vs_rho.py --abs_gap). "
                         "Signed values are still printed to stdout.")
    ap.add_argument("arms", nargs="+", help="rho=trainer_state.json")
    args = ap.parse_args()

    dense = losses(args.dense)
    l_start = dense[:5].mean()
    d_final = dense[-TAIL:].mean()
    d_se = dense[-TAIL:].std(ddof=1) / np.sqrt(TAIL)
    denom = l_start - d_final

    rows = []
    for spec in args.arms:
        rho, path = spec.split("=", 1)
        s = losses(path)
        s_final = s[-TAIL:].mean()
        s_se = s[-TAIL:].std(ddof=1) / np.sqrt(TAIL)
        # sparse - dense, the gap/regret convention: positive, and HIGHER = WORSE.
        # Deliberately not dense - sparse: that puts every point below the dense
        # baseline, where "below" collides with the lower-is-better reading of
        # loss itself and gets read as sparse beating dense. It does not.
        #
        # PAIRED SE, not hypot(se_sparse, se_dense). All five arms consume the
        # same data in the same order -- logps/chosen is bit-identical at step 1
        # across every trainer_state.json, and corr(L_sparse, L_dense) over the
        # last 50 steps is +0.96..+0.97. Treating them as independent inflates
        # the bars ~3.7x by double-counting shared batch-difficulty noise.
        paired = s[-TAIL:] - dense[-TAIL:]
        gap = paired.mean()
        gap_se = paired.std(ddof=1) / np.sqrt(TAIL)
        # retention = 1 - gap/denom, so it inherits the paired SE directly
        ret = (l_start - s_final) / denom
        ret_se = gap_se / denom
        rows.append((float(rho), s_final, gap, gap_se, ret, ret_se))

    rows.sort()
    rho = np.array([r[0] for r in rows])
    gap = np.array([r[2] for r in rows])
    gap_se = np.array([r[3] for r in rows])
    if args.abs_gap:
        # |mean gap|, not mean |per-step gap| -- the latter is biased upward by
        # sampling noise even when the true gap is zero. Paired SE unchanged.
        gap = np.abs(gap)

    pos = np.arange(len(rho))

    fig, ax = plt.subplots(figsize=(8.4, 5.6), facecolor="white")

    ax.axhline(0.0, color=MUTED, linewidth=1.6, linestyle=(0, (5, 2)))
    ax.text(pos[0] - 0.35, 0.0006, "dense", color=MUTED, fontsize=11, va="bottom")
    # Marker deliberately smaller than the dataviz default 8pt: the paired SE is
    # ~0.0002, so at this y-range the whole bar spans ~15px. An 8pt dot (22px)
    # swallows it and the point reads as having no error bar at all. Shrinking
    # the dot exposes the TRUE bar rather than inflating it.
    ax.errorbar(pos, gap, yerr=gap_se, color=BLUE, linewidth=2, marker="o",
                markersize=5, capsize=5, capthick=1.7, elinewidth=1.7, zorder=3)
    fmt = "{:.4f}" if args.abs_gap else "{:+.4f}"
    for x, y, se in zip(pos, gap, gap_se):
        ax.annotate(fmt.format(y), (x, y + se), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=11, color=INK)

    ax.set_title(f"DPO loss retention vs mask sparsity\n{args.dataset_label}",
                 fontsize=15, fontweight="700", color=INK, pad=12)
    # "frozen at zero" would be wrong: SparseAdamW just does not update masked
    # coords, so they keep their PRETRAINED values. This is a subnetwork result,
    # not pruning, and the mislabel invites exactly that misreading.
    ax.set_xlabel("ρ  (% of weights left frozen at pretrained values)",
                  fontsize=13, color=MUTED)
    ax.set_ylabel(
        f"|L_sparse − L_dense|   (last-{TAIL}-step mean)" if args.abs_gap
        else f"L_sparse − L_dense   (last-{TAIL}-step mean)",
        fontsize=13, color=MUTED)
    if args.abs_gap:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.set_ylim(-0.0015, max(gap + gap_se) * 1.22)
    style(ax, pos, [f"{r:g}" for r in rho])

    # Small boxed sidebar, bottom-right inside the axes. Bars this short read as
    # "no error bars" otherwise; the box states their size in the plotted units.
    # "±1 SE" is the load-bearing half — it says WHAT the bar is (68% coverage);
    # the number only says how long. Dropping the former would leave a magnitude
    # with no interpretation. The number is kept because the bars are ~15px and
    # cannot be measured off the axis by eye.
    # Top-left. The rho=99.75 cliff sets the y-range, so the plateau collapses to
    # the floor and the rise occupies the right half: bottom-right sits on the
    # plateau points, mid-right sits on the rising segment. Top-left is the only
    # region the data does not enter.
    ax.text(0.015, 0.97, f"Error bars: ±1 SE (≈{gap_se.mean():.4f})",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, color=MUTED,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=GRID, linewidth=1.2))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, facecolor="white")

    print(f"L_start {l_start:.4f}   L_dense {d_final:.4f} ± {d_se:.4f}")
    print(f"{'rho':>6} {'L_sparse':>10} {'gap':>10} {'±':>8} {'retention':>10} {'±':>8}")
    for r, sf, g, gs, rt, rs in rows:
        print(f"{r:>6g} {sf:>10.4f} {g:>+10.4f} {gs:>8.4f} {rt:>10.4f} {rs:>8.4f}")
    print(f"wrote {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
