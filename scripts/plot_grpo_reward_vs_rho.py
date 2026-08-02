"""Reward retention vs sparsity rho for the GRPO oracle sweep.

The GRPO sibling of scripts/plot_dpo_retention_vs_rho.py — same visual spec (single
panel, dense baseline at 0, paired per-step SE, boxed error-bar note). Keep the two in
sync by hand if the spec changes.

TWO DELIBERATE DIFFERENCES FROM THE DPO FIGURE, both forced by the objective:

1. It plots REWARD, not loss. GRPO's "loss" is a policy-gradient surrogate, not a fit
   measure: over the last 50 steps its own SE is 23-136 % of its value and it moves
   non-monotonically in rho (0.010 -> 0.042 -> 0.023). A loss-gap panel would be noise.
   Reward's SE is 2.2-2.8 % of its value. Reward is the quantity GRPO optimises.

2. The sign therefore means the OPPOSITE of the DPO figure: reward is higher-is-better,
   so a point ABOVE the dense line is an arm that BEAT dense. On the DPO loss figure,
   above the line means worse. Do not read the two side by side without saying so.

Usage:
  python scripts/plot_grpo_reward_vs_rho.py --dense dense.json --out prefix \
      70=sp70.json 80=sp80.json ... 99.75=sp99.75.json
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
TAIL = 50  # steps averaged for the "final" reward, matching the DPO figure


def rewards(path):
    h = json.load(open(path))["log_history"]
    pts = sorted((e["step"], e["reward"]) for e in h if "reward" in e)
    return np.array([p[1] for p in pts])


def style(ax, pos, labels):
    # ORDINAL x, unlike the DPO figure's linear rho axis. That figure had 4
    # levels (80/90/97.5/99) which fit on a true numeric axis; this sweep has 7
    # spanning 70..99.75, four of them inside the last 5 rho units, so linear
    # spacing collides both the ticks and the value labels. Equal spacing is the
    # usual convention for a sparsity ladder -- but it means horizontal distance
    # is NOT proportional to Δρ here.
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
    ap.add_argument("--dataset_label", default="MATH-220k · Llama-3.1-8B-Instruct")
    ap.add_argument("--absolute", action="store_true",
                    help="Plot R_sparse itself against a dense reference line, "
                         "instead of the R_sparse - R_dense gap.")
    ap.add_argument("--abs_gap", action="store_true",
                    help="Plot |R_sparse - R_dense| instead of the signed gap. "
                         "The sign (which arms BEAT dense) is folded away; it is "
                         "still printed to stdout.")
    ap.add_argument("arms", nargs="+", help="rho=trainer_state.json")
    args = ap.parse_args()
    if args.absolute and args.abs_gap:
        ap.error("--absolute and --abs_gap are mutually exclusive")

    dense = rewards(args.dense)
    d_final = dense[-TAIL:].mean()
    d_se = dense[-TAIL:].std(ddof=1) / np.sqrt(TAIL)

    rows = []
    for spec in args.arms:
        rho, path = spec.split("=", 1)
        s = rewards(path)
        # Paired SE, as in the DPO figure: the arms see the same prompts in the
        # same order. Pairing is weaker here (corr 0.41-0.62 vs 0.96 for DPO)
        # because GRPO samples completions, but it is still positive and still
        # the correct estimator -- hypot() would inflate these bars ~1.4x.
        paired = s[-TAIL:] - dense[-TAIL:]
        rows.append((float(rho), s[-TAIL:].mean(), paired.mean(),
                     paired.std(ddof=1) / np.sqrt(TAIL),
                     s[-TAIL:].std(ddof=1) / np.sqrt(TAIL)))

    rows.sort()
    rho = np.array([r[0] for r in rows])
    if args.absolute:
        # The paired SE estimates the uncertainty of a DIFFERENCE and is not the
        # error bar for an absolute level: it cancels the shared batch-difficulty
        # noise that an absolute reward still carries. Each arm gets its own
        # last-TAIL SE here, which is why these bars are wider than the gap bars.
        gap = np.array([r[1] for r in rows])
        gap_se = np.array([r[4] for r in rows])
    elif args.abs_gap:
        # |mean gap|, not mean |per-step gap|: the per-step |diff| would be
        # biased upward by sampling noise even for a truly zero gap. The paired
        # SE is unchanged -- it is the uncertainty of the underlying signed gap.
        gap = np.abs([r[2] for r in rows])
        gap_se = np.array([r[3] for r in rows])
    else:
        gap = np.array([r[2] for r in rows])
        gap_se = np.array([r[3] for r in rows])

    pos = np.arange(len(rho))

    fig, ax = plt.subplots(figsize=(8.4, 5.6), facecolor="white")

    ref = d_final if args.absolute else 0.0
    ax.axhline(ref, color=MUTED, linewidth=1.6, linestyle=(0, (5, 2)))
    if args.absolute:
        # Dense carries its own sampling noise; without the band the sparse bars
        # would appear to be measured against an exact number.
        ax.axhspan(ref - d_se, ref + d_se, color=MUTED, alpha=0.12, linewidth=0)
        # Right-hand end, below the line: every value label sits above its point,
        # and the leftmost of those lands exactly where a left-hand label would.
        ax.text(pos[-1] + 0.4, ref - d_se, f"dense {d_final:.4f}",
                color=MUTED, fontsize=11, va="top", ha="right")
    else:
        ax.text(pos[0] - 0.35, max(gap + gap_se) * 0.045, "dense",
                color=MUTED, fontsize=11, va="bottom")
    ax.errorbar(pos, gap, yerr=gap_se, color=BLUE, linewidth=2, marker="o",
                markersize=5, capsize=5, capthick=1.7, elinewidth=1.7, zorder=3)
    fmt = "{:.4f}" if (args.absolute or args.abs_gap) else "{:+.4f}"
    for x, y, se in zip(pos, gap, gap_se):
        ax.annotate(fmt.format(y), (x, y + se), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=10, color=INK)

    ax.set_title(f"GRPO reward retention vs mask sparsity\n{args.dataset_label}",
                 fontsize=15, fontweight="700", color=INK, pad=12)
    ax.set_xlabel("ρ  (% of weights left frozen at pretrained values)",
                  fontsize=13, color=MUTED)
    if args.absolute:
        ylabel = f"R_sparse   (last-{TAIL}-step mean)"
    elif args.abs_gap:
        ylabel = f"|R_sparse − R_dense|   (last-{TAIL}-step mean)"
    else:
        ylabel = f"R_sparse − R_dense   (last-{TAIL}-step mean)"
    ax.set_ylabel(ylabel, fontsize=13, color=MUTED)
    if args.abs_gap:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    lo = min(min(gap - gap_se), ref - d_se) if args.absolute else min(gap - gap_se)
    hi = max(max(gap + gap_se), ref + d_se) if args.absolute else max(gap + gap_se)
    pad = 0.32 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    style(ax, pos, [f"{r:g}" for r in rho])

    ax.text(0.985, 0.05, f"Error bars: ±1 SE (≈{gap_se.mean():.4f})",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color=MUTED,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=GRID, linewidth=1.2))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, facecolor="white")

    print(f"R_dense (last-{TAIL}) = {d_final:.4f} ± {d_se:.4f}")
    print(f"{'rho':>7} {'R_sparse':>10} {'±SE':>8} {'gap':>10} {'±SE':>8} {'sigma':>7}")
    for r, sf, g, gs, own in rows:
        print(f"{r:>7g} {sf:>10.4f} {own:>8.4f} {g:>+10.4f} {gs:>8.4f} {g/gs:>7.1f}")
    print(f"wrote {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
