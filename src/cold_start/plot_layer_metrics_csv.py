#!/usr/bin/env python3
"""Generate quick visualization panels from layer-metrics CSV files."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def to_float(x):
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def to_int_params(x):
    """Parse n_params from CSV; None if missing."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def has_valid(vals):
    return any(not math.isnan(v) for v in vals)


def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expected_random_jaccard(density_a: float, density_b: float) -> float:
    """Closed-form expectation: Jaccard of two independent Bernoulli masks with given densities."""
    denom = density_a + density_b - density_a * density_b
    if denom <= 0:
        return 0.0
    return (density_a * density_b) / denom


def plot_mask_overlap_reference(
    ax,
    x,
    expected_baseline,
    mc_mean,
    mc_lo,
    mc_hi,
    random_trials: int,
) -> None:
    """Draw closed-form E[Jaccard] + MC band for indep. Bernoulli masks (mask-overlap reference).

    On the CKA axes this is the **same** overlap reference as the Jaccard panel (comparable
    vertical scale), not the expectation of CKA under a random-activation null—that would
    require model forward passes or a separate derivation.
    """
    if has_valid(expected_baseline):
        ax.plot(
            x,
            expected_baseline,
            linestyle=":",
            linewidth=1.0,
            color="gray",
            alpha=0.85,
            label="E[Jaccard] (indep. Bernoulli, closed form)",
        )
    if (
        random_trials >= 2
        and has_valid(mc_mean)
        and has_valid(mc_lo)
        and has_valid(mc_hi)
    ):
        ax.fill_between(
            x,
            mc_lo,
            mc_hi,
            alpha=0.25,
            color="tab:orange",
            label=f"random mask MC (n={random_trials} trials/layer)",
        )
        ax.plot(
            x,
            mc_mean,
            linestyle="--",
            linewidth=1.0,
            color="tab:orange",
            alpha=0.85,
        )


def monte_carlo_jaccard_row(
    n_params: int,
    density_a: float,
    density_b: float,
    n_trials: int,
    rng: np.random.Generator,
) -> tuple:
    """Per layer: draw n_trials independent random mask pairs; return (mean, min, max) Jaccard."""
    if n_params is None or n_params < 1:
        return None, None, None
    if not (0.0 <= density_a <= 1.0 and 0.0 <= density_b <= 1.0):
        return None, None, None
    n = int(n_params)
    trials = np.empty(n_trials, dtype=np.float64)
    for t in range(n_trials):
        a = rng.random(n) < density_a
        b = rng.random(n) < density_b
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        trials[t] = float(inter) / float(union) if union > 0 else 0.0
    return float(trials.mean()), float(trials.min()), float(trials.max())


def plot_one(csv_path: Path, out_path: Path, random_trials: int, random_seed: int):
    rows = read_rows(csv_path)
    if not rows:
        return False

    rng = np.random.default_rng(random_seed)

    x = list(range(len(rows)))
    layer_labels = [r.get("layer", str(i)) for i, r in enumerate(rows)]

    jaccard = [to_float(r.get("jaccard")) for r in rows]
    cka = [to_float(r.get("cka")) for r in rows]
    sparsity_a = [to_float(r.get("sparsity_a")) for r in rows]
    sparsity_b = [to_float(r.get("sparsity_b")) for r in rows]
    erank_a = [to_float(r.get("effective_rank_a_norm")) for r in rows]
    erank_b = [to_float(r.get("effective_rank_b_norm")) for r in rows]
    n_params_col = [to_int_params(r.get("n_params")) for r in rows]

    # Closed-form E[Jaccard] under independent random masks with matched marginals
    expected_baseline = []
    for sa, sb in zip(sparsity_a, sparsity_b):
        if math.isnan(sa) or math.isnan(sb):
            s = sa if not math.isnan(sa) else sb
            d = (1.0 - s) if not math.isnan(s) else float("nan")
            if math.isnan(d):
                expected_baseline.append(float("nan"))
            else:
                expected_baseline.append(expected_random_jaccard(d, d))
        else:
            expected_baseline.append(expected_random_jaccard(1.0 - sa, 1.0 - sb))

    # Monte Carlo: n_trials independent random mask pairs per layer (shows spread, not a single draw)
    mc_mean = []
    mc_lo = []
    mc_hi = []
    for i, row in enumerate(rows):
        n_p = n_params_col[i]
        sa, sb = sparsity_a[i], sparsity_b[i]
        if math.isnan(sa) or math.isnan(sb):
            s = sa if not math.isnan(sa) else sb
            da = (1.0 - s) if not math.isnan(s) else float("nan")
            db = da
        else:
            da, db = 1.0 - sa, 1.0 - sb
        if math.isnan(da) or math.isnan(db):
            mc_mean.append(float("nan"))
            mc_lo.append(float("nan"))
            mc_hi.append(float("nan"))
            continue
        m, lo, hi = monte_carlo_jaccard_row(n_p, da, db, random_trials, rng)
        mc_mean.append(m if m is not None else float("nan"))
        mc_lo.append(lo if lo is not None else float("nan"))
        mc_hi.append(hi if hi is not None else float("nan"))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(csv_path.name, fontsize=12)

    # Jaccard + random reference (closed-form + MC band)
    ax = axes[0, 0]
    if has_valid(jaccard):
        ax.plot(x, jaccard, marker="o", linewidth=1.3, markersize=3, label="observed Jaccard", color="tab:blue")
        plot_mask_overlap_reference(
            ax, x, expected_baseline, mc_mean, mc_lo, mc_hi, random_trials
        )
        ax.set_ylim(
            0,
            max(
                1.0,
                max((v for v in jaccard if not math.isnan(v)), default=1.0) * 1.05,
            ),
        )
        ax.legend(fontsize=7, loc="best")
        ax.set_title("Per-layer Jaccard vs random baseline")
    else:
        ax.text(0.5, 0.5, "No Jaccard data", ha="center", va="center")
        ax.set_title("Per-layer Jaccard")
    ax.grid(alpha=0.3)

    # CKA + same mask-overlap reference (E[J] + MC band) for side-by-side context
    ax = axes[0, 1]
    if has_valid(cka):
        ax.plot(x, cka, marker="o", linewidth=1.3, markersize=3, color="tab:purple", label="CKA")
        show_overlap_ref = has_valid(expected_baseline) or (
            random_trials >= 2
            and has_valid(mc_mean)
            and has_valid(mc_lo)
            and has_valid(mc_hi)
        )
        if show_overlap_ref:
            plot_mask_overlap_reference(
                ax, x, expected_baseline, mc_mean, mc_lo, mc_hi, random_trials
            )
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6, loc="best")
        ax.set_title(
            "Per-layer CKA\n(gray/orange = indep. random mask overlap ref; not E[CKA] null)"
        )
    else:
        ax.text(0.5, 0.5, "No CKA data", ha="center", va="center")
        ax.set_title("Per-layer CKA")
    ax.grid(alpha=0.3)

    # Sparsity A/B
    ax = axes[1, 0]
    drawn = False
    if has_valid(sparsity_a):
        ax.plot(x, sparsity_a, marker="o", linewidth=1.3, markersize=3, label="sparsity_a")
        drawn = True
    if has_valid(sparsity_b):
        ax.plot(x, sparsity_b, marker="o", linewidth=1.3, markersize=3, label="sparsity_b")
        drawn = True
    if drawn:
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.set_title("Per-layer Sparsity")
    else:
        ax.text(0.5, 0.5, "No sparsity data", ha="center", va="center")
        ax.set_title("Per-layer Sparsity")
    ax.grid(alpha=0.3)

    # Effective rank normalized A/B (no Jaccard random line)
    ax = axes[1, 1]
    drawn = False
    if has_valid(erank_a):
        ax.plot(x, erank_a, marker="o", linewidth=1.3, markersize=3, label="erank_a_norm")
        drawn = True
    if has_valid(erank_b):
        ax.plot(x, erank_b, marker="o", linewidth=1.3, markersize=3, label="erank_b_norm")
        drawn = True
    if drawn:
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.set_title("Per-layer Effective Rank (normalized)")
    else:
        ax.text(0.5, 0.5, "No effective-rank data", ha="center", va="center")
        ax.set_title("Per-layer Effective Rank (normalized)")
    ax.grid(alpha=0.3)

    tick_step = max(1, len(x) // 12)
    tick_idx = list(range(0, len(x), tick_step))
    tick_text = [layer_labels[i].split(".")[-2] if "." in layer_labels[i] else str(i) for i in tick_idx]
    for ax in axes.ravel():
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_text, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Layer")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Create plot panels from layer-metrics CSV files.")
    parser.add_argument("--input-dir", type=str, default="masks")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--pattern", type=str, default="*.csv")
    parser.add_argument(
        "--random-trials",
        type=int,
        default=3,
        help="Monte Carlo trials per layer for random-mask Jaccard band (min–max shading).",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="RNG seed for Monte Carlo baselines.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    pattern = f"**/{args.pattern}" if args.recursive else args.pattern
    csv_files = sorted(input_dir.glob(pattern))

    # Ignore summary CSV artifacts.
    csv_files = [p for p in csv_files if not p.name.endswith("_summary.csv")]

    made = 0
    for p in csv_files:
        out = p.with_name(f"{p.stem}_plots.png")
        if plot_one(p, out, args.random_trials, args.random_seed):
            made += 1
            print(f"✓ {out}")

    print(f"Generated plot files: {made}")


if __name__ == "__main__":
    main()
