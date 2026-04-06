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


def _merge_rows(rows_a, rows_b):
    """Align two CSV row lists by layer name. Returns list of (row_a, row_b) pairs."""
    index_b = {r["layer"]: r for r in rows_b}
    merged = []
    for ra in rows_a:
        rb = index_b.get(ra["layer"])
        if rb is not None:
            merged.append((ra, rb))
    return merged


def plot_compare(csv_a: Path, csv_b: Path, label_a: str, label_b: str, out_path: Path):
    """4-panel comparison figure: two CSVs (e.g. SNIP vs CAV) overlaid on the same axes.

    Panel layout:
      [0,0] Per-layer Jaccard        — label_a line, label_b line, random baseline
      [0,1] Per-layer CKA            — same
      [1,0] Per-layer Sparsity       — 4 lines: A-GRPO, A-DPO, B-GRPO, B-DPO
      [1,1] Per-layer Eff. Rank norm — same 4 lines
    """
    rows_a = read_rows(csv_a)
    rows_b = read_rows(csv_b)
    if not rows_a or not rows_b:
        print(f"  [compare] Empty CSV — skipping {out_path.name}")
        return False

    merged = _merge_rows(rows_a, rows_b)
    if not merged:
        print(f"  [compare] No shared layers between {csv_a.name} and {csv_b.name}")
        return False

    pairs_a, pairs_b = zip(*merged)
    x = list(range(len(merged)))
    layer_labels = [r["layer"] for r in pairs_a]

    def col(rows, key):
        return [to_float(r.get(key)) for r in rows]

    jac_a    = col(pairs_a, "jaccard")
    jac_b    = col(pairs_b, "jaccard")
    cka_a    = col(pairs_a, "cka")
    cka_b    = col(pairs_b, "cka")
    spa_a    = col(pairs_a, "sparsity_a")   # label_a GRPO
    spb_a    = col(pairs_a, "sparsity_b")   # label_a DPO
    spa_b    = col(pairs_b, "sparsity_a")   # label_b GRPO
    spb_b    = col(pairs_b, "sparsity_b")   # label_b DPO
    era_a    = col(pairs_a, "effective_rank_a_norm")
    erb_a    = col(pairs_a, "effective_rank_b_norm")
    era_b    = col(pairs_b, "effective_rank_a_norm")
    erb_b    = col(pairs_b, "effective_rank_b_norm")

    # Random Jaccard baseline using label_a sparsity (both should be ~same target)
    rand_bl = []
    for sa, sb in zip(spa_a, spb_a):
        if math.isnan(sa) or math.isnan(sb):
            rand_bl.append(float("nan"))
        else:
            rand_bl.append(expected_random_jaccard(1.0 - sa, 1.0 - sb))

    # Color scheme: blue=label_a, orange=label_b; solid=GRPO, dashed=DPO
    CA_G = "tab:blue";   CA_D = "tab:blue"
    CB_G = "tab:orange"; CB_D = "tab:orange"

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"GRPO vs DPO  |  {label_a} (blue) vs {label_b} (orange)  |  solid=GRPO  dashed=DPO",
        fontsize=11,
    )

    # ── Panel 0,0 : Jaccard ────────────────────────────────────────────
    ax = axes[0, 0]
    kw = dict(linewidth=1.4, markersize=3, marker="o")
    if has_valid(jac_a):
        ax.plot(x, jac_a, color=CA_G, label=f"{label_a}", **kw)
    if has_valid(jac_b):
        ax.plot(x, jac_b, color=CB_G, label=f"{label_b}", **kw)
    if has_valid(rand_bl):
        ax.plot(x, rand_bl, linestyle="--", linewidth=1.0, color="gray", alpha=0.6, label="random baseline")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-layer Jaccard  (GRPO ∩ DPO mask overlap)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panel 0,1 : CKA ───────────────────────────────────────────────
    ax = axes[0, 1]
    if has_valid(cka_a):
        ax.plot(x, cka_a, color=CA_G, label=f"{label_a}", **kw)
    else:
        ax.text(0.5, 0.5, f"No CKA data for {label_a}", ha="center", va="center", fontsize=9)
    if has_valid(cka_b):
        ax.plot(x, cka_b, color=CB_G, label=f"{label_b}", **kw)
    else:
        ax.text(0.5, 0.45, f"No CKA data for {label_b}", ha="center", va="center", fontsize=9)
    if has_valid(rand_bl):
        ax.plot(x, rand_bl, linestyle="--", linewidth=1.0, color="gray", alpha=0.6, label="random baseline")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-layer CKA  (GRPO vs DPO representation similarity)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panel 1,0 : Sparsity ──────────────────────────────────────────
    ax = axes[1, 0]
    kw_s = dict(linewidth=1.3, markersize=2.5, marker="o")
    kw_d = dict(linewidth=1.3, markersize=2.5, marker="o", linestyle="--", alpha=0.75)
    if has_valid(spa_a): ax.plot(x, spa_a, color=CA_G, label=f"{label_a} GRPO", **kw_s)
    if has_valid(spb_a): ax.plot(x, spb_a, color=CA_D, label=f"{label_a} DPO",  **kw_d)
    if has_valid(spa_b): ax.plot(x, spa_b, color=CB_G, label=f"{label_b} GRPO", **kw_s)
    if has_valid(spb_b): ax.plot(x, spb_b, color=CB_D, label=f"{label_b} DPO",  **kw_d)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-layer Sparsity  (solid=GRPO  dashed=DPO)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panel 1,1 : Effective Rank ────────────────────────────────────
    ax = axes[1, 1]
    if has_valid(era_a): ax.plot(x, era_a, color=CA_G, label=f"{label_a} GRPO", **kw_s)
    if has_valid(erb_a): ax.plot(x, erb_a, color=CA_D, label=f"{label_a} DPO",  **kw_d)
    if has_valid(era_b): ax.plot(x, era_b, color=CB_G, label=f"{label_b} GRPO", **kw_s)
    if has_valid(erb_b): ax.plot(x, erb_b, color=CB_D, label=f"{label_b} DPO",  **kw_d)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-layer Effective Rank (normalized)  (solid=GRPO  dashed=DPO)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Shared x-axis ticks ───────────────────────────────────────────
    tick_step = max(1, len(x) // 12)
    tick_idx  = list(range(0, len(x), tick_step))
    tick_text = [
        layer_labels[i].split(".")[-2] if "." in layer_labels[i] else str(i)
        for i in tick_idx
    ]
    for ax in axes.ravel():
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_text, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Layer")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Create plot panels from layer-metrics CSV files.")
    parser.add_argument("--input-dir", type=str, default="masks")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--pattern", type=str, default="*.csv")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path.home() / "figs"),
        help="Directory to write plot PNGs (created if missing). Default: ~/figs.",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=3,
        help="Monte Carlo trials per layer for random-mask Jaccard band (min–max shading).",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="RNG seed for Monte Carlo baselines.")

    # Compare mode: overlay two CSVs on one 4-panel figure
    parser.add_argument(
        "--compare", action="store_true",
        help="Overlay two CSVs on one figure. Requires --csv-a and --csv-b.",
    )
    parser.add_argument("--csv-a", type=str, default=None, help="First CSV (e.g. SNIP GRPO vs DPO)")
    parser.add_argument("--csv-b", type=str, default=None, help="Second CSV (e.g. CAV GRPO vs DPO)")
    parser.add_argument("--label-a", type=str, default="A", help="Legend label for csv-a")
    parser.add_argument("--label-b", type=str, default="B", help="Legend label for csv-b")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output PNG path for --compare mode")

    args = parser.parse_args()

    if args.compare:
        if not args.csv_a or not args.csv_b:
            parser.error("--compare requires --csv-a and --csv-b")
        ca = Path(args.csv_a)
        cb = Path(args.csv_b)
        out = Path(args.output) if args.output else ca.parent / f"compare_{args.label_a}_vs_{args.label_b}.png"
        plot_compare(ca, cb, args.label_a, args.label_b, out)
        return

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = f"**/{args.pattern}" if args.recursive else args.pattern
    csv_files = sorted(input_dir.glob(pattern))

    # Ignore summary CSV artifacts.
    csv_files = [p for p in csv_files if not p.name.endswith("_summary.csv")]

    made = 0
    for p in csv_files:
        out = output_dir / f"{p.stem}_plots.png"
        if plot_one(p, out, args.random_trials, args.random_seed):
            made += 1
            print(f"✓ {out}")

    print(f"Generated plot files: {made}")


if __name__ == "__main__":
    main()
