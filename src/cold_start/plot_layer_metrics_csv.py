#!/usr/bin/env python3
"""Generate quick visualization panels from layer-metrics CSV files."""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Optional

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


def expected_linear_cka_null_reference(
    n_samples: int,
    mode: str = "inv_sq",
) -> float:
    """Order-of-magnitude floor for linear CKA (`mask_to_cka.py`) under unrelated representations.

    This is **not** the same object as mask-overlap Jaccard: CKA is computed on finite-sample
    activations with the biased linear CKA estimator. For independent high-dimensional
    activations, values are typically tiny; a common back-of-envelope visualization line is
    O(1/n) or smaller in finite ``n`` (calibration batch size).

    Modes (pick one; use ``--cka-null-value`` to override entirely):

    - ``inv_sq``: ``1 / n**2`` — very small “vanishing” reference (default).
    - ``inv_nm1``: ``1 / (n-1)`` — looser finite-``n`` reference (often still >> empirical CKA).

    For calibrated nulls, use permutation over sequences or bootstrap; this line is only a chart aid.
    """
    n = int(n_samples)
    if n < 2:
        return float("nan")
    if mode == "inv_sq":
        return 1.0 / (float(n) * float(n))
    if mode == "inv_nm1":
        return 1.0 / float(n - 1)
    raise ValueError(f"unknown cka null mode: {mode}")


def _clamp_positive_log(y: float, eps: float = 1e-12) -> float:
    if math.isnan(y):
        return y
    return max(eps, min(1.0, y))


def _series_for_log_scale(vals: list, eps: float = 1e-12) -> list:
    return [_clamp_positive_log(v, eps) if not math.isnan(v) else float("nan") for v in vals]


def _apply_metric_y_scale(
    ax,
    y_scale: str,
    *,
    ymin_floor: float = 1e-12,
    ymax: float = 1.05,
) -> None:
    ys = y_scale.lower().strip()
    if ys == "linear":
        return
    if ys == "log":
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(ymin_floor, 1e-30), top=ymax)
        return
    raise ValueError(f"unknown y_scale: {y_scale} (use linear or log)")


def plot_mask_overlap_reference(
    ax,
    x,
    expected_baseline,
    mc_mean,
    mc_lo,
    mc_hi,
    random_trials: int,
) -> None:
    """Draw closed-form E[Jaccard] and optionally an MC band for indep. Bernoulli masks.

    Used on the **Jaccard** panel only. CKA uses `expected_linear_cka_null_reference`, not overlap.
    """
    if has_valid(expected_baseline):
        ax.plot(
            x,
            expected_baseline,
            linestyle=":",
            linewidth=1.2,
            color="gray",
            alpha=0.9,
            label="Jaccard null: E[overlap] (indep. Bernoulli, closed form)",
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
            alpha=0.28,
            color="tab:orange",
            label=f"Jaccard null: MC band (n={random_trials} indep. mask pairs/layer)",
        )
        ax.plot(
            x,
            mc_mean,
            linestyle="--",
            linewidth=1.1,
            color="tab:orange",
            alpha=0.9,
        )


def monte_carlo_jaccard_row(
    n_params: int,
    density_a: float,
    density_b: float,
    n_trials: int,
    rng: np.random.Generator,
) -> tuple:
    """Per layer: draw n_trials independent random mask pairs; return (mean, min, max) Jaccard."""
    if n_trials < 2:
        return None, None, None
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


def plot_one(
    csv_path: Path,
    out_path: Path,
    random_trials: int,
    random_seed: int,
    *,
    y_scale: str = "linear",
    cka_n_samples: int = 64,
    cka_null_mode: str = "inv_sq",
    cka_null_scale: float = 1.0,
    cka_null_value: Optional[float] = None,
    log_y_floor: float = 1e-12,
):
    rows = read_rows(csv_path)
    if not rows:
        return False

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

    # Optional Monte Carlo band (only if random_trials >= 2). Otherwise use closed-form E[J] only.
    mc_mean = []
    mc_lo = []
    mc_hi = []
    if random_trials >= 2:
        rng = np.random.default_rng(random_seed)
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
    else:
        for _ in rows:
            mc_mean.append(float("nan"))
            mc_lo.append(float("nan"))
            mc_hi.append(float("nan"))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(csv_path.name, fontsize=12)

    # Jaccard + random reference (closed-form E[J]; optional MC band if random_trials >= 2)
    ax = axes[0, 0]
    if has_valid(jaccard):
        jac_plot = _series_for_log_scale(jaccard, log_y_floor) if y_scale == "log" else jaccard
        eb_plot = (
            _series_for_log_scale(expected_baseline, log_y_floor) if y_scale == "log" else expected_baseline
        )
        mc_m_plot = (
            _series_for_log_scale(mc_mean, log_y_floor) if y_scale == "log" else mc_mean
        )
        mc_lo_plot = (
            _series_for_log_scale(mc_lo, log_y_floor) if y_scale == "log" else mc_lo
        )
        mc_hi_plot = (
            _series_for_log_scale(mc_hi, log_y_floor) if y_scale == "log" else mc_hi
        )
        ax.plot(x, jac_plot, marker="o", linewidth=1.3, markersize=3, label="observed Jaccard", color="tab:blue")
        plot_mask_overlap_reference(
            ax, x, eb_plot, mc_m_plot, mc_lo_plot, mc_hi_plot, random_trials
        )
        if y_scale == "log":
            _apply_metric_y_scale(ax, y_scale, ymin_floor=log_y_floor)
        else:
            ax.set_ylim(
                0,
                max(
                    1.0,
                    max((v for v in jaccard if not math.isnan(v)), default=1.0) * 1.05,
                ),
            )
        ax.legend(fontsize=7, loc="best")
        ax.set_title("Per-layer Jaccard vs random (Bernoulli) null")
    else:
        ax.text(0.5, 0.5, "No Jaccard data", ha="center", va="center")
        ax.set_title("Per-layer Jaccard")
    ax.grid(alpha=0.3)

    # CKA: observed activations similarity + theoretical CKA null (not mask-overlap Jaccard)
    ax = axes[0, 1]
    if has_valid(cka):
        cka_plot = _series_for_log_scale(cka, log_y_floor) if y_scale == "log" else cka
        ax.plot(x, cka_plot, marker="o", linewidth=1.3, markersize=3, color="tab:purple", label="linear CKA (activations)")
        if cka_null_value is not None and not math.isnan(cka_null_value):
            if cka_null_value <= 0:
                y_cka_null = float("nan")
            else:
                y_cka_null = max(cka_null_value * cka_null_scale, log_y_floor * 0.1)
            mode_lbl = "fixed"
        else:
            raw = expected_linear_cka_null_reference(cka_n_samples, cka_null_mode)
            y_cka_null = max(float(raw) * float(cka_null_scale), log_y_floor * 0.1)
            mode_lbl = cka_null_mode
        if not math.isnan(y_cka_null) and y_cka_null > 0:
            ax.axhline(
                y=y_cka_null,
                color="forestgreen",
                linestyle=":",
                linewidth=1.35,
                alpha=0.95,
                zorder=1,
                label=f"CKA null ref (~{y_cka_null:.3e}, {mode_lbl}; n={cka_n_samples})",
            )
        if y_scale == "log":
            _apply_metric_y_scale(ax, y_scale, ymin_floor=log_y_floor)
        else:
            ax.set_ylim(0, 1)
        ax.legend(fontsize=6, loc="best")
        ax.set_title(
            "Per-layer linear CKA (mask_to_cka) vs CKA null ref\n"
            "(green ≈ vanishing unrelated-activation floor; not Jaccard overlap)"
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


def plot_compare(
    csv_a: Path,
    csv_b: Path,
    label_a: str,
    label_b: str,
    out_path: Path,
    *,
    y_scale: str = "linear",
    cka_n_samples: int = 64,
    cka_null_mode: str = "inv_sq",
    cka_null_scale: float = 1.0,
    log_y_floor: float = 1e-12,
):
    """4-panel comparison figure: two CSVs (e.g. SNIP vs CAV) overlaid on the same axes.

    Panel layout:
      [0,0] Per-layer Jaccard        — label_a line, label_b line, E[Jaccard] null
      [0,1] Per-layer CKA            — label_a, label_b, CKA null ref (not Jaccard)
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
    ja_p = _series_for_log_scale(jac_a, log_y_floor) if y_scale == "log" else jac_a
    jb_p = _series_for_log_scale(jac_b, log_y_floor) if y_scale == "log" else jac_b
    rb_p = _series_for_log_scale(rand_bl, log_y_floor) if y_scale == "log" else rand_bl
    if has_valid(jac_a):
        ax.plot(x, ja_p, color=CA_G, label=f"{label_a}", **kw)
    if has_valid(jac_b):
        ax.plot(x, jb_p, color=CB_G, label=f"{label_b}", **kw)
    if has_valid(rand_bl):
        ax.plot(
            x,
            rb_p,
            linestyle="--",
            linewidth=1.0,
            color="gray",
            alpha=0.6,
            label="E[Jaccard] null (indep. Bernoulli)",
        )
    if y_scale == "log":
        _apply_metric_y_scale(ax, y_scale, ymin_floor=log_y_floor)
    else:
        ax.set_ylim(0, 1.05)
    ax.set_title("Per-layer Jaccard  (GRPO ∩ DPO mask overlap)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panel 0,1 : CKA ───────────────────────────────────────────────
    ax = axes[0, 1]
    cka_a_p = _series_for_log_scale(cka_a, log_y_floor) if y_scale == "log" else cka_a
    cka_b_p = _series_for_log_scale(cka_b, log_y_floor) if y_scale == "log" else cka_b
    if has_valid(cka_a):
        ax.plot(x, cka_a_p, color=CA_G, label=f"{label_a}", **kw)
    else:
        ax.text(0.5, 0.5, f"No CKA data for {label_a}", ha="center", va="center", fontsize=9)
    if has_valid(cka_b):
        ax.plot(x, cka_b_p, color=CB_G, label=f"{label_b}", **kw)
    else:
        ax.text(0.5, 0.45, f"No CKA data for {label_b}", ha="center", va="center", fontsize=9)
    y_cka_null = max(
        float(cka_null_scale) * expected_linear_cka_null_reference(cka_n_samples, cka_null_mode),
        log_y_floor * 0.1,
    )
    if not math.isnan(y_cka_null) and y_cka_null > 0:
        ax.axhline(
            y=y_cka_null,
            color="forestgreen",
            linestyle=":",
            linewidth=1.25,
            alpha=0.95,
            label=f"CKA null ref (~{y_cka_null:.3e}; n={cka_n_samples})",
        )
    if y_scale == "log":
        _apply_metric_y_scale(ax, y_scale, ymin_floor=log_y_floor)
    else:
        ax.set_ylim(0, 1.05)
    ax.set_title("Per-layer CKA  (activations; green = vanishing null, not Jaccard)")
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
        help=(
            "Monte Carlo trials per layer for optional Jaccard null band (min–max shading). "
            "Use 0 or 1 to skip MC and plot only the closed-form E[Jaccard] (indep. Bernoulli) curve."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=42, help="RNG seed for Monte Carlo baselines.")
    parser.add_argument(
        "--y-scale",
        type=str,
        default="linear",
        choices=["linear", "log"],
        help="Y-axis for Jaccard + CKA panels: linear [0,1] or log (clamps zeros for display).",
    )
    parser.add_argument(
        "--log-y-floor",
        type=float,
        default=1e-12,
        help="Minimum positive value when using --y-scale log (avoids log(0)).",
    )
    parser.add_argument(
        "--cka-n-samples",
        type=int,
        default=64,
        help="Calibration batch size n used for CKA null reference (match mask_to_cka --n_samples).",
    )
    parser.add_argument(
        "--cka-null-mode",
        type=str,
        default="inv_sq",
        choices=["inv_sq", "inv_nm1"],
        help="CKA null reference: 1/n**2 (vanishing) or 1/(n-1) (looser).",
    )
    parser.add_argument(
        "--cka-null-scale",
        type=float,
        default=1.0,
        help="Multiplies the CKA null reference line.",
    )
    parser.add_argument(
        "--cka-null-value",
        type=float,
        default=None,
        help="If set, horizontal CKA null at this value (overrides --cka-null-mode).",
    )

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
        plot_compare(
            ca,
            cb,
            args.label_a,
            args.label_b,
            out,
            y_scale=args.y_scale,
            cka_n_samples=args.cka_n_samples,
            cka_null_mode=args.cka_null_mode,
            cka_null_scale=args.cka_null_scale,
            log_y_floor=args.log_y_floor,
        )
        return

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use rglob for --recursive: pathlib glob("**/pat") is filesystem-dependent for matches
    # at the root of input_dir; rglob always includes the directory itself.
    if args.recursive:
        csv_files = sorted(input_dir.rglob(args.pattern))
    else:
        csv_files = sorted(input_dir.glob(args.pattern))

    # Ignore summary CSV artifacts.
    csv_files = [p for p in csv_files if not p.name.endswith("_summary.csv")]

    print(
        f"plot_layer_metrics_csv: input_dir={input_dir}  pattern={args.pattern!r}  "
        f"recursive={args.recursive}  matched_csv={len(csv_files)}",
        flush=True,
    )

    made = 0
    for p in csv_files:
        out = output_dir / f"{p.stem}_plots.png"
        if plot_one(
            p,
            out,
            args.random_trials,
            args.random_seed,
            y_scale=args.y_scale,
            cka_n_samples=args.cka_n_samples,
            cka_null_mode=args.cka_null_mode,
            cka_null_scale=args.cka_null_scale,
            cka_null_value=args.cka_null_value,
            log_y_floor=args.log_y_floor,
        ):
            made += 1
            print(f"✓ {out}")

    print(f"Generated plot files: {made}", flush=True)
    if len(csv_files) > 0 and made == 0:
        print(
            "ERROR: matched one or more layer_metrics CSVs but wrote zero PNGs "
            "(empty CSVs or plot_one failed silently).",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(csv_files) == 0:
        print(
            "WARNING: no layer_metrics CSVs matched; no PNGs written.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
