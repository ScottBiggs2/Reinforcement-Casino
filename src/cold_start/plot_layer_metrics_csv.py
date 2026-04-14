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


def theoretical_bernoulli_jaccard_mean(density_a: float, density_b: float) -> float:
    """E[Jaccard] for independent Bernoulli masks with keep densities ρ_a, ρ_b (fraction of ones).

    E[J] = ρ_a ρ_b / (ρ_a + ρ_b − ρ_a ρ_b). If ρ_a = ρ_b = ρ, this equals ρ / (2 − ρ).
    """
    denom = density_a + density_b - density_a * density_b
    if denom <= 0:
        return 0.0
    return (density_a * density_b) / denom


def theoretical_jaccard_variance(rho: float, n_indices: int) -> float:
    """Var(J) = 2(1 − ρ) / ( N · (2 − ρ²)³ ) with ρ = keep density, N = tensor length."""
    if n_indices < 1 or not (0.0 <= rho <= 1.0):
        return float("nan")
    den = (2.0 - rho * rho) ** 3
    if den <= 0:
        return float("nan")
    return 2.0 * (1.0 - rho) / (float(n_indices) * den)


def theoretical_cka_mean_and_std(total_indices: int) -> tuple:
    """Reference null: E[CKA] = 1/(N−1), Var = 2/(N−1)² (user-specified large-N limit)."""
    n = int(total_indices)
    if n < 2:
        return float("nan"), float("nan")
    nm1 = float(n - 1)
    e = 1.0 / nm1
    var = 2.0 / (nm1 * nm1)
    return e, math.sqrt(var)


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


def draw_jaccard_theory_reference(
    ax,
    x,
    e_mean: list,
    e_lo: list,
    e_hi: list,
) -> None:
    """Draw theoretical E[Jaccard] and ±1σ band from closed-form mean and variance (no MC)."""
    if has_valid(e_mean):
        ax.plot(
            x,
            e_mean,
            linestyle=":",
            linewidth=1.25,
            color="gray",
            alpha=0.95,
            label="E[J] null (indep. Bernoulli, theory)",
        )
    if (
        has_valid(e_lo)
        and has_valid(e_hi)
        and has_valid(e_mean)
    ):
        ax.fill_between(
            x,
            e_lo,
            e_hi,
            alpha=0.22,
            color="tab:orange",
            label="Jaccard null: E ± 1σ (theory)",
        )


def plot_one(
    csv_path: Path,
    out_path: Path,
    random_trials: int,
    random_seed: int,
    *,
    y_scale: str = "linear",
    cka_total_n: Optional[int] = None,
    cka_null_scale: float = 1.0,
    cka_null_value: Optional[float] = None,
    log_y_floor: float = 1e-12,
):
    """random_trials / random_seed are ignored (kept for CLI compatibility)."""
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

    da_list: list = []
    db_list: list = []
    for r in rows:
        da = to_float(r.get("density_a"))
        db = to_float(r.get("density_b"))
        if math.isnan(da):
            sa = to_float(r.get("sparsity_a"))
            da = 1.0 - sa if not math.isnan(sa) else float("nan")
        if math.isnan(db):
            sb = to_float(r.get("sparsity_b"))
            db = 1.0 - sb if not math.isnan(sb) else float("nan")
        da_list.append(da)
        db_list.append(db)

    # Theory: E[J], Var(J) with ρ = keep density; band E ± 1σ (σ = sqrt(Var)).
    e_mean: list = []
    e_lo: list = []
    e_hi: list = []
    for i in range(len(rows)):
        da, db = da_list[i], db_list[i]
        n_p = n_params_col[i] or 0
        if math.isnan(da) or math.isnan(db) or n_p < 1:
            e_mean.append(float("nan"))
            e_lo.append(float("nan"))
            e_hi.append(float("nan"))
            continue
        em = theoretical_bernoulli_jaccard_mean(da, db)
        rho_m = 0.5 * (da + db)
        var = theoretical_jaccard_variance(rho_m, n_p)
        sig = math.sqrt(max(0.0, var)) if not math.isnan(var) else 0.0
        e_mean.append(em)
        e_lo.append(max(0.0, em - sig))
        e_hi.append(min(1.0, em + sig))

    n_tot = int(cka_total_n) if cka_total_n is not None and cka_total_n > 0 else sum(
        n for n in n_params_col if n is not None and n > 0
    )
    if cka_null_value is not None and not math.isnan(cka_null_value) and cka_null_value > 0:
        cka_e = float(cka_null_value) * float(cka_null_scale)
        cka_sig = 0.0
    else:
        cka_e, cka_sig = theoretical_cka_mean_and_std(n_tot)
        cka_e *= float(cka_null_scale)
        cka_sig *= float(cka_null_scale)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(csv_path.name, fontsize=12)

    # Jaccard + theoretical null (E and ±1σ)
    ax = axes[0, 0]
    if has_valid(jaccard):
        jac_plot = _series_for_log_scale(jaccard, log_y_floor) if y_scale == "log" else jaccard
        eb_plot = _series_for_log_scale(e_mean, log_y_floor) if y_scale == "log" else e_mean
        lo_plot = _series_for_log_scale(e_lo, log_y_floor) if y_scale == "log" else e_lo
        hi_plot = _series_for_log_scale(e_hi, log_y_floor) if y_scale == "log" else e_hi
        ax.plot(x, jac_plot, marker="o", linewidth=1.3, markersize=3, label="observed Jaccard", color="tab:blue")
        draw_jaccard_theory_reference(ax, x, eb_plot, lo_plot, hi_plot)
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
        ax.set_title("Per-layer Jaccard vs Bernoulli null (theory)")
    else:
        ax.text(0.5, 0.5, "No Jaccard data", ha="center", va="center")
        ax.set_title("Per-layer Jaccard")
    ax.grid(alpha=0.3)

    # CKA: activations + theory null E=1/(N−1) on total index count N
    ax = axes[0, 1]
    show_cka_obs = has_valid(cka)
    if show_cka_obs:
        cka_plot = _series_for_log_scale(cka, log_y_floor) if y_scale == "log" else cka
        ax.plot(
            x,
            cka_plot,
            marker="o",
            linewidth=1.3,
            markersize=3,
            color="tab:purple",
            label="linear CKA (activations)",
        )
    if n_tot >= 2 and not math.isnan(cka_e) and cka_e > 0:
        y0 = max(cka_e, log_y_floor * 0.1)
        ax.axhline(
            y=y0,
            color="forestgreen",
            linestyle=":",
            linewidth=1.35,
            alpha=0.95,
            zorder=1,
            label=f"E[CKA] null = 1/(N−1) ≈ {y0:.3e} (N={n_tot})",
        )
        if cka_sig > 0:
            y_lo = max(y0 - cka_sig, log_y_floor * 0.1)
            y_hi = y0 + cka_sig
            ax.axhspan(y_lo, y_hi, color="forestgreen", alpha=0.12, label="CKA null ±1σ (theory)")
    drew_cka_theory = n_tot >= 2 and not math.isnan(cka_e) and cka_e > 0
    if not show_cka_obs and not drew_cka_theory:
        ax.text(0.5, 0.5, "No CKA in CSV (re-export with CKA JSON)", ha="center", va="center", fontsize=9)
    if y_scale == "log":
        _apply_metric_y_scale(ax, y_scale, ymin_floor=log_y_floor)
    else:
        ax.set_ylim(0, 1)
    ax.legend(fontsize=6, loc="best")
    ax.set_title(
        "Per-layer linear CKA vs theory null\n"
        f"(E=1/(N−1), σ²=2/(N−1)²; N=total masked indices ≈ {n_tot})"
    )
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
    cka_total_n: Optional[int] = None,
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

    # Theoretical E[J] for label_a rows (same as plot_one)
    rand_bl = []
    for ra in pairs_a:
        da = to_float(ra.get("density_a"))
        db = to_float(ra.get("density_b"))
        if math.isnan(da):
            sa = to_float(ra.get("sparsity_a"))
            da = 1.0 - sa if not math.isnan(sa) else float("nan")
        if math.isnan(db):
            sb = to_float(ra.get("sparsity_b"))
            db = 1.0 - sb if not math.isnan(sb) else float("nan")
        if math.isnan(da) or math.isnan(db):
            rand_bl.append(float("nan"))
        else:
            rand_bl.append(theoretical_bernoulli_jaccard_mean(da, db))

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
    n_tot = int(cka_total_n) if cka_total_n is not None and cka_total_n > 0 else sum(
        to_int_params(r.get("n_params")) or 0 for r in pairs_a
    )
    ce, _sig_cka = theoretical_cka_mean_and_std(n_tot)
    ce *= float(cka_null_scale)
    y_cka_null = max(ce, log_y_floor * 0.1) if n_tot >= 2 and not math.isnan(ce) else float("nan")
    if not math.isnan(y_cka_null) and y_cka_null > 0:
        ax.axhline(
            y=y_cka_null,
            color="forestgreen",
            linestyle=":",
            linewidth=1.25,
            alpha=0.95,
            label=f"E[CKA] null ≈ {y_cka_null:.3e} (N={n_tot})",
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
        default=1,
        help="Ignored (kept for backward compatibility). Jaccard null uses closed-form E and Var only.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Ignored (kept for backward compatibility).",
    )
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
        "--cka-total-n",
        type=int,
        default=None,
        help=(
            "Total index count N for CKA theory null E=1/(N−1), σ²=2/(N−1)². "
            "Default: sum of n_params in the CSV (total masked parameters)."
        ),
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
        help="If set, overrides theoretical E[CKA] null (still scaled by --cka-null-scale).",
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
            cka_total_n=args.cka_total_n,
            cka_null_scale=float(args.cka_null_scale),
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
            cka_total_n=args.cka_total_n,
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
