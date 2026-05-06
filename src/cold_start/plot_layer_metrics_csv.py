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


def finite_count(vals):
    return sum(1 for v in vals if not math.isnan(v))


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


def global_density_from_rows(rows: list, which: str) -> float:
    """Compute a single global keep-density across all layers.

    This is the constant Bernoulli baseline the user expects: a *single* density per mask
    (weighted by layer size), not per-layer densities that vary with layer sparsity.
    """
    key_d = f"density_{which}"
    key_s = f"sparsity_{which}"
    num = 0.0
    den = 0.0
    for r in rows:
        n_p = to_int_params(r.get("n_params")) or 0
        if n_p <= 0:
            continue
        d = to_float(r.get(key_d))
        if math.isnan(d):
            s = to_float(r.get(key_s))
            d = 1.0 - s if not math.isnan(s) else float("nan")
        if math.isnan(d):
            continue
        num += float(n_p) * float(d)
        den += float(n_p)
    if den <= 0:
        return float("nan")
    return num / den


def theoretical_jaccard_variance(rho: float, n_indices: int) -> float:
    """Var(J) for iid random binary masks with keep rate ρ over n indices.

    Matches the closed-form used in the writeup:

        E[J] = ρ / (2 − ρ)
        Var[J] = 2(1 − ρ) / ( n (2 − ρ)^3 )

    where n is the number of compared indices (e.g. total parameter count).
    """
    if n_indices < 1 or not (0.0 <= rho <= 1.0):
        return float("nan")
    den = (2.0 - rho) ** 3
    if den <= 0:
        return float("nan")
    return 2.0 * (1.0 - rho) / (float(n_indices) * den)


def mc_global_jaccard_quantiles(
    density_a: float,
    density_b: float,
    n_indices: int,
    trials: int,
    seed: int,
) -> Optional[tuple]:
    """Empirical 5th/95th percentile of global Jaccard under iid Bernoulli masks.

    IMPORTANT: This must be fast for very large N (e.g. billions of parameters). We
    therefore sample *counts* using a multinomial distribution instead of instantiating
    boolean masks of length N.

    For independent Bernoulli masks A~Bern(ρ_a), B~Bern(ρ_b), each index falls into
    one of 4 categories:

      (A=1,B=1) with p11 = ρ_a ρ_b     => contributes to intersection and union
      (A=1,B=0) with p10 = ρ_a(1-ρ_b)  => contributes to union
      (A=0,B=1) with p01 = (1-ρ_a)ρ_b  => contributes to union
      (A=0,B=0) with p00 = (1-ρ_a)(1-ρ_b)

    If we sample counts (c11,c10,c01,c00) ~ Multinomial(N, p), then:
        inter = c11
        union = c11 + c10 + c01
        J = inter/union  (union>0 else 0)
    """
    if trials <= 0 or n_indices < 1:
        return None
    if math.isnan(density_a) or math.isnan(density_b):
        return None
    rng = np.random.default_rng(seed)
    da = float(density_a)
    db = float(density_b)
    if not (0.0 <= da <= 1.0 and 0.0 <= db <= 1.0):
        return None
    # Probabilities for the 4 outcomes
    p11 = da * db
    p10 = da * (1.0 - db)
    p01 = (1.0 - da) * db
    p00 = (1.0 - da) * (1.0 - db)
    ps = np.array([p11, p10, p01, p00], dtype=np.float64)
    s = ps.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    ps = ps / s

    samples = np.empty(trials, dtype=np.float64)
    for t in range(trials):
        c11, c10, c01, _c00 = rng.multinomial(int(n_indices), ps)
        union = c11 + c10 + c01
        samples[t] = (c11 / union) if union > 0 else 0.0
    return float(np.percentile(samples, 5)), float(np.percentile(samples, 95))


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


def _jaccard_display_floor(log_y_floor: float, jaccard_log_floor: float) -> float:
    """Floor for Jaccard panel in log mode (default 1e-4 so tiny nulls stay visible)."""
    return max(float(log_y_floor), float(jaccard_log_floor))


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


def draw_jaccard_null_horizontal(
    ax,
    *,
    em_g: float,
    j_lo: float,
    j_hi: float,
    sig_g: float,
    n_tot: int,
    rho_m: float,
    y_scale: str,
    display_floor: float,
) -> None:
    """Horizontal E[J] and ±1σ band — same layout as CKA theory panel (axhline + axhspan).

    Uses global E[J] from size-weighted ρ_a, ρ_b and the iid-mask variance:

        Var[J] = 2(1 − ρ) / ( n (2 − ρ)^3 )

    where we plug in ρ = average keep density ρ̄ = (ρ_a+ρ_b)/2 and n = total indices.
    """
    if math.isnan(em_g) or n_tot < 1:
        return
    if y_scale == "log":
        em_d = _clamp_positive_log(em_g, display_floor)
        lo_d = _clamp_positive_log(j_lo, display_floor)
        hi_d = _clamp_positive_log(j_hi, display_floor)
        if lo_d > hi_d:
            lo_d, hi_d = hi_d, lo_d
    else:
        em_d, lo_d, hi_d = em_g, j_lo, j_hi

    if (
        sig_g > 0
        and not math.isnan(j_lo)
        and not math.isnan(j_hi)
        and (j_hi - j_lo) > 1e-15
    ):
        ax.axhspan(
            lo_d,
            hi_d,
            color="tab:orange",
            alpha=0.2,
            zorder=0,
            label="Jaccard null: E ± 1σ (theory)",
        )
    ax.axhline(
        y=em_d,
        color="gray",
        linestyle=":",
        linewidth=1.35,
        alpha=0.95,
        zorder=1,
        label=f"E[J] null ≈ {em_g:.3e} (N={n_tot}, ρ̄={rho_m:.4f})",
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
    jaccard_log_floor: float = 1e-4,
    jaccard_mc_trials: int = 0,
    jaccard_mc_seed: int = 42,
):
    """random_trials is deprecated; use jaccard_mc_trials for Monte Carlo null band."""
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

    print(
        f"[plot_layer_metrics] {csv_path.name}: rows={len(rows)} "
        f"finite_cka={finite_count(cka)} finite_er_a_norm={finite_count(erank_a)} "
        f"finite_er_b_norm={finite_count(erank_b)}",
        flush=True,
    )

    mask_a_name = rows[0].get("mask_a_name") or "mask_a"
    mask_b_name = rows[0].get("mask_b_name") or "mask_b"

    # Theory baseline: a *single* global Bernoulli baseline (constant across layers).
    # Compute global densities (size-weighted) so the baseline does not vary by layer index.
    da_g = global_density_from_rows(rows, "a")
    db_g = global_density_from_rows(rows, "b")

    em_g = (
        theoretical_bernoulli_jaccard_mean(da_g, db_g)
        if not math.isnan(da_g) and not math.isnan(db_g)
        else float("nan")
    )
    rho_m_g = 0.5 * (da_g + db_g) if not math.isnan(da_g) and not math.isnan(db_g) else float("nan")

    n_tot = int(cka_total_n) if cka_total_n is not None and cka_total_n > 0 else sum(
        n for n in n_params_col if n is not None and n > 0
    )
    # Horizontal null band: same closed-form Var as CKA-style total-N baseline (one σ for all layers).
    var_j = (
        theoretical_jaccard_variance(rho_m_g, n_tot)
        if not math.isnan(rho_m_g) and n_tot >= 1
        else float("nan")
    )
    sig_j = math.sqrt(max(0.0, var_j)) if not math.isnan(var_j) else 0.0
    j_lo = max(0.0, em_g - sig_j) if not math.isnan(em_g) else float("nan")
    j_hi = min(1.0, em_g + sig_j) if not math.isnan(em_g) else float("nan")

    jac_floor = _jaccard_display_floor(log_y_floor, jaccard_log_floor)
    if cka_null_value is not None and not math.isnan(cka_null_value) and cka_null_value > 0:
        cka_e = float(cka_null_value) * float(cka_null_scale)
        cka_sig = 0.0
    else:
        cka_e, cka_sig = theoretical_cka_mean_and_std(n_tot)
        cka_e *= float(cka_null_scale)
        cka_sig *= float(cka_null_scale)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(csv_path.name, fontsize=12)

    # Jaccard + theoretical null (horizontal E[J] and ±1σ, same style as CKA panel)
    ax = axes[0, 0]
    if has_valid(jaccard):
        jac_plot = _series_for_log_scale(jaccard, jac_floor) if y_scale == "log" else jaccard
        if (
            not math.isnan(em_g)
            and n_tot >= 1
            and not math.isnan(rho_m_g)
        ):
            draw_jaccard_null_horizontal(
                ax,
                em_g=em_g,
                j_lo=j_lo,
                j_hi=j_hi,
                sig_g=sig_j,
                n_tot=n_tot,
                rho_m=rho_m_g,
                y_scale=y_scale,
                display_floor=jac_floor,
            )
        mc_trials = max(int(jaccard_mc_trials), int(random_trials))
        if mc_trials > 0:
            mc = mc_global_jaccard_quantiles(
                da_g, db_g, n_tot, mc_trials, int(jaccard_mc_seed or random_seed)
            )
            if mc is not None:
                q_lo, q_hi = mc
                if y_scale == "log":
                    lo_d = _clamp_positive_log(q_lo, jac_floor)
                    hi_d = _clamp_positive_log(q_hi, jac_floor)
                    if lo_d > hi_d:
                        lo_d, hi_d = hi_d, lo_d
                else:
                    lo_d, hi_d = q_lo, q_hi
                ax.axhspan(
                    lo_d,
                    hi_d,
                    color="mediumpurple",
                    alpha=0.14,
                    zorder=0,
                    label="MC global Jaccard 5–95% (iid; N_tot)",
                )
        ax.plot(
            x,
            jac_plot,
            marker="o",
            linewidth=1.3,
            markersize=3,
            label="observed Jaccard",
            color="tab:blue",
            zorder=2,
        )
        if y_scale == "log":
            _apply_metric_y_scale(ax, y_scale, ymin_floor=jac_floor)
        else:
            ax.set_ylim(
                0,
                max(
                    1.0,
                    max((v for v in jaccard if not math.isnan(v)), default=1.0) * 1.05,
                ),
            )
        ax.legend(fontsize=7, loc="best")
        ax.set_title(
            "Per-layer Jaccard vs random baselines\n"
            f"(orange band = theory E[J]±1σ; purple = MC 5–95%; N_tot≈{n_tot})"
        )
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
    if not show_cka_obs:
        ax.text(
            0.5,
            0.5,
            "No observed CKA values in CSV\n(re-export with --cka-json from mask_to_cka output)",
            ha="center",
            va="center",
            fontsize=9,
        )
    if y_scale == "log":
        _apply_metric_y_scale(ax, y_scale, ymin_floor=log_y_floor)
    else:
        ax.set_ylim(0, 1)
    ax.legend(fontsize=6, loc="best")
    ax.set_title(
        "Per-layer linear CKA vs theory null (green; unrelated reps)\n"
        f"(E=1/(N−1), σ²=2/(N−1)²; N=total masked indices ≈ {n_tot} — not a Jaccard baseline)"
    )
    ax.grid(alpha=0.3)

    # Sparsity A/B
    ax = axes[1, 0]
    drawn = False
    if has_valid(sparsity_a):
        ax.plot(x, sparsity_a, marker="o", linewidth=1.3, markersize=3, label=f"sparsity: {mask_a_name}")
        drawn = True
    if has_valid(sparsity_b):
        ax.plot(x, sparsity_b, marker="o", linewidth=1.3, markersize=3, label=f"sparsity: {mask_b_name}")
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
        ax.plot(x, erank_a, marker="o", linewidth=1.3, markersize=3, label=f"erank_norm: {mask_a_name}")
        drawn = True
    if has_valid(erank_b):
        ax.plot(x, erank_b, marker="o", linewidth=1.3, markersize=3, label=f"erank_norm: {mask_b_name}")
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
    jaccard_log_floor: float = 1e-4,
    jaccard_mc_trials: int = 0,
    jaccard_mc_seed: int = 42,
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

    # Theoretical E[J] as a constant global Bernoulli baseline (same across layers).
    da_g = global_density_from_rows(list(pairs_a), "a")
    db_g = global_density_from_rows(list(pairs_a), "b")
    em_g = (
        theoretical_bernoulli_jaccard_mean(da_g, db_g)
        if not math.isnan(da_g) and not math.isnan(db_g)
        else float("nan")
    )
    rho_m_g = 0.5 * (da_g + db_g) if not math.isnan(da_g) and not math.isnan(db_g) else float("nan")
    n_tot_j = (
        int(cka_total_n)
        if cka_total_n is not None and cka_total_n > 0
        else sum(to_int_params(r.get("n_params")) or 0 for r in pairs_a)
    )
    var_j = (
        theoretical_jaccard_variance(rho_m_g, n_tot_j)
        if not math.isnan(rho_m_g) and n_tot_j >= 1
        else float("nan")
    )
    sig_j = math.sqrt(max(0.0, var_j)) if not math.isnan(var_j) else 0.0
    j_lo = max(0.0, em_g - sig_j) if not math.isnan(em_g) else float("nan")
    j_hi = min(1.0, em_g + sig_j) if not math.isnan(em_g) else float("nan")
    jac_floor = _jaccard_display_floor(log_y_floor, jaccard_log_floor)

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
    ja_p = _series_for_log_scale(jac_a, jac_floor) if y_scale == "log" else jac_a
    jb_p = _series_for_log_scale(jac_b, jac_floor) if y_scale == "log" else jac_b
    if (
        not math.isnan(em_g)
        and n_tot_j >= 1
        and not math.isnan(rho_m_g)
    ):
        draw_jaccard_null_horizontal(
            ax,
            em_g=em_g,
            j_lo=j_lo,
            j_hi=j_hi,
            sig_g=sig_j,
            n_tot=n_tot_j,
            rho_m=rho_m_g,
            y_scale=y_scale,
            display_floor=jac_floor,
        )
    if jaccard_mc_trials > 0:
        mc = mc_global_jaccard_quantiles(
            da_g, db_g, n_tot_j, jaccard_mc_trials, jaccard_mc_seed
        )
        if mc is not None:
            q_lo, q_hi = mc
            if y_scale == "log":
                lo_d = _clamp_positive_log(q_lo, jac_floor)
                hi_d = _clamp_positive_log(q_hi, jac_floor)
                if lo_d > hi_d:
                    lo_d, hi_d = hi_d, lo_d
            else:
                lo_d, hi_d = q_lo, q_hi
            ax.axhspan(
                lo_d,
                hi_d,
                color="mediumpurple",
                alpha=0.14,
                zorder=0,
                label="MC global Jaccard 5–95% (iid; N_tot)",
            )
    if has_valid(jac_a):
        ax.plot(x, ja_p, color=CA_G, label=f"{label_a}", **kw, zorder=2)
    if has_valid(jac_b):
        ax.plot(x, jb_p, color=CB_G, label=f"{label_b}", **kw, zorder=2)
    if y_scale == "log":
        _apply_metric_y_scale(ax, y_scale, ymin_floor=jac_floor)
    else:
        ax.set_ylim(0, 1.05)
    ax.set_title(
        "Per-layer Jaccard  (GRPO ∩ DPO mask overlap)\n"
        f"(theory null: horizontal E±σ, N≈{n_tot_j})"
    )
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
        default=0,
        help="Deprecated alias: if >0 and --jaccard-mc-trials is 0, used as jaccard_mc_trials.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Deprecated: use --jaccard-mc-seed (fallback seed for MC band).",
    )
    parser.add_argument(
        "--jaccard-mc-trials",
        type=int,
        default=0,
        help="If >0, draw this many iid global-mask Jaccard samples (purple 5–95%% band).",
    )
    parser.add_argument(
        "--jaccard-mc-seed",
        type=int,
        default=42,
        help="RNG seed for --jaccard-mc-trials.",
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
        "--jaccard-log-floor",
        type=float,
        default=1e-4,
        help="Minimum y for Jaccard panel in log mode (floors observed + null; default 1e-4).",
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
        mc_t = int(args.jaccard_mc_trials) or int(args.random_trials)
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
            jaccard_log_floor=float(args.jaccard_log_floor),
            jaccard_mc_trials=mc_t,
            jaccard_mc_seed=int(args.jaccard_mc_seed),
        )
        return

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Discover CSVs: when --recursive, union glob + rglob so root-level files are always
    # included (some Python/pathlib versions are finicky about rglob-only discovery on
    # certain filesystems; flat `comparisons_vs_ground_truth/layer_metrics_*.csv` must match).
    if args.recursive:
        csv_files = sorted({*input_dir.glob(args.pattern), *input_dir.rglob(args.pattern)})
    else:
        csv_files = sorted(input_dir.glob(args.pattern))

    # Ignore summary CSV artifacts.
    csv_files = [p for p in csv_files if not p.name.endswith("_summary.csv")]

    print(
        f"plot_layer_metrics_csv: input_dir={input_dir}  pattern={args.pattern!r}  "
        f"recursive={args.recursive}  matched_csv={len(csv_files)}",
        flush=True,
    )
    for p in csv_files[:12]:
        print(f"  csv: {p}", flush=True)
    if len(csv_files) > 12:
        print(f"  ... and {len(csv_files) - 12} more", flush=True)

    mc_trials = int(args.jaccard_mc_trials) or int(args.random_trials)
    mc_seed = int(args.jaccard_mc_seed)
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
            jaccard_log_floor=float(args.jaccard_log_floor),
            jaccard_mc_trials=mc_trials,
            jaccard_mc_seed=mc_seed,
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
            "ERROR: no layer_metrics CSVs matched; no PNGs written. "
            f"Check --input-dir and --pattern (input_dir={input_dir!s}).",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
