#!/usr/bin/env python3
"""
Visualize outputs from src/analysis/mask_score_gap_analysis.py.

Reads mask_score_gap_histograms.npz and mask_score_gap_summary.csv under --analysis-dir.
Writes PNG and PDF figures (histograms + ECDFs) for magnitude vs oracle and random vs oracle gaps.

CPU-only; suitable for a small Slurm job after the analysis run::

    python scripts/report_mask_score_gap_plots.py --analysis-dir /path/to/out_dir

Interpretation (what you are looking at)
----------------------------------------

Each *gap* is v = |s_other − s_oracle| at every weight element: how far that method's score
differs from the oracle score at the same index. Oracle = |w_500 − w_base|; magnitude = sum of
|Δw| over snapshot steps ≤ T; random = Uniform(0,1).

**Histogram (top panels):** Estimated density of gaps. Where the curve is high, many weights have
gaps in that range. For raw gaps, x is in weight space (large dynamic range → often log-scaled x).
For *normalized* gaps, each parameter tensor's oracle and other scores are min–max scaled to [0,1]
before differencing, so v_norm is in [0,1].

**ECDF (bottom panels):** Empirical cumulative distribution function. The y-axis is the fraction of
all weight elements whose gap is **at or below** the x value. Example: at x = 1e−6, if ECDF = 0.4,
then 40% of gaps are ≤ 1e−6. A curve that rises steeply on the left means most gaps are tiny; a
flat line near y = 1 on the right means almost everything is below that x.

**Files produced:**
- ``magnitude_raw.*`` — raw |magnitude_score − oracle_score|; log-scaled x common.
- ``magnitude_norm.*`` — per-tensor min–max normalized gaps; linear x in [0, zoomed].
- ``random_raw_ecdf_multi_seed.*`` — raw random vs oracle ECDFs (log x).
- ``random_norm_ecdf_multi_seed.*`` — normalized random vs oracle ECDFs (linear x, zoomed).
- ``summary_mean_bar.*`` — global mean gap by case row from the summary CSV.

Axis zoom: for normalized (and similar) plots, the full [0,1] range is often empty except near 0.
By default we set xmax to the bin edge that first reaches ``--cdf-mass`` of the probability mass
(plus a little padding), so the figure zooms into where the data actually live.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _hist_density(counts: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    total = float(counts.sum())
    if total <= 0:
        return np.array([]), np.array([])
    widths = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    density = counts.astype(np.float64) / total / np.maximum(widths, 1e-30)
    return centers, density


def _ecdf(counts: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    total = float(counts.sum())
    if total <= 0:
        return np.array([]), np.array([])
    centers = (edges[:-1] + edges[1:]) / 2.0
    cdf = np.cumsum(counts.astype(np.float64)) / total
    return centers, cdf


def _tight_linear_xlim(
    counts: np.ndarray,
    edges: np.ndarray,
    *,
    cdf_mass: float,
    xmin_floor: float = 0.0,
    xmax_cap: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Upper x-limit at the right edge of the bin where cumulative mass first reaches cdf_mass,
    with small padding. Clamps to actual span of nonempty bins so near-degenerate data still shows.
    """
    total = float(counts.sum())
    if total <= 0:
        return xmin_floor, min(xmax_cap or 1.0, 1.0)
    cum = np.cumsum(counts.astype(np.float64)) / total
    j = int(np.searchsorted(cum, cdf_mass))
    j = min(max(j, 0), len(counts) - 1)
    xmax = float(edges[j + 1])
    nz = np.flatnonzero(counts > 0)
    if nz.size:
        xmax = max(xmax, float(edges[nz[-1] + 1]))
    if xmax_cap is not None:
        xmax = min(xmax, xmax_cap)
    pad = max(xmax * 0.08, 1e-18)
    return xmin_floor, xmax + pad


def _tight_log_xlim_lin(
    counts: np.ndarray,
    log_edges: np.ndarray,
    *,
    cdf_mass: float,
) -> Tuple[float, float]:
    """Return linear-space xlim for log-binned histograms (edges are log10)."""
    total = float(counts.sum())
    if total <= 0:
        return 1e-30, 1.0
    cum = np.cumsum(counts.astype(np.float64)) / total
    j = int(np.searchsorted(cum, cdf_mass))
    j = min(max(j, 0), len(counts) - 1)
    xmax_lin = float(10 ** log_edges[j + 1])
    nz = np.flatnonzero(counts > 0)
    xmin_lin = float(10 ** log_edges[nz[0]]) if nz.size else float(10 ** log_edges[0])
    xmax_lin = max(xmax_lin, float(10 ** log_edges[nz[-1] + 1]) if nz.size else xmax_lin)
    pad_hi = xmax_lin * 0.08
    pad_lo = xmin_lin * 0.85
    return max(pad_lo, 1e-30), xmax_lin + pad_hi


def _plot_lin_hist(
    ax,
    counts: np.ndarray,
    edges: np.ndarray,
    title: str,
    xlabel: str,
    *,
    xmax_override: Optional[float] = None,
) -> None:
    c, d = _hist_density(counts, edges)
    if c.size == 0:
        ax.set_title(title + " (empty)")
        return
    ax.bar(c, d, width=np.diff(edges), align="center", alpha=0.85, edgecolor="none")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    if xmax_override is not None:
        ax.set_xlim(0.0, xmax_override)


def _plot_log_hist(
    ax,
    counts: np.ndarray,
    log_edges: np.ndarray,
    title: str,
    *,
    xlim_lin: Optional[Tuple[float, float]] = None,
) -> None:
    """log_edges are log10(bin bounds); x-axis shown in linear space."""
    total = float(counts.sum())
    if total <= 0:
        ax.set_title(title + " (empty)")
        return
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2.0
    lin_centers = 10**log_centers
    widths_log = np.diff(log_edges)
    widths_lin = (10 ** (log_centers + widths_log / 2) - 10 ** (log_centers - widths_log / 2)).clip(
        min=1e-30
    )
    density = counts.astype(np.float64) / total / widths_lin
    ax.bar(lin_centers, density, width=widths_lin, align="center", alpha=0.85, edgecolor="none")
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("gap (linear scale, log-spaced bins)")
    ax.set_ylabel("density")
    if xlim_lin is not None:
        ax.set_xlim(xlim_lin[0], xlim_lin[1])


def _plot_ecdf_lin(
    ax,
    counts: np.ndarray,
    edges: np.ndarray,
    title: str,
    xlabel: str,
    *,
    xmax_override: Optional[float] = None,
) -> None:
    x, y = _ecdf(counts, edges)
    if x.size == 0:
        ax.set_title(title + " (empty)")
        return
    ax.step(x, y, where="mid")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of weights with gap ≤ x")
    if xmax_override is not None:
        ax.set_xlim(0.0, xmax_override)


def _plot_ecdf_log(
    ax,
    counts: np.ndarray,
    log_edges: np.ndarray,
    title: str,
    *,
    xlim_lin: Optional[Tuple[float, float]] = None,
) -> None:
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2.0
    x = 10**log_centers
    _, y = _ecdf(counts, log_edges)
    if x.size == 0:
        ax.set_title(title + " (empty)")
        return
    ax.step(x, y, where="mid")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("gap (linear scale)")
    ax.set_ylabel("fraction of weights with gap ≤ x")
    if xlim_lin is not None:
        ax.set_xlim(xlim_lin[0], xlim_lin[1])


def load_summary_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot mask score gap histograms/ECDFs from analysis output.")
    ap.add_argument("--analysis-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=None, help="Figure output dir (default: analysis-dir/figures)")
    ap.add_argument(
        "--cdf-mass",
        type=float,
        default=0.9995,
        help="For tight linear x-axes: xmax is set where cumulative mass reaches this fraction (default 0.9995).",
    )
    ap.add_argument(
        "--norm-xmax",
        type=float,
        default=None,
        help="Force max x for normalized-gap plots (overrides automatic tight limit).",
    )
    ap.add_argument(
        "--raw-xmax",
        type=float,
        default=None,
        help="Force max x (linear) for raw-gap log-scale plots (overrides automatic tight limit).",
    )
    args = ap.parse_args()

    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = analysis_dir / "mask_score_gap_histograms.npz"
    summary_path = analysis_dir / "mask_score_gap_summary.csv"
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"{npz_path}\n"
            "  Expected directory with mask_score_gap_histograms.npz from mask_score_gap_analysis. "
            "Example: --analysis-dir ~/rl_casino/results/mask_score_gap_<jobid> "
            "or scratch .../rl_casino_analysis/mask_score_gap_light_r1/<jobid>. "
            "Use the full path (unset OUT_DIR is empty; /results/ is not your home results folder)."
        )

    data = np.load(npz_path)
    cm = float(np.clip(args.cdf_mass, 0.5, 1.0))

    # Magnitude raw (log-binned x)
    counts_mr = data["magnitude_raw_counts"]
    edges_mr = data["magnitude_raw_log_edges"]
    xlim_raw = None
    if args.raw_xmax is not None:
        xlim_raw = (1e-30, float(args.raw_xmax))
    else:
        lo, hi = _tight_log_xlim_lin(counts_mr, edges_mr, cdf_mass=cm)
        xlim_raw = (lo, hi)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    _plot_log_hist(axes[0], counts_mr, edges_mr, "|mag − oracle| raw", xlim_lin=xlim_raw)
    _plot_ecdf_log(axes[1], counts_mr, edges_mr, "ECDF: raw magnitude gap vs oracle", xlim_lin=xlim_raw)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"magnitude_raw.{ext}", dpi=150)
    plt.close(fig)

    counts_mn = data["magnitude_norm_counts"]
    edges_mn = data["magnitude_norm_edges"]
    _x0, xmax_mn = _tight_linear_xlim(counts_mn, edges_mn, cdf_mass=cm, xmax_cap=1.0)
    if args.norm_xmax is not None:
        xmax_mn = float(args.norm_xmax)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    _plot_lin_hist(
        axes[0],
        counts_mn,
        edges_mn,
        "|mag − oracle| (per-tensor min–max normalized scores)",
        "normalized gap",
        xmax_override=xmax_mn,
    )
    _plot_ecdf_lin(
        axes[1],
        counts_mn,
        edges_mn,
        "ECDF: normalized magnitude gap vs oracle",
        "normalized gap",
        xmax_override=xmax_mn,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"magnitude_norm.{ext}", dpi=150)
    plt.close(fig)

    seed_tags: List[str] = []
    for k in data.files:
        if k.startswith("random_raw_seed") and k.endswith("_counts"):
            seed_tags.append(k.replace("random_raw_seed", "").replace("_counts", ""))

    if seed_tags:
        xlim_rr: Optional[Tuple[float, float]] = None
        if args.raw_xmax is not None:
            xlim_rr = (1e-30, float(args.raw_xmax))
        else:
            lo_g, hi_g = 1e300, 0.0
            for tag in seed_tags:
                ckey = f"random_raw_seed{tag}_counts"
                ekey = f"random_raw_seed{tag}_log_edges"
                if ckey not in data.files or ekey not in data.files:
                    continue
                lo, hi = _tight_log_xlim_lin(data[ckey], data[ekey], cdf_mass=cm)
                lo_g = min(lo_g, lo)
                hi_g = max(hi_g, hi)
            if hi_g > 0:
                xlim_rr = (max(lo_g, 1e-30), hi_g)

        fig, ax = plt.subplots(figsize=(9, 5))
        for tag in sorted(seed_tags, key=lambda x: int(x) if x.isdigit() else 0):
            ckey = f"random_raw_seed{tag}_counts"
            ekey = f"random_raw_seed{tag}_log_edges"
            if ckey not in data.files or ekey not in data.files:
                continue
            xc_log, y = _ecdf(data[ckey], data[ekey])
            if xc_log.size == 0:
                continue
            x_lin = 10**xc_log
            ax.step(x_lin, y, where="mid", label=f"seed {tag}")
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title("ECDF: raw random vs oracle gap (by seed)")
        ax.set_xlabel("gap (linear scale)")
        ax.set_ylabel("fraction of weights with gap ≤ x")
        if xlim_rr is not None:
            ax.set_xlim(xlim_rr[0], xlim_rr[1])
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"random_raw_ecdf_multi_seed.{ext}", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        xmax_rn = 0.0
        for tag in sorted(seed_tags, key=lambda x: int(x) if x.isdigit() else 0):
            ckey = f"random_norm_seed{tag}_counts"
            ekey = f"random_norm_seed{tag}_edges"
            if ckey not in data.files or ekey not in data.files:
                continue
            x, y = _ecdf(data[ckey], data[ekey])
            if x.size == 0:
                continue
            ax.step(x, y, where="mid", label=f"seed {tag}")
            _, xi = _tight_linear_xlim(data[ckey], data[ekey], cdf_mass=cm, xmax_cap=1.0)
            xmax_rn = max(xmax_rn, xi)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title("ECDF: normalized random vs oracle gap (by seed)")
        ax.set_xlabel("normalized gap")
        ax.set_ylabel("fraction of weights with gap ≤ x")
        if args.norm_xmax is not None:
            ax.set_xlim(0.0, args.norm_xmax)
        else:
            ax.set_xlim(0.0, xmax_rn if xmax_rn > 0 else 1.0)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"random_norm_ecdf_multi_seed.{ext}", dpi=150)
        plt.close(fig)

    if summary_path.is_file():
        rows = load_summary_csv(summary_path)
        fig, ax = plt.subplots(figsize=(8, 4))
        cases = [r["case"] for r in rows]
        means = [float(r["mean"]) if r.get("mean") else float("nan") for r in rows]
        ax.barh(cases, means)
        ax.set_xlabel("mean gap")
        ax.set_title("Mean gap by case (from summary CSV)")
        m = np.nanmax(np.abs(means)) if means else 1.0
        ax.set_xlim(0, max(m * 1.15, 1e-18))
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"summary_mean_bar.{ext}", dpi=150)
        plt.close(fig)

    print(f"Wrote figures under {out_dir}")


if __name__ == "__main__":
    main()
