#!/usr/bin/env python3
"""
Visualize outputs from src/analysis/mask_score_gap_analysis.py.

Reads mask_score_gap_histograms.npz (+ optional summary CSV) under --analysis-dir.

**Legacy layout** (single ``magnitude_raw_*``): still produces standalone magnitude/raw + norm PNGs.

**Milestone layout** (``magnitude_raw_step50_*``, ...): produces color-coded overlays of all warm
magnitude endpoints vs oracle, plus random, with log-spaced x for raw distributions and mostly
log x-axes elsewhere (normalized ECDF/hists use gap + floor for log readability).

Interpretation summaries live in ``mask_score_gap_gap_diagnostics.json`` (written by analysis).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_summary_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ecdf(counts: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    total = float(counts.sum())
    if total <= 0:
        return np.array([]), np.array([])
    centers = (edges[:-1] + edges[1:]) / 2.0
    cdf = np.cumsum(counts.astype(np.float64)) / total
    return centers, cdf


def _ecdf_log_edges(counts: np.ndarray, log_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xc = (log_edges[:-1] + log_edges[1:]) / 2.0
    x_lin = 10**xc
    _, y = _ecdf(counts, log_edges)
    return x_lin, y


def _hist_density_lin(counts: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    total = float(counts.sum())
    if total <= 0:
        return np.array([]), np.array([])
    w = np.diff(edges).astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    dens = counts.astype(np.float64) / total / np.maximum(w, 1e-30)
    return centers, dens


def _hist_density_from_log_bins(counts: np.ndarray, log_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return linear-space bin centers + density for plotting on log axis."""
    total = float(counts.sum())
    if total <= 0:
        return np.array([]), np.array([])
    lin_c = (10 ** log_edges[:-1] + 10 ** log_edges[1:]) / 2.0
    widths_lin = (10 ** log_edges[1:] - 10 ** log_edges[:-1]).clip(min=1e-30)
    dens = counts.astype(np.float64) / total / widths_lin
    return lin_c, dens


def _tight_linear_xlim(
    counts: np.ndarray, edges: np.ndarray, cdf_mass: float, xmax_cap: Optional[float]
) -> Tuple[float, float]:
    total = float(counts.sum())
    if total <= 0:
        return 0.0, 1.0
    cum = np.cumsum(counts.astype(np.float64)) / total
    j = int(np.searchsorted(cum, cdf_mass))
    j = min(max(j, 0), len(counts) - 1)
    xmax = float(edges[j + 1])
    nz = np.flatnonzero(counts > 0)
    if nz.size:
        xmax = max(xmax, float(edges[nz[-1] + 1]))
    if xmax_cap is not None:
        xmax = min(xmax, xmax_cap)
    return 0.0, xmax + max(xmax * 0.06, 1e-22)


def _tight_union_log_xlim(series: Sequence[Tuple[np.ndarray, np.ndarray]], cdf_mass: float) -> Tuple[float, float]:
    """Union of tight linear-display limits across log-binned histograms."""
    lo, hi = 1e300, 0.0
    for c, le in series:
        total = float(c.sum())
        if total <= 0:
            continue
        nz = np.flatnonzero(c > 0)
        if nz.size == 0:
            continue
        cum = np.cumsum(c.astype(np.float64)) / total
        j = int(np.searchsorted(cum, cdf_mass))
        j = min(max(j, 0), len(c) - 1)
        hi_lin = float(10 ** le[j + 1])
        lo_lin = float(10 ** le[nz[0]])
        lo = min(lo, lo_lin * 0.9)
        hi = max(hi, hi_lin * 1.08)
    if hi <= 0:
        return 1e-30, 1.0
    return max(lo, 1e-30), hi


def _parse_milestone_steps(npz_keys: Sequence[str]) -> List[int]:
    steps = []
    pat = re.compile(r"^magnitude_raw_step(\d+)_counts$")
    for k in npz_keys:
        m = pat.match(k)
        if m:
            steps.append(int(m.group(1)))
    return sorted(set(steps))


def _colors(milestones: List[int]) -> Dict[str, str]:
    base = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or []
    if not base:
        base = [f"C{i}" for i in range(10)]
    d: Dict[str, str] = {}
    for i, s in enumerate(milestones):
        d[f"step{s}"] = base[i % len(base)]
    d["random"] = "#6c6c6c"
    return d


def _plot_legacy(data, out_dir: Path, args) -> None:
    cm = float(np.clip(args.cdf_mass, 0.5, 1.0))
    counts_mr = data["magnitude_raw_counts"]
    edges_mr = data["magnitude_raw_log_edges"]
    xlim_raw = (
        (1e-30, float(args.raw_xmax))
        if args.raw_xmax is not None
        else _tight_union_log_xlim([(counts_mr, edges_mr)], cdf_mass=cm)
    )

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    c, d = _hist_density_from_log_bins(counts_mr, edges_mr)
    axes[0].plot(c, d, color="tab:blue")
    axes[0].set_xscale("log")
    axes[0].set_title("|warm mag − oracle| raw (legacy single target)")
    axes[0].set_xlim(xlim_raw[0], xlim_raw[1])
    x, y = _ecdf_log_edges(counts_mr, edges_mr)
    axes[1].step(x, y, where="mid", color="tab:blue")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlim(xlim_raw[0], xlim_raw[1])
    axes[1].set_ylabel("CDF")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"magnitude_raw_legacy.{ext}", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot mask score gap outputs (milestones + random vs oracle).")
    ap.add_argument("--analysis-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--cdf-mass", type=float, default=0.9995)
    ap.add_argument("--norm-xmax", type=float, default=None)
    ap.add_argument("--raw-xmax", type=float, default=None)
    args = ap.parse_args()

    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = float(np.clip(args.cdf_mass, 0.5, 1.0))

    npz_path = analysis_dir / "mask_score_gap_histograms.npz"
    summary_path = analysis_dir / "mask_score_gap_summary.csv"

    if not npz_path.is_file():
        raise FileNotFoundError(
            f"{npz_path}\n"
            "  Pass the analysis output dir (contains mask_score_gap_histograms.npz). "
            "Example: --analysis-dir /home/YOU/rl_casino/results/mask_score_gap_<jobid>"
        )

    data = np.load(npz_path)
    keys = sorted(data.files)
    milestones = _parse_milestone_steps(keys)

    cols = _colors(milestones) if milestones else {}

    seed_tags: List[str] = []
    for k in keys:
        if k.startswith("random_raw_seed") and k.endswith("_counts"):
            seed_tags.append(k.replace("random_raw_seed", "").replace("_counts", ""))
    seed_primary = sorted(seed_tags, key=lambda x: int(x) if x.isdigit() else 0)[0] if seed_tags else ""

    # ---------------- legacy single magnitude_raw_counts (optional)
    if "magnitude_raw_counts" in data.files:
        _plot_legacy(data, out_dir, args)

    if not milestones:
        print(
            "No magnitude_raw_step*_counts in npz — if this is intentional, regenerate analysis with milestones."
        )

    # ---------------- overlays: raw (log-x)
    if milestones:
        raw_series_lo = [(data[f"magnitude_raw_step{s}_counts"], data[f"magnitude_raw_step{s}_log_edges"]) for s in milestones]
        if seed_primary:
            raw_series_lo.append(
                (
                    data[f"random_raw_seed{seed_primary}_counts"],
                    data[f"random_raw_seed{seed_primary}_log_edges"],
                )
            )
        xlim_raw_u = (
            (1e-30, float(args.raw_xmax))
            if args.raw_xmax is not None
            else _tight_union_log_xlim(raw_series_lo, cdf_mass=cm)
        )

        fig, axes = plt.subplots(2, 1, figsize=(10, 9))
        for s in milestones:
            c_k = f"magnitude_raw_step{s}_counts"
            e_k = f"magnitude_raw_step{s}_log_edges"
            lin_c, dens = _hist_density_from_log_bins(data[c_k], data[e_k])
            axes[0].plot(lin_c, dens, label=f"|mag@≤{s} − oracle|", color=cols[f"step{s}"], lw=2, alpha=0.9)

        if seed_primary:
            c_k = f"random_raw_seed{seed_primary}_counts"
            e_k = f"random_raw_seed{seed_primary}_log_edges"
            lin_c, dens = _hist_density_from_log_bins(data[c_k], data[e_k])
            axes[0].plot(
                lin_c, dens, label=f"random (seed {seed_primary})", color=cols["random"], lw=2.2, ls="--"
            )

        axes[0].set_xscale("log")
        axes[0].set_title("Raw gap densities vs oracle (warm mag @ intermediate steps)")
        axes[0].set_xlabel("gap (linear, log axis)")
        axes[0].set_ylabel("density")
        axes[0].set_xlim(xlim_raw_u[0], xlim_raw_u[1])
        axes[0].legend(loc="best", fontsize=8)

        for s in milestones:
            ck, ek = f"magnitude_raw_step{s}_counts", f"magnitude_raw_step{s}_log_edges"
            x, y = _ecdf_log_edges(data[ck], data[ek])
            axes[1].step(x, y, where="mid", label=f"mag ≤ {s}", color=cols[f"step{s}"], lw=2)
        if seed_primary:
            ck, ek = f"random_raw_seed{seed_primary}_counts", f"random_raw_seed{seed_primary}_log_edges"
            x, y = _ecdf_log_edges(data[ck], data[ek])
            axes[1].step(x, y, where="mid", label=f"random s{seed_primary}", color=cols["random"], lw=2, ls="--")
        axes[1].set_xscale("log")
        axes[1].set_xlim(xlim_raw_u[0], xlim_raw_u[1])
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel("gap (linear)")
        axes[1].set_ylabel("fraction gap ≤ x")
        axes[1].set_title("ECDF overlays (raw)")
        axes[1].legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"combined_raw_logx.{ext}", dpi=175)
        plt.close(fig)

        # ---------------- normalized: log x via display floor (small gaps dominate)
        norm_series = [(data[f"magnitude_norm_step{s}_counts"], data[f"magnitude_norm_step{s}_edges"]) for s in milestones]
        if seed_primary:
            norm_series.append(
                (data[f"random_norm_seed{seed_primary}_counts"], data[f"random_norm_seed{seed_primary}_edges"])
            )
        xmin_plot = 1e-14
        _xmn = 0.0
        xmax_n = 1.0
        for nc, ne in norm_series:
            _, xi = _tight_linear_xlim(nc, ne, cm, xmax_cap=1.0)
            xmax_n = max(xmax_n, xi)
        if args.norm_xmax is not None:
            xmax_n = float(args.norm_xmax)

        fig, axes = plt.subplots(2, 1, figsize=(10, 9))
        for s in milestones:
            nk, ek = f"magnitude_norm_step{s}_counts", f"magnitude_norm_step{s}_edges"
            cx, dn = _hist_density_lin(data[nk], data[ek])
            cx_vis = np.maximum(cx.astype(np.float64), xmin_plot)
            axes[0].plot(cx_vis, dn, label=f"mag≤{s}", color=cols[f"step{s}"], lw=2)
        if seed_primary:
            nk = f"random_norm_seed{seed_primary}_counts"
            ek = f"random_norm_seed{seed_primary}_edges"
            cx, dn = _hist_density_lin(data[nk], data[ek])
            cx_vis = np.maximum(cx.astype(np.float64), xmin_plot)
            axes[0].plot(cx_vis, dn, label=f"random {seed_primary}", color=cols["random"], lw=2, ls="--")
        axes[0].set_xscale("log")
        axes[0].set_xlim(max(xmin_plot, _xmn), xmax_n)
        axes[0].set_title("Normalized gap densities (per-tensor min–max)")
        axes[0].set_xlabel("normalized gap")
        axes[0].set_ylabel("density")
        axes[0].legend(fontsize=8)

        for s in milestones:
            nk, ek = f"magnitude_norm_step{s}_counts", f"magnitude_norm_step{s}_edges"
            x, y = _ecdf(data[nk], data[ek])
            xv = np.maximum(x.astype(np.float64), xmin_plot)
            axes[1].step(xv, y, where="mid", label=f"mag≤{s}", color=cols[f"step{s}"], lw=2)
        if seed_primary:
            nk, ek = f"random_norm_seed{seed_primary}_counts", f"random_norm_seed{seed_primary}_edges"
            x, y = _ecdf(data[nk], data[ek])
            xv = np.maximum(x.astype(np.float64), xmin_plot)
            axes[1].step(xv, y, where="mid", label=f"random {seed_primary}", color=cols["random"], lw=2, ls="--")
        axes[1].set_xscale("log")
        axes[1].set_xlim(max(xmin_plot, _xmn), xmax_n)
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel("normalized gap")
        axes[1].set_ylabel("fraction gap ≤ x")
        axes[1].set_title("ECDF overlays (normalized)")
        axes[1].legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"combined_norm_logx.{ext}", dpi=175)
        plt.close(fig)

        # ------------ multi-seed random raw
        if len(seed_tags) > 1:
            fig, ax = plt.subplots(figsize=(9, 5))
            srs = [(data[f"random_raw_seed{t}_counts"], data[f"random_raw_seed{t}_log_edges"]) for t in seed_tags]
            x_lr = (
                (1e-30, float(args.raw_xmax))
                if args.raw_xmax is not None
                else _tight_union_log_xlim(srs, cdf_mass=cm)
            )
            for tag in sorted(seed_tags, key=lambda x: int(x) if x.isdigit() else 0):
                x, y = _ecdf_log_edges(data[f"random_raw_seed{tag}_counts"], data[f"random_raw_seed{tag}_log_edges"])
                ax.step(x, y, where="mid", label=f"seed {tag}")
            ax.set_xscale("log")
            ax.set_xlim(x_lr[0], x_lr[1])
            ax.set_ylim(0, 1)
            ax.legend()
            ax.set_title("ECDF raw random-only (multiple seeds)")
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(out_dir / f"random_raw_ecdf_multi_seed.{ext}", dpi=150)
            plt.close(fig)

    if summary_path.is_file():
        rows = load_summary_csv(summary_path)
        fig, ax = plt.subplots(figsize=(10, max(4.0, len(rows) * 0.22)))
        cases = [r["case"] for r in rows]
        means = np.array([float(r["mean"]) if r.get("mean") else float("nan") for r in rows], dtype=np.float64)
        ax.barh(cases, np.maximum(means, 0.0))
        ax.set_xlabel("mean gap")
        ax.set_title("Mean gap by case")
        xmax = float(np.nanmax(means[np.isfinite(means)])) if np.any(np.isfinite(means)) else 1.0
        ax.set_xlim(0, max(xmax * 1.2, 1e-22))
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"summary_mean_bar.{ext}", dpi=150)
        plt.close(fig)

    print(f"Wrote figures under {out_dir}")