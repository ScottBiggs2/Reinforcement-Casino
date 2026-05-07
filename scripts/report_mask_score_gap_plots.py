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
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
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


def _load_optional_npz(path: Path) -> Optional["np.lib.npyio.NpzFile"]:
    try:
        if path.is_file():
            return np.load(path)
    except Exception:
        return None
    return None


def _margin_milestone_steps(npz_keys: Sequence[str], milestones: Sequence[int]) -> List[int]:
    out: List[int] = []
    for s in milestones:
        k = f"magnitude_margin_raw_step{s}_counts"
        if k in npz_keys:
            out.append(int(s))
    return out


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
    align_npz_path = analysis_dir / "mask_score_gap_alignment_histograms.npz"

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

    align = _load_optional_npz(align_npz_path)

    # ---------------- legacy single magnitude_raw_counts (optional)
    if "magnitude_raw_counts" in data.files:
        _plot_legacy(data, out_dir, args)

    if not milestones:
        print(
            "No magnitude_raw_step*_counts in npz — if this is intentional, regenerate analysis with milestones."
        )

    # ---------------- overlays: raw (log-x), side-by-side PDF | ECDF
    if milestones:
        raw_series_lo = [(data[f"magnitude_raw_step{s}_counts"], data[f"magnitude_raw_step{s}_log_edges"]) for s in milestones]
        if seed_primary:
            raw_series_lo.append(
                (
                    data[f"random_raw_seed{seed_primary}_counts"],
                    data[f"random_raw_seed{seed_primary}_log_edges"],
                )
            )
        if align is not None and "alignment_gap_oracle_raw_counts" in align.files:
            raw_series_lo.append((align["alignment_gap_oracle_raw_counts"], align["alignment_gap_oracle_raw_log_edges"]))
        xlim_raw_u = (
            (1e-30, float(args.raw_xmax))
            if args.raw_xmax is not None
            else _tight_union_log_xlim(raw_series_lo, cdf_mass=cm)
        )
        # Trim heavy small-x tail: clamp left edge to 1e-12 for legibility.
        x_lo_disp = max(float(xlim_raw_u[0]), 1e-14)
        x_hi_disp = float(xlim_raw_u[1])

        # Bigger figure + bigger fonts for paper-ready legibility.
        with plt.rc_context(
            {
                "font.size": 16,
                "axes.titlesize": 18,
                "axes.labelsize": 17,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "legend.fontsize": 12,
                "mathtext.fontset": "cm",
            }
        ):
            fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
            ax_pdf, ax_ecdf = axes[0], axes[1]

            # ----- PDF / density panel (left)
            for s in milestones:
                c_k = f"magnitude_raw_step{s}_counts"
                e_k = f"magnitude_raw_step{s}_log_edges"
                lin_c, dens = _hist_density_from_log_bins(data[c_k], data[e_k])
                ax_pdf.plot(
                    lin_c,
                    dens,
                    label=rf"$|\,w^{{\mathrm{{warm}}}}_{{\leq {s}}} - w^{{*}}\,|$",
                    color=cols[f"step{s}"],
                    lw=2.2,
                    alpha=0.9,
                )
            if seed_primary:
                c_k = f"random_raw_seed{seed_primary}_counts"
                e_k = f"random_raw_seed{seed_primary}_log_edges"
                lin_c, dens = _hist_density_from_log_bins(data[c_k], data[e_k])
                ax_pdf.plot(
                    lin_c,
                    dens,
                    label=rf"random (seed {seed_primary})",
                    color=cols["random"],
                    lw=2.4,
                    ls="--",
                )
            if align is not None and "alignment_gap_oracle_raw_counts" in align.files:
                lin_c, dens = _hist_density_from_log_bins(
                    align["alignment_gap_oracle_raw_counts"], align["alignment_gap_oracle_raw_log_edges"]
                )
                ax_pdf.plot(
                    lin_c,
                    dens,
                    label=r"$|\,s^{*} - w^{*}\,|$ (alignment)",
                    color="#000000",
                    lw=2.6,
                    ls="-.",
                )

            ax_pdf.set_xscale("log")
            ax_pdf.set_title(r"PDF of raw gap $|s_i - w^{*}_i|$")
            ax_pdf.set_xlabel(r"gap $|s_i - w^{*}_i|$")
            ax_pdf.set_ylabel(r"density $\hat f(x)$")
            ax_pdf.set_xlim(x_lo_disp, x_hi_disp)
            ax_pdf.grid(True, which="both", ls=":", alpha=0.35)
            ax_pdf.legend(loc="best")

            # ----- ECDF panel (right)
            for s in milestones:
                ck, ek = f"magnitude_raw_step{s}_counts", f"magnitude_raw_step{s}_log_edges"
                x, y = _ecdf_log_edges(data[ck], data[ek])
                ax_ecdf.step(
                    x,
                    y,
                    where="mid",
                    label=rf"warm mag, step $\leq {s}$",
                    color=cols[f"step{s}"],
                    lw=2.2,
                )
            if seed_primary:
                ck, ek = f"random_raw_seed{seed_primary}_counts", f"random_raw_seed{seed_primary}_log_edges"
                x, y = _ecdf_log_edges(data[ck], data[ek])
                ax_ecdf.step(
                    x,
                    y,
                    where="mid",
                    label=rf"random (seed {seed_primary})",
                    color=cols["random"],
                    lw=2.2,
                    ls="--",
                )
            if align is not None and "alignment_gap_oracle_raw_counts" in align.files:
                x, y = _ecdf_log_edges(
                    align["alignment_gap_oracle_raw_counts"],
                    align["alignment_gap_oracle_raw_log_edges"],
                )
                ax_ecdf.step(
                    x,
                    y,
                    where="mid",
                    label=r"alignment $|s^{*} - w^{*}|$",
                    color="#000000",
                    lw=2.4,
                    ls="-.",
                )
            ax_ecdf.set_xscale("log")
            ax_ecdf.set_xlim(x_lo_disp, x_hi_disp)
            ax_ecdf.set_ylim(0, 1)
            ax_ecdf.set_xlabel(r"gap $|s_i - w^{*}_i|$")
            ax_ecdf.set_ylabel(r"$\mathrm{Pr}[\,|s_i - w^{*}_i| \leq x\,]$")
            ax_ecdf.set_title(r"ECDF of raw gap $|s_i - w^{*}_i|$")
            ax_ecdf.grid(True, which="both", ls=":", alpha=0.35)
            ax_ecdf.legend(loc="lower right")

            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(out_dir / f"combined_raw_logx.{ext}", dpi=200, bbox_inches="tight")
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

        # ---------------- margin ECDF overlays (Theorem 3): m_i(s) = |s_i - tau|
        m_steps = _margin_milestone_steps(keys, milestones)
        margin_series_lo: List[Tuple[np.ndarray, np.ndarray]] = []
        for s in m_steps:
            ck, ek = f"magnitude_margin_raw_step{s}_counts", f"magnitude_margin_raw_step{s}_log_edges"
            if ck in data.files and float(data[ck].sum()) > 0:
                margin_series_lo.append((data[ck], data[ek]))
        if seed_primary:
            cmr = f"random_margin_raw_seed{seed_primary}_counts"
            emr = f"random_margin_raw_seed{seed_primary}_log_edges"
            if cmr in data.files and float(data[cmr].sum()) > 0:
                margin_series_lo.append((data[cmr], data[emr]))
        if "oracle_margin_raw_counts" in data.files and float(data["oracle_margin_raw_counts"].sum()) > 0:
            margin_series_lo.append((data["oracle_margin_raw_counts"], data["oracle_margin_raw_log_edges"]))

        if margin_series_lo:
            xlim_m = (
                (1e-30, float(args.raw_xmax))
                if args.raw_xmax is not None
                else _tight_union_log_xlim(margin_series_lo, cdf_mass=cm)
            )
            fig, ax = plt.subplots(figsize=(10, 5.5))
            for s in m_steps:
                ck, ek = f"magnitude_margin_raw_step{s}_counts", f"magnitude_margin_raw_step{s}_log_edges"
                if ck not in data.files or float(data[ck].sum()) <= 0:
                    continue
                x, y = _ecdf_log_edges(data[ck], data[ek])
                ax.step(x, y, where="mid", label=f"|mag@<={s} - tau|", color=cols[f"step{s}"], lw=2)
            if seed_primary:
                cmr, emr = f"random_margin_raw_seed{seed_primary}_counts", f"random_margin_raw_seed{seed_primary}_log_edges"
                if cmr in data.files and float(data[cmr].sum()) > 0:
                    x, y = _ecdf_log_edges(data[cmr], data[emr])
                    ax.step(x, y, where="mid", label=f"random margin (seed {seed_primary})", color=cols["random"], lw=2, ls="--")
            if "oracle_margin_raw_counts" in data.files and float(data["oracle_margin_raw_counts"].sum()) > 0:
                x, y = _ecdf_log_edges(data["oracle_margin_raw_counts"], data["oracle_margin_raw_log_edges"])
                ax.step(x, y, where="mid", label="|oracle - tau*|", color="#2ca02c", lw=2, ls=":")
            ax.set_xscale("log")
            ax.set_xlim(xlim_m[0], xlim_m[1])
            ax.set_ylim(0, 1)
            ax.set_xlabel("margin (linear)")
            ax.set_ylabel("fraction margin <= x")
            ax.set_title("ECDF certifiability margins m_i(s) = |s_i - tau_rho(s)| (global top-k)")
            ax.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(out_dir / f"combined_margin_ecdf_logx.{ext}", dpi=175)
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

        cert_rows = [r for r in rows if str(r.get("case", "")).startswith("cert_strict_")]
        if cert_rows:
            fig, ax = plt.subplots(figsize=(10, max(3.5, len(cert_rows) * 0.28)))
            ccases = [r["case"] for r in cert_rows]
            fracs = np.array(
                [float(r["mean"]) if r.get("mean") else float("nan") for r in cert_rows], dtype=np.float64
            )
            ax.barh(ccases, np.clip(fracs, 0.0, 1.0))
            ax.set_xlim(0, 1.05)
            ax.set_xlabel("fraction weights with gap < margin (strict)")
            ax.set_title("Certifiability rate P[g_i < m_i] at configured sparsity")
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(out_dir / f"certifiability_frac_bar.{ext}", dpi=150)
            plt.close(fig)

    n_png = len(list(out_dir.glob("*.png")))
    print(f"Wrote {n_png} PNG file(s) under {out_dir}")


if __name__ == "__main__":
    main()