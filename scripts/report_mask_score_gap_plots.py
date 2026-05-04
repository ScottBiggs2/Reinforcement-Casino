#!/usr/bin/env python3
"""
Visualize outputs from src/analysis/mask_score_gap_analysis.py.

Reads mask_score_gap_histograms.npz and mask_score_gap_summary.csv under --analysis-dir.
Writes PNG and PDF figures (histograms + ECDFs) for magnitude vs oracle and random vs oracle gaps.

CPU-only; suitable for a small Slurm job after the analysis run::

    python scripts/report_mask_score_gap_plots.py --analysis-dir /path/to/out_dir
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

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


def _plot_lin_hist(ax, counts: np.ndarray, edges: np.ndarray, title: str, xlabel: str) -> None:
    c, d = _hist_density(counts, edges)
    if c.size == 0:
        ax.set_title(title + " (empty)")
        return
    ax.bar(c, d, width=np.diff(edges), align="center", alpha=0.85, edgecolor="none")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")


def _plot_log_hist(ax, counts: np.ndarray, log_edges: np.ndarray, title: str) -> None:
    """log_edges are log10(bin bounds); x-axis shown in linear space."""
    total = float(counts.sum())
    if total <= 0:
        ax.set_title(title + " (empty)")
        return
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2.0
    lin_centers = 10**log_centers
    widths_log = np.diff(log_edges)
    # approximate linear bin widths at center for bar display (log-spaced bins)
    widths_lin = (10 ** (log_centers + widths_log / 2) - 10 ** (log_centers - widths_log / 2)).clip(min=1e-30)
    density = counts.astype(np.float64) / total / widths_lin
    ax.bar(lin_centers, density, width=widths_lin, align="center", alpha=0.85, edgecolor="none")
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("gap (linear)")
    ax.set_ylabel("density")


def _plot_ecdf_lin(ax, counts: np.ndarray, edges: np.ndarray, title: str, xlabel: str) -> None:
    x, y = _ecdf(counts, edges)
    if x.size == 0:
        ax.set_title(title + " (empty)")
        return
    ax.step(x, y, where="mid")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")


def _plot_ecdf_log(ax, counts: np.ndarray, log_edges: np.ndarray, title: str) -> None:
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
    ax.set_xlabel("gap (linear)")
    ax.set_ylabel("CDF")


def load_summary_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot mask score gap histograms/ECDFs from analysis output.")
    ap.add_argument("--analysis-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=None, help="Figure output dir (default: analysis-dir/figures)")
    args = ap.parse_args()

    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = analysis_dir / "mask_score_gap_histograms.npz"
    summary_path = analysis_dir / "mask_score_gap_summary.csv"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path)

    # Magnitude raw (log-binned x)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    _plot_log_hist(axes[0], data["magnitude_raw_counts"], data["magnitude_raw_log_edges"], "|mag − oracle| raw")
    _plot_ecdf_log(axes[1], data["magnitude_raw_counts"], data["magnitude_raw_log_edges"], "ECDF raw magnitude gap")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"magnitude_raw.{ext}", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    _plot_lin_hist(
        axes[0],
        data["magnitude_norm_counts"],
        data["magnitude_norm_edges"],
        "|mag − oracle| per-tensor min–max norm",
        "gap",
    )
    _plot_ecdf_lin(
        axes[1],
        data["magnitude_norm_counts"],
        data["magnitude_norm_edges"],
        "ECDF normalized magnitude gap",
        "gap",
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"magnitude_norm.{ext}", dpi=150)
    plt.close(fig)

    # Random seeds: detect keys random_raw_*_counts
    seed_tags: List[str] = []
    for k in data.files:
        if k.startswith("random_raw_seed") and k.endswith("_counts"):
            seed_tags.append(k.replace("random_raw_seed", "").replace("_counts", ""))

    if seed_tags:
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
        ax.set_title("ECDF raw random vs oracle gap (by seed)")
        ax.set_xlabel("gap (linear)")
        ax.set_ylabel("CDF")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"random_raw_ecdf_multi_seed.{ext}", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for tag in sorted(seed_tags, key=lambda x: int(x) if x.isdigit() else 0):
            ckey = f"random_norm_seed{tag}_counts"
            ekey = f"random_norm_seed{tag}_edges"
            if ckey not in data.files or ekey not in data.files:
                continue
            x, y = _ecdf(data[ckey], data[ekey])
            if x.size == 0:
                continue
            ax.step(x, y, where="mid", label=f"seed {tag}")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title("ECDF normalized random vs oracle gap (by seed)")
        ax.set_xlabel("gap")
        ax.set_ylabel("CDF")
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
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"summary_mean_bar.{ext}", dpi=150)
        plt.close(fig)

    print(f"Wrote figures under {out_dir}")


if __name__ == "__main__":
    main()
