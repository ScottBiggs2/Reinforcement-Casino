#!/usr/bin/env python3
"""
Plot benchmark CSV from h200_sparse_dpo_bsr_benchmark.py (or compatible training_log.csv).

Example:
  python scripts/plot_h200_bsr_benchmark.py \\
    --csv /path/to/benchmark_training_log.csv \\
    --out_dir /path/to/plots
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _phase_label(row: pd.Series) -> str:
    ph = str(row.get("phase", "") or "").strip()
    if ph == "dense":
        return "dense"
    sp = row.get("sparsity_target_pct", "")
    if sp == "" or (isinstance(sp, float) and np.isnan(sp)):
        return ph or "dense"
    try:
        return f"{float(sp):g}% sp"
    except (TypeError, ValueError):
        return ph or "unknown"


def _add_phase_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "plot_group" not in out.columns:
        if "phase" in out.columns:
            out["plot_group"] = out["phase"].astype(str)
        else:
            out["plot_group"] = out.apply(_phase_label, axis=1)
    return out


def plot_throughput(df: pd.DataFrame, out_path: str, dpi: int) -> None:
    need = {"step", "cumulative_samples_per_s"}
    if not need.issubset(df.columns):
        print(f"Skip throughput plot (need columns {need}, have {set(df.columns)})", file=sys.stderr)
        return
    df = _add_phase_key(df)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for key, g in df.groupby("plot_group", sort=True):
        g = g.sort_values("step")
        ax.plot(g["step"], g["cumulative_samples_per_s"], label=key, linewidth=1.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative samples/s")
    ax.set_title("Training throughput (cumulative)")
    ax.legend(title="Phase", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_steps_per_sec(df: pd.DataFrame, out_path: str, dpi: int) -> None:
    need = {"step", "cumulative_steps_per_s"}
    if not need.issubset(df.columns):
        print(f"Skip steps/s plot (need {need})", file=sys.stderr)
        return
    df = _add_phase_key(df)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for key, g in df.groupby("plot_group", sort=True):
        g = g.sort_values("step")
        ax.plot(g["step"], g["cumulative_steps_per_s"], label=key, linewidth=1.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative steps/s")
    ax.set_title("Optimizer steps per second (cumulative)")
    ax.legend(title="Phase", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_loss(df: pd.DataFrame, out_path: str, dpi: int) -> None:
    loss_col = None
    for c in ("loss", "train_loss", "rewards/chosen"):
        if c in df.columns:
            loss_col = c
            break
    if loss_col is None or "step" not in df.columns:
        print("Skip loss plot (no loss column)", file=sys.stderr)
        return
    df = _add_phase_key(df)
    sub = df[df[loss_col].notna()].copy()
    if sub.empty:
        print("Skip loss plot (empty)", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for key, g in sub.groupby("plot_group", sort=True):
        g = g.sort_values("step")
        ax.plot(g["step"], g[loss_col], label=key, linewidth=1.5, alpha=0.9)
    ax.set_xlabel("Step")
    ax.set_ylabel(loss_col)
    ax.set_title("Training loss")
    ax.legend(title="Phase", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_summary_bars(df: pd.DataFrame, out_path: str, dpi: int, last_n: int) -> None:
    if "cumulative_samples_per_s" not in df.columns:
        print("Skip summary bars", file=sys.stderr)
        return
    df = _add_phase_key(df)
    rows = []
    for key, g in df.groupby("plot_group", sort=True):
        g = g.sort_values("step")
        tail = g.tail(max(1, last_n))
        rows.append(
            {
                "plot_group": key,
                "mean_samples_s": float(tail["cumulative_samples_per_s"].mean()),
                "mean_steps_s": float(tail["cumulative_steps_per_s"].mean())
                if "cumulative_steps_per_s" in tail.columns
                else float("nan"),
            }
        )
    if not rows:
        return
    summ = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    x = np.arange(len(summ))
    axes[0].bar(x, summ["mean_samples_s"], color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summ["plot_group"], rotation=35, ha="right")
    axes[0].set_ylabel("Mean cumulative samples/s")
    axes[0].set_title(f"Mean over last {last_n} logged rows")
    axes[0].grid(True, axis="y", alpha=0.3)

    if summ["mean_steps_s"].notna().any():
        axes[1].bar(x, summ["mean_steps_s"], color="seagreen")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(summ["plot_group"], rotation=35, ha="right")
        axes[1].set_ylabel("Mean cumulative steps/s")
        axes[1].set_title(f"Mean over last {last_n} logged rows")
        axes[1].grid(True, axis="y", alpha=0.3)
    else:
        axes[1].set_visible(False)

    fig.suptitle("Throughput summary by phase")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Plot H200 BSR benchmark CSV logs.")
    ap.add_argument("--csv", required=True, help="Path to benchmark_training_log.csv")
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for PNGs (default: same dir as CSV)",
    )
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--summary_last_n",
        type=int,
        default=20,
        help="Rows from the end of each phase used for bar-chart means",
    )
    args = ap.parse_args(argv)

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV is empty.", file=sys.stderr)
        sys.exit(1)

    base = os.path.join(out_dir, "h200_bsr_benchmark")
    plot_throughput(df, f"{base}_throughput_samples.png", args.dpi)
    plot_steps_per_sec(df, f"{base}_throughput_steps.png", args.dpi)
    plot_loss(df, f"{base}_loss.png", args.dpi)
    plot_summary_bars(df, f"{base}_summary_bars.png", args.dpi, args.summary_last_n)
    print(f"Done. Plots in: {out_dir}")


if __name__ == "__main__":
    main()
