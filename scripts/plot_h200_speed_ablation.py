#!/usr/bin/env python3
"""
Visual reporting for H200 optimizer benchmark CSVs.

Reads ``benchmark_training_log.csv`` (and optionally ``speed_ablation_report.json``
or ``--slurm-out`` for wall-time breakdown), writes PNGs under ``plots/`` and
``benchmark_visual_report.md`` under ``--run-dir``.

Usage::

  python3 scripts/plot_h200_speed_ablation.py --run-dir /path/to/run_or_results_folder
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _tail_mean(series: pd.Series, n: int) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    tail = s.tail(max(1, int(n)))
    return float(tail.mean())


def _to_float(x: Any) -> float:
    if x is None or (isinstance(x, str) and x.strip() == ""):
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_benchmark_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    skip_numeric = {"phase", "optimizer", "mask_type", "theory_doc_ref"}
    for c in df.columns:
        if c in skip_numeric:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def primary_throughput_column(df: pd.DataFrame) -> Tuple[str, str]:
    """Return (column_name, human label) for per-step / summary throughput."""
    if "inst_steps_per_s" in df.columns and df["inst_steps_per_s"].notna().any():
        return "inst_steps_per_s", "inst. steps/s"
    if "cumulative_steps_per_s_excl_first" in df.columns:
        if df["cumulative_steps_per_s_excl_first"].notna().any():
            return "cumulative_steps_per_s_excl_first", "cum. steps/s (excl. 1st)"
    return "cumulative_steps_per_s", "cum. steps/s"


def mask_family_from_row(row: pd.Series) -> str:
    mt = row.get("mask_type")
    if isinstance(mt, str) and mt.strip():
        return "block" if mt.strip().lower() == "block" else "element"
    ph = str(row.get("phase", ""))
    if ph.endswith("_blk") or "_blk" in ph:
        return "block"
    if ph.endswith("_elem") or "_elem" in ph:
        return "element"
    if ph == "dense":
        return "dense"
    return "unknown"


def compute_phase_table(
    df: pd.DataFrame, tp_col: str, tail_steps: int
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for phase in sorted(df["phase"].dropna().unique(), key=str):
        sub = df[df["phase"] == phase].sort_values("step")
        if sub.empty:
            continue
        tail_inst = _tail_mean(sub["inst_steps_per_s"], tail_steps) if "inst_steps_per_s" in sub.columns else float("nan")
        tail_excl = (
            _tail_mean(sub["cumulative_steps_per_s_excl_first"], tail_steps)
            if "cumulative_steps_per_s_excl_first" in sub.columns
            else float("nan")
        )
        sp = sub["sparsity_target_pct"].iloc[0] if "sparsity_target_pct" in sub.columns else None
        try:
            sp_f = float(sp) if sp is not None and str(sp).strip() != "" and not pd.isna(sp) else float("nan")
        except (TypeError, ValueError):
            sp_f = float("nan")
        last = sub.iloc[-1]
        mfamily = mask_family_from_row(last)
        t_frac = last.get("theory_active_fraction_global")
        try:
            t_frac_f = float(t_frac) if t_frac != "" and t_frac is not None and not pd.isna(t_frac) else float("nan")
        except (TypeError, ValueError):
            t_frac_f = float("nan")
        rows.append(
            {
                "phase": phase,
                "sparsity_target_pct": sp_f,
                "mask_family": mfamily,
                "max_step": int(sub["step"].max()) if "step" in sub.columns else 0,
                "tail_mean_inst_steps_per_s": tail_inst,
                "tail_mean_steps_per_s_excl_first": tail_excl,
                "summary_metric": _tail_mean(sub[tp_col], tail_steps) if tp_col in sub.columns else float("nan"),
                "theory_active_fraction_global": t_frac_f,
            }
        )
    out = pd.DataFrame(rows)
    dense_metric = float("nan")
    dense_row = out[out["phase"] == "dense"]
    if not dense_row.empty:
        dense_metric = float(dense_row["summary_metric"].iloc[0])
    ratios: List[float] = []
    for _, r in out.iterrows():
        if str(r["phase"]) == "dense" or math.isnan(dense_metric) or dense_metric <= 0:
            ratios.append(float("nan"))
        else:
            m = float(r["summary_metric"])
            ratios.append(round(m / dense_metric, 6) if m == m and m > 0 else float("nan"))
    out["ratio_vs_dense"] = ratios
    return out


def parse_slurm_bench_json(text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    mask_by_phase: Dict[str, Dict[str, Any]] = {}
    train_by_phase: Dict[str, Dict[str, Any]] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("BENCH_JSON "):
            continue
        raw = s[len("BENCH_JSON ") :].strip()
        try:
            b = json.loads(raw)
        except json.JSONDecodeError:
            continue
        ph = str(b.get("phase", ""))
        k = b.get("kind")
        if k == "mask" and ph:
            mask_by_phase[ph] = b
        elif k == "train_wall" and ph:
            train_by_phase[ph] = b
    m_rows = [{"phase": p, **v} for p, v in sorted(mask_by_phase.items())]
    t_rows = [{"phase": p, **v} for p, v in sorted(train_by_phase.items())]
    return m_rows, t_rows


def wall_seconds_from_report_json(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    summaries = data.get("phase_summaries")
    if not isinstance(summaries, list):
        return None
    rows: List[Dict[str, Any]] = []
    for s in summaries:
        ph = s.get("phase")
        if not ph:
            continue
        mb = s.get("mask_bench") or {}
        tw = s.get("train_wall_bench") or {}
        rows.append(
            {
                "phase": str(ph),
                "mask_s": _to_float(mb.get("seconds")),
                "prepare_s": _to_float(tw.get("prepare_s")),
                "trainer_s": _to_float(tw.get("trainer_s")),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def merge_wall_from_slurm(
    mask_rows: List[Dict[str, Any]], train_rows: List[Dict[str, Any]]
) -> pd.DataFrame:
    by_phase: Dict[str, Dict[str, float]] = {}
    for r in mask_rows:
        ph = str(r.get("phase", ""))
        by_phase.setdefault(ph, {})["mask_s"] = _to_float(r.get("seconds"))
    for r in train_rows:
        ph = str(r.get("phase", ""))
        e = by_phase.setdefault(ph, {})
        e["prepare_s"] = _to_float(r.get("prepare_s"))
        e["trainer_s"] = _to_float(r.get("trainer_s"))
    rows = [
        {"phase": ph, **vals}
        for ph, vals in sorted(by_phase.items(), key=lambda x: x[0])
    ]
    return pd.DataFrame(rows)


def plot_throughput_vs_step(
    df: pd.DataFrame,
    tp_col: str,
    tp_label: str,
    out_path: Path,
) -> None:
    phases = sorted(df["phase"].dropna().unique(), key=str)
    n = len(phases)
    if n == 0:
        return
    ncols = min(3, max(1, int(math.ceil(math.sqrt(n)))))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    for idx, phase in enumerate(phases):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sub = df[df["phase"] == phase].sort_values("step")
        if "inst_steps_per_s" in sub.columns and sub["inst_steps_per_s"].notna().any():
            y = sub["inst_steps_per_s"]
            ylab = "inst. steps/s"
        else:
            y = sub[tp_col] if tp_col in sub.columns else sub.get("cumulative_steps_per_s", pd.Series(dtype=float))
            ylab = tp_label
        ax.plot(sub["step"], y, color="#2c7fb8", linewidth=1.2)
        ax.set_title(phase, fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle("Throughput vs step (per phase)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_placeholder(path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, wrap=True)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_tail_throughput_bars(
    ptable: pd.DataFrame, dense_metric: float, tp_label: str, out_path: Path
) -> None:
    sparse = ptable[ptable["phase"] != "dense"].copy()
    if sparse.empty:
        _save_placeholder(out_path, "No sparse phases in CSV; skipped tail bar chart.")
        return
    sparse = sparse[sparse["mask_family"].isin(("element", "block"))]
    if sparse.empty:
        _save_placeholder(out_path, "No element/block phases; skipped tail bar chart.")
        return
    sparse["sparsity"] = sparse["sparsity_target_pct"]
    order = sorted(sparse["sparsity"].dropna().unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(order))
    width = 0.36
    for fam, color, off in [
        ("element", "#7fc97f", -width / 2),
        ("block", "#beaed4", width / 2),
    ]:
        heights = []
        for sp in order:
            row = sparse[(sparse["sparsity"] == sp) & (sparse["mask_family"] == fam)]
            heights.append(float(row["summary_metric"].iloc[0]) if len(row) else float("nan"))
        ax.bar(x + off, heights, width, label=fam, color=color)
    if dense_metric == dense_metric and dense_metric > 0:
        ax.axhline(dense_metric, color="#386cb0", linestyle="--", linewidth=1.5, label="dense (tail mean)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:g}%" for s in order])
    ax.set_xlabel("target sparsity (% zeros)")
    ax.set_ylabel(f"tail mean {tp_label}")
    ax.legend()
    ax.set_title("Tail throughput by sparsity × mask type")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_vs_dense(ptable: pd.DataFrame, out_path: Path) -> None:
    sparse = ptable[(ptable["phase"] != "dense") & ptable["mask_family"].isin(("element", "block"))].copy()
    if sparse.empty:
        _save_placeholder(out_path, "No sparse phases; skipped ratio chart.")
        return
    rvd = pd.to_numeric(sparse["ratio_vs_dense"], errors="coerce")
    if rvd.notna().sum() == 0:
        _save_placeholder(out_path, "No valid ratio_vs_dense values (missing dense baseline?).")
        return
    sparse["sparsity"] = sparse["sparsity_target_pct"]
    order = sorted(sparse["sparsity"].dropna().unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(order))
    width = 0.36
    for fam, color, off in [("element", "#7fc97f", -width / 2), ("block", "#beaed4", width / 2)]:
        heights: List[float] = []
        for sp in order:
            row = sparse[(sparse["sparsity"] == sp) & (sparse["mask_family"] == fam)]
            rv_raw = row["ratio_vs_dense"].iloc[0] if len(row) else np.nan
            rvf = float(pd.to_numeric(rv_raw, errors="coerce"))
            heights.append(rvf if not math.isnan(rvf) else float("nan"))
        ax.bar(x + off, heights, width, label=fam, color=color)
    ax.axhline(1.0, color="#666666", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:g}%" for s in order])
    ax.set_xlabel("target sparsity (% zeros)")
    ax.set_ylabel("ratio vs dense (tail throughput)")
    ax.set_title("Relative speed vs dense baseline")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_wall_breakdown(wall_df: pd.DataFrame, out_path: Path) -> None:
    if wall_df.empty:
        return
    df = wall_df.copy()
    for c in ("mask_s", "prepare_s", "trainer_s"):
        if c not in df.columns:
            df[c] = float("nan")
    phases = df["phase"].tolist()
    m = df["mask_s"].fillna(0).to_numpy()
    p = df["prepare_s"].fillna(0).to_numpy()
    t = df["trainer_s"].fillna(0).to_numpy()
    x = np.arange(len(phases))
    fig, ax = plt.subplots(figsize=(max(8, len(phases) * 0.45), 4.5))
    ax.bar(x, m, label="mask (gen/load)", color="#fdb462")
    ax.bar(x, p, bottom=m, label="prepare", color="#80b1d3")
    ax.bar(x, t, bottom=m + p, label="trainer.train", color="#bebada")
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("seconds")
    ax.set_title("Wall time composition (BENCH_JSON)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_theory_vs_metric(ptable: pd.DataFrame, out_path: Path) -> bool:
    sub = ptable[ptable["phase"] != "dense"].copy()
    sub = sub[sub["theory_active_fraction_global"].notna() & (sub["theory_active_fraction_global"] > 0)]
    if len(sub) < 2:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for fam, color in [("element", "#7fc97f"), ("block", "#beaed4")]:
        s = sub[sub["mask_family"] == fam]
        if s.empty:
            continue
        ax.scatter(
            s["theory_active_fraction_global"],
            s["summary_metric"],
            label=fam,
            color=color,
            s=55,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for _, r in s.iterrows():
            ax.annotate(
                str(r["phase"]),
                (r["theory_active_fraction_global"], r["summary_metric"]),
                fontsize=7,
                alpha=0.85,
            )
    ax.set_xlabel("theory active fraction (global)")
    ax.set_ylabel("tail mean throughput (summary metric)")
    ax.set_title("Theory active fraction vs measured tail throughput")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def write_markdown(
    md_path: Path,
    run_dir: Path,
    tp_col: str,
    tp_label: str,
    ptable: pd.DataFrame,
    plot_rel_paths: Dict[str, str],
    notes: List[str],
) -> None:
    lines: List[str] = [
        "# H200 benchmark visual report",
        "",
        f"- **run_dir:** `{run_dir}`",
        f"- **generated (UTC):** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
        f"- **throughput column used for summary:** `{tp_col}` ({tp_label})",
        "",
    ]
    if notes:
        lines.append("## Notes")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")
    lines.extend(
        [
            "## Per-phase tail metrics",
            "",
            "| phase | mask | sparsity % | max_step | tail inst | tail excl_first | summary | vs dense |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, r in ptable.iterrows():
        sp = r.get("sparsity_target_pct")
        sp_s = "" if sp != sp or pd.isna(sp) else f"{float(sp):g}"
        rv = r.get("ratio_vs_dense")
        try:
            rv_f = float(rv)
            rv_s = "" if math.isnan(rv_f) else f"{rv_f:.4f}"
        except (TypeError, ValueError):
            rv_s = ""
        lines.append(
            f"| `{r['phase']}` | {r['mask_family']} | {sp_s} | {int(r['max_step'])} | "
            f"{r['tail_mean_inst_steps_per_s']:.6g} | {r['tail_mean_steps_per_s_excl_first']:.6g} | "
            f"{r['summary_metric']:.6g} | {rv_s} |"
        )
    lines.extend(["", "## Figures", ""])

    order = [
        ("Throughput vs step (per phase)", "01_throughput_vs_step"),
        ("Tail throughput by sparsity × mask", "02_tail_throughput_sparsity_mask"),
        ("Ratio vs dense", "03_ratio_vs_dense"),
        ("Wall-time composition", "04_wall_breakdown"),
        ("Theory active fraction vs throughput", "05_theory_vs_metric"),
    ]
    for title, key in order:
        if key not in plot_rel_paths:
            continue
        rp = plot_rel_paths[key]
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}]({rp})")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- **Fig. 1** shows whether per-step interval throughput stabilizes by the end of each phase (short runs may still show compile noise at early steps).",
            "- **Fig. 2** compares sparse mask families against the horizontal dense baseline (tail mean).",
            "- **Fig. 3** expresses the same comparison as a ratio; values above 1 mean faster than dense on this summary metric.",
            "- **Fig. 4** (when present) separates one-time mask work and prepare from `trainer.train` wall time.",
            "- **Fig. 5** (when present) relates the theory proxy for global active fraction to measured throughput.",
            "",
        ]
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True, help="Folder with benchmark_training_log.csv")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="PNG output directory (default: <run-dir>/plots)",
    )
    ap.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Markdown output (default: <run-dir>/benchmark_visual_report.md)",
    )
    ap.add_argument("--tail-steps", type=int, default=8, help="Tail window for means (default 8)")
    ap.add_argument(
        "--slurm-out",
        type=Path,
        default=None,
        help="Slurm stdout with BENCH_JSON lines (optional wall breakdown source)",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "plots")).resolve()
    md_out = (args.md_out or (run_dir / "benchmark_visual_report.md")).resolve()
    csv_path = run_dir / "benchmark_training_log.csv"

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 100

    notes: List[str] = []
    try:
        df = load_benchmark_csv(csv_path)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    tp_col, tp_label = primary_throughput_column(df)
    if tp_col != "inst_steps_per_s":
        notes.append(f"Primary metric fallback: `{tp_col}` (no usable `inst_steps_per_s`).")

    ptable = compute_phase_table(df, tp_col, args.tail_steps)
    dense_row = ptable[ptable["phase"] == "dense"]
    dense_metric = float(dense_row["summary_metric"].iloc[0]) if not dense_row.empty else float("nan")

    out_dir.mkdir(parents=True, exist_ok=True)

    p01 = out_dir / "01_throughput_vs_step.png"
    plot_throughput_vs_step(df, tp_col, tp_label, p01)

    p02 = out_dir / "02_tail_throughput_sparsity_mask.png"
    plot_tail_throughput_bars(ptable, dense_metric, tp_label, p02)

    p03 = out_dir / "03_ratio_vs_dense.png"
    plot_ratio_vs_dense(ptable, p03)

    p04: Optional[Path] = None
    wall_df: Optional[pd.DataFrame] = None
    if args.slurm_out and args.slurm_out.is_file():
        text = args.slurm_out.read_text(encoding="utf-8", errors="replace")
        m_rows, t_rows = parse_slurm_bench_json(text)
        wall_df = merge_wall_from_slurm(m_rows, t_rows)
        if wall_df is not None and wall_df.empty:
            wall_df = None
    if wall_df is None or wall_df.empty:
        wall_df = wall_seconds_from_report_json(run_dir / "speed_ablation_report.json")
    if wall_df is not None and not wall_df.empty:
        p04 = out_dir / "04_wall_breakdown.png"
        plot_wall_breakdown(wall_df, p04)
    else:
        notes.append("Wall breakdown figure skipped (no BENCH_JSON in --slurm-out and no usable speed_ablation_report.json).")

    p05 = out_dir / "05_theory_vs_metric.png"
    theory_ok = plot_theory_vs_metric(ptable, p05)
    if not theory_ok:
        try:
            if p05.exists():
                p05.unlink()
        except OSError:
            pass
        notes.append("Theory scatter skipped (need ≥2 sparse rows with `theory_active_fraction_global` in CSV).")

    # Relative paths from md_out directory to plot files
    md_parent = md_out.parent
    plot_keys = {
        "01_throughput_vs_step": os.path.relpath(p01, start=md_parent),
        "02_tail_throughput_sparsity_mask": os.path.relpath(p02, start=md_parent),
        "03_ratio_vs_dense": os.path.relpath(p03, start=md_parent),
    }
    if p04 is not None and p04.is_file():
        plot_keys["04_wall_breakdown"] = os.path.relpath(p04, start=md_parent)
    if p05.is_file():
        plot_keys["05_theory_vs_metric"] = os.path.relpath(p05, start=md_parent)

    write_markdown(md_out, run_dir, tp_col, tp_label, ptable, plot_keys, notes)
    print(f"Wrote {md_out}", file=sys.stderr)
    print(f"Wrote plots under {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
