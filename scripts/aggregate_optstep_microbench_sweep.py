#!/usr/bin/env python3
"""
Aggregate independent ``optimizer.step()`` microbench shards into one sparsity-vs-time table.

Walks ``<root>/<JOBID>/{elem,block}/optimizer_step_microbench.csv`` (one jobid per chain),
extracts the target sparsity from each row's ``mask_path`` (filename tag like
``s99p75_element_b16_mlp0_floor0.0025.pt`` → 99.75), and writes:

  <out_dir>/optstep_microbench_long.csv      # one row per (jobid, mask_label, optimizer)
  <out_dir>/optstep_microbench_summary.md    # human-readable per-mask-label tables + speedups
  <out_dir>/optstep_microbench_mean_ms.png   # mean_ms_mid vs sparsity (matplotlib, optional)

Usage:
  python scripts/aggregate_optstep_microbench_sweep.py \\
      --root /scratch/$USER/rl_casino_optstep_microbench \\
      --out-dir /scratch/$USER/rl_casino_optstep_microbench/aggregate
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Filenames look like ``s50p0_element_b16_mlp0_floor0.0025.pt`` / ``s99p75_block_...pt``.
_SPARSITY_RE = re.compile(r"(?:^|/)s(\d+(?:p\d+)?)_(element|block)_b(\d+)")


def _parse_mask_path(p: str) -> Tuple[Optional[float], Optional[str], Optional[int]]:
    if not p:
        return None, None, None
    m = _SPARSITY_RE.search(p)
    if not m:
        return None, None, None
    sp = m.group(1).replace("p", ".")
    try:
        sp_f = float(sp)
    except Exception:
        return None, None, None
    return sp_f, m.group(2), int(m.group(3))


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _is_finite(x: float) -> bool:
    return isinstance(x, float) and x == x and not math.isinf(x)


def _walk_csvs(root: Path, only_jobids: Optional[List[str]]) -> List[Path]:
    """Find <root>/<jobid>/{elem,block}/optimizer_step_microbench.csv shards."""
    pats: List[str] = []
    if only_jobids:
        for j in only_jobids:
            pats.append(str(root / j / "elem" / "optimizer_step_microbench.csv"))
            pats.append(str(root / j / "block" / "optimizer_step_microbench.csv"))
    else:
        pats.append(str(root / "*" / "elem" / "optimizer_step_microbench.csv"))
        pats.append(str(root / "*" / "block" / "optimizer_step_microbench.csv"))
    found: List[Path] = []
    seen: set[str] = set()
    for pat in pats:
        for hit in glob.glob(pat):
            if hit not in seen:
                seen.add(hit)
                found.append(Path(hit))
    return sorted(found)


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _jobid_from_path(p: Path) -> str:
    # .../<root>/<JOBID>/<mask_label>/optimizer_step_microbench.csv
    return p.parent.parent.name


def _aggregate(rows_in: Iterable[Tuple[Path, Dict[str, str]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for csv_path, r in rows_in:
        opt = r.get("optimizer", "")
        case = r.get("case", "")
        mask_label_csv = r.get("mask_label", "")
        mask_path = r.get("mask_path", "")
        sp, mtype_from_path, bsr = _parse_mask_path(mask_path)
        # Prefer mask label embedded in row when available.
        mlabel = mask_label_csv or mtype_from_path or ""
        out.append(
            {
                "jobid": _jobid_from_path(csv_path),
                "csv_path": str(csv_path),
                "case": case,
                "optimizer": opt,
                "mask_label": mlabel,
                "sparsity_pct": sp,
                "mask_type_from_path": mtype_from_path,
                "block_size_bsr": bsr,
                "mask_path": mask_path,
                "tensors_used": int(_f(r.get("tensors_used")) or 0) if r.get("tensors_used") else None,
                "total_numel": int(_f(r.get("total_numel")) or 0) if r.get("total_numel") else None,
                "active_numel": int(_f(r.get("active_numel")) or 0) if r.get("active_numel") else None,
                "active_frac": _f(r.get("active_frac")),
                "mean_ms_mid": _f(r.get("mean_ms_mid")),
                "p50_ms_mid": _f(r.get("p50_ms_mid")),
                "mean_ms_all": _f(r.get("mean_ms_all")),
                "p50_ms_all": _f(r.get("p50_ms_all")),
                "steps_total": int(_f(r.get("steps_total")) or 0) if r.get("steps_total") else None,
                "trim_frac": _f(r.get("trim_frac")),
                "note": r.get("note", ""),
            }
        )
    return out


def _write_long_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "jobid",
        "mask_label",
        "sparsity_pct",
        "optimizer",
        "mean_ms_mid",
        "p50_ms_mid",
        "mean_ms_all",
        "p50_ms_all",
        "active_frac",
        "tensors_used",
        "total_numel",
        "active_numel",
        "steps_total",
        "trim_frac",
        "case",
        "mask_path",
        "csv_path",
        "block_size_bsr",
        "mask_type_from_path",
        "note",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _pivot_by_optimizer(rows: List[Dict[str, Any]], mask_label: str, metric: str) -> Tuple[List[float], Dict[str, Dict[float, float]]]:
    """Returns (sorted_sparsities, {optimizer: {sparsity: value}})."""
    series: Dict[str, Dict[float, float]] = {}
    sparsities: set[float] = set()
    for r in rows:
        if (r.get("mask_label") or "").lower() != mask_label.lower():
            continue
        sp = r.get("sparsity_pct")
        opt = r.get("optimizer") or ""
        v = _f(r.get(metric))
        if sp is None or not _is_finite(v) or not opt:
            continue
        sparsities.add(float(sp))
        series.setdefault(opt, {})[float(sp)] = v
    return sorted(sparsities), series


def _fmt_ms(v: float) -> str:
    return "" if not _is_finite(v) else f"{v:.4f}"


def _md_table_for_mask(rows: List[Dict[str, Any]], mask_label: str) -> List[str]:
    sparsities, series = _pivot_by_optimizer(rows, mask_label, "mean_ms_mid")
    if not sparsities:
        return []
    optimizers = sorted(series.keys())
    lines: List[str] = []
    lines.append(f"### mask_label = `{mask_label}` — `mean_ms_mid` (lower is better)")
    lines.append("")
    header = "| sparsity_pct | " + " | ".join(f"`{o}`" for o in optimizers) + " | speedup vs `adamw_torch` | speedup vs `adamw_8bit` |"
    sep = "|---:|" + "|".join(["---:"] * (len(optimizers) + 2)) + "|"
    lines.append(header)
    lines.append(sep)
    for sp in sparsities:
        cells = [f"{sp:g}"]
        for opt in optimizers:
            cells.append(_fmt_ms(series.get(opt, {}).get(sp, float("nan"))))
        s_v = series.get("sparse_adamw", {}).get(sp, float("nan"))
        d_t = series.get("adamw_torch", {}).get(sp, float("nan"))
        d_8 = series.get("adamw_8bit", {}).get(sp, float("nan"))
        sp_t = (d_t / s_v) if (_is_finite(s_v) and _is_finite(d_t) and s_v > 0) else float("nan")
        sp_8 = (d_8 / s_v) if (_is_finite(s_v) and _is_finite(d_8) and s_v > 0) else float("nan")
        cells.append("" if not _is_finite(sp_t) else f"x{sp_t:.3f}")
        cells.append("" if not _is_finite(sp_8) else f"x{sp_8:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _write_summary_md(rows: List[Dict[str, Any]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    mask_labels = sorted({(r.get("mask_label") or "") for r in rows if r.get("mask_label")})
    lines: List[str] = []
    lines.append("# optimizer.step() microbench sweep — sparsity aggregate")
    lines.append("")
    n_jobs = len({r.get("jobid") for r in rows})
    lines.append(f"- shards aggregated: **{n_jobs}** jobid(s), **{len(rows)}** CSV rows")
    if rows:
        st = next((r.get("steps_total") for r in rows if r.get("steps_total")), None)
        tf = next((r.get("trim_frac") for r in rows if _is_finite(_f(r.get("trim_frac")))), None)
        if st:
            lines.append(f"- microbench config (first row seen): `steps_total={st}`  `trim_frac={tf}`")
    lines.append("")
    for ml in mask_labels:
        block = _md_table_for_mask(rows, ml)
        if block:
            lines.extend(block)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _maybe_plot(rows: List[Dict[str, Any]], out_png: Path) -> Optional[str]:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        return f"(matplotlib unavailable, skipped plot: {exc})"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    mask_labels = sorted({(r.get("mask_label") or "") for r in rows if r.get("mask_label")})
    if not mask_labels:
        return "(no mask_label rows; skipped plot)"
    fig, axes = plt.subplots(1, len(mask_labels), figsize=(6 * len(mask_labels), 5), squeeze=False)
    for col, ml in enumerate(mask_labels):
        ax = axes[0][col]
        sparsities, series = _pivot_by_optimizer(rows, ml, "mean_ms_mid")
        for opt in sorted(series.keys()):
            xs = sparsities
            ys = [series[opt].get(s, float("nan")) for s in xs]
            ax.plot(xs, ys, marker="o", label=opt)
        ax.set_xlabel("target sparsity (%)")
        ax.set_ylabel("mean_ms_mid (ms / step)")
        ax.set_title(f"mask_label = {ml}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="OUT_BASE root containing <JOBID>/{elem,block}/...")
    ap.add_argument("--out-dir", type=Path, default=None, help="Where to write aggregate artifacts (default: <root>/aggregate)")
    ap.add_argument("--jobid", action="append", default=None, help="Restrict to specific JOBID dirs (repeatable). Default: all.")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: --root not a directory: {root}", file=sys.stderr)
        return 2
    out_dir = (args.out_dir or (root / "aggregate")).resolve()

    csvs = _walk_csvs(root, args.jobid)
    if not csvs:
        print(f"ERROR: no shard CSVs under {root}", file=sys.stderr)
        return 2

    pairs: List[Tuple[Path, Dict[str, str]]] = []
    for csv_path in csvs:
        for r in _read_rows(csv_path):
            pairs.append((csv_path, r))

    rows = _aggregate(pairs)
    out_csv = out_dir / "optstep_microbench_long.csv"
    out_md = out_dir / "optstep_microbench_summary.md"
    out_png = out_dir / "optstep_microbench_mean_ms.png"

    _write_long_csv(rows, out_csv)
    _write_summary_md(rows, out_md)
    plot_msg = _maybe_plot(rows, out_png)

    print(f"Aggregated {len(csvs)} shard CSV(s) → {len(rows)} rows")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    if plot_msg:
        print(plot_msg)
    else:
        print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
