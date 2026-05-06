#!/usr/bin/env python3
"""
Re-render a readable optimizer_step_microbench.md from an existing CSV.

Use this to clean up older runs where the MD accidentally contained literal '\\n' sequences.
This does NOT require rerunning the GPU microbench.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _get(rows: List[Dict[str, str]], opt: str) -> Optional[Dict[str, str]]:
    for r in rows:
        if r.get("optimizer") == opt:
            return r
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"CSV empty: {csv_path}")

    out_path = (args.out or (csv_path.parent / "optimizer_step_microbench.md")).resolve()

    # header fields (best-effort)
    first = rows[0]
    mask_label = first.get("mask_label", "")
    mask_path = first.get("mask_path", "")
    steps_total = first.get("steps_total", "")
    trim_frac = first.get("trim_frac", "")

    dense = _get(rows, "adamw_torch")
    dense8 = _get(rows, "adamw_8bit")
    sparse = _get(rows, "sparse_adamw")

    lines: List[str] = []
    lines.append("# SparseAdamW optimizer.step() microbench")
    lines.append("")
    if mask_label:
        lines.append(f"- **mask_label:** `{mask_label}`")
    if mask_path:
        lines.append(f"- **mask_path:** `{mask_path}`")
    if steps_total:
        lines.append(f"- **steps_total:** `{steps_total}`  **trim_frac:** `{trim_frac}`")
    lines.append("")

    lines.append("## Timing summary (trimmed mid-window)")
    lines.append("")
    lines.append("| case | optimizer | tensors | total_numel | active_frac | mean_ms_mid | p50_ms_mid | note |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| `{r.get('case','')}` | `{r.get('optimizer','')}` | {r.get('tensors_used','')} | "
            f"{r.get('total_numel','')} | {r.get('active_frac','')} | {r.get('mean_ms_mid','')} | "
            f"{r.get('p50_ms_mid','')} | {r.get('note','')} |"
        )

    if sparse is not None:
        s = _f(sparse.get("mean_ms_mid"))
        lines.append("")
        lines.append("## Key speedups (trimmed mean)")
        lines.append("")
        if dense is not None:
            d = _f(dense.get("mean_ms_mid"))
            if d == d and s == s and s > 0:
                lines.append(f"- **SparseAdamW vs torch AdamW:** x{(d/s):.3f} faster (lower is better).")
        if dense8 is not None:
            d8 = _f(dense8.get("mean_ms_mid"))
            if d8 == d8 and s == s and s > 0:
                lines.append(f"- **SparseAdamW vs AdamW 8-bit:** x{(d8/s):.3f} faster (lower is better).")

    lines.append("")
    lines.append("## Memory / traffic estimates (subset only)")
    lines.append("")
    lines.append("| case | est_param_MB | est_grad_MB | est_adam_state_MB_dense | est_adam_state_MB_sparse | traffic_proxy_MB |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        def mb(key: str) -> str:
            v = _f(r.get(key))
            return "" if not (v == v) else f"{v/1e6:.1f}"

        lines.append(
            f"| `{r.get('case','')}` | {mb('est_param_bytes')} | {mb('est_grad_bytes')} | "
            f"{mb('est_adam_state_bytes_fp32_dense')} | {mb('est_adam_state_bytes_fp32_sparse')} | "
            f"{mb('est_sparseadamw_traffic_bytes_proxy')} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

