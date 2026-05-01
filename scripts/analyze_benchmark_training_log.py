#!/usr/bin/env python3
"""
Summarize benchmark_training_log.csv from h200_sparse_dpo_bsr_benchmark.py.

Usage (on login node or laptop):
  python scripts/analyze_benchmark_training_log.py /path/to/benchmark_training_log.csv

Checks per-phase wall_time_s / throughput and, when t_* columns exist, warns that
t_forward_ms and t_backward_ms are last-micro-batch only when grad_accum > 1.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _num(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_path", type=Path)
    p.add_argument(
        "--grad-accum",
        type=int,
        default=64,
        help="Expected DPO gradient_accumulation_steps (for caveat messages)",
    )
    args = p.parse_args()
    path = args.csv_path
    if not path.is_file():
        print(f"ERROR: not a file: {path}", file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("ERROR: empty CSV", file=sys.stderr)
        sys.exit(1)

    fields = rows[0].keys()
    has_t_step = "t_step_total_ms" in fields
    has_t_fwd = "t_forward_ms" in fields
    has_t_bwd = "t_backward_ms" in fields
    if has_t_fwd or has_t_bwd:
        if args.grad_accum > 1:
            print(
                f"Note: t_forward_ms / t_backward_ms are last micro-batch only; "
                f"grad_accum={args.grad_accum} → do not multiply by grad_accum to match t_step_total_ms.\n"
            )

    by_phase: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        ph = r.get("phase") or "(unknown)"
        by_phase[str(ph)].append(r)

    print(f"File: {path}  rows={len(rows)}  phases={len(by_phase)}\n")

    for phase in sorted(by_phase.keys()):
        pr = by_phase[phase]
        walls = [_num(r.get("wall_time_s")) for r in pr]
        walls = [w for w in walls if w is not None]
        sps = [_num(r.get("cumulative_steps_per_s")) for r in pr]
        sps = [x for x in sps if x is not None]

        last_wall = walls[-1] if walls else None
        mean_sps = statistics.mean(sps) if sps else None

        line = f"phase={phase!r}  log_rows={len(pr)}"
        if last_wall is not None:
            line += f"  final_wall_time_s={last_wall:.3f}"
        if mean_sps is not None and not math.isnan(mean_sps):
            line += f"  mean_cumulative_steps_per_s={mean_sps:.6f}"
        print(line)

        if has_t_step:
            steps_ms = [_num(r.get("t_step_total_ms")) for r in pr]
            steps_ms = [x for x in steps_ms if x is not None]
            if steps_ms:
                print(
                    f"    t_step_total_ms: min={min(steps_ms):.1f} max={max(steps_ms):.1f} "
                    f"mean={statistics.mean(steps_ms):.1f} (ms)"
                )
            others = [_num(r.get("t_other_ms")) for r in pr]
            others = [x for x in others if x is not None]
            if others and max(others) > 1e5:
                print(
                    f"    WARNING: large t_other_ms (max={max(others):.0f} ms) often means "
                    f"t_fwd/t_bwd do not cover full optimizer step or detailed timing was misleading."
                )
        print()


if __name__ == "__main__":
    main()
