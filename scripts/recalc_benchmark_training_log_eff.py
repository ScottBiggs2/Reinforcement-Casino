#!/usr/bin/env python3
"""
Recompute eff_bsr_* columns in benchmark_training_log.csv without re-running training.

Older logs used:
  eff = theory_bsr_backward_flops_proxy / (t_backward_ms / 1e3)
but theory FLOPs are per **optimizer step** while t_backward_ms is one **micro-batch**
(last accum step). Correct scaling:
  eff = flops / ((t_backward_ms * grad_accum) / 1e3)

Usage:
  python scripts/recalc_benchmark_training_log_eff.py \\
    --csv /path/to/benchmark_training_log.csv \\
    --output /path/to/benchmark_training_log.fixed.csv

  # In-place (writes .bak then replaces)
  python scripts/recalc_benchmark_training_log_eff.py --csv path/to.csv --in-place

If the CSV has ``trainer_grad_accum_steps`` (newer runs), that value is used per row;
otherwise --grad-accum (default 64) applies to all rows.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from typing import Any, Dict, List, Optional


def _to_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _to_int(x: Any, default: int) -> int:
    if x is None or x == "":
        return default
    try:
        return max(1, int(float(x)))
    except (TypeError, ValueError):
        return default


def recalc_row(row: Dict[str, str], default_ga: int) -> Dict[str, str]:
    out = dict(row)
    flops = _to_float(row.get("theory_bsr_backward_flops_proxy"))
    ga = _to_int(row.get("trainer_grad_accum_steps"), default_ga)
    if not (row.get("trainer_grad_accum_steps") or "").strip():
        out["trainer_grad_accum_steps"] = str(ga)

    if flops is None or flops <= 0:
        return out
    bwd_ms = _to_float(row.get("t_backward_ms"))
    step_ms = _to_float(row.get("t_step_total_ms"))

    if bwd_ms is not None and bwd_ms > 0:
        bwd_step_ms = bwd_ms * float(ga)
        out["eff_bsr_backward_flops_per_s"] = f"{flops / (bwd_step_ms / 1e3):.6f}"
        out["eff_bsr_backward_tflops"] = f"{(flops / (bwd_step_ms / 1e3)) / 1e12:.6f}"
    if step_ms is not None and step_ms > 0:
        out["eff_bsr_backward_flops_per_s_over_e2e_step"] = f"{flops / (step_ms / 1e3):.6f}"
        out["eff_bsr_backward_tflops_over_e2e_step"] = f"{(flops / (step_ms / 1e3)) / 1e12:.6f}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="Input benchmark_training_log.csv")
    ap.add_argument("--output", default=None, help="Output path (default: stdout as CSV)")
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite input; keeps backup path.csv.bak (same dir)",
    )
    ap.add_argument(
        "--grad-accum",
        type=int,
        default=64,
        help="gradient_accumulation_steps if CSV lacks trainer_grad_accum_steps column",
    )
    args = ap.parse_args()

    path = os.path.abspath(args.csv)
    if not os.path.isfile(path):
        print(f"ERROR: not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print("ERROR: empty CSV", file=sys.stderr)
            sys.exit(1)
        rows: List[Dict[str, str]] = [recalc_row(r, args.grad_accum) for r in reader]

    # Preserve column order; ensure new trainer_* keys appear if missing
    extra_cols = []
    for c in ("trainer_grad_accum_steps", "trainer_per_device_train_batch_size"):
        if c not in fieldnames:
            extra_cols.append(c)
    fieldnames_out = list(fieldnames) + [c for c in extra_cols if c not in fieldnames]

    def write_to(fp) -> None:
        w = csv.DictWriter(fp, fieldnames=fieldnames_out, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames_out})

    if args.in_place:
        bak = path + ".bak"
        shutil.copy2(path, bak)
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as fp:
            write_to(fp)
        os.replace(tmp, path)
        print(f"Wrote {path} (backup {bak})", file=sys.stderr)
        return

    if args.output:
        outp = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
        with open(outp, "w", newline="", encoding="utf-8") as fp:
            write_to(fp)
        print(f"Wrote {outp}", file=sys.stderr)
        return

    write_to(sys.stdout)


if __name__ == "__main__":
    main()
