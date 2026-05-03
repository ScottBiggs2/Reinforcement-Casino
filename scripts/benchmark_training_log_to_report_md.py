#!/usr/bin/env python3
"""
Build a standalone Markdown report from ``benchmark_training_log.csv``.

Includes:
  - Explicit **dense baseline** row when the CSV contains a ``dense`` phase (steps/s, samples/s,
    and **relative throughput** vs that baseline for every other phase).
  - When no dense phase exists, states that clearly and still lists sparse phases (no fake ratios).
  - **Theory** columns (mask-sidecar FLOP proxy, active fraction, token proxy) whenever present.
  - **CUDA segment breakdown** (step / forward / backward / optim / non-optim / other ms) when
    ``RL_CASINO_BSR_DETAILED_TIMING=1`` populated those columns; otherwise a short note.

Usage (repo root or anywhere):

  python3 scripts/benchmark_training_log_to_report_md.py \\
    --csv /scratch/USER/rl_casino_h200_bsr/JOBID/benchmark_training_log.csv \\
    --out benchmark_h200_report.md

  python3 scripts/benchmark_training_log_to_report_md.py --csv run.csv --tail-rows 8
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _phase_sort_key(phase: str) -> Tuple[Any, ...]:
    if phase == "dense":
        return (0, 0.0, 0, 0, phase)
    m = re.match(r"^s([\dp]+)_(elem|blk)_gi([^_]+)_(block_[12]d)$", phase)
    if not m:
        return (1, 0.0, 99, 99, phase)
    sp_raw = m.group(1).replace("p", ".")
    try:
        sp = float(sp_raw)
    except ValueError:
        sp = 0.0
    mask_ord = 0 if m.group(2) == "elem" else 1
    adam_ord = 0 if m.group(4) == "block_1d" else 1
    return (1, -sp, mask_ord, adam_ord, phase)


def _to_float(x: Any) -> float:
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _mean(vals: List[float]) -> float:
    finite = [v for v in vals if not math.isnan(v)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _fmt(x: float, nd: int = 4, empty: str = "—") -> str:
    if x is None or math.isnan(x) or math.isinf(x):
        return empty
    return f"{float(x):.{nd}f}"


def _fmt_sci(x: float) -> str:
    if x is None or math.isnan(x) or math.isinf(x) or x <= 0:
        return "—"
    return f"{float(x):.4e}"


def _fmt_pct_ratio(x: float) -> str:
    """e.g. 0.42 -> '42.0% of baseline' ; >1 -> '124% of baseline'."""
    if x is None or math.isnan(x) or math.isinf(x) or x <= 0:
        return "—"
    return f"{100.0 * x:.1f}% of baseline"


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    raw = path.read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines(keepends=True) if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(StringIO("".join(lines))))


def aggregate_phases(rows: List[Dict[str, str]], tail_rows: int) -> List[Dict[str, Any]]:
    by_phase: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        ph = (r.get("phase") or "").strip()
        if not ph:
            continue
        by_phase[ph].append(r)

    phases_sorted = sorted(by_phase.keys(), key=_phase_sort_key)
    out: List[Dict[str, Any]] = []
    numeric_cols = (
        "wall_time_s",
        "cumulative_steps_per_s",
        "cumulative_samples_per_s",
        "t_step_total_ms",
        "t_forward_ms",
        "t_backward_ms",
        "t_optim_ms",
        "t_nonoptim_ms",
        "t_other_ms",
        "eff_bsr_backward_tflops",
        "eff_bsr_backward_tflops_over_e2e_step",
        "theory_bsr_backward_flops_proxy",
        "theory_active_fraction_global",
        "theory_b_tokens_proxy",
    )

    for phase in phases_sorted:
        g = by_phase[phase]
        g.sort(key=lambda x: int(float(x.get("step") or 0)))
        tail = g[-max(1, tail_rows) :]
        row: Dict[str, Any] = {"phase": phase}
        last = tail[-1]
        sp = last.get("sparsity_target_pct", "").strip()
        if sp == "":
            row["sparsity_target_pct"] = ""
        else:
            row["sparsity_target_pct"] = _to_float(sp)
        row["optimizer"] = (last.get("optimizer") or "").strip()

        for col in numeric_cols:
            row[col] = _mean([_to_float(x.get(col)) for x in tail])

        try:
            row["last_logged_step"] = int(float(tail[-1].get("step") or 0))
        except (TypeError, ValueError):
            row["last_logged_step"] = 0
        try:
            row["trainer_grad_accum_steps"] = int(float(tail[-1].get("trainer_grad_accum_steps") or 0))
        except (TypeError, ValueError):
            row["trainer_grad_accum_steps"] = 0
        try:
            row["trainer_per_device_train_batch_size"] = int(
                float(tail[-1].get("trainer_per_device_train_batch_size") or 0)
            )
        except (TypeError, ValueError):
            row["trainer_per_device_train_batch_size"] = 0

        out.append(row)
    return out


def _has_cuda_timing(agg: List[Dict[str, Any]]) -> bool:
    for r in agg:
        v = float(r.get("t_step_total_ms", float("nan")))
        if not math.isnan(v) and v > 0:
            return True
    return False


def _has_theory(agg: List[Dict[str, Any]]) -> bool:
    for r in agg:
        if str(r.get("phase")) == "dense":
            continue
        v = float(r.get("theory_bsr_backward_flops_proxy", float("nan")))
        if not math.isnan(v) and v > 0:
            return True
    return False


def build_markdown(
    *,
    csv_path: Path,
    agg: List[Dict[str, Any]],
    tail_rows: int,
) -> str:
    lines: List[str] = []
    lines.append(f"# Benchmark report: `{csv_path.name}`")
    lines.append("")
    lines.append(f"- **Source:** `{csv_path.resolve()}`")
    lines.append(f"- **Aggregation:** mean over the last **{tail_rows}** logged rows per `phase`.")
    lines.append("")

    dense = next((r for r in agg if r["phase"] == "dense"), None)
    if dense:
        ds = float(dense["cumulative_steps_per_s"])
        dm = float(dense["cumulative_samples_per_s"])
        lines.append("## Baseline (dense phase in this CSV)")
        lines.append("")
        lines.append(
            "This run includes a **dense** DPO phase (`nn.Linear` + dense optimizer). "
            "All sparse rows below are expressed **relative to that same job’s dense tail mean**."
        )
        lines.append("")
        lines.append("| Metric | Dense baseline (tail mean) |")
        lines.append("|--------|------------------------------|")
        lines.append(f"| Optimizer steps / s | {_fmt(ds, 6)} |")
        lines.append(f"| Samples / s | {_fmt(dm, 4)} |")
        lines.append(f"| Wall time (tail mean, s) | {_fmt(float(dense['wall_time_s']), 2)} |")
        lines.append("")
    else:
        lines.append("## Baseline")
        lines.append("")
        lines.append(
            "**No `dense` phase appears in this CSV** (typical for sparse-only throughput sweeps). "
            "End-to-end **steps/s** and **samples/s** are absolute; there is **no in-run dense ratio**. "
            "To obtain “vs dense” numbers, either (a) rerun with a dense baseline phase included, or "
            "(b) join against a separate dense benchmark CSV taken under matched batch, sequence, and steps."
        )
        lines.append("")

    lines.append("## Throughput (all phases)")
    lines.append("")
    if dense and not math.isnan(float(dense["cumulative_steps_per_s"])) and float(dense["cumulative_steps_per_s"]) > 0:
        lines.append(
            "| Phase | Sparse % | Optimizer | Last step | microBS / accum | steps/s | samples/s | "
            "vs dense steps/s | vs dense samples/s | wall (s) |"
        )
        lines.append(
            "|-------|------------|-----------|-----------|-----------------|---------|-----------|-------------------|--------------------|----------|"
        )
        d_sps = float(dense["cumulative_steps_per_s"])
        d_sms = float(dense["cumulative_samples_per_s"])
    else:
        lines.append(
            "| Phase | Sparse % | Optimizer | Last step | microBS / accum | steps/s | samples/s | wall (s) |"
        )
        lines.append(
            "|-------|------------|-----------|-----------|-----------------|---------|-----------|----------|"
        )
        d_sps = d_sms = float("nan")

    for r in agg:
        ph = str(r["phase"])
        sp = r.get("sparsity_target_pct", "")
        if sp == "" or (isinstance(sp, float) and math.isnan(sp)):
            sp_s = ""
        else:
            sp_s = str(sp)
        mbs = int(r.get("trainer_per_device_train_batch_size") or 0)
        ga = int(r.get("trainer_grad_accum_steps") or 0)
        bs = f"{mbs}/{ga}" if mbs > 0 and ga > 0 else "—"
        sps = float(r["cumulative_steps_per_s"])
        sms = float(r["cumulative_samples_per_s"])
        row_cells = [
            f"`{ph}`",
            sp_s,
            str(r.get("optimizer", "")),
            str(int(r.get("last_logged_step") or 0)),
            bs,
            _fmt(sps, 6),
            _fmt(sms, 4),
        ]
        if dense and not math.isnan(d_sps) and d_sps > 0 and not math.isnan(d_sms) and d_sms > 0:
            rs = sps / d_sps if ph != "dense" else 1.0
            rm = sms / d_sms if ph != "dense" else 1.0
            row_cells.extend([_fmt_pct_ratio(rs), _fmt_pct_ratio(rm)])
        row_cells.append(_fmt(float(r["wall_time_s"]), 2))
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    if _has_theory(agg):
        lines.append("## Theory / accounting (mask sidecar, sparse phases)")
        lines.append("")
        lines.append(
            "**`theory_bsr_backward_flops_proxy`:** masked-linear backward FLOPs **per optimizer step** "
            "(see `src/utils/bsr_theory_metrics.py`). Does not include full dense-forward FLOPs."
        )
        lines.append("")
        lines.append(
            "| Phase | Sparse % | Theory BWD FLOP/step | active fraction | b-token proxy |"
        )
        lines.append("|-------|------------|----------------------|-----------------|----------------|")
        for r in agg:
            if r["phase"] == "dense":
                continue
            sp = r.get("sparsity_target_pct", "")
            sp_s = "" if sp == "" or (isinstance(sp, float) and math.isnan(sp)) else str(sp)
            th = float(r.get("theory_bsr_backward_flops_proxy", float("nan")))
            th_s = _fmt_sci(th) if not math.isnan(th) and th > 0 else "—"
            af = float(r.get("theory_active_fraction_global", float("nan")))
            af_s = _fmt(af, 6) if not math.isnan(af) else "—"
            bt = float(r.get("theory_b_tokens_proxy", float("nan")))
            bt_s = str(int(bt)) if not math.isnan(bt) and bt > 0 else "—"
            lines.append(
                "| `" + str(r["phase"]) + "` | " + sp_s + " | " + th_s + " | " + af_s + " | " + bt_s + " |"
            )
        lines.append("")

    has_tim = _has_cuda_timing(agg)
    lines.append("## CUDA segment timings (micro-batch–level)")
    lines.append("")
    if not has_tim:
        lines.append(
            "*No non-empty `t_step_total_ms` in the tail window — this CSV was almost certainly logged with "
            "**`RL_CASINO_BSR_DETAILED_TIMING=0`**. Forward/backward/optim ms and effective TFLOP/s from "
            "backward slices are therefore unavailable here.*"
        )
        lines.append("")
    else:
        lines.append(
            "Means over the same tail rows. **Caveat:** `t_forward_ms` / `t_backward_ms` refer to the "
            "**last gradient-accumulation micro-batch**, not the full optimizer step; interpret `t_other_ms` "
            "with care."
        )
        lines.append("")
        lines.append(
            "| Phase | t_step ms | t_fwd ms | t_bwd ms | t_optim ms | t_nonoptim ms | t_other ms | "
            "TFLOP/s (bwd) | TFLOP/s (step) |"
        )
        lines.append(
            "|-------|-----------|----------|----------|------------|---------------|------------|---------------|----------------|"
        )
        for r in agg:
            ph = str(r["phase"])
            tb = float(r.get("eff_bsr_backward_tflops", float("nan")))
            ts = float(r.get("eff_bsr_backward_tflops_over_e2e_step", float("nan")))
            tb_s = "—" if ph == "dense" or math.isnan(tb) else _fmt(tb, 4)
            ts_s = "—" if ph == "dense" or math.isnan(ts) else _fmt(ts, 4)
            lines.append(
                "| `"
                + ph
                + "` | "
                + _fmt(float(r["t_step_total_ms"]), 2)
                + " | "
                + _fmt(float(r["t_forward_ms"]), 2)
                + " | "
                + _fmt(float(r["t_backward_ms"]), 2)
                + " | "
                + _fmt(float(r["t_optim_ms"]), 2)
                + " | "
                + _fmt(float(r.get("t_nonoptim_ms", float("nan"))), 2)
                + " | "
                + _fmt(float(r.get("t_other_ms", float("nan"))), 2)
                + " | "
                + tb_s
                + " | "
                + ts_s
                + " |"
            )
        lines.append("")
        lines.append(
            "_Effective TFLOP/s:_ from CSV columns `eff_bsr_backward_tflops` and "
            "`eff_bsr_backward_tflops_over_e2e_step` (see `logging_utils.BenchmarkThroughputCallback`). "
            "Re-run `scripts/recalc_benchmark_training_log_eff.py` if older rows omitted grad-accum scaling."
        )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Generated by `scripts/benchmark_training_log_to_report_md.py`.*")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True, help="Path to benchmark_training_log.csv")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output Markdown path (default: <csv_stem>_report.md next to the CSV)",
    )
    ap.add_argument(
        "--tail-rows",
        type=int,
        default=8,
        help="Mean metrics over the last N rows per phase",
    )
    args = ap.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        print(f"ERROR: not a file: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_csv_rows(csv_path)
    if not rows:
        print("ERROR: CSV empty after stripping comments", file=sys.stderr)
        sys.exit(1)
    if "phase" not in rows[0] or "step" not in rows[0]:
        print("ERROR: CSV needs columns phase, step", file=sys.stderr)
        sys.exit(1)

    agg = aggregate_phases(rows, args.tail_rows)
    md = build_markdown(csv_path=csv_path, agg=agg, tail_rows=args.tail_rows)

    out_path = args.out
    if out_path is None:
        out_path = csv_path.parent / f"{csv_path.stem}_report.md"
    else:
        out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
