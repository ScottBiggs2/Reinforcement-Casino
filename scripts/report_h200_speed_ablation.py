#!/usr/bin/env python3
"""
Post-hoc report for H200 BSR speed ablation runs.

Reads:
  - Slurm stdout (``BENCH_JSON`` lines, ``[throughput]`` lines, ``TIME LIMIT`` / ``CANCELLED``)
  - ``benchmark_training_log.csv`` under ``--run-dir``
  - ``benchmark_theory.json`` (optional)

Writes ``speed_ablation_report.md`` and ``speed_ablation_report.json`` under ``--run-dir`` by default.

Usage::

  python3 scripts/report_h200_speed_ablation.py \\
    --run-dir /scratch/USER/rl_casino_h200_bsr/JOB \\
    --slurm-out /path/to/repo/logs/h200_speed_ablat_12345.out
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_bench_json_lines(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return (json_objects, warning_lines)."""
    out: List[Dict[str, Any]] = []
    warns: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("BENCH_JSON "):
            continue
        raw = s[len("BENCH_JSON ") :].strip()
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError as e:
            warns.append(f"bad BENCH_JSON: {e}: {raw[:120]}")
    return out, warns


def parse_throughput_lines(text: str) -> List[Dict[str, Any]]:
    pat = re.compile(
        r"^\[throughput\] phase=(?P<phase>\S+)\s+sparsity=(?P<sp>\S+)\s+step=(?P<step>\d+)\s+"
        r"steps/s=(?P<sps>[\d.]+)\s+samples/s=(?P<sms>[\d.]+)\s+wall_s=(?P<wall>[\d.]+)"
        r"(?:\s+steps/s_excl_first=(?P<ex>[\d.]+))?"
    )
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        d = m.groupdict()
        rows.append(
            {
                "phase": d["phase"],
                "sparsity": d["sp"],
                "step": int(d["step"]),
                "steps_per_s": float(d["sps"]),
                "samples_per_s": float(d["sms"]),
                "wall_s": float(d["wall"]),
                "steps_per_s_excl_first": float(d["ex"]) if d.get("ex") else None,
            }
        )
    return rows


def slurm_flags(text: str) -> Dict[str, bool]:
    t = text.upper()
    return {
        "time_limit": "TIME LIMIT" in t or "DUE TO TIME LIMIT" in t,
        "cancelled": "CANCELLED" in t or "CANCELLED AT" in t,
    }


def _to_float(x: Any) -> float:
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tail_mean(vals: List[float], n: int = 8) -> float:
    finite = [v for v in vals if not math.isnan(v)]
    if not finite:
        return float("nan")
    tail = finite[-n:]
    return sum(tail) / len(tail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Benchmark output_dir (contains benchmark_training_log.csv).",
    )
    ap.add_argument(
        "--slurm-out",
        type=Path,
        default=None,
        help="Slurm .out file (BENCH_JSON + [throughput] + time limit detection).",
    )
    ap.add_argument(
        "--theory-json",
        type=Path,
        default=None,
        help="Default: <run-dir>/benchmark_theory.json",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Default: <run-dir>/speed_ablation_report.md",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Default: <run-dir>/speed_ablation_report.json",
    )
    args = ap.parse_args()

    run_dir: Path = args.run_dir.resolve()
    csv_path = run_dir / "benchmark_training_log.csv"
    theory_path = (args.theory_json or (run_dir / "benchmark_theory.json")).resolve()
    out_md = (args.out_md or (run_dir / "speed_ablation_report.md")).resolve()
    out_json = (args.out_json or (run_dir / "speed_ablation_report.json")).resolve()

    slurm_text = ""
    if args.slurm_out and args.slurm_out.is_file():
        slurm_text = _read_text(args.slurm_out.resolve())
    bench_rows, bench_warns = parse_bench_json_lines(slurm_text)
    thr_rows = parse_throughput_lines(slurm_text)
    flags = slurm_flags(slurm_text)

    csv_rows = load_csv_rows(csv_path)
    theory: Any = None
    if theory_path.is_file():
        try:
            theory = json.loads(_read_text(theory_path))
        except json.JSONDecodeError:
            theory = None

    # Group BENCH_JSON by phase
    mask_by_phase: Dict[str, Dict[str, Any]] = {}
    train_wall_by_phase: Dict[str, Dict[str, Any]] = {}
    for b in bench_rows:
        k = b.get("kind")
        ph = str(b.get("phase", ""))
        if k == "mask" and ph:
            mask_by_phase[ph] = b
        elif k == "train_wall" and ph:
            train_wall_by_phase[ph] = b

    # Per-phase CSV aggregates
    by_phase: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in csv_rows:
        ph = r.get("phase", "")
        if ph:
            by_phase[ph].append(r)

    phase_summaries: List[Dict[str, Any]] = []
    for phase, rows in sorted(by_phase.items()):
        if not rows:
            continue
        steps = [_to_float(r.get("step")) for r in rows]
        max_step = max((int(s) for s in steps if not math.isnan(s)), default=0)
        tgt = None
        for r in rows:
            v = r.get("benchmark_phase_target_steps", "")
            if v not in ("", None):
                try:
                    tgt = int(float(v))
                except (TypeError, ValueError):
                    tgt = None
                break
        inst = [_to_float(r.get("inst_steps_per_s")) for r in rows]
        excl = [_to_float(r.get("cumulative_steps_per_s_excl_first")) for r in rows]
        cum = [_to_float(r.get("cumulative_steps_per_s")) for r in rows]
        phase_summaries.append(
            {
                "phase": phase,
                "logged_rows": len(rows),
                "max_step": max_step,
                "benchmark_phase_target_steps": tgt,
                "complete": tgt is None or max_step >= tgt,
                "tail_mean_inst_steps_per_s": round(tail_mean(inst), 8),
                "tail_mean_cumulative_steps_per_s_excl_first": round(tail_mean(excl), 8),
                "tail_mean_cumulative_steps_per_s": round(tail_mean(cum), 8),
                "mask_bench": mask_by_phase.get(phase),
                "train_wall_bench": train_wall_by_phase.get(phase),
            }
        )

    dense_tail = float("nan")
    for s in phase_summaries:
        if s["phase"] == "dense":
            dense_tail = float(s["tail_mean_inst_steps_per_s"])
            break

    for s in phase_summaries:
        t = s["tail_mean_inst_steps_per_s"]
        if s["phase"] != "dense" and not math.isnan(dense_tail) and dense_tail > 0 and not math.isnan(t):
            s["ratio_vs_dense_tail_inst"] = round(t / dense_tail, 6)
        else:
            s["ratio_vs_dense_tail_inst"] = None

    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "csv_path": str(csv_path),
        "csv_exists": csv_path.is_file(),
        "theory_json_path": str(theory_path),
        "theory_loaded": theory is not None,
        "slurm_out": str(args.slurm_out) if args.slurm_out else None,
        "slurm_flags": flags,
        "bench_json_parse_warnings": bench_warns,
        "throughput_lines_parsed": len(thr_rows),
        "phase_summaries": phase_summaries,
        "theory_records": theory,
    }

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# H200 speed ablation report (auto-generated)\n")
    lines.append(f"- **run_dir:** `{run_dir}`\n")
    lines.append(f"- **CSV present:** {csv_path.is_file()}\n")
    lines.append(f"- **Theory JSON present:** {theory_path.is_file()}\n")
    if args.slurm_out:
        lines.append(f"- **Slurm .out:** `{args.slurm_out}`\n")
    lines.append(f"- **Slurm time-limit string seen:** {flags['time_limit']}\n")
    lines.append(f"- **Slurm cancelled string seen:** {flags['cancelled']}\n")
    if bench_warns:
        lines.append("\n## Parse warnings\n")
        for w in bench_warns:
            lines.append(f"- {w}\n")

    lines.append("\n## Per-phase summary\n")
    lines.append(
        "| phase | complete | max_step | target_steps | tail_mean_inst_steps/s | "
        "tail_mean_steps/s_excl_first | mask_s | prepare_s | trainer_s | vs_dense_inst |\n"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for s in phase_summaries:
        mb = s.get("mask_bench") or {}
        tw = s.get("train_wall_bench") or {}
        m_s = mb.get("seconds", "")
        p_s = tw.get("prepare_s", "")
        t_s = tw.get("trainer_s", "")
        rv = s.get("ratio_vs_dense_tail_inst")
        rv_s = "" if rv is None else f"{rv:.4f}"
        tgt = s.get("benchmark_phase_target_steps")
        tgt_s = "" if tgt is None else str(tgt)
        lines.append(
            f"| `{s['phase']}` | {s['complete']} | {s['max_step']} | {tgt_s} | "
            f"{s['tail_mean_inst_steps_per_s']:.6g} | {s['tail_mean_cumulative_steps_per_s_excl_first']:.6g} | "
            f"{m_s} | {p_s} | {t_s} | {rv_s} |\n"
        )

    lines.append("\n## Raw BENCH_JSON (chronological)\n")
    for b in bench_rows:
        lines.append(f"- `{json.dumps(b, separators=(',', ':'))}`\n")

    lines.append("\n## Parsed [throughput] lines\n")
    for t in thr_rows[-40:]:
        lines.append(f"- `{t}`\n")

    out_md.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}", file=sys.stderr)
    print(f"Wrote {out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
