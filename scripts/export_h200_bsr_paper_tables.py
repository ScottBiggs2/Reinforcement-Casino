#!/usr/bin/env python3
"""
Aggregate benchmark_training_log.csv into Markdown + LaTeX snippets for paper_snippets.md.
Uses stdlib only (no pandas).

  python scripts/export_h200_bsr_paper_tables.py \\
    --csv /path/to/benchmark_training_log.csv \\
    --inject paper_snippets.md [--compact]

`--compact` emits Markdown throughput summary plus **one** LaTeX table (omit duplicate timing + appendix blocks).

Markers in paper_snippets.md (or ``hpc_paper_snippets.md``):

  <!-- H200_BSR_PAPER_EXPORT_START -->
  <!-- H200_BSR_PAPER_EXPORT_END -->
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


MARK_START = "<!-- H200_BSR_PAPER_EXPORT_START -->"
MARK_END = "<!-- H200_BSR_PAPER_EXPORT_END -->"


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

        nums = {}
        for col in (
            "wall_time_s",
            "wall_delta_s",
            "cumulative_steps_per_s",
            "cumulative_samples_per_s",
            "inst_steps_per_s",
            "inst_samples_per_s",
            "t_step_total_ms",
            "t_forward_ms",
            "t_backward_ms",
            "t_optim_ms",
            "eff_bsr_backward_tflops",
            "eff_bsr_backward_tflops_over_e2e_step",
            "theory_bsr_backward_flops_proxy",
            "theory_active_fraction_global",
            "theory_b_tokens_proxy",
        ):
            nums[col] = [_to_float(x.get(col)) for x in tail]
            row[col] = _mean(nums[col])

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


def _fmt_num(x: float, nd: int = 4) -> str:
    if x is None or math.isnan(x) or math.isinf(x):
        return "---"
    return f"{float(x):.{nd}f}"


def _fmt_theory_flops_per_step(x: float) -> str:
    """Accounting-scale FLOP counts per optimizer step (scientific notation)."""
    if x is None or math.isnan(x) or math.isinf(x) or x <= 0:
        return "---"
    return f"{float(x):.4e}"


def _latex_tt(s: str) -> str:
    return "\\texttt{" + str(s).replace("_", "\\_") + "}"


def build_markdown_summary(agg: List[Dict[str, Any]], csv_basename: str) -> str:
    lines = [
        f"<!-- Generated from `{csv_basename}` — re-run export after updating the CSV. -->",
        "",
        "### Aggregated throughput (mean of last logged rows per phase)",
        "",
        "| Phase | Sparse % | Optimizer | Last step | microBS / accum | Steps/s | Samples/s | Wall (s) | Theory BWD FLOP/step | TFLOP/s (bwd) | TFLOP/s (step) |",
        "|-------|----------|-----------|-----------|-----------------|---------|-----------|----------|----------------------|---------------|----------------|",
    ]
    for r in agg:
        ph = str(r["phase"])
        sp = r.get("sparsity_target_pct", "")
        if sp == "":
            sp_s = ""
        elif isinstance(sp, float) and math.isnan(sp):
            sp_s = ""
        else:
            sp_s = str(sp)
        mbs = int(r.get("trainer_per_device_train_batch_size") or 0)
        ga = int(r.get("trainer_grad_accum_steps") or 0)
        if mbs > 0 and ga > 0:
            bs_accum = f"{mbs}/{ga}"
        else:
            bs_accum = "---"
        tfl_b = float(r.get("eff_bsr_backward_tflops", float("nan")))
        tfl_b_s = "---" if ph == "dense" or math.isnan(tfl_b) else _fmt_num(tfl_b, 4)
        tfl_st = float(r.get("eff_bsr_backward_tflops_over_e2e_step", float("nan")))
        tfl_st_s = "---" if ph == "dense" or math.isnan(tfl_st) else _fmt_num(tfl_st, 4)
        th = float(r.get("theory_bsr_backward_flops_proxy", float("nan")))
        th_s = "---" if ph == "dense" or math.isnan(th) or th <= 0 else _fmt_theory_flops_per_step(th)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{ph}`",
                    sp_s,
                    str(r.get("optimizer", "")),
                    str(int(r.get("last_logged_step") or 0)),
                    bs_accum,
                    _fmt_num(float(r["cumulative_steps_per_s"]), 6),
                    _fmt_num(float(r["cumulative_samples_per_s"]), 4),
                    _fmt_num(float(r["wall_time_s"]), 2),
                    th_s,
                    tfl_b_s,
                    tfl_st_s,
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def build_latex_highlight_timing(agg: List[Dict[str, Any]], has_timing: bool) -> str:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        (
            r"  \caption{H200 BSR — mean step/component times (ms) per phase when \texttt{t\_*} "
            r"columns exist in CSV; otherwise use throughput table only.}"
        ),
        r"  \label{tab:bsr-timing-highlights-autogen}",
        r"  \begin{tabular}{@{}lcccccc@{}}",
        r"    \toprule",
        r"    Phase & Sparse (\%) & Optimizer & $t_{\mathrm{step}}$ & $t_{\mathrm{fwd}}$ & $t_{\mathrm{bwd}}$ & $t_{\mathrm{opt}}$ \\",
        r"    \midrule",
    ]
    want = {
        "dense",
        "s99p75_elem_gidense_block_1d",
        "s99p75_elem_gidense_block_2d",
        "s97p5_blk_gidense_block_1d",
        "s97p5_blk_gidense_block_2d",
    }
    sub = [r for r in agg if r["phase"] in want]
    if len(sub) < 2:
        sub = agg[: min(6, len(agg))]
    for r in sub:
        ph = _latex_tt(str(r["phase"]))
        sp = r.get("sparsity_target_pct", "")
        if sp == "" or (isinstance(sp, float) and math.isnan(sp)):
            sp_l = "---"
        else:
            sp_l = str(sp)
        opt = str(r["optimizer"]).replace("_", "\\_")
        if has_timing:
            ts = _fmt_num(float(r["t_step_total_ms"]), 2)
            tf = _fmt_num(float(r["t_forward_ms"]), 2)
            tb = _fmt_num(float(r["t_backward_ms"]), 2)
            to = _fmt_num(float(r["t_optim_ms"]), 2)
        else:
            ts, tf, tb, to = "---", "---", "---", "---"
        lines.append(f"    {ph} & {sp_l} & {opt} & {ts} & {tf} & {tb} & {to} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def build_latex_throughput(agg: List[Dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        (
            r"  \caption{H200 BSR --- throughput (tail-mean). "
            r"\emph{Theory} $\tilde F_{\mathrm{BSR\,bwd}}$: masked-linear backward FLOPs per optimizer step (mask sidecar; sparse phases only). "
            r"\emph{Timed} rates use CUDA segment timings when enabled: \textsubscript{bwd} scales the last micro-batch backward by grad accumulation; "
            r"\textsubscript{step} divides the same theory proxy by the measured full-step wall slice. Otherwise \texttt{---}.}"
        ),
        r"  \label{tab:bsr-throughput-autogen}",
        r"  \begin{tabular}{@{}lcccccc@{}}",
        r"    \toprule",
        r"    Phase & Steps/s & Samples/s & Last step & $\tilde F_{\mathrm{BSR\,bwd}}$ / step & "
        r"TFLOP/s\textsubscript{bwd} & TFLOP/s\textsubscript{step} \\",
        r"    \midrule",
    ]
    for r in agg:
        ph = _latex_tt(str(r["phase"]))
        th = float(r.get("theory_bsr_backward_flops_proxy", float("nan")))
        if str(r["phase"]) == "dense" or math.isnan(th) or th <= 0:
            th_s = "---"
        else:
            th_s = _fmt_theory_flops_per_step(th)
        tflo_b = float(r.get("eff_bsr_backward_tflops", float("nan")))
        tflo_b_s = "---" if str(r["phase"]) == "dense" or math.isnan(tflo_b) else _fmt_num(tflo_b, 4)
        tflo_st = float(r.get("eff_bsr_backward_tflops_over_e2e_step", float("nan")))
        tflo_st_s = "---" if str(r["phase"]) == "dense" or math.isnan(tflo_st) else _fmt_num(tflo_st, 4)
        try:
            last_s = str(int(r.get("last_logged_step") or 0))
        except (TypeError, ValueError):
            last_s = "---"
        lines.append(
            f"    {ph} & {_fmt_num(float(r['cumulative_steps_per_s']), 6)} & "
            f"{_fmt_num(float(r['cumulative_samples_per_s']), 4)} & {last_s} & {th_s} & {tflo_b_s} & {tflo_st_s} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def build_latex_appendix_grid(agg: List[Dict[str, Any]], has_timing: bool) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \footnotesize",
        r"  \caption{H200 BSR — all phases in CSV (partial runs omit unfinished sparsity levels).}",
        r"  \label{tab:bsr-appendix-autogen}",
        r"  \begin{tabular}{@{}lccccccccccc@{}}",
        r"    \toprule",
        r"    Phase tag & Sparse (\%) & Mask & GI & Kernel & $\bar{t}_{\mathrm{step}}$ & $\bar{t}_{\mathrm{fwd}}$ & "
        r"$\bar{t}_{\mathrm{bwd}}$ & $\bar{t}_{\mathrm{opt}}$ & $\tilde F_{\mathrm{bwd}}$/step & "
        r"TFLOP/s\textsubscript{bwd} & TFLOP/s\textsubscript{step} \\",
        r"    \midrule",
    ]
    for r in agg:
        ph = str(r["phase"])
        sp = r.get("sparsity_target_pct", "")
        if ph == "dense":
            mask, gi, kern = "---", "---", r"adamw\textsubscript{dense}"
            sp_l = "---"
        else:
            if sp == "" or (isinstance(sp, float) and math.isnan(sp)):
                sp_l = ""
            else:
                sp_l = str(sp)
            mask = "elem" if "_elem_" in ph else ("blk" if "_blk_" in ph else "---")
            gi = "dense" if "_gidense_" in ph else "---"
            m_adam = re.search(r"(block_[12]d)$", ph)
            kern = m_adam.group(1).replace("_", "\\_") if m_adam else "---"
        if has_timing:
            ts = _fmt_num(float(r["t_step_total_ms"]), 2)
            tf = _fmt_num(float(r["t_forward_ms"]), 2)
            tb = _fmt_num(float(r["t_backward_ms"]), 2)
            to = _fmt_num(float(r["t_optim_ms"]), 2)
        else:
            ts, tf, tb, to = "---", "---", "---", "---"
        th = float(r.get("theory_bsr_backward_flops_proxy", float("nan")))
        if ph == "dense" or math.isnan(th) or th <= 0:
            th_s = "---"
        else:
            th_s = _fmt_theory_flops_per_step(th)
        tflo_b = float(r.get("eff_bsr_backward_tflops", float("nan")))
        tflo_b_s = "---" if ph == "dense" or math.isnan(tflo_b) else _fmt_num(tflo_b, 4)
        tflo_s = float(r.get("eff_bsr_backward_tflops_over_e2e_step", float("nan")))
        tflo_s_s = "---" if ph == "dense" or math.isnan(tflo_s) else _fmt_num(tflo_s, 4)
        lines.append(
            f"    {_latex_tt(ph)} & {sp_l} & {mask} & {gi} & {kern} & {ts} & {tf} & {tb} & {to} & {th_s} & "
            f"{tflo_b_s} & {tflo_s_s} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def inject_into_paper(paper_path: Path, body: str) -> None:
    text = paper_path.read_text(encoding="utf-8")
    if MARK_START not in text or MARK_END not in text:
        print(
            f"ERROR: {paper_path} must contain {MARK_START!r} and {MARK_END!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    pre, rest = text.split(MARK_START, 1)
    _, post = rest.split(MARK_END, 1)
    new_text = pre + MARK_START + "\n\n" + body.rstrip() + "\n\n" + MARK_END + post
    paper_path.write_text(new_text, encoding="utf-8")
    print(f"Updated {paper_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, type=Path, help="benchmark_training_log.csv")
    ap.add_argument(
        "--inject",
        type=Path,
        default=None,
        help="paper_snippets.md to rewrite between export markers",
    )
    ap.add_argument(
        "--tail-rows",
        type=int,
        default=8,
        help="Mean metrics over the last N logged rows per phase",
    )
    ap.add_argument("--stdout-only", action="store_true", help="Print export body; do not inject")
    ap.add_argument(
        "--compact",
        action="store_true",
        help="Markdown + throughput LaTeX only (omit timing highlight + appendix LaTeX blocks)",
    )
    args = ap.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        print(f"ERROR: not a file: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_csv_rows(csv_path)
    if not rows:
        print("ERROR: CSV is empty after stripping comments", file=sys.stderr)
        sys.exit(1)
    if "phase" not in rows[0] or "step" not in rows[0]:
        print("ERROR: CSV needs columns phase, step", file=sys.stderr)
        sys.exit(1)

    agg = aggregate_phases(rows, args.tail_rows)
    ts_vals = [float(r["t_step_total_ms"]) for r in agg]
    has_timing = any(not math.isnan(v) and v > 0 for v in ts_vals)

    md_summary = build_markdown_summary(agg, csv_path.name)
    throughput_tex = build_latex_throughput(agg).strip()
    timing_tex = build_latex_highlight_timing(agg, has_timing).strip()
    appendix_tex = build_latex_appendix_grid(agg, has_timing).strip()

    body_parts = [
        "## Auto-filled from benchmark CSV (edit upstream CSV + re-export; do not hand-tune numbers here)",
        "",
    ]
    if args.compact:
        body_parts.extend(
            [
                "_Compact mode_ (`--compact`): Markdown summary and one throughput LaTeX block only. Omit `--compact` for timing-highlight + appendix LaTeX.",
                "",
                md_summary,
                "```latex",
                throughput_tex,
                "```",
                "",
            ]
        )
    else:
        body_parts.extend(
            [
                md_summary,
                "```latex",
                timing_tex,
                "```",
                "",
                "```latex",
                throughput_tex,
                "```",
                "",
                "```latex",
                appendix_tex,
                "```",
                "",
            ]
        )
    body = "\n".join(body_parts)

    if args.stdout_only:
        print(body)
        return

    if args.inject:
        inject_into_paper(args.inject.resolve(), body)
        return

    print(body)


if __name__ == "__main__":
    main()
