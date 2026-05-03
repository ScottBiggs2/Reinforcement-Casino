#!/usr/bin/env python3
"""
Build a Markdown + optional LaTeX report from ``benchmark_training_log.csv``.

Features:
  - **Dense baseline** from this CSV or from ``--baseline-csv`` (dense tail mean for ratios).
  - **Variance:** mean and sample std over the tail for ``cumulative_*`` and ``inst_*`` when present.
  - **Incomplete phases:** compare max logged ``step`` to ``benchmark_phase_target_steps`` when present.
  - **Warnings** for incomplete phases or high coefficient of variation (``--cv-warn``).
  - **``--emit-tex DIR``:** booktabs-ready ``.tex`` fragments (throughput + stability).

Newer benchmark rows may include ``wall_delta_s``, ``inst_steps_per_s`` (interval rates since the
previous log). ``cumulative_steps_per_s`` still includes cold start from ``on_train_begin``; prefer
``inst_steps_per_s`` tail statistics when logging every step.

Usage::

  python3 scripts/benchmark_training_log_to_report_md.py \\
    --csv /scratch/USER/rl_casino_h200_bsr/JOBID/benchmark_training_log.csv \\
    --out benchmark_h200_report.md \\
    --emit-tex ./paper_tables/

  python3 scripts/benchmark_training_log_to_report_md.py \\
    --csv sparse_only.csv --baseline-csv dense_baseline.csv --tail-rows 16
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
from typing import Any, Dict, List, Optional, Tuple


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


def _std_sample(vals: List[float]) -> float:
    finite = [v for v in vals if not math.isnan(v)]
    if len(finite) < 2:
        return float("nan")
    m = sum(finite) / len(finite)
    var = sum((x - m) ** 2 for x in finite) / (len(finite) - 1)
    return math.sqrt(var)


def _cv(mean_v: float, std_v: float) -> float:
    if mean_v is None or std_v is None or math.isnan(mean_v) or math.isnan(std_v):
        return float("nan")
    if abs(mean_v) < 1e-18:
        return float("nan")
    return std_v / abs(mean_v)


def _fmt(x: float, nd: int = 4, empty: str = "—") -> str:
    if x is None or math.isnan(x) or math.isinf(x):
        return empty
    return f"{float(x):.{nd}f}"


def _fmt_sci(x: float) -> str:
    if x is None or math.isnan(x) or math.isinf(x) or x <= 0:
        return "—"
    return f"{float(x):.4e}"


def _fmt_pct_ratio(x: float) -> str:
    if x is None or math.isnan(x) or math.isinf(x) or x <= 0:
        return "—"
    return f"{100.0 * x:.1f}% of baseline"


def _latex_tt(s: str) -> str:
    return "\\texttt{" + str(s).replace("_", "\\_").replace("%", "\\%") + "}"


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    raw = path.read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines(keepends=True) if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(StringIO("".join(lines))))


def _expected_target_steps(rows_for_phase: List[Dict[str, str]], override: Optional[int]) -> Optional[int]:
    if override is not None:
        return int(override)
    for x in reversed(rows_for_phase):
        v = (x.get("benchmark_phase_target_steps") or "").strip()
        if not v:
            continue
        try:
            return int(float(v))
        except (TypeError, ValueError):
            continue
    return None


def compute_phase_statistics(
    rows: List[Dict[str, str]],
    tail_rows: int,
    expected_steps_override: Optional[int],
) -> List[Dict[str, Any]]:
    by_phase: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        ph = (r.get("phase") or "").strip()
        if not ph:
            continue
        by_phase[ph].append(r)

    out: List[Dict[str, Any]] = []
    for phase in sorted(by_phase.keys(), key=_phase_sort_key):
        g = sorted(by_phase[phase], key=lambda x: int(float(x.get("step") or 0)))
        tail = g[-max(1, tail_rows) :]
        last = tail[-1]
        max_step = max(int(float(x.get("step") or 0)) for x in g)
        exp = _expected_target_steps(g, expected_steps_override)
        incomplete = bool(exp is not None and max_step < exp)

        sp = last.get("sparsity_target_pct", "").strip()
        if sp == "":
            sp_val: Any = ""
        else:
            sp_val = _to_float(sp)

        cum_sps = [_to_float(x.get("cumulative_steps_per_s")) for x in tail]
        cum_sms = [_to_float(x.get("cumulative_samples_per_s")) for x in tail]
        inst_sps = [_to_float(x.get("inst_steps_per_s")) for x in tail]
        inst_sms = [_to_float(x.get("inst_samples_per_s")) for x in tail]
        walls = [_to_float(x.get("wall_time_s")) for x in tail]

        m_cum_sps = _mean(cum_sps)
        s_cum_sps = _std_sample(cum_sps)
        cv_cum = _cv(m_cum_sps, s_cum_sps)
        m_inst_sps = _mean(inst_sps)
        s_inst_sps = _std_sample(inst_sps)

        row: Dict[str, Any] = {
            "phase": phase,
            "sparsity_target_pct": sp_val,
            "optimizer": (last.get("optimizer") or "").strip(),
            "last_logged_step": max_step,
            "expected_steps": exp,
            "incomplete": incomplete,
            "mean_cumulative_steps_per_s": m_cum_sps,
            "std_cumulative_steps_per_s": s_cum_sps,
            "cv_cumulative_steps_per_s": cv_cum,
            "mean_cumulative_samples_per_s": _mean(cum_sms),
            "std_cumulative_samples_per_s": _std_sample(cum_sms),
            "mean_inst_steps_per_s": m_inst_sps,
            "std_inst_steps_per_s": s_inst_sps,
            "mean_inst_samples_per_s": _mean(inst_sms),
            "mean_wall_time_s": _mean(walls),
            "trainer_grad_accum_steps": int(float(last.get("trainer_grad_accum_steps") or 0) or 0),
            "trainer_per_device_train_batch_size": int(
                float(last.get("trainer_per_device_train_batch_size") or 0) or 0
            ),
            "benchmark_mlp_only": (last.get("benchmark_mlp_only") or "").strip(),
            "benchmark_mask_key_count": (last.get("benchmark_mask_key_count") or "").strip(),
        }

        for col in (
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
            row[f"mean_{col}"] = _mean([_to_float(x.get(col)) for x in tail])

        out.append(row)
    return out


def _dense_baseline_rates(stats: List[Dict[str, Any]]) -> Tuple[float, float]:
    dense = next((s for s in stats if s["phase"] == "dense"), None)
    if dense is None:
        return float("nan"), float("nan")
    return float(dense["mean_cumulative_steps_per_s"]), float(dense["mean_cumulative_samples_per_s"])


def _load_external_dense_baseline(path: Path, tail_rows: int) -> Tuple[float, float]:
    rows = load_csv_rows(path)
    stats = compute_phase_statistics(rows, tail_rows, None)
    ds, dm = _dense_baseline_rates(stats)
    if math.isnan(ds) or ds <= 0:
        raise SystemExit(
            f"ERROR: --baseline-csv {path} has no usable `dense` phase for baseline rates."
        )
    return ds, dm


def _collect_warnings(stats: List[Dict[str, Any]], cv_threshold: float) -> List[str]:
    w: List[str] = []
    for s in stats:
        if s.get("incomplete"):
            exp = s.get("expected_steps")
            w.append(
                f"Phase `{s['phase']}` appears **incomplete**: max step {s['last_logged_step']} "
                f"< expected {exp}."
            )
        cv = float(s.get("cv_cumulative_steps_per_s", float("nan")))
        if not math.isnan(cv) and cv > cv_threshold:
            w.append(
                f"Phase `{s['phase']}` has high tail CV on cumulative steps/s "
                f"(CV={cv:.3f} > threshold {cv_threshold:.3f}); treat tail mean as noisy."
            )
    return w


def build_markdown(
    *,
    csv_path: Path,
    stats: List[Dict[str, Any]],
    tail_rows: int,
    warnings: List[str],
    external_baseline: Optional[Tuple[float, float]],
) -> str:
    lines: List[str] = []
    lines.append(f"# Benchmark report: `{csv_path.name}`")
    lines.append("")
    lines.append(f"- **Source:** `{csv_path.resolve()}`")
    lines.append(
        f"- **Per-phase stats:** mean and sample **std** over the last **{tail_rows}** rows; "
        "**CV** = std / |mean| on cumulative steps/s."
    )
    lines.append(
        "- **Throughput:** `cumulative_steps_per_s` includes cold start from train begin (load, compile). "
        "When present, **`inst_steps_per_s`** is the interval rate since the previous log (closer to steady-state if you log every step)."
    )
    lines.append("")

    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for msg in warnings:
            lines.append(f"- {msg}")
        lines.append("")

    d_sps, d_sms = _dense_baseline_rates(stats)
    baseline_label = "dense phase in this CSV"
    if external_baseline is not None:
        d_sps, d_sms = external_baseline
        baseline_label = "external `--baseline-csv` (dense tail mean)"

    if not math.isnan(d_sps) and d_sps > 0:
        lines.append("## Baseline (for ratios)")
        lines.append("")
        lines.append(f"Steps/s and samples/s are taken from **{baseline_label}**.")
        lines.append("")
        lines.append("| Metric | Baseline |")
        lines.append("|--------|----------|")
        lines.append(f"| Optimizer steps / s | {_fmt(d_sps, 6)} |")
        lines.append(f"| Samples / s | {_fmt(d_sms, 4)} |")
        lines.append("")
    else:
        lines.append("## Baseline")
        lines.append("")
        lines.append(
            "No dense phase in this CSV and no ``--baseline-csv`` provided — **no vs-dense ratios**."
        )
        lines.append("")

    lines.append("## Throughput and stability (tail window)")
    lines.append("")
    if not math.isnan(d_sps) and d_sps > 0:
        head = (
            "| Phase | Sparse % | Opt | Target steps | Max step | Complete? | "
            "cum steps/s μ | σ | CV | inst steps/s μ | σ | wall μ (s) | mlp_only | #mask keys | "
            "vs baseline steps/s |"
        )
        sep = (
            "|-------|------------|-----|----------------|----------|-----------|"
            "-----------------|---|---|------------------|---|------------|----------|"
            "------------|--------------------|"
        )
    else:
        head = (
            "| Phase | Sparse % | Opt | Target steps | Max step | Complete? | "
            "cum steps/s μ | σ | CV | inst steps/s μ | σ | wall μ (s) | mlp_only | #mask keys |"
        )
        sep = (
            "|-------|------------|-----|----------------|----------|-----------|"
            "-----------------|---|---|------------------|---|------------|----------|------------|"
        )
    lines.append(head)
    lines.append(sep)

    for s in stats:
        sp = s.get("sparsity_target_pct", "")
        sp_s = "" if sp == "" or (isinstance(sp, float) and math.isnan(sp)) else str(sp)
        exp_s = str(s["expected_steps"]) if s.get("expected_steps") is not None else "—"
        ok = "yes" if not s.get("incomplete") else "**no**"
        mlp = s.get("benchmark_mlp_only", "") or "—"
        mk = s.get("benchmark_mask_key_count", "") or "—"
        row_cells = [
            f"`{s['phase']}`",
            sp_s,
            str(s.get("optimizer", "")),
            exp_s,
            str(s["last_logged_step"]),
            ok,
            _fmt(float(s["mean_cumulative_steps_per_s"]), 6),
            _fmt(float(s["std_cumulative_steps_per_s"]), 6),
            _fmt(float(s["cv_cumulative_steps_per_s"]), 4),
            _fmt(float(s["mean_inst_steps_per_s"]), 6),
            _fmt(float(s["std_inst_steps_per_s"]), 6),
            _fmt(float(s["mean_wall_time_s"]), 2),
            mlp,
            mk,
        ]
        if not math.isnan(d_sps) and d_sps > 0:
            ratio = float(s["mean_cumulative_steps_per_s"]) / d_sps if s["phase"] != "dense" else 1.0
            row_cells.append(_fmt_pct_ratio(ratio))
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    # Theory
    has_theory = any(
        not math.isnan(float(s.get("mean_theory_bsr_backward_flops_proxy", float("nan"))))
        and float(s.get("mean_theory_bsr_backward_flops_proxy", 0)) > 0
        for s in stats
        if s["phase"] != "dense"
    )
    if has_theory:
        lines.append("## Theory (mask sidecar, tail mean)")
        lines.append("")
        lines.append("| Phase | Sparse % | Theory BWD FLOP/step | active fraction | b-token |")
        lines.append("|-------|------------|----------------------|-----------------|---------|")
        for s in stats:
            if s["phase"] == "dense":
                continue
            th = float(s.get("mean_theory_bsr_backward_flops_proxy", float("nan")))
            if math.isnan(th) or th <= 0:
                continue
            sp = s.get("sparsity_target_pct", "")
            sp_s = "" if sp == "" or (isinstance(sp, float) and math.isnan(sp)) else str(sp)
            lines.append(
                "| `"
                + s["phase"]
                + "` | "
                + sp_s
                + " | "
                + _fmt_sci(th)
                + " | "
                + _fmt(float(s.get("mean_theory_active_fraction_global", float("nan"))), 6)
                + " | "
                + (
                    str(int(float(s.get("mean_theory_b_tokens_proxy"))))
                    if not math.isnan(float(s.get("mean_theory_b_tokens_proxy", float("nan"))))
                    and float(s.get("mean_theory_b_tokens_proxy", 0)) > 0
                    else "—"
                )
                + " |"
            )
        lines.append("")

    # CUDA timing
    has_tim = any(
        not math.isnan(float(s.get("mean_t_step_total_ms", float("nan"))))
        and float(s.get("mean_t_step_total_ms", 0)) > 0
        for s in stats
    )
    lines.append("## CUDA segment timings (tail mean)")
    lines.append("")
    if not has_tim:
        lines.append(
            "*No `t_step_total_ms` in tail — run with **`RL_CASINO_BSR_DETAILED_TIMING=1`** for segment ms and effective TFLOP/s.*"
        )
        lines.append("")
    else:
        lines.append(
            "| Phase | t_step | t_fwd | t_bwd | t_opt | TFLOP/s bwd | TFLOP/s step |"
        )
        lines.append("|-------|--------|-------|-------|-------|-------------|--------------|")
        for s in stats:
            lines.append(
                "| `"
                + s["phase"]
                + "` | "
                + _fmt(float(s.get("mean_t_step_total_ms", float("nan"))), 2)
                + " | "
                + _fmt(float(s.get("mean_t_forward_ms", float("nan"))), 2)
                + " | "
                + _fmt(float(s.get("mean_t_backward_ms", float("nan"))), 2)
                + " | "
                + _fmt(float(s.get("mean_t_optim_ms", float("nan"))), 2)
                + " | "
                + _fmt(float(s.get("mean_eff_bsr_backward_tflops", float("nan"))), 4)
                + " | "
                + _fmt(float(s.get("mean_eff_bsr_backward_tflops_over_e2e_step", float("nan"))), 4)
                + " |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Generated by `scripts/benchmark_training_log_to_report_md.py`.*")
    lines.append("")
    return "\n".join(lines)


def emit_latex_tables(
    *,
    out_dir: Path,
    csv_stem: str,
    stats: List[Dict[str, Any]],
    warnings: List[str],
    d_sps: float,
    d_sms: float,
) -> None:
    del warnings  # surfaced in Markdown; LaTeX captions should stay ASCII-safe
    out_dir.mkdir(parents=True, exist_ok=True)
    pfx = csv_stem.replace(" ", "_")

    # Throughput + stability
    body: List[str] = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \footnotesize",
        r"  \caption{H200 BSR benchmark: tail mean ($\mu$) and sample std ($\sigma$) of cumulative and interval steps/s.}",
        r"  \label{tab:" + pfx + r"_throughput_stability}",
        r"  \begin{tabular}{@{}lccccccc@{}}",
        r"    \toprule",
        r"    Phase & $\mu_{\mathrm{cum}}$ s$^{-1}$ & $\sigma_{\mathrm{cum}}$ & CV & "
        r"$\mu_{\mathrm{inst}}$ s$^{-1}$ & max step & target & ok? \\",
        r"    \midrule",
    ]
    for s in stats:
        exp = s.get("expected_steps")
        exp_s = str(exp) if exp is not None else "---"
        ok = "yes" if not s.get("incomplete") else "no"
        body.append(
            "    "
            + _latex_tt(s["phase"])
            + " & "
            + _fmt(float(s["mean_cumulative_steps_per_s"]), 6)
            + " & "
            + _fmt(float(s["std_cumulative_steps_per_s"]), 6)
            + " & "
            + _fmt(float(s["cv_cumulative_steps_per_s"]), 4)
            + " & "
            + _fmt(float(s["mean_inst_steps_per_s"]), 6)
            + " & "
            + str(s["last_logged_step"])
            + " & "
            + exp_s
            + " & "
            + ok
            + r" \\"
        )
    body += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    (out_dir / f"{pfx}_throughput_stability.tex").write_text("\n".join(body), encoding="utf-8")

    # Baseline ratios if dense baseline known
    if not math.isnan(d_sps) and d_sps > 0:
        rb: List[str] = [
            r"\begin{table}[t]",
            r"  \centering",
            r"  \small",
            r"  \caption{Throughput relative to dense baseline ($\mu_{\mathrm{cum}}$ steps/s).}",
            r"  \label{tab:" + pfx + r"_vs_dense}",
            r"  \begin{tabular}{@{}lc@{}}",
            r"    \toprule",
            r"    Phase & frac of dense \\",
            r"    \midrule",
        ]
        for s in stats:
            if s["phase"] == "dense":
                frac = 1.0
            else:
                frac = float(s["mean_cumulative_steps_per_s"]) / d_sps if not math.isnan(d_sps) else float("nan")
            rb.append(f"    {_latex_tt(s['phase'])} & {_fmt(frac, 4)} \\\\")
        rb += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
        (out_dir / f"{pfx}_vs_dense.tex").write_text("\n".join(rb), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True, help="Path to benchmark_training_log.csv")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output Markdown path (default: <csv_stem>_report.md next to the CSV)",
    )
    ap.add_argument("--tail-rows", type=int, default=8, help="Tail window size per phase")
    ap.add_argument(
        "--baseline-csv",
        type=Path,
        default=None,
        help="Optional second CSV containing a `dense` phase for baseline ratios",
    )
    ap.add_argument(
        "--expected-steps",
        type=int,
        default=None,
        help="Override expected optimizer steps per phase (else use benchmark_phase_target_steps column)",
    )
    ap.add_argument(
        "--cv-warn",
        type=float,
        default=0.35,
        help="Warn when tail CV of cumulative steps/s exceeds this threshold",
    )
    ap.add_argument(
        "--emit-tex",
        type=Path,
        default=None,
        help="Write LaTeX table fragments under this directory",
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

    stats = compute_phase_statistics(rows, args.tail_rows, args.expected_steps)
    warnings = _collect_warnings(stats, args.cv_warn)

    external: Optional[Tuple[float, float]] = None
    if args.baseline_csv is not None:
        bp = args.baseline_csv.resolve()
        if not bp.is_file():
            print(f"ERROR: baseline CSV not found: {bp}", file=sys.stderr)
            sys.exit(1)
        external = _load_external_dense_baseline(bp, args.tail_rows)

    d_sps, d_sms = _dense_baseline_rates(stats)
    if external is not None:
        d_sps, d_sms = external

    md = build_markdown(
        csv_path=csv_path,
        stats=stats,
        tail_rows=args.tail_rows,
        warnings=warnings,
        external_baseline=external,
    )

    out_path = args.out if args.out is not None else csv_path.parent / f"{csv_path.stem}_report.md"
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path}")

    if args.emit_tex is not None:
        emit_latex_tables(
            out_dir=args.emit_tex.resolve(),
            csv_stem=csv_path.stem,
            stats=stats,
            warnings=warnings,
            d_sps=d_sps,
            d_sms=d_sms,
        )
        print(f"Wrote LaTeX tables under {args.emit_tex.resolve()}")


if __name__ == "__main__":
    main()
