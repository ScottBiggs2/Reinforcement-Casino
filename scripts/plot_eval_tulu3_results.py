#!/usr/bin/env python3
"""Aggregate the 7 Tulu3 eval result dirs into a single grouped bar chart.

Reads:  ~/rc-sparse-speed/results/eval_tulu3_<TAG>_<jobid>/all_benchmarks_summary.json
Writes: ~/rc-sparse-speed/results/tulu3_eval_summary.{png,csv}

Usage:
    python scripts/plot_eval_tulu3_results.py
    python scripts/plot_eval_tulu3_results.py --results_root ~/rc-sparse-speed/results --out_dir .
"""
import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


# Same extraction logic as run_all_benchmarks.print_summary, but returns floats.
def extract_score(benchmark: str, results_blob: dict) -> Optional[float]:
    res = results_blob.get("results")
    if not isinstance(res, dict):
        return None

    if benchmark == "mmlu":
        for k, v in res.items():
            if "mmlu" in k.lower() and isinstance(v, dict):
                for ak in ("acc,none", "acc"):
                    if ak in v and isinstance(v[ak], (int, float)):
                        return float(v[ak])
    elif benchmark == "math":
        for k, v in res.items():
            if "math" in k.lower() and isinstance(v, dict):
                for ak in ("acc,none", "acc", "exact_match,none", "exact_match"):
                    if ak in v and isinstance(v[ak], (int, float)):
                        return float(v[ak])
    elif benchmark == "gsm8k":
        for k, v in res.items():
            if "gsm8k" in k.lower() and isinstance(v, dict):
                for ak in (
                    "exact_match,strict-match",
                    "exact_match,flexible-extract",
                    "exact_match,none",
                    "exact_match",
                    "acc,none",
                    "acc",
                ):
                    if ak in v and isinstance(v[ak], (int, float)):
                        return float(v[ak])
    elif benchmark == "ifeval":
        for k, v in res.items():
            if "ifeval" in k.lower() and isinstance(v, dict):
                for ak in ("prompt_level_strict_acc,none", "prompt_level_strict_acc"):
                    if ak in v and isinstance(v[ak], (int, float)):
                        return float(v[ak])
    elif benchmark == "squad":
        for k, v in res.items():
            if "squad" in k.lower() and isinstance(v, dict):
                for ak in ("contains,none", "exact_match,none", "exact_match"):
                    if ak in v and isinstance(v[ak], (int, float)):
                        return float(v[ak])
    elif benchmark in ("gpqa", "gpqa_diamond"):
        for k, v in res.items():
            if ("gpqa" in k.lower() or "diamond" in k.lower()) and isinstance(v, dict):
                for ak in ("acc_norm,none", "acc_norm", "acc,none", "acc"):
                    if ak in v and isinstance(v[ak], (int, float)):
                        return float(v[ak])
    elif benchmark == "coding":
        # Aggregate humaneval pass@1 if present, else mbpp
        scores = []
        for task, td in res.items():
            if isinstance(td, dict):
                for ak in (
                    "pass@1,create_test",
                    "pass_at_1,create_test",
                    "pass@1,none",
                    "pass_at_1,none",
                    "pass@1",
                    "pass_at_1",
                ):
                    if ak in td and isinstance(td[ak], (int, float)):
                        scores.append(float(td[ak]))
                        break
        if scores:
            return float(np.mean(scores))
    return None


# Pretty model names + a stable ordering / color group.
MODEL_DISPLAY = {
    "base_llama31_8b_instruct":   ("Base (no train)",         "#888888"),
    "dense_dpo_tulu3_ckpt500":    ("Dense DPO ckpt-500",      "#1f77b4"),
    "sparse_warm_mag_step200":    ("Sparse · warm-mag",       "#2ca02c"),
    "sparse_oracle_dpo_tulu3":    ("Sparse · oracle DPO Tulu3 (in-task)", "#9467bd"),
    "sparse_oracle_dpo_lightr1":  ("Sparse · oracle DPO Light-R1",        "#d62728"),
    "sparse_oracle_grpo_math":    ("Sparse · oracle GRPO Math",           "#ff7f0e"),
    "sparse_random":              ("Sparse · random mask",                "#7f7f7f"),
}
MODEL_ORDER = list(MODEL_DISPLAY.keys())
# gpqa_diamond dropped — Idavidrein/gpqa is a gated HF dataset and our token
# does not have access. Re-add once HF access is granted.
BENCHMARK_ORDER = ["mmlu", "ifeval", "gsm8k", "math", "squad", "coding"]


# Suffixes that mean "partial retry of an earlier run for the same model".
# When we see e.g. base_llama31_8b_instruct_mmlu_only_<jid>, fold it back into
# base_llama31_8b_instruct so per-benchmark scores get merged.
_RETRY_SUFFIXES = ("_mmlu_only", "_retry")


def _canonical_tag(tag: str) -> str:
    for s in _RETRY_SUFFIXES:
        if tag.endswith(s):
            return tag[: -len(s)]
    return tag


def find_summaries(results_root: Path) -> Dict[str, list[Path]]:
    """Map canonical model tag -> list of summary jsons (sorted by job id ascending)."""
    pat = str(results_root / "eval_tulu3_*" / "all_benchmarks_summary.json")
    grouped: Dict[str, list[tuple[int, Path]]] = {}
    rx = re.compile(r"eval_tulu3_(?P<tag>.+)_(?P<jid>\d+)$")
    for p in glob.glob(pat):
        d = Path(p).parent
        m = rx.match(d.name)
        if not m:
            continue
        tag = _canonical_tag(m.group("tag"))
        jid = int(m.group("jid"))
        grouped.setdefault(tag, []).append((jid, Path(p)))
    return {t: [p for _, p in sorted(v)] for t, v in grouped.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_root",
        default=os.path.expanduser("~/rc-sparse-speed/results"),
        help="Dir holding eval_tulu3_<TAG>_<jobid>/ subdirs",
    )
    ap.add_argument(
        "--out_dir",
        default=os.path.expanduser("~/rc-sparse-speed/results"),
        help="Where to write the PNG + CSV",
    )
    args = ap.parse_args()
    root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = find_summaries(root)
    if not summaries:
        raise SystemExit(f"No summaries found under {root}/eval_tulu3_*/")

    # tag -> benchmark -> score (merged across retry runs; later non-error wins)
    table: Dict[str, Dict[str, Optional[float]]] = {}
    for tag, paths in summaries.items():
        per_bench: Dict[str, Optional[float]] = {b: None for b in BENCHMARK_ORDER}
        for p in paths:
            try:
                blob = json.loads(p.read_text())
            except Exception as e:
                print(f"  ! failed to read {p}: {e}")
                continue
            for bench in BENCHMARK_ORDER:
                sub = blob.get(bench)
                if not isinstance(sub, dict) or "error" in sub:
                    continue
                score = extract_score(bench, sub)
                if score is not None:
                    per_bench[bench] = score
        table[tag] = per_bench

    # CSV
    csv_path = out_dir / "tulu3_eval_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_tag", "model_display"] + BENCHMARK_ORDER)
        for tag in MODEL_ORDER:
            if tag not in table:
                continue
            row = [tag, MODEL_DISPLAY[tag][0]]
            for b in BENCHMARK_ORDER:
                v = table[tag].get(b)
                row.append("" if v is None else f"{v:.4f}")
            w.writerow(row)
    print(f"[csv] wrote {csv_path}")

    # Plot — grouped bars: x = benchmark, group = model
    present_models = [m for m in MODEL_ORDER if m in table]
    if not present_models:
        print("no model data, skipping plot")
        return
    n_models = len(present_models)
    n_bench = len(BENCHMARK_ORDER)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(12, 1.5 * n_bench + 2), 6))
    x = np.arange(n_bench)
    for i, tag in enumerate(present_models):
        scores = [table[tag].get(b) for b in BENCHMARK_ORDER]
        # matplotlib will skip None bars if we mask
        ys = np.array([np.nan if s is None else s for s in scores], dtype=float)
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            ys,
            width,
            label=MODEL_DISPLAY[tag][0],
            color=MODEL_DISPLAY[tag][1],
            edgecolor="black",
            linewidth=0.4,
        )
        for b, v in zip(bars, ys):
            if not np.isnan(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.005,
                    f"{v:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in BENCHMARK_ORDER], rotation=20)
    ax.set_ylabel("Score (acc / EM / pass@1, higher is better)")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Tulu3 eval suite — Llama-3.1-8B-Instruct\n"
        "(base / dense DPO ckpt-500 / 5 sparse @ 97.5% sparsity)"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    png_path = out_dir / "tulu3_eval_summary.png"
    fig.savefig(png_path, dpi=160)
    print(f"[png] wrote {png_path}")


if __name__ == "__main__":
    main()
