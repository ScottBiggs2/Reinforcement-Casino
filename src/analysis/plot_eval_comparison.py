#!/usr/bin/env python3
"""Aggregate the 7 eval runs (dense + 6 masks) into one grouped-bar chart.

Reads a manifest TSV produced by `scripts/run_eval_6masks.sh`:
    label  job_id  output_dir

For each row, reads every `*_results.json` under `output_dir` and picks out
the primary metric per benchmark, then draws bars grouped by benchmark.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Benchmark → (result file stub, metric key)
# Metric keys match what each evaluator writes out.
METRICS = [
    ("MATH",         "math",   ["groups", "minerva_math", "math_verify,none"]),
    ("GSM8K",        "gsm8k",  ["gsm8k", "exact_match,strict-match"]),
    ("IFEval-Loose", "ifeval", ["ifeval", "prompt_level_loose_acc,none"]),
    ("IFEval-Strict", "ifeval", ["ifeval", "prompt_level_strict_acc,none"]),
    ("HumanEval",    "coding", ["humaneval", "pass@1,create_test"]),
    ("MBPP",         "coding", ["mbpp", "pass_at_1,none"]),
    ("SQuAD",        "squad",  ["squad_completion", "contains,none"]),
    ("GPQA-Diamond", "gpqa",   ["gpqa_diamond", "acc"]),
    ("MMLU",         "mmlu",   ["mmlu", "acc,none"]),
]

LABEL_ORDER = [
    "Dense",
    "Random-DPO", "Cold-CAV-DPO", "Oracle-DPO",
    "Random-GRPO", "Cold-CAV-GRPO", "Oracle-GRPO",
]
COLORS = {
    "Dense":          "#ff7f0e",
    "Random-DPO":     "#d62728",
    "Cold-CAV-DPO":   "#1f77b4",
    "Oracle-DPO":     "#2ca02c",
    "Random-GRPO":    "#ff9896",
    "Cold-CAV-GRPO":  "#aec7e8",
    "Oracle-GRPO":    "#98df8a",
}


def _walk_dict(d, path=None):
    path = path or []
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _walk_dict(v, path + [k])
    else:
        yield path, d


def _pick_metric(results_json: dict, keys: list) -> float:
    """Scan leaves of a nested dict for a path where every required `key`
    appears as a standalone segment (not substring of a larger segment).

    Each key may be a plain segment (e.g. "mmlu") or a segment-prefix match
    (e.g. "acc,none" matches "acc,none" or "acc,flex"). Path segments are
    compared case-insensitively. Skips any path containing "stderr".
    Returns the first numeric match.
    """
    keys_lc = [k.lower() for k in keys]
    for path, val in _walk_dict(results_json):
        if not isinstance(val, (int, float)):
            continue
        segs_lc = [s.lower() for s in path]
        if any("stderr" in s for s in segs_lc):
            continue
        if all(any(k == s or s.startswith(k) for s in segs_lc) for k in keys_lc):
            return float(val)
    return float("nan")


def load_one(output_dir: str) -> dict:
    """Read every *_results.json in a dir and pick metrics per benchmark."""
    found = {}
    seen_stubs = set()
    for _, stub, _ in METRICS:
        if stub in seen_stubs:
            continue
        seen_stubs.add(stub)
        path = os.path.join(output_dir, f"{stub}_results.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [warn] could not parse {path}: {e}")
            continue
        found[stub] = data
    # Also accept an aggregate results.json if the runner writes one.
    agg = os.path.join(output_dir, "results.json")
    if os.path.exists(agg):
        try:
            with open(agg) as f:
                data = json.load(f)
            found["_agg"] = data
        except Exception:
            pass
    return found


def metric_for_label(label: str, dumps: dict, bench_stub: str, keys: list):
    if bench_stub in dumps:
        return _pick_metric(dumps[bench_stub], keys)
    if "_agg" in dumps:
        return _pick_metric(dumps["_agg"], keys)
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="manifest.tsv path")
    ap.add_argument("--output", default=None,
                    help="Output PNG (default: alongside manifest)")
    args = ap.parse_args()

    rows = []
    with open(args.manifest) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    by_label = {r["label"]: r for r in rows}

    # Load dumps
    print("[load] reading per-run result JSONs...")
    dumps_by_label = {}
    for lbl, row in by_label.items():
        dumps_by_label[lbl] = load_one(row["output_dir"])
        print(f"  {lbl:16s}  {len(dumps_by_label[lbl])} result files "
              f"← {row['output_dir']}")

    # Build matrix: benchmarks × labels
    bench_names = [name for name, _, _ in METRICS]
    labels = [lbl for lbl in LABEL_ORDER if lbl in by_label]
    M = np.full((len(bench_names), len(labels)), np.nan)
    for j, lbl in enumerate(labels):
        for i, (_, stub, keys) in enumerate(METRICS):
            M[i, j] = metric_for_label(lbl, dumps_by_label[lbl], stub, keys)

    # --- plot -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 5.5))
    n_bench = len(bench_names)
    n_lbl = len(labels)
    width = 0.9 / n_lbl
    x = np.arange(n_bench)
    for j, lbl in enumerate(labels):
        xs = x + (j - (n_lbl - 1) / 2) * width
        ax.bar(xs, M[:, j], width, label=lbl,
               color=COLORS.get(lbl, None), edgecolor="black", linewidth=0.3)
        for xi, v in zip(xs, M[:, j]):
            if not np.isnan(v):
                ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                            xytext=(0, 2), ha="center", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(bench_names, rotation=15, ha="right")
    ax.set_ylabel("Score (benchmark-primary metric)")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"LLaMA-3.1-8B-Instruct — Eval comparison "
                 f"(sparsity 97.5%, EVAL_LIMIT={os.environ.get('EVAL_LIMIT', '?')})",
                 fontsize=12, fontweight="bold")
    ax.legend(ncol=4, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis="y", alpha=0.3)

    out = args.output or os.path.join(os.path.dirname(args.manifest),
                                      "eval_comparison.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] → {out}")

    # Print the table
    print("\n=== Score table ===")
    print(f"{'benchmark':16s} " + " ".join(f"{l:>14s}" for l in labels))
    for i, name in enumerate(bench_names):
        print(f"{name:16s} " + " ".join(f"{M[i, j]:14.3f}" for j in range(n_lbl)))


if __name__ == "__main__":
    main()
