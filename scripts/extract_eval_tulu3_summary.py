#!/usr/bin/env python3
"""Collapse the per-arm eval_tulu3 result JSONs into one small summary JSON
(arms x benchmarks -> headline metric). Run on AICR login; scp the output down
for plotting. Missing benchmarks -> null."""

import glob
import json
import os

ROOT = "/scratch/xie_yiyi_neu/rebuttal_analysis/eval_tulu3"
OUT = os.path.join(ROOT, "table1_summary.json")


def headline(bench, results):
    if bench == "mmlu":
        return results.get("mmlu", {}).get("acc,none")
    if bench == "math":
        vals = [m.get("exact_match,none") for t, m in results.items()
                if t.startswith(("minerva_math", "hendrycks_math"))
                and isinstance(m.get("exact_match,none"), float)]
        return sum(vals) / len(vals) if vals else None
    if bench == "gsm8k":
        return results.get("gsm8k", {}).get("exact_match,strict-match")
    if bench == "humaneval":
        return results.get("humaneval", {}).get("pass@1,create_test")
    if bench == "mbpp":
        return results.get("mbpp", {}).get("pass_at_1,none")
    if bench == "ifeval":
        m = results.get("ifeval", {})
        for k in ("prompt_level_strict_acc,none", "inst_level_strict_acc,none"):
            if isinstance(m.get(k), float):
                return m[k]
        return None
    if bench == "squad":
        for t, m in results.items():
            if "squad" in t:
                for k in ("exact_match,none", "f1,none", "contains,none", "acc,none"):
                    if isinstance(m.get(k), float):
                        return m[k]
                floats = [v for k, v in m.items()
                          if isinstance(v, float) and "stderr" not in k]
                if floats:
                    return floats[0]
        return None
    if bench == "gpqa_diamond":
        for t, m in results.items():
            if "gpqa" in t and isinstance(m.get("acc,none"), float):
                return m["acc,none"]
        return None
    return None


FILES = {"mmlu": "mmlu", "math": "math", "gsm8k": "gsm8k",
         "humaneval": "coding", "mbpp": "coding", "ifeval": "ifeval",
         "squad": "squad", "gpqa_diamond": "gpqa_diamond"}

summary = {}
for d in sorted(glob.glob(os.path.join(ROOT, "*_2466*/"))):
    arm = os.path.basename(d.rstrip("/")).rsplit("_", 1)[0]
    row = {}
    for bench, stem in FILES.items():
        path = os.path.join(d, f"{stem}_results.json")
        val = None
        if os.path.exists(path):
            try:
                val = headline(bench, json.load(open(path)).get("results", {}))
            except Exception:
                val = None
        row[bench] = val
    summary[arm] = row

with open(OUT, "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
print("->", OUT)
