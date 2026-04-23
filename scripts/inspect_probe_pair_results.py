#!/usr/bin/env python3
"""Inspect pairwise probe results and summarize debug-relevant health metrics.

Usage:
    python scripts/inspect_probe_pair_results.py \
        --results /scratch/$USER/probe_pair_masks/probe_pair_results.json
"""

import argparse
import json
import math
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Path to probe_pair_results.json")
    p.add_argument(
        "--top_k",
        type=int,
        default=8,
        help="How many worst layers (largest |gap| or NaN-heavy) to display per config/property",
    )
    return p.parse_args()


def _is_nan(x):
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def _safe_mean(vals):
    clean = [v for v in vals if not _is_nan(v)]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


def summarize_config(cfg_name, cfg_data, top_k):
    prop_names = sorted(cfg_data.keys())

    total_layers = 0
    degen_layers = 0
    nonconv_layers = 0
    nan_test_layers = 0
    nan_train_layers = 0
    nonfinite_total = 0
    reason_counts = defaultdict(int)

    all_test = []
    all_train = []
    all_gap = []
    all_holdout = []

    print(f"\n=== {cfg_name} ===")
    for prop in prop_names:
        layer_map = cfg_data[prop]
        total_layers += len(layer_map)

        prop_test = []
        prop_train = []
        prop_gap = []
        prop_holdout = []
        worst = []

        for layer_name, m in layer_map.items():
            test = m.get("test", float("nan"))
            train = m.get("train", float("nan"))
            gap = m.get("gap", float("nan"))
            holdout = m.get("holdout_test")
            degen = bool(m.get("degenerate", False))
            conv = bool(m.get("converged", False))
            nonfinite = int(m.get("nonfinite_count", 0))
            reason = m.get("degenerate_reason")

            if degen:
                degen_layers += 1
            if not conv:
                nonconv_layers += 1
            if _is_nan(test):
                nan_test_layers += 1
            if _is_nan(train):
                nan_train_layers += 1

            nonfinite_total += nonfinite
            if reason:
                reason_counts[str(reason)] += 1

            prop_test.append(test)
            prop_train.append(train)
            prop_gap.append(gap)
            if holdout is not None:
                prop_holdout.append(holdout)
                all_holdout.append(holdout)
            all_test.append(test)
            all_train.append(train)
            all_gap.append(gap)

            score = float("inf") if (_is_nan(test) or _is_nan(gap)) else abs(gap)
            worst.append((score, layer_name, test, train, gap, degen, conv, nonfinite, reason))

        print(
            f"  {prop:12s} "
            f"test={_safe_mean(prop_test):.3f} "
            f"train={_safe_mean(prop_train):.3f} "
            f"gap={_safe_mean(prop_gap):+.3f} "
            f"holdout={_safe_mean(prop_holdout):.3f}"
        )

        worst_sorted = sorted(worst, key=lambda x: x[0], reverse=True)
        print("    worst layers:")
        for _, lname, test, train, gap, degen, conv, nonfinite, reason in worst_sorted[:top_k]:
            reason_txt = reason if reason is not None else "-"
            print(
                f"      {lname:36s} "
                f"test={test!s:>8} train={train!s:>8} gap={gap!s:>8} "
                f"degen={int(degen)} conv={int(conv)} nonfinite={nonfinite} reason={reason_txt}"
            )

    reason_msg = ", ".join(f"{k}:{v}" for k, v in sorted(reason_counts.items()))
    if not reason_msg:
        reason_msg = "none"

    print("  summary:")
    print(f"    layers_total={total_layers}")
    print(f"    degenerate_layers={degen_layers} ({degen_layers / max(total_layers, 1):.1%})")
    print(f"    non_converged_layers={nonconv_layers} ({nonconv_layers / max(total_layers, 1):.1%})")
    print(f"    nan_test_layers={nan_test_layers} ({nan_test_layers / max(total_layers, 1):.1%})")
    print(f"    nan_train_layers={nan_train_layers} ({nan_train_layers / max(total_layers, 1):.1%})")
    print(f"    nonfinite_total={nonfinite_total}")
    print(f"    degenerate_reasons={reason_msg}")
    print(f"    overall_test_mean={_safe_mean(all_test):.3f}")
    print(f"    overall_train_mean={_safe_mean(all_train):.3f}")
    print(f"    overall_gap_mean={_safe_mean(all_gap):+.3f}")
    print(f"    overall_holdout_mean={_safe_mean(all_holdout):.3f}")


def main():
    args = parse_args()
    with open(args.results) as f:
        data = json.load(f)

    cfg_names = list(data.keys())
    print(f"Loaded {len(cfg_names)} configs from: {args.results}")
    for cfg in cfg_names:
        summarize_config(cfg, data[cfg], top_k=args.top_k)


if __name__ == "__main__":
    main()
