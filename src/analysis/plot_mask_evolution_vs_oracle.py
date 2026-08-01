#!/usr/bin/env python
"""Plot how the GRPO active subnetwork evolves toward the 500-step oracle.

Reads a `suite_summary.json` produced by src/cold_start/mask_interpretation_suite.py
(which contains `labels`, an N x N `jaccard_matrix`, and an N x N `cka_matrix` or null)
and draws similarity-to-oracle **as a function of training step T** for the
magnitude masks (mag_step{T}), with the random baseline shown as a flat floor.

The story: early magnitude masks overlap the 500-step oracle far less than late
ones (rising toward T=500) while staying well above the random floor — i.e. the
selected subnetwork drifts over training, so a single endpoint diff
(theta_final - theta_initial) cannot represent it.

Usage:
    python src/analysis/plot_mask_evolution_vs_oracle.py \
        --summary /path/to/suite_summary.json \
        --output  /path/to/mask_evolution_vs_oracle.png \
        [--oracle-label oracle_gt] [--random-label random] [--oracle-step 500]

Labels are mapped to a training step by the integer following "step" in the label
(e.g. "mag_step50" -> 50). The oracle label is pinned to --oracle-step (default 500)
and anchors the curve at similarity 1.0.
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _step_of(label: str) -> Optional[int]:
    """Extract the training step encoded in a label, e.g. 'mag_step250' -> 250."""
    m = re.search(r"step[_-]?(\d+)", label)
    return int(m.group(1)) if m else None


def _series_vs_oracle(
    labels: List[str],
    matrix: Optional[List[List[Optional[float]]]],
    oracle_idx: int,
    oracle_step: int,
    random_idx: Optional[int],
):
    """Return (steps, values) sorted by step for step-bearing masks vs the oracle,
    plus the random->oracle floor value (or None)."""
    if matrix is None:
        return [], [], None
    pts: Dict[int, float] = {}
    for i, lab in enumerate(labels):
        if i == oracle_idx:
            continue
        if random_idx is not None and i == random_idx:
            continue
        step = _step_of(lab)
        if step is None:
            continue
        val = matrix[i][oracle_idx]
        if val is not None:
            pts[step] = float(val)
    # anchor the oracle at its own step (self-similarity == 1.0)
    pts.setdefault(oracle_step, float(matrix[oracle_idx][oracle_idx]))
    steps = sorted(pts)
    values = [pts[s] for s in steps]
    floor = None
    if random_idx is not None:
        fv = matrix[random_idx][oracle_idx]
        floor = None if fv is None else float(fv)
    return steps, values, floor


def _panel(ax, steps, values, floor, metric_name, oracle_step):
    if not steps:
        ax.text(0.5, 0.5, f"no {metric_name} data", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.plot(steps, values, "-o", color="#2b6cb0", lw=2, ms=7, label=f"magnitude mask vs oracle")
    # highlight the oracle anchor point
    ax.plot([oracle_step], [values[steps.index(oracle_step)]], "D", color="#2b6cb0",
            ms=9, label="oracle (T=500)")
    if floor is not None:
        ax.axhline(floor, ls="--", color="#a0aec0", lw=1.5, label=f"random floor ({floor:.3f})")
    ax.set_xlabel("training step T")
    ax.set_ylabel(f"{metric_name} vs 500-step oracle")
    ax.set_title(f"{metric_name}: subnetwork convergence to oracle")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    # annotate each point
    for s, v in zip(steps, values):
        ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True, help="Path to suite_summary.json")
    ap.add_argument("--output", "-o", required=True, help="Output PNG path")
    ap.add_argument("--oracle-label", default="oracle_gt")
    ap.add_argument("--random-label", default="random")
    ap.add_argument("--oracle-step", type=int, default=500)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    with open(args.summary, encoding="utf-8") as f:
        summary = json.load(f)

    labels: List[str] = summary["labels"]
    if args.oracle_label not in labels:
        raise SystemExit(
            f"oracle label {args.oracle_label!r} not in labels {labels}; pass --oracle-label"
        )
    oracle_idx = labels.index(args.oracle_label)
    random_idx = labels.index(args.random_label) if args.random_label in labels else None

    jac = summary.get("jaccard_matrix")
    cka = summary.get("cka_matrix")

    j_steps, j_vals, j_floor = _series_vs_oracle(labels, jac, oracle_idx, args.oracle_step, random_idx)
    c_steps, c_vals, c_floor = _series_vs_oracle(labels, cka, oracle_idx, args.oracle_step, random_idx)

    has_cka = bool(c_steps)
    ncols = 2 if has_cka else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.4 * ncols, 5.2), squeeze=False)
    _panel(axes[0][0], j_steps, j_vals, j_floor, "Jaccard", args.oracle_step)
    if has_cka:
        _panel(axes[0][1], c_steps, c_vals, c_floor, "CKA", args.oracle_step)

    fig.suptitle("GRPO active subnetwork evolves over training\n"
                 "(similarity of step-T magnitude mask to the 500-step oracle)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.output}")
    # also echo the numbers for quick sanity/logging
    print("Jaccard vs oracle:", list(zip(j_steps, [round(v, 4) for v in j_vals])),
          f"| random floor={None if j_floor is None else round(j_floor, 4)}")
    if has_cka:
        print("CKA vs oracle:", list(zip(c_steps, [round(v, 4) for v in c_vals])),
              f"| random floor={None if c_floor is None else round(c_floor, 4)}")


if __name__ == "__main__":
    main()
