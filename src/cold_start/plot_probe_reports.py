#!/usr/bin/env python3
"""Bar + line charts from mask_probe_report.py JSON files (suite probe_reports/ directory)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_LAYER_IDX = re.compile(r"model\.layers\.(\d+)\.")


def _layer_index(name: str) -> Optional[int]:
    m = _LAYER_IDX.search(str(name))
    return int(m.group(1)) if m else None


def _load_probe_json(path: Path) -> Tuple[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stem = path.stem
    label = stem.replace("_probe_report", "") if "_probe_report" in stem else stem
    return label, data


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot linear-probe summaries from mask interpretation suite probe_reports/*.json."
    )
    p.add_argument(
        "--suite-dir",
        type=str,
        required=True,
        help="Suite output directory containing probe_reports/*.json",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write PNGs (default: <suite-dir>/probe_plots).",
    )
    args = p.parse_args()
    suite = Path(args.suite_dir).resolve()
    probe_dir = suite / "probe_reports"
    if not probe_dir.is_dir():
        print(f"No probe_reports directory: {probe_dir}", file=sys.stderr)
        raise SystemExit(2)

    json_paths = sorted(probe_dir.glob("*_probe_report.json"))
    if not json_paths:
        print(f"No *_probe_report.json under {probe_dir}", file=sys.stderr)
        raise SystemExit(2)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"matplotlib required: {e}", file=sys.stderr)
        raise SystemExit(1)

    out = Path(args.output_dir).resolve() if args.output_dir else (suite / "probe_plots")
    out.mkdir(parents=True, exist_ok=True)

    series: List[Tuple[str, Dict[str, Any]]] = [_load_probe_json(jp) for jp in json_paths]

    # --- Bar: mean train accuracy per mask (from summary) ---
    labels = [lb for lb, _ in series]
    means = []
    cv_means = []
    for lb, data in series:
        summ = data.get("summary") or {}
        m = summ.get("mean_train_accuracy")
        cv = summ.get("mean_cv_accuracy_mean")
        means.append(float(m) if m is not None and m == m else float("nan"))
        cv_means.append(float(cv) if cv is not None and cv == cv else float("nan"))

    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(labels)), 4.0))
    x = range(len(labels))
    w = 0.35
    ax.bar([i - w / 2 for i in x], means, width=w, label="mean train acc", color="steelblue")
    if any(cv == cv for cv in cv_means):
        ax.bar([i + w / 2 for i in x], cv_means, width=w, label="mean CV acc", color="darkorange")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(loc="lower right")
    ax.set_title("Linear probes (chosen vs rejected / contrast on MLP hooks)")
    fig.tight_layout()
    bar_path = out / "probe_summary_bars.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {bar_path}")

    # --- Lines: per-layer train accuracy ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.tab10(range(len(series)))
    for idx, (lb, data) in enumerate(series):
        layers = (data.get("probe_report") or {}).get("layers") or {}
        pts: List[Tuple[int, float]] = []
        for lname, row in layers.items():
            if not isinstance(row, dict):
                continue
            li = _layer_index(lname)
            ta = row.get("train_accuracy")
            if li is None or ta is None:
                continue
            try:
                fv = float(ta)
            except (TypeError, ValueError):
                continue
            if fv == fv:
                pts.append((li, fv))
        pts.sort(key=lambda t: t[0])
        if pts:
            xs = [t[0] for t in pts]
            ys = [t[1] for t in pts]
            ax2.plot(xs, ys, marker="o", ms=3, linewidth=1.2, label=lb, color=cmap[idx])
    ax2.set_xlabel("decoder layer index")
    ax2.set_ylabel("train accuracy")
    ax2.set_ylim(0.0, 1.05)
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_title("Per-layer linear probe train accuracy")
    fig2.tight_layout()
    line_path = out / "probe_per_layer_train_accuracy.png"
    fig2.savefig(line_path, dpi=150)
    plt.close(fig2)
    print(f"Wrote {line_path}")


if __name__ == "__main__":
    main()
