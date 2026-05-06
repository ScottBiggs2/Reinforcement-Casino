#!/usr/bin/env python3
"""
Concatenate shard outputs from scripts/h200_bsr_orchestrate.sh into RUN_ROOT/merged/.

Expects sibling directories RUN_ROOT/bench_*/benchmark_training_log.csv and benchmark_theory.json
(theory files are JSON lists of phase records).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List


def _shard_dirs(run_root: Path) -> List[Path]:
    return sorted(
        d for d in run_root.iterdir() if d.is_dir() and d.name.startswith("bench_") and d.name != "merged"
    )


def _merge_csvs(paths: Iterable[Path], out_path: Path) -> None:
    paths = list(paths)
    if not paths:
        print("merge_bsr_shard_csvs: no benchmark_training_log.csv found under bench_*.", file=sys.stderr)
        sys.exit(1)
    first = paths[0]
    with first.open(newline="", encoding="utf-8") as rf:
        reader = csv.reader(rf)
        header = next(reader)
        rows: List[list] = [row for row in reader]
    for p in paths[1:]:
        with p.open(newline="", encoding="utf-8") as rf:
            reader = csv.reader(rf)
            h2 = next(reader)
            if h2 != header:
                raise ValueError(f"CSV header mismatch: {first} vs {p}")
            rows.extend(reader)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as wf:
        w = csv.writer(wf)
        w.writerow(header)
        w.writerows(rows)
    print(f"Wrote {out_path}  ({len(rows)} data rows from {len(paths)} shards)")


def _merge_theory(paths: Iterable[Path], out_path: Path) -> None:
    paths = list(paths)
    combined: List[Any] = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            chunk = json.load(f)
        if not isinstance(chunk, list):
            raise ValueError(f"Expected JSON list in {p}")
        combined.extend(chunk)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"Wrote {out_path}  ({len(combined)} records from {len(paths)} shards)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge BSR orchestrator shard CSV + theory JSON.")
    ap.add_argument("--run-root", type=Path, required=True, help="RUN_ROOT containing bench_*/ dirs")
    args = ap.parse_args()
    run_root = args.run_root.resolve()
    shards = _shard_dirs(run_root)
    if not shards:
        print(f"No bench_* dirs under {run_root}", file=sys.stderr)
        sys.exit(1)
    merged = run_root / "merged"
    csv_paths = [(s / "benchmark_training_log.csv") for s in shards]
    csv_ok = [p for p in csv_paths if p.is_file()]
    if len(csv_ok) != len(csv_paths):
        missing = [str(p) for p in csv_paths if not p.is_file()]
        print("Missing CSV(s):\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)
    theory_paths = [(s / "benchmark_theory.json") for s in shards]
    theory_ok = [p for p in theory_paths if p.is_file()]

    _merge_csvs(csv_ok, merged / "benchmark_training_log.csv")
    if theory_ok:
        _merge_theory(theory_ok, merged / "benchmark_theory.json")
    else:
        print("(No benchmark_theory.json found in shards; skipped theory merge)")


if __name__ == "__main__":
    main()
