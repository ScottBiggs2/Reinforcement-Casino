#!/usr/bin/env python3
"""Deep checks for mask .pt files: identity across warm methods, direct Jaccard, file hashes.

Use after a pipeline mask stage when comparisons look "stuck" or Jaccard repeats across pairs.

Example (cluster):
  python src/cold_start/diagnose_mask_files.py \\
    /scratch/$USER/rl_casino_masks/<run_id>

Or compare two files:
  python src/cold_start/diagnose_mask_files.py --mask-a PATH --mask-b PATH
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.cold_start.mask_to_jaccard import compute_jaccard, load_masks


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _mask_dict_equality(a: dict, b: dict) -> tuple[float, int, int]:
    """Fraction of (key, tensor equal) among common keys; count of keys."""
    common = sorted(set(a.keys()) & set(b.keys()))
    if not common:
        return 0.0, 0, 0
    same = 0
    for k in common:
        if a[k].shape != b[k].shape:
            continue
        if torch.equal(a[k].bool(), b[k].bool()):
            same += 1
    return same / len(common), same, len(common)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "mask_dir",
        nargs="?",
        default=None,
        help="Directory containing warm_magnitude_*.pt, warm_fisher_*.pt, cold_fisher_*.pt",
    )
    parser.add_argument("--mask-a", type=str, default=None, help="First mask .pt (optional if mask_dir set)")
    parser.add_argument("--mask-b", type=str, default=None, help="Second mask .pt")
    args = parser.parse_args()

    if args.mask_a and args.mask_b:
        print("=== Direct two-file comparison ===")
        for label, p in ("A", args.mask_a), ("B", args.mask_b):
            print(f"  {label}: {p}")
            print(f"       sha256: {_sha256(p)}  size: {os.path.getsize(p)} bytes")
        ma, _ = load_masks(args.mask_a)
        mb, _ = load_masks(args.mask_b)
        eq_frac, n_same, n_tot = _mask_dict_equality(ma, mb)
        print(f"  Layers with identical binary masks: {n_same}/{n_tot} ({eq_frac:.4%})")
        j = compute_jaccard(ma, mb, device="cpu")
        print(f"  Jaccard (aggregate): {j['aggregate_jaccard']:.6f}")
        print(f"  Jaccard (mean):      {j['mean_jaccard']:.6f}")
        return

    if not args.mask_dir:
        print("Provide mask_dir or both --mask-a and --mask-b", file=sys.stderr)
        sys.exit(1)

    d = Path(args.mask_dir)
    if not d.is_dir():
        print(f"Not a directory: {d}", file=sys.stderr)
        sys.exit(1)

    files = sorted(d.glob("*.pt"))
    if not files:
        print(f"No .pt files in {d}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Mask directory: {d.resolve()} ===\n")
    print("Files (name, bytes, sha256[:16]):")
    for p in files:
        h = _sha256(str(p))
        print(f"  {p.name}")
        print(f"    {p.stat().st_size:>12}  {h[:16]}…")

    # Pair warm magnitude vs warm fisher (same sparsity/step naming in pipeline)
    mag = [f for f in files if "warm_magnitude_" in f.name]
    fish = [f for f in files if "warm_fisher_" in f.name]
    cold = [f for f in files if "cold_fisher_" in f.name]

    if mag and fish:
        print("\n=== warm_magnitude vs warm_fisher (first pair of each) ===")
        m0, f0 = str(mag[0]), str(fish[0])
        print(f"  Using:\n    {m0}\n    {f0}")
        ma, _ = load_masks(m0)
        mb, _ = load_masks(f0)
        eq_frac, n_same, n_tot = _mask_dict_equality(ma, mb)
        print(f"  Identical binary masks per layer: {n_same}/{n_tot} ({eq_frac:.4%})")
        if eq_frac > 0.99:
            print(
                "  NOTE: If ~100% identical, Jaccard vs a shared cold mask will match for wm_vs_cf and wf_vs_cf."
            )
        j = compute_jaccard(ma, mb, device="cpu")
        print(f"  Direct Jaccard(magnitude, warm_fisher): aggregate={j['aggregate_jaccard']:.6f}")

    if mag and cold:
        print("\n=== warm_magnitude vs cold_fisher (first pair) ===")
        c0 = str(cold[0]) if cold else None
        if c0:
            print(f"  Using:\n    {str(mag[0])}\n    {c0}")
            ma, _ = load_masks(str(mag[0]))
            mc, _ = load_masks(c0)
            j = compute_jaccard(ma, mc, device="cpu")
            print(f"  Jaccard: aggregate={j['aggregate_jaccard']:.6f}")

    print("\n=== Hints ===")
    print("  - Repeated aggregate J vs cold_fisher: compare warm_mag vs warm_fisher above.")
    print("  - Slow stage 3: layer_metrics CSV used to run SVD per layer; pipeline now uses")
    print("    EXPORT_LAYER_METRICS_SKIP_EFFECTIVE_RANK=1 by default (see pipeline_common.sh).")


if __name__ == "__main__":
    main()
