"""
What fraction of a warm-start mask is chosen by tie-break noise rather than by signal?

The warm-start selection score is the accumulated |Delta theta| up to step k, read from the delta
logs. Where that score is exactly zero, the mask builder cannot rank coordinates and falls back to
tie-break noise (torch.manual_seed(42), scale = max|score| * 1e-6, mask_utils.py:169/:513). If the
keep budget exceeds the number of coordinates with a nonzero score, the remainder of the mask is
selected arbitrarily.

That matters twice over:

  1. Interpretation. It bounds how much of the subnetwork the warm-start signal actually
     determines, which speaks directly to the claim that warm-start is a good oracle estimator.
  2. Provenance. torch.rand on CPU and on CUDA are different RNG streams, so two masks built at
     different k are only comparable if they were built on the same device kind. If the
     tie-broken share is large, a device mismatch shows up as a spurious drop in Jaccard that
     has nothing to do with k.

Note on interpretation: delta logs store bf16, and at lr=5e-7 a single step's update to a given
weight can round to zero in bf16. So "exactly zero score" means "no displacement resolvable at
bf16 in the logged deltas", not necessarily "no true displacement". Either way it is zero as far
as the mask builder can see, so the tie-break share is real.

  python scripts/measure_mask_tiebreak_share.py \
      --mask 50=/path/warm_magnitude_..._step50.pt \
      --cache 50=/path/magnitude_caches/mag_aggregate_step_50.pt \
      --out_json /path/tiebreak_share.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.utils.mask_utils import load_masks_file


def parse_kv(specs, what):
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--{what} expects key=path, got {spec!r}")
        k, v = spec.split("=", 1)
        out[k] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mask", action="append", required=True, help="label=mask_path")
    p.add_argument("--cache", action="append", required=True,
                   help="label=mag_aggregate_step_<k>.pt (same label as its mask)")
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    masks_in = parse_kv(args.mask, "mask")
    caches_in = parse_kv(args.cache, "cache")
    if set(masks_in) != set(caches_in):
        raise SystemExit(f"label mismatch: masks {sorted(masks_in)} vs caches {sorted(caches_in)}")

    results = {}
    for label in sorted(masks_in, key=lambda x: (len(x), x)):
        masks, meta, _ = load_masks_file(masks_in[label])
        cache = torch.load(caches_in[label], map_location="cpu")

        n = k = nz = sel_zero = nz_sel = 0
        missing = []
        for name, m in masks.items():
            if name not in cache:
                missing.append(name)
                continue
            mb = m.bool()
            pos = cache[name] > 0
            n += int(mb.numel())
            k += int(mb.sum())
            nz += int(pos.sum())
            sel_zero += int((mb & ~pos).sum())
            nz_sel += int((mb & pos).sum())

        rec = {
            "label": label,
            "mask_path": masks_in[label],
            "cache_path": caches_in[label],
            "mask_tensors": len(masks),
            "tensors_missing_from_cache": len(missing),
            "covered_elements": n,
            "selected": k,
            "nonzero_score_coords": nz,
            "nonzero_score_fraction": nz / n if n else None,
            "selected_with_zero_score": sel_zero,
            "tiebreak_share_of_mask": sel_zero / k if k else None,
            "selected_with_nonzero_score": nz_sel,
            "signal_coverage": nz_sel / nz if nz else None,
            "target_step": (meta or {}).get("target_step"),
        }
        results[label] = rec

        print(f"k={label}: tensors={rec['mask_tensors']} "
              f"missing_from_cache={rec['tensors_missing_from_cache']}")
        print(f"  covered elements      : {n:,}")
        print(f"  selected (keep budget): {k:,}")
        print(f"  coords with score > 0 : {nz:,}  ({rec['nonzero_score_fraction']:.6f} of coverage)")
        print(f"  selected, score == 0  : {sel_zero:,}  "
              f"({rec['tiebreak_share_of_mask']:.4f} of the mask chosen by tie-break)")
        print(f"  selected, score > 0   : {nz_sel:,}  "
              f"(captures {rec['signal_coverage']:.4f} of all nonzero coords)", flush=True)
        del masks, cache

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
