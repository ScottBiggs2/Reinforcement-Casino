"""
Restrict a mask to the tensor coverage of a reference mask.

Why this is needed. The delta-log mask finder de-duplicates tied weights with a content hash
(shape + delta.sum(), even_better_mask_finder.py:105-116). When most deltas are exactly zero in
bf16, same-shape tensors collide spuriously and get dropped, so a delta-log mask covers fewer
tensors than the model has (216 of 226 for Olmo-3-7B/Light-R1). Checkpoint-diff and random masks
do not go through that path and cover all 226.

That difference is not cosmetic. sparse_dpo_efficiency.py enforces sparsity only through
SparseAdamW, and sparse_adamw.py:119-131 gives any parameter without a mask entry a full dense
AdamW update. So a tensor missing from the mask file is trained densely, and two arms whose masks
differ in coverage differ in trainable parameter count -- an uncontrolled difference that shows up
in any comparison between them.

Subsetting a globally-uniform-rate mask to a smaller tensor set preserves its keep *rate*, so the
result is coverage-matched to the reference while remaining the same random draw on the tensors it
retains. Comparing the original against the subset therefore isolates the coverage effect alone.

  python scripts/subset_mask_to_reference_coverage.py \
      --reference .../warm_magnitude_..._step250.pt \
      --mask .../random_..._seed42.pt \
      --out .../random_..._seed42_coverage216.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.utils.mask_utils import load_masks_file, save_masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="mask whose tensor coverage defines the target")
    p.add_argument("--mask", required=True, help="mask to restrict")
    p.add_argument("--out", required=True)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    if os.path.exists(args.out):
        raise SystemExit(f"refusing to overwrite {args.out}")

    ref_masks, ref_meta, _ = load_masks_file(args.reference)
    src_masks, src_meta, _ = load_masks_file(args.mask)

    ref_keys = set(ref_masks)
    src_keys = set(src_masks)
    missing = sorted(ref_keys - src_keys)
    if missing:
        raise SystemExit(
            f"reference covers {len(missing)} tensors the mask does not, so the mask cannot be "
            f"restricted to it: {missing[:5]}"
        )

    for name in ref_keys:
        if tuple(ref_masks[name].shape) != tuple(src_masks[name].shape):
            raise SystemExit(f"shape mismatch on {name}: "
                             f"{tuple(ref_masks[name].shape)} vs {tuple(src_masks[name].shape)}")

    dropped = sorted(src_keys - ref_keys)
    out_masks = {name: src_masks[name].bool().clone() for name in sorted(ref_keys)}

    def totals(d):
        n = sum(int(v.numel()) for v in d.values())
        k = sum(int(v.bool().sum()) for v in d.values())
        return n, k

    n_src, k_src = totals(src_masks)
    n_out, k_out = totals(out_masks)
    n_ref, k_ref = totals(ref_masks)

    meta = dict(src_meta or {})
    meta.update({
        "subset_of": os.path.abspath(args.mask),
        "coverage_reference": os.path.abspath(args.reference),
        "coverage_note": (
            "Restricted to the reference mask's tensor coverage so that both arms leave the same "
            "tensors unmasked. Unmasked tensors receive dense AdamW updates "
            "(sparse_adamw.py:119-131), so coverage differences change the trainable parameter "
            "count."
        ),
        "dropped_tensors": dropped,
        "n_tensors": len(out_masks),
        "n_elements": n_out,
        "k_kept": k_out,
    })
    save_masks(out_masks, args.out, metadata=meta)

    report = {
        "source": {"path": args.mask, "tensors": len(src_masks), "elements": n_src, "kept": k_src,
                   "keep_rate": k_src / n_src},
        "reference": {"path": args.reference, "tensors": len(ref_masks), "elements": n_ref,
                      "kept": k_ref, "keep_rate": k_ref / n_ref},
        "output": {"path": args.out, "tensors": len(out_masks), "elements": n_out, "kept": k_out,
                   "keep_rate": k_out / n_out},
        "dropped_tensors": dropped,
    }
    print(f"source    : {len(src_masks)} tensors, {n_src:,} elements, "
          f"{k_src:,} kept ({k_src/n_src:.6f})")
    print(f"reference : {len(ref_masks)} tensors, {n_ref:,} elements, "
          f"{k_ref:,} kept ({k_ref/n_ref:.6f})")
    print(f"output    : {len(out_masks)} tensors, {n_out:,} elements, "
          f"{k_out:,} kept ({k_out/n_out:.6f})")
    print(f"dropped {len(dropped)} tensors: {dropped}")
    print(f"\nwrote {args.out}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
