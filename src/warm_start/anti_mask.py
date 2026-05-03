"""
Anti-mask generator: per-layer complement of an oracle mask.

For each masked param, output_mask = 1 - input_mask. Same shapes, same
keys, complementary density (e.g. 97.5% sparse oracle -> 2.5% sparse
anti-mask). This is the necessity control for probe_pair_masks.py: keeping
ONLY the anti-mask weights (or, with --patch_mode anti_delta_only, applying
RL Δθ on the complement) should DESTROY the probe signal if M genuinely
identifies the task subnetwork.

Usage:
    python src/warm_start/anti_mask.py \\
        --reference_mask /path/to/oracle.pt \\
        --output_file   /path/to/anti_oracle.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.mask_utils import save_masks  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Per-layer anti-mask (1-M) generator")
    p.add_argument("--reference_mask", required=True,
                   help="Oracle/CAV/etc. mask .pt to invert.")
    p.add_argument("--output_file", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[anti_mask] loading {args.reference_mask}")
    ref = torch.load(args.reference_mask, map_location="cpu", weights_only=False)
    ref_masks = ref["masks"] if isinstance(ref, dict) and "masks" in ref else ref
    ref_meta = ref.get("metadata", {}) if isinstance(ref, dict) else {}

    anti = {}
    total = 0
    kept = 0
    for name, m in ref_masks.items():
        if not torch.all((m == 0) | (m == 1)):
            raise ValueError(
                f"{name} is non-binary; anti-mask requires {{0,1}} input."
            )
        am = (1 - m.to(torch.uint8)).to(m.dtype)
        anti[name] = am
        total += am.numel()
        kept += int(am.sum().item())

    density = kept / total if total else 0.0
    print(f"[anti_mask] params: {len(anti)}  kept/total: {kept}/{total}  "
          f"density: {density:.4%} (anti-density)")

    metadata = dict(ref_meta)
    metadata.update({
        "method": "anti_mask",
        "reference_mask": args.reference_mask,
        "anti_density": density,
        "original_method": ref_meta.get("method", "unknown"),
    })

    save_masks(anti, args.output_file, metadata)
    print(f"[anti_mask] saved -> {args.output_file}")


if __name__ == "__main__":
    main()
