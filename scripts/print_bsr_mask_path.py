#!/usr/bin/env python3
"""Print canonical mask .pt path (matches h200_sparse_dpo_bsr_benchmark._mask_cache_path)."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.full_training.h200_sparse_dpo_bsr_benchmark import _mask_cache_path  # noqa: E402
from src.utils.mask_utils import DEFAULT_MIN_LAYER_KEEP_RATIO  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-cache-dir", type=str, required=True, help="Parent dir containing masks/")
    ap.add_argument("--sparsity", type=float, required=True)
    ap.add_argument("--mask-type", choices=["element", "block"], required=True)
    ap.add_argument("--block-size-bsr", type=int, default=16)
    ap.add_argument("--mlp-only", action="store_true")
    ap.add_argument("--min-layer-keep-ratio", type=float, default=DEFAULT_MIN_LAYER_KEEP_RATIO)
    args = ap.parse_args()
    mt = "block" if args.mask_type == "block" else "element"
    p = _mask_cache_path(
        out_dir=args.mask_cache_dir,
        sparsity_pct=args.sparsity,
        mask_type=mt,
        block_size=args.block_size_bsr,
        mlp_only=args.mlp_only,
        min_layer_keep_ratio=args.min_layer_keep_ratio,
    )
    print(p)


if __name__ == "__main__":
    main()
