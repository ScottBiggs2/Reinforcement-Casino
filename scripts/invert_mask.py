#!/usr/bin/env python3
"""CLI: write complement mask (1 - M) without modifying the source file.

Default output: <stem>_inverse.pt next to the input (same format as the source).

Examples (repo root):

  python scripts/invert_mask.py masks/cold_fisher_x_sparsity97.5pct_n256.pt
  python scripts/invert_mask.py masks/warm_magnitude_....pt -o /tmp/custom_inverse.pt
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils.mask_utils import invert_mask_file  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Invert a binary inclusion mask .pt (1=include, 0=exclude): "
            "output tensor is 1 where the input was 0 and 0 where the input was 1."
        )
    )
    p.add_argument(
        "input",
        help="Path to mask .pt (wrapped {'masks', 'metadata'} or legacy dict)",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (default: <input_stem>_inverse.pt beside the input)",
    )
    args = p.parse_args()
    out = invert_mask_file(args.input, args.output)
    print(out, file=sys.stdout)


if __name__ == "__main__":
    main()
