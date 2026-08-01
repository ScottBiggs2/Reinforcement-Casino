#!/usr/bin/env python3
"""
Standalone mask coverage verifier.

Usage:
  python src/utils/verify_mask_coverage.py --model_name meta-llama/Llama-3.1-8B-Instruct --mask_file masks/foo.pt

This is used as a fast regression check and as a gate for new mask generators.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM

# Ensure repo root is importable (so `import src...` works even without PYTHONPATH).
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.utils.mask_coverage_report import compute_mask_coverage_report


def main(args) -> None:
    print(f"Loading model (CPU): {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"Loading mask file: {args.mask_file}")
    data = torch.load(args.mask_file, map_location="cpu", weights_only=False)

    report = compute_mask_coverage_report(model=model, masks=data, topk_missing=args.topk_missing)
    print(json.dumps(report, indent=2))

    if report["shape_mismatch_count"] != 0:
        raise SystemExit(2)
    if report["numel_covered_frac_2d"] < args.min_2d_coverage:
        raise SystemExit(3)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Verify model/mask coverage")
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--mask_file", type=str, required=True)
    p.add_argument("--topk_missing", type=int, default=20)
    p.add_argument("--min_2d_coverage", type=float, default=0.995)
    p.add_argument("--out_json", type=str, default=None)
    main(p.parse_args())

