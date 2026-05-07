#!/usr/bin/env python3
"""
Export a raw (non-normalized) Fisher diagonal proxy for H_ii for the alignment-optimal scoring algorithm.

This is a thin wrapper around src/cold_start/cold_mask_finder.py's Fisher accumulator with per-layer
normalization DISABLED, and writes a torch-save dict name->tensor (fp32) that alignment code expects.

Note: this script is GPU + dataset heavy; run it as its own Slurm job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

import os
import sys

# Ensure imports work when invoked from sbatch --wrap or any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.cold_start.cold_mask_finder import load_calibration_data, compute_fisher_scores, sanitize_model_name
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export raw Fisher diagonal for alignment scoring.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="qihoo360/Light-R1-DPOData")
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--mini-batch-size", type=int, default=4)
    ap.add_argument("--mlp-only", action="store_true")
    ap.add_argument("--out", type=str, required=True, help="Output .pt path for fisher diag.")
    ap.add_argument("--meta-out", type=str, default=None, help="Optional JSON metadata output path.")
    ap.add_argument("--force-cpu", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print("device:", device)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    data = load_calibration_data(args.dataset, tok, n_samples=int(args.n_samples), max_length=int(args.max_length), device=device)
    fisher_scores: Dict[str, torch.Tensor] = compute_fisher_scores(
        model,
        data,
        device=device,
        mlp_only=bool(args.mlp_only),
        mini_batch_size=int(args.mini_batch_size),
        normalize_per_layer=False,  # CRITICAL for alignment absolute scale
    )
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fisher_scores, out_p)
    print("Wrote fisher diag:", out_p)

    meta = {
        "method": "empirical_fisher_diag_raw",
        "model": args.model,
        "model_sanitized": sanitize_model_name(args.model),
        "dataset": args.dataset,
        "n_samples": int(args.n_samples),
        "max_length": int(args.max_length),
        "mini_batch_size": int(args.mini_batch_size),
        "mlp_only": bool(args.mlp_only),
        "normalize_per_layer": False,
        "device": device,
        "dtype": "float32",
    }
    meta_path = Path(args.meta_out) if args.meta_out else out_p.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote meta:", meta_path)


if __name__ == "__main__":
    main()

