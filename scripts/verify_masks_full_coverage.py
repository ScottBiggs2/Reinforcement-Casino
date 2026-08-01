#!/usr/bin/env python3
"""
Batch gate: assert every mask in a set covers *all* of a model's 2-D weight tensors.

This is the check that would have caught RESULTS.md §3.2 — the delta-log builder silently
dropping attention projections, so a mask covered 213/226 tensors and the 13 missing ones were
then trained densely (§3.3). ``src/utils/verify_mask_coverage.py`` does the same coverage math
for ONE mask but reloads the fp32 model each call; here the model is loaded once and every mask is
checked against it, which is what you want right after a rebuild sweep.

A mask PASSES when, over the model's 2-D weights:
  - numel_covered_frac_2d == 1.0 (every 2-D weight tensor has a mask entry), and
  - no 2-D tensor is missing, and
  - no shape mismatch.
1-D tensors (RMSNorm/QK-norm, biases) are expected to be absent — they are never maskable — so
they are ignored by this gate.

Usage:
  python scripts/verify_masks_full_coverage.py \
      --model_name allenai/OLMo-3-7B-Instruct \
      --masks /scratch/$USER/rl_casino_masks/olmo3_7b_light_r1/*.pt \
      --out_json /scratch/$USER/rl_casino_analysis/mask_coverage_gate.json

Exits non-zero if ANY mask fails, so it drops straight into a Slurm dependency chain before the
training arms are submitted.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoModelForCausalLM

from src.utils.mask_coverage_report import compute_mask_coverage_report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name", required=True, help="HF id or local dir; loaded once on CPU.")
    p.add_argument("--masks", nargs="+", required=True,
                   help="Mask .pt paths and/or globs. All are checked against the one model.")
    p.add_argument("--out_json", default=None, help="Optional path to write the full report.")
    p.add_argument("--topk_missing", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()

    paths = []
    for spec in args.masks:
        hits = sorted(glob.glob(spec))
        paths.extend(hits if hits else [spec])
    # de-dup, preserve order (assign `seen` first — a single tuple-unpack line evaluates the RHS
    # before binding, so the comprehension would hit an unbound `seen`).
    seen = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]
    missing_files = [p for p in paths if not os.path.isfile(p)]
    if missing_files:
        raise SystemExit(f"mask file(s) not found: {missing_files}")
    if not paths:
        raise SystemExit("no mask files matched")

    print(f"Loading model (CPU, fp32): {args.model_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, device_map=None, low_cpu_mem_usage=True
    )
    model.eval()
    n_model_2d = sum(1 for _, p in model.named_parameters() if p.dim() == 2)
    print(f"Model has {n_model_2d} two-dimensional weight tensors (the maskable universe).\n")

    results, all_pass = {}, True
    print(f"{'mask':<52} {'2D cov':>8} {'2D miss':>8} {'shape≠':>7}  verdict")
    print("-" * 92)
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        rep = compute_mask_coverage_report(model=model, masks=data, topk_missing=args.topk_missing)
        missing_2d = [m for m in rep["missing_topk_by_numel"] if m["dim"] == 2]
        cov2d = rep["numel_covered_frac_2d"]
        ok = (cov2d >= 0.999999) and (not missing_2d) and (rep["shape_mismatch_count"] == 0)
        all_pass = all_pass and ok
        rep["_verdict"] = "PASS" if ok else "FAIL"
        rep["_missing_2d_names"] = [m["name"] for m in missing_2d]
        results[path] = rep
        label = os.path.basename(path)
        label = label if len(label) <= 51 else "…" + label[-50:]
        print(f"{label:<52} {cov2d:>8.4f} {len(missing_2d):>8} "
              f"{rep['shape_mismatch_count']:>7}  {'✓ PASS' if ok else '✗ FAIL'}")
        if not ok and missing_2d:
            for m in missing_2d[:8]:
                print(f"      dropped 2-D tensor: {m['name']}  ({m['numel']:,} params → trained DENSE)")

    print("-" * 92)
    print(f"\n{'ALL MASKS PASS' if all_pass else 'ONE OR MORE MASKS FAILED'} "
          f"({sum(r['_verdict']=='PASS' for r in results.values())}/{len(results)} pass)")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"model_name": args.model_name, "n_model_2d": n_model_2d,
                       "all_pass": all_pass, "masks": results}, f, indent=2)
        print(f"wrote {args.out_json}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
