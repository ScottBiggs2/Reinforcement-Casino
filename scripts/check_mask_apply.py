#!/usr/bin/env python3
"""Sanity check that a mask file matches model parameter names/shapes.

Usage:
    python scripts/check_mask_apply.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --mask masks/cold_snip_llama_90pct.pt
"""
import argparse
import torch
from transformers import AutoModelForCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--mask", required=True)
    args = p.parse_args()

    print(f"[load] model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    print(f"[load] mask: {args.mask}")
    data = torch.load(args.mask, map_location="cpu", weights_only=False)
    mask_dict = data["masks"] if isinstance(data, dict) and "masks" in data else data

    param_names = {n for n, _ in model.named_parameters()}
    mask_keys = set(mask_dict.keys())

    matched = mask_keys & param_names
    missing = mask_keys - param_names
    print(f"\n[check] mask keys matched: {len(matched)}/{len(mask_keys)}")
    print(f"[check] missing (in mask but not in model): {len(missing)}")
    if missing:
        print("  examples:", list(missing)[:5])

    print("\n[check] per-param shape + keep_frac (first 3 matched):")
    shown = 0
    total_kept, total_elems = 0, 0
    for n, p in model.named_parameters():
        if n not in mask_dict:
            continue
        m = mask_dict[n]
        assert m.shape == p.shape, (
            f"SHAPE MISMATCH {n}: mask {tuple(m.shape)} vs param {tuple(p.shape)}"
        )
        keep_frac = (m == 1).float().mean().item()
        total_kept += int((m == 1).sum().item())
        total_elems += m.numel()
        if shown < 3:
            print(f"  {n}: shape OK, keep_frac={keep_frac:.3f}")
            shown += 1

    print(f"\n[check] overall keep_frac across all matched params: "
          f"{total_kept / max(total_elems, 1):.4f}  "
          f"(sparsity = {1 - total_kept / max(total_elems, 1):.4f})")
    print("\n[check] done.")


if __name__ == "__main__":
    main()
