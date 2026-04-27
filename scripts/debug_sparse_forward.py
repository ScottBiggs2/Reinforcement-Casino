#!/usr/bin/env python3
"""
Isolate where sparse path goes NaN.

Steps:
  1. Load Llama-3.1-8B (bf16, CUDA).
  2. Run a forward pass on a FIXED token sequence with the unmodified model.
     Save baseline output stats: mean / std / min / max / num_nan.
  3. Replace MLP linears with SparseLinearLayer carrying an ALL-ONES mask
     (i.e., dense compute, mask never skips anything). Run the same forward.
     Compare to baseline. If they diverge → SparseLinearFunction autograd
     wrapper itself is the bug, before any sparsity actually kicks in.
  4. Repeat with a real 90% block-sparse mask. If THIS diverges from step 3
     but step 3 matches step 2 → mask handling / kernel skip is the bug.

No optimizer involvement. No training. Pure forward sanity check.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.mlps.bsr_sparse_mlp import replace_linear_modules
from src.utils.mask_manager import SparseMaskManager


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "The capital of France is"
SEED = 42


def stats(name: str, t: torch.Tensor) -> None:
    """Print stats; if any NaN/Inf, mark loudly."""
    nan = torch.isnan(t).sum().item()
    inf = torch.isinf(t).sum().item()
    if nan or inf:
        print(f"  {name}: NaN={nan} Inf={inf}  shape={tuple(t.shape)} dtype={t.dtype}  ❌")
    else:
        finite = t.float()
        print(
            f"  {name}: mean={finite.mean().item():+.4e} "
            f"std={finite.std().item():.4e} "
            f"min={finite.min().item():+.4e} "
            f"max={finite.max().item():+.4e} "
            f"shape={tuple(t.shape)} dtype={t.dtype}"
        )


def fwd_and_stats(model, tokenizer, label: str):
    """Run forward, return last-layer logits stats."""
    print(f"\n=== {label} ===")
    enc = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc)
    stats("logits", out.logits)
    return out.logits.detach().clone()


def build_block_mask(shape, sparsity: float, block_size: int, all_ones: bool, device) -> torch.Tensor:
    """Bool mask, block-aligned. all_ones=True → fully dense."""
    out_dim, in_dim = int(shape[0]), int(shape[1])
    if all_ones:
        return torch.ones(out_dim, in_dim, dtype=torch.bool, device=device)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    nb_out = max(1, out_dim // block_size)
    nb_in = max(1, in_dim // block_size)
    n_blocks = nb_out * nb_in
    n_keep = max(1, int(round((1.0 - sparsity) * n_blocks)))
    perm = torch.randperm(n_blocks, generator=g)[:n_keep]
    keep = torch.zeros(n_blocks, dtype=torch.bool)
    keep[perm] = True
    keep = keep.view(nb_out, nb_in)
    mask = keep.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    return mask[:out_dim, :in_dim].contiguous().to(device)


def build_mask_dict(model, sparsity: float, block_size: int, mlp_only: bool, all_ones: bool):
    masks = {}
    for name, p in model.named_parameters():
        if "weight" not in name or p.dim() != 2:
            continue
        if mlp_only and "mlp" not in name.lower():
            continue
        masks[name] = build_block_mask(p.shape, sparsity, block_size, all_ones, p.device)
    return masks


def main():
    device = torch.device("cuda")
    torch.manual_seed(SEED)

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    print(f"  on {device} dtype={model.dtype}")

    # 1. Baseline forward (unmodified Llama).
    baseline = fwd_and_stats(model, tokenizer, "BASELINE  (unmodified Llama-3.1-8B)")

    # 2. Wrap MLP linears with SparseLinearLayer + all-ones mask.
    print("\nReplacing MLP linears with SparseLinearLayer (mask = all 1s)...")
    masks_ones = build_mask_dict(model, sparsity=0.0, block_size=16, mlp_only=True, all_ones=True)
    print(f"  {len(masks_ones)} layers to wrap")

    # Save masks via SparseMaskManager so we go through the same code path.
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(masks_ones, tmp)
    try:
        mm_ones = SparseMaskManager(tmp, device=device)
        replace_linear_modules(model, mm_ones.masks, block_size=16, use_tf32=True, verbose=False)
        print("  done")
        wrapped_dense = fwd_and_stats(model, tokenizer, "WRAPPED  (SparseLinearLayer, mask=all-1s, mlp-only)")

        diff = (baseline.float() - wrapped_dense.float()).abs()
        print(f"\nbaseline vs wrapped-dense:")
        stats("absdiff", diff)
        if torch.isnan(wrapped_dense).any():
            print("  → Forward through SparseLinearFunction wrapper produced NaN even with all-1s mask.")
            print("  → Bug is in SparseLinearFunction.forward / replace_linear_modules itself.")
            return
        if diff.max().item() > 0.5:
            print("  → Wrapped output diverged from baseline despite mask=1.")
            print("  → SparseLinearFunction wrapper is changing outputs. Suspect.")
        else:
            print("  → Wrapped output ≈ baseline (within bf16 noise). Wrapper is innocent so far.")

    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    main()
