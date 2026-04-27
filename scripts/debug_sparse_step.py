#!/usr/bin/env python3
"""
Run one full train-style step and check after each substep where NaN appears.

Substeps:
  1. forward (already known clean from debug_sparse_forward.py)
  2. fake loss + backward → check param.grad for NaN/Inf
  3. optimizer step → check param weights for NaN/Inf
  4. second forward → check logits for NaN

If (2) is clean but (3) corrupts → block_sparse_adam_2d_kernel is the bug.
If (2) shows NaN grads → BSR backward kernel is producing them.
If both clean and (4) NaN → something in TRL/dpo loop, not these kernels.

Uses 90% block-sparse mask (real sparse path, not all-1s).
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.mlps.bsr_sparse_mlp import replace_linear_modules
from src.optimizers.sparse_adamw import SparseAdamW
from src.utils.mask_manager import SparseMaskManager


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "The capital of France is"
SEED = 42


def check_nan(label, t):
    nan = torch.isnan(t).sum().item()
    inf = torch.isinf(t).sum().item()
    if nan or inf:
        print(f"  ❌ {label}: NaN={nan} Inf={inf} shape={tuple(t.shape)}")
        return True
    print(f"  ✓ {label}: clean (mean={t.float().mean().item():+.4e} max={t.float().max().item():+.4e})")
    return False


def check_params_nan(model, label):
    bad = []
    for name, p in model.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            bad.append(name)
    if bad:
        print(f"  ❌ {label}: NaN/Inf in {len(bad)} params, e.g. {bad[:3]}")
        return True
    print(f"  ✓ {label}: all params finite")
    return False


def check_grads_nan(model, label):
    bad = []
    sane_count = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            bad.append(name)
        else:
            sane_count += 1
    if bad:
        print(f"  ❌ {label}: NaN/Inf in {len(bad)} grads, e.g. {bad[:3]}")
        return True
    print(f"  ✓ {label}: {sane_count} grads, all finite")
    return False


def build_block_mask(shape, sparsity, block_size, device):
    out_dim, in_dim = int(shape[0]), int(shape[1])
    g = torch.Generator(device="cpu").manual_seed(SEED)
    nb_out = max(1, out_dim // block_size)
    nb_in = max(1, in_dim // block_size)
    n = nb_out * nb_in
    keep = torch.zeros(n, dtype=torch.bool)
    keep[torch.randperm(n, generator=g)[:max(1, int(round((1 - sparsity) * n)))]] = True
    keep = keep.view(nb_out, nb_in)
    m = keep.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    return m[:out_dim, :in_dim].contiguous().to(device)


def main():
    device = torch.device("cuda")
    torch.manual_seed(SEED)

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True
    )
    model.to(device)
    model.train()
    print(f"  on {device} dtype={model.dtype}")

    # Build 90% mask, mlp-only.
    print("\nBuilding 90% block-sparse mlp-only mask...")
    masks = {}
    for name, p in model.named_parameters():
        if "weight" not in name or p.dim() != 2:
            continue
        if "mlp" not in name.lower():
            continue
        masks[name] = build_block_mask(p.shape, 0.9, 16, p.device)
    print(f"  {len(masks)} layers")

    fd, tmp = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(masks, tmp)
    try:
        mm = SparseMaskManager(tmp, device=device)
        replace_linear_modules(model, mm.masks, block_size=16, use_tf32=True, verbose=False)

        check_params_nan(model, "params after replace_linear_modules")

        # ==================== Step 1: forward ====================
        print("\n=== STEP 1: forward ===")
        enc = tokenizer(PROMPT, return_tensors="pt").to(device)
        # Need labels for a loss; use input_ids shifted as a proxy
        out = model(**enc, labels=enc["input_ids"])
        check_nan("logits", out.logits)
        loss = out.loss
        check_nan("loss", loss.unsqueeze(0))

        # ==================== Step 2: backward ====================
        print("\n=== STEP 2: backward ===")
        loss.backward()
        if check_grads_nan(model, "grads after backward"):
            print("  → BSR backward kernel is producing NaN gradients. Bug here.")
            return

        # ==================== Step 3: optimizer ====================
        print("\n=== STEP 3: optimizer ===")
        opt = SparseAdamW(
            model.named_parameters(),
            mask_manager=mm,
            lr=5e-7,
            block_size=16,
            mlp_only=True,
        )
        opt.step()
        opt.zero_grad()
        if check_params_nan(model, "params after optimizer.step()"):
            print("  → block_sparse_adam_2d_kernel is corrupting weights. Bug here.")
            return

        # ==================== Step 4: second forward ====================
        print("\n=== STEP 4: second forward (after one full update) ===")
        out2 = model(**enc, labels=enc["input_ids"])
        if check_nan("logits", out2.logits):
            print("  → Forward produces NaN AFTER optimizer touched weights, even though weights are finite.")
            print("  → Subtle issue, likely in mask buffer or gradient checkpointing or activations.")
            return
        check_nan("loss", out2.loss.unsqueeze(0))

        print("\n✓ ✓ ✓  Single full step is clean. Bug must appear after ≥2 steps.")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    main()
