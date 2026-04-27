#!/usr/bin/env python3
"""
Multi-step debug — simulate TRL's grad_accum + gradient_checkpointing.

Single-step debug came out clean (params + grads + 2nd forward all finite).
Bug must accumulate across steps. This runs N actual optim steps (each =
GRAD_ACCUM micro forward+backward) with gradient_checkpointing enabled,
checking for NaN after every micro and every optimizer step.

Goal: identify the exact step / micro / substep where corruption first
appears.
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
PROMPTS = [
    "The capital of France is",
    "Python is a programming language that",
    "Photosynthesis is the process",
    "The theory of relativity",
]
SEED = 42
N_STEPS = 5
GRAD_ACCUM = 4


def has_nan_inf(t):
    return torch.isnan(t).any().item() or torch.isinf(t).any().item()


def first_bad_param(model):
    for name, p in model.named_parameters():
        if has_nan_inf(p):
            return name, "param"
        if p.grad is not None and has_nan_inf(p.grad):
            return name, "grad"
    return None, None


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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True
    )
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()  # MATCH the convergence test setting
    model.train()
    print(f"  on {device} dtype={model.dtype}  gradient_checkpointing=on")

    masks = {}
    for name, p in model.named_parameters():
        if "weight" not in name or p.dim() != 2:
            continue
        if "mlp" not in name.lower():
            continue
        masks[name] = build_block_mask(p.shape, 0.9, 16, p.device)
    print(f"  built {len(masks)} masks")

    fd, tmp = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(masks, tmp)
    try:
        mm = SparseMaskManager(tmp, device=device)
        replace_linear_modules(model, mm.masks, block_size=16, use_tf32=True, verbose=False)

        opt = SparseAdamW(
            model.named_parameters(),
            mask_manager=mm,
            lr=5e-7,
            block_size=16,
            mlp_only=True,
        )

        # Pre-tokenize a few inputs to cycle through.
        encoded = []
        for p in PROMPTS:
            enc = tokenizer(p, return_tensors="pt").to(device)
            encoded.append(enc)

        for step in range(1, N_STEPS + 1):
            print(f"\n========== OPTIM STEP {step} ==========")
            opt.zero_grad()

            for micro in range(GRAD_ACCUM):
                enc = encoded[(step * GRAD_ACCUM + micro) % len(encoded)]
                out = model(**enc, labels=enc["input_ids"])
                loss = out.loss / GRAD_ACCUM
                loss.backward()

                logits_nan = has_nan_inf(out.logits)
                loss_nan = has_nan_inf(out.loss)
                bad_name, bad_kind = first_bad_param(model)

                tag = f"micro {micro+1}/{GRAD_ACCUM}"
                if logits_nan or loss_nan or bad_name:
                    print(f"  [{tag}] ❌ logits_nan={logits_nan} loss_nan={loss_nan} "
                          f"bad_param={bad_name} ({bad_kind})")
                    print(f"    loss = {out.loss.item():.4e}")
                    if bad_name:
                        return
                else:
                    print(f"  [{tag}] ✓ loss={out.loss.item():.4e}")

            opt.step()
            bad_name, bad_kind = first_bad_param(model)
            if bad_name:
                print(f"  optimizer.step() → ❌ {bad_name} ({bad_kind}) corrupted")
                return
            print(f"  ✓ optimizer.step() done, all params finite")

        print("\n✓ ✓ ✓  All N_STEPS clean. Bug needs more steps to appear.")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    main()
