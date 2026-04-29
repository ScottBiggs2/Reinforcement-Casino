#!/usr/bin/env python3
"""
GPU parity: dense grad_input vs sparse_grad_input_triton for block-masked weights.

Run from repo root:
  python src/sandbox/bsr_grad_input_parity.py

Expectations:
- fp32: tight match (max_abs ~1e-5 or better depending on GPU)
- bf16: larger tolerance; TF32 on dot paths adds noise

Environment:
  RL_CASINO_BSR_GRAD_INPUT_MODE is ignored here (script calls kernel directly).
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.kernels.bsr_backward import sparse_grad_input_triton


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a - b).detach()
    denom = b.detach().abs().max().clamp_min(1e-12)
    return {
        "max_abs": float(d.abs().max().item()),
        "max_rel": float((d.abs().max() / denom).item()),
        "cos": float(
            torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        ),
    }


def run_once(*, dtype: torch.dtype, use_tf32: bool, device: torch.device) -> None:
    torch.manual_seed(0)
    B = 512
    out_dim = 4096
    in_dim = 4096
    block = 16

    grad_out = torch.randn(B, out_dim, device=device, dtype=dtype)
    w = torch.randn(out_dim, in_dim, device=device, dtype=dtype)

    # Random block mask ~5% blocks kept
    bm = out_dim // block
    bn = in_dim // block
    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    block_keep = torch.rand((bm, bn), generator=g) < 0.05
    mask = (
        block_keep.repeat_interleave(block, 0)
        .repeat_interleave(block, 1)[:out_dim, :in_dim]
        .to(device)
    )
    with torch.no_grad():
        w.mul_(mask.to(w.dtype))

    # Make the reference path explicit. On H100/H200, PyTorch fp32 matmul often uses TF32
    # by default, which can create ~1e-1 max_abs deltas vs a true-fp32 accumulation kernel.
    # For the fp32 parity check, we disable TF32 so the dense reference is strict fp32.
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    if dtype == torch.float32:
        torch.backends.cuda.matmul.allow_tf32 = False
    dense = grad_out @ w
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    active_blocks = torch.nonzero(
        block_keep.flatten(), as_tuple=False
    ).squeeze(-1).to(torch.int32).to(device)

    os.environ.setdefault("BSR_BATCH_CHUNKS", "8")
    sparse = sparse_grad_input_triton(
        grad_out, w, mask, active_blocks=active_blocks, block_size=block, use_tf32=use_tf32
    )

    m = metrics(sparse, dense)
    print(f"dtype={dtype} use_tf32={use_tf32} metrics={m}")

    if dtype == torch.float32:
        assert m["max_abs"] < 1e-3, m
        assert m["cos"] > 0.9999, m
    else:
        # bf16 + optional TF32: looser bar
        assert m["max_abs"] < 0.25, m
        assert m["cos"] > 0.99, m


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    print("=== fp32 ===")
    run_once(dtype=torch.float32, use_tf32=False, device=device)
    print("=== bf16 ===")
    run_once(dtype=torch.bfloat16, use_tf32=True, device=device)
    print("OK")


if __name__ == "__main__":
    main()
