#!/usr/bin/env python3
"""
GPU micro-test to verify Triton kernel caching for the BSR grad_weight kernel.

Expected behavior:
- First call: slow(er) due to compilation
- Second call (same signature): fast
- Third call after unrelated GPU work: still fast (no \"reset\")

Run:
  python src/sandbox/bsr_recompile_microtest.py
"""

import os
import time
import torch

from src.kernels.bsr_backward import sparse_weight_gradient_triton, bsr_recompile_diag_summary


def _time_call(fn, *, sync: bool = True) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this micro-test.")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    if sync:
        torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def main() -> None:
    assert torch.cuda.is_available(), "CUDA runtime required"
    device = torch.device("cuda")

    # Stabilize constexprs for this test.
    os.environ.setdefault("BSR_USE_ATOMIC", "0")
    os.environ.setdefault("BSR_BATCH_CHUNKS", "8")
    os.environ.setdefault("BSR_BATCH_BLOCK_SIZE", "64")
    os.environ.setdefault("BSR_NUM_WARPS", "4")
    os.environ.setdefault("BSR_NUM_STAGES", "2")

    # Enable diag timing + sync for high confidence.
    os.environ.setdefault("RL_CASINO_BSR_RECOMPILE_DIAG", "1")
    os.environ.setdefault("RL_CASINO_BSR_RECOMPILE_DIAG_SYNC", "1")
    os.environ["RL_CASINO_BSR_RECOMPILE_DIAG_STEP"] = "0"

    torch.manual_seed(0)
    B = 1024
    out_dim = 4096
    in_dim = 4096
    block = 16

    grad_out = torch.randn((B, out_dim), device=device, dtype=torch.bfloat16)
    inp = torch.randn((B, in_dim), device=device, dtype=torch.bfloat16)

    # Block-structured mask with a small number of active blocks to mimic extreme sparsity.
    blocks_m = (out_dim + block - 1) // block
    blocks_n = (in_dim + block - 1) // block
    block_mask = torch.zeros((blocks_m, blocks_n), device=device, dtype=torch.bool)
    # Activate a few blocks deterministically
    active_coords = [(0, 0), (1, 2), (10, 7), (25, 31), (127, 3)]
    for r, c in active_coords:
        block_mask[r % blocks_m, c % blocks_n] = True
    # Expand to element mask
    mask = (
        block_mask.repeat_interleave(block, 0)
        .repeat_interleave(block, 1)[:out_dim, :in_dim]
        .contiguous()
    )
    active_blocks = torch.nonzero(block_mask.flatten(), as_tuple=True)[0].to(torch.int32).contiguous()

    def call():
        _ = sparse_weight_gradient_triton(
            grad_out, inp, mask, active_blocks=active_blocks, block_size=block, use_tf32=True
        )

    # First call (compile expected)
    t1 = _time_call(call)

    # Second call, identical signature
    os.environ["RL_CASINO_BSR_RECOMPILE_DIAG_STEP"] = "1"
    t2 = _time_call(call)

    # Unrelated GPU work
    x = torch.randn((8192, 8192), device=device, dtype=torch.float16)
    y = x @ x.t()
    del x, y
    torch.cuda.synchronize()

    # Third call, still identical signature
    os.environ["RL_CASINO_BSR_RECOMPILE_DIAG_STEP"] = "2"
    t3 = _time_call(call)

    print(f"call1_ms={t1:.3f} call2_ms={t2:.3f} call3_ms={t3:.3f}")
    print(bsr_recompile_diag_summary(reset=False))


if __name__ == "__main__":
    main()

