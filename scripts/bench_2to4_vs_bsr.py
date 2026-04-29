#!/usr/bin/env python3
"""
A2 bench: 2:4 structured sparse (torchao + cuSPARSELt) vs our 16x16 BSR vs dense.

Measures forward + backward time per linear layer at Llama-3.1-8B MLP shapes,
matched at rho=0.5 sparsity. Single H200, no model loading, random tensors.

Output: CSV + console table at <output_dir>/bench_2to4_vs_bsr.{csv,md}
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mlps.bsr_sparse_mlp import SparseLinearLayer

try:
    from torchao.sparsity import SparseSemiStructuredTensor
    from torchao.sparsity.training import SemiSparseLinear, swap_linear_with_semi_sparse_linear
    HAS_TORCHAO = True
except ImportError as e:
    HAS_TORCHAO = False
    _TORCHAO_ERR = str(e)


@dataclass
class Result:
    layer: str
    shape: str
    method: str
    fwd_ms: float
    bwd_ms: float
    total_ms: float
    peak_mem_mb: float
    output_max_abs_diff: float
    n_iters: int


def make_block_mask(out_dim: int, in_dim: int, rho: float, block_size: int = 16, device="cuda") -> torch.Tensor:
    nbm = (out_dim + block_size - 1) // block_size
    nbn = (in_dim + block_size - 1) // block_size
    block_active = (torch.rand(nbm, nbn, device=device) < rho)
    mask = block_active.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    return mask[:out_dim, :in_dim].to(torch.bool)


def make_2to4_mask(out_dim: int, in_dim: int, device="cuda") -> torch.Tensor:
    """2:4 pattern: in every group of 4 along the contraction dim, 2 are zero."""
    assert in_dim % 4 == 0, f"in_dim {in_dim} must be divisible by 4"
    g = torch.zeros(out_dim, in_dim, device=device, dtype=torch.bool)
    # In each row, first 2 of every 4 = True (deterministic; not random — 2:4 hardware path is fixed pattern)
    g.view(out_dim, in_dim // 4, 4)[..., :2] = True
    return g


def cuda_time(fn, n_warmup=5, n_iters=50):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters


def bench_module(
    module: nn.Module,
    x: torch.Tensor,
    label: str,
    n_warmup: int,
    n_iters: int,
    dense_output_ref: Optional[torch.Tensor] = None,
):
    module = module.cuda().to(x.dtype)
    x = x.detach().requires_grad_(True)

    # Numerical sanity (single shot)
    with torch.no_grad():
        out = module(x)
    if dense_output_ref is not None:
        diff = (out.float() - dense_output_ref.float()).abs().max().item()
    else:
        diff = 0.0

    # Forward time
    def _fwd():
        return module(x)
    fwd_ms = cuda_time(_fwd, n_warmup=n_warmup, n_iters=n_iters)

    # Forward + backward time
    grad_out = torch.randn_like(out)
    def _fwd_bwd():
        if x.grad is not None:
            x.grad = None
        for p in module.parameters():
            if p.grad is not None:
                p.grad = None
        y = module(x)
        y.backward(grad_out)

    torch.cuda.reset_peak_memory_stats()
    total_ms = cuda_time(_fwd_bwd, n_warmup=n_warmup, n_iters=n_iters)
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    bwd_ms = max(0.0, total_ms - fwd_ms)

    return {
        "method": label,
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "total_ms": total_ms,
        "peak_mem_mb": peak_mem_mb,
        "output_max_abs_diff": diff,
    }, out.detach()


def run_one_shape(
    name: str,
    in_dim: int,
    out_dim: int,
    batch_tokens: int,
    rho: float,
    dtype: torch.dtype,
    n_warmup: int,
    n_iters: int,
):
    device = "cuda"
    print(f"\n{'='*70}\nShape: {name}  in={in_dim}  out={out_dim}  B={batch_tokens}  rho={rho}  dtype={dtype}\n{'='*70}")

    x = torch.randn(batch_tokens, in_dim, device=device, dtype=dtype)
    weight_init = torch.randn(out_dim, in_dim, device=device, dtype=dtype) * 0.02

    results = []

    # ---- (1) Dense baseline ----
    dense = nn.Linear(in_dim, out_dim, bias=False).to(device=device, dtype=dtype)
    with torch.no_grad():
        dense.weight.copy_(weight_init)
    r, dense_out = bench_module(dense, x, "dense", n_warmup, n_iters, dense_output_ref=None)
    results.append(r)
    del dense
    torch.cuda.empty_cache()

    # ---- (2) Our BSR-16 ----
    bsr_mask = make_block_mask(out_dim, in_dim, rho=rho, block_size=16, device=device)
    bsr = SparseLinearLayer(in_dim, out_dim, bias=False, mask=bsr_mask, block_size=16, use_tf32=False)
    with torch.no_grad():
        bsr.weight.copy_(weight_init * bsr_mask.to(dtype))
    r, _ = bench_module(bsr, x, "bsr16", n_warmup, n_iters, dense_output_ref=dense_out * bsr_mask.to(dtype).any().float())
    # Note: BSR forward is dense F.linear, output won't equal dense unless weight is masked the same way.
    # We pre-masked weight above so output IS the masked-weight matmul.
    results.append(r)
    del bsr
    torch.cuda.empty_cache()

    # ---- (3) torchao 2:4 ----
    if HAS_TORCHAO:
        try:
            ao_lin = nn.Linear(in_dim, out_dim, bias=False).to(device=device, dtype=dtype)
            mask_2to4 = make_2to4_mask(out_dim, in_dim, device=device)
            with torch.no_grad():
                ao_lin.weight.copy_(weight_init * mask_2to4.to(dtype))

            # Wrap in a tiny holder so swap_linear_with_semi_sparse_linear can find it
            holder = nn.Sequential()
            holder.add_module("layer", ao_lin)
            swap_linear_with_semi_sparse_linear(holder, {"layer": SemiSparseLinear})

            r, _ = bench_module(holder, x, "2to4_torchao", n_warmup, n_iters, dense_output_ref=None)
            results.append(r)
            del holder, ao_lin
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  torchao 2:4 phase FAILED: {type(e).__name__}: {e}")
            results.append({
                "method": "2to4_torchao",
                "fwd_ms": float("nan"),
                "bwd_ms": float("nan"),
                "total_ms": float("nan"),
                "peak_mem_mb": 0.0,
                "output_max_abs_diff": 0.0,
            })
    else:
        print(f"  torchao not available: {_TORCHAO_ERR}")
        results.append({
            "method": "2to4_torchao_skipped",
            "fwd_ms": float("nan"),
            "bwd_ms": float("nan"),
            "total_ms": float("nan"),
            "peak_mem_mb": 0.0,
            "output_max_abs_diff": 0.0,
        })

    # Print
    print(f"\n  {'method':<22} {'fwd_ms':>10} {'bwd_ms':>10} {'total_ms':>12} {'peak_MB':>10}")
    for r in results:
        print(f"  {r['method']:<22} {r['fwd_ms']:>10.3f} {r['bwd_ms']:>10.3f} {r['total_ms']:>12.3f} {r['peak_mem_mb']:>10.1f}")

    rows = [
        Result(
            layer=name,
            shape=f"{out_dim}x{in_dim}",
            n_iters=n_iters,
            **r,
        )
        for r in results
    ]
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_tokens", type=int, default=2048, help="Total tokens (batch * seq_len), default matches sparse DPO bs=2 × max_len=1024")
    p.add_argument("--rho", type=float, default=0.5, help="BSR active block fraction (2:4 is fixed at 0.5 by hardware)")
    p.add_argument("--n_warmup", type=int, default=5)
    p.add_argument("--n_iters", type=int, default=100)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"PyTorch {torch.__version__}  CUDA {torch.version.cuda}  device={torch.cuda.get_device_name(0)}")
    print(f"torchao available: {HAS_TORCHAO}")
    print(f"Config: batch_tokens={args.batch_tokens}  rho={args.rho}  dtype={dtype}")
    print(f"        n_warmup={args.n_warmup}  n_iters={args.n_iters}")

    # Llama-3.1-8B MLP shapes
    shapes = [
        ("gate_proj_or_up_proj", 4096, 14336),  # 4096 in -> 14336 out
        ("down_proj",            14336, 4096),  # 14336 in -> 4096 out
    ]

    all_rows = []
    for name, in_dim, out_dim in shapes:
        rows = run_one_shape(name, in_dim, out_dim, args.batch_tokens, args.rho, dtype, args.n_warmup, args.n_iters)
        all_rows.extend(rows)

    # CSV
    csv_path = os.path.join(args.output_dir, "bench_2to4_vs_bsr.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(asdict(r))
    print(f"\nWrote {csv_path}")

    # Markdown summary table
    md_path = os.path.join(args.output_dir, "bench_2to4_vs_bsr.md")
    with open(md_path, "w") as f:
        f.write(f"# 2:4 vs BSR-16 vs dense bench\n\n")
        f.write(f"`{torch.cuda.get_device_name(0)}` / dtype={args.dtype} / rho={args.rho} / B={args.batch_tokens}\n\n")
        f.write(f"| layer | shape | method | fwd_ms | bwd_ms | total_ms | peak_MB | speedup_vs_dense |\n")
        f.write(f"|---|---|---|---:|---:|---:|---:|---:|\n")
        # Group by layer; compute speedup vs dense baseline within each layer
        by_layer: dict = {}
        for r in all_rows:
            by_layer.setdefault(r.layer, []).append(r)
        for layer, rs in by_layer.items():
            dense_total = next((r.total_ms for r in rs if r.method == "dense"), float("nan"))
            for r in rs:
                speedup = dense_total / r.total_ms if r.total_ms > 0 and r.total_ms == r.total_ms else float("nan")
                f.write(f"| {r.layer} | {r.shape} | {r.method} | {r.fwd_ms:.3f} | {r.bwd_ms:.3f} | {r.total_ms:.3f} | {r.peak_mem_mb:.1f} | {speedup:.2f}× |\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
