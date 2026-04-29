#!/usr/bin/env python3
"""
Kernel-level breakdown of BSR backward at the user's training sparsity (rho=0.025).

Goal: of the ~0.6 ms backward time at rho=0.025, how much is:
  (a) sparse_weight_gradient_triton (our custom Triton kernel)
  (b) grad_input = grad_out @ W (dense matmul, runs in F.linear backward)
  (c) autograd / Python overhead (full bwd time minus a+b)

This decides whether kernel optimization (B1, fused, 8-bit) is worth it
or whether bottleneck is elsewhere (input-grad path / autograd).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.kernels.bsr_backward import sparse_grad_input_triton, sparse_weight_gradient_triton
from src.mlps.bsr_sparse_mlp import SparseLinearLayer


def make_block_mask(out_dim: int, in_dim: int, rho: float, block_size: int = 16, device="cuda"):
    nbm = (out_dim + block_size - 1) // block_size
    nbn = (in_dim + block_size - 1) // block_size
    block_active = (torch.rand(nbm, nbn, device=device) < rho)
    mask = block_active.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    return mask[:out_dim, :in_dim].to(torch.bool)


def cuda_time(fn, n_warmup=10, n_iters=200):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iters


def bench_one_shape(name, in_dim, out_dim, B, rho, dtype, device, n_warmup, n_iters):
    print(f"\n{'='*70}\n{name}  in={in_dim}  out={out_dim}  B={B}  rho={rho}\n{'='*70}")

    x = torch.randn(B, in_dim, device=device, dtype=dtype)
    w = torch.randn(out_dim, in_dim, device=device, dtype=dtype) * 0.02
    grad_out = torch.randn(B, out_dim, device=device, dtype=dtype)
    mask = make_block_mask(out_dim, in_dim, rho=rho, device=device)
    w_masked = w * mask.to(dtype)

    # ---- (1) DENSE: F.linear forward + autograd full backward ----
    x_d = x.detach().requires_grad_(True)
    w_d = w.clone().detach().requires_grad_(True)
    def dense_full():
        if x_d.grad is not None: x_d.grad = None
        if w_d.grad is not None: w_d.grad = None
        out = F.linear(x_d, w_d)
        out.backward(grad_out)
    dense_full_ms = cuda_time(dense_full, n_warmup, n_iters)

    # ---- (2) DENSE: just grad_input = grad_out @ W ----
    def dense_input_grad():
        return grad_out @ w
    dense_input_grad_ms = cuda_time(dense_input_grad, n_warmup, n_iters)

    # ---- (3) DENSE: just grad_weight = grad_out.T @ x  (cublas) ----
    def dense_weight_grad():
        return grad_out.T @ x
    dense_weight_grad_ms = cuda_time(dense_weight_grad, n_warmup, n_iters)

    # ---- (4) BSR full bwd, dense input-grad mode (current main branch) ----
    bsr = SparseLinearLayer(in_dim, out_dim, bias=False, mask=mask, block_size=16, use_tf32=False).to(device=device, dtype=dtype)
    with torch.no_grad():
        bsr.weight.copy_(w_masked)
    x_b = x.detach().requires_grad_(True)
    def bsr_full_dense_ig():
        os.environ["RL_CASINO_BSR_GRAD_INPUT_MODE"] = "dense"
        if x_b.grad is not None: x_b.grad = None
        if bsr.weight.grad is not None: bsr.weight.grad = None
        out = bsr(x_b)
        out.backward(grad_out)
    bsr_full_ms = cuda_time(bsr_full_dense_ig, n_warmup, n_iters)

    # ---- (4b) BSR full bwd, SPARSE input-grad mode (B1 atomic baseline) ----
    def bsr_full_sparse_ig():
        os.environ["RL_CASINO_BSR_GRAD_INPUT_MODE"] = "sparse"
        if x_b.grad is not None: x_b.grad = None
        if bsr.weight.grad is not None: bsr.weight.grad = None
        out = bsr(x_b)
        out.backward(grad_out)
    bsr_full_b1_ms = cuda_time(bsr_full_sparse_ig, n_warmup, n_iters)

    # ---- (5) Just sparse_weight_gradient_triton (custom kernel only) ----
    def bsr_kernel_only():
        return sparse_weight_gradient_triton(grad_out, x, mask, block_size=16, use_tf32=False)
    bsr_kernel_ms = cuda_time(bsr_kernel_only, n_warmup, n_iters)

    # ---- (6) Just sparse_grad_input_triton (B1 input-grad kernel only) ----
    def b1_input_grad_only():
        return sparse_grad_input_triton(grad_out, w_masked, mask, block_size=16, use_tf32=False)
    b1_kernel_ms = cuda_time(b1_input_grad_only, n_warmup, n_iters)

    # ---- Print breakdown ----
    print(f"\n  DENSE BREAKDOWN")
    print(f"    full bwd (F.linear autograd) : {dense_full_ms:.3f} ms")
    print(f"    just grad_input (gO @ W)     : {dense_input_grad_ms:.3f} ms  ({dense_input_grad_ms/dense_full_ms*100:.0f}% of full)")
    print(f"    just grad_weight (gO.T @ x)  : {dense_weight_grad_ms:.3f} ms  ({dense_weight_grad_ms/dense_full_ms*100:.0f}% of full)")
    print(f"    fwd + autograd overhead      : {dense_full_ms - dense_input_grad_ms - dense_weight_grad_ms:.3f} ms")

    print(f"\n  BSR-16 (DENSE input-grad — current branch baseline)")
    print(f"    full bwd                     : {bsr_full_ms:.3f} ms")
    print(f"    just sparse_weight_grad      : {bsr_kernel_ms:.3f} ms  ({bsr_kernel_ms/bsr_full_ms*100:.0f}%)")
    print(f"    dense gO @ W                 : {dense_input_grad_ms:.3f} ms  ({dense_input_grad_ms/bsr_full_ms*100:.0f}%)")
    print(f"    autograd overhead            : {bsr_full_ms - bsr_kernel_ms - dense_input_grad_ms:.3f} ms")

    print(f"\n  BSR-16 (B1 SPARSE input-grad — Scott atomic kernel)")
    print(f"    full bwd                     : {bsr_full_b1_ms:.3f} ms")
    print(f"    just sparse_weight_grad      : {bsr_kernel_ms:.3f} ms  ({bsr_kernel_ms/bsr_full_b1_ms*100:.0f}%)")
    print(f"    sparse_grad_input_triton     : {b1_kernel_ms:.3f} ms  ({b1_kernel_ms/bsr_full_b1_ms*100:.0f}%)")
    print(f"    autograd overhead            : {bsr_full_b1_ms - bsr_kernel_ms - b1_kernel_ms:.3f} ms")

    print(f"\n  SPEEDUPS vs DENSE")
    print(f"    BSR (dense ig)               : {dense_full_ms / bsr_full_ms:.2f}×")
    print(f"    BSR + B1 (sparse ig)         : {dense_full_ms / bsr_full_b1_ms:.2f}×")
    print(f"    input-grad kernel speedup    : {dense_input_grad_ms / b1_kernel_ms:.2f}×  (B1 vs cublas dense)")
    print(f"    weight-grad kernel speedup   : {dense_weight_grad_ms / bsr_kernel_ms:.2f}×  (vs cublas)")

    return {
        "shape": name,
        "rho": rho,
        "dense_full_ms": dense_full_ms,
        "dense_input_grad_ms": dense_input_grad_ms,
        "dense_weight_grad_ms": dense_weight_grad_ms,
        "bsr_full_ms": bsr_full_ms,
        "bsr_full_b1_ms": bsr_full_b1_ms,
        "bsr_kernel_ms": bsr_kernel_ms,
        "b1_kernel_ms": b1_kernel_ms,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--rho", type=float, default=0.025)
    p.add_argument("--batch_tokens", type=int, default=2048)
    p.add_argument("--n_warmup", type=int, default=10)
    p.add_argument("--n_iters", type=int, default=200)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"
    dtype = torch.bfloat16

    print(f"PyTorch {torch.__version__}  CUDA {torch.version.cuda}  device={torch.cuda.get_device_name(0)}")
    print(f"Config: B={args.batch_tokens}  rho={args.rho}  bf16  warmup={args.n_warmup}  iters={args.n_iters}")

    shapes = [
        ("gate/up_proj", 4096, 14336),
        ("down_proj",    14336, 4096),
    ]

    rows = []
    for name, in_dim, out_dim in shapes:
        rows.append(bench_one_shape(name, in_dim, out_dim, args.batch_tokens, args.rho, dtype, device, args.n_warmup, args.n_iters))

    # Summary
    print(f"\n{'='*70}\nSUMMARY (rho={args.rho})\n{'='*70}")
    print(f"  {'shape':<14} {'dense':>10} {'bsr':>10} {'bsr+B1':>10} {'bsr_sp':>9} {'B1_sp':>9}")
    for r in rows:
        sp = r["dense_full_ms"] / r["bsr_full_ms"]
        sp_b1 = r["dense_full_ms"] / r["bsr_full_b1_ms"]
        print(f"  {r['shape']:<14} {r['dense_full_ms']:>8.3f}ms {r['bsr_full_ms']:>8.3f}ms {r['bsr_full_b1_ms']:>8.3f}ms {sp:>8.2f}× {sp_b1:>8.2f}×")

    # Markdown out
    md_path = os.path.join(args.output_dir, "kernel_breakdown.md")
    with open(md_path, "w") as f:
        f.write(f"# BSR backward kernel breakdown @ rho={args.rho}\n\n")
        f.write(f"`{torch.cuda.get_device_name(0)}` / bf16 / B={args.batch_tokens}\n\n")
        f.write("| shape | dense_full | bsr (dense ig) | bsr+B1 (sparse ig) | speedup baseline | speedup +B1 | input-grad kernel speedup |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            sp = r["dense_full_ms"] / r["bsr_full_ms"]
            sp_b1 = r["dense_full_ms"] / r["bsr_full_b1_ms"]
            ig_sp = r["dense_input_grad_ms"] / r["b1_kernel_ms"]
            f.write(f"| {r['shape']} | {r['dense_full_ms']:.3f} | {r['bsr_full_ms']:.3f} | {r['bsr_full_b1_ms']:.3f} | {sp:.2f}× | {sp_b1:.2f}× | {ig_sp:.2f}× |\n")
        f.write("\n**bsr (dense ig)**: current main branch (input-grad still dense `gO @ W` via cublas)\n\n")
        f.write("**bsr+B1 (sparse ig)**: input-grad routed through `sparse_grad_input_triton` (Scott atomic baseline ported from cav_fixes b9fec55)\n\n")
        f.write("**input-grad kernel speedup**: dense cublas `gO @ W` time / `sparse_grad_input_triton` time. > 1× means B1 helps.\n")
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
