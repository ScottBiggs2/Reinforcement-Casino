#!/usr/bin/env python3
"""
Microbenchmark: optimizer.step() cost only (no forward/backward).

This isolates the part SparseAdamW accelerates and is useful when end-to-end DPO
step time is dominated by dense forward/backward.

It creates synthetic 2D weight tensors (matching typical model matrix shapes)
and runs optimizer.step() repeatedly with fixed random gradients.

Outputs a CSV and Markdown summary under --out-dir.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import sys

from src.optimizers.sparse_adamw import SparseAdamW
from src.utils.mask_manager import SparseMaskManager


@dataclass
class Case:
    name: str
    optimizer: str  # adamw_torch, adamw_8bit, sparse_adamw
    mask_path: Optional[str] = None


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_steps(opt, n: int, warmup: int, sync_cuda: bool) -> Tuple[float, float]:
    # returns (mean_ms, p50_ms) across measured steps
    times: List[float] = []
    for i in range(warmup + n):
        if sync_cuda:
            _sync()
        t0 = time.perf_counter()
        opt.step()
        if sync_cuda:
            _sync()
        dt = (time.perf_counter() - t0) * 1e3
        if i >= warmup:
            times.append(dt)
    times.sort()
    mean = sum(times) / max(1, len(times))
    p50 = times[len(times) // 2] if times else float("nan")
    return mean, p50


def _make_params(
    *,
    device: torch.device,
    dtype: torch.dtype,
    shapes: List[Tuple[int, int]],
) -> Tuple[List[Tuple[str, torch.nn.Parameter]], Dict[str, torch.Tensor]]:
    named: List[Tuple[str, torch.nn.Parameter]] = []
    grads: Dict[str, torch.Tensor] = {}
    for i, (m, n) in enumerate(shapes):
        name = f"w{i}"
        p = torch.nn.Parameter(torch.randn((m, n), device=device, dtype=dtype))
        g = torch.randn_like(p)
        p.grad = g
        named.append((name, p))
        grads[name] = g
    return named, grads


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--block_size", type=int, default=32)
    ap.add_argument("--mask_path", type=str, default=None, help="Mask .pt for sparse_adamw case")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--sync_cuda", type=int, default=1)
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # Representative large GEMM-ish shapes (Llama 8B-ish): q/k/v/o + up/down/gate
    shapes = [
        (4096, 4096),
        (4096, 4096),
        (4096, 4096),
        (4096, 4096),
        (14336, 4096),
        (4096, 14336),
        (14336, 4096),
    ]

    named_params, _ = _make_params(device=dev, dtype=dtype, shapes=shapes)

    cases: List[Case] = [
        Case(name="adamw_torch", optimizer="adamw_torch"),
        Case(name="adamw_8bit", optimizer="adamw_8bit"),
        Case(name="sparse_adamw", optimizer="sparse_adamw", mask_path=args.mask_path),
    ]

    rows: List[dict] = []
    for c in cases:
        # fresh params per case so state sizes don't interfere
        named_params, _ = _make_params(device=dev, dtype=dtype, shapes=shapes)
        params = [p for _, p in named_params]

        if c.optimizer == "adamw_torch":
            opt = torch.optim.AdamW(params, lr=args.lr)
        elif c.optimizer == "adamw_8bit":
            try:
                from bitsandbytes.optim import AdamW8bit
            except ImportError:
                rows.append(
                    {
                        "case": c.name,
                        "optimizer": c.optimizer,
                        "mean_step_ms": "",
                        "p50_step_ms": "",
                        "note": "bitsandbytes not importable",
                    }
                )
                continue
            opt = AdamW8bit(params, lr=args.lr)
        elif c.optimizer == "sparse_adamw":
            if not c.mask_path:
                rows.append(
                    {
                        "case": c.name,
                        "optimizer": c.optimizer,
                        "mean_step_ms": "",
                        "p50_step_ms": "",
                        "note": "missing --mask_path",
                    }
                )
                continue
            mm = SparseMaskManager(c.mask_path, device=dev)
            opt = SparseAdamW(named_params, mm, lr=args.lr, block_size=args.block_size, mlp_only=False)
        else:
            raise ValueError(c.optimizer)

        mean_ms, p50_ms = _timed_steps(opt, n=args.steps, warmup=args.warmup, sync_cuda=bool(args.sync_cuda))
        rows.append(
            {
                "case": c.name,
                "optimizer": c.optimizer,
                "mean_step_ms": round(mean_ms, 6),
                "p50_step_ms": round(p50_ms, 6),
                "note": "",
            }
        )

    csv_path = out_dir / "optimizer_step_microbench.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "optimizer_step_microbench.md"
    lines = ["# Optimizer.step() microbenchmark", ""]
    lines.append(f"- **device:** `{dev}`")
    lines.append(f"- **dtype:** `{args.dtype}`")
    lines.append(f"- **warmup:** {args.warmup}  **steps:** {args.steps}  **sync_cuda:** {bool(args.sync_cuda)}")
    lines.append("")
    lines.append("| case | mean_step_ms | p50_step_ms | note |")
    lines.append("|---|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| `{r['case']}` | {r['mean_step_ms']} | {r['p50_step_ms']} | {r['note']} |"
        )
    md_path.write_text("\\n".join(lines), encoding="utf-8")

    print(f"Wrote {csv_path}", file=sys.stderr)  # type: ignore[name-defined]
    print(f"Wrote {md_path}", file=sys.stderr)  # type: ignore[name-defined]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

