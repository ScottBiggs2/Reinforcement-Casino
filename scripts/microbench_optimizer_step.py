#!/usr/bin/env python3
"""
Microbenchmark: optimizer.step() cost only (no forward/backward).

Design goals:
- Highlight SparseAdamW kernel memory/speed savings in isolation.
- Use REAL mask shapes/keys to create parameters so SparseAdamW update patterns match reality.
- Fixed step count (default 50) with trimmed stats excluding first/last 10% for fairness.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import sys

from src.optimizers.sparse_adamw import SparseAdamW
from src.utils.mask_manager import SparseMaskManager


@dataclass
class ResultRow:
    case: str
    optimizer: str
    mask_label: str
    mask_path: str
    steps_total: int
    trim_frac: float
    mean_ms_mid: float
    p50_ms_mid: float
    mean_ms_all: float
    p50_ms_all: float
    note: str


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = int(round((len(xs) - 1) * q))
    i = max(0, min(i, len(xs) - 1))
    return float(xs[i])

def _trimmed(xs: List[float], trim_frac: float) -> List[float]:
    if not xs:
        return []
    n = len(xs)
    k = int(math.floor(n * float(trim_frac)))
    if 2 * k >= n:
        return xs
    return xs[k : n - k]


def _timed_steps(opt, steps: int, sync_cuda: bool) -> List[float]:
    times: List[float] = []
    for _ in range(int(steps)):
        if sync_cuda:
            _sync()
        t0 = time.perf_counter()
        opt.step()
        if sync_cuda:
            _sync()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def _make_params_from_mask(
    *,
    mask_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[Tuple[str, torch.nn.Parameter]], Dict[str, torch.Tensor]]:
    named: List[Tuple[str, torch.nn.Parameter]] = []
    bool_masks: Dict[str, torch.Tensor] = {}
    mm = SparseMaskManager(mask_path, device=device)
    for k, m in mm.masks.items():
        if m.dim() != 2:
            continue
        p = torch.nn.Parameter(torch.randn(tuple(m.shape), device=device, dtype=dtype))
        p.grad = torch.randn_like(p)
        named.append((k, p))
        bool_masks[k] = m.bool() if m.dtype != torch.bool else m
    if not named:
        raise RuntimeError(f"No 2D mask tensors found in {mask_path}")
    return named, bool_masks


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

    # NOTE: the original script used fixed synthetic shapes. We now build params from the mask
    # file to keep SparseAdamW's update pattern realistic.
    rows: List[ResultRow] = []

    def _run_case(case: str, optimizer: str, build_opt_fn):
        try:
            named_params, _ = _make_params_from_mask(
                mask_path=str(args.mask_path),
                device=dev,
                dtype=dtype,
            )
            params = [p for _, p in named_params]
            opt = build_opt_fn(named_params, params)
            times = _timed_steps(opt, steps=args.steps, sync_cuda=bool(args.sync_cuda))
            mid = _trimmed(times, float(args.trim_frac))
            rows.append(
                ResultRow(
                    case=f"{case}_{args.mask_label}",
                    optimizer=optimizer,
                    mask_label=str(args.mask_label),
                    mask_path=str(args.mask_path),
                    steps_total=int(args.steps),
                    trim_frac=float(args.trim_frac),
                    mean_ms_mid=float(sum(mid) / len(mid)) if mid else float("nan"),
                    p50_ms_mid=_percentile(mid, 0.50),
                    mean_ms_all=float(sum(times) / len(times)) if times else float("nan"),
                    p50_ms_all=_percentile(times, 0.50),
                    note="",
                )
            )
        except Exception as e:
            rows.append(
                ResultRow(
                    case=f"{case}_{args.mask_label}",
                    optimizer=optimizer,
                    mask_label=str(args.mask_label),
                    mask_path=str(args.mask_path),
                    steps_total=int(args.steps),
                    trim_frac=float(args.trim_frac),
                    mean_ms_mid=float("nan"),
                    p50_ms_mid=float("nan"),
                    mean_ms_all=float("nan"),
                    p50_ms_all=float("nan"),
                    note=str(e),
                )
            )

    if int(args.run_dense_torch) == 1:
        _run_case("dense", "adamw_torch", lambda named, params: torch.optim.AdamW(params, lr=args.lr))

    if int(args.run_dense_8bit) == 1:
        try:
            from bitsandbytes.optim import AdamW8bit
        except Exception as e:
            rows.append(
                ResultRow(
                    case=f"dense8bit_{args.mask_label}",
                    optimizer="adamw_8bit",
                    mask_label=str(args.mask_label),
                    mask_path=str(args.mask_path),
                    steps_total=int(args.steps),
                    trim_frac=float(args.trim_frac),
                    mean_ms_mid=float("nan"),
                    p50_ms_mid=float("nan"),
                    mean_ms_all=float("nan"),
                    p50_ms_all=float("nan"),
                    note=f"bitsandbytes import failed: {e}",
                )
            )
        else:
            _run_case("dense8bit", "adamw_8bit", lambda named, params: AdamW8bit(params, lr=args.lr))

    if int(args.run_sparse) == 1:
        _run_case(
            "sparse",
            "sparse_adamw",
            lambda named, params: SparseAdamW(
                named,
                SparseMaskManager(str(args.mask_path), device=dev),
                lr=args.lr,
                block_size=args.block_size,
                mlp_only=False,
            ),
        )

    csv_path = out_dir / "optimizer_step_microbench.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ResultRow.__annotations__.keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)

    md_path = out_dir / "optimizer_step_microbench.md"
    lines = ["# SparseAdamW optimizer.step() microbench", ""]
    lines.append(f"- **mask_label:** `{args.mask_label}`")
    lines.append(f"- **mask_path:** `{args.mask_path}`")
    lines.append(f"- **device:** `{dev}`  **dtype:** `{args.dtype}`")
    lines.append(f"- **lr:** `{args.lr}`  **block_size:** `{args.block_size}`")
    lines.append(f"- **steps_total:** `{args.steps}`  **trim_frac:** `{args.trim_frac}` (excludes first/last 10%)")
    lines.append(f"- **sync_cuda:** `{bool(args.sync_cuda)}`")
    lines.append("")
    lines.append("| case | optimizer | mean_ms_mid | p50_ms_mid | mean_ms_all | p50_ms_all | note |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| `{r.case}` | `{r.optimizer}` | {r.mean_ms_mid:.6g} | {r.p50_ms_mid:.6g} | "
            f"{r.mean_ms_all:.6g} | {r.p50_ms_all:.6g} | {r.note} |"
        )
    md_path.write_text("\\n".join(lines), encoding="utf-8")

    print(f"Wrote {csv_path}", file=sys.stderr)
    print(f"Wrote {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

