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
from typing import Dict, List, Tuple

import torch
import sys

import os

# Ensure `src.*` imports work when running from Slurm.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimizers.sparse_adamw import SparseAdamW
from src.utils.mask_manager import SparseMaskManager


@dataclass
class ResultRow:
    case: str
    optimizer: str
    mask_label: str
    mask_path: str
    tensors_used: int
    total_numel: int
    active_numel: int
    active_frac: float
    est_param_bytes: int
    est_grad_bytes: int
    est_adam_state_bytes_fp32_dense: int
    est_adam_state_bytes_fp32_sparse: int
    est_sparseadamw_traffic_bytes_proxy: int
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
    max_total_numel: int,
    max_tensors: int,
) -> Tuple[List[Tuple[str, torch.nn.Parameter]], Dict[str, torch.Tensor]]:
    named: List[Tuple[str, torch.nn.Parameter]] = []
    bool_masks: Dict[str, torch.Tensor] = {}
    mm = SparseMaskManager(mask_path, device=device)
    items: List[Tuple[str, torch.Tensor, int]] = []
    for k, m in mm.masks.items():
        if m.dim() != 2:
            continue
        items.append((k, m, int(m.numel())))
    items.sort(key=lambda x: x[2], reverse=True)

    total = 0
    for k, m, n in items:
        if len(named) >= int(max_tensors):
            break
        if total + n > int(max_total_numel) and len(named) > 0:
            break
        total += n
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
    ap.add_argument("--mask_path", type=str, required=True, help="Mask .pt (element OR block)")
    ap.add_argument("--mask_label", type=str, default="elem", help="Label for this run (elem/block/etc.)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--block_size", type=int, default=32)
    # NOTE: kept for backward compatibility; this microbench no longer uses warmup loops.
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--trim_frac", type=float, default=0.10, help="Exclude first/last frac from stats")
    ap.add_argument("--sync_cuda", type=int, default=1)
    ap.add_argument("--run_dense_torch", type=int, default=1)
    ap.add_argument("--run_dense_8bit", type=int, default=1)
    ap.add_argument("--run_sparse", type=int, default=1)
    ap.add_argument("--max_total_numel", type=int, default=25_000_000, help="Cap total elements across tensors.")
    ap.add_argument("--max_tensors", type=int, default=64, help="Cap number of tensors selected from mask.")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # NOTE: We build params from the mask file (capped) to keep the update pattern realistic.
    rows: List[ResultRow] = []

    bytes_per_elem = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4}.get(dtype, 2)

    def _run_case(case: str, optimizer: str, build_opt_fn):
        try:
            named_params, _ = _make_params_from_mask(
                mask_path=str(args.mask_path),
                device=dev,
                dtype=dtype,
                max_total_numel=int(args.max_total_numel),
                max_tensors=int(args.max_tensors),
            )
            params = [p for _, p in named_params]
            opt = build_opt_fn(named_params, params)
            times = _timed_steps(opt, steps=args.steps, sync_cuda=bool(args.sync_cuda))
            mid = _trimmed(times, float(args.trim_frac))

            mm_cpu = SparseMaskManager(str(args.mask_path), device=torch.device("cpu"))
            total_numel = 0
            active_numel = 0
            for name, _p in named_params:
                m = mm_cpu.get_mask(name)
                if m is None:
                    continue
                mb = m.bool() if m.dtype != torch.bool else m
                total_numel += int(mb.numel())
                active_numel += int(mb.sum().item())
            active_frac = (active_numel / total_numel) if total_numel > 0 else float("nan")

            est_param_bytes = int(total_numel * bytes_per_elem)
            est_grad_bytes = int(total_numel * bytes_per_elem)
            est_adam_dense = int(total_numel * 8)  # fp32 m,v
            est_adam_sparse = int(active_numel * 8)
            est_sparse_traffic = int(active_numel * 112)  # proxy
            rows.append(
                ResultRow(
                    case=f"{case}_{args.mask_label}",
                    optimizer=optimizer,
                    mask_label=str(args.mask_label),
                    mask_path=str(args.mask_path),
                    tensors_used=int(len(named_params)),
                    total_numel=int(total_numel),
                    active_numel=int(active_numel),
                    active_frac=float(active_frac),
                    est_param_bytes=int(est_param_bytes),
                    est_grad_bytes=int(est_grad_bytes),
                    est_adam_state_bytes_fp32_dense=int(est_adam_dense),
                    est_adam_state_bytes_fp32_sparse=int(est_adam_sparse),
                    est_sparseadamw_traffic_bytes_proxy=int(est_sparse_traffic),
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
                    tensors_used=0,
                    total_numel=0,
                    active_numel=0,
                    active_frac=float("nan"),
                    est_param_bytes=0,
                    est_grad_bytes=0,
                    est_adam_state_bytes_fp32_dense=0,
                    est_adam_state_bytes_fp32_sparse=0,
                    est_sparseadamw_traffic_bytes_proxy=0,
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
    lines.append(f"- **max_total_numel:** `{int(args.max_total_numel)}`  **max_tensors:** `{int(args.max_tensors)}`")
    lines.append("")

    lines.append("## Timing summary (trimmed mid-window)")
    lines.append("")
    lines.append("| case | optimizer | tensors | total_numel | active_frac | mean_ms_mid | p50_ms_mid | note |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        af = r.active_frac
        af_s = "" if af != af else f"{af:.4g}"
        lines.append(
            f"| `{r.case}` | `{r.optimizer}` | {r.tensors_used} | {r.total_numel} | {af_s} | "
            f"{r.mean_ms_mid:.6g} | {r.p50_ms_mid:.6g} | {r.note} |"
        )

    lines.append("")
    lines.append("## Memory / traffic estimates (subset only)")
    lines.append("")
    lines.append("- `est_param_bytes` / `est_grad_bytes` use the chosen dtype bytes-per-element.")
    lines.append("- AdamW state estimate assumes fp32 `m`+`v` (8 bytes/element).")
    lines.append("- Sparse traffic proxy uses 112 bytes per active element (see `src/utils/bsr_theory_metrics.py`).")
    lines.append("")
    lines.append("| case | est_param_MB | est_grad_MB | est_adam_state_MB_dense | est_adam_state_MB_sparse | traffic_proxy_MB |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| `{r.case}` | {r.est_param_bytes/1e6:.3g} | {r.est_grad_bytes/1e6:.3g} | "
            f"{r.est_adam_state_bytes_fp32_dense/1e6:.3g} | {r.est_adam_state_bytes_fp32_sparse/1e6:.3g} | "
            f"{r.est_sparseadamw_traffic_bytes_proxy/1e6:.3g} |"
        )

    md_path.write_text("\\n".join(lines), encoding="utf-8")

    print(f"Wrote {csv_path}", file=sys.stderr)
    print(f"Wrote {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

