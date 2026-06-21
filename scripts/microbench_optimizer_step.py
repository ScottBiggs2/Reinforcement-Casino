#!/usr/bin/env python3
"""
Microbenchmark: optimizer.step() cost only (no forward/backward).

Two isolated phases run back-to-back so they cannot contaminate each other:

  Phase 1 – Speed: pure wall-clock timing, zero memory instrumentation.
  Phase 2 – Memory: measured GPU footprint and peak scratch, zero timing pressure.

Design goals:
- Highlight SparseAdamW kernel memory/speed savings in isolation.
- Use REAL mask shapes/keys to create parameters so SparseAdamW update patterns match reality.
- Fixed step count (default 50) with trimmed stats excluding first/last 10% for fairness.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import sys
import os

# Ensure `src.*` imports work when running from Slurm.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimizers.sparse_adamw import SparseAdamW
from src.utils.mask_manager import SparseMaskManager


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class ResultRow:
    """Speed phase output — identical schema to prior versions for backward compat."""
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


@dataclass
class MemRow:
    """Memory phase output — measured GPU allocations, not estimates."""
    case: str
    optimizer: str
    mask_label: str
    mask_path: str
    tensors_used: int
    total_numel: int
    active_numel: int
    active_frac: float
    # Measured GPU memory (MB) ------------------------------------------------
    params_grad_mb: float       # params + grads allocated on GPU (measured)
    opt_state_mb: float         # optimizer state after lazy init via first step (measured)
    total_footprint_mb: float   # params_grad_mb + opt_state_mb
    peak_scratch_mb: float      # peak temp alloc above baseline during one step (measured)
    # Bandwidth estimate -------------------------------------------------------
    bw_ref_steps: int           # steps run solely for bandwidth reference timing
    bw_ref_mean_ms: float       # mean step time from those reference steps
    est_traffic_bytes: int      # theoretical bytes transferred per step (proxy)
    bw_est_gb_s: float          # est_traffic_bytes / bw_ref_mean_ms in GB/s
    note: str


# ── Low-level helpers ─────────────────────────────────────────────────────────

_BW_REF_STEPS = 5  # steps used only for bandwidth reference timing in the memory phase


def _sync(device: Optional[torch.device] = None) -> None:
    if torch.cuda.is_available():
        if device is not None:
            torch.cuda.synchronize(device)
        else:
            torch.cuda.synchronize()


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = max(0, min(int(round((len(xs) - 1) * q)), len(xs) - 1))
    return float(xs[i])


def _trimmed(xs: List[float], trim_frac: float) -> List[float]:
    if not xs:
        return []
    n = len(xs)
    k = int(math.floor(n * float(trim_frac)))
    if 2 * k >= n:
        return xs
    return xs[k : n - k]


def _timed_steps(
    opt, steps: int, sync_cuda: bool, device: Optional[torch.device] = None
) -> List[float]:
    times: List[float] = []
    for _ in range(int(steps)):
        if sync_cuda:
            _sync(device)
        t0 = time.perf_counter()
        opt.step()
        if sync_cuda:
            _sync(device)
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def _make_params_from_mask(
    *,
    mask_path: str,
    device: torch.device,
    dtype: torch.dtype,
    max_total_numel: int,
    max_tensors: int,
    selection_order: str = "model_order",
    cap_behavior: str = "break",
) -> Tuple[List[Tuple[str, torch.nn.Parameter]], Dict[str, torch.Tensor]]:
    """
    Subset selection policy for the synthetic optimizer.step() microbench.

    selection_order:
      - "model_order"   : keep mm.masks insertion order (named_parameters order at save time).
      - "largest_first" : sort by tensor numel descending.

    cap_behavior:
      - "break" : Always include the first eligible tensor, then break the moment another tensor
                  would push total numel past max_total_numel.
                  Matches historical commit b9d6ba2 behavior used to produce the 97.5% row.
      - "skip"  : skip any single tensor whose numel > max_total_numel; otherwise pack greedily
                  until adding another would exceed the cap.
    """
    named: List[Tuple[str, torch.nn.Parameter]] = []
    bool_masks: Dict[str, torch.Tensor] = {}
    mm = SparseMaskManager(mask_path, device=device)
    items: List[Tuple[str, torch.Tensor, int]] = []
    for k, m in mm.masks.items():
        if m.dim() != 2:
            continue
        items.append((k, m, int(m.numel())))
    if selection_order == "largest_first":
        items.sort(key=lambda x: x[2], reverse=True)
    elif selection_order != "model_order":
        raise ValueError(f"Unknown selection_order: {selection_order!r}")

    total = 0
    cap = int(max_total_numel)
    for k, m, n in items:
        if len(named) >= int(max_tensors):
            break
        if cap_behavior == "skip":
            if n > cap:
                continue
            if total + n > cap:
                break
        elif cap_behavior == "break":
            if len(named) > 0 and (total + n > cap):
                break
        else:
            raise ValueError(f"Unknown cap_behavior: {cap_behavior!r}")
        total += n
        p = torch.nn.Parameter(torch.randn(tuple(m.shape), device=device, dtype=dtype))
        p.grad = torch.randn_like(p)
        named.append((k, p))
        bool_masks[k] = m.bool() if m.dtype != torch.bool else m
    if not named:
        raise RuntimeError(f"No 2D mask tensors found in {mask_path}")
    return named, bool_masks


def _mask_stats(
    mask_path: str, named_params: List[Tuple[str, torch.nn.Parameter]]
) -> Tuple[int, int]:
    """Return (total_numel, active_numel) for the given param subset."""
    mm_cpu = SparseMaskManager(mask_path, device=torch.device("cpu"))
    total_numel = 0
    active_numel = 0
    for name, _ in named_params:
        m = mm_cpu.get_mask(name)
        if m is None:
            continue
        mb = m.bool() if m.dtype != torch.bool else m
        total_numel += int(mb.numel())
        active_numel += int(mb.sum().item())
    return total_numel, active_numel


# ── Phase 1: Speed ────────────────────────────────────────────────────────────

def run_speed_phase(
    args,
    dev: torch.device,
    dtype: torch.dtype,
    bytes_per_elem: int,
) -> List[ResultRow]:
    """
    Pure timing phase — no memory instrumentation whatsoever.
    Each case creates its own fresh params and optimizer to avoid state bleed.
    """
    rows: List[ResultRow] = []

    def _null_row(case: str, optimizer: str, exc: Exception) -> ResultRow:
        return ResultRow(
            case=f"{case}_{args.mask_label}",
            optimizer=optimizer,
            mask_label=str(args.mask_label),
            mask_path=str(args.mask_path),
            tensors_used=0, total_numel=0, active_numel=0, active_frac=float("nan"),
            est_param_bytes=0, est_grad_bytes=0,
            est_adam_state_bytes_fp32_dense=0, est_adam_state_bytes_fp32_sparse=0,
            est_sparseadamw_traffic_bytes_proxy=0,
            steps_total=int(args.steps), trim_frac=float(args.trim_frac),
            mean_ms_mid=float("nan"), p50_ms_mid=float("nan"),
            mean_ms_all=float("nan"), p50_ms_all=float("nan"),
            note=str(exc),
        )

    def _run_case(case: str, optimizer: str, build_opt_fn) -> None:
        try:
            named_params, _ = _make_params_from_mask(
                mask_path=str(args.mask_path),
                device=dev,
                dtype=dtype,
                max_total_numel=int(args.max_total_numel),
                max_tensors=int(args.max_tensors),
                selection_order=str(args.selection_order),
                cap_behavior=str(args.cap_behavior),
            )
            opt = build_opt_fn(named_params)
            times = _timed_steps(opt, steps=args.steps, sync_cuda=bool(args.sync_cuda), device=dev)
            mid = _trimmed(times, float(args.trim_frac))

            total_numel, active_numel = _mask_stats(str(args.mask_path), named_params)
            active_frac = (active_numel / total_numel) if total_numel > 0 else float("nan")

            rows.append(ResultRow(
                case=f"{case}_{args.mask_label}",
                optimizer=optimizer,
                mask_label=str(args.mask_label),
                mask_path=str(args.mask_path),
                tensors_used=len(named_params),
                total_numel=int(total_numel),
                active_numel=int(active_numel),
                active_frac=float(active_frac),
                est_param_bytes=int(total_numel * bytes_per_elem),
                est_grad_bytes=int(total_numel * bytes_per_elem),
                est_adam_state_bytes_fp32_dense=int(total_numel * 8),
                est_adam_state_bytes_fp32_sparse=int(active_numel * 8),
                est_sparseadamw_traffic_bytes_proxy=int(active_numel * 112),
                steps_total=int(args.steps),
                trim_frac=float(args.trim_frac),
                mean_ms_mid=float(sum(mid) / len(mid)) if mid else float("nan"),
                p50_ms_mid=_percentile(mid, 0.50),
                mean_ms_all=float(sum(times) / len(times)) if times else float("nan"),
                p50_ms_all=_percentile(times, 0.50),
                note="",
            ))
        except Exception as e:
            rows.append(_null_row(case, optimizer, e))

    if int(args.run_dense_torch) == 1:
        _run_case("dense", "adamw_torch",
                  lambda named: torch.optim.AdamW([p for _, p in named], lr=args.lr))

    if int(args.run_dense_8bit) == 1:
        try:
            from bitsandbytes.optim import AdamW8bit
        except Exception as e:
            rows.append(ResultRow(
                case=f"dense8bit_{args.mask_label}", optimizer="adamw_8bit",
                mask_label=str(args.mask_label), mask_path=str(args.mask_path),
                tensors_used=0, total_numel=0, active_numel=0, active_frac=float("nan"),
                est_param_bytes=0, est_grad_bytes=0,
                est_adam_state_bytes_fp32_dense=0, est_adam_state_bytes_fp32_sparse=0,
                est_sparseadamw_traffic_bytes_proxy=0,
                steps_total=int(args.steps), trim_frac=float(args.trim_frac),
                mean_ms_mid=float("nan"), p50_ms_mid=float("nan"),
                mean_ms_all=float("nan"), p50_ms_all=float("nan"),
                note=f"bitsandbytes import failed: {e}",
            ))
        else:
            _run_case("dense8bit", "adamw_8bit",
                      lambda named: AdamW8bit([p for _, p in named], lr=args.lr))

    if int(args.run_sparse) == 1:
        _run_case(
            "sparse", "sparse_adamw",
            lambda named: SparseAdamW(
                named,
                SparseMaskManager(str(args.mask_path), device=dev),
                lr=args.lr,
                block_size=args.block_size,
                mlp_only=False,
            ),
        )

    return rows


# ── Phase 2: Memory ───────────────────────────────────────────────────────────

def run_memory_phase(
    args,
    dev: torch.device,
    dtype: torch.dtype,
) -> List[MemRow]:
    """
    Measured memory phase — completely isolated from speed phase.

    For each optimizer:
      1. Clean GPU slate (gc + empty_cache).
      2. Allocate params+grads → measure delta → params_grad_mb.
      3. Build optimizer → run first step (lazy state init) → measure delta → opt_state_mb.
      4. Warm up 3 steps, then run one step under peak-memory tracking → peak_scratch_mb.
      5. Run _BW_REF_STEPS timed steps → bw_ref_mean_ms for bandwidth estimate.
      6. Delete all tensors and optimizer, flush cache before next case.
    """
    if not torch.cuda.is_available():
        return []

    rows: List[MemRow] = []

    def _null_mem_row(case: str, optimizer: str, exc: Exception) -> MemRow:
        return MemRow(
            case=f"{case}_{args.mask_label}",
            optimizer=optimizer,
            mask_label=str(args.mask_label),
            mask_path=str(args.mask_path),
            tensors_used=0, total_numel=0, active_numel=0, active_frac=float("nan"),
            params_grad_mb=float("nan"), opt_state_mb=float("nan"),
            total_footprint_mb=float("nan"), peak_scratch_mb=float("nan"),
            bw_ref_steps=_BW_REF_STEPS, bw_ref_mean_ms=float("nan"),
            est_traffic_bytes=0, bw_est_gb_s=float("nan"),
            note=str(exc),
        )

    def _run_mem_case(case: str, optimizer_name: str, build_opt_fn) -> None:
        try:
            # ── Step 0: clean slate ──────────────────────────────────────────
            gc.collect()
            torch.cuda.empty_cache()
            _sync(dev)
            mem_0 = torch.cuda.memory_allocated(dev)

            # ── Step 1: allocate params + grads ─────────────────────────────
            named_params, _ = _make_params_from_mask(
                mask_path=str(args.mask_path),
                device=dev,
                dtype=dtype,
                max_total_numel=int(args.max_total_numel),
                max_tensors=int(args.max_tensors),
                selection_order=str(args.selection_order),
                cap_behavior=str(args.cap_behavior),
            )
            _sync(dev)
            mem_1 = torch.cuda.memory_allocated(dev)
            params_grad_mb = (mem_1 - mem_0) / 1e6

            # ── Step 2: build optimizer + lazy state init ────────────────────
            # AdamW (torch and 8-bit) initializes momentum buffers on the first
            # step, not on construction. SparseAdamW may do the same. We capture
            # everything from post-params to post-first-step as "optimizer state".
            opt = build_opt_fn(named_params)
            opt.step()  # triggers lazy state allocation
            _sync(dev)
            mem_3 = torch.cuda.memory_allocated(dev)
            opt_state_mb = (mem_3 - mem_1) / 1e6

            # ── Step 3: warm up, then measure peak scratch ───────────────────
            for _ in range(3):
                opt.step()
            _sync(dev)

            mem_baseline = torch.cuda.memory_allocated(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            opt.step()
            _sync(dev)
            peak_scratch_mb = max(0.0, torch.cuda.max_memory_allocated(dev) - mem_baseline) / 1e6

            # ── Step 4: short timing run for bandwidth estimate ──────────────
            bw_times = _timed_steps(opt, steps=_BW_REF_STEPS, sync_cuda=True, device=dev)
            bw_ref_mean_ms = sum(bw_times) / len(bw_times) if bw_times else float("nan")

            # ── Step 5: compute bandwidth estimate ───────────────────────────
            total_numel, active_numel = _mask_stats(str(args.mask_path), named_params)
            active_frac = (active_numel / total_numel) if total_numel > 0 else float("nan")
            # Sparse operates only on active elements; dense touches all of them.
            working_numel = active_numel if optimizer_name == "sparse_adamw" else total_numel
            est_traffic = int(working_numel * 112)  # same proxy as speed-phase estimates
            bw_est_gb_s = (
                (est_traffic / (bw_ref_mean_ms * 1e-3)) / 1e9
                if bw_ref_mean_ms > 0 and bw_ref_mean_ms == bw_ref_mean_ms
                else float("nan")
            )

            rows.append(MemRow(
                case=f"{case}_{args.mask_label}",
                optimizer=optimizer_name,
                mask_label=str(args.mask_label),
                mask_path=str(args.mask_path),
                tensors_used=len(named_params),
                total_numel=int(total_numel),
                active_numel=int(active_numel),
                active_frac=float(active_frac),
                params_grad_mb=float(params_grad_mb),
                opt_state_mb=float(opt_state_mb),
                total_footprint_mb=float(params_grad_mb + opt_state_mb),
                peak_scratch_mb=float(peak_scratch_mb),
                bw_ref_steps=_BW_REF_STEPS,
                bw_ref_mean_ms=float(bw_ref_mean_ms),
                est_traffic_bytes=int(est_traffic),
                bw_est_gb_s=float(bw_est_gb_s),
                note="",
            ))

        except Exception as e:
            rows.append(_null_mem_row(case, optimizer_name, e))

        finally:
            # Explicit cleanup so each case starts with a clean GPU state.
            try:
                del opt  # type: ignore[possibly-undefined]
            except NameError:
                pass
            try:
                del params, named_params  # type: ignore[possibly-undefined]
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()

    if int(args.run_dense_torch) == 1:
        _run_mem_case("dense", "adamw_torch",
                      lambda named: torch.optim.AdamW([p for _, p in named], lr=args.lr))

    if int(args.run_dense_8bit) == 1:
        try:
            from bitsandbytes.optim import AdamW8bit
        except Exception as e:
            rows.append(MemRow(
                case=f"dense8bit_{args.mask_label}", optimizer="adamw_8bit",
                mask_label=str(args.mask_label), mask_path=str(args.mask_path),
                tensors_used=0, total_numel=0, active_numel=0, active_frac=float("nan"),
                params_grad_mb=float("nan"), opt_state_mb=float("nan"),
                total_footprint_mb=float("nan"), peak_scratch_mb=float("nan"),
                bw_ref_steps=_BW_REF_STEPS, bw_ref_mean_ms=float("nan"),
                est_traffic_bytes=0, bw_est_gb_s=float("nan"),
                note=f"bitsandbytes import failed: {e}",
            ))
        else:
            _run_mem_case("dense8bit", "adamw_8bit",
                          lambda named: AdamW8bit([p for _, p in named], lr=args.lr))

    if int(args.run_sparse) == 1:
        _run_mem_case(
            "sparse", "sparse_adamw",
            lambda named: SparseAdamW(
                named,
                SparseMaskManager(str(args.mask_path), device=dev),
                lr=args.lr,
                block_size=args.block_size,
                mlp_only=False,
            ),
        )

    return rows


# ── Output writers ────────────────────────────────────────────────────────────

def _write_speed_csv(rows: List[ResultRow], out_dir: Path) -> Path:
    csv_path = out_dir / "optimizer_step_microbench.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ResultRow.__annotations__.keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)
    return csv_path


def _write_memory_csv(rows: List[MemRow], out_dir: Path) -> Path:
    csv_path = out_dir / "optimizer_step_memory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(MemRow.__annotations__.keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)
    return csv_path


def _fmt_mb(v: float) -> str:
    return "" if v != v else f"{v:.2f}"


def _fmt_gb(v: float) -> str:
    return "" if v != v else f"{v:.1f}"


def _row_by_opt(rows: list, key: str):
    for rr in rows:
        if rr.optimizer == key:
            return rr
    return None


def _write_markdown(
    args,
    dev: torch.device,
    speed_rows: List[ResultRow],
    mem_rows: List[MemRow],
    out_dir: Path,
) -> Path:
    md_path = out_dir / "optimizer_step_microbench.md"
    lines: List[str] = ["# SparseAdamW optimizer.step() microbench", ""]
    lines.append(f"- **mask_label:** `{args.mask_label}`")
    lines.append(f"- **mask_path:** `{args.mask_path}`")
    lines.append(f"- **device:** `{dev}`  **dtype:** `{args.dtype}`")
    lines.append(f"- **lr:** `{args.lr}`  **block_size:** `{args.block_size}`")
    lines.append(
        f"- **max_total_numel:** `{int(args.max_total_numel)}`  "
        f"**max_tensors:** `{int(args.max_tensors)}`  "
        f"**selection_order:** `{args.selection_order}`  "
        f"**cap_behavior:** `{args.cap_behavior}`"
    )
    lines.append("")

    # ── Phase 1: Speed ────────────────────────────────────────────────────────
    lines.append("## Phase 1 — Speed (no memory instrumentation)")
    lines.append("")
    lines.append(
        f"- **steps_total:** `{args.steps}`  "
        f"**trim_frac:** `{args.trim_frac}` (excludes first/last {int(args.trim_frac * 100)}%)"
    )
    lines.append(f"- **sync_cuda:** `{bool(args.sync_cuda)}`")
    lines.append("")
    lines.append("| case | optimizer | tensors | total_numel | active_frac | mean_ms_mid | p50_ms_mid | note |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in speed_rows:
        af_s = "" if r.active_frac != r.active_frac else f"{r.active_frac:.4g}"
        lines.append(
            f"| `{r.case}` | `{r.optimizer}` | {r.tensors_used} | {r.total_numel} | {af_s} | "
            f"{r.mean_ms_mid:.6g} | {r.p50_ms_mid:.6g} | {r.note} |"
        )

    sparse_s = _row_by_opt(speed_rows, "sparse_adamw")
    if sparse_s is not None:
        lines.append("")
        lines.append("### Key speedups (trimmed mean)")
        lines.append("")
        dense_t_s = _row_by_opt(speed_rows, "adamw_torch")
        dense_8_s = _row_by_opt(speed_rows, "adamw_8bit")
        if (dense_t_s is not None and dense_t_s.mean_ms_mid == dense_t_s.mean_ms_mid
                and sparse_s.mean_ms_mid == sparse_s.mean_ms_mid):
            lines.append(
                f"- **SparseAdamW vs torch AdamW:** "
                f"x{dense_t_s.mean_ms_mid / max(1e-9, sparse_s.mean_ms_mid):.3f} faster"
            )
        if (dense_8_s is not None and dense_8_s.mean_ms_mid == dense_8_s.mean_ms_mid
                and sparse_s.mean_ms_mid == sparse_s.mean_ms_mid):
            lines.append(
                f"- **SparseAdamW vs AdamW 8-bit:** "
                f"x{dense_8_s.mean_ms_mid / max(1e-9, sparse_s.mean_ms_mid):.3f} faster"
            )

    lines.append("")

    # ── Phase 2: Memory ───────────────────────────────────────────────────────
    lines.append("## Phase 2 — Memory (measured GPU footprint, isolated from speed phase)")
    lines.append("")
    lines.append(f"- **bw_ref_steps:** `{_BW_REF_STEPS}` (short timing used only for bandwidth estimate, not the Phase 1 numbers)")
    lines.append("- `params_grad_mb`: GPU bytes for params + grads, measured before optimizer is built.")
    lines.append("- `opt_state_mb`: GPU bytes added by optimizer (build + lazy first-step state init), measured.")
    lines.append("- `peak_scratch_mb`: peak temp allocations above steady-state baseline during one step.")
    lines.append("- `bw_est_gb_s`: theoretical traffic proxy / bw_ref_mean_ms (dense uses total_numel × 112 B; sparse uses active_numel × 112 B).")
    lines.append("")
    lines.append(
        "| case | optimizer | active_frac | params_grad_MB | opt_state_MB | "
        "total_footprint_MB | peak_scratch_MB | bw_est_GB_s | note |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in mem_rows:
        af_s = "" if r.active_frac != r.active_frac else f"{r.active_frac:.4g}"
        lines.append(
            f"| `{r.case}` | `{r.optimizer}` | {af_s} | "
            f"{_fmt_mb(r.params_grad_mb)} | {_fmt_mb(r.opt_state_mb)} | "
            f"{_fmt_mb(r.total_footprint_mb)} | {_fmt_mb(r.peak_scratch_mb)} | "
            f"{_fmt_gb(r.bw_est_gb_s)} | {r.note} |"
        )

    sparse_m = _row_by_opt(mem_rows, "sparse_adamw")
    if sparse_m is not None:
        lines.append("")
        lines.append("### Memory savings (measured optimizer state)")
        lines.append("")
        dense_t_m = _row_by_opt(mem_rows, "adamw_torch")
        dense_8_m = _row_by_opt(mem_rows, "adamw_8bit")
        for label, dense_m in [("torch AdamW", dense_t_m), ("AdamW 8-bit", dense_8_m)]:
            if dense_m is None:
                continue
            d, s = dense_m.opt_state_mb, sparse_m.opt_state_mb
            if d == d and s == s and d > 0:
                lines.append(
                    f"- **SparseAdamW vs {label} state:** "
                    f"x{d / max(1e-6, s):.3f} smaller ({(1 - s / d) * 100:.1f}% reduction)"
                )

    lines.append("")

    # ── Estimated traffic / state (legacy section, kept for render_optimizer_step_microbench_md.py) ──
    lines.append("## Memory / traffic estimates (subset only)")
    lines.append("")
    lines.append("- `est_param_bytes` / `est_grad_bytes` use the chosen dtype bytes-per-element.")
    lines.append("- AdamW state estimate assumes fp32 `m`+`v` (8 bytes/element).")
    lines.append("- Sparse traffic proxy uses 112 bytes per active element (see `src/utils/bsr_theory_metrics.py`).")
    lines.append("")
    lines.append("| case | est_param_MB | est_grad_MB | est_adam_state_MB_dense | est_adam_state_MB_sparse | traffic_proxy_MB |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in speed_rows:
        lines.append(
            f"| `{r.case}` | {r.est_param_bytes / 1e6:.1f} | {r.est_grad_bytes / 1e6:.1f} | "
            f"{r.est_adam_state_bytes_fp32_dense / 1e6:.1f} | "
            f"{r.est_adam_state_bytes_fp32_sparse / 1e6:.1f} | "
            f"{r.est_sparseadamw_traffic_bytes_proxy / 1e6:.1f} |"
        )

    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--mask_path", type=str, required=True, help="Mask .pt (element OR block)")
    ap.add_argument("--mask_label", type=str, default="elem", help="Label for this run (elem/block/etc.)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--block_size", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=0)  # kept for backward compat; unused
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--trim_frac", type=float, default=0.10, help="Exclude first/last frac from stats")
    ap.add_argument("--sync_cuda", type=int, default=1)
    ap.add_argument("--run_dense_torch", type=int, default=1)
    ap.add_argument("--run_dense_8bit", type=int, default=1)
    ap.add_argument("--run_sparse", type=int, default=1)
    ap.add_argument("--max_total_numel", type=int, default=25_000_000, help="Cap total elements across tensors.")
    ap.add_argument("--max_tensors", type=int, default=64, help="Cap number of tensors selected from mask.")
    ap.add_argument(
        "--selection_order",
        choices=["model_order", "largest_first"],
        default="model_order",
    )
    ap.add_argument(
        "--cap_behavior",
        choices=["break", "skip"],
        default="break",
        help=(
            "How to honor --max_total_numel. 'break' always includes the first tensor and stops on "
            "the next overage (matches historical 97.5%% row). 'skip' skips per-tensor overflows."
        ),
    )
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    bytes_per_elem = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4}.get(dtype, 2)

    # ── Phase 1: Speed ────────────────────────────────────────────────────────
    print("=== Phase 1: Speed (no memory instrumentation) ===", file=sys.stderr)
    speed_rows = run_speed_phase(args, dev, dtype, bytes_per_elem)

    # ── Phase 2: Memory ───────────────────────────────────────────────────────
    print("=== Phase 2: Memory (isolated from speed phase) ===", file=sys.stderr)
    mem_rows = run_memory_phase(args, dev, dtype)

    # ── Write outputs ─────────────────────────────────────────────────────────
    speed_csv = _write_speed_csv(speed_rows, out_dir)
    mem_csv = _write_memory_csv(mem_rows, out_dir)
    md_path = _write_markdown(args, dev, speed_rows, mem_rows, out_dir)

    print(f"Wrote {speed_csv}", file=sys.stderr)
    print(f"Wrote {mem_csv}", file=sys.stderr)
    print(f"Wrote {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
