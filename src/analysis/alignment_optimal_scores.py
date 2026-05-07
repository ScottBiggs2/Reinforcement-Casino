"""
Alignment-optimal scoring fixed point (Appendix optimal scoring).

Implements:
  s*(theta_i, B) = (H_ii/2) * theta_i^2 + (C*B + eps_grad) * |theta_i|
  B_{t+1} = sum_{i not in M_t} |theta_i| where M_t keeps top-(1-rho) scores by s*(·,B_t)

This module is written to be usable on huge models without materializing full flat score vectors.
We use the same keep-count convention as `global_keep_count` from `src.analysis.certifiability_margin`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

from src.analysis.certifiability_margin import global_keep_count, streaming_global_topk_threshold


@dataclass(frozen=True)
class AlignmentFixedPointResult:
    B_star: float
    trace_B: List[float]
    converged: bool
    iters: int
    eta: float
    max_iter: int
    tau_star: float
    keep_count: int
    total_n: int


def _eligible_names_in_order(
    theta: Dict[str, torch.Tensor], hdiag: Dict[str, torch.Tensor], keys: Sequence[str]
) -> List[str]:
    out: List[str] = []
    for k in keys:
        if k not in theta or k not in hdiag:
            continue
        t = theta[k]
        h = hdiag[k]
        if t.shape != h.shape:
            continue
        if not t.is_floating_point() or not h.is_floating_point():
            continue
        if t.numel() == 0:
            continue
        out.append(k)
    return out


def s_star_tensor(
    theta: torch.Tensor,
    hdiag: torch.Tensor,
    *,
    B: float,
    C: float,
    eps_grad: float,
) -> torch.Tensor:
    """
    Compute elementwise s*(theta, B) on a single tensor.

    Returns fp32 tensor on CPU by default (caller can cast/move as needed).
    """
    th = theta.float()
    hd = hdiag.float()
    torch.nan_to_num_(th, nan=0.0, posinf=0.0, neginf=0.0)
    torch.nan_to_num_(hd, nan=0.0, posinf=0.0, neginf=0.0)
    return 0.5 * hd * (th * th) + (float(C) * float(B) + float(eps_grad)) * th.abs()


def iter_s_star_chunks_fp32(
    *,
    theta: Dict[str, torch.Tensor],
    hdiag: Dict[str, torch.Tensor],
    keys: Sequence[str],
    B: float,
    C: float,
    eps_grad: float,
    chunk_numel: int = 10_000_000,
) -> Iterator[torch.Tensor]:
    """
    Yield 1D fp32 chunks of s*(·,B) in the same global key order used elsewhere.

    We do not add tie-break noise here; ties in fp32 are extremely unlikely for real models.
    If you need exact mask_utils tie-break semantics, add it in a higher-level driver.
    """
    for name in keys:
        if name not in theta or name not in hdiag:
            continue
        t = theta[name]
        h = hdiag[name]
        if t.shape != h.shape or t.numel() == 0:
            continue
        if not t.is_floating_point() or not h.is_floating_point():
            continue
        s = s_star_tensor(t, h, B=B, C=C, eps_grad=eps_grad).reshape(-1)
        n = int(s.numel())
        for start in range(0, n, int(chunk_numel)):
            yield s[start : min(n, start + int(chunk_numel))]


def fixed_point_B_star(
    *,
    theta: Dict[str, torch.Tensor],
    hdiag: Dict[str, torch.Tensor],
    keys: Sequence[str],
    sparsity_percent: float,
    C: float,
    eps_grad: float,
    eta: float = 1e-6,
    max_iter: int = 25,
    chunk_numel: int = 10_000_000,
) -> AlignmentFixedPointResult:
    """
    Streaming fixed-point iteration for B*.

    Note: this requires multiple passes over all parameters per iteration (tau, then B update).
    Empirically the appendix suggests <10 iterations.
    """
    keys = list(keys)
    elig = _eligible_names_in_order(theta, hdiag, keys)
    total_n = int(sum(int(theta[n].numel()) for n in elig))
    keep = int(global_keep_count(total_n, float(sparsity_percent))) if total_n > 0 else 0

    B = 0.0
    trace: List[float] = [float(B)]
    tau_last = float("nan")
    converged = False

    for it in range(int(max_iter)):
        if total_n <= 0 or keep <= 0:
            tau_last = float("nan")
            converged = True
            break

        # Pass A: threshold tau for top-k selection on s*(·,B)
        tau, k_keep, _N = streaming_global_topk_threshold(
            total_n=total_n,
            sparsity_percent=float(sparsity_percent),
            iter_sel_chunks_fp32=iter_s_star_chunks_fp32(
                theta=theta,
                hdiag=hdiag,
                keys=elig,
                B=B,
                C=C,
                eps_grad=eps_grad,
                chunk_numel=chunk_numel,
            ),
        )
        tau_last = float(tau)

        # Pass B: compute B_next by summing |theta| over pruned coords (score < tau)
        B_next = 0.0
        kept = 0
        for name in elig:
            t = theta[name].float()
            h = hdiag[name].float()
            s = s_star_tensor(t, h, B=B, C=C, eps_grad=eps_grad)
            keep_mask = s >= tau_last
            kept += int(keep_mask.sum().item())
            # pruned mass
            B_next += float(t.abs()[~keep_mask].double().sum().item())

        trace.append(float(B_next))
        if abs(float(B_next) - float(B)) < float(eta):
            B = float(B_next)
            converged = True
            break
        B = float(B_next)

        # If for some reason kept count is wildly off, still continue; ties are the common cause.
        # Users can enable tie-break noise in a higher-level wrapper if needed.

    return AlignmentFixedPointResult(
        B_star=float(B),
        trace_B=trace,
        converged=bool(converged),
        iters=int(len(trace) - 1),
        eta=float(eta),
        max_iter=int(max_iter),
        tau_star=float(tau_last),
        keep_count=int(keep),
        total_n=int(total_n),
    )

