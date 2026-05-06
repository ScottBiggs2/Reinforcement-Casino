"""
Certifiability margins (Theorem 3): global top-k threshold τ_ρ(s) and per-weight margins m_i(s)=|s_i-τ|.

Uses pure global ranking (min_layer_keep_ratio=0 in theory): τ is the smallest score among the
k largest scores, matching hybrid-free mask_utils semantics. Production masks with per-layer
floors are not represented by a single scalar τ — callers should document cert_mode accordingly.

See mask_utils._create_mask_global_flat tie-break noise (seed 42, scale ∝ max|score|).
"""

from __future__ import annotations

import heapq
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


def flatten_scores_in_order(scores: Dict[str, torch.Tensor], keys: Sequence[str]) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    ref_dt = torch.float32
    for k in keys:
        if k not in scores:
            continue
        t = scores[k].reshape(-1)
        ref_dt = t.dtype
        parts.append(t)
    if not parts:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(parts, dim=0).to(dtype=ref_dt)


def apply_tie_break_noise_flat(
    flat: torch.Tensor,
    *,
    tie_break_noise_scale: float = 1e-6,
    seed: int = 42,
) -> torch.Tensor:
    """Match create_mask_from_scores_gpu_efficient defaults (seed 42, scale from max abs)."""
    x = flat.clone()
    torch.nan_to_num_(x, nan=0.0, posinf=0.0, neginf=0.0)
    xf = x.float()
    scale = max(float(xf.abs().max().item()) * tie_break_noise_scale, 1e-12)
    g = torch.Generator(device=flat.device)
    g.manual_seed(int(seed))
    noise = torch.randn(x.shape, generator=g, device=flat.device, dtype=torch.float32)
    y = xf + noise * scale
    if x.dtype != torch.float32:
        return y.to(dtype=x.dtype)
    return y


def global_keep_count(total_params: int, sparsity_percent: float) -> int:
    """Same keep budget as mask_utils global path: keep_percent = 100 - sparsity_percent."""
    keep_percent = 100.0 - float(sparsity_percent)
    k = max(1, int(keep_percent / 100.0 * total_params))
    return min(k, total_params)


def global_topk_threshold(
    flat_for_selection: torch.Tensor,
    sparsity_percent: float,
) -> Tuple[float, int, int]:
    """
    τ_ρ = minimum score among the k largest scores (boundary of global top-k keep set).

    Returns (tau, k_keep, N).
    """
    N = int(flat_for_selection.numel())
    if N == 0:
        return float("nan"), 0, 0
    k = global_keep_count(N, sparsity_percent)
    vals, _ = torch.topk(flat_for_selection, k, largest=True)
    tau = float(vals.min().item())
    return tau, k, N


def merge_tensor_into_topk_heap(heap: List[float], k: int, values_fp32: torch.Tensor) -> None:
    """
    Maintain the k largest floats seen so far in a min-heap of size at most k.
    heap[0] is the smallest among those k (the k-th largest overall once full).
    """
    if k <= 0:
        return
    v = values_fp32.reshape(-1)
    if v.numel() == 0:
        return
    take = min(k, int(v.numel()))
    topv, _ = torch.topk(v, take, largest=True)
    for val in topv.tolist():
        x = float(val)
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:
            heapq.heapreplace(heap, x)


def scaled_hybrid_floor_counts_per_layer(
    layer_numels: Sequence[int],
    min_layer_keep_ratio: float,
    keep_count: int,
) -> List[int]:
    """
    Per-layer keep floors matching mask_utils hybrid scaling when sum(floors) > keep_count.

    Returns one non-negative floor count per layer, in the same order as ``layer_numels``.
    """
    ratio = float(max(0.0, min(1.0, min_layer_keep_ratio)))
    floors: List[int] = []
    for layer_n in layer_numels:
        lf = int(ratio * int(layer_n))
        lf = max(0, min(lf, int(layer_n)))
        floors.append(lf)
    requested = int(sum(floors))
    if requested > int(keep_count) and requested > 0:
        scale = float(keep_count) / float(requested)
        floors = [max(0, min(int(f * scale), int(n))) for f, n in zip(floors, layer_numels)]
    return floors


def streaming_min_of_global_top_r(
    iter_fp32_1d_pieces: Iterable[torch.Tensor],
    R: int,
    *,
    max_cat: int = 4_000_000,
) -> float:
    """
    Return min(top-R values) over the logical concatenation of all 1D fp32 pieces, without
    materializing the full concatenation (buffer holds at most R floats).
    """
    if R <= 0:
        return float("nan")
    buf = torch.empty(0, dtype=torch.float32)
    for piece in iter_fp32_1d_pieces:
        p = piece.reshape(-1).float()
        if p.numel() == 0:
            continue
        ps = 0
        pe = int(p.numel())
        while ps < pe:
            sub = p[ps : min(ps + max_cat, pe)]
            ps += int(sub.numel())
            cat = torch.cat([buf, sub], dim=0) if buf.numel() else sub
            m = min(R, int(cat.numel()))
            buf, _ = torch.topk(cat, m, largest=True)
    if buf.numel() == 0:
        return float("nan")
    return float(buf.min().item())


def tau_hybrid_global_phase_from_flat(
    flat_sel: torch.Tensor,
    layer_numels: Sequence[int],
    floors: Sequence[int],
    keep_count: int,
) -> Tuple[float, int, int, int, int]:
    """
    Global-phase threshold after per-layer floors (mask_utils flat hybrid semantics).

    Floors: within each layer slice, mark top ``floors[i]`` selection scores as -inf, then
    take the smallest value among the top ``R = keep_count - sum(floors)`` remaining scores.

    Returns (tau, keep_count, N, floor_total, R_remaining).
    """
    x = flat_sel.float().clone()
    torch.nan_to_num_(x, nan=0.0, posinf=0.0, neginf=0.0)
    off = 0
    for n_layer, f in zip(layer_numels, floors):
        n_layer = int(n_layer)
        f = int(f)
        if f > 0:
            sl = x[off : off + n_layer]
            kk = min(f, n_layer)
            _, idx = torch.topk(sl, kk, largest=True)
            sl[idx] = float("-inf")
        off += n_layer
    if off != int(x.numel()):
        raise ValueError("layer_numels do not partition flat_sel")
    floor_total = int(sum(int(f) for f in floors))
    R = int(keep_count) - floor_total
    N = int(x.numel())
    if R <= 0:
        return float("nan"), int(keep_count), N, floor_total, R
    vals, _ = torch.topk(x, min(R, N), largest=True)
    return float(vals.min().item()), int(keep_count), N, floor_total, R


def streaming_global_topk_threshold(
    *,
    total_n: int,
    sparsity_percent: float,
    iter_sel_chunks_fp32: Iterable[torch.Tensor],
) -> Tuple[float, int, int]:
    """Same tau as global_topk_threshold without materializing the full flat tensor."""
    if total_n <= 0:
        return float("nan"), 0, 0
    k = global_keep_count(total_n, sparsity_percent)
    heap: List[float] = []
    for ch in iter_sel_chunks_fp32:
        merge_tensor_into_topk_heap(heap, k, ch)
    if not heap:
        return float("nan"), k, total_n
    tau = float(heap[0])
    return tau, k, total_n


def scores_for_cert_selection(
    flat_raw: torch.Tensor,
    *,
    match_tie_break: bool,
    tie_break_noise_scale: float = 1e-6,
    tie_break_seed: int = 42,
) -> torch.Tensor:
    if flat_raw.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        dt = flat_raw.dtype
        x = flat_raw.clone().to(dtype=dt)
    else:
        dt = torch.float32
        x = flat_raw.float().clone()
    torch.nan_to_num_(x, nan=0.0, posinf=0.0, neginf=0.0)
    if match_tie_break:
        return apply_tie_break_noise_flat(x, tie_break_noise_scale=tie_break_noise_scale, seed=tie_break_seed)
    return x


def certifiability_strict_fraction(gap: torch.Tensor, margin: torch.Tensor) -> Tuple[int, int]:
    """Count / denom where gap_i < margin_i (strict inequality)."""
    ok = (gap < margin).sum().item()
    return int(ok), int(gap.numel())


def smoke_check() -> None:
    """Deterministic sanity: N=10, ρ=50% → k=5, τ known from sorted scores."""
    torch.manual_seed(0)
    flat = torch.arange(10.0, 20.0)  # distinct ascending
    sel = scores_for_cert_selection(flat, match_tie_break=False)
    tau, k, n = global_topk_threshold(sel, sparsity_percent=50.0)
    assert n == 10 and k == 5
    assert abs(tau - 15.0) < 1e-5, tau  # top-5 largest: 19..15 → τ = min = 15
    margin = (sel - tau).abs()
    gap = torch.zeros_like(sel)  # oracle equals s -> gap=0; strict 0 < margin fails where margin==0 (on tau)
    ok, tot = certifiability_strict_fraction(gap, margin)
    assert tot == 10 and ok == int((margin > 0).sum().item())

    # Hand-check cert fraction: N=4, rho=50% -> k=2, tau=2 on scores [0,1,2,3]
    s = torch.tensor([0.0, 1.0, 2.0, 3.0])
    star = torch.zeros(4)
    sel = scores_for_cert_selection(s, match_tie_break=False)
    tau2, k2, n2 = global_topk_threshold(sel, sparsity_percent=50.0)
    assert n2 == 4 and k2 == 2 and abs(tau2 - 2.0) < 1e-5
    mrg = (sel - tau2).abs()
    g2 = (sel - star).abs()
    ok2, tot2 = certifiability_strict_fraction(g2, mrg)
    assert tot2 == 4 and ok2 == 1


if __name__ == "__main__":
    smoke_check()
    print("certifiability_margin smoke_check OK")
