"""
Certifiability margins (Theorem 3): global top-k threshold τ_ρ(s) and per-weight margins m_i(s)=|s_i-τ|.

Uses pure global ranking (min_layer_keep_ratio=0 in theory): τ is the smallest score among the
k largest scores, matching hybrid-free mask_utils semantics. Production masks with per-layer
floors are not represented by a single scalar τ — callers should document cert_mode accordingly.

See mask_utils._create_mask_global_flat tie-break noise (seed 42, scale ∝ max|score|).
"""

from __future__ import annotations

import heapq
import math
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

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


def tie_break_scale(max_abs: float, relative_scale: float) -> float:
    """Absolute tie-break amplitude from a *purely relative* setting; 0 disables perturbation.

    Deliberately has no absolute floor. ``mask_utils`` uses ``max(max|s| * 1e-6, 1e-12)``, and in the
    lr=5e-7 / bf16 regime both terms outrank genuine scores (RESULTS.md §3.1: selection captured only
    ~73% of nonzero-score coordinates). A relative amplitude keeps the perturbation below the
    resolution of the scores it is breaking ties among.
    """
    if relative_scale <= 0.0 or not (max_abs > 0.0):
        return 0.0
    return float(max_abs) * float(relative_scale)


def add_tie_break_noise_(
    x: torch.Tensor,
    *,
    scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """In-place tie-break perturbation of a 1-D selection vector, drawn in ``x``'s dtype.

    At the relative amplitudes we use (~1e-12 of max|s|) the addition rounds away entirely for
    coordinates carrying real signal, so ranking among them is untouched; it only separates
    coordinates that are numerically indistinguishable. Consequence: exact ties above the noise
    floor stay tied and fall back to ``topk``'s deterministic index order rather than to RNG.
    """
    if scale <= 0.0:
        return x
    noise = torch.randn(x.numel(), generator=generator, dtype=x.dtype)
    x.add_(noise, alpha=scale)
    return x


def _finite_min_max(chunks: Iterable[torch.Tensor]) -> Tuple[float, float, int, int]:
    """(min, max, n_finite, n_total) over finite entries; -inf floors are excluded, not clamped."""
    lo = float("inf")
    hi = float("-inf")
    n_finite = 0
    n_total = 0
    for ch in chunks:
        v = ch.reshape(-1)
        n_total += int(v.numel())
        f = v[torch.isfinite(v)]
        if f.numel() == 0:
            continue
        n_finite += int(f.numel())
        lo = min(lo, float(f.min().item()))
        hi = max(hi, float(f.max().item()))
    return lo, hi, n_finite, n_total


def streaming_exact_kth_largest(
    chunk_source: Callable[[], Iterable[torch.Tensor]],
    k: int,
    *,
    num_bins: int = 4096,
    candidate_cap: int = 40_000_000,
    max_rounds: int = 8,
) -> Tuple[float, Dict[str, object]]:
    """k-th largest value over the logical concatenation of all chunks, in bounded memory.

    ``chunk_source`` must return a *fresh* iterable each call — the algorithm makes several passes.
    Non-finite entries (``-inf`` per-layer floors) are ignored.

    Range-narrowing histogram: each round bins the surviving interval into ``num_bins`` linear bins
    and keeps the bin the k-th largest falls in, so the interval shrinks by ~``num_bins`` per round
    and reaches float resolution in 2-3 rounds. This replaces ``streaming_min_of_global_top_r``'s
    top-R buffer (R ~ 1.6e8 here, one ``topk`` per 4 M-element chunk) and the size-k Python heap in
    ``streaming_global_topk_threshold``, neither of which is tractable at ρ=97.5% on a 7 B model.

    Mass points are handled without materializing them: if the interval collapses to a single float
    the answer *is* that float, which is the common case here because bf16-quantized deltas pile onto
    exact multiples of ``ulp(w)``.

    Returns ``(tau, info)``; ``info`` carries the bracket and how it terminated.
    """
    lo, hi, n_finite, n_total = _finite_min_max(chunk_source())
    info: Dict[str, object] = {"n_total": n_total, "n_finite": n_finite, "rounds": 0}
    if k <= 0 or n_finite == 0:
        info["status"] = "empty"
        return float("nan"), info
    if k > n_finite:
        # Fewer finite candidates than the keep budget: the k-th largest does not exist.
        info["status"] = "k_exceeds_finite"
        info["finite_min"] = lo
        info["finite_max"] = hi
        return float("nan"), info
    if lo == hi:
        info["status"] = "degenerate_range"
        info["tau_bracket"] = (lo, hi)
        return float(lo), info

    n_above = 0  # count of values strictly greater than the current bracket's top
    for rnd in range(max_rounds):
        edges = torch.linspace(lo, hi, num_bins + 1, dtype=torch.float64)
        counts = torch.zeros(num_bins, dtype=torch.int64)
        above = 0
        for ch in chunk_source():
            v = ch.reshape(-1).double()
            v = v[torch.isfinite(v)]
            if v.numel() == 0:
                continue
            above += int((v > hi).sum().item())
            sel = v[(v >= lo) & (v <= hi)]
            if sel.numel() == 0:
                continue
            # right=False → bin i holds (edges[i], edges[i+1]], so counts[b+1:] is exactly
            # count(v > edges[b+1]) and the next bracket cannot double-count its own top edge.
            # v == lo lands in bin 0 after the clamp.
            idx = torch.bucketize(sel, edges, right=False) - 1
            idx.clamp_(0, num_bins - 1)
            counts += torch.bincount(idx, minlength=num_bins).to(torch.int64)

        n_above = above
        need = k - n_above  # rank within the bracket
        if need <= 0:
            # Everything at or above `hi` already fills the budget; boundary sits at hi.
            info["rounds"] = rnd + 1
            info["status"] = "boundary_at_bracket_top"
            info["tau_bracket"] = (lo, hi)
            return float(hi), info

        cum_from_top = torch.cumsum(counts.flip(0), dim=0)
        j = int(torch.searchsorted(cum_from_top, torch.tensor(need, dtype=torch.int64)).item())
        j = min(max(j, 0), num_bins - 1)
        b = num_bins - 1 - j  # bin index (ascending) holding the k-th largest
        n_above = n_above + int(counts[b + 1 :].sum().item())
        new_lo = float(edges[b].item())
        new_hi = float(edges[b + 1].item())
        bin_count = int(counts[b].item())
        lo, hi = new_lo, new_hi
        info["rounds"] = rnd + 1

        if new_lo == new_hi:
            info["status"] = "mass_point"
            info["tau_bracket"] = (lo, hi)
            return float(new_lo), info
        if bin_count <= candidate_cap:
            break

    # Materialize the surviving bracket and finish exactly.
    need = k - n_above
    if need <= 0:
        info["status"] = "boundary_at_bracket_top"
        info["tau_bracket"] = (lo, hi)
        return float(hi), info
    parts: List[torch.Tensor] = []
    held = 0
    for ch in chunk_source():
        v = ch.reshape(-1).double()
        v = v[torch.isfinite(v)]
        sel = v[(v >= lo) & (v <= hi)]
        if sel.numel():
            parts.append(sel)
            held += int(sel.numel())
        if held > candidate_cap * 4:
            # Refusing to grow without bound; the bracket is already at ~1e-12 relative width.
            info["status"] = "bracket_width_fallback"
            info["tau_bracket"] = (lo, hi)
            return float(lo), info
    if not parts:
        info["status"] = "empty_bracket"
        info["tau_bracket"] = (lo, hi)
        return float(lo), info
    cand = torch.cat(parts, dim=0)
    take = min(need, int(cand.numel()))
    vals, _ = torch.topk(cand, take, largest=True)
    info["status"] = "exact"
    info["tau_bracket"] = (lo, hi)
    info["candidates"] = int(cand.numel())
    return float(vals.min().item()), info


# A tie-break draw is N(0, scale); over ~7e9 coordinates the largest is ~6.3 sigma, so a boundary
# within ~6.5 sigma of zero is indistinguishable from the perturbation that produced it.
NOISE_GUARD_SIGMA = 6.5


def tau_is_noise_determined(tau: float, tie_break_scale: float, *, guard: float = NOISE_GUARD_SIGMA) -> bool:
    """True when tau_rho sits inside the tie-break noise, so it is not a property of the scores.

    This is the test that matters, and it is *not* implied by comparing the raw support to the keep
    budget: a score vector can have far more positive coordinates than the budget and still have
    fewer than R of them *above the noise floor*, because ``ulp ∝ |w|`` gives the warm-magnitude
    score a tail running below any fixed relative amplitude (RESULTS.md §3.0). Measured on
    Olmo-3-7B at rho=99%: 71.7 M positives survived the per-layer floors, but only 53.6 M exceeded
    1e-14 against R = 54.7 M, so tau landed at 7.3e-15 with support well above the budget.
    """
    if not (tie_break_scale > 0.0):
        return False
    if not math.isfinite(tau):
        return True
    return tau <= guard * float(tie_break_scale)


def tau_relative(x: torch.Tensor, tau: float) -> torch.Tensor:
    """Scale-free score s/τ̂_ρ. Rank-preserving for τ̂>0, so the top-k set is unchanged."""
    if not (tau > 0.0):
        raise ValueError(f"tau_relative requires tau > 0, got {tau!r}")
    return x / float(tau)


def normalized_margin(x: torch.Tensor, tau: float) -> torch.Tensor:
    """m̃_i = |s_i − τ̂_ρ| / τ̂_ρ — distance to the selection boundary in units of the boundary.

    Coordinates with no resolvable signal (s_i = 0) land at exactly 1.0, so the mass at 1.0 is the
    fraction of the score vector the mask cannot distinguish (see RESULTS.md §3.1).
    """
    return (tau_relative(x, tau) - 1.0).abs()


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
