"""
Certifiability margins (Theorem 3): global top-k threshold τ_ρ(s) and per-weight margins m_i(s)=|s_i-τ|.

Uses pure global ranking (min_layer_keep_ratio=0 in theory): τ is the smallest score among the
k largest scores, matching hybrid-free mask_utils semantics. Production masks with per-layer
floors are not represented by a single scalar τ — callers should document cert_mode accordingly.

See mask_utils._create_mask_global_flat tie-break noise (seed 42, scale ∝ max|score|).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch


def flatten_scores_in_order(scores: Dict[str, torch.Tensor], keys: Sequence[str]) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for k in keys:
        if k not in scores:
            continue
        parts.append(scores[k].reshape(-1).float())
    if not parts:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(parts, dim=0)


def apply_tie_break_noise_flat(
    flat: torch.Tensor,
    *,
    tie_break_noise_scale: float = 1e-6,
    seed: int = 42,
) -> torch.Tensor:
    """Match create_mask_from_scores_gpu_efficient defaults (seed 42, scale from max abs)."""
    x = flat.clone()
    torch.nan_to_num_(x, nan=0.0, posinf=0.0, neginf=0.0)
    scale = max(float(x.abs().max().item()) * tie_break_noise_scale, 1e-12)
    g = torch.Generator(device=flat.device)
    g.manual_seed(int(seed))
    noise = torch.randn(x.shape, generator=g, device=flat.device, dtype=x.dtype)
    return x + noise * scale


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


def scores_for_cert_selection(
    flat_raw: torch.Tensor,
    *,
    match_tie_break: bool,
    tie_break_noise_scale: float = 1e-6,
    tie_break_seed: int = 42,
) -> torch.Tensor:
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
