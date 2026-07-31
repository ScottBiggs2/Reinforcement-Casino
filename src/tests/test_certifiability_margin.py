"""Smoke tests for global tau, strict certifiability fraction, and the streaming k-th largest."""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.analysis.certifiability_margin import (  # noqa: E402
    add_tie_break_noise_,
    certifiability_strict_fraction,
    global_keep_count,
    global_topk_threshold,
    normalized_margin,
    scores_for_cert_selection,
    smoke_check,
    streaming_exact_kth_largest,
    tau_relative,
    tie_break_scale,
)


def test_smoke_check_runs():
    smoke_check()


def test_partial_cert_fraction_matches_hand_count():
    s = torch.tensor([0.0, 1.0, 2.0, 3.0])
    star = torch.zeros(4)
    sel = scores_for_cert_selection(s, match_tie_break=False)
    tau, k, n = global_topk_threshold(sel, sparsity_percent=50.0)
    assert n == 4 and k == 2 and abs(float(tau) - 2.0) < 1e-5
    mrg = (sel - tau).abs()
    gap = (sel - star).abs()
    ok, tot = certifiability_strict_fraction(gap, mrg)
    assert tot == 4 and ok == 1


# ---------------------------------------------------------------- streaming k-th largest


def _chunks_of(x: torch.Tensor, sizes):
    def src():
        off = 0
        for n in sizes:
            yield x[off : off + n]
            off += n
        if off < x.numel():
            yield x[off:]

    return src


def _brute_kth(x: torch.Tensor, k: int) -> float:
    finite = x[torch.isfinite(x)].double()
    vals, _ = torch.topk(finite, k, largest=True)
    return float(vals.min().item())


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_streaming_kth_matches_brute_force_on_continuous_scores(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(20_000, generator=g, dtype=torch.float64) * 1e-4
    for k in (1, 7, 500, 10_000, 19_999, 20_000):
        tau, info = streaming_exact_kth_largest(_chunks_of(x, [3_333, 7, 9_000]), k, num_bins=64)
        assert tau == pytest.approx(_brute_kth(x, k), rel=1e-12, abs=0.0), (k, info)


def test_streaming_kth_handles_heavy_zero_mass_point():
    """The regime that matters here: ~99% of a bf16 delta is exactly zero (RESULTS.md §3.0)."""
    x = torch.zeros(100_000, dtype=torch.float64)
    x[:900] = torch.linspace(1e-6, 1e-4, 900, dtype=torch.float64)
    # k inside the signal support -> a real positive boundary
    tau, info = streaming_exact_kth_largest(_chunks_of(x, [10_000] * 9), 500, num_bins=128)
    assert tau == pytest.approx(_brute_kth(x, 500), rel=1e-9)
    assert tau > 0.0
    # k beyond the signal support -> the boundary is the zero mass point itself
    tau0, info0 = streaming_exact_kth_largest(_chunks_of(x, [10_000] * 9), 50_000, num_bins=128)
    assert tau0 == 0.0, info0


def test_streaming_kth_ignores_neg_inf_layer_floors():
    """Hybrid tau marks each layer's floor picks as -inf; they must not enter the ranking."""
    x = torch.arange(1.0, 1001.0, dtype=torch.float64)
    x[:100] = float("-inf")
    tau, info = streaming_exact_kth_largest(_chunks_of(x, [250, 250, 250]), 10, num_bins=32)
    assert tau == pytest.approx(991.0)
    assert int(info["n_finite"]) == 900
    assert int(info["n_total"]) == 1000


def test_streaming_kth_reports_when_budget_exceeds_finite_support():
    x = torch.full((100,), float("-inf"), dtype=torch.float64)
    x[:10] = torch.arange(1.0, 11.0, dtype=torch.float64)
    tau, info = streaming_exact_kth_largest(_chunks_of(x, [50]), 20, num_bins=16)
    assert torch.isnan(torch.tensor(tau))
    assert info["status"] == "k_exceeds_finite"


def test_streaming_kth_all_equal_is_that_value():
    x = torch.full((5_000,), 3.25, dtype=torch.float64)
    tau, info = streaming_exact_kth_largest(_chunks_of(x, [1_000]), 2_500, num_bins=64)
    assert tau == pytest.approx(3.25)


def test_streaming_kth_agrees_with_global_topk_threshold():
    g = torch.Generator().manual_seed(11)
    x = torch.rand(50_000, generator=g, dtype=torch.float64)
    rho = 97.5
    k = global_keep_count(int(x.numel()), rho)
    tau_ref, k_ref, n_ref = global_topk_threshold(x, rho)
    tau, _ = streaming_exact_kth_largest(_chunks_of(x, [7_777]), k, num_bins=256)
    assert k == k_ref and n_ref == 50_000
    assert tau == pytest.approx(float(tau_ref), rel=1e-12)


# ---------------------------------------------------------------- tie-break amplitude


def test_tie_break_scale_is_purely_relative():
    assert tie_break_scale(2.0e-4, 1e-12) == pytest.approx(2.0e-16)
    assert tie_break_scale(2.0e-4, 0.0) == 0.0
    assert tie_break_scale(0.0, 1e-12) == 0.0


def _realistic_scores(n_signal: int = 4_000, n_zero: int = 200_000) -> torch.Tensor:
    """Scores with the shape the real ones have: a heavy zero mass plus a signal tail spanning
    many decades (fp32-accumulated |Δθ| runs from ~1e-11 up to ~2e-4; RESULTS.md §3.0/§3.1)."""
    g = torch.Generator().manual_seed(3)
    x = torch.zeros(n_signal + n_zero, dtype=torch.float32)
    expo = torch.rand(n_signal, generator=g, dtype=torch.float32) * 7.0 - 11.0  # log10 in [-11, -4]
    x[:n_signal] = torch.pow(torch.tensor(10.0), expo) * 2.0
    return x


def test_tiny_tie_break_noise_does_not_reorder_genuine_scores():
    """1e-12 relative must separate exact zeros without permuting anything carrying signal."""
    n_signal = 4_000
    x = _realistic_scores(n_signal=n_signal)
    scale = tie_break_scale(float(x.max().item()), 1e-12)

    sel = x.clone()
    add_tie_break_noise_(sel, scale=scale, generator=torch.Generator().manual_seed(42))

    # No *distinct* signal value moves: sorting by the perturbed scores reproduces the exact
    # descending value sequence of the raw scores. (Coordinates that were exactly tied may swap
    # with each other — that is the tie-break working, not a reordering.)
    signal = x[:n_signal]
    order_sel = torch.argsort(sel[:n_signal], descending=True)
    assert torch.equal(signal[order_sel], torch.sort(signal, descending=True).values)
    # Every signal coordinate still outranks every zero coordinate.
    assert float(sel[:n_signal].min().item()) > float(sel[n_signal:].max().item())
    # ... and the zeros are no longer all tied, so a tie-break exists at all.
    assert int(torch.unique(sel[n_signal:]).numel()) > 1_000


def test_legacy_tie_break_amplitude_lets_zeros_outrank_real_signal():
    """Why the amplitude had to change: at 1e-6 relative the noise is not breaking ties, it is
    randomizing the bottom of the ranking (RESULTS.md §3.1)."""
    n_signal = 4_000
    x = _realistic_scores(n_signal=n_signal)
    legacy = x.clone()
    add_tie_break_noise_(
        legacy,
        scale=tie_break_scale(float(x.max().item()), 1e-6),
        generator=torch.Generator().manual_seed(42),
    )
    zero_ceiling = float(legacy[n_signal:].max().item())
    outranked = int((legacy[:n_signal] < zero_ceiling).sum().item())
    assert outranked > n_signal // 10, outranked
    # Signal ordering is also permuted, which the 1e-12 amplitude never does.
    assert not torch.equal(
        torch.argsort(x[:n_signal], descending=True),
        torch.argsort(legacy[:n_signal], descending=True),
    )


# ---------------------------------------------------------------- tau-relative normalization


def test_tau_relative_is_rank_preserving():
    g = torch.Generator().manual_seed(5)
    x = torch.rand(10_000, generator=g, dtype=torch.float64) * 3e-5
    tau, _, _ = global_topk_threshold(x, 97.5)
    xr = tau_relative(x, tau)
    assert torch.equal(torch.argsort(x, descending=True), torch.argsort(xr, descending=True))
    # tau maps to exactly 1.0, so the boundary is a shared origin across arms.
    tau_r, _, _ = global_topk_threshold(xr, 97.5)
    assert tau_r == pytest.approx(1.0, rel=1e-12)


def test_normalized_margin_puts_zero_score_coordinates_at_one():
    x = torch.tensor([0.0, 0.0, 1e-5, 4e-5], dtype=torch.float64)
    m = normalized_margin(x, 2e-5)
    assert m[0] == pytest.approx(1.0) and m[1] == pytest.approx(1.0)
    assert m[2] == pytest.approx(0.5)
    assert m[3] == pytest.approx(1.0)


def test_tau_relative_refuses_degenerate_tau():
    with pytest.raises(ValueError):
        tau_relative(torch.ones(4), 0.0)
