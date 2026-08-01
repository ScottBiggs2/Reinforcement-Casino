"""streaming_global_topk_threshold matches brute-force global_topk_threshold on chunked data."""

import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.analysis.certifiability_margin import (  # noqa: E402
    global_keep_count,
    global_topk_threshold,
    scaled_hybrid_floor_counts_per_layer,
    streaming_global_topk_threshold,
    streaming_min_of_global_top_r,
    tau_hybrid_global_phase_from_flat,
)


def test_streaming_tau_matches_cat_bruteforce():
    torch.manual_seed(7)
    parts = [torch.randn(120), torch.randn(89), torch.randn(55)]
    flat = torch.cat(parts)
    N = flat.numel()
    rho = 97.5

    def chunks():
        for p in parts:
            yield p.float()

    t_brute, k_b, n_b = global_topk_threshold(flat.float(), rho)
    t_stream, k_s, n_s = streaming_global_topk_threshold(
        total_n=N, sparsity_percent=rho, iter_sel_chunks_fp32=chunks()
    )
    assert n_b == n_s == N
    assert k_b == k_s
    assert abs(t_brute - t_stream) < 1e-4 * max(1.0, abs(t_brute))


def test_streaming_tau_hand_small():
    parts = [torch.tensor([1.0, 5.0, 3.0]), torch.tensor([4.0, 2.0])]
    flat = torch.cat(parts)
    N = flat.numel()

    def chunks():
        for p in parts:
            yield p

    t_stream, k_s, _ = streaming_global_topk_threshold(
        total_n=N, sparsity_percent=40.0, iter_sel_chunks_fp32=chunks()
    )
    t_brute, _, _ = global_topk_threshold(flat, 40.0)
    assert abs(t_stream - t_brute) < 1e-6


def test_streaming_min_top_r_matches_global_tau_when_r_is_k():
    """No floors: min(top-R) over the stream with R=k equals global τ."""
    torch.manual_seed(11)
    parts = [torch.randn(30), torch.randn(41)]
    flat = torch.cat(parts).float()
    N = flat.numel()
    rho = 97.5
    k = global_keep_count(N, rho)

    def chunks():
        for p in parts:
            yield p.float()

    t_global, _, _ = global_topk_threshold(flat, rho)
    t_min_r = streaming_min_of_global_top_r(chunks(), k)
    assert abs(t_global - t_min_r) < 1e-4 * max(1.0, abs(t_global))


def test_hybrid_tau_stream_matches_materialize():
    # Two layers [1,5,3] and [4,2,6,0]; one floor weight per layer (largest each); keep budget 5 → R=3.
    flat = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0, 6.0, 0.0], dtype=torch.float32)
    layer_numels = [3, 4]
    floors = [1, 1]
    keep_count = 5
    tau_mat, _, _, _, _ = tau_hybrid_global_phase_from_flat(flat, layer_numels, floors, keep_count)

    def hybrid_pieces():
        x0 = flat[:3].clone()
        _, i0 = torch.topk(x0, 1, largest=True)
        x0[i0] = float("-inf")
        yield x0
        x1 = flat[3:].clone()
        _, i1 = torch.topk(x1, 1, largest=True)
        x1[i1] = float("-inf")
        yield x1

    R = keep_count - sum(floors)
    tau_stream = streaming_min_of_global_top_r(hybrid_pieces(), R)
    assert abs(tau_mat - tau_stream) < 1e-5
    assert abs(float(tau_mat) - 2.0) < 1e-5


def test_scaled_floors_used_by_hybrid():
    numels = [100, 100]
    keep = 10
    floors = scaled_hybrid_floor_counts_per_layer(numels, min_layer_keep_ratio=0.2, keep_count=keep)
    assert sum(floors) <= keep
    assert len(floors) == 2
