import torch


def test_fixed_point_converges_toy() -> None:
    # Toy: 2 tensors, simple Fisher diag, should converge quickly.
    from src.analysis.alignment_optimal_scores import fixed_point_B_star

    theta = {
        "a": torch.tensor([1.0, -2.0, 3.0]),
        "b": torch.tensor([0.5, -0.25]),
    }
    h = {
        "a": torch.tensor([1.0, 1.0, 1.0]),
        "b": torch.tensor([2.0, 2.0]),
    }
    keys = ["a", "b"]
    res = fixed_point_B_star(theta=theta, hdiag=h, keys=keys, sparsity_percent=50.0, C=0.1, eps_grad=0.0, eta=1e-8)
    assert res.iters >= 1
    assert len(res.trace_B) == res.iters + 1
    assert res.total_n == 5
    assert res.keep_count == 2  # keep 50% of 5 -> int(0.5*5)=2 in global_keep_count
    assert res.B_star >= 0.0

