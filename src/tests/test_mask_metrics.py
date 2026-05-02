"""Unit tests for mask Jaccard, CKA core, effective rank, and sparsity stats."""

import os
import sys

import torch

# Repo root on path for `src.*` imports
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.cold_start.export_layer_metrics_csv import compute_basic_stats, effective_rank
from src.cold_start.mask_jaccard_aggregates import (
    classify_param_bucket,
    jaccard_by_param_bucket,
)
from src.cold_start.mask_to_cka import linear_cka
from src.cold_start.mask_to_jaccard import compute_jaccard
from src.utils.mask_utils import compute_jaccard_similarity


def test_jaccard_agrees_mask_to_jaccard_and_mask_utils():
    masks_a = {
        "model.layers.0.mlp.gate_proj.weight": torch.tensor([[1, 1, 0], [0, 1, 0]]).bool(),
        "model.layers.0.self_attn.q_proj.weight": torch.tensor([[1, 0]]).bool(),
    }
    masks_b = {
        "model.layers.0.mlp.gate_proj.weight": torch.tensor([[1, 0, 0], [0, 1, 1]]).bool(),
        "model.layers.0.self_attn.q_proj.weight": torch.tensor([[1, 1]]).bool(),
    }
    j1 = compute_jaccard(masks_a, masks_b, device="cpu")
    j2 = compute_jaccard_similarity(masks_a, masks_b)
    assert j2 is not None
    assert abs(j1["aggregate_jaccard"] - j2["aggregate_jaccard"]) < 1e-9
    assert abs(j1["mean_jaccard"] - j2["mean_jaccard"]) < 1e-5
    for k in j1["per_layer"]:
        assert abs(j1["per_layer"][k] - j2["per_layer"][k]) < 1e-5


def test_jaccard_identical_masks():
    m = {"w": torch.ones(2, 3).bool()}
    j = compute_jaccard(m, m, device="cpu")
    assert j["aggregate_jaccard"] == 1.0
    assert j["mean_jaccard"] == 1.0


def test_jaccard_no_common_keys():
    a = {"x": torch.ones(1).bool()}
    b = {"y": torch.ones(1).bool()}
    j = compute_jaccard(a, b, device="cpu")
    assert j["aggregate_jaccard"] == 0.0
    assert j["per_layer"] == {}


def test_jaccard_all_zero_union():
    a = {"w": torch.zeros(2, 2).bool()}
    b = {"w": torch.zeros(2, 2).bool()}
    j = compute_jaccard(a, b, device="cpu")
    assert j["aggregate_jaccard"] == 0.0


def test_linear_cka_identical_activations():
    torch.manual_seed(0)
    X = torch.randn(8, 5)
    c = linear_cka(X, X.clone())
    assert abs(c - 1.0) < 1e-5


def test_linear_cka_range():
    torch.manual_seed(1)
    X = torch.randn(12, 6)
    Y = torch.randn(12, 6)
    c = linear_cka(X, Y)
    assert c == c  # not nan
    assert 0.0 <= c <= 1.0 + 1e-5


def test_effective_rank_rank_one_matrix():
    W = torch.ones(4, 4)
    er, er_n = effective_rank(W)
    assert er is not None and er_n is not None
    assert abs(er - 1.0) < 0.01
    assert abs(er_n - 1.0 / 4.0) < 0.01  # normalized by min(4,4)=4


def test_effective_rank_ndim1_returns_none():
    er, er_n = effective_rank(torch.ones(10))
    assert er is None and er_n is None


def test_compute_basic_stats_density():
    t = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    s = compute_basic_stats(t)
    assert s["n_params"] == 4
    assert s["n_kept"] == 3
    assert abs(s["density"] - 0.75) < 1e-6
    assert abs(s["sparsity"] - 0.25) < 1e-6


def test_classify_param_bucket():
    assert classify_param_bucket("model.layers.0.self_attn.q_proj.weight") == "attn"
    assert classify_param_bucket("model.layers.0.mlp.down_proj.weight") == "mlp"
    assert classify_param_bucket("model.layers.0.input_layernorm.weight") == "norm"


def test_jaccard_by_param_bucket_weighted():
    masks_a = {
        "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2).bool(),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros(2, 2).bool(),
    }
    masks_b = {
        "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2).bool(),
        "model.layers.0.mlp.gate_proj.weight": torch.ones(2, 2).bool(),
    }
    buckets = jaccard_by_param_bucket(masks_a, masks_b, device="cpu")
    assert buckets["attn"]["aggregate_jaccard"] == 1.0
    assert buckets["mlp"]["aggregate_jaccard"] == 0.0
