"""Smoke tests for global tau and strict certifiability fraction."""

import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.analysis.certifiability_margin import (  # noqa: E402
    certifiability_strict_fraction,
    global_topk_threshold,
    scores_for_cert_selection,
    smoke_check,
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
