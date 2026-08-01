"""Merge path for parallel mask-score-gap shards (no checkpoint loads)."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.analysis.mask_score_gap_analysis import (  # noqa: E402
    PARALLEL_SHARD_VERSION,
    LogSpaceHistogram,
    LinSpaceHistogram,
    MomentAccum,
    merge_parallel_shards,
    parallel_shards_dir,
    _atomic_torch_save,
    _lin_hist_to_dict,
    _log_hist_to_dict,
    _moment_to_dict,
    _write_shard_done,
)


def _tiny_baseline_shard(sd: Path, milestones: list[int]) -> None:
    h = 8
    seeds = [42]
    oracle_h = LogSpaceHistogram(h)
    oracle_h.counts[1] = 3
    oracle_w = MomentAccum(n=10, sum=1.0, sumsq=0.5)
    gap_diag = {
        "_note": "test",
        "cert_oracle_tau": 0.5,
        "cert_random_seed42_tau": 0.25,
        "cert_strict_frac_random_seed42": 0.1,
        "cert_strict_numer_random_seed42": 1,
        "cert_strict_denom_random_seed42": 10,
    }
    layer_rows = [{"param_name": "p1", "numel": 4, "mean_v_rand_raw_seed42": 0.1}]
    payload = {
        "version": PARALLEL_SHARD_VERSION,
        "kind": "baseline",
        "histogram_bins": h,
        "milestones": milestones,
        "random_seeds": seeds,
        "gap_diag": gap_diag,
        "layer_rows": layer_rows,
        "oracle_margin_hist": _log_hist_to_dict(oracle_h),
        "oracle_margin_w": _moment_to_dict(oracle_w),
        "rnd_raw_hists": {42: _log_hist_to_dict(LogSpaceHistogram(h))},
        "rnd_norm_hists": {42: _lin_hist_to_dict(LinSpaceHistogram(h))},
        "rnd_w_raw": {42: _moment_to_dict(MomentAccum())},
        "rnd_w_norm": {42: _moment_to_dict(MomentAccum())},
        "rnd_margin_raw_hists": {42: _log_hist_to_dict(LogSpaceHistogram(h))},
        "rnd_margin_w_raw": {42: _moment_to_dict(MomentAccum())},
        "max_verify_diff": 0.0,
        "verify_key_worst": "",
        "cert_match_tie_break": True,
        "cert_min_layer_keep_ratio": 0.0,
        "cert_global_mode": "stream",
        "cert_tau_rule": "global",
        "cert_hybrid_min_layer_keep_ratio": 0.0,
        "sparsity_percent": 50.0,
        "flatten_dtype": "bfloat16",
    }
    bp = sd / "baseline_shard.pt"
    _atomic_torch_save(payload, bp)
    _write_shard_done(bp)


def _milestone_shard(sd: Path, step: int) -> None:
    h = 8
    mraw = LogSpaceHistogram(h)
    mraw.counts[2] = int(step)
    mnorm = LinSpaceHistogram(h)
    mw_raw = MomentAccum(n=int(step), sum=float(step), sumsq=float(step * step))
    mw_norm = MomentAccum()
    mm = LogSpaceHistogram(h)
    mm_w = MomentAccum(n=5, sum=1.0, sumsq=1.0)
    gap = {
        f"milestone_{step}_raw_frac_zero_exact": 0.0,
        f"cert_magnitude_step{step}_tau": 0.3,
        f"cert_strict_denom_magnitude_step{step}": 100,
        f"cert_strict_numer_magnitude_step{step}": 10,
    }
    payload = {
        "version": PARALLEL_SHARD_VERSION,
        "kind": "milestone",
        "step": step,
        "gap_diag": gap,
        "mag_raw_hist": _log_hist_to_dict(mraw),
        "mag_norm_hist": _lin_hist_to_dict(mnorm),
        "mag_w_raw": _moment_to_dict(mw_raw),
        "mag_w_norm": _moment_to_dict(mw_norm),
        "mag_margin_hist": _log_hist_to_dict(mm),
        "mag_margin_w": _moment_to_dict(mm_w),
        "layer_mag_updates": {"p1": {f"mean_v_mag_raw_step{step}": 0.01, f"mean_v_mag_norm_step{step}": 0.02}},
        "max_verify_diff": 0.0,
        "verify_key_worst": "",
    }
    mp = sd / f"milestone_{step}_shard.pt"
    _atomic_torch_save(payload, mp)
    _write_shard_done(mp)


def test_merge_parallel_shards_writes_outputs():
    milestones = [50, 100]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        sd = parallel_shards_dir(out)
        sd.mkdir(parents=True, exist_ok=True)
        _tiny_baseline_shard(sd, milestones)
        for m in milestones:
            _milestone_shard(sd, m)
        merge_parallel_shards(out_dir=out, milestones=milestones)
        assert (out / "mask_score_gap_gap_diagnostics.json").is_file()
        assert (out / "mask_score_gap_summary.csv").is_file()
        assert (out / "mask_score_gap_histograms.npz").is_file()
        raw = np.load(out / "mask_score_gap_histograms.npz")
        assert raw[f"magnitude_raw_step50_counts"][2] == 50
        assert raw[f"magnitude_raw_step100_counts"][2] == 100
