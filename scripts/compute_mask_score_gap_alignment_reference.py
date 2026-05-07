#!/usr/bin/env python3
"""
Compute an alignment-optimal scoring reference (Appendix optimal scoring) and emit *sidecar* artifacts
next to an existing mask-score-gap run directory.

This never overwrites:
  mask_score_gap_histograms.npz, mask_score_gap_summary.csv, mask_score_gap_gap_diagnostics.json

It writes:
  mask_score_gap_alignment_gap_diagnostics.json
  mask_score_gap_alignment_histograms.npz

Requires:
  - existing analysis dir with mask_score_gap_run.json and magnitude_caches/
  - Fisher diagonal proxy for H_ii as a torch-save dict name->tensor (fp32 recommended)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.analysis.alignment_optimal_scores import fixed_point_B_star, s_star_tensor
from src.analysis.mask_score_gap_analysis import (  # reuse histogram helpers
    LogSpaceHistogram,
    MomentAccum,
    _torch_load_analysis_shard,
    checkpoint_dtype_arg,
)
from src.warm_start.checkpoint_diff_mask_finder import load_state_dict


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _torch_load_any(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _iter_keys_in_order(final_sd: Dict[str, torch.Tensor]) -> List[str]:
    return list(final_sd.keys())


def _load_mag_cache_dict(cache_dir: Path, step: int) -> Dict[str, torch.Tensor]:
    p = cache_dir / f"mag_aggregate_step_{int(step)}.pt"
    if not p.is_file():
        raise FileNotFoundError(p)
    d = _torch_load_any(p)
    if not isinstance(d, dict):
        raise TypeError(f"Expected dict in {p}, got {type(d)}")
    return d


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute alignment-optimal s* reference sidecars for mask-score-gap.")
    ap.add_argument("--analysis-dir", type=str, required=True, help="Existing mask-score-gap OUT_DIR.")
    ap.add_argument("--fisher-diag", type=str, required=True, help="Path to fisher diagonal .pt (dict name->tensor).")
    ap.add_argument("--C", type=float, required=True, help="Remainder bound constant C (see appendix).")
    ap.add_argument("--epsilon-grad", type=float, required=True, help="Gradient-norm bound epsilon (see appendix).")
    ap.add_argument("--eta", type=float, default=1e-6, help="Convergence tolerance for B iteration.")
    ap.add_argument("--max-iter", type=int, default=25)
    ap.add_argument("--chunk-numel", type=int, default=10_000_000, help="Chunk size for streaming top-k tau.")
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument(
        "--checkpoint-dtype",
        type=checkpoint_dtype_arg,
        default=checkpoint_dtype_arg("bfloat16"),
        help="Dtype for loading checkpoints into RAM (bf16 recommended).",
    )
    ap.add_argument("--histogram-bins", type=int, default=1024)
    ap.add_argument("--max-keys", type=int, default=None, help="Optional truncate key list for smoke.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.analysis_dir)
    run_meta_path = out_dir / "mask_score_gap_run.json"
    if not run_meta_path.is_file():
        raise FileNotFoundError(f"Missing {run_meta_path} (pass a mask-score-gap analysis dir).")
    run = _load_json(run_meta_path)

    cache_dir = out_dir / "magnitude_caches"
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Missing magnitude_caches/ under {out_dir}")

    initial_model = str(run["initial_model"])
    final_model = str(run["final_model"])
    sparsity_percent = float(run["sparsity_percent"])
    milestones = [int(x) for x in run.get("magnitude_milestones", [])]
    if not milestones:
        raise ValueError("mask_score_gap_run.json missing magnitude_milestones")

    print("Loading Fisher diagonal:", args.fisher_diag)
    fisher = _torch_load_any(Path(args.fisher_diag))
    if not isinstance(fisher, dict):
        raise TypeError("fisher-diag must be torch-saved dict name->tensor")

    print("Loading initial/final state dicts on CPU:", initial_model, final_model)
    ckpt_dt = args.checkpoint_dtype
    initial_sd = load_state_dict(initial_model, device="cpu", torch_dtype=ckpt_dt)
    final_sd = load_state_dict(final_model, device="cpu", torch_dtype=ckpt_dt)

    keys = _iter_keys_in_order(final_sd)
    keys = [k for k in keys if k in initial_sd and k in fisher]
    if args.max_keys is not None:
        keys = keys[: max(0, int(args.max_keys))]
    print(f"Eligible keys: {len(keys)}")

    # Fisher must match final shapes
    fisher_ok: Dict[str, torch.Tensor] = {}
    for k in keys:
        if k in fisher and isinstance(fisher[k], torch.Tensor) and fisher[k].shape == final_sd[k].shape:
            fisher_ok[k] = fisher[k]
    keys = [k for k in keys if k in fisher_ok]
    print(f"Shape-matched keys: {len(keys)}")

    theta = {k: final_sd[k] for k in keys}

    fp = fixed_point_B_star(
        theta=theta,
        hdiag=fisher_ok,
        keys=keys,
        sparsity_percent=sparsity_percent,
        C=float(args.C),
        eps_grad=float(args.epsilon_grad),
        eta=float(args.eta),
        max_iter=int(args.max_iter),
        chunk_numel=int(args.chunk_numel),
    )
    print("Fixed point:", fp)

    # Histograms: gaps between s* and (oracle |Δw|), warm magnitude (per milestone), and random U(0,1)
    bins = int(args.histogram_bins)
    h_oracle = LogSpaceHistogram(bins)
    h_rand = LogSpaceHistogram(bins)
    h_mag = {int(s): LogSpaceHistogram(bins) for s in milestones}
    w_oracle = MomentAccum()
    w_rand = MomentAccum()
    w_mag = {int(s): MomentAccum() for s in milestones}

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(args.random_seed))

    # Preload mag caches (dict name->tensor fp32) per milestone.
    mag_by_step: Dict[int, Dict[str, torch.Tensor]] = {}
    for step in milestones:
        print("Loading mag cache step", step)
        mag_by_step[int(step)] = _load_mag_cache_dict(cache_dir, int(step))

    B_star = float(fp.B_star)
    for name in keys:
        w0 = initial_sd[name]
        w1 = final_sd[name]
        h = fisher_ok[name]
        if not w0.is_floating_point() or not w1.is_floating_point() or not h.is_floating_point():
            continue
        if w0.shape != w1.shape or w1.shape != h.shape:
            continue
        theta_t = w1
        sopt = s_star_tensor(theta_t, h, B=B_star, C=float(args.C), eps_grad=float(args.epsilon_grad))

        soracle = (w1 - w0).abs().float()
        v_or = (sopt - soracle).abs()
        h_oracle.update(v_or)
        w_oracle.update(v_or)

        s_rand = torch.rand(sopt.shape, generator=gen, dtype=torch.float32)
        v_r = (sopt - s_rand).abs()
        h_rand.update(v_r)
        w_rand.update(v_r)

        for step in milestones:
            md = mag_by_step[int(step)]
            if name not in md:
                continue
            sm = md[name].float()
            if sm.shape != sopt.shape:
                continue
            v_m = (sopt - sm).abs()
            h_mag[int(step)].update(v_m)
            w_mag[int(step)].update(v_m)

    diag = {
        "s_star_version": 1,
        "analysis_dir": str(out_dir),
        "fisher_diag": str(args.fisher_diag),
        "C": float(args.C),
        "epsilon_grad": float(args.epsilon_grad),
        "eta": float(args.eta),
        "max_iter": int(args.max_iter),
        "converged": bool(fp.converged),
        "iters": int(fp.iters),
        "B_trace": list(map(float, fp.trace_B)),
        "B_star": float(fp.B_star),
        "tau_star": float(fp.tau_star),
        "sparsity_percent": float(sparsity_percent),
        "histogram_bins": int(bins),
        "num_keys": int(len(keys)),
    }
    (out_dir / "mask_score_gap_alignment_gap_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    npz_payload: Dict[str, np.ndarray] = {"hist_bins": np.array([bins], dtype=np.int64)}
    npz_payload["alignment_gap_oracle_raw_counts"] = h_oracle.counts
    npz_payload["alignment_gap_oracle_raw_log_edges"] = h_oracle.edges
    npz_payload["alignment_gap_random_raw_counts"] = h_rand.counts
    npz_payload["alignment_gap_random_raw_log_edges"] = h_rand.edges
    for step in milestones:
        npz_payload[f"alignment_gap_magnitude_step{int(step)}_raw_counts"] = h_mag[int(step)].counts
        npz_payload[f"alignment_gap_magnitude_step{int(step)}_raw_log_edges"] = h_mag[int(step)].edges

    np.savez(out_dir / "mask_score_gap_alignment_histograms.npz", **npz_payload)
    print("Wrote mask_score_gap_alignment_histograms.npz + diagnostics JSON under", out_dir)


if __name__ == "__main__":
    main()

