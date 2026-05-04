#!/usr/bin/env python3
"""
Compare mask construction scores against an oracle: per-weight gaps v_i = |s_i^other - s_i^oracle|.

Oracle scores: |w_final - w_initial| (same as checkpoint_diff_mask_finder).
Warm magnitude @T: sum over delta snapshots with step <= T of |w_t - w_base| (even_better_mask_finder magnitude).
Random: Uniform(0,1) per element.

Writes summary CSV, per-layer CSV, histogram NPZ, and run metadata JSON.

Smoke test (few layers, low RAM):
    python src/analysis/mask_score_gap_analysis.py \\
        --initial_model meta-llama/Llama-3.1-8B-Instruct \\
        --final_model /path/to/checkpoint-500 \\
        --delta_log_dir /path/to/deltas/run \\
        --magnitude_target_step 200 \\
        --out_dir /tmp/mask_gap_smoke \\
        --max_keys 3 \\
        --histogram_bins 256

Slurm: see scripts/slurm_mask_score_gap_light_r1.slurm
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.warm_start.checkpoint_diff_mask_finder import (  # noqa: E402
    is_mlp_param,
    load_state_dict,
)
from src.warm_start.even_better_mask_finder import load_deltas_streaming  # noqa: E402


def _git_rev() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def safe_torch_load(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass
class MomentAccum:
    """Streaming mean / population std over many tensor chunks."""

    n: int = 0
    sum: float = 0.0
    sumsq: float = 0.0

    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        xf = x.reshape(-1).double()
        self.n += xf.numel()
        self.sum += xf.sum().item()
        self.sumsq += (xf * xf).sum().item()

    def mean_value(self) -> float:
        return self.sum / self.n if self.n else 0.0

    def std_value(self) -> float:
        if self.n == 0:
            return 0.0
        m = self.mean_value()
        v = self.sumsq / self.n - m * m
        return float(np.sqrt(max(v, 0.0)))


class LogSpaceHistogram:
    """Histogram for nonnegative values with wide dynamic range (log10 bins)."""

    def __init__(self, num_bins: int, log_min: float = -30.0, log_max: float = 6.0):
        self.num_bins = num_bins
        self.log_min = log_min
        self.log_max = log_max
        self.edges = np.linspace(log_min, log_max, num_bins + 1)
        self.counts = np.zeros(num_bins, dtype=np.int64)

    def update(self, v: torch.Tensor, chunk: int = 50_000_000) -> None:
        if v.numel() == 0:
            return
        flat = v.reshape(-1).float().clamp(min=1e-30).cpu()
        n = flat.numel()
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            lx = torch.log10(flat[start:end]).numpy()
            c, _ = np.histogram(lx, bins=self.edges)
            self.counts += c.astype(np.int64)


class LinSpaceHistogram:
    """Histogram on [0, upper] with fixed upper (default 1.0 for normalized gaps)."""

    def __init__(self, num_bins: int, upper: float = 1.0):
        self.num_bins = num_bins
        self.upper = upper
        self.edges = np.linspace(0.0, upper, num_bins + 1)
        self.counts = np.zeros(num_bins, dtype=np.int64)

    def update(self, v: torch.Tensor, chunk: int = 50_000_000) -> None:
        if v.numel() == 0:
            return
        flat = v.reshape(-1).float().clamp(min=0.0, max=self.upper).cpu()
        n = flat.numel()
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            x = flat[start:end].numpy()
            c, _ = np.histogram(x, bins=self.edges)
            self.counts += c.astype(np.int64)


def _quantiles_from_hist_counts(counts: np.ndarray, edges: np.ndarray, qs: Sequence[float]) -> Dict[str, float]:
    """Piecewise-uniform inverse-CDF inside each bin (linear bin edges)."""
    total = float(counts.sum())
    out: Dict[str, float] = {}
    if total <= 0:
        for q in qs:
            out[f"p{int(round(q * 100)):02d}"] = float("nan")
        return out

    cum = np.concatenate([[0.0], np.cumsum(counts.astype(np.float64))])
    for q in qs:
        target = q * total
        idx = int(np.searchsorted(cum, target, side="right") - 1)
        idx = max(0, min(idx, len(counts) - 1))
        cum_before = cum[idx]
        bin_mass = float(counts[idx])
        if bin_mass <= 0:
            val = float(edges[idx])
        else:
            frac = (target - cum_before) / bin_mass
            frac = float(np.clip(frac, 0.0, 1.0))
            lo, hi = float(edges[idx]), float(edges[idx + 1])
            val = lo + frac * (hi - lo)
        out[f"p{int(round(q * 100)):02d}"] = val

    return out


def _quantiles_log_hist(counts: np.ndarray, log_edges: np.ndarray, qs: Sequence[float]) -> Dict[str, float]:
    """Quantiles on linear scale when histogram counts are over log10(values)."""
    total = float(counts.sum())
    out: Dict[str, float] = {}
    if total <= 0:
        for q in qs:
            out[f"p{int(round(q * 100)):02d}"] = float("nan")
        return out

    cum = np.concatenate([[0.0], np.cumsum(counts.astype(np.float64))])
    for q in qs:
        target = q * total
        idx = int(np.searchsorted(cum, target, side="right") - 1)
        idx = max(0, min(idx, len(counts) - 1))
        cum_before = cum[idx]
        bin_mass = float(counts[idx])
        if bin_mass <= 0:
            log_val = float(log_edges[idx])
        else:
            frac = (target - cum_before) / bin_mass
            frac = float(np.clip(frac, 0.0, 1.0))
            lo_l, hi_l = float(log_edges[idx]), float(log_edges[idx + 1])
            log_val = lo_l + frac * (hi_l - lo_l)
        out[f"p{int(round(q * 100)):02d}"] = float(10**log_val)

    return out


def compute_magnitude_scores(
    delta_log_dir: str,
    target_step: Optional[int],
    param_names_filter: Optional[Sequence[str]],
    mlp_only: bool,
) -> Dict[str, torch.Tensor]:
    steps_and_paths = load_deltas_streaming(delta_log_dir, target_step)
    if not steps_and_paths:
        raise FileNotFoundError(f"No delta snapshots under {delta_log_dir!r} for target_step={target_step!r}")

    aggregated: Dict[str, torch.Tensor] = {}
    param_names: Optional[List[str]] = None

    for step_idx, (step, delta_path) in enumerate(steps_and_paths):
        print(f"  [{step_idx + 1}/{len(steps_and_paths)}] magnitude accumulate step {step} ← {delta_path}")
        deltas = safe_torch_load(delta_path, map_location="cpu")

        if param_names is None:
            if param_names_filter is not None:
                param_names = [n for n in param_names_filter if n in deltas]
            else:
                param_names = [name for name in deltas.keys() if not mlp_only or is_mlp_param(name)]

            for name in param_names:
                if name in deltas:
                    aggregated[name] = torch.zeros_like(deltas[name], dtype=torch.float32)

        for name in param_names:
            if name in deltas:
                aggregated[name] += deltas[name].to(torch.float32).abs()

        del deltas
        gc.collect()

    return aggregated


def _minmax_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    xf = x.float()
    mn = xf.min()
    mx = xf.max()
    rng = mx - mn
    if float(rng) <= eps:
        return torch.zeros_like(xf)
    return (xf - mn) / rng


def process_keys(
    *,
    keys: List[str],
    initial_sd: Dict[str, torch.Tensor],
    final_sd: Dict[str, torch.Tensor],
    mag_scores: Dict[str, torch.Tensor],
    delta_verify: Optional[Dict[str, torch.Tensor]],
    random_seeds: List[int],
    histogram_bins: int,
    out_dir: Path,
) -> None:
    qs = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)

    mag_raw_hist = LogSpaceHistogram(histogram_bins)
    mag_norm_hist = LinSpaceHistogram(histogram_bins, upper=1.0)
    rnd_raw_hists = {s: LogSpaceHistogram(histogram_bins) for s in random_seeds}
    rnd_norm_hists = {s: LinSpaceHistogram(histogram_bins, upper=1.0) for s in random_seeds}

    mag_w_raw = MomentAccum()
    mag_w_norm = MomentAccum()
    rnd_w_raw = {s: MomentAccum() for s in random_seeds}
    rnd_w_norm = {s: MomentAccum() for s in random_seeds}

    layer_rows: List[Dict[str, object]] = []

    max_verify_diff = 0.0
    verify_key_worst = ""

    for ki, name in enumerate(keys):
        if name not in initial_sd or name not in final_sd:
            print(f"  skip {name} (missing in initial or final sd)")
            continue
        if name not in mag_scores:
            print(f"  skip {name} (missing in magnitude aggregate)")
            continue

        w0 = initial_sd[name].to(torch.float32)
        w1 = final_sd[name].to(torch.float32)
        s_oracle = (w1 - w0).abs()

        if delta_verify is not None and name in delta_verify:
            dv = delta_verify[name].to(torch.float32).abs()
            if dv.shape == s_oracle.shape:
                diff = (s_oracle - dv).abs().max().item()
                if diff > max_verify_diff:
                    max_verify_diff = diff
                    verify_key_worst = name

        s_mag = mag_scores[name].to(torch.float32)
        if s_mag.shape != s_oracle.shape:
            print(f"  skip {name} (shape mismatch mag {s_mag.shape} vs oracle {s_oracle.shape})")
            del w0, w1, s_oracle, s_mag
            gc.collect()
            continue

        v_mag_raw = (s_mag - s_oracle).abs()
        o_n = _minmax_norm(s_oracle)
        m_n = _minmax_norm(s_mag)
        v_mag_norm = (m_n - o_n).abs()

        mag_raw_hist.update(v_mag_raw)
        mag_norm_hist.update(v_mag_norm)
        mag_w_raw.update(v_mag_raw)
        mag_w_norm.update(v_mag_norm)

        row_base = {
            "param_name": name,
            "numel": int(s_oracle.numel()),
            "mean_v_mag_raw": float(v_mag_raw.mean().item()),
            "mean_v_mag_norm": float(v_mag_norm.mean().item()),
        }

        for seed in random_seeds:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            s_rand = torch.rand(s_oracle.shape, generator=g, dtype=torch.float32)
            v_rr = (s_rand - s_oracle).abs()
            r_n = _minmax_norm(s_rand)
            v_rn = (r_n - o_n).abs()
            rnd_raw_hists[seed].update(v_rr)
            rnd_norm_hists[seed].update(v_rn)
            rnd_w_raw[seed].update(v_rr)
            rnd_w_norm[seed].update(v_rn)
            row_base[f"mean_v_rand_raw_seed{seed}"] = float(v_rr.mean().item())
            row_base[f"mean_v_rand_norm_seed{seed}"] = float(v_rn.mean().item())

        layer_rows.append(row_base)

        del w0, w1, s_oracle, s_mag, v_mag_raw, o_n, m_n, v_mag_norm
        gc.collect()

        if (ki + 1) % 20 == 0:
            print(f"  processed {ki + 1}/{len(keys)} keys")

    if delta_verify is not None:
        print(f"verify deltas max |oracle - |delta500||: {max_verify_diff:.6e} (worst key: {verify_key_worst!r})")

    # Summary CSV + NPZ
    summary_path = out_dir / "mask_score_gap_summary.csv"
    by_layer_path = out_dir / "mask_score_gap_by_layer.csv"

    summary_rows: List[Dict[str, object]] = []

    def pack_row(
        case: str,
        acc: MomentAccum,
        h_log: Optional[LogSpaceHistogram],
        h_lin: Optional[LinSpaceHistogram],
    ) -> Dict[str, object]:
        if h_log is not None:
            qdict = _quantiles_log_hist(h_log.counts, h_log.edges, qs)
        elif h_lin is not None:
            qdict = _quantiles_from_hist_counts(h_lin.counts, h_lin.edges, qs)
        else:
            raise ValueError("pack_row requires h_log or h_lin")
        return {
            "case": case,
            "total_numel": acc.n,
            "mean": acc.mean_value(),
            "std": acc.std_value(),
            **qdict,
        }

    summary_rows.append(pack_row("magnitude_raw", mag_w_raw, mag_raw_hist, None))
    summary_rows.append(pack_row("magnitude_norm", mag_w_norm, None, mag_norm_hist))

    for seed in random_seeds:
        summary_rows.append(pack_row(f"random_raw_seed{seed}", rnd_w_raw[seed], rnd_raw_hists[seed], None))
        summary_rows.append(pack_row(f"random_norm_seed{seed}", rnd_w_norm[seed], None, rnd_norm_hists[seed]))

    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    if layer_rows:
        lf = list(layer_rows[0].keys())
        with by_layer_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=lf)
            writer.writeheader()
            for row in layer_rows:
                writer.writerow(row)

    npz_path = out_dir / "mask_score_gap_histograms.npz"
    npz_payload: Dict[str, np.ndarray] = {
        "hist_bins": np.array([histogram_bins], dtype=np.int64),
        "magnitude_raw_counts": mag_raw_hist.counts,
        "magnitude_raw_log_edges": mag_raw_hist.edges,
        "magnitude_norm_counts": mag_norm_hist.counts,
        "magnitude_norm_edges": mag_norm_hist.edges,
    }
    for seed in random_seeds:
        npz_payload[f"random_raw_seed{seed}_counts"] = rnd_raw_hists[seed].counts
        npz_payload[f"random_raw_seed{seed}_log_edges"] = rnd_raw_hists[seed].edges
        npz_payload[f"random_norm_seed{seed}_counts"] = rnd_norm_hists[seed].counts
        npz_payload[f"random_norm_seed{seed}_edges"] = rnd_norm_hists[seed].edges

    np.savez(npz_path, **npz_payload)
    print(f"Wrote {summary_path}, {by_layer_path}, {npz_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mask score gaps vs oracle (magnitude @T, random).")
    p.add_argument("--initial_model", type=str, required=True)
    p.add_argument("--final_model", type=str, required=True)
    p.add_argument("--delta_log_dir", type=str, required=True)
    p.add_argument("--magnitude_target_step", type=int, default=200)
    p.add_argument("--mlp_only", action="store_true")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument(
        "--random_seeds",
        type=str,
        default=None,
        help="Comma-separated extra seeds (includes --random_seed by default). Example: 42,43,44",
    )
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--histogram_bins", type=int, default=2048)
    p.add_argument("--max_keys", type=int, default=None, help="Debug: only process first N intersecting keys")
    p.add_argument(
        "--verify_deltas_dir",
        type=str,
        default=None,
        help="If set, load deltas_step_500.pt from this dir and print max |oracle-|delta500||.",
    )
    p.add_argument(
        "--magnitude_scores_cache",
        type=str,
        default=None,
        help="Optional path to read/write aggregated magnitude scores .pt to skip recomputation.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds: List[int] = [int(args.random_seed)]
    if args.random_seeds:
        for s in args.random_seeds.split(","):
            s = s.strip()
            if s:
                seeds.append(int(s))
    seeds = sorted(set(seeds))

    print("Loading initial / final state dicts (CPU fp32, high RAM)...")
    initial_sd = load_state_dict(args.initial_model, device="cpu")
    final_sd = load_state_dict(args.final_model, device="cpu")

    keys_all = [n for n in final_sd if n in initial_sd]
    if args.mlp_only:
        keys_all = [n for n in keys_all if is_mlp_param(n)]

    mag_cache_path = Path(args.magnitude_scores_cache) if args.magnitude_scores_cache else None
    if mag_cache_path and mag_cache_path.is_file():
        print(f"Loading magnitude aggregates from cache {mag_cache_path}")
        mag_scores = safe_torch_load(str(mag_cache_path), map_location="cpu")
    else:
        mag_scores = compute_magnitude_scores(
            args.delta_log_dir,
            args.magnitude_target_step,
            param_names_filter=keys_all,
            mlp_only=args.mlp_only,
        )
        if mag_cache_path:
            torch.save(mag_scores, mag_cache_path)
            print(f"Saved magnitude aggregates to {mag_cache_path}")

    keys = [k for k in keys_all if k in mag_scores]
    if args.max_keys is not None:
        keys = keys[: max(0, args.max_keys)]

    delta_verify: Optional[Dict[str, torch.Tensor]] = None
    if args.verify_deltas_dir:
        cand = Path(args.verify_deltas_dir) / "deltas_step_500.pt"
        if not cand.is_file():
            raise FileNotFoundError(f"verify expected {cand}")
        print(f"Loading verification snapshot {cand}")
        delta_verify = safe_torch_load(str(cand), map_location="cpu")

    meta = {
        "initial_model": args.initial_model,
        "final_model": args.final_model,
        "delta_log_dir": args.delta_log_dir,
        "magnitude_target_step": args.magnitude_target_step,
        "mlp_only": args.mlp_only,
        "random_seeds": seeds,
        "histogram_bins": args.histogram_bins,
        "num_keys": len(keys),
        "verify_deltas_dir": args.verify_deltas_dir,
        "git_rev": _git_rev(),
    }
    with (out_dir / "mask_score_gap_run.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Intersecting keys to process: {len(keys)}")
    process_keys(
        keys=keys,
        initial_sd=initial_sd,
        final_sd=final_sd,
        mag_scores=mag_scores,
        delta_verify=delta_verify,
        random_seeds=seeds,
        histogram_bins=args.histogram_bins,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
