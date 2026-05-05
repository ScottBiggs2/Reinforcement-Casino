#!/usr/bin/env python3
"""
Compare mask construction scores against an oracle: per-weight gaps v_i = |s_i^other - s_i^oracle|.

Oracle scores: |w_final - w_initial| (same as checkpoint_diff_mask_finder).
Warm magnitude milestones: sum of |w_snapshot - w_base| over snapshots with step ≤ t
(even_better_mask_finder semantics), evaluated at each intermediate step (default 50,100,150,200).
Random: Uniform(0,1) per element.

Writes summary CSV, per-layer CSV, histogram NPZ, gap diagnostics JSON, metadata JSON.

Log histogram bins use log10(v + eps_batch) where eps adapts per chunk from small positive values,
so stacking at 1e-30 from a hard clamp is avoided (zeros still map near log(eps)).

Slurm: see scripts/slurm_mask_score_gap_light_r1.slurm
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
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

from src.analysis.certifiability_margin import (  # noqa: E402
    certifiability_strict_fraction,
    flatten_scores_in_order,
    global_topk_threshold,
    scores_for_cert_selection,
)


def ensure_hf_hub_token() -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    candidates: List[Path] = []
    tfile = os.environ.get("HF_TOKEN_FILE")
    if tfile:
        candidates.append(Path(tfile).expanduser())
    candidates.append(Path.home() / ".cache" / "huggingface" / "token")
    for p in candidates:
        try:
            if p.is_file():
                tok = p.read_text(encoding="utf-8").strip()
                if tok:
                    os.environ["HF_TOKEN"] = tok
                    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", tok)
                    return
        except OSError:
            continue


def _git_rev() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def safe_torch_load(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass
class MomentAccum:
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


def _histogram_eps_positive(flat_nn: torch.Tensor) -> float:
    """Epsilon for log10(v + eps): avoids artefactual stacking at log(1e-30) while preserving tiny positives."""
    if flat_nn.numel() == 0:
        return 1e-30
    a = flat_nn.detach().float().cpu().numpy()
    q = float(np.quantile(a, 0.01))
    return max(1e-30, abs(q) * 1e-4)


class LogSpaceHistogram:
    """log10 bins on adjusted values log10(v + eps) with per-chunk epsilon from positive quantile."""

    def __init__(self, num_bins: int, log_min: float = -30.0, log_max: float = 6.0):
        self.num_bins = num_bins
        self.log_min = log_min
        self.log_max = log_max
        self.edges = np.linspace(log_min, log_max, num_bins + 1)
        self.counts = np.zeros(num_bins, dtype=np.int64)

    def update(self, v: torch.Tensor, chunk: int = 50_000_000) -> None:
        if v.numel() == 0:
            return
        flat = v.reshape(-1).float().cpu().clamp(min=0.0)
        n = flat.numel()
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            part = flat[start:end]
            pos = part[part > 0]
            eps = _histogram_eps_positive(pos)
            adj = part + eps
            lx = torch.log10(adj).numpy()
            lx = np.clip(lx, self.log_min + 1e-9, self.log_max)
            c, _ = np.histogram(lx, bins=self.edges)
            self.counts += c.astype(np.int64)


class LinSpaceHistogram:
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


def _quantiles_from_hist_counts(
    counts: np.ndarray, edges: np.ndarray, qs: Sequence[float]
) -> Dict[str, float]:
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
            frac = float(np.clip((target - cum_before) / bin_mass, 0.0, 1.0))
            lo, hi = float(edges[idx]), float(edges[idx + 1])
            val = lo + frac * (hi - lo)
        out[f"p{int(round(q * 100)):02d}"] = val
    return out


def _quantiles_log_hist(counts: np.ndarray, log_edges: np.ndarray, qs: Sequence[float]) -> Dict[str, float]:
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
            frac = float(np.clip((target - cum_before) / bin_mass, 0.0, 1.0))
            lo_l, hi_l = float(log_edges[idx]), float(log_edges[idx + 1])
            log_val = lo_l + frac * (hi_l - lo_l)
        out[f"p{int(round(q * 100)):02d}"] = float(10**log_val)
    return out


def build_magnitude_milestone_caches(
    delta_log_dir: str,
    milestones: Sequence[int],
    param_names_filter: Sequence[str],
    mlp_only: bool,
    cache_dir: Path,
    force_rebuild: bool,
) -> Dict[int, Path]:
    milestones = sorted({int(x) for x in milestones})
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {s: cache_dir / f"mag_aggregate_step_{s}.pt" for s in milestones}
    if not force_rebuild and all(p.is_file() for p in paths.values()):
        print(f"Magnitude caches present under {cache_dir}; use --force_magnitude_cache_rebuild to re-stream.")
        return paths

    max_step = max(milestones)
    steps_and_paths = load_deltas_streaming(delta_log_dir, max_step)
    if not steps_and_paths:
        raise FileNotFoundError(f"No delta snapshots under {delta_log_dir!r} up to step {max_step}")

    aggregated: Dict[str, torch.Tensor] = {}
    param_names: Optional[List[str]] = None
    milestones_set = set(milestones)
    pending_save = milestones_set.copy()

    for step_idx, (step, delta_path) in enumerate(steps_and_paths):
        print(f"  [{step_idx + 1}/{len(steps_and_paths)}] stream step {step} ← {delta_path}")
        deltas = safe_torch_load(delta_path, map_location="cpu")
        if param_names is None:
            param_names = [n for n in param_names_filter if n in deltas]
            if not param_names:
                param_names = [name for name in deltas.keys() if not mlp_only or is_mlp_param(name)]
            for name in param_names:
                if name in deltas:
                    aggregated[name] = torch.zeros_like(deltas[name], dtype=torch.float32)

        for name in param_names:
            if name in deltas:
                aggregated[name] += deltas[name].to(torch.float32).abs()

        del deltas
        gc.collect()

        if step in milestones_set and step in pending_save:
            out_p = paths[step]
            to_save = {k: aggregated[k].clone() for k in aggregated}
            torch.save(to_save, out_p)
            print(f"    saved milestone step {step} → {out_p}")
            pending_save.discard(step)

    if pending_save:
        seen = [s for s, _ in steps_and_paths]
        raise RuntimeError(
            f"Did not persist milestones {sorted(pending_save)}; delta steps observed: {seen}. "
            "Check --magnitude_milestones matches your deltas_step_*.pt schedule."
        )
    for m, pth in paths.items():
        if not pth.is_file():
            raise RuntimeError(f"Expected cache missing: {pth} (step {m})")

    return paths


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
    milestone_cache_paths: Dict[int, Path],
    milestones: List[int],
    delta_verify: Optional[Dict[str, torch.Tensor]],
    random_seeds: List[int],
    histogram_bins: int,
    out_dir: Path,
    sparsity_percent: float,
    cert_match_tie_break: bool,
    cert_min_layer_keep_ratio: float,
) -> None:
    qs = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    milestones = sorted(milestones)

    rnd_raw_hists = {s: LogSpaceHistogram(histogram_bins) for s in random_seeds}
    rnd_norm_hists = {s: LinSpaceHistogram(histogram_bins, upper=1.0) for s in random_seeds}
    rnd_w_raw = {s: MomentAccum() for s in random_seeds}
    rnd_w_norm = {s: MomentAccum() for s in random_seeds}
    rnd_margin_raw_hists = {s: LogSpaceHistogram(histogram_bins) for s in random_seeds}
    rnd_margin_w_raw = {s: MomentAccum() for s in random_seeds}

    mag_raw_hists = {step: LogSpaceHistogram(histogram_bins) for step in milestones}
    mag_norm_hists = {step: LinSpaceHistogram(histogram_bins, upper=1.0) for step in milestones}
    mag_w_raw = {step: MomentAccum() for step in milestones}
    mag_w_norm = {step: MomentAccum() for step in milestones}
    mag_margin_raw_hists = {step: LogSpaceHistogram(histogram_bins) for step in milestones}
    mag_margin_w_raw = {step: MomentAccum() for step in milestones}

    oracle_margin_hist = LogSpaceHistogram(histogram_bins)
    oracle_margin_w = MomentAccum()

    gap_diag: Dict[str, object] = {
        "_note": (
            "frac_zero_exact counts exact v=0 floats (oracle match at that coordinate). "
            "Tiny-but-positive gaps are distinct; log histogram uses adaptive eps."
        ),
        "cert_mode": (
            "global_topk_tau: flatten scores in analysis key order; τ = min(top-k values); "
            "m_i = |s_i^(sel) − τ|; cert_strict = mean_i 1[|s_i^(sel) − s*_i| < m_i]. "
            "Hybrid masks (min_layer_keep_ratio > 0 in mask_utils) are not equivalent to a single scalar τ."
        ),
        "cert_sparsity_percent": float(sparsity_percent),
        "cert_match_tie_break": bool(cert_match_tie_break),
        "cert_min_layer_keep_ratio": float(cert_min_layer_keep_ratio),
    }
    if cert_min_layer_keep_ratio > 0:
        gap_diag["cert_hybrid_warning"] = (
            f"cert_min_layer_keep_ratio={cert_min_layer_keep_ratio} > 0 only affects this diagnostic metadata; "
            "τ and m_i are still computed with pure global top-k (theorem-style). Training masks may use per-layer floors."
        )

    layer_rows: List[Dict[str, object]] = []
    max_verify_diff = 0.0
    verify_key_worst = ""

    rnd_flat_parts: Dict[int, List[torch.Tensor]] = {s: [] for s in random_seeds}
    oracle_flat_parts_rand: List[torch.Tensor] = []

    # ---------- random vs oracle (one pass over keys)
    print("Computing random-mask gaps vs oracle (one pass)...")
    for ki, name in enumerate(keys):
        if name not in initial_sd or name not in final_sd:
            continue
        w0 = initial_sd[name].to(torch.float32)
        w1 = final_sd[name].to(torch.float32)
        s_oracle = (w1 - w0).abs()
        oracle_flat_parts_rand.append(s_oracle.reshape(-1))
        row_base: Dict[str, object] = {
            "param_name": name,
            "numel": int(s_oracle.numel()),
        }
        o_n_r = _minmax_norm(s_oracle)
        for seed in random_seeds:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            s_rand = torch.rand(s_oracle.shape, generator=g, dtype=torch.float32)
            rnd_flat_parts[seed].append(s_rand.reshape(-1))
            v_rr = (s_rand - s_oracle).abs()
            r_n = _minmax_norm(s_rand)
            v_rn = (r_n - o_n_r).abs()
            rnd_raw_hists[seed].update(v_rr)
            rnd_norm_hists[seed].update(v_rn)
            rnd_w_raw[seed].update(v_rr)
            rnd_w_norm[seed].update(v_rn)
            row_base[f"mean_v_rand_raw_seed{seed}"] = float(v_rr.mean().item())
            row_base[f"mean_v_rand_norm_seed{seed}"] = float(v_rn.mean().item())

        layer_rows.append(row_base)
        del w0, w1, s_oracle, o_n_r
        gc.collect()

    oracle_flat_vec = torch.cat(oracle_flat_parts_rand) if oracle_flat_parts_rand else torch.empty(0)
    if oracle_flat_vec.numel() > 0:
        o_sel = scores_for_cert_selection(oracle_flat_vec, match_tie_break=cert_match_tie_break)
        tau_o, k_o, N_o = global_topk_threshold(o_sel, sparsity_percent)
        margin_o = (o_sel - tau_o).abs()
        oracle_margin_hist.update(margin_o)
        oracle_margin_w.update(margin_o)
        gap_diag["cert_oracle_tau"] = tau_o
        gap_diag["cert_oracle_k_keep"] = k_o
        gap_diag["cert_oracle_N"] = N_o
    for seed in random_seeds:
        parts = rnd_flat_parts.get(seed, [])
        if not parts or oracle_flat_vec.numel() == 0:
            continue
        flat_r = torch.cat(parts)
        if flat_r.numel() != oracle_flat_vec.numel():
            continue
        r_sel = scores_for_cert_selection(flat_r, match_tie_break=cert_match_tie_break)
        tau_r, k_r, N_r = global_topk_threshold(r_sel, sparsity_percent)
        margin_r = (r_sel - tau_r).abs()
        gap_cert_r = (r_sel - oracle_flat_vec).abs()
        c_ok, c_den = certifiability_strict_fraction(gap_cert_r, margin_r)
        rnd_margin_raw_hists[seed].update(margin_r)
        rnd_margin_w_raw[seed].update(margin_r)
        gap_diag[f"cert_random_seed{seed}_tau"] = tau_r
        gap_diag[f"cert_random_seed{seed}_k_keep"] = k_r
        gap_diag[f"cert_random_seed{seed}_N"] = N_r
        gap_diag[f"cert_strict_frac_random_seed{seed}"] = (c_ok / c_den) if c_den else float("nan")
        gap_diag[f"cert_strict_numer_random_seed{seed}"] = c_ok
        gap_diag[f"cert_strict_denom_random_seed{seed}"] = c_den

    # ---------- per milestone magnitude caches
    layer_by_name = {row["param_name"]: row for row in layer_rows}

    for step in milestones:
        path = milestone_cache_paths[step]
        print(f"Magnitude @{step}: load cached aggregates from {path}")
        mag_scores = safe_torch_load(str(path), map_location="cpu")
        n_elems = 0
        n_zero_raw = 0
        vmin_pos_accum = torch.tensor(float("inf"))
        vmax_accum = torch.tensor(float("-inf"))
        oracle_match_sample: List[Dict[str, object]] = []
        flat_mag_chunks: List[torch.Tensor] = []
        flat_oracle_chunks: List[torch.Tensor] = []

        for ki, name in enumerate(keys):
            if name not in initial_sd or name not in final_sd or name not in mag_scores:
                continue
            w0 = initial_sd[name].to(torch.float32)
            w1 = final_sd[name].to(torch.float32)
            s_oracle = (w1 - w0).abs()
            s_mag = mag_scores[name].to(torch.float32)
            if s_mag.shape != s_oracle.shape:
                del w0, w1, s_oracle, s_mag
                continue

            flat_mag_chunks.append(s_mag.reshape(-1).clone())
            flat_oracle_chunks.append(s_oracle.reshape(-1).clone())

            if delta_verify is not None and name in delta_verify:
                dv = delta_verify[name].to(torch.float32).abs()
                if dv.shape == s_oracle.shape:
                    diff = (s_oracle - dv).abs().max().item()
                    if diff > max_verify_diff:
                        max_verify_diff = diff
                        verify_key_worst = name

            v_mag_raw = (s_mag - s_oracle).abs()
            v_zero = v_mag_raw == 0
            ne = int(v_mag_raw.numel())
            nz = int(v_zero.sum().item())
            n_elems += ne
            n_zero_raw += nz

            vp = v_mag_raw[v_mag_raw > 0]
            if vp.numel():
                vmin_pos_accum = torch.minimum(vmin_pos_accum, vp.min())
                vmax_accum = torch.maximum(vmax_accum, v_mag_raw.max())

            o_n = _minmax_norm(s_oracle)
            m_n = _minmax_norm(s_mag)
            v_mag_norm = (m_n - o_n).abs()

            mag_raw_hists[step].update(v_mag_raw)
            mag_norm_hists[step].update(v_mag_norm)
            mag_w_raw[step].update(v_mag_raw)
            mag_w_norm[step].update(v_mag_norm)

            ln = layer_by_name.get(name)
            if ln is None:
                ln = {"param_name": name, "numel": int(s_oracle.numel())}
                layer_by_name[name] = ln
                layer_rows.append(ln)
            ln[f"mean_v_mag_raw_step{step}"] = float(v_mag_raw.mean().item())
            ln[f"mean_v_mag_norm_step{step}"] = float(v_mag_norm.mean().item())

            if len(oracle_match_sample) < 2:
                oracle_match_sample.append(
                    {"key": name, "frac_zero": nz / max(ne, 1), "mean_v_raw": float(v_mag_raw.mean().item())}
                )

            del w0, w1, s_oracle, s_mag, v_mag_raw, o_n, m_n, v_mag_norm
            gc.collect()

            if (ki + 1) % 60 == 0:
                print(f"  @{step}: processed key {ki + 1}/{len(keys)}")

        gap_diag[f"milestone_{step}_raw_frac_zero_exact"] = (n_zero_raw / n_elems) if n_elems else 0.0
        _mn = float(vmin_pos_accum.item()) if n_elems else float("nan")
        gap_diag[f"milestone_{step}_raw_min_positive"] = (
            _mn if math.isfinite(_mn) else float("nan")
        )
        _mx = float(vmax_accum.item()) if n_elems else float("nan")
        gap_diag[f"milestone_{step}_raw_global_max"] = _mx if math.isfinite(_mx) else float("nan")
        gap_diag[f"milestone_{step}_sample_tensors"] = oracle_match_sample[:2]

        if flat_mag_chunks:
            flat_mag = torch.cat(flat_mag_chunks)
            flat_oracle_m = torch.cat(flat_oracle_chunks)
            m_sel = scores_for_cert_selection(flat_mag, match_tie_break=cert_match_tie_break)
            tau_m, k_m, N_m = global_topk_threshold(m_sel, sparsity_percent)
            margin_m = (m_sel - tau_m).abs()
            gap_cert_m = (m_sel - flat_oracle_m).abs()
            c_ok, c_den = certifiability_strict_fraction(gap_cert_m, margin_m)
            mag_margin_raw_hists[step].update(margin_m)
            mag_margin_w_raw[step].update(margin_m)
            gap_diag[f"cert_magnitude_step{step}_tau"] = tau_m
            gap_diag[f"cert_magnitude_step{step}_k_keep"] = k_m
            gap_diag[f"cert_magnitude_step{step}_N"] = N_m
            gap_diag[f"cert_strict_frac_magnitude_step{step}"] = (c_ok / c_den) if c_den else float("nan")
            gap_diag[f"cert_strict_numer_magnitude_step{step}"] = c_ok
            gap_diag[f"cert_strict_denom_magnitude_step{step}"] = c_den

        del mag_scores
        gc.collect()

    if delta_verify is not None:
        print(f"verify deltas max |oracle - |delta500||: {max_verify_diff:.6e} (worst key: {verify_key_worst!r})")

    with (out_dir / "mask_score_gap_gap_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(gap_diag, f, indent=2)

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
        return {"case": case, "total_numel": acc.n, "mean": acc.mean_value(), "std": acc.std_value(), **qdict}

    def pack_cert_row(case: str, numer: int, denom: int) -> Dict[str, object]:
        frac = (numer / denom) if denom else float("nan")
        qnan = {f"p{int(round(q * 100)):02d}": float("nan") for q in qs}
        return {"case": case, "total_numel": denom, "mean": frac, "std": 0.0, **qnan}

    for step in milestones:
        summary_rows.append(pack_row(f"magnitude_raw_step{step}", mag_w_raw[step], mag_raw_hists[step], None))
        summary_rows.append(pack_row(f"magnitude_norm_step{step}", mag_w_norm[step], None, mag_norm_hists[step]))
        summary_rows.append(
            pack_row(f"magnitude_margin_raw_step{step}", mag_margin_w_raw[step], mag_margin_raw_hists[step], None)
        )
        dn = int(gap_diag.get(f"cert_strict_denom_magnitude_step{step}", 0) or 0)
        nr = int(gap_diag.get(f"cert_strict_numer_magnitude_step{step}", 0) or 0)
        if dn > 0:
            summary_rows.append(pack_cert_row(f"cert_strict_magnitude_step{step}", nr, dn))
    if oracle_margin_w.n > 0:
        summary_rows.append(pack_row("oracle_margin_raw", oracle_margin_w, oracle_margin_hist, None))
    for seed in random_seeds:
        summary_rows.append(pack_row(f"random_raw_seed{seed}", rnd_w_raw[seed], rnd_raw_hists[seed], None))
        summary_rows.append(pack_row(f"random_norm_seed{seed}", rnd_w_norm[seed], None, rnd_norm_hists[seed]))
        summary_rows.append(
            pack_row(f"random_margin_raw_seed{seed}", rnd_margin_w_raw[seed], rnd_margin_raw_hists[seed], None)
        )
        dn = int(gap_diag.get(f"cert_strict_denom_random_seed{seed}", 0) or 0)
        nr = int(gap_diag.get(f"cert_strict_numer_random_seed{seed}", 0) or 0)
        if dn > 0:
            summary_rows.append(pack_cert_row(f"cert_strict_random_seed{seed}", nr, dn))

    normalized_rows = sorted(layer_rows, key=lambda r: str(r.get("param_name", "")))

    fn = sorted(summary_rows[0].keys()) if summary_rows else []
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=fn)
        wcsv.writeheader()
        for row in summary_rows:
            wcsv.writerow(row)

    if normalized_rows:
        lf = sorted(set().union(*(r.keys() for r in normalized_rows)))
        with by_layer_path.open("w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=lf)
            wcsv.writeheader()
            for row in normalized_rows:
                wcsv.writerow(row)

    npz_payload: Dict[str, np.ndarray] = {"hist_bins": np.array([histogram_bins], dtype=np.int64)}
    for step in milestones:
        npz_payload[f"magnitude_raw_step{step}_counts"] = mag_raw_hists[step].counts
        npz_payload[f"magnitude_raw_step{step}_log_edges"] = mag_raw_hists[step].edges
        npz_payload[f"magnitude_norm_step{step}_counts"] = mag_norm_hists[step].counts
        npz_payload[f"magnitude_norm_step{step}_edges"] = mag_norm_hists[step].edges
    for seed in random_seeds:
        npz_payload[f"random_raw_seed{seed}_counts"] = rnd_raw_hists[seed].counts
        npz_payload[f"random_raw_seed{seed}_log_edges"] = rnd_raw_hists[seed].edges
        npz_payload[f"random_norm_seed{seed}_counts"] = rnd_norm_hists[seed].counts
        npz_payload[f"random_norm_seed{seed}_edges"] = rnd_norm_hists[seed].edges
        npz_payload[f"random_margin_raw_seed{seed}_counts"] = rnd_margin_raw_hists[seed].counts
        npz_payload[f"random_margin_raw_seed{seed}_log_edges"] = rnd_margin_raw_hists[seed].edges
    for step in milestones:
        npz_payload[f"magnitude_margin_raw_step{step}_counts"] = mag_margin_raw_hists[step].counts
        npz_payload[f"magnitude_margin_raw_step{step}_log_edges"] = mag_margin_raw_hists[step].edges
    if oracle_margin_hist.counts.sum() > 0:
        npz_payload["oracle_margin_raw_counts"] = oracle_margin_hist.counts
        npz_payload["oracle_margin_raw_log_edges"] = oracle_margin_hist.edges

    np.savez(out_dir / "mask_score_gap_histograms.npz", **npz_payload)
    print(f"Wrote {summary_path}, {by_layer_path}, mask_score_gap_histograms.npz, mask_score_gap_gap_diagnostics.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mask score gaps vs oracle (warm magnitude milestones, random).")
    p.add_argument("--initial_model", type=str, required=True)
    p.add_argument("--final_model", type=str, required=True)
    p.add_argument("--delta_log_dir", type=str, required=True)
    p.add_argument(
        "--magnitude_milestones",
        type=str,
        default="50,100,150,200",
        help="Comma-separated snapshot endpoints for partial warm magnitude scores (sums |Δ_w| through each).",
    )
    p.add_argument("--mlp_only", action="store_true")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument(
        "--random_seeds",
        type=str,
        default=None,
        help="Comma-separated extra seeds (always includes --random_seed).",
    )
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--histogram_bins", type=int, default=2048)
    p.add_argument("--max_keys", type=int, default=None)
    p.add_argument("--verify_deltas_dir", type=str, default=None)
    p.add_argument(
        "--force_magnitude_cache_rebuild",
        action="store_true",
        help="Re-stream deltas and overwrite milestone caches under magnitude_caches/",
    )
    _env_sp = os.environ.get("CERT_SPARSITY_PERCENT") or os.environ.get("SPARSITY_PERCENT")
    _default_sparsity = float(_env_sp) if _env_sp else 97.5
    p.add_argument(
        "--sparsity_percent",
        type=float,
        default=_default_sparsity,
        help="Global sparsity rho (percent pruned): keep k = floor((100-rho)/100*N); tau from top-k boundary. "
        "Default from CERT_SPARSITY_PERCENT or SPARSITY_PERCENT env, else 97.5.",
    )
    p.add_argument(
        "--cert_min_layer_keep_ratio",
        type=float,
        default=float(os.environ.get("CERT_MIN_LAYER_KEEP_RATIO", "0")),
        help="Document-only for hybrid masks: τ/m_i stay pure-global; >0 logs a warning in gap diagnostics.",
    )
    p.add_argument(
        "--cert_no_match_tie_break",
        action="store_true",
        help="Use raw scores for τ and margins (no tie-break noise). Default matches mask_utils tie-break.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_hf_hub_token()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "magnitude_caches"

    milestones = sorted({int(x.strip()) for x in args.magnitude_milestones.split(",") if x.strip()})

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

    milestone_paths = build_magnitude_milestone_caches(
        args.delta_log_dir,
        milestones,
        keys_all,
        args.mlp_only,
        cache_dir,
        force_rebuild=args.force_magnitude_cache_rebuild,
    )

    keys = list(keys_all)
    if args.max_keys is not None:
        keys = keys[: max(0, args.max_keys)]

    delta_verify: Optional[Dict[str, torch.Tensor]] = None
    if args.verify_deltas_dir:
        cand = Path(args.verify_deltas_dir) / "deltas_step_500.pt"
        if not cand.is_file():
            raise FileNotFoundError(f"verify expected {cand}")
        print(f"Loading verification snapshot {cand}")
        delta_verify = safe_torch_load(str(cand), map_location="cpu")

    cert_match_tie_break = True
    if getattr(args, "cert_no_match_tie_break", False):
        cert_match_tie_break = False
    elif os.environ.get("CERT_MATCH_TIE_BREAK", "").strip().lower() in ("0", "false", "no"):
        cert_match_tie_break = False

    meta = {
        "initial_model": args.initial_model,
        "final_model": args.final_model,
        "delta_log_dir": args.delta_log_dir,
        "magnitude_milestones": milestones,
        "magnitude_cache_dir": str(cache_dir),
        "mlp_only": args.mlp_only,
        "random_seeds": seeds,
        "histogram_bins": args.histogram_bins,
        "num_keys": len(keys),
        "verify_deltas_dir": args.verify_deltas_dir,
        "git_rev": _git_rev(),
        "sparsity_percent": float(args.sparsity_percent),
        "cert_match_tie_break": cert_match_tie_break,
        "cert_min_layer_keep_ratio": float(args.cert_min_layer_keep_ratio),
        "cert_note": (
            "Margins use pure global τ on flattened scores (key order = analysis iteration order). "
            "Training hybrid masks with min_layer_keep_ratio>0 are not described by this single τ."
        ),
    }
    with (out_dir / "mask_score_gap_run.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Milestones {milestones}; intersecting keys: {len(keys)}")
    process_keys(
        keys=keys,
        initial_sd=initial_sd,
        final_sd=final_sd,
        milestone_cache_paths=milestone_paths,
        milestones=milestones,
        delta_verify=delta_verify,
        random_seeds=seeds,
        histogram_bins=args.histogram_bins,
        out_dir=out_dir,
        sparsity_percent=float(args.sparsity_percent),
        cert_match_tie_break=cert_match_tie_break,
        cert_min_layer_keep_ratio=float(args.cert_min_layer_keep_ratio),
    )


if __name__ == "__main__":
    main()
