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
from typing import Dict, List, Optional, Sequence, Tuple

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
    global_keep_count,
    global_topk_threshold,
    scaled_hybrid_floor_counts_per_layer,
    scores_for_cert_selection,
    streaming_global_topk_threshold,
    streaming_min_of_global_top_r,
    tau_hybrid_global_phase_from_flat,
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


def checkpoint_dtype_arg(s: str) -> torch.dtype:
    t = s.strip().lower()
    aliases = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "f32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if t not in aliases:
        raise argparse.ArgumentTypeError(
            f"Unknown checkpoint dtype {s!r}; use one of: {', '.join(sorted(aliases))}"
        )
    return aliases[t]


def _checkpoint_dtype_cli_default() -> str:
    env = os.environ.get("CHECKPOINT_DTYPE") or os.environ.get("MASK_GAP_CHECKPOINT_DTYPE")
    return env.strip().lower() if env else "bfloat16"


def _explain_dtype(dt: torch.dtype) -> str:
    if dt == torch.bfloat16:
        return "bfloat16 (lower RAM; small numerical deltas vs fp32)"
    if dt == torch.float16:
        return "float16 (lower RAM; may be numerically brittle on CPU)"
    return "float32"


def _cert_global_mode_default() -> str:
    v = (os.environ.get("MASK_GAP_CERT_GLOBAL_MODE") or os.environ.get("CERT_GLOBAL_MODE") or "").strip().lower()
    if v in ("materialize", "stream"):
        return v
    return "materialize"


def _cert_tau_rule_default() -> str:
    v = (os.environ.get("MASK_GAP_CERT_TAU_RULE") or os.environ.get("CERT_TAU_RULE") or "").strip().lower()
    if v in ("global", "hybrid_global_phase"):
        return v
    return "global"


def _cert_hybrid_min_layer_keep_ratio_default() -> float:
    env = os.environ.get("CERT_HYBRID_MIN_LAYER_KEEP_RATIO") or os.environ.get("MASK_GAP_CERT_HYBRID_MIN_LAYER_KEEP_RATIO")
    if env is not None and str(env).strip() != "":
        return float(env)
    return 0.0


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
    flatten_dtype: torch.dtype,
    sparsity_percent: float,
    cert_match_tie_break: bool,
    cert_min_layer_keep_ratio: float,
    cert_global_mode: str,
    cert_tau_rule: str,
    cert_hybrid_min_layer_keep_ratio: float,
) -> None:
    qs = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    milestones = sorted(milestones)
    use_stream = cert_global_mode == "stream"
    use_hybrid_tau = cert_tau_rule == "hybrid_global_phase"
    if use_hybrid_tau and float(cert_hybrid_min_layer_keep_ratio) <= 0.0:
        raise ValueError("cert_tau_rule=hybrid_global_phase requires --cert_hybrid_min_layer_keep_ratio > 0")

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
        "flatten_dtype_for_globals": str(flatten_dtype).replace("torch.", ""),
        "cert_global_mode": str(cert_global_mode),
        "cert_tau_rule": str(cert_tau_rule),
        "cert_hybrid_min_layer_keep_ratio": float(cert_hybrid_min_layer_keep_ratio),
    }
    if cert_min_layer_keep_ratio > 0 and not use_hybrid_tau:
        gap_diag["cert_hybrid_warning"] = (
            f"cert_min_layer_keep_ratio={cert_min_layer_keep_ratio} > 0 is metadata-only unless "
            "cert_tau_rule=hybrid_global_phase (use --cert_hybrid_min_layer_keep_ratio for τ)."
        )
    if use_hybrid_tau:
        gap_diag["cert_tau_note"] = (
            "hybrid_global_phase: τ is the global-phase cutoff after per-layer floors (mask_utils-style), "
            "not the pure-global Theorem-3 τ."
        )

    layer_rows: List[Dict[str, object]] = []
    max_verify_diff = 0.0
    verify_key_worst = ""

    oracle_eligible: List[str] = []
    oracle_numels: List[int] = []
    for _name in keys:
        if _name not in initial_sd or _name not in final_sd:
            continue
        w0n, w1n = initial_sd[_name], final_sd[_name]
        if not w0n.is_floating_point() or not w1n.is_floating_point():
            continue
        oracle_eligible.append(_name)
        oracle_numels.append(int(w0n.numel()))
    flat_oracle_n = int(sum(oracle_numels))
    oracle_keep = global_keep_count(flat_oracle_n, sparsity_percent) if flat_oracle_n > 0 else 0
    oracle_floors: List[int] = []
    oracle_floor_by_name: Dict[str, int] = {}
    if use_hybrid_tau and flat_oracle_n > 0:
        oracle_floors = scaled_hybrid_floor_counts_per_layer(
            oracle_numels, float(cert_hybrid_min_layer_keep_ratio), int(oracle_keep)
        )
        gap_diag["cert_oracle_hybrid_floor_total"] = int(sum(oracle_floors))
        gap_diag["cert_oracle_hybrid_R"] = int(oracle_keep - sum(oracle_floors))
        for _nm, _f in zip(oracle_eligible, oracle_floors):
            oracle_floor_by_name[_nm] = int(_f)

    oracle_offsets: List[Tuple[str, int, int]] = []

    if use_stream:
        oracle_flat_vec = torch.empty(0, dtype=flatten_dtype)
        rnd_flat_bufs: Dict[int, torch.Tensor] = {}
    else:
        oracle_flat_vec = torch.empty((flat_oracle_n,), dtype=flatten_dtype) if flat_oracle_n else torch.empty(0)
        rnd_flat_bufs = {s: torch.empty((flat_oracle_n,), dtype=flatten_dtype) for s in random_seeds}
    oracle_write_off = 0

    # ---------- random vs oracle (one pass over keys)
    print("Computing random-mask gaps vs oracle (one pass)...")
    for ki, name in enumerate(keys):
        if name not in initial_sd or name not in final_sd:
            continue
        w0 = initial_sd[name]
        w1 = final_sd[name]
        if not w0.is_floating_point() or not w1.is_floating_point():
            del w0, w1
            continue
        s_oracle = (w1 - w0).abs().to(dtype=flatten_dtype)
        nloc = int(s_oracle.numel())
        if not use_stream:
            oracle_offsets.append((name, oracle_write_off, oracle_write_off + nloc))
            oracle_flat_vec[oracle_write_off : oracle_write_off + nloc].copy_(s_oracle.reshape(-1))
        row_base: Dict[str, object] = {
            "param_name": name,
            "numel": int(s_oracle.numel()),
        }
        o_n_r = _minmax_norm(s_oracle)
        for seed in random_seeds:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            s_rand = torch.rand(s_oracle.shape, generator=g, dtype=flatten_dtype)
            if not use_stream:
                rnd_flat_bufs[seed][oracle_write_off : oracle_write_off + nloc].copy_(s_rand.reshape(-1))
            v_rr = (s_rand.to(torch.float32) - s_oracle.to(torch.float32)).abs()
            r_n = _minmax_norm(s_rand)
            v_rn = (r_n - o_n_r).abs()
            rnd_raw_hists[seed].update(v_rr)
            rnd_norm_hists[seed].update(v_rn)
            rnd_w_raw[seed].update(v_rr)
            rnd_w_norm[seed].update(v_rn)
            row_base[f"mean_v_rand_raw_seed{seed}"] = float(v_rr.mean().item())
            row_base[f"mean_v_rand_norm_seed{seed}"] = float(v_rn.mean().item())

        layer_rows.append(row_base)
        oracle_write_off += nloc
        del w0, w1, s_oracle, o_n_r
        gc.collect()

    if oracle_write_off != flat_oracle_n:
        raise RuntimeError(
            f"Oracle flatten size mismatch ({oracle_write_off} vs expected {flat_oracle_n}); key order/filtering bug."
        )

    if use_stream and flat_oracle_n > 0:
        print("Cert/margins: streaming global tau (no full-model flat buffers)...")

        max_abs_o = 0.0
        for name in keys:
            if name not in initial_sd or name not in final_sd:
                continue
            w0, w1 = initial_sd[name], final_sd[name]
            if not w0.is_floating_point() or not w1.is_floating_point():
                continue
            x = (w1 - w0).abs().to(dtype=flatten_dtype).reshape(-1).float()
            torch.nan_to_num_(x, nan=0.0, posinf=0.0, neginf=0.0)
            max_abs_o = max(max_abs_o, float(x.abs().max().item()))
        scale_o = max(max_abs_o * 1e-6, 1e-12) if cert_match_tie_break else 0.0

        def _oracle_sel_chunks_stream():
            gen_tb = torch.Generator(device="cpu")
            gen_tb.manual_seed(42)
            for name in keys:
                if name not in initial_sd or name not in final_sd:
                    continue
                w0, w1 = initial_sd[name], final_sd[name]
                if not w0.is_floating_point() or not w1.is_floating_point():
                    continue
                x = (w1 - w0).abs().to(dtype=flatten_dtype).reshape(-1).float().clone()
                torch.nan_to_num_(x, nan=0.0, posinf=0.0, neginf=0.0)
                if cert_match_tie_break:
                    noise = torch.randn(x.numel(), generator=gen_tb, dtype=torch.float32)
                    x = x + noise * scale_o
                yield x

        if use_hybrid_tau:
            floor_sum_o = int(sum(oracle_floors)) if oracle_floors else 0
            R_o = int(oracle_keep) - floor_sum_o
            if R_o <= 0 or flat_oracle_n <= 0:
                tau_o = float("nan")
                k_o, N_o = int(oracle_keep), int(flat_oracle_n)
            else:

                def _oracle_hybrid_pieces_inner():
                    gen_tb = torch.Generator(device="cpu")
                    gen_tb.manual_seed(42)
                    for name in keys:
                        if name not in initial_sd or name not in final_sd:
                            continue
                        w0, w1 = initial_sd[name], final_sd[name]
                        if not w0.is_floating_point() or not w1.is_floating_point():
                            continue
                        x = (w1 - w0).abs().to(dtype=flatten_dtype).reshape(-1).float().clone()
                        torch.nan_to_num_(x, nan=0.0, posinf=0.0, neginf=0.0)
                        if cert_match_tie_break:
                            noise = torch.randn(x.numel(), generator=gen_tb, dtype=torch.float32)
                            x = x + noise * scale_o
                        f = int(oracle_floor_by_name.get(name, 0))
                        if f > 0:
                            kk = min(f, int(x.numel()))
                            _, idx = torch.topk(x, kk, largest=True)
                            x[idx] = float("-inf")
                        yield x

                tau_o = streaming_min_of_global_top_r(_oracle_hybrid_pieces_inner(), R_o)
                k_o, N_o = int(oracle_keep), int(flat_oracle_n)
        else:
            tau_o, k_o, N_o = streaming_global_topk_threshold(
                total_n=flat_oracle_n,
                sparsity_percent=sparsity_percent,
                iter_sel_chunks_fp32=_oracle_sel_chunks_stream(),
            )
        gap_diag["cert_oracle_tau"] = tau_o
        gap_diag["cert_oracle_k_keep"] = k_o
        gap_diag["cert_oracle_N"] = N_o

        gen_tb2 = torch.Generator(device="cpu")
        gen_tb2.manual_seed(42)
        for name in keys:
            if name not in initial_sd or name not in final_sd:
                continue
            w0, w1 = initial_sd[name], final_sd[name]
            if not w0.is_floating_point() or not w1.is_floating_point():
                continue
            x = (w1 - w0).abs().to(dtype=flatten_dtype).reshape(-1).float().clone()
            torch.nan_to_num_(x, nan=0.0, posinf=0.0, neginf=0.0)
            if cert_match_tie_break:
                noise = torch.randn(x.numel(), generator=gen_tb2, dtype=torch.float32)
                x = x + noise * scale_o
            margin_o = (x - tau_o).abs()
            oracle_margin_hist.update(margin_o)
            oracle_margin_w.update(margin_o)

        for seed in random_seeds:
            max_abs_r = 0.0
            for name in keys:
                if name not in initial_sd or name not in final_sd:
                    continue
                w0, w1 = initial_sd[name], final_sd[name]
                if not w0.is_floating_point() or not w1.is_floating_point():
                    continue
                sh = w0.shape
                g0 = torch.Generator(device="cpu")
                g0.manual_seed(int(seed))
                raw = torch.rand(sh, generator=g0, dtype=flatten_dtype).reshape(-1).float()
                torch.nan_to_num_(raw, nan=0.0, posinf=0.0, neginf=0.0)
                max_abs_r = max(max_abs_r, float(raw.abs().max().item()))
            scale_r = max(max_abs_r * 1e-6, 1e-12) if cert_match_tie_break else 0.0

            def _rand_sel_chunks_stream():
                gen_tb = torch.Generator(device="cpu")
                gen_tb.manual_seed(42)
                for name in keys:
                    if name not in initial_sd or name not in final_sd:
                        continue
                    w0, w1 = initial_sd[name], final_sd[name]
                    if not w0.is_floating_point() or not w1.is_floating_point():
                        continue
                    sh = w0.shape
                    g0 = torch.Generator(device="cpu")
                    g0.manual_seed(int(seed))
                    raw = torch.rand(sh, generator=g0, dtype=flatten_dtype).reshape(-1).float().clone()
                    torch.nan_to_num_(raw, nan=0.0, posinf=0.0, neginf=0.0)
                    x = raw
                    if cert_match_tie_break:
                        noise = torch.randn(x.numel(), generator=gen_tb, dtype=torch.float32)
                        x = x + noise * scale_r
                    yield x

            if use_hybrid_tau:
                floor_sum_r = int(sum(oracle_floors)) if oracle_floors else 0
                R_r = int(oracle_keep) - floor_sum_r
                if R_r <= 0 or flat_oracle_n <= 0:
                    tau_r = float("nan")
                    k_r, N_r = int(oracle_keep), int(flat_oracle_n)
                else:

                    def _rand_hybrid_pieces():
                        gen_tb = torch.Generator(device="cpu")
                        gen_tb.manual_seed(42)
                        for name in keys:
                            if name not in initial_sd or name not in final_sd:
                                continue
                            w0, w1 = initial_sd[name], final_sd[name]
                            if not w0.is_floating_point() or not w1.is_floating_point():
                                continue
                            sh = w0.shape
                            g0 = torch.Generator(device="cpu")
                            g0.manual_seed(int(seed))
                            raw = torch.rand(sh, generator=g0, dtype=flatten_dtype).reshape(-1).float().clone()
                            torch.nan_to_num_(raw, nan=0.0, posinf=0.0, neginf=0.0)
                            x = raw
                            if cert_match_tie_break:
                                noise = torch.randn(x.numel(), generator=gen_tb, dtype=torch.float32)
                                x = x + noise * scale_r
                            f = int(oracle_floor_by_name.get(name, 0))
                            if f > 0:
                                kk = min(f, int(x.numel()))
                                _, idx = torch.topk(x, kk, largest=True)
                                x[idx] = float("-inf")
                            yield x

                    tau_r = streaming_min_of_global_top_r(_rand_hybrid_pieces(), R_r)
                    k_r, N_r = int(oracle_keep), int(flat_oracle_n)
            else:
                tau_r, k_r, N_r = streaming_global_topk_threshold(
                    total_n=flat_oracle_n,
                    sparsity_percent=sparsity_percent,
                    iter_sel_chunks_fp32=_rand_sel_chunks_stream(),
                )

            gen_tb3 = torch.Generator(device="cpu")
            gen_tb3.manual_seed(42)
            c_ok = 0
            c_den = 0
            for name in keys:
                if name not in initial_sd or name not in final_sd:
                    continue
                w0, w1 = initial_sd[name], final_sd[name]
                if not w0.is_floating_point() or not w1.is_floating_point():
                    continue
                sh = w0.shape
                g0 = torch.Generator(device="cpu")
                g0.manual_seed(int(seed))
                raw = torch.rand(sh, generator=g0, dtype=flatten_dtype).reshape(-1).float().clone()
                torch.nan_to_num_(raw, nan=0.0, posinf=0.0, neginf=0.0)
                r_sel = raw
                if cert_match_tie_break:
                    noise = torch.randn(r_sel.numel(), generator=gen_tb3, dtype=torch.float32)
                    r_sel = r_sel + noise * scale_r
                s_star = (w1 - w0).abs().to(dtype=flatten_dtype).reshape(-1).float()
                torch.nan_to_num_(s_star, nan=0.0, posinf=0.0, neginf=0.0)
                gap = (r_sel - s_star).abs()
                margin = (r_sel - tau_r).abs()
                c_ok += int((gap < margin).sum().item())
                c_den += int(gap.numel())
                rnd_margin_raw_hists[seed].update(margin)
                rnd_margin_w_raw[seed].update(margin)

            gap_diag[f"cert_random_seed{seed}_tau"] = tau_r
            gap_diag[f"cert_random_seed{seed}_k_keep"] = k_r
            gap_diag[f"cert_random_seed{seed}_N"] = N_r
            gap_diag[f"cert_strict_frac_random_seed{seed}"] = (c_ok / c_den) if c_den else float("nan")
            gap_diag[f"cert_strict_numer_random_seed{seed}"] = c_ok
            gap_diag[f"cert_strict_denom_random_seed{seed}"] = c_den

    elif not use_stream and oracle_flat_vec.numel() > 0:
        o_sel = scores_for_cert_selection(oracle_flat_vec, match_tie_break=cert_match_tie_break)
        if use_hybrid_tau:
            tau_o, k_o, N_o, _ft_o, _rr_o = tau_hybrid_global_phase_from_flat(
                o_sel, oracle_numels, oracle_floors, int(oracle_keep)
            )
        else:
            tau_o, k_o, N_o = global_topk_threshold(o_sel, sparsity_percent)
        margin_o = (o_sel - tau_o).abs()
        oracle_margin_hist.update(margin_o)
        oracle_margin_w.update(margin_o)
        gap_diag["cert_oracle_tau"] = tau_o
        gap_diag["cert_oracle_k_keep"] = k_o
        gap_diag["cert_oracle_N"] = N_o
        for seed in random_seeds:
            flat_r = rnd_flat_bufs.get(seed)
            if flat_r is None or flat_r.numel() != oracle_flat_vec.numel():
                continue
            r_sel = scores_for_cert_selection(flat_r, match_tie_break=cert_match_tie_break)
            if use_hybrid_tau:
                tau_r, k_r, N_r, _ft_r, _rr_r = tau_hybrid_global_phase_from_flat(
                    r_sel, oracle_numels, oracle_floors, int(oracle_keep)
                )
            else:
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

    if not use_stream:
        del rnd_flat_bufs
    gc.collect()

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

        mag_cert_n = 0
        mag_eligible: List[str] = []
        mag_numels: List[int] = []
        for nm in keys:
            if nm not in initial_sd or nm not in final_sd or nm not in mag_scores:
                continue
            om = mag_scores[nm].shape
            oo = initial_sd[nm].shape
            of = final_sd[nm].shape
            if om == oo == of:
                n_mag_layer = int(math.prod(tuple(int(x) for x in om)))
                mag_cert_n += n_mag_layer
                mag_eligible.append(nm)
                mag_numels.append(n_mag_layer)
        if use_stream:
            flat_mag_flat = torch.empty(0, dtype=flatten_dtype)
            flat_oracle_flat_m = torch.empty(0, dtype=flatten_dtype)
        else:
            flat_mag_flat = torch.empty((mag_cert_n,), dtype=flatten_dtype) if mag_cert_n > 0 else torch.empty(0)
            flat_oracle_flat_m = torch.empty((mag_cert_n,), dtype=flatten_dtype) if mag_cert_n > 0 else torch.empty(0)
        mag_cert_off = 0

        for ki, name in enumerate(keys):
            if name not in initial_sd or name not in final_sd or name not in mag_scores:
                continue
            w0 = initial_sd[name]
            w1 = final_sd[name]
            if not w0.is_floating_point() or not w1.is_floating_point():
                del w0, w1
                continue
            s_oracle = (w1 - w0).abs()
            if s_oracle.is_floating_point():
                s_oracle = s_oracle.to(dtype=flatten_dtype)
            s_mag = mag_scores[name]
            if s_mag.is_floating_point():
                s_mag = s_mag.to(dtype=flatten_dtype)
            if s_mag.shape != s_oracle.shape:
                del w0, w1, s_oracle, s_mag
                continue

            nloc_mag = int(s_mag.numel())
            if not use_stream:
                flat_mag_flat[mag_cert_off : mag_cert_off + nloc_mag].copy_(s_mag.reshape(-1))
                flat_oracle_flat_m[mag_cert_off : mag_cert_off + nloc_mag].copy_(s_oracle.reshape(-1))
            mag_cert_off += nloc_mag

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

        if mag_cert_off != mag_cert_n:
            raise RuntimeError(
                f"Magnitude cert flatten mismatch at step {step}: wrote {mag_cert_off} vs expected {mag_cert_n}"
            )

        mag_keep = global_keep_count(mag_cert_n, sparsity_percent) if mag_cert_n > 0 else 0
        mag_floors: List[int] = []
        mag_floor_by_name: Dict[str, int] = {}
        if use_hybrid_tau and mag_cert_n > 0:
            mag_floors = scaled_hybrid_floor_counts_per_layer(
                mag_numels, float(cert_hybrid_min_layer_keep_ratio), int(mag_keep)
            )
            gap_diag[f"cert_magnitude_step{step}_hybrid_floor_total"] = int(sum(mag_floors))
            gap_diag[f"cert_magnitude_step{step}_hybrid_R"] = int(mag_keep - sum(mag_floors))
            for _nm, _f in zip(mag_eligible, mag_floors):
                mag_floor_by_name[_nm] = int(_f)

        if use_stream and mag_cert_n > 0:
            max_abs_m = 0.0
            for nm in keys:
                if nm not in initial_sd or nm not in final_sd or nm not in mag_scores:
                    continue
                w0m, w1m = initial_sd[nm], final_sd[nm]
                if not w0m.is_floating_point() or not w1m.is_floating_point():
                    continue
                sm = mag_scores[nm]
                om_sh = sm.shape
                oo_sh = w0m.shape
                of_sh = w1m.shape
                if om_sh != oo_sh or om_sh != of_sh:
                    continue
                xm = sm.to(dtype=flatten_dtype).reshape(-1).float()
                torch.nan_to_num_(xm, nan=0.0, posinf=0.0, neginf=0.0)
                max_abs_m = max(max_abs_m, float(xm.abs().max().item()))
            scale_m = max(max_abs_m * 1e-6, 1e-12) if cert_match_tie_break else 0.0

            def _mag_sel_chunks_stream():
                gen_tb = torch.Generator(device="cpu")
                gen_tb.manual_seed(42)
                for nm in keys:
                    if nm not in initial_sd or nm not in final_sd or nm not in mag_scores:
                        continue
                    w0m, w1m = initial_sd[nm], final_sd[nm]
                    if not w0m.is_floating_point() or not w1m.is_floating_point():
                        continue
                    sm = mag_scores[nm]
                    som = (w1m - w0m).abs()
                    if sm.shape != som.shape:
                        continue
                    x = sm.to(dtype=flatten_dtype).reshape(-1).float().clone()
                    torch.nan_to_num_(x, nan=0.0, posinf=0.0, neginf=0.0)
                    if cert_match_tie_break:
                        noise = torch.randn(x.numel(), generator=gen_tb, dtype=torch.float32)
                        x = x + noise * scale_m
                    yield x

            if use_hybrid_tau:
                floor_sum_m = int(sum(mag_floors)) if mag_floors else 0
                R_m = int(mag_keep) - floor_sum_m
                if R_m <= 0 or mag_cert_n <= 0:
                    tau_m = float("nan")
                    k_m, N_m = int(mag_keep), int(mag_cert_n)
                else:

                    def _mag_hybrid_pieces():
                        gen_tb = torch.Generator(device="cpu")
                        gen_tb.manual_seed(42)
                        for nm in keys:
                            if nm not in initial_sd or nm not in final_sd or nm not in mag_scores:
                                continue
                            w0m, w1m = initial_sd[nm], final_sd[nm]
                            if not w0m.is_floating_point() or not w1m.is_floating_point():
                                continue
                            sm = mag_scores[nm]
                            som = (w1m - w0m).abs()
                            if sm.shape != som.shape:
                                continue
                            x = sm.to(dtype=flatten_dtype).reshape(-1).float().clone()
                            torch.nan_to_num_(x, nan=0.0, posinf=0.0, neginf=0.0)
                            if cert_match_tie_break:
                                noise = torch.randn(x.numel(), generator=gen_tb, dtype=torch.float32)
                                x = x + noise * scale_m
                            f = int(mag_floor_by_name.get(nm, 0))
                            if f > 0:
                                kk = min(f, int(x.numel()))
                                _, idx = torch.topk(x, kk, largest=True)
                                x[idx] = float("-inf")
                            yield x

                    tau_m = streaming_min_of_global_top_r(_mag_hybrid_pieces(), R_m)
                    k_m, N_m = int(mag_keep), int(mag_cert_n)
            else:
                tau_m, k_m, N_m = streaming_global_topk_threshold(
                    total_n=mag_cert_n,
                    sparsity_percent=sparsity_percent,
                    iter_sel_chunks_fp32=_mag_sel_chunks_stream(),
                )

            gen_tb4 = torch.Generator(device="cpu")
            gen_tb4.manual_seed(42)
            c_ok_m = 0
            c_den_m = 0
            for nm in keys:
                if nm not in initial_sd or nm not in final_sd or nm not in mag_scores:
                    continue
                w0m, w1m = initial_sd[nm], final_sd[nm]
                if not w0m.is_floating_point() or not w1m.is_floating_point():
                    continue
                sm = mag_scores[nm]
                som = (w1m - w0m).abs().to(dtype=flatten_dtype)
                if sm.shape != som.shape:
                    continue
                m_sel = sm.to(dtype=flatten_dtype).reshape(-1).float().clone()
                torch.nan_to_num_(m_sel, nan=0.0, posinf=0.0, neginf=0.0)
                if cert_match_tie_break:
                    noise = torch.randn(m_sel.numel(), generator=gen_tb4, dtype=torch.float32)
                    m_sel = m_sel + noise * scale_m
                s_star_m = som.reshape(-1).float()
                torch.nan_to_num_(s_star_m, nan=0.0, posinf=0.0, neginf=0.0)
                gap_m = (m_sel - s_star_m).abs()
                margin_m = (m_sel - tau_m).abs()
                c_ok_m += int((gap_m < margin_m).sum().item())
                c_den_m += int(gap_m.numel())
                mag_margin_raw_hists[step].update(margin_m)
                mag_margin_w_raw[step].update(margin_m)

            gap_diag[f"cert_magnitude_step{step}_tau"] = tau_m
            gap_diag[f"cert_magnitude_step{step}_k_keep"] = k_m
            gap_diag[f"cert_magnitude_step{step}_N"] = N_m
            gap_diag[f"cert_strict_frac_magnitude_step{step}"] = (c_ok_m / c_den_m) if c_den_m else float("nan")
            gap_diag[f"cert_strict_numer_magnitude_step{step}"] = c_ok_m
            gap_diag[f"cert_strict_denom_magnitude_step{step}"] = c_den_m

        elif not use_stream and mag_cert_n > 0 and flat_mag_flat.numel() > 0:
            flat_mag = flat_mag_flat
            flat_oracle_m = flat_oracle_flat_m
            m_sel = scores_for_cert_selection(flat_mag, match_tie_break=cert_match_tie_break)
            if use_hybrid_tau:
                tau_m, k_m, N_m, _ft_m, _rr_m = tau_hybrid_global_phase_from_flat(
                    m_sel, mag_numels, mag_floors, int(mag_keep)
                )
            else:
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
        help="Training-mask metadata (logged in gap diagnostics). For hybrid τ, use --cert_hybrid_min_layer_keep_ratio.",
    )
    p.add_argument(
        "--cert_no_match_tie_break",
        action="store_true",
        help="Use raw scores for τ and margins (no tie-break noise). Default matches mask_utils tie-break.",
    )
    p.add_argument(
        "--checkpoint_dtype",
        type=checkpoint_dtype_arg,
        default=checkpoint_dtype_arg(_checkpoint_dtype_cli_default()),
        help=(
            "Storage dtype for checkpoint weights loaded into RAM (hub + local shards). "
            "Default: MASK_GAP_CHECKPOINT_DTYPE or CHECKPOINT_DTYPE env, else bfloat16. "
            "fp16/bf16 cut memory (~2×) versus fp32; intermediate ops still promote as needed."
        ),
    )
    p.add_argument(
        "--cert_global_mode",
        type=str,
        choices=("materialize", "stream"),
        default=_cert_global_mode_default(),
        help=(
            "How to compute global tau/margins/cert: stream avoids full-model flat tensors (lower RAM). "
            "Default from MASK_GAP_CERT_GLOBAL_MODE or CERT_GLOBAL_MODE env, else materialize."
        ),
    )
    p.add_argument(
        "--cert_tau_rule",
        type=str,
        choices=("global", "hybrid_global_phase"),
        default=_cert_tau_rule_default(),
        help=(
            "global: Theorem-3 style τ = min(top-k) on flattened selection scores. "
            "hybrid_global_phase: per-layer floor top-k then global top-R on remainder (mask_utils flat hybrid). "
            "Default from MASK_GAP_CERT_TAU_RULE or CERT_TAU_RULE env, else global."
        ),
    )
    p.add_argument(
        "--cert_hybrid_min_layer_keep_ratio",
        type=float,
        default=_cert_hybrid_min_layer_keep_ratio_default(),
        help=(
            "When --cert_tau_rule=hybrid_global_phase, per-layer keep floor as a fraction of layer size "
            "(same scaling as mask_utils when floors exceed budget). "
            "Default from CERT_HYBRID_MIN_LAYER_KEEP_RATIO env, else 0 (invalid for hybrid; must be >0)."
        ),
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

    ckpt_dt = getattr(args, "checkpoint_dtype", torch.bfloat16)
    print(f"Loading initial / final state dicts on CPU ({_explain_dtype(ckpt_dt)})...")
    initial_sd = load_state_dict(args.initial_model, device="cpu", torch_dtype=ckpt_dt)
    final_sd = load_state_dict(args.final_model, device="cpu", torch_dtype=ckpt_dt)

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

    if str(getattr(args, "cert_tau_rule", "global")) == "hybrid_global_phase":
        if float(getattr(args, "cert_hybrid_min_layer_keep_ratio", 0.0)) <= 0.0:
            raise ValueError(
                "--cert_tau_rule=hybrid_global_phase requires --cert_hybrid_min_layer_keep_ratio > 0 "
                "(or set CERT_HYBRID_MIN_LAYER_KEEP_RATIO)."
            )

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
        "checkpoint_dtype": str(ckpt_dt).replace("torch.", ""),
        "sparsity_percent": float(args.sparsity_percent),
        "cert_match_tie_break": cert_match_tie_break,
        "cert_min_layer_keep_ratio": float(args.cert_min_layer_keep_ratio),
        "cert_global_mode": str(getattr(args, "cert_global_mode", "materialize")),
        "cert_tau_rule": str(getattr(args, "cert_tau_rule", "global")),
        "cert_hybrid_min_layer_keep_ratio": float(getattr(args, "cert_hybrid_min_layer_keep_ratio", 0.0)),
        "cert_note": (
            "Margins use τ from cert_tau_rule (global = pure top-k on flattened selection scores; "
            "hybrid_global_phase = floors then global top-R). Key order = analysis iteration order."
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
        flatten_dtype=ckpt_dt,
        sparsity_percent=float(args.sparsity_percent),
        cert_match_tie_break=cert_match_tie_break,
        cert_min_layer_keep_ratio=float(args.cert_min_layer_keep_ratio),
        cert_global_mode=str(getattr(args, "cert_global_mode", "materialize")),
        cert_tau_rule=str(getattr(args, "cert_tau_rule", "global")),
        cert_hybrid_min_layer_keep_ratio=float(getattr(args, "cert_hybrid_min_layer_keep_ratio", 0.0)),
    )


if __name__ == "__main__":
    main()
