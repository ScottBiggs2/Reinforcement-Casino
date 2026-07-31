#!/usr/bin/env python3
r"""Certifiability margins (Eq. 7) for warm-start magnitude scores against the oracle.

Measures, per weight coordinate i, over the *same* coverage the fixed mask builder scores:

    delta_i = | s_warm(theta^{<=k}) - s_oracle(theta^T, theta^0) |     (score gap)
    m_i     = | s_i - tau_hat_rho(s) |                                 (certifiability margin)

for k in {50,100,150,200}, an oracle arm, and a random arm, at one or more sparsities rho.

Three things differ from ``mask_score_gap_analysis.py``, which this supersedes for Figure 5:

1. **Coverage.** Scores are restricted to ``_select_2d_weight_names`` — exactly the 2-D ``*.weight``
   tensors the corrected builder scores (254 on Qwen3-8B, 226 on Olmo3-7B). The old script scored all
   parameters including 145/129 one-dimensional norm weights, so its ``tau_hat_rho`` was computed
   over a different N than the mask it described, and the arms were not mutually comparable.

2. **Tie-break amplitude.** ``max|s| * 1e-12`` with no absolute floor, drawn per tensor from a
   name-derived seed, instead of ``max(max|s| * 1e-6, 1e-12)`` off a single shared stream. The old
   amplitude randomized the bottom of the ranking rather than breaking ties (RESULTS.md §3.1), and
   the shared stream made the random arm non-independent of the warm arms (§3.4).

3. **Scale.** The warm score at k is ``sum_{j<=k} |theta^j - theta^0|``, so its scale grows ~linearly
   in k while the oracle is a single difference. Raw margins therefore shrink toward the oracle as k
   grows *by construction*. Reported margins are tau-relative,

       m~_i = |s_i - tau_hat_rho(s)| / tau_hat_rho(s),

   which is rank-preserving (the mask is unchanged) and puts every arm's selection boundary at 1.0.
   Coordinates with no resolvable displacement (s_i = 0) land at exactly 1.0, so the mass at 1.0 is
   the share of the mask that carries no signal. Raw-space histograms are emitted alongside.

Degeneracy is a real possibility here and is detected, not hidden: at rho=97.5% the keep budget is
2.5% of ~7e9 coordinates while only ~1% have resolvable |dtheta| in bf16 at lr=5e-7, in which case
the true tau_rho is exactly 0 and no tau-relative margin exists. See ``tau_degenerate`` in the
diagnostics, and use several ``--sparsity_percents`` to find a regime where the margin is defined.

Stages (``--execution_mode``), so the expensive pieces are done once and in parallel:

    cache_only -> tau -> arm (array, one task per arm) -> merge

Slurm: ``scripts/slurm_certifiability_margins.slurm`` (driver:
``scripts/submit_certifiability_margins.sh``).
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.certifiability_margin import (  # noqa: E402
    add_tie_break_noise_,
    global_keep_count,
    scaled_hybrid_floor_counts_per_layer,
    streaming_exact_kth_largest,
    tie_break_scale,
)
from src.analysis.mask_score_gap_analysis import build_magnitude_milestone_caches  # noqa: E402
from src.warm_start.checkpoint_diff_mask_finder import load_state_dict  # noqa: E402
from src.warm_start.even_better_mask_finder import (  # noqa: E402
    _select_2d_weight_names,
    is_mlp_param,
)

SHARD_VERSION = 2
DEFAULT_MILESTONES = "50,100,150,200"
DEFAULT_SPARSITIES = "97.5"
# Production hybrid floor (mask_utils.DEFAULT_MIN_LAYER_KEEP_RATIO, and the Olmo3 mask slurm scripts).
DEFAULT_HYBRID_MIN_LAYER_KEEP_RATIO = 0.0025
DEFAULT_TIE_BREAK_RELATIVE_SCALE = 1e-12


# ------------------------------------------------------------------ small utilities


def _git_rev() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _dtype_arg(s: str) -> torch.dtype:
    aliases = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "f32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    t = s.strip().lower()
    if t not in aliases:
        raise argparse.ArgumentTypeError(f"Unknown dtype {s!r}; use one of {sorted(aliases)}")
    return aliases[t]


def stream_seed(*parts: object) -> int:
    """Deterministic per-tensor seed from a label, so every pass regenerates the same draw and
    distinct purposes (tie-break vs random mask) never share an RNG stream."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") & 0x7FFF_FFFF_FFFF_FFFF


def guard_not_under_home(path: Path, *, label: str, allow: bool) -> None:
    """Refuse to write bulk artifacts under $HOME.

    Explorer's ``/home/biggs.s`` is at 66 G and a single magnitude cache is ~30 G; filling it wedges
    the account. Cheap to check, expensive to get wrong, so it is a hard failure by default.
    """
    if allow:
        return
    try:
        home = Path.home().resolve()
    except (RuntimeError, OSError):
        return
    p = path.expanduser().resolve()
    if p == home or home in p.parents:
        raise SystemExit(
            f"REFUSING: {label} resolves under $HOME ({p}).\n"
            "  This pipeline writes tens of GB of magnitude caches; put it on scratch, e.g.\n"
            "    /scratch/$USER/rl_casino_analysis/certifiability_margins/<cell>\n"
            "  Override with --allow_home_out_dir only if you are certain."
        )


def safe_torch_load(path: str | Path, map_location: str = "cpu"):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except (TypeError, RuntimeError):
        return torch.load(str(path), map_location=map_location)


# ------------------------------------------------------------------ accumulators


@dataclass
class Moments:
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    n_zero: int = 0

    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        xf = x.reshape(-1).double()
        self.n += int(xf.numel())
        self.total += float(xf.sum().item())
        self.total_sq += float((xf * xf).sum().item())
        self.n_zero += int((xf == 0).sum().item())

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else float("nan")

    @property
    def std(self) -> float:
        if not self.n:
            return float("nan")
        var = self.total_sq / self.n - self.mean**2
        return float(math.sqrt(max(var, 0.0)))

    def to_dict(self) -> Dict[str, float]:
        return {"n": float(self.n), "total": self.total, "total_sq": self.total_sq, "n_zero": float(self.n_zero)}

    @staticmethod
    def from_dict(d: Dict[str, float]) -> "Moments":
        return Moments(n=int(d["n"]), total=float(d["total"]), total_sq=float(d["total_sq"]), n_zero=int(d.get("n_zero", 0)))


@dataclass
class LogHist:
    """Fixed log10 bins plus an explicit underflow bin, so exact zeros are counted rather than
    smeared by an adaptive epsilon (the old LogSpaceHistogram chose eps per chunk, which made bin
    positions depend on chunk composition)."""

    num_bins: int = 2000
    log_min: float = -12.0
    log_max: float = 8.0
    counts: np.ndarray = field(default=None)  # type: ignore[assignment]
    n_underflow: int = 0
    n_overflow: int = 0

    def __post_init__(self) -> None:
        if self.counts is None:
            self.counts = np.zeros(self.num_bins, dtype=np.int64)

    @property
    def edges(self) -> np.ndarray:
        return np.linspace(self.log_min, self.log_max, self.num_bins + 1)

    def update(self, v: torch.Tensor, chunk: int = 33_554_432) -> None:
        if v.numel() == 0:
            return
        flat = v.reshape(-1)
        edges = torch.linspace(self.log_min, self.log_max, self.num_bins + 1, dtype=torch.float64)
        for start in range(0, int(flat.numel()), chunk):
            part = flat[start : start + chunk].double()
            part = part[torch.isfinite(part)]
            if part.numel() == 0:
                continue
            lo_mask = part <= 0
            self.n_underflow += int(lo_mask.sum().item())
            pos = part[~lo_mask]
            if pos.numel() == 0:
                continue
            lx = torch.log10(pos)
            under = lx < self.log_min
            over = lx > self.log_max
            self.n_underflow += int(under.sum().item())
            self.n_overflow += int(over.sum().item())
            mid = lx[(~under) & (~over)]
            if mid.numel() == 0:
                continue
            idx = torch.bucketize(mid, edges, right=False) - 1
            idx.clamp_(0, self.num_bins - 1)
            self.counts += np.bincount(idx.numpy(), minlength=self.num_bins).astype(np.int64)

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_bins": int(self.num_bins),
            "log_min": float(self.log_min),
            "log_max": float(self.log_max),
            "counts": self.counts.copy(),
            "n_underflow": int(self.n_underflow),
            "n_overflow": int(self.n_overflow),
        }

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "LogHist":
        h = LogHist(num_bins=int(d["num_bins"]), log_min=float(d["log_min"]), log_max=float(d["log_max"]))
        h.counts = np.asarray(d["counts"], dtype=np.int64).copy()
        h.n_underflow = int(d["n_underflow"])
        h.n_overflow = int(d["n_overflow"])
        return h

    def merge_(self, other: "LogHist") -> None:
        if (self.num_bins, self.log_min, self.log_max) != (other.num_bins, other.log_min, other.log_max):
            raise ValueError("LogHist geometry mismatch")
        self.counts += other.counts
        self.n_underflow += other.n_underflow
        self.n_overflow += other.n_overflow

    def quantiles(self, qs: Sequence[float]) -> Dict[str, float]:
        total = float(self.counts.sum() + self.n_underflow + self.n_overflow)
        out: Dict[str, float] = {}
        if total <= 0:
            return {f"p{int(round(q * 100)):02d}": float("nan") for q in qs}
        edges = self.edges
        cum = np.concatenate([[float(self.n_underflow)], float(self.n_underflow) + np.cumsum(self.counts.astype(np.float64))])
        for q in qs:
            target = q * total
            key = f"p{int(round(q * 100)):02d}"
            if target <= self.n_underflow:
                out[key] = 0.0
                continue
            j = int(np.searchsorted(cum, target, side="left")) - 1
            j = min(max(j, 0), self.num_bins - 1)
            span = float(self.counts[j])
            frac = float(np.clip((target - cum[j]) / span, 0.0, 1.0)) if span > 0 else 0.0
            out[key] = float(10 ** (edges[j] + frac * (edges[j + 1] - edges[j])))
        return out


# ------------------------------------------------------------------ score providers


@dataclass
class ScoreContext:
    names: List[str]
    initial_sd: Dict[str, torch.Tensor]
    final_sd: Dict[str, torch.Tensor]
    mag_scores: Optional[Dict[str, torch.Tensor]] = None
    random_seed: int = 42
    work_dtype: torch.dtype = torch.float64


def oracle_score(ctx: ScoreContext, name: str) -> torch.Tensor:
    """s*_i = |theta^T_i - theta^0_i|, the checkpoint-diff oracle."""
    w0 = ctx.initial_sd[name]
    w1 = ctx.final_sd[name]
    return (w1.to(ctx.work_dtype) - w0.to(ctx.work_dtype)).abs().reshape(-1)


def warm_score(ctx: ScoreContext, name: str) -> torch.Tensor:
    """s_i = sum_{j<=k} |theta^j_i - theta^0_i| — production semantics
    (``compute_absolute_magnitude_mask_streaming``), read from the milestone cache."""
    assert ctx.mag_scores is not None
    return ctx.mag_scores[name].to(ctx.work_dtype).reshape(-1)


def random_score(ctx: ScoreContext, name: str) -> torch.Tensor:
    """Uniform(0,1) per coordinate. Seeded per *tensor name*, so the stream is independent of the
    tie-break streams (the old builder reused one global seed-42 stream for both — RESULTS.md §3.4)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(stream_seed("random_mask", ctx.random_seed, name))
    shape = ctx.initial_sd[name].shape
    return torch.rand(shape, generator=g, dtype=torch.float32).to(ctx.work_dtype).reshape(-1)


SCORE_FNS: Dict[str, Callable[[ScoreContext, str], torch.Tensor]] = {
    "oracle": oracle_score,
    "warm": warm_score,
    "random": random_score,
}


@dataclass
class ArmSpec:
    name: str  # e.g. "oracle", "warm_k50", "random_seed42"
    kind: str  # "oracle" | "warm" | "random"
    milestone: Optional[int] = None
    seed: Optional[int] = None

    @property
    def label(self) -> str:
        return self.name


def build_arm_specs(milestones: Sequence[int], seeds: Sequence[int]) -> List[ArmSpec]:
    arms = [ArmSpec(name="oracle", kind="oracle")]
    arms += [ArmSpec(name=f"warm_k{k}", kind="warm", milestone=int(k)) for k in sorted(milestones)]
    arms += [ArmSpec(name=f"random_seed{s}", kind="random", seed=int(s)) for s in sorted(seeds)]
    return arms


# ------------------------------------------------------------------ selection scores + tau


def _raw_chunks(ctx: ScoreContext, kind: str) -> Iterator[torch.Tensor]:
    fn = SCORE_FNS[kind]
    for name in ctx.names:
        yield fn(ctx, name)


def arm_max_abs_and_support(ctx: ScoreContext, kind: str) -> Tuple[float, int, int]:
    """(max|s|, count of s>0, N) in one pass — the tie-break amplitude and the degeneracy test."""
    max_abs = 0.0
    n_pos = 0
    n_tot = 0
    for x in _raw_chunks(ctx, kind):
        torch.nan_to_num_(x, nan=0.0, posinf=0.0, neginf=0.0)
        n_tot += int(x.numel())
        n_pos += int((x > 0).sum().item())
        m = float(x.abs().max().item()) if x.numel() else 0.0
        max_abs = max(max_abs, m)
        del x
    return max_abs, n_pos, n_tot


def selection_chunks(
    ctx: ScoreContext,
    kind: str,
    *,
    tie_break_scale_abs: float,
    floors_by_name: Optional[Dict[str, int]] = None,
) -> Iterator[torch.Tensor]:
    """Selection scores: raw + tie-break perturbation, with hybrid per-layer floors marked -inf.

    Regenerated identically on every pass because each tensor's noise seed is derived from its name.
    """
    fn = SCORE_FNS[kind]
    for name in ctx.names:
        x = fn(ctx, name)
        torch.nan_to_num_(x, nan=0.0, posinf=0.0, neginf=0.0)
        if tie_break_scale_abs > 0.0:
            g = torch.Generator(device="cpu")
            g.manual_seed(stream_seed("tie_break", kind, ctx.random_seed if kind == "random" else 0, name))
            add_tie_break_noise_(x, scale=tie_break_scale_abs, generator=g)
        if floors_by_name:
            f = int(floors_by_name.get(name, 0))
            if f > 0:
                kk = min(f, int(x.numel()))
                _, idx = torch.topk(x, kk, largest=True)
                x[idx] = float("-inf")
        yield x


@dataclass
class TauResult:
    arm: str
    sparsity_percent: float
    tau: float
    keep_count: int
    n_total: int
    n_raw_positive: int
    tau_degenerate: bool
    floor_total: int
    r_remaining: int
    max_abs: float
    tie_break_scale: float
    tau_rule: str
    info: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        d = dict(self.__dict__)
        d["info"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.info.items()}
        return d


def compute_tau(
    ctx: ScoreContext,
    arm: ArmSpec,
    *,
    sparsity_percent: float,
    tau_rule: str,
    hybrid_min_layer_keep_ratio: float,
    tie_break_relative_scale: float,
    max_abs: float,
    n_raw_positive: int,
    n_total: int,
    kth_num_bins: int,
) -> TauResult:
    keep = global_keep_count(n_total, sparsity_percent) if n_total else 0
    scale = tie_break_scale(max_abs, tie_break_relative_scale)

    floors_by_name: Dict[str, int] = {}
    floor_total = 0
    if tau_rule == "hybrid_global_phase":
        numels = [int(ctx.initial_sd[n].numel()) for n in ctx.names]
        floors = scaled_hybrid_floor_counts_per_layer(numels, hybrid_min_layer_keep_ratio, keep)
        floors_by_name = {n: int(f) for n, f in zip(ctx.names, floors)}
        floor_total = int(sum(floors))
    r_remaining = int(keep - floor_total)

    # Degeneracy is a property of the *raw* scores: if fewer coordinates carry signal than the keep
    # budget demands, the true boundary is 0 and only the tie-break decides the remainder.
    degenerate = bool(n_raw_positive < keep)

    if r_remaining <= 0 or n_total <= 0:
        return TauResult(
            arm=arm.name,
            sparsity_percent=sparsity_percent,
            tau=float("nan"),
            keep_count=int(keep),
            n_total=int(n_total),
            n_raw_positive=int(n_raw_positive),
            tau_degenerate=degenerate,
            floor_total=floor_total,
            r_remaining=r_remaining,
            max_abs=max_abs,
            tie_break_scale=scale,
            tau_rule=tau_rule,
            info={"status": "no_global_phase_budget"},
        )

    def source() -> Iterable[torch.Tensor]:
        return selection_chunks(
            ctx, arm.kind, tie_break_scale_abs=scale, floors_by_name=floors_by_name or None
        )

    tau, info = streaming_exact_kth_largest(source, r_remaining, num_bins=kth_num_bins)
    return TauResult(
        arm=arm.name,
        sparsity_percent=sparsity_percent,
        tau=float(tau),
        keep_count=int(keep),
        n_total=int(n_total),
        n_raw_positive=int(n_raw_positive),
        tau_degenerate=degenerate,
        floor_total=floor_total,
        r_remaining=r_remaining,
        max_abs=max_abs,
        tie_break_scale=scale,
        tau_rule=tau_rule,
        info=dict(info),
    )


# ------------------------------------------------------------------ margins / gaps / cert


@dataclass
class ArmMeasurement:
    arm: str
    sparsity_percent: float
    margin_raw: LogHist
    margin_rel: LogHist
    gap_raw: LogHist
    gap_rel: LogHist
    m_margin_raw: Moments
    m_margin_rel: Moments
    m_gap_raw: Moments
    m_gap_rel: Moments
    cert_numer: int = 0
    cert_denom: int = 0
    selected_with_zero_score: int = 0
    nonzero_captured: int = 0
    nonzero_total: int = 0
    n_score_zero: int = 0
    n_score_total: int = 0
    # |{i : sel_i >= tau}| — the global-phase keep set. Reported explicitly rather than reusing
    # keep_count, because under the hybrid rule keep_count also covers the per-layer floors and the
    # two are only approximately equal.
    n_at_or_above_tau: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "arm": self.arm,
            "sparsity_percent": float(self.sparsity_percent),
            "margin_raw": self.margin_raw.to_dict(),
            "margin_rel": self.margin_rel.to_dict(),
            "gap_raw": self.gap_raw.to_dict(),
            "gap_rel": self.gap_rel.to_dict(),
            "m_margin_raw": self.m_margin_raw.to_dict(),
            "m_margin_rel": self.m_margin_rel.to_dict(),
            "m_gap_raw": self.m_gap_raw.to_dict(),
            "m_gap_rel": self.m_gap_rel.to_dict(),
            "cert_numer": int(self.cert_numer),
            "cert_denom": int(self.cert_denom),
            "selected_with_zero_score": int(self.selected_with_zero_score),
            "nonzero_captured": int(self.nonzero_captured),
            "nonzero_total": int(self.nonzero_total),
            "n_score_zero": int(self.n_score_zero),
            "n_score_total": int(self.n_score_total),
            "n_at_or_above_tau": int(self.n_at_or_above_tau),
        }


def _tau_usable(tr: "TauResult") -> bool:
    """A tau is usable as a normalizer only if it is a real score boundary.

    ``tau_degenerate`` means the keep budget exceeded the number of coordinates with any resolvable
    score, so the reported tau is whatever the tie-break perturbation happened to put at rank k —
    order 1e-16, not a property of the scoring function.
    """
    return bool(math.isfinite(tr.tau) and tr.tau > 0.0 and not tr.tau_degenerate)


def _new_measurement(arm: str, rho: float, bins: int) -> ArmMeasurement:
    return ArmMeasurement(
        arm=arm,
        sparsity_percent=rho,
        margin_raw=LogHist(num_bins=bins, log_min=-30.0, log_max=4.0),
        margin_rel=LogHist(num_bins=bins, log_min=-12.0, log_max=8.0),
        gap_raw=LogHist(num_bins=bins, log_min=-30.0, log_max=4.0),
        gap_rel=LogHist(num_bins=bins, log_min=-12.0, log_max=8.0),
        m_margin_raw=Moments(),
        m_margin_rel=Moments(),
        m_gap_raw=Moments(),
        m_gap_rel=Moments(),
    )


def measure_arm(
    ctx: ScoreContext,
    arm: ArmSpec,
    *,
    taus: Dict[float, TauResult],
    tau_oracle: Dict[float, TauResult],
    hist_bins: int,
) -> Dict[float, ArmMeasurement]:
    """One pass over the coverage, accumulating margins/gaps/cert for every rho at once."""
    out = {rho: _new_measurement(arm.name, rho, hist_bins) for rho in sorted(taus)}
    fn = SCORE_FNS[arm.kind]

    for ti, name in enumerate(ctx.names):
        s_raw = fn(ctx, name)
        torch.nan_to_num_(s_raw, nan=0.0, posinf=0.0, neginf=0.0)
        s_star = oracle_score(ctx, name)
        torch.nan_to_num_(s_star, nan=0.0, posinf=0.0, neginf=0.0)

        gap_raw = (s_raw - s_star).abs()
        n_pos_local = int((s_raw > 0).sum().item())
        n_zero_local = int((s_raw == 0).sum().item())

        for rho, tr in taus.items():
            m = out[rho]
            m.n_score_zero += n_zero_local
            m.n_score_total += int(s_raw.numel())
            tau = tr.tau
            tau_star = tau_oracle[rho].tau

            margin_raw = (s_raw - tau).abs() if math.isfinite(tau) else torch.full_like(s_raw, float("nan"))
            if math.isfinite(tau):
                m.margin_raw.update(margin_raw)
                m.m_margin_raw.update(margin_raw)
            m.gap_raw.update(gap_raw)
            m.m_gap_raw.update(gap_raw)

            # tau-relative space: each score vector divided by its own boundary. Refused when
            # either boundary is degenerate — a degenerate tau sits *inside* the tie-break noise
            # (~1e-16), so dividing by it would manufacture 16 decades of meaningless margin.
            usable = _tau_usable(tr) and _tau_usable(tau_oracle[rho])
            if usable:
                s_rel = s_raw / tau
                margin_rel = (s_rel - 1.0).abs()
                gap_rel = (s_rel - s_star / tau_star).abs()
                m.margin_rel.update(margin_rel)
                m.m_margin_rel.update(margin_rel)
                m.gap_rel.update(gap_rel)
                m.m_gap_rel.update(gap_rel)
                m.cert_numer += int((gap_rel < margin_rel).sum().item())
                m.cert_denom += int(gap_rel.numel())
                del s_rel, margin_rel, gap_rel
            elif math.isfinite(tau):
                # Fall back to raw-space certifiability so the number still exists when tau == 0.
                m.cert_numer += int((gap_raw < margin_raw).sum().item())
                m.cert_denom += int(gap_raw.numel())

            # How much of the keep set is decided by signal vs by the tie-break (RESULTS.md §3.1).
            if math.isfinite(tau):
                sel = s_raw + 0.0
                if tr.tie_break_scale > 0.0:
                    g = torch.Generator(device="cpu")
                    g.manual_seed(
                        stream_seed("tie_break", arm.kind, ctx.random_seed if arm.kind == "random" else 0, name)
                    )
                    add_tie_break_noise_(sel, scale=tr.tie_break_scale, generator=g)
                kept = sel >= tau
                m.n_at_or_above_tau += int(kept.sum().item())
                m.selected_with_zero_score += int((kept & (s_raw == 0)).sum().item())
                m.nonzero_captured += int((kept & (s_raw > 0)).sum().item())
                m.nonzero_total += n_pos_local
                del sel, kept
            del margin_raw

        del s_raw, s_star, gap_raw
        if (ti + 1) % 40 == 0:
            print(f"    {arm.name}: {ti + 1}/{len(ctx.names)} tensors", flush=True)
            gc.collect()

    return out


# ------------------------------------------------------------------ artifact IO


def shards_dir(out_dir: Path) -> Path:
    return out_dir / "arm_shards"


def tau_path(out_dir: Path) -> Path:
    return out_dir / "certifiability_tau.json"


def _atomic_save(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def write_merged_artifacts(
    *,
    out_dir: Path,
    arms: Sequence[ArmSpec],
    sparsities: Sequence[float],
    measurements: Dict[Tuple[str, float], ArmMeasurement],
    taus: Dict[Tuple[str, float], TauResult],
    meta: Dict[str, object],
) -> None:
    qs = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    rows: List[Dict[str, object]] = []
    npz: Dict[str, np.ndarray] = {}
    diagnostics: Dict[str, object] = {"meta": meta, "arms": {}}

    for rho in sorted(sparsities):
        for arm in arms:
            key = (arm.name, rho)
            m = measurements.get(key)
            tr = taus.get(key)
            if m is None or tr is None:
                continue
            tag = f"{arm.name}_rho{rho:g}"
            for field_name, hist, mom in (
                ("margin_raw", m.margin_raw, m.m_margin_raw),
                ("margin_rel", m.margin_rel, m.m_margin_rel),
                ("gap_raw", m.gap_raw, m.m_gap_raw),
                ("gap_rel", m.gap_rel, m.m_gap_rel),
            ):
                npz[f"{tag}_{field_name}_counts"] = hist.counts
                npz[f"{tag}_{field_name}_log_edges"] = hist.edges
                npz[f"{tag}_{field_name}_underflow"] = np.array([hist.n_underflow], dtype=np.int64)
                npz[f"{tag}_{field_name}_overflow"] = np.array([hist.n_overflow], dtype=np.int64)
                rows.append(
                    {
                        "arm": arm.name,
                        "sparsity_percent": rho,
                        "quantity": field_name,
                        "n": mom.n,
                        "mean": mom.mean,
                        "std": mom.std,
                        "frac_exact_zero": (mom.n_zero / mom.n) if mom.n else float("nan"),
                        **hist.quantiles(qs),
                    }
                )
            cert = (m.cert_numer / m.cert_denom) if m.cert_denom else float("nan")
            rows.append(
                {
                    "arm": arm.name,
                    "sparsity_percent": rho,
                    "quantity": "cert_strict_fraction",
                    "n": m.cert_denom,
                    "mean": cert,
                    "std": 0.0,
                    "frac_exact_zero": float("nan"),
                    **{f"p{int(round(q * 100)):02d}": float("nan") for q in qs},
                }
            )
            diagnostics["arms"][tag] = {  # type: ignore[index]
                "tau": tr.to_dict(),
                "cert_strict_fraction": cert,
                "cert_numer": m.cert_numer,
                "cert_denom": m.cert_denom,
                "selected_with_zero_score": m.selected_with_zero_score,
                "nonzero_captured": m.nonzero_captured,
                "nonzero_total": m.nonzero_total,
                "n_at_or_above_tau": m.n_at_or_above_tau,
                # Share of the global-phase keep set that carries no signal, so its selection was
                # decided purely by the tie-break perturbation.
                "frac_of_selection_from_tie_break": (
                    m.selected_with_zero_score / m.n_at_or_above_tau if m.n_at_or_above_tau else float("nan")
                ),
                "frac_of_nonzero_captured": (
                    m.nonzero_captured / m.nonzero_total if m.nonzero_total else float("nan")
                ),
                # s_i == 0  =>  m~_i = |0/tau - 1| = 1 exactly, so this *is* the mass of the
                # tau-relative margin sitting on the selection boundary.
                "frac_score_exactly_zero": (
                    m.n_score_zero / m.n_score_total if m.n_score_total else float("nan")
                ),
                "frac_margin_rel_exactly_at_boundary": (
                    m.m_margin_rel.n_zero / m.m_margin_rel.n if m.m_margin_rel.n else float("nan")
                ),
            }

    summary = out_dir / "certifiability_summary.csv"
    fieldnames = sorted({k for r in rows for k in r})
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    np.savez(out_dir / "certifiability_margins.npz", **npz)
    with (out_dir / "certifiability_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, default=str)
    print(f"Wrote {summary}, certifiability_margins.npz, certifiability_diagnostics.json")


# ------------------------------------------------------------------ stages


def resolve_scope(
    initial_sd: Dict[str, torch.Tensor],
    final_sd: Dict[str, torch.Tensor],
    *,
    score_scope: str,
    scope_source: str,
    mlp_only: bool,
    max_keys: Optional[int],
) -> List[str]:
    """Coverage. ``builder_2d`` reproduces the corrected builder's selection, so tau_hat_rho is
    computed over the same N as the mask it characterises.

    Tie de-duplication runs on ``scope_source`` (default the initial weights theta^0). theta^0 is
    where parameter sharing actually lives — two projections can share *deltas* by accident once ~99%
    of a bf16 delta is zero, which is the false-positive class that dropped attention tensors in the
    first place (RESULTS.md §3.2).
    """
    src = {"initial": initial_sd, "final": final_sd}[scope_source]
    if score_scope == "builder_2d":
        names = _select_2d_weight_names(src, mlp_only)
    else:
        names = [n for n in src if (not mlp_only or is_mlp_param(n))]
    names = [
        n
        for n in names
        if n in initial_sd
        and final_sd[n].is_floating_point()
        and initial_sd[n].is_floating_point()
        and tuple(final_sd[n].shape) == tuple(initial_sd[n].shape)
    ]
    if max_keys is not None:
        names = names[: max(0, int(max_keys))]
    return names


def load_context(args: argparse.Namespace) -> Tuple[ScoreContext, Dict[str, object]]:
    dt = args.checkpoint_dtype
    print(f"Loading theta^0 from {args.initial_model}")
    initial_sd = load_state_dict(args.initial_model, device="cpu", torch_dtype=dt)
    print(f"Loading theta^T from {args.final_model}")
    final_sd = load_state_dict(args.final_model, device="cpu", torch_dtype=dt)
    names = resolve_scope(
        initial_sd,
        final_sd,
        score_scope=args.score_scope,
        scope_source=args.scope_source,
        mlp_only=args.mlp_only,
        max_keys=args.max_keys,
    )
    covered = int(sum(int(final_sd[n].numel()) for n in names))
    print(f"Scope={args.score_scope}: {len(names)} tensors, {covered} covered elements")
    ctx = ScoreContext(
        names=names,
        initial_sd=initial_sd,
        final_sd=final_sd,
        random_seed=int(args.random_seed),
        work_dtype=torch.float64 if args.work_dtype == "float64" else torch.float32,
    )
    scope_meta = {
        "score_scope": args.score_scope,
        "scope_source": args.scope_source,
        "n_tensors": len(names),
        "covered_elements": covered,
        "checkpoint_dtype": str(dt).replace("torch.", ""),
        "work_dtype": args.work_dtype,
    }
    return ctx, scope_meta


def stage_cache(args: argparse.Namespace, ctx: ScoreContext) -> Dict[int, Path]:
    cache_dir = Path(args.out_dir) / "magnitude_caches"
    guard_not_under_home(cache_dir, label="magnitude cache dir", allow=args.allow_home_out_dir)
    return build_magnitude_milestone_caches(
        args.delta_log_dir,
        args.milestones,
        ctx.names,
        args.mlp_only,
        cache_dir,
        force_rebuild=args.force_magnitude_cache_rebuild,
    )


def stage_tau(args: argparse.Namespace, ctx: ScoreContext, scope_meta: Dict[str, object]) -> None:
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "magnitude_caches"
    arms = build_arm_specs(args.milestones, [int(args.random_seed)])
    results: Dict[str, Dict[str, object]] = {}

    for arm in arms:
        if arm.kind == "warm":
            p = cache_dir / f"mag_aggregate_step_{arm.milestone}.pt"
            if not p.is_file():
                raise FileNotFoundError(f"Missing magnitude cache {p}; run --execution_mode cache_only first")
            ctx.mag_scores = safe_torch_load(p)
            missing = [n for n in ctx.names if n not in ctx.mag_scores]
            if missing:
                raise RuntimeError(
                    f"{arm.name}: magnitude cache is missing {len(missing)} in-scope tensors "
                    f"(first: {missing[:3]}). Rebuild caches with the current --score_scope."
                )
        else:
            ctx.mag_scores = None

        print(f"[tau] {arm.name}: max|s| / support pass", flush=True)
        max_abs, n_pos, n_tot = arm_max_abs_and_support(ctx, arm.kind)
        print(f"[tau] {arm.name}: max|s|={max_abs:.6e} n_positive={n_pos} N={n_tot}", flush=True)

        for rho in args.sparsity_percents:
            tr = compute_tau(
                ctx,
                arm,
                sparsity_percent=rho,
                tau_rule=args.tau_rule,
                hybrid_min_layer_keep_ratio=args.hybrid_min_layer_keep_ratio,
                tie_break_relative_scale=args.tie_break_relative_scale,
                max_abs=max_abs,
                n_raw_positive=n_pos,
                n_total=n_tot,
                kth_num_bins=args.kth_num_bins,
            )
            results[f"{arm.name}|{rho:g}"] = tr.to_dict()
            flag = "  <-- DEGENERATE (keep budget exceeds signal support)" if tr.tau_degenerate else ""
            print(
                f"[tau] {arm.name} rho={rho:g}: tau={tr.tau:.6e} keep={tr.keep_count} "
                f"R={tr.r_remaining} status={tr.info.get('status')}{flag}",
                flush=True,
            )

        ctx.mag_scores = None
        gc.collect()

    payload = {
        "version": SHARD_VERSION,
        "scope": scope_meta,
        "tau_rule": args.tau_rule,
        "hybrid_min_layer_keep_ratio": args.hybrid_min_layer_keep_ratio,
        "tie_break_relative_scale": args.tie_break_relative_scale,
        "sparsity_percents": list(args.sparsity_percents),
        "milestones": list(args.milestones),
        "random_seed": int(args.random_seed),
        "git_rev": _git_rev(),
        "taus": results,
    }
    with tau_path(out_dir).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Wrote {tau_path(out_dir)}")


def _load_taus(out_dir: Path, arms: Sequence[ArmSpec], sparsities: Sequence[float]) -> Dict[Tuple[str, float], TauResult]:
    p = tau_path(out_dir)
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}; run --execution_mode tau first")
    payload = json.loads(p.read_text(encoding="utf-8"))
    raw = payload["taus"]
    out: Dict[Tuple[str, float], TauResult] = {}
    for arm in arms:
        for rho in sparsities:
            key = f"{arm.name}|{rho:g}"
            if key not in raw:
                raise KeyError(f"tau JSON has no entry for {key}; rerun the tau stage with matching args")
            d = dict(raw[key])
            d["info"] = dict(d.get("info") or {})
            out[(arm.name, float(rho))] = TauResult(**d)
    return out


def stage_arm(args: argparse.Namespace, ctx: ScoreContext) -> None:
    out_dir = Path(args.out_dir)
    arms = build_arm_specs(args.milestones, [int(args.random_seed)])
    by_name = {a.name: a for a in arms}
    if args.arm not in by_name:
        raise ValueError(f"--arm {args.arm!r} not in {sorted(by_name)}")
    arm = by_name[args.arm]

    taus_all = _load_taus(out_dir, arms, args.sparsity_percents)
    taus = {rho: taus_all[(arm.name, rho)] for rho in args.sparsity_percents}
    tau_oracle = {rho: taus_all[("oracle", rho)] for rho in args.sparsity_percents}

    if arm.kind == "warm":
        p = out_dir / "magnitude_caches" / f"mag_aggregate_step_{arm.milestone}.pt"
        ctx.mag_scores = safe_torch_load(p)

    print(f"[arm] {arm.name}: measuring margins/gaps over {len(ctx.names)} tensors", flush=True)
    measurements = measure_arm(ctx, arm, taus=taus, tau_oracle=tau_oracle, hist_bins=args.histogram_bins)

    payload = {
        "version": SHARD_VERSION,
        "arm": arm.name,
        "kind": arm.kind,
        "milestone": arm.milestone,
        "seed": arm.seed,
        "sparsity_percents": list(args.sparsity_percents),
        "measurements": {f"{rho:g}": m.to_dict() for rho, m in measurements.items()},
    }
    sd = shards_dir(out_dir)
    path = sd / f"arm_{arm.name}.pt"
    _atomic_save(payload, path)
    Path(str(path) + ".done").write_text("ok\n", encoding="utf-8")
    print(f"[arm] wrote {path}")


def stage_merge(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    arms = build_arm_specs(args.milestones, [int(args.random_seed)])
    taus = _load_taus(out_dir, arms, args.sparsity_percents)
    sd = shards_dir(out_dir)

    measurements: Dict[Tuple[str, float], ArmMeasurement] = {}
    missing: List[str] = []
    for arm in arms:
        path = sd / f"arm_{arm.name}.pt"
        if not path.is_file() or not Path(str(path) + ".done").is_file():
            missing.append(arm.name)
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("version", -1)) != SHARD_VERSION:
            raise ValueError(f"{path}: shard version mismatch")
        for rho_key, d in payload["measurements"].items():
            rho = float(rho_key)
            measurements[(arm.name, rho)] = ArmMeasurement(
                arm=str(d["arm"]),
                sparsity_percent=float(d["sparsity_percent"]),
                margin_raw=LogHist.from_dict(d["margin_raw"]),
                margin_rel=LogHist.from_dict(d["margin_rel"]),
                gap_raw=LogHist.from_dict(d["gap_raw"]),
                gap_rel=LogHist.from_dict(d["gap_rel"]),
                m_margin_raw=Moments.from_dict(d["m_margin_raw"]),
                m_margin_rel=Moments.from_dict(d["m_margin_rel"]),
                m_gap_raw=Moments.from_dict(d["m_gap_raw"]),
                m_gap_rel=Moments.from_dict(d["m_gap_rel"]),
                cert_numer=int(d["cert_numer"]),
                cert_denom=int(d["cert_denom"]),
                selected_with_zero_score=int(d["selected_with_zero_score"]),
                nonzero_captured=int(d["nonzero_captured"]),
                nonzero_total=int(d["nonzero_total"]),
                n_score_zero=int(d.get("n_score_zero", 0)),
                n_score_total=int(d.get("n_score_total", 0)),
                n_at_or_above_tau=int(d.get("n_at_or_above_tau", 0)),
            )
    if missing:
        raise FileNotFoundError(f"Missing arm shards for {missing} under {sd}")

    tau_payload = json.loads(tau_path(out_dir).read_text(encoding="utf-8"))
    meta = {
        "initial_model": args.initial_model,
        "final_model": args.final_model,
        "delta_log_dir": args.delta_log_dir,
        "milestones": list(args.milestones),
        "sparsity_percents": list(args.sparsity_percents),
        "tau_rule": args.tau_rule,
        "hybrid_min_layer_keep_ratio": args.hybrid_min_layer_keep_ratio,
        "tie_break_relative_scale": args.tie_break_relative_scale,
        "random_seed": int(args.random_seed),
        "model_label": args.model_label,
        "dataset_label": args.dataset_label,
        "objective_label": args.objective_label,
        "scope": tau_payload.get("scope"),
        "git_rev": _git_rev(),
    }
    write_merged_artifacts(
        out_dir=out_dir,
        arms=arms,
        sparsities=args.sparsity_percents,
        measurements=measurements,
        taus=taus,
        meta=meta,
    )


# ------------------------------------------------------------------ cli


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--initial_model", required=True, help="theta^0: prefer the run's deltas/base_state.pt")
    p.add_argument("--final_model", required=True, help="theta^T: the final HF checkpoint dir")
    p.add_argument("--delta_log_dir", required=True, help="dir holding base_state.pt + deltas_step_*.pt")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--magnitude_milestones", default=DEFAULT_MILESTONES)
    p.add_argument(
        "--sparsity_percents",
        default=DEFAULT_SPARSITIES,
        help="Comma-separated rho values (percent pruned). Sweep these to find a regime where the "
        "keep budget does not exceed the signal support (see tau_degenerate).",
    )
    p.add_argument("--score_scope", choices=("builder_2d", "all_params"), default="builder_2d")
    p.add_argument(
        "--scope_source",
        choices=("initial", "final"),
        default="initial",
        help="Which state dict the 2-D selection and tie de-dup run on (default theta^0).",
    )
    p.add_argument("--mlp_only", action="store_true")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--histogram_bins", type=int, default=2000)
    p.add_argument("--kth_num_bins", type=int, default=4096)
    p.add_argument("--max_keys", type=int, default=None, help="Debug only: truncates coverage, so tau is not comparable.")
    p.add_argument("--checkpoint_dtype", type=_dtype_arg, default=_dtype_arg(os.environ.get("CHECKPOINT_DTYPE", "bfloat16")))
    p.add_argument("--work_dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--tau_rule", choices=("global", "hybrid_global_phase"), default="hybrid_global_phase")
    p.add_argument("--hybrid_min_layer_keep_ratio", type=float, default=DEFAULT_HYBRID_MIN_LAYER_KEEP_RATIO)
    p.add_argument(
        "--tie_break_relative_scale",
        type=float,
        default=DEFAULT_TIE_BREAK_RELATIVE_SCALE,
        help="Tie-break amplitude as a fraction of max|s|; 0 disables. Default 1e-12 (mask_utils uses 1e-6, "
        "which randomizes the bottom of the ranking rather than breaking ties).",
    )
    p.add_argument("--force_magnitude_cache_rebuild", action="store_true")
    p.add_argument("--execution_mode", choices=("cache_only", "tau", "arm", "merge", "full"), default="full")
    p.add_argument("--arm", default=None, help="For --execution_mode arm, e.g. oracle | warm_k50 | random_seed42")
    p.add_argument("--allow_home_out_dir", action="store_true", help="Escape hatch for the $HOME write guard.")
    p.add_argument("--model_label", default="")
    p.add_argument("--dataset_label", default="")
    p.add_argument("--objective_label", default="")
    args = p.parse_args(argv)

    args.milestones = sorted({int(x) for x in str(args.magnitude_milestones).split(",") if x.strip()})
    args.sparsity_percents = sorted({float(x) for x in str(args.sparsity_percents).split(",") if x.strip()})
    if not args.milestones:
        p.error("--magnitude_milestones is empty")
    if not args.sparsity_percents:
        p.error("--sparsity_percents is empty")
    if args.tau_rule == "hybrid_global_phase" and args.hybrid_min_layer_keep_ratio <= 0:
        p.error("--tau_rule=hybrid_global_phase needs --hybrid_min_layer_keep_ratio > 0")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    guard_not_under_home(out_dir, label="--out_dir", allow=args.allow_home_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.execution_mode == "merge":
        stage_merge(args)
        return

    ctx, scope_meta = load_context(args)

    if args.execution_mode == "cache_only":
        stage_cache(args, ctx)
        return
    if args.execution_mode == "tau":
        stage_cache(args, ctx)
        stage_tau(args, ctx, scope_meta)
        return
    if args.execution_mode == "arm":
        if not args.arm:
            raise SystemExit("--execution_mode arm requires --arm")
        stage_arm(args, ctx)
        return

    # full: everything in-process (small models / smoke tests)
    stage_cache(args, ctx)
    stage_tau(args, ctx, scope_meta)
    for arm in build_arm_specs(args.milestones, [int(args.random_seed)]):
        args.arm = arm.name
        ctx.mag_scores = None
        stage_arm(args, ctx)
    stage_merge(args)


if __name__ == "__main__":
    main()
