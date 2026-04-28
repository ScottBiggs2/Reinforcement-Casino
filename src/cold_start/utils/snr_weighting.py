"""
Utilities for optional SNR-based reweighting of pruning scores.

This module is intentionally standalone and lightweight so SNIP/GRaSP can share:
- online moment accumulation across calibration minibatches
- SNR -> multiplier transforms
- guardrails for per-weight accumulation (RAM-heavy for large models)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Tuple

import torch

ScoreSNRMode = Literal["off", "per_tensor", "per_weight"]
ScoreSNRTransform = Literal["identity", "log1p", "clamp"]


@dataclass(frozen=True)
class SNRConfig:
    mode: ScoreSNRMode = "off"
    eps: float = 1e-8
    transform: ScoreSNRTransform = "log1p"
    clamp_min: float = 0.0
    clamp_max: float = 50.0
    ram_budget_gb: float = 8.0
    allow_large_ram: bool = False


def _to_cpu_f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cpu", dtype=torch.float32, non_blocking=False).contiguous()


def estimate_welford_ram_gb(param_numel: int) -> float:
    """Rough RAM for Welford per-weight mean+M2 in float32."""
    bytes_per_f32 = 4
    # mean + M2
    total_bytes = int(param_numel) * bytes_per_f32 * 2
    return total_bytes / (1024**3)


def estimate_params_welford_ram_gb(params: Iterable[torch.Tensor]) -> float:
    return sum(estimate_welford_ram_gb(int(p.numel())) for p in params)


def _apply_transform(snr: torch.Tensor, cfg: SNRConfig) -> torch.Tensor:
    if cfg.transform == "identity":
        return snr
    if cfg.transform == "log1p":
        return torch.log1p(torch.clamp(snr, min=0.0))
    if cfg.transform == "clamp":
        return torch.clamp(snr, min=float(cfg.clamp_min), max=float(cfg.clamp_max))
    raise ValueError(f"Unknown SNR transform: {cfg.transform!r}")


class _WelfordTensor:
    """Welford moments for a tensor on CPU float32."""

    def __init__(self, shape: torch.Size):
        self.count = 0
        self.mean = torch.zeros(shape, dtype=torch.float32, device="cpu")
        self.m2 = torch.zeros(shape, dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def update(self, x_cpu_f32: torch.Tensor) -> None:
        x = x_cpu_f32
        if x.device.type != "cpu" or x.dtype != torch.float32:
            raise ValueError("Welford update expects CPU float32 tensor.")
        self.count += 1
        if self.count == 1:
            self.mean.copy_(x)
            self.m2.zero_()
            return
        delta = x - self.mean
        self.mean.add_(delta / float(self.count))
        delta2 = x - self.mean
        self.m2.add_(delta * delta2)

    @torch.no_grad()
    def finalize_mean_std(self, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.count <= 1:
            std = torch.zeros_like(self.mean)
            return self.mean.clone(), std
        var = self.m2 / float(self.count - 1)
        std = torch.sqrt(torch.clamp(var, min=0.0) + float(eps))
        return self.mean.clone(), std


class _WelfordScalar:
    """Welford moments for scalar floats."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / float(self.count)
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def finalize_mean_std(self, eps: float) -> Tuple[float, float]:
        if self.count <= 1:
            return float(self.mean), float(eps)
        var = self.m2 / float(self.count - 1)
        std = (max(var, 0.0) + float(eps)) ** 0.5
        return float(self.mean), float(std)


class GradientSNRAccumulator:
    """
    Accumulate gradient SNR statistics across minibatches.

    Modes:
    - per_tensor: track scalar moments of grad magnitude per parameter tensor per minibatch.
    - per_weight: track per-element moments of grad per parameter tensor per minibatch (RAM heavy).
    """

    def __init__(self, *, cfg: SNRConfig, params_by_name: Dict[str, torch.Tensor]):
        self.cfg = cfg
        self.params_by_name = params_by_name
        self._scalar_stats: Dict[str, _WelfordScalar] = {}
        self._tensor_stats: Dict[str, _WelfordTensor] = {}

        if cfg.mode == "per_tensor":
            self._scalar_stats = {name: _WelfordScalar() for name in params_by_name}
        elif cfg.mode == "per_weight":
            est_gb = estimate_params_welford_ram_gb(params_by_name.values())
            if (est_gb > cfg.ram_budget_gb) and not cfg.allow_large_ram:
                raise MemoryError(
                    "Per-weight SNR accumulation is RAM-heavy. "
                    f"Estimated mean+M2 RAM ≈ {est_gb:.2f} GB, budget={cfg.ram_budget_gb:.2f} GB. "
                    "Use --score-snr per_tensor, or increase --score-snr-ram-budget-gb, "
                    "or pass --score-snr-allow-large-ram to override."
                )
            self._tensor_stats = {name: _WelfordTensor(p.shape) for name, p in params_by_name.items()}
        elif cfg.mode == "off":
            pass
        else:
            raise ValueError(f"Unknown SNR mode: {cfg.mode!r}")

    @torch.no_grad()
    def update_from_batch_grads(self, batch_grads_by_name: Dict[str, torch.Tensor]) -> None:
        if self.cfg.mode == "off":
            return

        if self.cfg.mode == "per_tensor":
            for name, g in batch_grads_by_name.items():
                if name not in self._scalar_stats:
                    continue
                gf = g.detach().float()
                # Scalar proxy for gradient magnitude on this tensor for this minibatch.
                val = float(gf.abs().mean().item())
                self._scalar_stats[name].update(val)
            return

        if self.cfg.mode == "per_weight":
            for name, g in batch_grads_by_name.items():
                if name not in self._tensor_stats:
                    continue
                self._tensor_stats[name].update(_to_cpu_f32(g))
            return

        raise ValueError(f"Unsupported SNR mode: {self.cfg.mode!r}")

    @torch.no_grad()
    def snr_multipliers(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Returns
        -------
        multipliers:
            Per-parameter tensor multipliers. For per_tensor mode these are scalar tensors.
        summary:
            Small float summary stats: mean of multipliers per tensor (useful for metadata).
        """
        if self.cfg.mode == "off":
            return {}, {}

        multipliers: Dict[str, torch.Tensor] = {}
        summary: Dict[str, float] = {}

        if self.cfg.mode == "per_tensor":
            for name, stat in self._scalar_stats.items():
                mu, std = stat.finalize_mean_std(self.cfg.eps)
                snr = abs(mu) / float(std)
                s = torch.tensor(snr, dtype=torch.float32, device="cpu")
                m = _apply_transform(s, self.cfg)
                multipliers[name] = m
                summary[name] = float(m.item())
            return multipliers, summary

        if self.cfg.mode == "per_weight":
            for name, stat in self._tensor_stats.items():
                mu, std = stat.finalize_mean_std(self.cfg.eps)
                snr = mu.abs() / (std + float(self.cfg.eps))
                m = _apply_transform(snr, self.cfg)
                multipliers[name] = m
                summary[name] = float(m.mean().item()) if m.numel() else 0.0
            return multipliers, summary

        raise ValueError(f"Unsupported SNR mode: {self.cfg.mode!r}")


def summarize_multiplier_stats(mult_summary: Dict[str, float]) -> Dict[str, float]:
    if not mult_summary:
        return {"n_tensors": 0}
    vals = torch.tensor(list(mult_summary.values()), dtype=torch.float32)
    return {
        "n_tensors": int(vals.numel()),
        "min": float(vals.min().item()),
        "max": float(vals.max().item()),
        "mean": float(vals.mean().item()),
        "std": float(vals.std(unbiased=False).item()),
    }

