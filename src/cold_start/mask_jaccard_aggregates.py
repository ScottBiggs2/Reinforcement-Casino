"""Layer-type and decoder-index aggregations for mask Jaccard (size-weighted global per bucket)."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

DECODER_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def classify_param_bucket(param_name: str) -> str:
    """Coarse module class for transformer LMs (HF-style names)."""
    n = param_name.lower()
    if any(
        tok in n
        for tok in (
            ".q_proj",
            ".k_proj",
            ".v_proj",
            ".o_proj",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        )
    ):
        return "attn"
    if "mlp" in n and any(
        tok in n for tok in ("gate_proj", "up_proj", "down_proj")
    ):
        return "mlp"
    if "layernorm" in n or "layer_norm" in n or ".norm" in n:
        return "norm"
    return "other"


def decoder_layer_index(param_name: str) -> Optional[int]:
    m = DECODER_LAYER_RE.search(param_name)
    return int(m.group(1)) if m else None


def _tensor_inter_union(
    a: torch.Tensor, b: torch.Tensor, device: str
) -> Tuple[int, int]:
    a = a.to(device).bool()
    b = b.to(device).bool()
    inter = int((a & b).sum().item())
    union = int((a | b).sum().item())
    return inter, union


def aggregate_jaccard_for_keys(
    masks_a: Dict[str, torch.Tensor],
    masks_b: Dict[str, torch.Tensor],
    keys: Iterable[str],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Global Jaccard = sum(inter)/sum(union) over tensors in ``keys`` (size-weighted)."""
    per_layer: Dict[str, float] = {}
    total_intersection = 0
    total_union = 0
    key_list = sorted(keys)
    for name in key_list:
        inter, union = _tensor_inter_union(masks_a[name], masks_b[name], device)
        per_layer[name] = round(inter / union if union > 0 else 0.0, 6)
        total_intersection += inter
        total_union += union
    aggregate = (
        total_intersection / total_union if total_union > 0 else 0.0
    )
    vals = list(per_layer.values())
    return {
        "aggregate_jaccard": round(aggregate, 6),
        "mean_jaccard": round(sum(vals) / len(vals), 6) if vals else 0.0,
        "min_jaccard": round(min(vals), 6) if vals else 0.0,
        "max_jaccard": round(max(vals), 6) if vals else 0.0,
        "per_layer": per_layer,
        "n_layers": len(per_layer),
        "total_intersection": int(total_intersection),
        "total_union": int(total_union),
    }


def jaccard_by_param_bucket(
    masks_a: Dict[str, torch.Tensor],
    masks_b: Dict[str, torch.Tensor],
    device: str = "cpu",
) -> Dict[str, Any]:
    common = set(masks_a.keys()) & set(masks_b.keys())
    bucket_to_keys: Dict[str, List[str]] = {}
    for name in common:
        b = classify_param_bucket(name)
        bucket_to_keys.setdefault(b, []).append(name)
    out: Dict[str, Any] = {}
    for b in sorted(bucket_to_keys.keys()):
        out[b] = aggregate_jaccard_for_keys(
            masks_a, masks_b, bucket_to_keys[b], device=device
        )
    return out


def jaccard_by_decoder_layer(
    masks_a: Dict[str, torch.Tensor],
    masks_b: Dict[str, torch.Tensor],
    device: str = "cpu",
) -> Dict[str, Any]:
    common = set(masks_a.keys()) & set(masks_b.keys())
    layer_to_keys: Dict[str, List[str]] = {}
    for name in common:
        idx = decoder_layer_index(name)
        key = str(idx) if idx is not None else "non_decoder"
        layer_to_keys.setdefault(key, []).append(name)
    out: Dict[str, Any] = {}
    # numeric layers first, then non_decoder
    def sort_key(k: str) -> Tuple[int, str]:
        if k == "non_decoder":
            return (10**9, k)
        try:
            return (int(k), k)
        except ValueError:
            return (10**9 - 1, k)

    for layer_key in sorted(layer_to_keys.keys(), key=sort_key):
        out[layer_key] = aggregate_jaccard_for_keys(
            masks_a, masks_b, layer_to_keys[layer_key], device=device
        )
    return out


def extended_jaccard_report(
    masks_a: Dict[str, torch.Tensor],
    masks_b: Dict[str, torch.Tensor],
    *,
    device: str = "cpu",
    include_param_buckets: bool = False,
    include_decoder_layers: bool = False,
) -> Dict[str, Any]:
    """Optional sections for JSON reports (pairwise masks; no duplicate global Jaccard)."""
    report: Dict[str, Any] = {}
    if include_param_buckets:
        report["by_param_bucket"] = jaccard_by_param_bucket(
            masks_a, masks_b, device=device
        )
    if include_decoder_layers:
        report["by_decoder_layer"] = jaccard_by_decoder_layer(
            masks_a, masks_b, device=device
        )
    return report
