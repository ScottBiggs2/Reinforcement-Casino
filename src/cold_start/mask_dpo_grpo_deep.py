"""Phase-1 deep DPO-vs-GRPO mask comparison primitives.

All ops are CPU + bool-tensor only; one mask in memory at a time where possible.
Designed for Llama-3.1-8B-Instruct: 32 layers, 32 query heads, 8 KV heads,
head_dim=128, intermediate_size=14336.

Public surface (called by scripts/mask_dpo_grpo_deep.py):

    classify_subbucket(param_name) -> str         # q,k,v,o,gate,up,down,attn_norm,mlp_norm,final_norm,embed,lm_head,other
    decoder_layer_index(param_name) -> int|None
    per_mask_density_layer_subbucket(mask)        -> list[dict]
    per_mask_attn_head_density(mask, n_q, n_kv, head_dim) -> list[dict]
    per_mask_mlp_neuron_density(mask)             -> list[dict]   # one row per (layer, neuron_idx)
    pair_jaccard_by_subbucket(mask_a, mask_b)     -> list[dict]
    disjoint_magnitude_histogram(mask_a, mask_b, base_state_dict, bins) -> list[dict]
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

DECODER_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def classify_subbucket(param_name: str) -> str:
    n = param_name.lower()
    if "self_attn.q_proj" in n:
        return "q"
    if "self_attn.k_proj" in n:
        return "k"
    if "self_attn.v_proj" in n:
        return "v"
    if "self_attn.o_proj" in n:
        return "o"
    if "mlp.gate_proj" in n:
        return "gate"
    if "mlp.up_proj" in n:
        return "up"
    if "mlp.down_proj" in n:
        return "down"
    if "input_layernorm" in n:
        return "attn_norm"
    if "post_attention_layernorm" in n:
        return "mlp_norm"
    if n.endswith("model.norm.weight"):
        return "final_norm"
    if "embed_tokens" in n:
        return "embed"
    if "lm_head" in n:
        return "lm_head"
    return "other"


def decoder_layer_index(param_name: str) -> Optional[int]:
    m = DECODER_LAYER_RE.search(param_name)
    return int(m.group(1)) if m else None


def per_mask_density_layer_subbucket(masks: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    """Density per (layer, subbucket). Layer = -1 for non-decoder params."""
    rows: List[Dict[str, Any]] = []
    for name, t in masks.items():
        bucket = classify_subbucket(name)
        layer = decoder_layer_index(name)
        layer_key = -1 if layer is None else layer
        b = t.bool()
        n_total = b.numel()
        n_active = int(b.sum().item())
        rows.append(
            {
                "param": name,
                "layer": layer_key,
                "subbucket": bucket,
                "n_total": n_total,
                "n_active": n_active,
                "density": n_active / n_total if n_total > 0 else 0.0,
            }
        )
    return rows


def per_mask_attn_head_density(
    masks: Dict[str, torch.Tensor],
    n_q_heads: int = 32,
    n_kv_heads: int = 8,
    head_dim: int = 128,
) -> List[Dict[str, Any]]:
    """For attn projections, compute density per query-head (Q+O) and per kv-head (K+V).

    Q (out, in)  -> reshape (n_q, head_dim, in)         -> mean over (head_dim, in) per head
    K, V (out, in) where out=n_kv*head_dim -> (n_kv, head_dim, in) -> mean per kv-head
    O (out, in) where in=n_q*head_dim -> (out, n_q, head_dim) -> mean per query-head
    """
    rows: List[Dict[str, Any]] = []
    for name, t in masks.items():
        layer = decoder_layer_index(name)
        if layer is None:
            continue
        bucket = classify_subbucket(name)
        b = t.bool()
        if bucket == "q":
            shp = b.shape
            mat = b.reshape(n_q_heads, head_dim, shp[1])
            for h in range(n_q_heads):
                rows.append(
                    {
                        "layer": layer,
                        "head_kind": "q",
                        "head_idx": h,
                        "param": name,
                        "density": mat[h].float().mean().item(),
                    }
                )
        elif bucket == "o":
            shp = b.shape
            mat = b.reshape(shp[0], n_q_heads, head_dim)
            for h in range(n_q_heads):
                rows.append(
                    {
                        "layer": layer,
                        "head_kind": "o",
                        "head_idx": h,
                        "param": name,
                        "density": mat[:, h, :].float().mean().item(),
                    }
                )
        elif bucket in ("k", "v"):
            shp = b.shape
            mat = b.reshape(n_kv_heads, head_dim, shp[1])
            for h in range(n_kv_heads):
                rows.append(
                    {
                        "layer": layer,
                        "head_kind": bucket,
                        "head_idx": h,
                        "param": name,
                        "density": mat[h].float().mean().item(),
                    }
                )
    return rows


def per_mask_mlp_neuron_density(masks: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    """Density of mask along the hidden-neuron axis (4*hidden) of MLP.

    gate, up: (intermediate, hidden) -> per-neuron density = density of row n
    down:     (hidden, intermediate) -> per-neuron density = density of column n
    Writes one row per (layer, neuron_idx).
    """
    layers: Dict[int, Dict[str, torch.Tensor]] = {}
    for name, t in masks.items():
        layer = decoder_layer_index(name)
        if layer is None:
            continue
        bucket = classify_subbucket(name)
        if bucket not in ("gate", "up", "down"):
            continue
        layers.setdefault(layer, {})[bucket] = t.bool()
    rows: List[Dict[str, Any]] = []
    for layer, ts in layers.items():
        if not all(k in ts for k in ("gate", "up", "down")):
            continue
        gate_d = ts["gate"].float().mean(dim=1)
        up_d = ts["up"].float().mean(dim=1)
        down_d = ts["down"].float().mean(dim=0)
        n = gate_d.shape[0]
        for i in range(n):
            rows.append(
                {
                    "layer": layer,
                    "neuron_idx": i,
                    "gate_density": gate_d[i].item(),
                    "up_density": up_d[i].item(),
                    "down_density": down_d[i].item(),
                }
            )
    return rows


def pair_jaccard_by_subbucket(
    mask_a: Dict[str, torch.Tensor],
    mask_b: Dict[str, torch.Tensor],
) -> List[Dict[str, Any]]:
    """Per-(layer, subbucket) Jaccard, plus a layer-collapsed `_all_layers` row per subbucket."""
    common = sorted(set(mask_a) & set(mask_b))
    bucket_layer_io: Dict[Tuple[str, int], Tuple[int, int]] = {}
    for k in common:
        a = mask_a[k].bool()
        b = mask_b[k].bool()
        inter = int((a & b).sum().item())
        union = int((a | b).sum().item())
        bucket = classify_subbucket(k)
        layer = decoder_layer_index(k)
        layer_key = -1 if layer is None else layer
        prev = bucket_layer_io.get((bucket, layer_key), (0, 0))
        bucket_layer_io[(bucket, layer_key)] = (prev[0] + inter, prev[1] + union)
    rows: List[Dict[str, Any]] = []
    for (bucket, layer), (inter, union) in sorted(bucket_layer_io.items()):
        rows.append(
            {
                "subbucket": bucket,
                "layer": layer,
                "intersection": inter,
                "union": union,
                "jaccard": inter / union if union > 0 else 0.0,
            }
        )
    bucket_io: Dict[str, Tuple[int, int]] = {}
    for (bucket, _), (inter, union) in bucket_layer_io.items():
        prev = bucket_io.get(bucket, (0, 0))
        bucket_io[bucket] = (prev[0] + inter, prev[1] + union)
    for bucket, (inter, union) in sorted(bucket_io.items()):
        rows.append(
            {
                "subbucket": bucket,
                "layer": -2,
                "intersection": inter,
                "union": union,
                "jaccard": inter / union if union > 0 else 0.0,
            }
        )
    return rows


def disjoint_magnitude_histogram(
    mask_a: Dict[str, torch.Tensor],
    mask_b: Dict[str, torch.Tensor],
    base_state_dict: Dict[str, torch.Tensor],
    *,
    bins: torch.Tensor,
) -> List[Dict[str, Any]]:
    """For each region in {a_only, b_only, both}, accumulate |W| histogram across all params.

    `bins` defines bin EDGES; output rows have (region, bin_lo, bin_hi, count).
    Final extra rows: region totals (region, bin_lo=NaN, bin_hi=NaN, count=total, mean_abs=...).
    """
    counts = {r: torch.zeros(len(bins) - 1, dtype=torch.float64) for r in ("both", "a_only", "b_only")}
    sums = {r: 0.0 for r in counts}
    totals = {r: 0 for r in counts}
    for k in sorted(set(mask_a) & set(mask_b) & set(base_state_dict)):
        a = mask_a[k].bool()
        b = mask_b[k].bool()
        w = base_state_dict[k].abs().float()
        if w.shape != a.shape:
            continue
        regions = {
            "both": a & b,
            "a_only": a & ~b,
            "b_only": (~a) & b,
        }
        for region_name, sel in regions.items():
            if sel.any():
                vals = w[sel]
                counts[region_name] += torch.histc(
                    vals, bins=len(bins) - 1, min=float(bins[0]), max=float(bins[-1])
                ).double()
                sums[region_name] += float(vals.sum().item())
                totals[region_name] += int(vals.numel())
    rows: List[Dict[str, Any]] = []
    bin_lo = bins[:-1].tolist()
    bin_hi = bins[1:].tolist()
    for region, hist in counts.items():
        for i, (lo, hi) in enumerate(zip(bin_lo, bin_hi)):
            rows.append(
                {
                    "region": region,
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "count": int(hist[i].item()),
                }
            )
    for region in counts:
        n = totals[region]
        rows.append(
            {
                "region": region,
                "bin_lo": float("nan"),
                "bin_hi": float("nan"),
                "count": n,
                "mean_abs": (sums[region] / n) if n > 0 else 0.0,
                "summary": True,
            }
        )
    return rows
