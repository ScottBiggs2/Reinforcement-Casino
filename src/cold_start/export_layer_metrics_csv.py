#!/usr/bin/env python3
"""Export per-layer mask comparison metrics to CSV.

This script is designed for charting and analysis workflows.

For each shared layer between `mask_a` and `mask_b`, it exports:
  - per-mask sparsity / density
  - per-mask effective rank (entropy-based, from singular values)
  - per-layer Jaccard similarity
  - optional per-layer CKA (if a CKA JSON is provided)
"""

import argparse
import csv
import json
import math
import os
import sys
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple

import torch


LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
# Decoder block anywhere in the name (mask keys may be prefixed e.g. base_model.model.layers...)
DECODER_BLOCK_SEARCH = re.compile(r"(model\.layers\.\d+)")


def _normalize_cka_module_key(name: str) -> str:
    """Strip common PEFT / wrapper prefixes so hook names align with mask checkpoints."""
    s = str(name)
    for prefix in ("base_model.model.", "base_model.", "model.model."):
        if s.startswith(prefix):
            return s[len(prefix) :]
    return s


def resolve_cka_for_canonical_layer(
    canonical: str,
    per_layer_cka: Dict[str, float],
) -> float:
    """Map a mask tensor name to CKA from ``mask_to_cka`` JSON.

    CKA hooks attach to ``*.mlp.down_proj`` modules (one scalar per decoder layer).
    Mask CSV rows use per-weight names (``...q_proj``, ``...down_proj``, ...). We
    broadcast the down_proj CKA to every tensor in the same ``model.layers.L`` block.

    Module strings may differ between hooks and masks (e.g. ``base_model.model.layers`` vs
    ``model.layers``); we match on ``model.layers.N`` anywhere in the canonical name.
    """
    if not per_layer_cka:
        return float("nan")

    def _coerce(v) -> float:
        if v is None:
            return float("nan")
        try:
            fv = float(v)
            return fv
        except (TypeError, ValueError):
            return float("nan")

    c_norm = _normalize_cka_module_key(canonical)
    for key in (canonical, c_norm, canonical_name(canonical), canonical_name(c_norm)):
        if key in per_layer_cka:
            return _coerce(per_layer_cka[key])

    m = DECODER_BLOCK_SEARCH.search(canonical) or DECODER_BLOCK_SEARCH.search(c_norm)
    if not m:
        return float("nan")
    block = m.group(1)
    down_key = f"{block}.mlp.down_proj"
    for dk in (
        down_key,
        _normalize_cka_module_key(down_key),
        canonical_name(down_key),
        canonical_name(_normalize_cka_module_key(down_key)),
    ):
        if dk in per_layer_cka:
            return _coerce(per_layer_cka[dk])

    prefix = block + "."
    for k, v in per_layer_cka.items():
        ks = str(k)
        if not ks.startswith(prefix) and not _normalize_cka_module_key(ks).startswith(prefix):
            continue
        fv = _coerce(v)
        if fv == fv:
            return fv
    return float("nan")


def cka_value_for_layer_row(
    canonical: str,
    raw_cka: Dict[str, float],
) -> float:
    """CKA scalar for one CSV row: hook/broadcast resolution, then Irene-style direct lookup."""
    v = resolve_cka_for_canonical_layer(canonical, raw_cka)
    if v == v:  # not NaN
        return v
    for key in (canonical, canonical_name(canonical)):
        if key in raw_cka:
            x = raw_cka[key]
            try:
                fv = float(x)
                if fv == fv:
                    return fv
            except (TypeError, ValueError):
                pass
    return float("nan")


def load_masks(path: str) -> Tuple[Dict[str, torch.Tensor], Optional[dict]]:
    """Load a mask dict from a `.pt` file."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict) and "masks" in data:
        return data["masks"], data.get("metadata")
    if isinstance(data, dict):
        return data, None
    raise ValueError(f"Unrecognized mask format: {path}")


def canonical_name(name: str) -> str:
    """Normalize parameter/layer names so JSON sources can be merged cleanly."""
    if name.endswith(".weight"):
        return name[:-7]
    return name


def parse_layer_index(name: str) -> Optional[int]:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def compute_basic_stats(mask_tensor: torch.Tensor) -> dict:
    """Compute count, density, and sparsity stats for a mask tensor."""
    m = mask_tensor.bool()
    total = int(m.numel())
    kept = int(m.sum().item())
    density = (kept / total) if total > 0 else 0.0
    sparsity = 1.0 - density
    return {
        "n_params": total,
        "n_kept": kept,
        "density": density,
        "sparsity": sparsity,
    }


def effective_rank(mask_tensor: torch.Tensor, eps: float = 1e-12) -> Tuple[Optional[float], Optional[float]]:
    """Compute entropy-based effective rank and normalized effective rank.

    Effective rank is defined as:
      erank(W) = exp(H(p)), where p_i = sigma_i / sum_j sigma_j
      and sigma_i are singular values of W.

    Returns (erank, erank_normalized). For tensors with <2 dims, returns (None, None).
    """
    if mask_tensor.ndim < 2:
        return None, None

    # Match RL-irene: full `svdvals` on the 2D mask slice (plots + behavior validated there).
    W = mask_tensor.float().reshape(mask_tensor.shape[0], -1)
    m, n = W.shape
    max_rank = min(m, n)
    if max_rank == 0:
        return 0.0, 0.0

    s = torch.linalg.svdvals(W)
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0, 0.0

    p = s / s.sum()
    entropy = -(p * torch.log(torch.clamp(p, min=eps))).sum().item()
    erank = float(math.exp(entropy))
    return erank, erank / float(max_rank)


def _both_effective_ranks(
    pair: Tuple[str, torch.Tensor, torch.Tensor],
) -> Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float]]:
    """One layer: effective rank for mask A and mask B (for parallel map)."""
    name, t_a, t_b = pair
    er_a, er_a_norm = effective_rank(t_a)
    er_b, er_b_norm = effective_rank(t_b)
    return name, er_a, er_a_norm, er_b, er_b_norm


def compute_per_layer_jaccard(masks_a: Dict[str, torch.Tensor], masks_b: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute per-layer Jaccard for shared layers."""
    out = {}
    for name in sorted(set(masks_a.keys()) & set(masks_b.keys())):
        a = masks_a[name].bool()
        b = masks_b[name].bool()
        union = int((a | b).sum().item())
        inter = int((a & b).sum().item())
        out[name] = (inter / union) if union > 0 else 0.0
    return out


def load_per_layer_json(path: Optional[str], key: str) -> Dict[str, float]:
    """Load a per-layer metric dict from JSON; index by raw and canonical names.

    - Keeps JSON ``null`` as NaN so keys remain addressable (export used to drop nulls).
    - Merges nested layouts: ``per_layer_cka``, ``per_layer``, ``cka.per_layer``, etc.
    - Registers normalized module prefixes for CKA (hooks vs mask naming).
    """
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get(key)
    if not isinstance(raw, dict):
        raw = {}
    # Alternate layouts (only when primary key missing or explicitly empty for CKA)
    if not raw and key == "per_layer_cka":
        cka_block = data.get("cka")
        if isinstance(cka_block, dict):
            nested = cka_block.get("per_layer_cka") or cka_block.get("per_layer")
            if isinstance(nested, dict):
                raw = nested

    out: Dict[str, float] = {}
    for k, v in raw.items():
        ks = str(k)
        if v is None:
            fv = float("nan")
        else:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
        kn = _normalize_cka_module_key(ks)
        for alias in (ks, canonical_name(ks), kn, canonical_name(kn)):
            out[alias] = fv
    return out


def default_output_path(mask_a: str, mask_b: str) -> str:
    base_a = os.path.splitext(os.path.basename(mask_a))[0]
    base_b = os.path.splitext(os.path.basename(mask_b))[0]
    try:
        out_dir = os.path.commonpath([os.path.abspath(mask_a), os.path.abspath(mask_b)])
    except ValueError:
        out_dir = "."
    return os.path.join(out_dir, f"layer_metrics_{base_a}_vs_{base_b}.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Export per-layer mask metrics (sparsity, effective rank, Jaccard, optional CKA) to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mask_a", type=str, help="First mask file (.pt)")
    parser.add_argument("mask_b", type=str, help="Second mask file (.pt)")
    parser.add_argument("--cka-json", type=str, default=None, help="Optional CKA JSON containing `per_layer_cka`.")
    parser.add_argument("--jaccard-json", type=str, default=None, help="Optional Jaccard JSON containing `per_layer_jaccard`.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV path.")
    parser.add_argument(
        "--skip_effective_rank",
        action="store_true",
        help="Skip SVD-based effective rank (much faster; rank columns left empty). "
        "Recommended for large models: per-layer svdvals dominates wall time.",
    )
    parser.add_argument(
        "--effective_rank_workers",
        type=int,
        default=None,
        help="Parallel threads for per-layer SVD (ignored with --skip_effective_rank). "
        "Default: min(8, max(1, cpu_count//2)). Use 1 for fully serial.",
    )
    args = parser.parse_args()

    for path in (args.mask_a, args.mask_b):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Mask file not found: {path}")

    masks_a, meta_a = load_masks(args.mask_a)
    masks_b, meta_b = load_masks(args.mask_b)

    common = sorted(set(masks_a.keys()) & set(masks_b.keys()))
    if not common:
        raise ValueError("The two masks share no common layer names.")

    if args.jaccard_json and not os.path.isfile(args.jaccard_json):
        print(
            f"WARNING: --jaccard-json not found ({args.jaccard_json}); computing Jaccard from masks.",
            file=sys.stderr,
        )
        args.jaccard_json = None

    if args.cka_json and not os.path.isfile(args.cka_json):
        print(
            f"WARNING: --cka-json not found ({args.cka_json}); CKA column will be NaN. "
            "Run mask_to_cka for this pair or omit --cka-json.",
            file=sys.stderr,
        )
        args.cka_json = None

    if args.jaccard_json:
        per_layer_jaccard = load_per_layer_json(args.jaccard_json, "per_layer_jaccard")
    else:
        per_layer_jaccard = {
            canonical_name(k): v
            for k, v in compute_per_layer_jaccard(masks_a, masks_b).items()
        }

    raw_cka = load_per_layer_json(args.cka_json, "per_layer_cka")

    if args.skip_effective_rank:
        er_out = None
    else:
        if args.effective_rank_workers is not None:
            workers = max(1, int(args.effective_rank_workers))
        else:
            workers = min(8, max(1, (os.cpu_count() or 4) // 2))
        # Avoid BLAS oversubscription: many threads each calling svdvals
        if workers > 1:
            torch.set_num_threads(max(1, (os.cpu_count() or 4) // workers))
        pairs = [(n, masks_a[n], masks_b[n]) for n in common]
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                er_out = list(ex.map(_both_effective_ranks, pairs))
        else:
            er_out = [_both_effective_ranks(p) for p in pairs]

    rows = []
    for idx, name in enumerate(common):
        ca = canonical_name(name)
        t_a = masks_a[name]
        t_b = masks_b[name]

        s_a = compute_basic_stats(t_a)
        s_b = compute_basic_stats(t_b)

        if args.skip_effective_rank:
            er_a = er_a_norm = er_b = er_b_norm = None
        else:
            assert er_out is not None
            _, er_a, er_a_norm, er_b, er_b_norm = er_out[idx]

        rows.append(
            {
                "mask_a_name": os.path.basename(args.mask_a),
                "mask_b_name": os.path.basename(args.mask_b),
                "layer": ca,
                "layer_index": parse_layer_index(ca),
                "shape_a": str(tuple(t_a.shape)),
                "shape_b": str(tuple(t_b.shape)),
                "n_params": s_a["n_params"],
                "kept_a": s_a["n_kept"],
                "density_a": round(s_a["density"], 8),
                "sparsity_a": round(s_a["sparsity"], 8),
                "effective_rank_a": None if er_a is None else round(er_a, 8),
                "effective_rank_a_norm": None if er_a_norm is None else round(er_a_norm, 8),
                "kept_b": s_b["n_kept"],
                "density_b": round(s_b["density"], 8),
                "sparsity_b": round(s_b["sparsity"], 8),
                "effective_rank_b": None if er_b is None else round(er_b, 8),
                "effective_rank_b_norm": None if er_b_norm is None else round(er_b_norm, 8),
                "jaccard": round(per_layer_jaccard.get(ca, float("nan")), 8),
                "cka": round(cka_value_for_layer_row(ca, raw_cka), 8),
            }
        )

    rows.sort(
        key=lambda r: (
            r["layer_index"] if r["layer_index"] is not None else 10**9,
            r["layer"],
        )
    )

    output_path = args.output or default_output_path(args.mask_a, args.mask_b)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    fieldnames = [
        "mask_a_name",
        "mask_b_name",
        "layer",
        "layer_index",
        "shape_a",
        "shape_b",
        "n_params",
        "kept_a",
        "density_a",
        "sparsity_a",
        "effective_rank_a",
        "effective_rank_a_norm",
        "kept_b",
        "density_b",
        "sparsity_b",
        "effective_rank_b",
        "effective_rank_b_norm",
        "jaccard",
        "cka",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} layer rows to: {output_path}")
    if args.skip_effective_rank:
        print("(effective_rank columns skipped; omit --skip_effective_rank for full SVD — slow on large models)")
    else:
        print(f"(effective_rank: {workers} worker(s); set --effective_rank_workers 1 for serial)")
    if meta_a:
        print(f"metadata_a: {meta_a}")
    if meta_b:
        print(f"metadata_b: {meta_b}")
    if args.cka_json:
        print(f"CKA source: {args.cka_json}")
    if args.jaccard_json:
        print(f"Jaccard source: {args.jaccard_json}")


if __name__ == "__main__":
    main()
