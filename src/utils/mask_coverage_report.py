import json
from typing import Any, Dict, List, Tuple

import torch


def _unwrap_masks(maybe_wrapped: Any) -> Dict[str, torch.Tensor]:
    """
    Accept either:
    - dict[name -> tensor] (legacy)
    - {"masks": {name -> tensor}, "metadata": ...} (wrapped)
    and return the masks dict.
    """
    if isinstance(maybe_wrapped, dict) and "masks" in maybe_wrapped and isinstance(maybe_wrapped["masks"], dict):
        return maybe_wrapped["masks"]
    if isinstance(maybe_wrapped, dict):
        return maybe_wrapped
    raise TypeError("Unsupported mask object; expected dict or wrapped dict with 'masks'.")


def compute_mask_coverage_report(
    *,
    model: torch.nn.Module,
    masks: Any,
    topk_missing: int = 20,
) -> Dict[str, Any]:
    """
    Compute coverage metrics between a model's parameters and a mask dict.

    Returns a JSON-serializable dict suitable for logging and gating.
    """
    masks_dict = _unwrap_masks(masks)
    model_params = list(model.named_parameters())

    total_params = len(model_params)
    numel_total_model = 0
    numel_total_2d = 0
    numel_covered_total = 0
    numel_covered_2d = 0

    param_key_coverage_count = 0
    shape_mismatch: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    missing: List[Tuple[str, int, int]] = []  # (name, numel, dim)

    for name, p in model_params:
        n = int(p.numel())
        d = int(p.dim())
        numel_total_model += n
        if d == 2:
            numel_total_2d += n

        if name in masks_dict:
            param_key_coverage_count += 1
            m = masks_dict[name]
            if tuple(m.shape) != tuple(p.shape):
                shape_mismatch.append((name, tuple(p.shape), tuple(m.shape)))
                continue
            numel_covered_total += n
            if d == 2:
                numel_covered_2d += n
        else:
            missing.append((name, n, d))

    missing_sorted = sorted(missing, key=lambda x: x[1], reverse=True)[: max(0, int(topk_missing))]

    report: Dict[str, Any] = {
        "param_key_coverage_count": int(param_key_coverage_count),
        "param_key_coverage_frac": float(param_key_coverage_count / max(1, total_params)),
        "numel_covered_total": int(numel_covered_total),
        "numel_total_model": int(numel_total_model),
        "numel_covered_frac_total": float(numel_covered_total / max(1, numel_total_model)),
        "numel_covered_2d": int(numel_covered_2d),
        "numel_total_2d": int(numel_total_2d),
        "numel_covered_frac_2d": float(numel_covered_2d / max(1, numel_total_2d)),
        "shape_mismatch_count": int(len(shape_mismatch)),
        "shape_mismatches": [
            {"name": n, "param_shape": list(ps), "mask_shape": list(ms)}
            for (n, ps, ms) in shape_mismatch
        ],
        "missing_topk_by_numel": [
            {"name": n, "numel": int(numel), "dim": int(dim)}
            for (n, numel, dim) in missing_sorted
        ],
    }
    return report


def write_coverage_report(path: str, report: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

