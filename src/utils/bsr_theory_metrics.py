"""
Theoretical proxies for BSR backward FLOPs and SparseAdamW VRAM traffic.

Conventions match ``src/analysis/sparse_training_complexity.md``:
  - §1b: per masked linear, sparse backward FLOPs ~ A_l * B * D_in * D_out with A_l = active fraction.
  - §2c: SparseAdamW memory movement ~ 28 * 4 * (A * P) = 112 bytes per active element (summed).

``B`` is an approximate token count per optimizer step (microbatch matmul contraction).
This does **not** include the dense forward pass on full weights (DPO still runs dense forwards).
"""

from __future__ import annotations

from typing import Any, Dict, Union

import torch

Tensor = torch.Tensor


def default_b_tokens_proxy(
    *,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    chosen_rejected_pairs: bool = True,
) -> int:
    """
    Upper-bound style token count for matmuls: batch * accum * (2 if DPO pair) * max_length.
    """
    b = int(per_device_train_batch_size) * int(gradient_accumulation_steps) * int(max_length)
    return b * (2 if chosen_rejected_pairs else 1)


def compute_sparse_mask_theory_metrics(
    masks: Dict[str, Tensor],
    b_tokens: int,
) -> Dict[str, Union[int, float, str]]:
    """
    Aggregate proxies over 2D weight masks (bool or float).

    Returns JSON/CSV-friendly scalars.
    """
    if not masks:
        return {
            "theory_b_tokens_proxy": int(b_tokens),
            "theory_active_weight_count": 0,
            "theory_masked_param_elements": 0,
            "theory_active_fraction_global": 0.0,
            "theory_bsr_backward_flops_proxy": 0.0,
            "theory_dense_backward_flops_masked_layers_proxy": 0.0,
            "theory_sparse_adamw_optimizer_bytes_proxy": 0,
            "theory_doc_ref": "src/analysis/sparse_training_complexity.md §1b §2c",
        }

    b = float(max(int(b_tokens), 1))
    active_total = 0
    p_total = 0
    bsr_flops = 0.0
    dense_flops = 0.0

    for m in masks.values():
        if m.dim() != 2:
            continue
        d_out, d_in = int(m.shape[0]), int(m.shape[1])
        p = d_in * d_out
        mt = m.bool() if m.dtype != torch.bool else m
        active = int(mt.sum().item())
        a = active / float(p) if p > 0 else 0.0
        active_total += active
        p_total += p
        # §1b: O(A * B * Din * Dout)
        bsr_flops += a * b * float(d_in) * float(d_out)
        dense_flops += b * float(d_in) * float(d_out)

    frac = (active_total / float(p_total)) if p_total > 0 else 0.0
    # §2c: 7 reads/writes * 4 bytes per active element
    adam_bytes = 28 * 4 * active_total

    return {
        "theory_b_tokens_proxy": int(b_tokens),
        "theory_active_weight_count": active_total,
        "theory_masked_param_elements": p_total,
        "theory_active_fraction_global": round(frac, 8),
        "theory_bsr_backward_flops_proxy": round(bsr_flops, 6),
        "theory_dense_backward_flops_masked_layers_proxy": round(dense_flops, 6),
        "theory_sparse_adamw_optimizer_bytes_proxy": int(adam_bytes),
        "theory_doc_ref": "src/analysis/sparse_training_complexity.md §1b §2c",
    }


def dense_phase_theory_stub(
    *,
    b_tokens: int,
) -> Dict[str, Any]:
    """Placeholder row for dense baseline (no sparse mask): same B proxy, no mask aggregates."""
    return {
        "theory_b_tokens_proxy": int(b_tokens),
        "theory_active_weight_count": "",
        "theory_masked_param_elements": "",
        "theory_active_fraction_global": "",
        "theory_bsr_backward_flops_proxy": "",
        "theory_dense_backward_flops_masked_layers_proxy": "",
        "theory_sparse_adamw_optimizer_bytes_proxy": "",
        "theory_doc_ref": "dense_baseline_no_bsr_mask",
    }
