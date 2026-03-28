import torch
import os
import json
from typing import Any, Dict

# Default masking mode:
# - global ranking across all scored weights
# - plus a small per-tensor keep floor to reduce collapse under high sparsity
DEFAULT_MIN_LAYER_KEEP_RATIO = 0.0025

# Global top-k over ~5B+ scores exceeds 2^32 elements; CUDA topk (gatherTopK) can assert
# on such 1D tensors. Use CPU top-k above this threshold unless RL_CASINO_MASK_TOPK_ALLOW_GPU=1.
_CUDA_TOPK_SAFE_NUMEL = 2_000_000_000


def pooling_metadata(
    *,
    local_pool: bool,
    min_layer_keep_ratio: float = DEFAULT_MIN_LAYER_KEEP_RATIO,
) -> Dict[str, Any]:
    """
    Fields to merge into saved mask metadata so runs are reproducible and comparable.

    - pooling_mode ``global``: one ranking across all scored weight elements.
    - pooling_mode ``global_with_layer_floor``: global top-k after reserving a per-layer
      keep floor (the default when ``min_layer_keep_ratio`` is left unchanged).
    - pooling_mode ``local_per_tensor``: each 2D weight matrix keeps keep_frac of its
      own elements independently (``--local_pool``).

    Random baselines: ``generate_random_mask.py`` uses a single global threshold on
    concatenated scores; ``random_mask_baseline.py`` uses
    ``create_mask_from_scores_gpu_efficient``.

    For very large models, materializing ``torch.cat`` over all scores may be expensive;
    future work: threshold search with per-chunk counts (exact global sparsity) or
    oversampled per-block top-k then a final global top-k (approximate unless oversample
    is large enough to include all true global top-k).
    """
    if local_pool:
        return {"pooling_mode": "local_per_tensor", "local_pool": True}
    if min_layer_keep_ratio > 0:
        return {
            "pooling_mode": "global_with_layer_floor",
            "local_pool": False,
            "min_layer_keep_ratio": float(min_layer_keep_ratio),
        }
    return {"pooling_mode": "global", "local_pool": False}


def _topk_indices_safe(scores_1d: torch.Tensor, k: int, largest: bool = True) -> torch.Tensor:
    """Top-k indices; CPU fallback for very large tensors to avoid CUDA gatherTopK failures."""
    n = scores_1d.numel()
    k = max(0, min(int(k), n))
    if k == 0:
        return scores_1d.new_empty((0,), dtype=torch.long)
    allow_gpu = os.environ.get("RL_CASINO_MASK_TOPK_ALLOW_GPU", "").lower() in ("1", "true", "yes")
    force_cpu = os.environ.get("RL_CASINO_MASK_TOPK_CPU", "").lower() in ("1", "true", "yes")
    use_cpu = scores_1d.is_cuda and (force_cpu or (n > _CUDA_TOPK_SAFE_NUMEL and not allow_gpu))
    if use_cpu:
        print(
            f"  Note: top-k on CPU ({n:,} scores, k={k:,}) — "
            "CUDA top-k can fail when the score vector is extremely large."
        )
        idx = torch.topk(scores_1d.detach().cpu(), k=k, largest=largest, sorted=False).indices
        return idx.to(scores_1d.device, non_blocking=False)
    return torch.topk(scores_1d, k=k, largest=largest, sorted=False).indices


def _create_mask_local(scores_dict, sparsity_percent, device, add_tie_break_noise, tie_break_noise_scale):
    """Per-layer top-k: each weight matrix independently keeps keep_frac of its elements."""
    keep_frac = 1.0 - sparsity_percent / 100.0
    print(f"\n=== Creating Per-layer Local Masks (target sparsity: {sparsity_percent}%) ===")

    masks = {}
    total_params = 0
    total_kept = 0

    for name, score in scores_dict.items():
        if score is None or score.numel() == 0:
            masks[name] = torch.zeros_like(score, dtype=torch.float32).cpu()
            continue

        s = score.to(device=device, dtype=torch.float32)
        n = s.numel()
        n_keep = max(1, int(keep_frac * n))

        flat = s.reshape(-1)
        if add_tie_break_noise:
            scale = max(flat.abs().max().item() * tie_break_noise_scale, 1e-12)
            flat = flat + torch.randn_like(flat) * scale

        idx = _topk_indices_safe(flat, k=n_keep, largest=True)
        mask_flat = torch.zeros(n, device=device, dtype=torch.float32)
        mask_flat[idx] = 1.0

        masks[name] = mask_flat.reshape(score.shape).cpu()
        total_params += n
        total_kept += n_keep

    actual_sparsity = 100.0 - (total_kept / max(total_params, 1) * 100.0)
    print(f"Total parameters: {total_params:,}")
    print(f"Actual keep: {total_kept:,} | Actual sparsity: {actual_sparsity:.4f}%")
    return masks


def create_mask_from_scores_gpu_efficient(
    scores_dict,
    sparsity_percent,
    device='cuda',
    add_tie_break_noise: bool = True,
    tie_break_noise_scale: float = 1e-10,
    min_layer_keep_ratio: float = DEFAULT_MIN_LAYER_KEEP_RATIO,
    local_pool: bool = False,
):
    """
    Create sparse masks from score tensors.

        Default (local_pool=False): *global* ranking with a small per-tensor keep floor,
        so high-scoring layers still compete globally but each scored tensor keeps at least
        a small fraction of weights unless the global budget is too small.

        local_pool=True: *per-layer* ranking — each weight matrix independently
        keeps its top keep_frac elements, giving uniform sparsity per layer.

        Optional hybrid mode (local_pool=False only):
            - If min_layer_keep_ratio > 0, each non-empty layer keeps at least
                floor(min_layer_keep_ratio * layer_numel) parameters.
            - Remaining budget is allocated by global top-k over the full model.
            - Pass min_layer_keep_ratio=0.0 for pure global selection with no floor.
    """
    if local_pool:
        print("Mask pooling: local (each weight matrix ranked independently; use --local_pool)")
        return _create_mask_local(scores_dict, sparsity_percent, device, add_tie_break_noise, tie_break_noise_scale)

    if min_layer_keep_ratio > 0:
        print(
            "Mask pooling: global with per-layer keep floor "
            f"(min_layer_keep_ratio={min_layer_keep_ratio}; remaining budget is global top-k)"
        )
    else:
        print("Mask pooling: global (single ranking across all scored weights)")

    print(f"\n=== Creating Exact Global Masks (target sparsity: {sparsity_percent}%) ===")

    if not scores_dict:
        raise ValueError(
            "create_mask_from_scores_gpu_efficient received an empty scores_dict. "
            "Upstream scoring produced no valid weight-score tensors."
        )

    keep_percent = 100.0 - sparsity_percent

    # Normalize tensors onto target device and collect valid entries.
    valid_scores = {}
    total_params = 0
    for name, score in scores_dict.items():
        if score is None or score.numel() == 0:
            continue
        s = score.to(device=device, dtype=torch.float32)
        valid_scores[name] = s
        total_params += s.numel()

    if not valid_scores or total_params == 0:
        raise ValueError(
            "No non-empty score tensors were available for global ranking. "
            "Check upstream scoring/mapping logic."
        )

    keep_count = max(1, int(keep_percent / 100.0 * total_params))
    keep_count = min(keep_count, total_params)

    print(f"Total parameters: {total_params:,}")
    print(f"Target keep count: {keep_count:,} ({keep_percent:.2f}%)")
    if min_layer_keep_ratio > 0:
        print(f"Using hybrid global mask with per-layer keep floor ratio={min_layer_keep_ratio:.4f}")

    # Flatten all scores globally.
    offsets = []
    flat_chunks = []
    cursor = 0
    for name, s in valid_scores.items():
        n = s.numel()
        offsets.append((name, cursor, cursor + n, s.shape))
        flat_chunks.append(s.reshape(-1))
        cursor += n

    all_scores = torch.cat(flat_chunks, dim=0)
    all_scores = torch.nan_to_num(all_scores, nan=0.0, posinf=0.0, neginf=0.0)

    if add_tie_break_noise:
        # Very small jitter for deterministic tie-breaking when many scores are equal.
        scale = max(all_scores.abs().max().item() * tie_break_noise_scale, 1e-12)
        all_scores = all_scores + torch.randn_like(all_scores) * scale
        print(f"Applied tie-break noise (scale={scale:.3e})")

    global_mask_flat = torch.zeros_like(all_scores, dtype=torch.float32)

    # Optional per-layer minimum keep floor.
    floor_selected = 0
    if min_layer_keep_ratio > 0:
        min_layer_keep_ratio = float(max(0.0, min(1.0, min_layer_keep_ratio)))
        layer_floors = []
        for name, start, end, _shape in offsets:
            layer_n = end - start
            local_floor = int(min_layer_keep_ratio * layer_n)
            local_floor = max(0, min(local_floor, layer_n))
            layer_floors.append((name, start, end, local_floor))

        requested_floor_total = sum(f for _, _, _, f in layer_floors)
        if requested_floor_total > keep_count:
            print(
                f"⚠ Requested per-layer floor keeps {requested_floor_total:,} params, "
                f"but global keep budget is {keep_count:,}. Scaling floors down proportionally."
            )
            scale = keep_count / max(1, requested_floor_total)
            scaled_floors = []
            for name, start, end, f in layer_floors:
                layer_n = end - start
                nf = int(f * scale)
                nf = max(0, min(nf, layer_n))
                scaled_floors.append((name, start, end, nf))
            layer_floors = scaled_floors

        for name, start, end, local_floor in layer_floors:
            if local_floor <= 0:
                continue
            layer_scores = all_scores[start:end]
            local_idx = _topk_indices_safe(layer_scores, k=local_floor, largest=True)
            global_mask_flat[start:end][local_idx] = 1.0
            floor_selected += int(local_floor)

        print(f"Per-layer floor selected: {floor_selected:,} parameters")

    # Fill remaining budget via exact global top-k among unselected positions.
    remaining = keep_count - floor_selected
    if remaining > 0:
        # Exclude already selected floor positions by setting scores to -inf.
        candidate_scores = all_scores.clone()
        candidate_scores[global_mask_flat > 0] = float("-inf")
        keep_indices = _topk_indices_safe(candidate_scores, k=remaining, largest=True)
        global_mask_flat[keep_indices] = 1.0

    masks = {}
    total_kept = 0
    print("Applying global mask to layers...")
    for idx, (name, start, end, shape) in enumerate(offsets):
        if idx % 50 == 0:
            print(f"  Processing layer {idx+1}/{len(offsets)}")
        m = global_mask_flat[start:end].reshape(shape)
        total_kept += int(m.sum().item())
        masks[name] = m.cpu()

    # Pass through empty entries (if any) as all-zero masks to preserve keys.
    for name, score in scores_dict.items():
        if name in masks:
            continue
        if score is None:
            continue
        masks[name] = torch.zeros_like(score, dtype=torch.float32).cpu()

    actual_sparsity = 100.0 - (total_kept / total_params * 100.0)
    print("\nVerification:")
    print(f"  Target keep: {keep_count:,} ({keep_percent:.2f}%)")
    print(f"  Actual keep: {int(total_kept):,} ({100.0 - actual_sparsity:.2f}%)")
    print(f"  Actual sparsity: {actual_sparsity:.4f}% (target: {sparsity_percent}%)")
    print(f"  Error: {abs(actual_sparsity - sparsity_percent):.6f}%")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return masks


def compute_jaccard_similarity(pred_masks, true_masks):
    """
    Computes Jaccard similarity, spatial overlap, and cosine similarity
    between predicted and reference masks. Uses GPU for faster computation.
    """
    print("\n=== Computing Similarity Metrics ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    per_layer_jaccard = {}
    total_intersection = 0
    total_union = 0
    total_pred_kept = 0
    total_true_kept = 0
    
    for name in pred_masks.keys():
        if name not in true_masks:
            continue
        
        pred = pred_masks[name].to(device).bool()
        true = true_masks[name].to(device).bool()
        
        intersection = (pred & true).sum().item()
        union = (pred | true).sum().item()
        
        jaccard = intersection / union if union > 0 else 0.0
        per_layer_jaccard[name] = jaccard
        
        total_intersection += intersection
        total_union += union
        total_pred_kept += pred.sum().item()
        total_true_kept += true.sum().item()
        
        del pred, true
    
    torch.cuda.empty_cache()
    
    aggregate_jaccard = total_intersection / total_union if total_union > 0 else 0.0
    
    # Calculate additional metrics
    overlap_pred = total_intersection / total_pred_kept if total_pred_kept > 0 else 0.0
    overlap_true = total_intersection / total_true_kept if total_true_kept > 0 else 0.0
    cosine_similarity = total_intersection / ((total_pred_kept * total_true_kept) ** 0.5) if (total_pred_kept * total_true_kept) > 0 else 0.0
    
    if len(per_layer_jaccard) > 0:
        mean_jaccard = sum(per_layer_jaccard.values()) / len(per_layer_jaccard)
        min_jaccard = min(per_layer_jaccard.values())
        max_jaccard = max(per_layer_jaccard.values())
        
        print(f"Aggregate Jaccard Similarity:     {aggregate_jaccard:.4f}")
        print(f"Overlap (Intersect / Pred_Size):  {overlap_pred:.4f} ({overlap_pred*100:.1f}%)")
        print(f"Overlap (Intersect / True_Size):  {overlap_true:.4f} ({overlap_true*100:.1f}%)")
        print(f"Global Cosine Similarity:         {cosine_similarity:.4f}")
        print(f"Mean per-layer Jaccard:           {mean_jaccard:.4f}")
        
        return {
            "aggregate_jaccard": aggregate_jaccard,
            "mean_jaccard": mean_jaccard,
            "min_jaccard": min_jaccard,
            "max_jaccard": max_jaccard,
            "per_layer": per_layer_jaccard,
            "overlap_fraction_predicted": overlap_pred,
            "overlap_fraction_reference": overlap_true,
            "cosine_similarity": cosine_similarity
        }
    else:
        print("No matching layers found for similarity computation.")
        return None

def save_masks(masks, output_file, metadata=None):
    """Saves masks with optional metadata."""
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    save_dict = {"masks": masks}
    if metadata:
        save_dict["metadata"] = metadata
    
    torch.save(save_dict, output_file)
    print(f"\nMasks saved to: {output_file}")
    
    total_params = sum(m.numel() for m in masks.values())
    kept_params = sum(m.sum().item() for m in masks.values())
    actual_sparsity = 100.0 - (kept_params / total_params * 100)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Kept parameters: {int(kept_params):,}")
    print(f"Final sparsity: {actual_sparsity:.2f}%")
