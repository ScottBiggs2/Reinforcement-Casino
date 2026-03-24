import torch
import os
import json

def create_mask_from_scores_gpu_efficient(
    scores_dict,
    sparsity_percent,
    device='cuda',
    add_tie_break_noise: bool = True,
    tie_break_noise_scale: float = 1e-10,
    min_layer_keep_ratio: float = 0.0,
):
    """
    Create exact global top-k masks from score tensors.

        IMPORTANT: this function performs *global* ranking across all provided
        parameter tensors.

        Optional hybrid mode:
            - If min_layer_keep_ratio > 0, each non-empty layer keeps at least
                floor(min_layer_keep_ratio * layer_numel) parameters.
            - Remaining budget is allocated by global top-k over the full model.
    """
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
            local_idx = torch.topk(layer_scores, k=local_floor, largest=True, sorted=False).indices
            global_mask_flat[start:end][local_idx] = 1.0
            floor_selected += int(local_floor)

        print(f"Per-layer floor selected: {floor_selected:,} parameters")

    # Fill remaining budget via exact global top-k among unselected positions.
    remaining = keep_count - floor_selected
    if remaining > 0:
        # Exclude already selected floor positions by setting scores to -inf.
        candidate_scores = all_scores.clone()
        candidate_scores[global_mask_flat > 0] = float("-inf")
        keep_indices = torch.topk(candidate_scores, k=remaining, largest=True, sorted=False).indices
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
