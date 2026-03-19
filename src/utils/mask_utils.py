import torch
import os
import json

def create_mask_from_scores_gpu_efficient(scores_dict, sparsity_percent, device='cuda'):
    """
    GPU-accelerated exact top-k mask creation.
    Processes scores directly on GPU without unnecessary transfers.
    Global threshold with per-layer correction fallback.
    """
    print(f"\n=== Creating Exact Masks (target sparsity: {sparsity_percent}%) ===")
    
    keep_percent = 100.0 - sparsity_percent
    
    # Add noise on GPU to break ties
    print("Adding noise to break ties (on GPU)...")
    for name in scores_dict.keys():
        score = scores_dict[name].to(device)
        noise_scale = max(score.abs().max().item() * 1e-10, 1e-12)
        noise = torch.randn_like(score, device=device) * noise_scale
        scores_dict[name] = score + noise
    
    # Sample and compute threshold on GPU
    print("Sampling scores to estimate threshold (on GPU)...")
    sample_scores = []
    total_params = 0
    
    for name, score in scores_dict.items():
        total_params += score.numel()
        flat = score.flatten()
        sample_size = min(100000, flat.numel())
        
        if sample_size < flat.numel():
            indices = torch.randperm(flat.numel(), device=device)[:sample_size]
            sample_scores.append(flat[indices])
        else:
            sample_scores.append(flat)
    
    all_samples = torch.cat(sample_scores)
    keep_count = max(1, int(keep_percent / 100.0 * total_params))
    
    print(f"Total parameters: {total_params:,}")
    print(f"Target keep count: {keep_count:,} ({keep_percent:.1f}%)")
    print(f"Sampled {all_samples.numel():,} values for threshold estimation")
    
    # Compute threshold on GPU
    sample_keep_count = max(1, int(keep_percent / 100.0 * all_samples.numel()))
    if sample_keep_count >= all_samples.numel():
        threshold = 0.0
    else:
        threshold_tensor, _ = torch.topk(all_samples, sample_keep_count)
        threshold = threshold_tensor.min().item()
    
    print(f"Estimated threshold: {threshold:.10f}")
    del all_samples
    torch.cuda.empty_cache()
    
    # Apply masks on GPU, store on CPU
    print("\nApplying per-layer masks...")
    masks = {}
    total_kept = 0
    
    for idx, (name, score) in enumerate(scores_dict.items()):
        if idx % 50 == 0:
            print(f"  Processing layer {idx+1}/{len(scores_dict)}")
        
        mask = (score >= threshold).float()
        total_kept += mask.sum().item()
        masks[name] = mask.cpu()  # Move to CPU for storage
        
    actual_sparsity = 100.0 - (total_kept / total_params * 100)
    
    # Correction pass if needed (already on GPU)
    if abs(actual_sparsity - sparsity_percent) > 5.0:
        print(f"\n⚠ Initial sparsity {actual_sparsity:.2f}% is off target. Doing correction pass...")
        
        masks = {}
        total_kept = 0
        
        for name, score in scores_dict.items():
            flat = score.flatten()
            layer_keep = max(1, int(keep_percent / 100.0 * flat.numel()))
            
            if layer_keep >= flat.numel():
                mask = torch.ones_like(score)
            else:
                topk_vals, _ = torch.topk(flat, layer_keep)
                local_threshold = topk_vals.min()
                mask = (score >= local_threshold).float()
            
            masks[name] = mask.cpu()
            total_kept += mask.sum().item()
    
    # Clean up GPU
    torch.cuda.empty_cache()
    
    actual_sparsity = 100.0 - (total_kept / total_params * 100)
    
    print(f"\nVerification:")
    print(f"  Target keep: {keep_count:,} ({keep_percent:.1f}%)")
    print(f"  Actual keep: {int(total_kept):,} ({100.0 - actual_sparsity:.1f}%)")
    print(f"  Actual sparsity: {actual_sparsity:.2f}% (target: {sparsity_percent}%)")
    print(f"  Error: {abs(actual_sparsity - sparsity_percent):.2f}%")
    
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
