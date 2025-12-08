import torch
import os
import argparse
from collections import defaultdict
import json

def load_deltas_from_log_dir(delta_log_dir, target_step=None):
    """
    Loads delta files from the new delta_logs format.
    If target_step is specified, loads only up to that step.
    
    Returns:
        dict: {step: {param_name: delta_tensor}}
    """
    print(f"Loading deltas from: {delta_log_dir}")
    
    deltas_by_step = {}
    
    # Find all delta files and sort by step number
    delta_files = [f for f in os.listdir(delta_log_dir) if f.startswith("deltas_step_") and f.endswith(".pt")]
    
    # Sort by step number, not filename
    def extract_step(filename):
        return int(filename.split("_")[-1].replace(".pt", ""))
    
    delta_files = sorted(delta_files, key=extract_step)
    
    for delta_file in delta_files:
        step = extract_step(delta_file)
        
        # Skip if beyond target step
        if target_step is not None and step > target_step:
            continue
            
        file_path = os.path.join(delta_log_dir, delta_file)
        try:
            deltas = torch.load(file_path, map_location='cpu')
            deltas_by_step[step] = deltas
            print(f"Loaded deltas for step {step} ({len(deltas)} parameters)")
        except Exception as e:
            print(f"Warning: Could not load {delta_file}: {e}")
            continue
    
    if not deltas_by_step:
        print("No delta files found!")
        return None
        
    return deltas_by_step


def compute_ground_truth_mask(deltas_by_step, sparsity_percent, device='cuda'):
    """
    Computes the 'ground truth' mask using the final checkpoint's absolute changes.
    This represents what we're trying to predict early in training.
    
    Args:
        deltas_by_step: Dictionary of deltas
        sparsity_percent: Target sparsity (% of weights to mask OUT)
        device: Device for computation
    
    Returns:
        dict: Ground truth masks (1 = keep, 0 = mask out)
    """
    print(f"\n=== Computing Ground Truth Mask (target sparsity: {sparsity_percent}%) ===")
    
    # Use final checkpoint
    final_step = max(deltas_by_step.keys())
    final_deltas = deltas_by_step[final_step]
    
    print(f"Using final checkpoint at step {final_step}")
    
    masks = {}
    total_params = 0
    total_kept = 0
    
    keep_percent = 100.0 - sparsity_percent  # If 90% sparsity, keep 10%
    
    for idx, (name, delta) in enumerate(final_deltas.items()):
        if idx % 20 == 0:
            print(f"  Processing parameter {idx+1}/{len(final_deltas)}")
        
        # Move to GPU for fast sorting
        delta_gpu = delta.abs().to(device)
        flat_changes = delta_gpu.flatten()
        
        # Calculate how many to keep
        k = max(1, int(keep_percent / 100.0 * flat_changes.numel()))
        
        if k >= flat_changes.numel():
            threshold = 0.0
        else:
            # Get the k-th largest value
            threshold = torch.kthvalue(flat_changes, flat_changes.numel() - k).values
        
        # Mask: 1 = keep, 0 = mask out
        mask = (delta_gpu >= threshold).float().cpu()
        masks[name] = mask
        
        total_params += mask.numel()
        total_kept += mask.sum().item()
        
        del delta_gpu, flat_changes
        torch.cuda.empty_cache()
    
    actual_sparsity = 100.0 - (total_kept / total_params * 100) if total_params > 0 else 0
    print(f"Ground truth - Total params: {total_params:,}, Kept: {total_kept:,}")
    print(f"Ground truth - Actual sparsity: {actual_sparsity:.2f}% (target: {sparsity_percent}%)")
    
    return masks


def compute_jaccard_similarity(pred_masks, true_masks):
    """
    Computes Jaccard similarity between predicted and ground truth masks.
    
    Jaccard = |intersection| / |union|
    
    Returns:
        dict: Per-layer and aggregate Jaccard scores
    """
    print("\n=== Computing Jaccard Similarity ===")
    
    per_layer_jaccard = {}
    total_intersection = 0
    total_union = 0
    
    for name in pred_masks.keys():
        if name not in true_masks:
            continue
        
        pred = pred_masks[name].bool()
        true = true_masks[name].bool()
        
        intersection = (pred & true).sum().item()
        union = (pred | true).sum().item()
        
        jaccard = intersection / union if union > 0 else 0.0
        per_layer_jaccard[name] = jaccard
        
        total_intersection += intersection
        total_union += union
    
    # Aggregate Jaccard
    aggregate_jaccard = total_intersection / total_union if total_union > 0 else 0.0
    
    print(f"Aggregate Jaccard Similarity: {aggregate_jaccard:.4f}")
    print(f"Mean per-layer Jaccard: {sum(per_layer_jaccard.values()) / len(per_layer_jaccard):.4f}")
    print(f"Min per-layer Jaccard: {min(per_layer_jaccard.values()):.4f}")
    print(f"Max per-layer Jaccard: {max(per_layer_jaccard.values()):.4f}")
    
    return {
        "aggregate_jaccard": aggregate_jaccard,
        "mean_jaccard": sum(per_layer_jaccard.values()) / len(per_layer_jaccard),
        "min_jaccard": min(per_layer_jaccard.values()),
        "max_jaccard": max(per_layer_jaccard.values()),
        "per_layer": per_layer_jaccard
    }


def compute_absolute_magnitude_mask(deltas_by_step, sparsity_percent, device='cuda'):
    """
    GPU-accelerated magnitude mask computation.
    
    Args:
        sparsity_percent: Target sparsity (% of weights to MASK OUT)
    """
    print(f"\n=== Computing Absolute Magnitude Mask (target sparsity: {sparsity_percent}%) ===")
    
    aggregated = defaultdict(lambda: None)
    
    # Aggregate absolute changes
    for step, deltas in deltas_by_step.items():
        print(f"  Processing step {step}...")
        for name, delta in deltas.items():
            if aggregated[name] is None:
                aggregated[name] = torch.zeros_like(delta)
            aggregated[name] += delta.abs()
    
    # Create masks ON GPU
    print("Creating masks on GPU...")
    masks = {}
    total_params = 0
    total_kept = 0
    
    keep_percent = 100.0 - sparsity_percent  # If 90% sparsity, keep 10%
    
    for idx, (name, total_change) in enumerate(aggregated.items()):
        if idx % 20 == 0:
            print(f"  Creating mask {idx+1}/{len(aggregated)}")
        
        # Move to GPU for fast sorting
        total_change_gpu = total_change.to(device)
        flat_changes = total_change_gpu.flatten()
        
        k = max(1, int(keep_percent / 100.0 * flat_changes.numel()))
        
        if k >= flat_changes.numel():
            threshold = 0.0
        else:
            # kthvalue is FAST on GPU
            threshold = torch.kthvalue(flat_changes, flat_changes.numel() - k).values
        
        mask = (total_change_gpu >= threshold).float().cpu()  # Move back to CPU for storage
        masks[name] = mask
        
        total_params += mask.numel()
        total_kept += mask.sum().item()
        
        # Clean up GPU memory
        del total_change_gpu, flat_changes
        torch.cuda.empty_cache()
    
    actual_sparsity = 100.0 - (total_kept / total_params * 100) if total_params > 0 else 0
    print(f"Total params: {total_params:,}, Kept: {total_kept:,}")
    print(f"Actual sparsity: {actual_sparsity:.2f}% (target: {sparsity_percent}%)")
    
    return masks


def compute_momentum_mask(deltas_by_step, sparsity_percent, window_size=5, device='cuda'):
    """
    GPU-accelerated momentum mask computation.
    Uses recent velocity consistency to identify important weights.
    
    Args:
        sparsity_percent: Target sparsity (% of weights to MASK OUT)
    """
    print(f"\n=== Computing Momentum-Based Mask (target sparsity: {sparsity_percent}%, window={window_size}) ===")
    
    sorted_steps = sorted(deltas_by_step.keys())
    
    if len(sorted_steps) < 2:
        print("Warning: Need at least 2 steps for momentum calculation. Falling back to magnitude.")
        return compute_absolute_magnitude_mask(deltas_by_step, sparsity_percent, device)
    
    # Compute velocity (change between consecutive steps)
    print("Computing velocities...")
    velocities_by_step = {}
    for i in range(1, len(sorted_steps)):
        prev_step = sorted_steps[i-1]
        curr_step = sorted_steps[i]
        
        velocities = {}
        for name in deltas_by_step[curr_step].keys():
            if name in deltas_by_step[prev_step]:
                velocities[name] = deltas_by_step[curr_step][name] - deltas_by_step[prev_step][name]
        
        velocities_by_step[curr_step] = velocities
    
    # Compute momentum scores
    print("Computing momentum scores on GPU...")
    momentum_scores = {}
    
    for idx, name in enumerate(deltas_by_step[sorted_steps[-1]].keys()):
        if idx % 20 == 0:
            print(f"  Processing parameter {idx+1}/{len(deltas_by_step[sorted_steps[-1]])}")
        
        # Get recent velocities
        recent_velocities = []
        for step in sorted_steps[-window_size:]:
            if step in velocities_by_step and name in velocities_by_step[step]:
                recent_velocities.append(velocities_by_step[step][name])
        
        if len(recent_velocities) < 2:
            # Fallback to magnitude if insufficient velocity data
            momentum_scores[name] = deltas_by_step[sorted_steps[-1]][name].abs()
            continue
        
        # Stack and compute ON GPU
        vel_stack = torch.stack(recent_velocities).to(device)
        
        mean_velocity = vel_stack.mean(dim=0)
        std_velocity = vel_stack.std(dim=0) + 1e-8
        
        # Consistency score: high when velocity is consistent (low variance)
        consistency = mean_velocity.abs() / std_velocity
        magnitude = mean_velocity.abs()
        
        # Combined score: favor weights with large, consistent movement
        score = (magnitude * (1 + consistency)).cpu()
        momentum_scores[name] = score
        
        del vel_stack
        torch.cuda.empty_cache()
    
    # Create masks ON GPU
    print("Creating masks on GPU...")
    masks = {}
    total_params = 0
    total_kept = 0
    
    keep_percent = 100.0 - sparsity_percent
    
    for idx, (name, score) in enumerate(momentum_scores.items()):
        if idx % 20 == 0:
            print(f"  Creating mask {idx+1}/{len(momentum_scores)}")
        
        score_gpu = score.to(device)
        flat_scores = score_gpu.flatten()
        
        k = max(1, int(keep_percent / 100.0 * flat_scores.numel()))
        
        if k >= flat_scores.numel():
            threshold = 0.0
        else:
            threshold = torch.kthvalue(flat_scores, flat_scores.numel() - k).values
        
        mask = (score_gpu >= threshold).float().cpu()
        masks[name] = mask
        
        total_params += mask.numel()
        total_kept += mask.sum().item()
        
        del score_gpu, flat_scores
        torch.cuda.empty_cache()
    
    actual_sparsity = 100.0 - (total_kept / total_params * 100) if total_params > 0 else 0
    print(f"Total params: {total_params:,}, Kept: {total_kept:,}")
    print(f"Actual sparsity: {actual_sparsity:.2f}% (target: {sparsity_percent}%)")
    
    return masks


def compute_fisher_mask(delta_log_dir, base_state_path, sparsity_percent, target_step=None, device='cuda'):
    """
    GPU-accelerated Fisher approximation.
    
    Args:
        sparsity_percent: Target sparsity (% of weights to MASK OUT)
    """
    print(f"\n=== Computing Fisher-Approximation Mask (target sparsity: {sparsity_percent}%) ===")
    print("Note: This is an approximation since we don't have true gradients.")
    
    # Load base state
    if not os.path.exists(base_state_path):
        print(f"Error: Base state not found at {base_state_path}")
        return None
    
    base_state = torch.load(base_state_path, map_location='cpu')
    print(f"Loaded base state with {len(base_state)} parameters")
    
    # Load deltas
    deltas_by_step = load_deltas_from_log_dir(delta_log_dir, target_step)
    if not deltas_by_step:
        return None
    
    # Approximate Fisher: F ≈ E[g²] where g is gradient
    print("Computing Fisher scores on GPU...")
    fisher_scores = {}
    
    for idx, name in enumerate(base_state.keys()):
        if idx % 20 == 0:
            print(f"  Processing parameter {idx+1}/{len(base_state)}")
        
        # Collect all deltas for this parameter
        deltas = []
        for step, step_deltas in deltas_by_step.items():
            if name in step_deltas:
                deltas.append(step_deltas[name])
        
        if not deltas:
            fisher_scores[name] = torch.zeros_like(base_state[name])
            continue
        
        # Stack deltas and compute variance ON GPU
        delta_stack = torch.stack(deltas).to(device)
        
        # Fisher ≈ variance of pseudo-gradients
        fisher_approx = (delta_stack.var(dim=0) + delta_stack.mean(dim=0).abs()).cpu()
        
        fisher_scores[name] = fisher_approx
        
        del delta_stack
        torch.cuda.empty_cache()
    
    # Create masks ON GPU
    print("Creating masks on GPU...")
    masks = {}
    total_params = 0
    total_kept = 0
    
    keep_percent = 100.0 - sparsity_percent
    
    for idx, (name, score) in enumerate(fisher_scores.items()):
        if idx % 20 == 0:
            print(f"  Creating mask {idx+1}/{len(fisher_scores)}")
        
        score_gpu = score.to(device)
        flat_scores = score_gpu.flatten()
        
        k = max(1, int(keep_percent / 100.0 * flat_scores.numel()))
        
        if k >= flat_scores.numel():
            threshold = 0.0
        else:
            threshold = torch.kthvalue(flat_scores, flat_scores.numel() - k).values
        
        mask = (score_gpu >= threshold).float().cpu()
        masks[name] = mask
        
        total_params += mask.numel()
        total_kept += mask.sum().item()
        
        del score_gpu, flat_scores
        torch.cuda.empty_cache()
    
    actual_sparsity = 100.0 - (total_kept / total_params * 100) if total_params > 0 else 0
    print(f"Total params: {total_params:,}, Kept: {total_kept:,}")
    print(f"Actual sparsity: {actual_sparsity:.2f}% (target: {sparsity_percent}%)")
    
    return masks


def save_masks(masks, output_file, metadata=None):
    """Saves masks with optional metadata."""
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
    print(f"Kept parameters: {kept_params:,}")
    print(f"Final sparsity: {actual_sparsity:.2f}%")


def verify_masks(masks, delta_log_dir):
    """Verify mask shapes match delta shapes."""
    print("\n=== Verifying Masks ===")
    
    # Load one set of deltas to check shapes
    delta_files = [f for f in os.listdir(delta_log_dir) if f.startswith("deltas_step_")]
    if not delta_files:
        print("No delta files found for verification")
        return False
    
    first_deltas = torch.load(os.path.join(delta_log_dir, delta_files[0]), map_location='cpu')
    
    for name, mask in masks.items():
        if name not in first_deltas:
            print(f"Warning: Mask for {name} not in deltas")
            continue
        
        if mask.shape != first_deltas[name].shape:
            print(f"ERROR: Shape mismatch for {name}")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Delta shape: {first_deltas[name].shape}")
            return False
    
    print("✓ All mask shapes verified")
    return True


def main(args):
    delta_log_dir = args.delta_log_dir or "./delta_logs"
    base_state_path = os.path.join(delta_log_dir, "base_state.pt")
    
    # Create output directory
    os.makedirs("masks", exist_ok=True)
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load deltas
    deltas_by_step = load_deltas_from_log_dir(delta_log_dir, args.target_step)
    if not deltas_by_step:
        print("Failed to load deltas. Exiting.")
        return
    
    print(f"\nLoaded deltas for steps: {sorted(deltas_by_step.keys())}")
    
    # Compute ground truth mask if requested
    ground_truth_masks = None
    if args.compute_jaccard:
        # Load ALL deltas (no target_step limit) for ground truth
        all_deltas = load_deltas_from_log_dir(delta_log_dir, target_step=None)
        ground_truth_masks = compute_ground_truth_mask(all_deltas, args.sparsity_percent, device)
    
    # Compute masks based on method
    if args.method == "magnitude":
        masks = compute_absolute_magnitude_mask(deltas_by_step, args.sparsity_percent, device)
        method_suffix = "magnitude"
    
    elif args.method == "momentum":
        masks = compute_momentum_mask(deltas_by_step, args.sparsity_percent, args.momentum_window, device)
        method_suffix = f"momentum_w{args.momentum_window}"
    
    elif args.method == "fisher":
        masks = compute_fisher_mask(delta_log_dir, base_state_path, args.sparsity_percent, args.target_step, device)
        method_suffix = "fisher"
    
    else:
        print(f"Unknown method: {args.method}")
        return
    
    if masks is None:
        print("Failed to compute masks. Exiting.")
        return
    
    # Verify masks
    if not verify_masks(masks, delta_log_dir):
        print("Mask verification failed!")
        return
    
    # Compute Jaccard similarity if requested
    jaccard_results = None
    if args.compute_jaccard and ground_truth_masks is not None:
        jaccard_results = compute_jaccard_similarity(masks, ground_truth_masks)
    
    # Save masks
    step_suffix = f"_step{args.target_step}" if args.target_step else ""
    output_file = args.output_file or f"masks/sparsity_{args.sparsity_percent}pct_{method_suffix}{step_suffix}.pt"
    
    metadata = {
        "method": args.method,
        "sparsity_percent": args.sparsity_percent,
        "target_step": args.target_step,
        "num_steps_used": len(deltas_by_step),
        "steps": sorted(deltas_by_step.keys()),
        "device": device,
    }
    
    if args.method == "momentum":
        metadata["momentum_window"] = args.momentum_window
    
    if jaccard_results:
        metadata["jaccard_similarity"] = {
            "aggregate": jaccard_results["aggregate_jaccard"],
            "mean": jaccard_results["mean_jaccard"],
            "min": jaccard_results["min_jaccard"],
            "max": jaccard_results["max_jaccard"],
        }
    
    save_masks(masks, output_file, metadata)
    
    if jaccard_results:
        # Save detailed Jaccard results
        jaccard_file = output_file.replace(".pt", "_jaccard.json")
        with open(jaccard_file, "w") as f:
            json.dump(jaccard_results, f, indent=2)
        print(f"Detailed Jaccard results saved to: {jaccard_file}")
    
    print("\n✓ Mask generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate sparse training masks with Jaccard similarity metrics (GPU-accelerated)"
    )
    parser.add_argument(
        "--delta_log_dir",
        type=str,
        default="./delta_logs",
        help="Directory containing delta_logs (with base_state.pt and deltas_step_*.pt files)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["magnitude", "momentum", "fisher"],
        default="magnitude",
        help="Method for mask generation"
    )
    parser.add_argument(
        "--sparsity_percent",
        type=float,
        default=90.0,
        help="Target sparsity: percentage of weights to MASK OUT (default: 90.0 = keep 10%)"
    )
    parser.add_argument(
        "--target_step",
        type=int,
        default=25,
        help="Generate mask using deltas up to this step (default: 25)"
    )
    parser.add_argument(
        "--momentum_window",
        type=int,
        default=5,
        help="Window size for momentum calculation (only used with --method momentum)"
    )
    parser.add_argument(
        "--compute_jaccard",
        action="store_true",
        help="Compute Jaccard similarity against ground truth mask from final checkpoint"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path (default: auto-generated in masks/ directory)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution (default: use GPU if available)"
    )
    
    args = parser.parse_args()
    main(args)


# Example usage:
# python improved_mask_finder.py --method magnitude --sparsity_percent 90.0 --target_step 25 --compute_jaccard
# python improved_mask_finder.py --method momentum --sparsity_percent 90.0 --target_step 25 --momentum_window 5 --compute_jaccard
# python improved_mask_finder.py --method fisher --sparsity_percent 90.0 --target_step 25 --compute_jaccard