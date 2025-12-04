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


def compute_absolute_magnitude_mask(deltas_by_step, top_k_percent, device='cuda'):
    """
    GPU-accelerated magnitude mask computation.
    """
    print(f"\n=== Computing Absolute Magnitude Mask (top {top_k_percent}%) on {device} ===")
    
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
    total_masked = 0
    
    for idx, (name, total_change) in enumerate(aggregated.items()):
        if idx % 20 == 0:
            print(f"  Creating mask {idx+1}/{len(aggregated)}")
        
        # Move to GPU for fast sorting
        total_change_gpu = total_change.to(device)
        flat_changes = total_change_gpu.flatten()
        
        k = max(1, int(top_k_percent / 100.0 * flat_changes.numel()))
        
        if k >= flat_changes.numel():
            threshold = 0.0
        else:
            # kthvalue is FAST on GPU
            threshold = torch.kthvalue(flat_changes, flat_changes.numel() - k).values
        
        mask = (total_change_gpu >= threshold).float().cpu()  # Move back to CPU for storage
        masks[name] = mask
        
        total_params += mask.numel()
        total_masked += mask.sum().item()
        
        # Clean up GPU memory
        del total_change_gpu, flat_changes
        torch.cuda.empty_cache()
    
    actual_sparsity = (total_masked / total_params * 100) if total_params > 0 else 0
    print(f"Actual sparsity: {actual_sparsity:.2f}%")
    
    return masks


def compute_momentum_mask(deltas_by_step, top_k_percent, window_size=5, device='cuda'):
    """
    GPU-accelerated momentum mask computation.
    """
    print(f"\n=== Computing Momentum-Based Mask (top {top_k_percent}%, window={window_size}) on {device} ===")
    
    sorted_steps = sorted(deltas_by_step.keys())
    
    if len(sorted_steps) < 2:
        print("Warning: Need at least 2 steps for momentum calculation. Falling back to magnitude.")
        return compute_absolute_magnitude_mask(deltas_by_step, top_k_percent, device)
    
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
            momentum_scores[name] = deltas_by_step[sorted_steps[-1]][name].abs()
            continue
        
        # Stack and compute ON GPU
        vel_stack = torch.stack(recent_velocities).to(device)
        
        mean_velocity = vel_stack.mean(dim=0)
        std_velocity = vel_stack.std(dim=0) + 1e-8
        
        consistency = mean_velocity.abs() / std_velocity
        magnitude = mean_velocity.abs()
        
        score = (magnitude * (1 + consistency)).cpu()  # Move back to CPU
        momentum_scores[name] = score
        
        del vel_stack
        torch.cuda.empty_cache()
    
    # Create masks ON GPU
    print("Creating masks on GPU...")
    masks = {}
    total_params = 0
    total_masked = 0
    
    for idx, (name, score) in enumerate(momentum_scores.items()):
        if idx % 20 == 0:
            print(f"  Creating mask {idx+1}/{len(momentum_scores)}")
        
        score_gpu = score.to(device)
        flat_scores = score_gpu.flatten()
        
        k = max(1, int(top_k_percent / 100.0 * flat_scores.numel()))
        
        if k >= flat_scores.numel():
            threshold = 0.0
        else:
            threshold = torch.kthvalue(flat_scores, flat_scores.numel() - k).values
        
        mask = (score_gpu >= threshold).float().cpu()
        masks[name] = mask
        
        total_params += mask.numel()
        total_masked += mask.sum().item()
        
        del score_gpu, flat_scores
        torch.cuda.empty_cache()
    
    actual_sparsity = (total_masked / total_params * 100) if total_params > 0 else 0
    print(f"Actual sparsity: {actual_sparsity:.2f}%")
    
    return masks


def compute_fisher_mask(delta_log_dir, base_state_path, top_k_percent, target_step=None, device='cuda'):
    """
    GPU-accelerated Fisher approximation.
    """
    print(f"\n=== Computing Fisher-Approximation Mask (top {top_k_percent}%) on {device} ===")
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
    total_masked = 0
    
    for idx, (name, score) in enumerate(fisher_scores.items()):
        if idx % 20 == 0:
            print(f"  Creating mask {idx+1}/{len(fisher_scores)}")
        
        score_gpu = score.to(device)
        flat_scores = score_gpu.flatten()
        
        k = max(1, int(top_k_percent / 100.0 * flat_scores.numel()))
        
        if k >= flat_scores.numel():
            threshold = 0.0
        else:
            threshold = torch.kthvalue(flat_scores, flat_scores.numel() - k).values
        
        mask = (score_gpu >= threshold).float().cpu()
        masks[name] = mask
        
        total_params += mask.numel()
        total_masked += mask.sum().item()
        
        del score_gpu, flat_scores
        torch.cuda.empty_cache()
    
    actual_sparsity = (total_masked / total_params * 100) if total_params > 0 else 0
    print(f"Actual sparsity: {actual_sparsity:.2f}%")
    
    return masks


def save_masks(masks, output_file, metadata=None):
    """Saves masks with optional metadata."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    save_dict = {"masks": masks}
    if metadata:
        save_dict["metadata"] = metadata
    
    torch.save(save_dict, output_file)
    print(f"\nMasks saved to: {output_file}")
    print(f"Total parameters: {sum(m.numel() for m in masks.values())}")
    print(f"Masked parameters: {sum(m.sum().item() for m in masks.values())}")


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
    
    # Compute masks based on method
    if args.method == "magnitude":
        masks = compute_absolute_magnitude_mask(deltas_by_step, args.top_k_percent, device)
        method_suffix = "magnitude"
    
    elif args.method == "momentum":
        masks = compute_momentum_mask(deltas_by_step, args.top_k_percent, args.momentum_window, device)
        method_suffix = f"momentum_w{args.momentum_window}"
    
    elif args.method == "fisher":
        masks = compute_fisher_mask(delta_log_dir, base_state_path, args.top_k_percent, args.target_step, device)
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
    
    # Save masks
    step_suffix = f"_step{args.target_step}" if args.target_step else ""
    output_file = args.output_file or f"masks/top_{args.top_k_percent}pct_{method_suffix}{step_suffix}.pt"
    
    metadata = {
        "method": args.method,
        "top_k_percent": args.top_k_percent,
        "target_step": args.target_step,
        "num_steps_used": len(deltas_by_step),
        "steps": sorted(deltas_by_step.keys()),
        "device": device,
    }
    
    if args.method == "momentum":
        metadata["momentum_window"] = args.momentum_window
    
    save_masks(masks, output_file, metadata)
    
    print("\n✓ Mask generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate sparse training masks using various methods (GPU-accelerated)"
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
        "--top_k_percent",
        type=float,
        default=10.0,
        help="Percentage of weights to include in mask"
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
# python better_mask_finder.py --method magnitude --top_k_percent 10.0 --target_step 25
# python better_mask_finder.py --method momentum --top_k_percent 10.0 --target_step 25 --momentum_window 5
# python better_mask_finder.py --method fisher --top_k_percent 10.0 --target_step 25
# python better_mask_finder.py --method magnitude --top_k_percent 10.0 --target_step 25 --cpu  # Force CPU