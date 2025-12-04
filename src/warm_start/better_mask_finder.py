import torch
import os
import argparse
from collections import defaultdict
import json

def load_deltas_streaming(delta_log_dir, target_step=None):
    """
    Generator that yields deltas one at a time to save memory.
    
    Yields:
        tuple: (step, deltas_dict)
    """
    print(f"Streaming deltas from: {delta_log_dir}")
    
    # Find all delta files
    delta_files = sorted([f for f in os.listdir(delta_log_dir) if f.startswith("deltas_step_") and f.endswith(".pt")])
    
    for delta_file in delta_files:
        # Extract step number
        step = int(delta_file.split("_")[-1].replace(".pt", ""))
        
        # Skip if beyond target step
        if target_step is not None and step > target_step:
            continue
            
        file_path = os.path.join(delta_log_dir, delta_file)
        try:
            deltas = torch.load(file_path, map_location='cpu')
            print(f"Loaded deltas for step {step} ({len(deltas)} parameters)")
            yield step, deltas
        except Exception as e:
            print(f"Warning: Could not load {delta_file}: {e}")
            continue


def compute_absolute_magnitude_mask(delta_log_dir, top_k_percent, target_step=None):
    """
    Memory-efficient streaming version: accumulates deltas without loading all at once.
    """
    print(f"\n=== Computing Absolute Magnitude Mask (top {top_k_percent}%) ===")
    
    aggregated = None
    steps_processed = []
    
    for step, deltas in load_deltas_streaming(delta_log_dir, target_step):
        steps_processed.append(step)
        
        # Initialize on first file
        if aggregated is None:
            aggregated = {name: torch.zeros_like(delta) for name, delta in deltas.items()}
        
        # Accumulate absolute changes
        for name, delta in deltas.items():
            aggregated[name] += delta.abs()
        
        # Free memory immediately
        del deltas
    
    if aggregated is None:
        print("No delta files found!")
        return None, []
    
    # Create masks
    print("Creating masks...")
    masks = {}
    total_params = 0
    total_masked = 0
    
    for name, total_change in aggregated.items():
        flat_changes = total_change.flatten()
        k = max(1, int(top_k_percent / 100.0 * flat_changes.numel()))
        
        if k >= flat_changes.numel():
            threshold = 0.0
        else:
            threshold = torch.kthvalue(flat_changes, flat_changes.numel() - k).values
        
        mask = (total_change > threshold).float()
        masks[name] = mask
        
        total_params += mask.numel()
        total_masked += mask.sum().item()
    
    actual_sparsity = (total_masked / total_params * 100) if total_params > 0 else 0
    print(f"Actual sparsity: {actual_sparsity:.2f}%")
    
    return masks, steps_processed


def compute_momentum_mask(delta_log_dir, top_k_percent, target_step=None, window_size=5):
    """
    Memory-efficient momentum calculation with streaming.
    Keeps only a sliding window of recent deltas in memory.
    """
    print(f"\n=== Computing Momentum-Based Mask (top {top_k_percent}%, window={window_size}) ===")
    
    # We need to keep a window of recent deltas for momentum calculation
    delta_window = []  # List of (step, deltas_dict)
    steps_processed = []
    
    for step, deltas in load_deltas_streaming(delta_log_dir, target_step):
        steps_processed.append(step)
        delta_window.append((step, deltas))
        
        # Keep only the window size
        if len(delta_window) > window_size + 1:  # +1 for velocity calculation
            del delta_window[0]
    
    if len(steps_processed) < 2:
        print("Warning: Need at least 2 steps for momentum calculation. Falling back to magnitude.")
        return compute_absolute_magnitude_mask(delta_log_dir, top_k_percent, target_step)
    
    print(f"Computing momentum scores using {len(delta_window)} recent steps...")
    
    # Get parameter names from the last deltas
    param_names = list(delta_window[-1][1].keys())
    
    # Compute momentum scores
    momentum_scores = {}
    
    for name in param_names:
        # Collect deltas for this parameter from the window
        param_deltas = []
        for step, deltas in delta_window:
            if name in deltas:
                param_deltas.append(deltas[name])
        
        if len(param_deltas) < 2:
            # Not enough history, use absolute magnitude from last step
            momentum_scores[name] = delta_window[-1][1][name].abs()
            continue
        
        # Compute velocities (differences between consecutive deltas)
        velocities = []
        for i in range(1, len(param_deltas)):
            velocities.append(param_deltas[i] - param_deltas[i-1])
        
        if len(velocities) < 1:
            momentum_scores[name] = param_deltas[-1].abs()
            continue
        
        # Stack velocities
        vel_stack = torch.stack(velocities)  # [steps, *param_shape]
        
        # Compute momentum: magnitude of mean velocity weighted by consistency
        mean_velocity = vel_stack.mean(dim=0)
        std_velocity = vel_stack.std(dim=0) + 1e-8
        
        # Momentum score: high if changes are large and consistent
        consistency = mean_velocity.abs() / std_velocity
        magnitude = mean_velocity.abs()
        
        # Combined score
        momentum_scores[name] = magnitude * (1 + consistency)
    
    # Create masks based on momentum scores
    masks = {}
    total_params = 0
    total_masked = 0
    
    for name, score in momentum_scores.items():
        flat_scores = score.flatten()
        k = max(1, int(top_k_percent / 100.0 * flat_scores.numel()))
        
        if k >= flat_scores.numel():
            threshold = 0.0
        else:
            threshold = torch.kthvalue(flat_scores, flat_scores.numel() - k).values
        
        mask = (score > threshold).float()
        masks[name] = mask
        
        total_params += mask.numel()
        total_masked += mask.sum().item()
    
    actual_sparsity = (total_masked / total_params * 100) if total_params > 0 else 0
    print(f"Actual sparsity: {actual_sparsity:.2f}%")
    
    return masks, steps_processed


def compute_fisher_mask(delta_log_dir, base_state_path, top_k_percent, target_step=None):
    """
    Memory-efficient Fisher approximation with streaming.
    Computes running mean and variance without loading all deltas at once.
    """
    print(f"\n=== Computing Fisher-Approximation Mask (top {top_k_percent}%) ===")
    print("Note: This is an approximation since we don't have true gradients.")
    
    # Load base state
    if not os.path.exists(base_state_path):
        print(f"Error: Base state not found at {base_state_path}")
        return None, []
    
    base_state = torch.load(base_state_path, map_location='cpu')
    print(f"Loaded base state with {len(base_state)} parameters")
    
    # Running statistics for Fisher approximation
    running_mean = {}
    running_m2 = {}  # For Welford's online variance algorithm
    count = 0
    steps_processed = []
    
    for step, deltas in load_deltas_streaming(delta_log_dir, target_step):
        steps_processed.append(step)
        count += 1
        
        for name, delta in deltas.items():
            if name not in running_mean:
                running_mean[name] = torch.zeros_like(delta)
                running_m2[name] = torch.zeros_like(delta)
            
            # Welford's online algorithm for variance
            delta_from_mean = delta - running_mean[name]
            running_mean[name] += delta_from_mean / count
            delta2_from_mean = delta - running_mean[name]
            running_m2[name] += delta_from_mean * delta2_from_mean
        
        del deltas
    
    if count == 0:
        print("No delta files found!")
        return None, []
    
    print(f"Computing Fisher scores from {count} steps...")
    
    # Compute Fisher scores: variance + abs(mean)
    fisher_scores = {}
    
    for name in base_state.keys():
        if name not in running_mean:
            fisher_scores[name] = torch.zeros_like(base_state[name])
            continue
        
        # Variance = M2 / count
        variance = running_m2[name] / count if count > 1 else torch.zeros_like(running_m2[name])
        
        # Fisher ≈ variance + abs(mean)
        fisher_scores[name] = variance + running_mean[name].abs()
    
    # Create masks
    masks = {}
    total_params = 0
    total_masked = 0
    
    for name, score in fisher_scores.items():
        flat_scores = score.flatten()
        k = max(1, int(top_k_percent / 100.0 * flat_scores.numel()))
        
        if k >= flat_scores.numel():
            threshold = 0.0
        else:
            threshold = torch.kthvalue(flat_scores, flat_scores.numel() - k).values
        
        mask = (score > threshold).float()
        masks[name] = mask
        
        total_params += mask.numel()
        total_masked += mask.sum().item()
    
    actual_sparsity = (total_masked / total_params * 100) if total_params > 0 else 0
    print(f"Actual sparsity: {actual_sparsity:.2f}%")
    
    return masks, steps_processed


def save_masks(masks, output_file, metadata=None):
    """Saves masks with optional metadata."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save in the format expected by Triton_DPO_train.py
    torch.save(masks, output_file)
    
    # Also save metadata separately if provided
    if metadata:
        metadata_file = output_file.replace('.pt', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_file}")
    
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
    
    # Compute masks based on method (now all streaming!)
    if args.method == "magnitude":
        masks, steps_used = compute_absolute_magnitude_mask(delta_log_dir, args.top_k_percent, args.target_step)
        method_suffix = "magnitude"
    
    elif args.method == "momentum":
        masks, steps_used = compute_momentum_mask(delta_log_dir, args.top_k_percent, args.target_step, args.momentum_window)
        method_suffix = f"momentum_w{args.momentum_window}"
    
    elif args.method == "fisher":
        masks, steps_used = compute_fisher_mask(delta_log_dir, base_state_path, args.top_k_percent, args.target_step)
        method_suffix = "fisher"
    
    else:
        print(f"Unknown method: {args.method}")
        return
    
    if masks is None:
        print("Failed to compute masks. Exiting.")
        return
    
    print(f"\nUsed deltas from steps: {steps_used}")
    
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
        "num_steps_used": len(steps_used),
        "steps": steps_used,
    }
    
    if args.method == "momentum":
        metadata["momentum_window"] = args.momentum_window
    
    save_masks(masks, output_file, metadata)
    
    print("\n✓ Mask generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate sparse training masks using various methods"
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
    
    args = parser.parse_args()
    main(args)


# Example usage:
# python better_mask_finder.py --method magnitude --top_k_percent 10.0 --target_step 25
# python better_mask_finder.py --method momentum --top_k_percent 10.0 --target_step 25 --momentum_window 5
# python better_mask_finder.py --method fisher --top_k_percent 10.0 --target_step 25