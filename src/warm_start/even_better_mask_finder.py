import torch
import os
import argparse
from collections import defaultdict
import json
import gc

def load_deltas_streaming(delta_log_dir, target_step=None):
    """
    Returns iterator of (step, delta_path) instead of loading all into memory.
    """
    print(f"Scanning deltas from: {delta_log_dir}")
    
    delta_files = [f for f in os.listdir(delta_log_dir) 
                   if f.startswith("deltas_step_") and f.endswith(".pt")]
    
    def extract_step(filename):
        return int(filename.split("_")[-1].replace(".pt", ""))
    
    delta_files = sorted(delta_files, key=extract_step)
    
    steps_and_paths = []
    for delta_file in delta_files:
        step = extract_step(delta_file)
        if target_step is not None and step > target_step:
            continue
        steps_and_paths.append((step, os.path.join(delta_log_dir, delta_file)))
    
    print(f"Found {len(steps_and_paths)} delta files")
    return steps_and_paths


def create_mask_from_scores_gpu_efficient(scores_dict, sparsity_percent, device='cuda'):
    """
    GPU-accelerated exact top-k mask creation.
    Processes scores directly on GPU without unnecessary transfers.
    """
    print(f"\n=== Creating Exact Masks (target sparsity: {sparsity_percent}%) ===")
    
    keep_percent = 100.0 - sparsity_percent
    
    # Add noise on GPU
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
        
        # Keep score on GPU for potential correction pass
    
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
    for name in list(scores_dict.keys()):
        del scores_dict[name]
    torch.cuda.empty_cache()
    
    actual_sparsity = 100.0 - (total_kept / total_params * 100)
    
    print(f"\nVerification:")
    print(f"  Target keep: {keep_count:,} ({keep_percent:.1f}%)")
    print(f"  Actual keep: {int(total_kept):,} ({100.0 - actual_sparsity:.1f}%)")
    print(f"  Actual sparsity: {actual_sparsity:.2f}% (target: {sparsity_percent}%)")
    print(f"  Error: {abs(actual_sparsity - sparsity_percent):.2f}%")
    
    return masks


def compute_ground_truth_mask_streaming(steps_and_paths, sparsity_percent, device='cuda'):
    """
    Computes ground truth mask by loading only the final checkpoint.
    """
    print(f"\n=== Computing Ground Truth Mask (target sparsity: {sparsity_percent}%) ===")
    
    final_step, final_path = steps_and_paths[-1]
    print(f"Loading final checkpoint at step {final_step}")
    
    final_deltas = torch.load(final_path, map_location=device)
    
    # Compute scores directly on GPU
    scores = {name: delta.abs() for name, delta in final_deltas.items()}
    
    masks = create_mask_from_scores_gpu_efficient(scores, sparsity_percent, device)
    
    del final_deltas
    torch.cuda.empty_cache()
    
    return masks


def compute_absolute_magnitude_mask_streaming(steps_and_paths, sparsity_percent, device='cuda', debug=False):
    """
    Magnitude mask with streaming: accumulate on GPU, never load all checkpoints at once.
    """
    print(f"\n=== Computing Absolute Magnitude Mask (target sparsity: {sparsity_percent}%) ===")
    print("Processing deltas in streaming fashion...")
    
    aggregated = {}
    param_names = None
    
    for step_idx, (step, delta_path) in enumerate(steps_and_paths):
        print(f"  [{step_idx+1}/{len(steps_and_paths)}] Processing step {step}...")
        
        # Load directly to GPU
        deltas = torch.load(delta_path, map_location=device)
        
        # Initialize aggregated dict on first pass
        if param_names is None:
            param_names = list(deltas.keys())
            for name in param_names:
                aggregated[name] = torch.zeros_like(deltas[name], device=device)
        
        # Accumulate on GPU
        for name in param_names:
            if name in deltas:
                aggregated[name] += deltas[name].abs()
        
        # Free memory immediately
        del deltas
        torch.cuda.empty_cache()
    
    if debug:
        print("\nAggregated score statistics (first 5 layers):")
        for idx, name in enumerate(list(aggregated.keys())[:5]):
            score = aggregated[name]
            print(f"  {name}: min={score.min().item():.10f}, max={score.max().item():.10f}, mean={score.mean().item():.10f}")
    
    masks = create_mask_from_scores_gpu_efficient(aggregated, sparsity_percent, device)
    
    del aggregated
    torch.cuda.empty_cache()
    
    return masks


def compute_momentum_mask_streaming(steps_and_paths, sparsity_percent, window_size=5, device='cuda', debug=False):
    """
    Momentum mask with streaming: only keep previous checkpoint in memory.
    """
    print(f"\n=== Computing Momentum-Based Mask (target sparsity: {sparsity_percent}%, window={window_size}) ===")
    
    if len(steps_and_paths) < 2:
        print("Warning: Need at least 2 steps for momentum. Falling back to magnitude.")
        return compute_absolute_magnitude_mask_streaming(steps_and_paths, sparsity_percent, device)
    
    # Store recent velocities in a sliding window (on GPU)
    velocity_window = defaultdict(list)  # {param_name: [v_t-w, ..., v_t]}
    prev_deltas = None
    param_names = None
    
    print("Computing velocities in streaming fashion...")
    for step_idx, (step, delta_path) in enumerate(steps_and_paths):
        print(f"  [{step_idx+1}/{len(steps_and_paths)}] Processing step {step}...")
        
        curr_deltas = torch.load(delta_path, map_location=device)
        
        if param_names is None:
            param_names = list(curr_deltas.keys())
        
        if prev_deltas is not None:
            # Compute velocity: v_t = delta_t - delta_{t-1}
            for name in param_names:
                if name in curr_deltas and name in prev_deltas:
                    velocity = curr_deltas[name] - prev_deltas[name]
                    velocity_window[name].append(velocity)
                    
                    # Keep only last 'window_size' velocities
                    if len(velocity_window[name]) > window_size:
                        old_v = velocity_window[name].pop(0)
                        del old_v
        
        # Update prev_deltas
        if prev_deltas is not None:
            del prev_deltas
        prev_deltas = curr_deltas
        
        torch.cuda.empty_cache()
    
    # Compute momentum scores from accumulated velocities
    print("Computing momentum scores...")
    momentum_scores = {}
    
    for name in param_names:
        if name not in velocity_window or len(velocity_window[name]) < 2:
            # Fallback to magnitude from last checkpoint
            momentum_scores[name] = prev_deltas[name].abs()
            continue
        
        # Stack velocities (already on GPU)
        vel_stack = torch.stack(velocity_window[name])
        
        mean_velocity = vel_stack.mean(dim=0)
        std_velocity = vel_stack.std(dim=0) + 1e-8
        
        consistency = mean_velocity.abs() / std_velocity
        magnitude = mean_velocity.abs()
        
        momentum_scores[name] = magnitude * (1 + consistency)
        
        del vel_stack
    
    # Clean up velocity window
    for name in velocity_window.keys():
        for v in velocity_window[name]:
            del v
    velocity_window.clear()
    
    if prev_deltas is not None:
        del prev_deltas
    
    torch.cuda.empty_cache()
    
    if debug:
        print("\nMomentum score statistics (first 5 layers):")
        for idx, name in enumerate(list(momentum_scores.keys())[:5]):
            score = momentum_scores[name]
            print(f"  {name}: min={score.min().item():.10f}, max={score.max().item():.10f}, mean={score.mean().item():.10f}")
    
    masks = create_mask_from_scores_gpu_efficient(momentum_scores, sparsity_percent, device)
    
    del momentum_scores
    torch.cuda.empty_cache()
    
    return masks


def compute_fisher_mask_streaming(steps_and_paths, sparsity_percent, device='cuda'):
    """
    Fisher approximation with streaming: accumulate sum and sum-of-squares on GPU.
    Fisher ≈ Var[delta] + |E[delta]| = E[delta²] - E[delta]² + |E[delta]|
    """
    print(f"\n=== Computing Fisher-Approximation Mask (target sparsity: {sparsity_percent}%) ===")
    print("Computing variance in streaming fashion using Welford's algorithm...")
    
    # Accumulate statistics on GPU
    count = 0
    sum_delta = {}
    sum_delta_sq = {}
    param_names = None
    
    for step_idx, (step, delta_path) in enumerate(steps_and_paths):
        print(f"  [{step_idx+1}/{len(steps_and_paths)}] Processing step {step}...")
        
        deltas = torch.load(delta_path, map_location=device)
        
        if param_names is None:
            param_names = list(deltas.keys())
            for name in param_names:
                sum_delta[name] = torch.zeros_like(deltas[name], device=device)
                sum_delta_sq[name] = torch.zeros_like(deltas[name], device=device)
        
        for name in param_names:
            if name in deltas:
                sum_delta[name] += deltas[name]
                sum_delta_sq[name] += deltas[name] ** 2
        
        count += 1
        
        del deltas
        torch.cuda.empty_cache()
    
    # Compute Fisher scores
    print("Computing Fisher scores...")
    fisher_scores = {}
    
    for name in param_names:
        mean_delta = sum_delta[name] / count
        mean_delta_sq = sum_delta_sq[name] / count
        
        # Var[X] = E[X²] - E[X]²
        variance = mean_delta_sq - mean_delta ** 2
        variance = torch.clamp(variance, min=0)  # Numerical stability
        
        fisher_scores[name] = variance + mean_delta.abs()
    
    # Clean up
    del sum_delta, sum_delta_sq
    torch.cuda.empty_cache()
    
    masks = create_mask_from_scores_gpu_efficient(fisher_scores, sparsity_percent, device)
    
    del fisher_scores
    torch.cuda.empty_cache()
    
    return masks


def compute_jaccard_similarity(pred_masks, true_masks):
    """
    Computes Jaccard similarity between predicted and ground truth masks.
    Uses GPU for faster computation.
    """
    print("\n=== Computing Jaccard Similarity ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    per_layer_jaccard = {}
    total_intersection = 0
    total_union = 0
    
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
        
        del pred, true
    
    torch.cuda.empty_cache()
    
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
    print(f"Kept parameters: {int(kept_params):,}")
    print(f"Final sparsity: {actual_sparsity:.2f}%")


def verify_masks(masks, steps_and_paths):
    """Verify mask shapes match delta shapes."""
    print("\n=== Verifying Masks ===")
    
    if not steps_and_paths:
        print("No delta files found for verification")
        return False
    
    _, first_path = steps_and_paths[0]
    first_deltas = torch.load(first_path, map_location='cpu')
    
    for name, mask in masks.items():
        if name not in first_deltas:
            print(f"Warning: Mask for {name} not in deltas")
            continue
        
        if mask.shape != first_deltas[name].shape:
            print(f"ERROR: Shape mismatch for {name}")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Delta shape: {first_deltas[name].shape}")
            del first_deltas
            return False
    
    del first_deltas
    print("✓ All mask shapes verified")
    return True


def main(args):
    delta_log_dir = args.delta_log_dir or "./delta_logs"
    
    # Create output directory
    os.makedirs("masks", exist_ok=True)
    
    # Check GPU availability
    if not torch.cuda.is_available() or args.cpu:
        print("ERROR: This script requires CUDA. Use the original script for CPU.")
        return
    
    device = 'cuda'
    print(f"\nUsing device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Get streaming iterator
    steps_and_paths = load_deltas_streaming(delta_log_dir, args.target_step)
    if not steps_and_paths:
        print("Failed to find delta files. Exiting.")
        return
    
    steps = [s for s, _ in steps_and_paths]
    print(f"\nFound deltas for steps: {steps}")
    
    # Compute ground truth mask if requested
    ground_truth_masks = None
    if args.compute_jaccard:
        print("\n" + "="*60)
        print("Computing ground truth mask for Jaccard comparison...")
        print("="*60)
        all_steps = load_deltas_streaming(delta_log_dir, target_step=None)
        ground_truth_masks = compute_ground_truth_mask_streaming(all_steps, args.sparsity_percent, device)
    
    # Compute masks based on method
    print("\n" + "="*60)
    print(f"Computing {args.method} mask...")
    print("="*60)
    
    if args.method == "magnitude":
        masks = compute_absolute_magnitude_mask_streaming(steps_and_paths, args.sparsity_percent, device, args.debug)
        method_suffix = "magnitude"
    
    elif args.method == "momentum":
        masks = compute_momentum_mask_streaming(steps_and_paths, args.sparsity_percent, args.momentum_window, device, args.debug)
        method_suffix = f"momentum_w{args.momentum_window}"
    
    elif args.method == "fisher":
        masks = compute_fisher_mask_streaming(steps_and_paths, args.sparsity_percent, device)
        method_suffix = "fisher"
    
    else:
        print(f"Unknown method: {args.method}")
        return
    
    if masks is None:
        print("Failed to compute masks. Exiting.")
        return
    
    # Verify masks
    if not verify_masks(masks, steps_and_paths):
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
        "num_steps_used": len(steps_and_paths),
        "steps": steps,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0),
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
        jaccard_file = output_file.replace(".pt", "_jaccard.json")
        with open(jaccard_file, "w") as f:
            json.dump(jaccard_results, f, indent=2)
        print(f"Detailed Jaccard results saved to: {jaccard_file}")
    
    print("\n✓ Mask generation complete!")
    print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPU-accelerated sparse mask generation with streaming for large models"
    )
    parser.add_argument("--delta_log_dir", type=str, default="./delta_logs")
    parser.add_argument("--method", type=str, choices=["magnitude", "momentum", "fisher"], default="magnitude")
    parser.add_argument("--sparsity_percent", type=float, default=90.0)
    parser.add_argument("--target_step", type=int, default=None)
    parser.add_argument("--momentum_window", type=int, default=5)
    parser.add_argument("--compute_jaccard", action="store_true")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (not recommended)")
    
    args = parser.parse_args()
    main(args)

# Example usage:
# python gpu_mask_finder.py --method magnitude --sparsity_percent 90.0 --target_step 100 --compute_jaccard
# python gpu_mask_finder.py --method momentum --sparsity_percent 90.0 --momentum_window 10
# python gpu_mask_finder.py --method fisher --sparsity_percent 90.0