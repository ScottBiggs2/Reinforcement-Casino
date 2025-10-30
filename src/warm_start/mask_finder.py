import torch
import glob
import os
from collections import defaultdict
import argparse

def find_and_aggregate_deltas(run_dir):
    """
    Finds all weight delta files in a run directory, aggregates them,
    and returns a dictionary of total absolute changes for each parameter.
    """
    print(f"Aggregating deltas from: {run_dir}")
    step_dirs = sorted(glob.glob(os.path.join(run_dir, "step_*")))
    aggregated_deltas = defaultdict(lambda: None)

    if not step_dirs:
        print("No step directories found.")
        return {}

    # Get all parameter names from the first step
    first_step_files = glob.glob(os.path.join(step_dirs[0], "*.pt"))
    if not first_step_files:
        print(f"No .pt files found in {step_dirs[0]}")
        return {}

    param_names = [os.path.basename(f).replace('_', '.')[:-3] for f in first_step_files]

    # Initialize aggregated_deltas with zero tensors
    for param_name in param_names:
        try:
            delta_tensor = torch.load(os.path.join(step_dirs[0], param_name.replace('.', '_') + ".pt"))
            aggregated_deltas[param_name] = torch.zeros_like(delta_tensor)
        except Exception as e:
            print(f"Could not load tensor for {param_name}: {e}")
            continue

    # Aggregate deltas
    for step_dir in step_dirs:
        for param_name in param_names:
            file_path = os.path.join(step_dir, param_name.replace('.', '_') + ".pt")
            if os.path.exists(file_path):
                try:
                    delta_tensor = torch.load(file_path)
                    if aggregated_deltas[param_name] is not None:
                        aggregated_deltas[param_name] += delta_tensor.abs()
                except Exception as e:
                    print(f"Could not load or aggregate tensor {file_path}: {e}")
                    continue
    
    return aggregated_deltas

def create_masks(aggregated_deltas, top_k_percent):
    """
    Creates sparse masks from aggregated weight deltas based on the top-k percentage.
    """
    print(f"Creating masks for top {top_k_percent}% of changes...")
    masks = {}
    for name, total_change in aggregated_deltas.items():
        if total_change is None:
            continue
        
        flat_changes = total_change.flatten()
        if flat_changes.numel() == 0:
            masks[name] = torch.zeros_like(total_change)
            continue

        k = int(top_k_percent / 100.0 * flat_changes.numel())
        if k == 0:
            # If k is 0, no weights should be selected.
            threshold = float('inf')
        else:
            # Find the k-th largest value. We need to find the (n-k)-th smallest value.
            threshold = torch.kthvalue(flat_changes, flat_changes.numel() - k).values
        
        masks[name] = (total_change > threshold).float()
        
    return masks

def save_masks(masks, output_file):
    """
    Saves the generated masks to a .pt file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(masks, output_file)
    print(f"Masks saved to {output_file}")

def verify_deltas(run_dir):
    """
    Verifies that the delta files are loadable and have consistent shapes.
    """
    print("--- Verifying Delta Files ---")
    step_dirs = sorted(glob.glob(os.path.join(run_dir, "step_*")))
    if not step_dirs:
        print("Verification failed: No step directories found.")
        return False

    param_shapes = {}
    all_params = set()

    # First pass to get all param names and their shapes from the first step
    first_step_files = glob.glob(os.path.join(step_dirs[0], "*.pt"))
    if not first_step_files:
        print(f"Verification failed: No .pt files in {step_dirs[0]}")
        return False
    for f in first_step_files:
        param_name = os.path.basename(f).replace('_', '.')[:-3]
        all_params.add(param_name)
        try:
            tensor = torch.load(f)
            param_shapes[param_name] = tensor.shape
        except Exception as e:
            print(f"Verification failed: Error loading {f}: {e}")
            return False

    # Check subsequent steps for consistency
    for step_dir in step_dirs[1:]:
        current_params = set()
        delta_files = glob.glob(os.path.join(step_dir, "*.pt"))
        for f in delta_files:
            param_name = os.path.basename(f).replace('_', '.')[:-3]
            current_params.add(param_name)
            if param_name not in param_shapes:
                print(f"Verification failed: New parameter {param_name} found in {step_dir}")
                return False
            try:
                tensor = torch.load(f)
                if param_shapes[param_name] != tensor.shape:
                    print(f"Verification failed: Shape mismatch for {param_name} in {step_dir}")
                    return False
            except Exception as e:
                print(f"Verification failed: Error loading {f}: {e}")
                return False
        if current_params != all_params:
            print(f"Verification failed: Parameter set mismatch in {step_dir}")
            return False

    print("Delta files verification successful: all files are loadable and shapes are consistent.")
    return True

def verify_masks(mask_file, run_dir, top_k_percent):
    """
    Verifies the generated mask file.
    """
    print("--- Verifying Mask File ---")
    if not os.path.exists(mask_file):
        print(f"Verification failed: Mask file not found at {mask_file}")
        return False

    try:
        masks = torch.load(mask_file)
    except Exception as e:
        print(f"Verification failed: Error loading mask file {mask_file}: {e}")
        return False
    
    # Get expected shapes from delta files
    step_dirs = glob.glob(os.path.join(run_dir, "step_*"))
    if not step_dirs:
        print("Verification failed: No step directories found to verify mask shapes.")
        return False
    
    first_step_files = glob.glob(os.path.join(step_dirs[0], "*.pt"))
    if not first_step_files:
        print("Verification failed: No delta files found to verify mask shapes.")
        return False

    param_shapes = {}
    for f in first_step_files:
        param_name = os.path.basename(f).replace('_', '.')[:-3]
        tensor = torch.load(f)
        param_shapes[param_name] = tensor.shape

    # Check shapes and sparsity
    total_params = 0
    total_masked_params = 0
    for name, mask in masks.items():
        if name not in param_shapes:
            print(f"Warning: Mask for {name} not found in reference delta files.")
            continue
        if mask.shape != param_shapes[name]:
            print(f"Verification failed: Shape mismatch for mask {name}. Expected {param_shapes[name]}, got {mask.shape}")
            return False
        
        num_masked = torch.sum(mask).item()
        total_elements = mask.numel()
        
        if total_elements > 0:
            sparsity = num_masked / total_elements * 100
            print(f"Mask for {name}: sparsity = {sparsity:.2f}% (target ~{top_k_percent}%)")
            total_params += total_elements
            total_masked_params += num_masked

    if total_params > 0:
        overall_sparsity = total_masked_params / total_params * 100
        print(f"\nOverall mask sparsity: {overall_sparsity:.2f}% (target ~{top_k_percent}%)")
    
    print("Mask verification successful: shapes are consistent and sparsity is as expected.")
    return True

def main(args):
    run_dir = os.path.join("results", args.run_name)
    output_file = args.output_file or f"masks/top_{args.top_k_percent}_percent_mask.pt"

    # Verification of deltas
    if not verify_deltas(run_dir):
        return

    # Main logic
    aggregated_deltas = find_and_aggregate_deltas(run_dir)
    if not aggregated_deltas:
        print("No deltas found to process.")
        return
        
    masks = create_masks(aggregated_deltas, args.top_k_percent)
    save_masks(masks, output_file)

    # Verification of masks
    verify_masks(output_file, run_dir, args.top_k_percent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find top-k% changing weights and create a mask.")
    parser.add_argument("--run_name", type=str, default="gemma_dpo_training", help="Name of the training run directory in results/.")
    parser.add_argument("--top_k_percent", type=float, default=10.0, help="Top-k percentage of weights to include in the mask.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save the output mask file.")
    
    args = parser.parse_args()
    main(args)
