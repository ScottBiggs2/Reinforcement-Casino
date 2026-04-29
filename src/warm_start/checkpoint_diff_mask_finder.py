import torch
import os
import sys
import argparse
import json
import gc
from typing import Dict, Optional

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.mask_utils import (
    DEFAULT_MIN_LAYER_KEEP_RATIO,
    create_mask_from_scores_gpu_efficient,
    save_masks,
    pooling_metadata,
)

def load_state_dict(path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Loads a state dict from a .pt file, .safetensors file, or a HuggingFace model (local/remote).
    """
    # 1. Check if it's a local file or directory that exists
    if os.path.exists(path):
        if os.path.isfile(path) and path.endswith((".pt", ".safetensors")):
            print(f"Loading state dict from local file: {path}")
            if path.endswith(".safetensors"):
                from safetensors.torch import load_file
                return load_file(path, device=device)
            else:
                return torch.load(path, map_location=device, weights_only=True)
        else:
            # Treat as local HuggingFace directory
            print(f"Loading local HuggingFace model directory: {path}")
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                path, 
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True
            )
            state_dict = model.state_dict()
            state_dict = {k: v.cpu().detach() for k, v in state_dict.items()}
            del model
            gc.collect()
            return state_dict

    # 2. If path doesn't exist locally, check if it might be a HuggingFace Hub ID
    # Hub IDs usually have 0 or 1 slashes (e.g., 'gpt2' or 'meta-llama/Llama-2-7b')
    if "/" in path and path.count("/") > 1:
        raise FileNotFoundError(
            f"Local path not found: '{path}'. \n"
            f"If this is a local checkpoint, verify the path exists. \n"
            f"HuggingFace Hub IDs typically don't have this many slashes."
        )

    print(f"Path not found locally, attempting HuggingFace Hub load: {path}")
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, 
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True
        )
        state_dict = model.state_dict()
        state_dict = {k: v.cpu().detach() for k, v in state_dict.items()}
        del model
        gc.collect()
        return state_dict
    except Exception as e:
        raise ValueError(f"Could not load HuggingFace model or file from '{path}': {e}")



def is_mlp_param(name):
    # MLP layer name patterns -- covers LLaMA, Gemma, Mistral, Qwen naming conventions
    MLP_KEYWORDS = ["gate_proj", "up_proj", "down_proj", "fc1", "fc2",
                    "feed_forward", "ffn", "mlp.c_fc", "mlp.c_proj"]
    return any(kw in name.lower() for kw in MLP_KEYWORDS)

def main(args):
    print(f"\n=== Checkpoint Difference Mask Finder ===")
    print(f"Initial model: {args.initial_model}")
    print(f"Final model:   {args.final_model}")
    print(f"Sparsity:      {args.sparsity_percent}%")

    # Load state dicts on CPU to avoid prompt OOM
    initial_sd = load_state_dict(args.initial_model, device="cpu")
    final_sd = load_state_dict(args.final_model, device="cpu")

    print("\nComputing weight differences (scores)...")
    scores = {}
    param_count = 0
    match_count = 0
    
    for name in final_sd:
        if name in initial_sd:
            if args.mlp_only and not is_mlp_param(name):
                continue
            
            # Use float32 for scores to maintain precision during subtraction
            diff = (final_sd[name].to(torch.float32) - initial_sd[name].to(torch.float32)).abs()
            scores[name] = diff
            match_count += 1
        param_count += 1

    print(f"Matched {match_count} parameters for scoring (out of {param_count} total).")
    print(f"Key coverage (final ∩ initial): {100.0 * match_count / max(param_count, 1):.2f}%")
    
    # Clean up to save memory
    del initial_sd
    del final_sd
    gc.collect()

    device = "cuda" if (torch.cuda.is_available() and not args.force_cpu) else "cpu"
    print(f"Using device for mask generation: {device}")

    # Generate masks
    masks = create_mask_from_scores_gpu_efficient(
        scores,
        args.sparsity_percent,
        device=device,
        local_pool=args.local_pool,
        min_layer_keep_ratio=args.min_layer_keep_ratio,
    )

    # Convert to boolean for space efficiency
    print("Converting masks to boolean...")
    for name in masks:
        masks[name] = masks[name].to(torch.bool)

    # Prepare metadata
    model_name = os.path.basename(os.path.normpath(args.final_model))
    output_file = args.output_file or f"masks/checkpoint_diff_ground_truth_{model_name}_sparsity{args.sparsity_percent}pct.pt"
    
    metadata = {
        "method": "checkpoint_difference_ground_truth",
        "sparsity_percent": args.sparsity_percent,
        "initial_model": args.initial_model,
        "final_model": args.final_model,
        "mlp_only": args.mlp_only,
        "device": device,
        **pooling_metadata(
            local_pool=args.local_pool,
            min_layer_keep_ratio=args.min_layer_keep_ratio,
        ),
    }

    # Save
    save_masks(masks, output_file, metadata)
    print(f"\n✓ Ground truth mask saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find ground truth mask from initial and final checkpoints.")
    parser.add_argument("--initial_model", type=str, required=True, help="Path to initial model (HF dir or .pt)")
    parser.add_argument("--final_model", type=str, required=True, help="Path to final model (HF dir or .pt)")
    parser.add_argument("--sparsity_percent", type=float, default=90.0, help="Target sparsity percentage")
    parser.add_argument("--output_file", type=str, default=None, help="Output path for the mask .pt file")
    parser.add_argument("--mlp_only", action="store_true", help="Only mask MLP parameters")
    parser.add_argument("--local_pool", action="store_true", help="Use local (per-layer) pooling")
    parser.add_argument("--min_layer_keep_ratio", type=float, default=DEFAULT_MIN_LAYER_KEEP_RATIO, help="Per-layer keep floor")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU for mask generation")
    
    args = parser.parse_args()
    main(args)
