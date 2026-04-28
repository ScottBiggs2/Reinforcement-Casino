import torch
import torch.nn as nn
import argparse
import os
import json
import sys
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# On-disk masks are torch.bool via ``save_masks`` (see ``src.utils.mask_utils`` module docstring).
from src.utils.mask_utils import (
    DEFAULT_MIN_LAYER_KEEP_RATIO,
    build_binary_masks_from_scores_blockwise,
    create_mask_from_scores_gpu_efficient,
    pooling_metadata,
    save_masks,
)


def generate_random_mask(
    model_name,
    sparsity_percent,
    output_file,
    mlp_only=False,
    compare_mask=None,
    min_layer_keep_ratio=DEFAULT_MIN_LAYER_KEEP_RATIO,
    seed=None,
    mask_granularity: str = "element",
    mask_block_size: int = 256,
    mask_block_reduction: str = "mean",
):
    print(f"Generating random mask for {model_name} at {sparsity_percent}% sparsity...")
    if min_layer_keep_ratio > 0:
        print(
            "Mask pooling: global with a small per-tensor keep floor "
            f"(min_layer_keep_ratio={min_layer_keep_ratio})"
        )
    else:
        print("Mask pooling: pure global (single threshold across all random scores)")

    mask_granularity = str(mask_granularity or "element").strip().lower()
    if mask_granularity not in ("element", "block"):
        raise ValueError(f"--mask_granularity must be one of: element, block (got {mask_granularity!r})")
    if mask_granularity == "block":
        if int(mask_block_size) < 1:
            raise ValueError(f"--mask_block_size must be >= 1 (got {mask_block_size})")
        mask_block_reduction = str(mask_block_reduction or "mean").strip().lower()
        if mask_block_reduction not in ("mean", "max"):
            raise ValueError(
                f"--mask_block_reduction must be one of: mean, max (got {mask_block_reduction!r})"
            )
        print(
            f"Mask layout: block (block_size={int(mask_block_size)}, reduction={mask_block_reduction}) "
            "(note: nominal sparsity applies to block-grid; realized weight sparsity may differ slightly)"
        )
    else:
        print("Mask layout: element")
    
    # Load model to get parameter shapes
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    import time

    if seed is not None:
        torch.manual_seed(int(seed))
    else:
        torch.manual_seed(int(time.time()))

    total_params = 0
    scores = {}

    # Identify target parameters
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
        if mlp_only and 'mlp' not in name.lower():
            continue
        if param.dim() != 2: # Only linear layers
            continue
        total_params += param.numel()
        scores[name] = torch.rand(param.shape, dtype=torch.float32)

    print(f"Targeting {len(scores)} layers ({total_params:,} total parameters)")
    
    if mask_granularity == "block":
        masks = build_binary_masks_from_scores_blockwise(
            scores,
            sparsity_percent=float(sparsity_percent),
            block_size=int(mask_block_size),
            device="cpu",
            local_pool=False,
            min_layer_keep_ratio=float(min_layer_keep_ratio),
            reduction=mask_block_reduction,
            add_tie_break_noise=False,
        )
    else:
        masks = create_mask_from_scores_gpu_efficient(
            scores,
            sparsity_percent,
            device="cpu",
            min_layer_keep_ratio=min_layer_keep_ratio,
            local_pool=False,
        )
    total_selected = sum(mask.sum().item() for mask in masks.values())
    
    # Save
    metadata = {
        "method": "random_global" if mask_granularity == "element" else "random_global_block",
        "model_name": model_name,
        "sparsity_percent": sparsity_percent,
        "mlp_only": mlp_only,
        "mask_granularity": mask_granularity,
        "mask_block_size": int(mask_block_size) if mask_granularity == "block" else None,
        "mask_block_reduction": mask_block_reduction if mask_granularity == "block" else None,
        "total_params": total_params,
        "selected_params": total_selected,
        **pooling_metadata(local_pool=False, min_layer_keep_ratio=min_layer_keep_ratio),
    }
    
    save_masks(masks, output_file, metadata=metadata)
    print(f"✓ Saved random mask to {output_file}")
    
    # Diagnostic Jaccard check if a mask is provided
    if compare_mask and os.path.exists(compare_mask):
        print(f"\n--- Diagnostic: Jaccard vs {os.path.basename(compare_mask)} ---")
        comp = torch.load(compare_mask, map_location='cpu')
        comp_masks = comp['masks'] if 'masks' in comp else comp
        
        inter = 0
        union = 0
        common_keys = set(masks.keys()).intersection(set(comp_masks.keys()))
        for k_name in common_keys:
            m1 = masks[k_name].bool()
            m2 = comp_masks[k_name].bool()
            inter += (m1 & m2).sum().item()
            union += (m1 | m2).sum().item()
        
        jaccard = inter / union if union > 0 else 0
        density = 1.0 - sparsity_percent/100.0
        expected = density / (2.0 - density)
        print(f"  Observed Jaccard: {jaccard:.6f}")
        print(f"  Expected (Chance): {expected:.6f}")
        print(f"  Ratio: {jaccard/expected:.2f}x (Closer to 1.0x means truer random)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--sparsity_percent", type=float, default=97.5)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--mlp_only", action="store_true", default=False)
    parser.add_argument("--compare_mask", type=str, default=None, help="Diagnostic: compare to this mask")
    parser.add_argument(
        "--min_layer_keep_ratio",
        type=float,
        default=DEFAULT_MIN_LAYER_KEEP_RATIO,
        help=(
            "Small per-tensor keep floor for hybrid global masking. "
            "Set to 0.0 for pure global selection."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible random masks (default: time-based).",
    )
    parser.add_argument(
        "--mask-granularity",
        "--mask_granularity",
        type=str,
        default="element",
        choices=["element", "block"],
        help="Mask layout: element-wise (default) or block-structured.",
    )
    parser.add_argument(
        "--mask-block-size",
        "--mask_block_size",
        type=int,
        default=256,
        help="Block size B for block-structured masks (B×B). Used only when --mask_granularity=block.",
    )
    parser.add_argument(
        "--mask-block-reduction",
        "--mask_block_reduction",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="How to pool element scores into blocks. Used only when --mask_granularity=block.",
    )
    args = parser.parse_args()

    generate_random_mask(
        args.model_name,
        args.sparsity_percent,
        args.output_file,
        args.mlp_only,
        args.compare_mask,
        args.min_layer_keep_ratio,
        args.seed,
        args.mask_granularity,
        args.mask_block_size,
        args.mask_block_reduction,
    )
