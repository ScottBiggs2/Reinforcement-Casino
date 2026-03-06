import torch
import torch.nn as nn
import argparse
import os
from transformers import AutoModelForCausalLM

def generate_random_mask(model_name, sparsity_percent, output_file, mlp_only=True):
    print(f"Generating random mask for {model_name} at {sparsity_percent}% sparsity...")
    
    # Load model to get parameter shapes
    # We use meta device or low_cpu_mem_usage to avoid OOM
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    masks = {}
    total_params = 0
    total_selected = 0
    
    # Identify target parameters
    target_params = []
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
        if mlp_only and 'mlp' not in name.lower():
            continue
        if param.dim() != 2: # Only linear layers
            continue
        target_params.append((name, param))
        total_params += param.numel()
    
    print(f"Targeting {len(target_params)} layers ({total_params:,} total parameters)")
    
    # Draw uniform random scores globally across target parameters
    # This matches the global top-k behavior of our other mask finders
    all_scores = []
    for name, param in target_params:
        all_scores.append(torch.rand(param.numel()))
    
    flat_scores = torch.cat(all_scores)
    
    # Determine threshold
    k = int((1.0 - sparsity_percent/100.0) * total_params)
    if k <= 0:
        k = 1
    
    threshold = torch.topk(flat_scores, k).values[-1]
    print(f"Global threshold at {k} tokens ({100-sparsity_percent:.2f}% density): {threshold:.6f}")
    
    # Apply threshold to each layer
    for (name, param), scores in zip(target_params, all_scores):
        mask = (scores >= threshold).view(param.shape).to(torch.float32)
        masks[name] = mask
        total_selected += mask.sum().item()
    
    # Save
    metadata = {
        "method": "random_global",
        "model_name": model_name,
        "sparsity_percent": sparsity_percent,
        "mlp_only": mlp_only,
        "total_params": total_params,
        "selected_params": total_selected,
        "actual_density": total_selected / total_params if total_params > 0 else 0
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save({"masks": masks, "metadata": metadata}, output_file)
    print(f"✓ Saved random mask to {output_file}")
    print(f"  Actual sparsity: {100.0 * (1.0 - total_selected / total_params):.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--sparsity_percent", type=float, default=97.5)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--mlp_only", action="store_true", default=True)
    args = parser.parse_args()
    
    generate_random_mask(args.model_name, args.sparsity_percent, args.output_file, args.mlp_only)
