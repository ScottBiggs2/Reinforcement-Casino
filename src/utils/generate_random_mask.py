import torch
import torch.nn as nn
import argparse
import os
import json
from transformers import AutoModelForCausalLM

def generate_random_mask(model_name, sparsity_percent, output_file, mlp_only=True, compare_mask=None):
    print(f"Generating random mask for {model_name} at {sparsity_percent}% sparsity...")
    
    # Load model to get parameter shapes
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
    
    # Random seeds for true independence
    import time
    torch.manual_seed(int(time.time()))
    
    all_scores = []
    for name, param in target_params:
        all_scores.append(torch.rand(param.numel()))
    
    flat_scores = torch.cat(all_scores)
    k = int((1.0 - sparsity_percent/100.0) * total_params)
    if k <= 0: k = 1
    
    threshold = torch.topk(flat_scores, k).values[-1]
    
    # Apply threshold
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
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save({"masks": masks, "metadata": metadata}, output_file)
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
    args = parser.parse_args()
    
    generate_random_mask(args.model_name, args.sparsity_percent, args.output_file, args.mlp_only, args.compare_mask)
