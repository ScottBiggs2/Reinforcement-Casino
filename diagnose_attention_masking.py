
import os
import sys
import torch
import torch.nn as nn
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.mlps.bsr_sparse_mlp import replace_linear_modules
from src.optimizers.sparse_adamw import SparseAdamW
from src.utils.mask_manager import SparseMaskManager

def run_diagnostic(model_name="google/gemma-2b-it"):
    print(f"============================================================")
    print(f"DIAGNOSTIC: Attention Masking Isolation")
    print(f"============================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load a tiny model component or create a dummy structure similar to the real one
    # For speed and reliability, we create a structure with the same layer names
    class AttentionLikeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(128, 128, bias=False)  # Attention-like
            self.mlp_up = nn.Linear(128, 512, bias=False)  # MLP-like
        
        def forward(self, x):
            return self.mlp_up(self.q_proj(x)).sum()

    model = AttentionLikeModel().to(device).bfloat16()

    # 2. Create a dummy mask file
    mask_path = "/tmp/diagnostic_mask.pt"
    # Create 1% sparse masks (very sparse to be sure)
    masks = {
        "q_proj.weight": (torch.rand(128, 128) > 0.99).float().to(device),
        "mlp_up.weight": (torch.rand(512, 128) > 0.99).float().to(device)
    }
    torch.save(masks, mask_path)
    print(f"✓ Created diagnostic mask at {mask_path}")

    # 3. Load Mask Manager
    mask_manager = SparseMaskManager(mask_path, device=device)

    # 4. Inject Sparse Layers (mlp_only=False)
    mask_dict = {n: mask_manager.get_mask(n) for n, _ in model.named_parameters() 
                 if 'weight' in n and mask_manager.has_mask(n)}
    
    print(f"✓ Injecting sparse modules for: {list(mask_dict.keys())}")
    replace_linear_modules(model, mask_dict, block_size=16)

    # 5. Initialize SparseAdamW (mlp_only=False)
    optimizer = SparseAdamW(
        list(model.named_parameters()), 
        mask_manager, 
        lr=1e-3,
        mlp_only=False
    )

    # Record initial weights
    initial_weights = {n: p.clone().detach() for n, p in model.named_parameters()}

    # 6. Run 5 training steps
    print(f"✓ Running 5 diagnostic steps...")
    for _ in range(5):
        optimizer.zero_grad()
        x = torch.randn(4, 128, device=device).bfloat16()
        loss = model(x)
        loss.backward()
        
        # Immediate gradient check
        for name, param in model.named_parameters():
            if param.grad is not None:
                mask = mask_dict[name]
                violation = (param.grad != 0) & (mask == 0)
                if violation.any():
                    print(f"  FAILED: Non-zero gradient found outside mask for {name}")
                    return
        
        optimizer.step()

    # 7. Final Verification
    print("\nFINAL SCAN:")
    all_passed = True
    for name, param in model.named_parameters():
        mask = mask_dict[name]
        
        # Check updates isolation
        diff = (param - initial_weights[name]).abs()
        diff_outside_mask = diff[mask == 0]
        max_diff_outside = diff_outside_mask.max().item()
        
        # Check if updates actually happened inside mask
        diff_inside_mask = diff[mask == 1]
        max_diff_inside = diff_inside_mask.max().item()
        
        print(f"  Layer: {name}")
        print(f"    - Max change outside mask: {max_diff_outside:.2e}")
        print(f"    - Max change inside mask:  {max_diff_inside:.2e}")
        
        if max_diff_outside > 1e-10: # Accounting for epsilon float errors if any
            print(f"    ❌ ISOLATION FAILURE: Weights changed outside mask.")
            all_passed = False
        elif max_diff_inside == 0:
            print(f"    ⚠️ WARNING: No weights changed inside mask (check gradient flow).")
            # This is usually okay if the model converged or x was small, 
            # but in this random test it should be non-zero.
        else:
            print(f"    ✅ ISOLATION PASSED: Only masked weights were updated.")

    # Clean up
    if os.path.exists(mask_path):
        os.remove(mask_path)
    
    if all_passed:
        print("\nSUMMARY: Sparse Attention Masking is WORKING correctly.")
        print("Weights outside the mask received zero gradients and zero updates.")
    else:
        print("\nSUMMARY: Sparse Attention Masking has ISOLATION ISSUES.")

if __name__ == "__main__":
    run_diagnostic()
