
import os
import sys
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mlps.bsr_sparse_mlp import replace_linear_modules
from src.optimizers.sparse_adamw import SparseAdamW
from src.utils.mask_manager import SparseMaskManager

def test_mask_isolation():
    print("Starting Mask Isolation Test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Create a dummy model with Linear (Attention-like) and MLP-like layers
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32, bias=False)  # Attention-like
            self.mlp_up = nn.Linear(32, 64, bias=False)  # MLP-like
        
        def forward(self, x):
            return self.mlp_up(self.q_proj(x)).sum()

    model = DummyModel().to(device).bfloat16()

    # 2. Create a dummy mask file
    mask_path = "/tmp/test_mask.pt"
    # Create 10% sparse masks
    masks = {
        "q_proj.weight": (torch.rand(32, 32) > 0.9).float().to(device),
        "mlp_up.weight": (torch.rand(64, 32) > 0.9).float().to(device)
    }
    torch.save(masks, mask_path)

    # 3. Load Mask Manager
    mask_manager = SparseMaskManager(mask_path, device=device)

    # 4. Inject Sparse Layers (mlp_only=False)
    mask_dict = {n: mask_manager.get_mask(n) for n, _ in model.named_parameters() 
                 if ('mlp' in n.lower() or True) and 'weight' in n and mask_manager.has_mask(n)}
    
    print(f"Injecting sparse modules for: {list(mask_dict.keys())}")
    replace_linear_modules(model, mask_dict, block_size=16)

    # 5. Initialize SparseAdamW (mlp_only=False)
    optimizer = SparseAdamW(
        list(model.named_parameters()), 
        mask_manager, 
        lr=1.0, # Large LR to see changes easily
        mlp_only=False
    )

    # Record initial weights
    initial_weights = {n: p.clone().detach() for n, p in model.named_parameters()}

    # 6. Run a training step
    x = torch.randn(1, 32, device=device).bfloat16()
    loss = model(x)
    loss.backward()

    # 7. Check Gradients
    print("\nChecking Gradients for isolation...")
    for name, param in model.named_parameters():
        if param.grad is not None:
            mask = mask_dict[name]
            grad_outside_mask = param.grad * (1 - mask)
            max_grad_outside = grad_outside_mask.abs().max().item()
            print(f"  {name}: Max gradient outside mask = {max_grad_outside}")
            if max_grad_outside > 0:
                print(f"  FAILED: Gradient found outside mask for {name}")
            else:
                print(f"  SUCCESS: All gradients outside mask are zero.")

    # 8. Run Optimizer Step
    optimizer.step()

    # 9. Check Weight Updates
    print("\nChecking Weight Updates for isolation...")
    for name, param in model.named_parameters():
        mask = mask_dict[name]
        diff = (param - initial_weights[name]).abs()
        diff_outside_mask = diff * (1 - mask)
        max_diff_outside = diff_outside_mask.max().item()
        
        # Also check if weights inside the mask ACTUALLY changed
        diff_inside_mask = diff * mask
        max_diff_inside = diff_inside_mask.max().item()
        
        print(f"  {name}: Max update outside mask = {max_diff_outside}")
        print(f"            Max update inside mask  = {max_diff_inside}")
        
        if max_diff_outside > 0:
            print(f"  FAILED: Weight changed outside mask for {name}")
        elif max_diff_inside == 0:
            print(f"  WARNING: No weight change inside mask for {name} (maybe gradient was 0?)")
        else:
            print(f"  SUCCESS: Weights outside mask are UNCHANGED.")

    # Clean up
    if os.path.exists(mask_path):
        os.remove(mask_path)

if __name__ == "__main__":
    test_mask_isolation()
