
import torch
import torch.nn as nn
from src.mlps.bsr_sparse_mlp import SparseLinearLayer
from src.optimizers.sparse_adamw import SparseAdamW
from src.utils.mask_manager import SparseMaskManager
import os
import tempfile

def test_sparse_consistency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    in_features = 64
    out_features = 64
    block_size = 16
    
    # Create a dummy mask (10% sparse)
    mask = torch.zeros(out_features, in_features)
    mask[:block_size, :block_size] = 1.0
    mask[block_size:2*block_size, block_size:2*block_size] = 1.0
    
    # Save mask to temp file
    fd, mask_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save({"mask.weight": mask.bool()}, mask_path)
    
    mask_manager = SparseMaskManager(mask_path, device=device)
    
    # Create SparseLinearLayer
    layer = SparseLinearLayer(in_features, out_features, bias=True, mask=mask, block_size=block_size)
    layer.to(device)
    
    # Verify weight masking
    assert (layer.weight * (1.0 - mask.to(device))).abs().sum() == 0, "Weights not masked correctly in set_mask"
    print("✓ Weight masking verified")
    
    # Create optimizer
    optimizer = SparseAdamW(
        [("mask.weight", layer.weight)],
        mask_manager,
        lr=1e-3,
        eager_state_init=True
    )
    
    # Verify sparse state allocation
    state = optimizer.state[layer.weight]
    assert state["exp_avg"].dim() == 1, f"exp_avg should be 1D for sparse states, got {state['exp_avg'].dim()}"
    assert state["exp_avg"].shape[0] == mask.sum().item(), f"exp_avg size mismatch: {state['exp_avg'].shape[0]} vs {mask.sum().item()}"
    print("✓ Sparse optimizer states verified (VRAM saving active)")
    
    # Dummy forward/backward
    input = torch.randn(8, in_features, device=device, requires_grad=True)
    output = layer(input)
    loss = output.sum()
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    print("✓ Optimizer step completed without crash")
    
    # Check that non-masked weights remained zero
    assert (layer.weight * (1.0 - mask.to(device))).abs().sum() == 0, "Update leaked into masked weights"
    print("✓ Mathematical equivalence (sparsity preservation) verified")
    
    os.unlink(mask_path)
    print("\nALL TESTS PASSED")

if __name__ == "__main__":
    test_sparse_consistency()
