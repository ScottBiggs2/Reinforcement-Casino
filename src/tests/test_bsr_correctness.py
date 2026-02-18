
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import will fail if we don't have src in path, but creating the file works locally
try:
    from src.mlps.bsr_sparse_mlp import SparseLinearLayer
except ImportError:
    pass

def test_bsr_correctness():
    """
    Test "Dense Forward, Sparse Backward" behavior.
    
    Expectations:
    1. Forward pass: Identical to standard nn.Linear (uses all weights).
    2. Input Grad: Identical to standard nn.Linear (uses all weights).
    3. Bias Grad: Identical to standard nn.Linear.
    4. Weight Grad: Sparse! Should equal (dense_weight_grad * mask).
    """
    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return

    device = torch.device("cuda")
    torch.manual_seed(42)

    # Config
    B, S, Din, Dout = 2, 16, 64, 128
    block_size = 16
    
    print(f"Testing on {device}")
    
    # 1. Setup Dense Reference
    dense_layer = nn.Linear(Din, Dout).to(device)
    
    # Create a block-sparse mask
    mask = torch.zeros(Dout, Din, device=device)
    n_blocks_out = Dout // block_size
    n_blocks_in = Din // block_size
    for i in range(n_blocks_out):
        for j in range(n_blocks_in):
            if torch.rand(1).item() > 0.5:
                mask[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = 1.0
    
    print(f"Sparsity: {1.0 - mask.mean().item():.2%}")
    
    # Copy weights to sparse layer EXACTLY
    # Note: We do NOT mask the weights here. The user wants dense forward.
    sparse_layer = SparseLinearLayer(Din, Dout, mask=mask, block_size=block_size).to(device)
    with torch.no_grad():
        sparse_layer.weight.data = dense_layer.weight.data.clone()
        if dense_layer.bias is not None:
            sparse_layer.bias.data = dense_layer.bias.data.clone()

    # Inputs
    x = torch.randn(B, S, Din, device=device, requires_grad=True)
    x_sparse = x.clone().detach().requires_grad_(True)
    
    # 2. Forward Pass Check
    print("--- Forward Pass Check ---")
    y_dense = dense_layer(x)
    y_sparse = sparse_layer(x_sparse)
    
    # Should be identical because SparseLinearLayer forward calls F.linear(input, weight)
    diff = (y_dense - y_sparse).abs().max().item()
    print(f"Forward Max Diff: {diff}")
    if diff > 1e-5:
        print("FAIL: Forward pass mismatch! (Should be identical to dense)")
    else:
        print("SUCCESS: Forward pass matches dense.")
    
    # 3. Backward Pass
    print("\n--- Backward Pass Check ---")
    grad_out = torch.randn_like(y_dense)
    
    # Dense Backward
    y_dense.backward(grad_out)
    
    # Standard dense gradients
    dense_w_grad = dense_layer.weight.grad
    dense_b_grad = dense_layer.bias.grad
    dense_x_grad = x.grad
    
    # Sparse Backward
    y_sparse.backward(grad_out)
    sparse_w_grad = sparse_layer.weight.grad
    sparse_b_grad = sparse_layer.bias.grad
    sparse_x_grad = x_sparse.grad
    
    # A. Check Bias Grad (Should match dense)
    b_diff = (dense_b_grad - sparse_b_grad).abs().max().item()
    print(f"Bias Grad Max Diff: {b_diff}")
    if b_diff > 1e-3:
        print("FAIL: Bias gradient mismatch!")
    else:
        print("SUCCESS: Bias gradient matches dense.")

    # B. Check Input Grad (Should match dense because forward weight was dense)
    x_diff = (dense_x_grad - sparse_x_grad).abs().max().item()
    print(f"Input Grad Max Diff: {x_diff}")
    if x_diff > 1e-3:
        print("FAIL: Input gradient mismatch!")
    else:
        print("SUCCESS: Input gradient matches dense (as expected).")

    # C. Check Weight Grad (Should be SPARSE: dense_grad * mask)
    expected_sparse_w_grad = dense_w_grad * mask
    w_diff = (expected_sparse_w_grad - sparse_w_grad).abs().max().item()
    print(f"Weight Grad Max Diff (vs Masked Dense): {w_diff}")
    
    # Also check that we didn't just compute dense grad
    dense_vs_sparse_w_diff = (dense_w_grad - sparse_w_grad).abs().max().item()
    print(f"Weight Grad Diff vs Unmasked Dense: {dense_vs_sparse_w_diff}")
    
    if w_diff > 1e-3:
        print("FAIL: Sparse weight gradient does not match (dense_grad * mask)!")
        
        # Debugging: check if it's completely zero or something
        if sparse_w_grad.abs().max().item() == 0:
            print("DEBUG: Sparse gradient is all zeros!")
    else:
        print("SUCCESS: Weight gradient is correctly masked.")

if __name__ == "__main__":
    test_bsr_correctness()
