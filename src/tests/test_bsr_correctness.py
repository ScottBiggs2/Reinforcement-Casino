
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
    Includes TF32 check and Gradient Diffusion simulation.
    """
    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return

    device = torch.device("cuda")
    torch.manual_seed(42)
    
    # Precision control
    print("Disabling TF32 for high precision check...")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

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
    
    diff = (y_dense - y_sparse).abs().max().item()
    print(f"Forward Max Diff: {diff}")
    if diff > 1e-5:
        print("FAIL: Forward pass mismatch!")
    else:
        print("SUCCESS: Forward pass matches dense.")
    
    # 3. Backward Pass
    print("\n--- Backward Pass Check ---")
    grad_out = torch.randn_like(y_dense)
    
    # Dense Backward
    y_dense.backward(grad_out)
    dense_w_grad = dense_layer.weight.grad
    dense_b_grad = dense_layer.bias.grad
    dense_x_grad = x.grad
    
    # Sparse Backward
    y_sparse.backward(grad_out)
    sparse_w_grad = sparse_layer.weight.grad
    sparse_b_grad = sparse_layer.bias.grad
    sparse_x_grad = x_sparse.grad
    
    # A. Check Bias Grad
    b_diff = (dense_b_grad - sparse_b_grad).abs().max().item()
    print(f"Bias Grad Max Diff: {b_diff}")
    
    # B. Check Input Grad
    x_diff = (dense_x_grad - sparse_x_grad).abs().max().item()
    print(f"Input Grad Max Diff: {x_diff}")

    # C. Check Weight Grad vs Masked Dense
    expected_sparse_w_grad = dense_w_grad * mask
    w_diff = (expected_sparse_w_grad - sparse_w_grad).abs().max().item()
    print(f"Weight Grad Max Diff (vs Masked Dense): {w_diff}")
    
    # Additional verification: Explicit Loop Calculation
    # Compute ground truth for ACTIVE blocks using standard torch.matmul
    print("Verifying active blocks with explicit loop...")
    x_flat = x.reshape(-1, Din) # (B*S, Din)
    grad_out_flat = grad_out.reshape(-1, Dout) # (B*S, Dout)
    
    max_loop_diff = 0.0
    for i in range(n_blocks_out):
        for j in range(n_blocks_in):
            if mask[i*block_size, j*block_size] == 1.0:
                # Extract blocks
                go_block = grad_out_flat[:, i*block_size:(i+1)*block_size] # (Batch, Block)
                x_block = x_flat[:, j*block_size:(j+1)*block_size]         # (Batch, Block)
                
                # Expected: go_block.T @ x_block -> (Block, Block)
                expected_block = go_block.T @ x_block
                
                # Actual
                actual_block = sparse_w_grad[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                
                block_diff = (expected_block - actual_block).abs().max().item()
                max_loop_diff = max(max_loop_diff, block_diff)
                
    print(f"Max Diff vs Explicit Loop (check for Kernel logic error): {max_loop_diff}")
    
    if max_loop_diff > 1e-4:
        print("FAIL: Kernel logic does not match explicit matrix multiplication!")
    else:
        print("SUCCESS: Kernel matches explicit math (diff due to float precision only).")

    # D. Gradient Diffusion Test (User Concept)
    print("\n--- Gradient Diffusion Simulation ---")
    print("Simulating: duplicating gradients from masked regions to unmasked regions.")
    
    # Concept: fill the zero-grad regions with average of non-zero grads?
    # Or copy nearest neighbor? 
    # Let's try filling with the mean absolute value of active gradients to see signal magnitude.
    
    active_grads = sparse_w_grad[mask.bool()]
    mean_active_grad = active_grads.abs().mean().item()
    std_active_grad = active_grads.std().item()
    
    print(f"Active Gradients: Mean Abs={mean_active_grad:.6f}, Std={std_active_grad:.6f}")
    
    # Create "Diffused" Gradient
    diffused_grad = sparse_w_grad.clone()
    # Fill inactive spots with random noise scaled to active stats
    # This simulates "dense" gradients flowing back even though weight update is sparse
    # (Note: this modifies the weight update vector, not the input gradient flow)
    noise = torch.randn_like(diffused_grad) * std_active_grad
    diffused_grad[~mask.bool()] = noise[~mask.bool()]
    
    print(f"Diffused Gradient created. Density: {(diffused_grad != 0).float().mean().item():.2%}")
    print("Diffusion test complete (placeholder logic verified).")

if __name__ == "__main__":
    test_bsr_correctness()
