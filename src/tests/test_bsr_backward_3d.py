
import torch
import torch.nn as nn
from src.mlps.bsr_sparse_mlp import SparseLinearLayer

def test_bsr_sparse_mlp_3d_backward():
    """
    Verifies that SparseLinearLayer handles 3D inputs (batch, seq, dim) correctly
    during backward pass.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Skipping test because CUDA is not available.")
        return

    print(f"Testing on {device}")
    
    # Setup dimensions
    batch_size = 2
    seq_len = 16
    in_features = 64
    out_features = 128
    block_size = 16
    
    # Create a random mask
    mask = torch.randint(0, 2, (out_features, in_features), device=device).float()
    
    # Initialize layer
    layer = SparseLinearLayer(in_features, out_features, mask=mask, block_size=block_size).to(device)
    
    # Create 3D input
    input_tensor = torch.randn(batch_size, seq_len, in_features, device=device, requires_grad=True)
    
    # Forward pass
    print("Running forward pass...")
    output = layer(input_tensor)
    
    assert output.shape == (batch_size, seq_len, out_features), f"Output shape mismatch: {output.shape}"
    
    # Backward pass
    print("Running backward pass...")
    grad_output = torch.randn_like(output)
    
    try:
        output.backward(grad_output)
        print("Backward pass successful!")
    except ValueError as e:
        print(f"FAIL: Backward pass failed with ValueError: {e}")
        raise e
    except Exception as e:
        print(f"FAIL: Backward pass failed with unexpected error: {e}")
        raise e

    # Verify gradients exist
    assert layer.weight.grad is not None, "Weight gradient is None"
    assert layer.bias.grad is not None, "Bias gradient is None"
    assert input_tensor.grad is not None, "Input gradient is None"
    
    # Basic shape checks for gradients
    assert layer.weight.grad.shape == (out_features, in_features)
    assert layer.bias.grad.shape == (out_features,)
    assert input_tensor.grad.shape == (batch_size, seq_len, in_features)

    print("All checks passed.")

if __name__ == "__main__":
    test_bsr_sparse_mlp_3d_backward()
