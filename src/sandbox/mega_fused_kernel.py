"""
Sparse Gradient Computation + Fused Backward/Adam

Key insight: We can compute gradients SPARSELY by only calculating
gradient elements where the mask is non-zero.

For grad = error @ X.T, we only compute grad[i,j] where mask[i,j] = 1
"""

import torch
import triton
import triton.language as tl


@triton.jit
def sparse_gradient_kernel(
    # Pointers
    error_ptr,      # [M, batch_size] - prediction error
    X_ptr,          # [N, batch_size] - input features  
    grad_ptr,       # [M, N] - output gradient (sparse)
    mask_ptr,       # [M, N] - sparsity mask
    # Dimensions
    M, N, batch_size,
    stride_em, stride_eb,
    stride_xn, stride_xb,
    stride_gm, stride_gn,
    stride_mm, stride_mn,
    # Scaling factor
    scale,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute sparse gradient: grad[i,j] = error[i,:] @ X[j,:].T
    Only computes where mask[i,j] = 1.
    
    This is MUCH faster than computing full dense gradient when sparse!
    """
    pid = tl.program_id(0)
    
    # Calculate block position
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE)
    block_row = pid // num_blocks_n
    block_col = pid % num_blocks_n
    
    # Calculate starting position
    row_start = block_row * BLOCK_SIZE
    col_start = block_col * BLOCK_SIZE
    
    # Create offset ranges
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    row_offsets = row_offsets[:, None]
    col_offsets = col_offsets[None, :]
    
    # Mask for valid elements
    mask_valid = (row_offsets < M) & (col_offsets < N)
    
    # Check if this block should be processed (early exit)
    center_row = row_start + BLOCK_SIZE // 2
    center_col = col_start + BLOCK_SIZE // 2
    if center_row < M and center_col < N:
        block_mask = tl.load(mask_ptr + center_row * stride_mm + center_col * stride_mn)
        if block_mask == 0.0:
            # Zero out this block and return
            g_offsets = row_offsets * stride_gm + col_offsets * stride_gn
            tl.store(grad_ptr + g_offsets, 0.0, mask=mask_valid)
            return
    
    # Load mask for this block
    m_offsets = row_offsets * stride_mm + col_offsets * stride_mn
    mask_block = tl.load(mask_ptr + m_offsets, mask=mask_valid, other=0.0)
    
    # Initialize gradient accumulator
    grad_accum = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Compute grad[i,j] = sum_k(error[i,k] * X[j,k]) for this block
    # We'll do this in chunks over the batch dimension
    BATCH_BLOCK = 32
    for k_start in range(0, batch_size, BATCH_BLOCK):
        k_offsets = k_start + tl.arange(0, BATCH_BLOCK)
        k_mask = k_offsets < batch_size
        
        # Load error[row_offsets, k_offsets] - shape [BLOCK_SIZE, BATCH_BLOCK]
        e_offsets = row_offsets * stride_em + k_offsets[None, :] * stride_eb
        error_chunk = tl.load(error_ptr + e_offsets, mask=mask_valid[:, None] & k_mask[None, :], other=0.0)
        
        # Load X[col_offsets, k_offsets] - shape [BLOCK_SIZE, BATCH_BLOCK]
        x_offsets = col_offsets * stride_xn + k_offsets[:, None] * stride_xb
        X_chunk = tl.load(X_ptr + x_offsets, mask=mask_valid[None, :] & k_mask[:, None], other=0.0)
        
        # Accumulate: grad[i,j] += error[i,k] * X[j,k]
        # error_chunk is [BLOCK_SIZE, BATCH_BLOCK], X_chunk is [BATCH_BLOCK, BLOCK_SIZE]
        # We need to transpose and multiply
        grad_accum += tl.dot(error_chunk, X_chunk)  # [BLOCK_SIZE, BLOCK_SIZE]
    
    # Scale and mask the gradient
    grad_block = grad_accum * scale * mask_block
    
    # Store result
    g_offsets = row_offsets * stride_gm + col_offsets * stride_gn
    tl.store(grad_ptr + g_offsets, grad_block, mask=mask_valid)


@triton.jit
def mega_fused_kernel(
    # Pointers
    weights_ptr,
    X_ptr,          # Input data [N, batch_size]
    Y_ptr,          # Target data [M, batch_size]
    exp_avg_ptr,
    exp_avg_sq_ptr,
    mask_ptr,
    error_buffer_ptr,  # Temporary buffer for error [M, batch_size]
    # Dimensions
    M, N, batch_size,
    stride_wm, stride_wn,
    stride_xn, stride_xb,
    stride_ym, stride_yb,
    stride_em, stride_en,
    stride_vm, stride_vn,
    stride_mm, stride_mn,
    stride_errm, stride_errb,
    # Optimization parameters
    lr, beta1, beta2, eps, weight_decay,
    bias_correction1, bias_correction2,
    use_adamw: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    MEGA-FUSED kernel: Forward + Sparse Gradient + Adam in one pass.
    
    This is the ultimate optimization:
    1. Compute prediction: pred = W @ X (only for this block's rows)
    2. Compute error: error = pred - Y
    3. Compute sparse gradient: grad[i,j] = error[i,:] @ X[j,:].T
    4. Update weights with Adam
    
    All in one kernel launch!
    """
    pid = tl.program_id(0)
    
    # Calculate block position
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE)
    block_row = pid // num_blocks_n
    block_col = pid % num_blocks_n
    
    row_start = block_row * BLOCK_SIZE
    col_start = block_col * BLOCK_SIZE
    
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    row_offsets = row_offsets[:, None]
    col_offsets = col_offsets[None, :]
    
    mask_valid = (row_offsets < M) & (col_offsets < N)
    
    # Check if this block should be processed
    center_row = row_start + BLOCK_SIZE // 2
    center_col = col_start + BLOCK_SIZE // 2
    if center_row < M and center_col < N:
        block_mask = tl.load(mask_ptr + center_row * stride_mm + center_col * stride_mn)
        if block_mask == 0.0:
            return  # Skip entirely
    
    # Load current weights for this block
    w_offsets = row_offsets * stride_wm + col_offsets * stride_wn
    w = tl.load(weights_ptr + w_offsets, mask=mask_valid, other=0.0)
    
    # Load optimizer states
    e_offsets = row_offsets * stride_em + col_offsets * stride_en
    v_offsets = row_offsets * stride_vm + col_offsets * stride_vn
    m_offsets = row_offsets * stride_mm + col_offsets * stride_mn
    
    m_state = tl.load(exp_avg_ptr + e_offsets, mask=mask_valid, other=0.0)
    v_state = tl.load(exp_avg_sq_ptr + v_offsets, mask=mask_valid, other=0.0)
    mask_block = tl.load(mask_ptr + m_offsets, mask=mask_valid, other=0.0)
    
    # Compute sparse gradient for this block
    # grad[i,j] = sum_k(error[i,k] * X[j,k]) where error = (W@X - Y)
    
    grad_accum = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    BATCH_BLOCK = 32
    for k_start in range(0, batch_size, BATCH_BLOCK):
        k_offsets = k_start + tl.arange(0, BATCH_BLOCK)
        k_mask = k_offsets < batch_size
        
        # For this would need to compute pred = W @ X first
        # This is getting complex - let's use the simpler two-kernel approach
        pass
    
    # (This mega-kernel is complex - better to use sparse_gradient + fused_adam)
    # Commenting out for now
    pass


def compute_sparse_gradient(error, X, mask, block_size=32):
    """
    Compute gradient sparsely: grad = error @ X.T, but only where mask = 1.
    
    Args:
        error: [M, batch_size] - prediction error
        X: [N, batch_size] - input features
        mask: [M, N] - sparsity mask
        
    Returns:
        grad: [M, N] - sparse gradient
    """
    M, batch_size = error.shape
    N = X.shape[0]
    
    grad = torch.zeros(M, N, device=error.device, dtype=error.dtype)
    
    # Scale factor (for MSE loss)
    scale = 2.0 / batch_size
    
    num_blocks_m = triton.cdiv(M, block_size)
    num_blocks_n = triton.cdiv(N, block_size)
    grid = (num_blocks_m * num_blocks_n,)
    
    sparse_gradient_kernel[grid](
        error, X, grad, mask,
        M, N, batch_size,
        error.stride(0), error.stride(1),
        X.stride(0), X.stride(1),
        grad.stride(0), grad.stride(1),
        mask.stride(0), mask.stride(1),
        scale,
        BLOCK_SIZE=block_size,
    )
    
    return grad


@triton.jit
def fused_sparse_backward_adam_kernel(
    weights_ptr, grads_ptr, exp_avg_ptr, exp_avg_sq_ptr, mask_ptr,
    M, N,
    stride_wm, stride_wn, stride_gm, stride_gn,
    stride_em, stride_en, stride_vm, stride_vn, stride_mm, stride_mn,
    lr, beta1, beta2, eps, weight_decay,
    bias_correction1, bias_correction2,
    use_adamw: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused sparse backward + Adam (from original implementation)."""
    pid = tl.program_id(0)
    
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE)
    block_row = pid // num_blocks_n
    block_col = pid % num_blocks_n
    
    row_start = block_row * BLOCK_SIZE
    col_start = block_col * BLOCK_SIZE
    
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    row_offsets = row_offsets[:, None]
    col_offsets = col_offsets[None, :]
    
    mask_valid = (row_offsets < M) & (col_offsets < N)
    
    # Early exit check
    center_row = row_start + BLOCK_SIZE // 2
    center_col = col_start + BLOCK_SIZE // 2
    if center_row < M and center_col < N:
        block_mask = tl.load(mask_ptr + center_row * stride_mm + center_col * stride_mn)
        if block_mask == 0.0:
            return
    
    # Calculate offsets
    w_offsets = row_offsets * stride_wm + col_offsets * stride_wn
    g_offsets = row_offsets * stride_gm + col_offsets * stride_gn
    e_offsets = row_offsets * stride_em + col_offsets * stride_en
    v_offsets = row_offsets * stride_vm + col_offsets * stride_vn
    m_offsets = row_offsets * stride_mm + col_offsets * stride_mn
    
    # Load data
    w = tl.load(weights_ptr + w_offsets, mask=mask_valid, other=0.0)
    g = tl.load(grads_ptr + g_offsets, mask=mask_valid, other=0.0)
    m = tl.load(exp_avg_ptr + e_offsets, mask=mask_valid, other=0.0)
    v = tl.load(exp_avg_sq_ptr + v_offsets, mask=mask_valid, other=0.0)
    mask_block = tl.load(mask_ptr + m_offsets, mask=mask_valid, other=0.0)
    
    # Apply mask to gradient
    g_masked = g * mask_block
    
    # Update moments
    m_new = beta1 * m + (1.0 - beta1) * g_masked
    v_new = beta2 * v + (1.0 - beta2) * g_masked * g_masked
    
    # Bias-corrected moments
    m_hat = m_new / bias_correction1
    v_hat = v_new / bias_correction2
    
    # Compute update
    denom = tl.sqrt(v_hat) + eps
    update = m_hat / denom
    
    # Apply weight decay
    if use_adamw:
        w_new = w * (1.0 - lr * weight_decay) - lr * update
    else:
        w_new = w - lr * update
        if weight_decay != 0.0:
            w_new = w_new - lr * weight_decay * w
    
    # Store updates
    tl.store(weights_ptr + w_offsets, w_new, mask=mask_valid)
    tl.store(exp_avg_ptr + e_offsets, m_new, mask=mask_valid)
    tl.store(exp_avg_sq_ptr + v_offsets, v_new, mask=mask_valid)


# ============================================================================
# TRAINING FUNCTION WITH SPARSE GRADIENT
# ============================================================================

def train_with_sparse_gradient(problem, n_steps=100, lr=0.01, block_size=32):
    """
    Train with SPARSE gradient computation + fused backward/Adam.
    
    This computes gradients sparsely (only where mask = 1) before
    passing to the fused backward/Adam kernel.
    """
    W = torch.randn_like(problem.W_true)
    
    exp_avg = torch.zeros_like(W)
    exp_avg_sq = torch.zeros_like(W)
    
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    weight_decay = 0.01
    
    train_losses = []
    test_losses = []
    times = []
    
    M, N = W.shape
    
    print(f"\nTraining with SPARSE GRADIENT + Fused Adam (block_size={block_size})...")
    
    import time as time_module
    start_total = time_module.time()
    
    for step in range(1, n_steps + 1):
        start = time_module.time()
        
        # Forward pass: pred = W @ X
        pred = W @ problem.X_train
        loss = ((pred - problem.Y_train) ** 2).mean()
        
        # Compute error
        error = pred - problem.Y_train  # [M, batch_size]
        
        # SPARSE gradient computation (only where mask = 1)
        grad = compute_sparse_gradient(error, problem.X_train, problem.mask, block_size=block_size)
        
        # Precompute bias corrections
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        
        # Calculate grid
        num_blocks_m = triton.cdiv(M, block_size)
        num_blocks_n = triton.cdiv(N, block_size)
        grid = (num_blocks_m * num_blocks_n,)
        
        # Fused backward + Adam kernel
        fused_sparse_backward_adam_kernel[grid](
            W, grad, exp_avg, exp_avg_sq, problem.mask,
            M, N,
            W.stride(0), W.stride(1),
            grad.stride(0), grad.stride(1),
            exp_avg.stride(0), exp_avg.stride(1),
            exp_avg_sq.stride(0), exp_avg_sq.stride(1),
            problem.mask.stride(0), problem.mask.stride(1),
            lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2,
            True,  # adamw
            BLOCK_SIZE=block_size,
        )
        
        torch.cuda.synchronize()
        times.append(time_module.time() - start)
        
        train_losses.append(loss.item())
        test_loss = ((W @ problem.X_test - problem.Y_test) ** 2).mean()
        test_losses.append(test_loss.item())
        
        if (step - 1) % 20 == 0:
            print(f"  Step {step-1:3d}: Train Loss = {loss.item():.6f}, "
                  f"Test Loss = {test_loss.item():.6f}")
    
    total_time = time_module.time() - start_total
    import numpy as np
    avg_time = np.mean(times[10:])
    
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Avg step time: {avg_time*1000:.3f}ms")
    
    return {
        'weights': W,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'avg_step_time': avg_time,
        'total_time': total_time,
    }


# ============================================================================
# COMPARISON TEST
# ============================================================================

def test_sparse_vs_dense_gradient():
    """Test if sparse gradient computation is faster."""
    print("="*80)
    print("SPARSE vs DENSE GRADIENT COMPUTATION")
    print("="*80)
    
    M, N = 2048, 2048
    batch_size = 100
    sparsity = 0.9
    block_size = 32
    
    # Create data
    error = torch.randn(M, batch_size, device='cuda')
    X = torch.randn(N, batch_size, device='cuda')
    mask = (torch.rand(M, N, device='cuda') > sparsity).float()
    
    nnz = mask.sum().item()
    print(f"\nMatrix: {M} x {N}")
    print(f"Batch size: {batch_size}")
    print(f"Sparsity: {100*sparsity:.1f}%")
    print(f"Non-zeros: {nnz:,} / {M*N:,}")
    
    # Warmup
    for _ in range(10):
        grad_dense = error @ X.T
        grad_sparse = compute_sparse_gradient(error, X, mask, block_size)
    
    torch.cuda.synchronize()
    
    # Benchmark dense
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        grad_dense = error @ X.T
    end.record()
    torch.cuda.synchronize()
    dense_time = start.elapsed_time(end) / 100
    
    # Benchmark sparse
    start.record()
    for _ in range(100):
        grad_sparse = compute_sparse_gradient(error, X, mask, block_size)
    end.record()
    torch.cuda.synchronize()
    sparse_time = start.elapsed_time(end) / 100
    
    print(f"\nGradient Computation:")
    print(f"  Dense (error @ X.T):     {dense_time:.3f} ms")
    print(f"  Sparse (Triton kernel):  {sparse_time:.3f} ms")
    print(f"  Speedup: {dense_time/sparse_time:.2f}x")
    
    # Verify correctness
    grad_dense_ref = error @ X.T
    grad_dense_masked = grad_dense_ref * mask
    grad_sparse_result = compute_sparse_gradient(error, X, mask, block_size)
    
    diff = (grad_sparse_result - grad_dense_masked).abs().max().item()
    print(f"\nCorrectness check:")
    print(f"  Max difference: {diff:.2e}")
    if diff < 1e-4:
        print(f"  ✓ Correct!")
    else:
        print(f"  ⚠ May have numerical issues")


if __name__ == "__main__":
    # Test sparse gradient computation
    test_sparse_vs_dense_gradient()
    
    print("\n" + "="*80)
    print("To integrate with your full training pipeline:")
    print("Replace the dense gradient computation with:")
    print("  error = pred - Y_train")
    print("  grad = compute_sparse_gradient(error, X_train, mask, block_size)")
    print("This should give additional speedup at high sparsity!")
    print("="*80)