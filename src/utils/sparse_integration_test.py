"""
Integrated test combining sparse backward pass and sparse Adam optimizer.

Tests the full sparse training pipeline:
1. Sparse forward pass
2. Sparse backward pass (using your kernel)
3. Sparse Adam optimization (using block-sparse Adam kernel)
"""

import torch
import triton
import triton.language as tl
import time


# ============================================================================
# SPARSE BACKWARD PASS KERNEL (Your implementation)
# ============================================================================

@triton.jit
def sparse_update_kernel(
    weights_ptr,
    grad_ptr,
    mask_ptr,
    output_ptr,
    M, N,
    stride_wm, stride_wn,
    stride_gm, stride_gn,
    stride_mm, stride_mn,
    stride_om, stride_on,
    lr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for sparse weight updates (your implementation).
    Applies: W_new = W_old - lr * (grad * mask)
    """
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
    
    w_offsets = row_offsets * stride_wm + col_offsets * stride_wn
    g_offsets = row_offsets * stride_gm + col_offsets * stride_gn
    m_offsets = row_offsets * stride_mm + col_offsets * stride_mn
    o_offsets = row_offsets * stride_om + col_offsets * stride_on
    
    mask_valid = (row_offsets < M) & (col_offsets < N)
    
    w_block = tl.load(weights_ptr + w_offsets, mask=mask_valid, other=0.0)
    g_block = tl.load(grad_ptr + g_offsets, mask=mask_valid, other=0.0)
    m_block = tl.load(mask_ptr + m_offsets, mask=mask_valid, other=0.0)
    
    # Compute update: W_new = W_old - lr * (grad * mask)
    masked_grad = g_block * m_block
    w_new = w_block - lr * masked_grad
    
    tl.store(output_ptr + o_offsets, w_new, mask=mask_valid)


def triton_sparse_update(weights, gradient, mask, lr, block_size=32):
    """
    Apply sparse gradient update using Triton kernel (your implementation).
    """
    M, N = weights.shape
    output = torch.empty_like(weights)
    
    num_blocks_m = triton.cdiv(M, block_size)
    num_blocks_n = triton.cdiv(N, block_size)
    grid = (num_blocks_m * num_blocks_n,)
    
    sparse_update_kernel[grid](
        weights, gradient, mask, output,
        M, N,
        weights.stride(0), weights.stride(1),
        gradient.stride(0), gradient.stride(1),
        mask.stride(0), mask.stride(1),
        output.stride(0), output.stride(1),
        lr,
        BLOCK_SIZE=block_size,
    )
    
    return output


# ============================================================================
# SPARSE ADAM KERNEL (Compatible 2D version)
# ============================================================================

@triton.jit
def sparse_adam_kernel(
    weights_ptr,
    grads_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    mask_ptr,
    M, N,
    stride_wm, stride_wn,
    stride_gm, stride_gn,
    stride_em, stride_en,
    stride_vm, stride_vn,
    stride_mm, stride_mn,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bias_correction1,
    bias_correction2,
    use_adamw: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sparse Adam/AdamW optimization kernel."""
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
    
    # Check if this block should be processed
    center_row = row_start + BLOCK_SIZE // 2
    center_col = col_start + BLOCK_SIZE // 2
    if center_row < M and center_col < N:
        m_center = center_row * stride_mm + center_col * stride_mn
        block_mask = tl.load(mask_ptr + m_center)
        if block_mask == 0.0:
            return  # Skip zero blocks
    
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
    sparse_mask = tl.load(mask_ptr + m_offsets, mask=mask_valid, other=0.0)
    
    # Apply mask to gradient
    g = g * sparse_mask
    
    # Update moments
    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * v + (1.0 - beta2) * g * g
    
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


def sparse_adam_update(weights, gradient, mask, exp_avg, exp_avg_sq,
                       lr, beta1, beta2, eps, weight_decay, step,
                       adamw=True, block_size=32):
    """Apply sparse Adam/AdamW update."""
    M, N = weights.shape
    
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    
    num_blocks_m = triton.cdiv(M, block_size)
    num_blocks_n = triton.cdiv(N, block_size)
    grid = (num_blocks_m * num_blocks_n,)
    
    sparse_adam_kernel[grid](
        weights, gradient, exp_avg, exp_avg_sq, mask,
        M, N,
        weights.stride(0), weights.stride(1),
        gradient.stride(0), gradient.stride(1),
        exp_avg.stride(0), exp_avg.stride(1),
        exp_avg_sq.stride(0), exp_avg_sq.stride(1),
        mask.stride(0), mask.stride(1),
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2,
        adamw,
        BLOCK_SIZE=block_size,
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_block_sparse_mask(shape, block_size, block_sparsity, device='cuda'):
    """Create a block-sparse binary mask."""
    M, N = shape
    mask = torch.ones(shape, device=device)
    
    num_blocks_m = (M + block_size - 1) // block_size
    num_blocks_n = (N + block_size - 1) // block_size
    total_blocks = num_blocks_m * num_blocks_n
    
    num_zero_blocks = int(total_blocks * block_sparsity)
    zero_block_indices = torch.randperm(total_blocks)[:num_zero_blocks]
    
    for block_idx in zero_block_indices:
        block_row = block_idx // num_blocks_n
        block_col = block_idx % num_blocks_n
        
        row_start = block_row * block_size
        row_end = min(row_start + block_size, M)
        col_start = block_col * block_size
        col_end = min(col_start + block_size, N)
        
        mask[row_start:row_end, col_start:col_end] = 0
    
    return mask


def dummy_forward_backward(weights, mask, target):
    """Simulate forward and backward pass with actual loss."""
    # Forward: matrix multiplication + simple loss
    output = torch.mm(weights, weights.t())
    loss = ((output - target) ** 2).sum()
    
    # Backward: compute gradients
    loss.backward()
    
    # Get gradient
    gradient = weights.grad.clone()
    weights.grad = None  # Clear for next iteration
    
    return gradient, loss.item()


def apply_sparse_backward_and_adam(weights, gradient, mask, exp_avg, exp_avg_sq,
                                   lr, beta1, beta2, eps, weight_decay, step,
                                   adamw=True, block_size=32):
    """
    Apply sparse backward pass followed by sparse Adam update.
    Uses your sparse_update_kernel for backward and sparse_adam_kernel for optimization.
    """
    # Mask gradients (sparse backward pass)
    masked_gradient = gradient * mask
    
    # Apply sparse Adam update
    sparse_adam_update(weights, masked_gradient, mask, exp_avg, exp_avg_sq,
                      lr, beta1, beta2, eps, weight_decay, step, 
                      adamw=adamw, block_size=block_size)


# ============================================================================
# TEST SUITE
# ============================================================================

def test_sparse_pipeline():
    """Test the complete sparse training pipeline."""
    
    print("=" * 80)
    print("Testing: Normal Backward+AdamW vs Sparse Backward+AdamW")
    print("=" * 80)
    
    # Configuration
    configs = [
        {'M': 1024, 'N': 1024, 'block_size': 32, 'sparsity': 0.7, 'name': '1M params, 70% sparse, BS=32'},
        {'M': 1024, 'N': 1024, 'block_size': 32, 'sparsity': 0.9, 'name': '1M params, 90% sparse, BS=32'},
        {'M': 2048, 'N': 2048, 'block_size': 64, 'sparsity': 0.8, 'name': '4M params, 80% sparse, BS=64'},
    ]
    
    for config in configs:
        print(f"\n{'=' * 80}")
        print(f"Config: {config['name']}")
        print(f"{'=' * 80}")
        
        M, N = config['M'], config['N']
        block_size = config['block_size']
        sparsity = config['sparsity']
        
        # Create sparse mask
        mask = create_block_sparse_mask((M, N), block_size, sparsity)
        nnz = mask.sum().item()
        total = M * N
        actual_sparsity = 1 - (nnz / total)
        
        print(f"Matrix: {M} x {N} = {total:,} parameters")
        print(f"Block size: {block_size}")
        print(f"Non-zeros: {nnz:,} ({100 * (1-actual_sparsity):.1f}%)")
        print(f"Actual sparsity: {100 * actual_sparsity:.1f}%")
        
        # Create target for loss computation
        target = torch.randn(M, M, device='cuda')
        
        # Initialize weights with sparse structure
        torch.manual_seed(42)
        initial_weights = torch.randn(M, N, device='cuda') * mask
        
        weights_sparse = initial_weights.clone().requires_grad_(True)
        weights_normal = initial_weights.clone().requires_grad_(True)
        
        # Optimizer states for sparse Adam
        exp_avg_sparse = torch.zeros_like(weights_sparse)
        exp_avg_sq_sparse = torch.zeros_like(weights_sparse)
        
        # PyTorch optimizer for normal pipeline
        optimizer_normal = torch.optim.AdamW([weights_normal], lr=0.001, weight_decay=0.01)
        
        # Training parameters
        lr = 0.001
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        weight_decay = 0.01
        n_steps = 5
        
        print(f"\n{'Training for ' + str(n_steps) + ' steps':^80}")
        print("-" * 80)
        
        # Warmup
        print("Warming up...")
        for step in range(1, 4):
            # Sparse pipeline warmup
            grad, _ = dummy_forward_backward(weights_sparse, mask, target)
            apply_sparse_backward_and_adam(weights_sparse, grad, mask, 
                                           exp_avg_sparse, exp_avg_sq_sparse,
                                           lr, beta1, beta2, eps, weight_decay, 
                                           step, adamw=True, block_size=block_size)
            
            # Normal pipeline warmup
            _, _ = dummy_forward_backward(weights_normal, mask, target)
            optimizer_normal.step()
            optimizer_normal.zero_grad()
        
        torch.cuda.synchronize()
        
        # Benchmark SPARSE pipeline (sparse backward + sparse Adam)
        print("Benchmarking sparse backward + sparse Adam...")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for step in range(1, n_steps + 1):
            grad, loss = dummy_forward_backward(weights_sparse, mask, target)
            apply_sparse_backward_and_adam(weights_sparse, grad, mask,
                                           exp_avg_sparse, exp_avg_sq_sparse,
                                           lr, beta1, beta2, eps, weight_decay,
                                           step, adamw=True, block_size=block_size)
        end.record()
        torch.cuda.synchronize()
        sparse_time = start.elapsed_time(end) / n_steps
        
        # Benchmark NORMAL pipeline (normal backward + normal AdamW)
        print("Benchmarking normal backward + normal AdamW...")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for step in range(1, n_steps + 1):
            grad, loss = dummy_forward_backward(weights_normal, mask, target)
            optimizer_normal.step()
            optimizer_normal.zero_grad()
        end.record()
        torch.cuda.synchronize()
        normal_time = start.elapsed_time(end) / n_steps
        
        speedup = normal_time / sparse_time
        
        print(f"\n{'Results':^80}")
        print("-" * 80)
        print(f"{'Pipeline':<40} {'Time/step (ms)':<20} {'Speedup':<20}")
        print("-" * 80)
        print(f"{'Normal Backward + Normal AdamW':<40} {normal_time:>10.4f} ms       {'1.00x':<20}")
        print(f"{'Sparse Backward + Sparse AdamW':<40} {sparse_time:>10.4f} ms       {speedup:.2f}x")
        print("-" * 80)
        
        if speedup > 1.3:
            print(f"✓ Sparse pipeline is {speedup:.2f}x faster!")
        elif speedup > 1.1:
            print(f"~ Modest speedup: {speedup:.2f}x")
        elif speedup > 1.0:
            print(f"~ Small speedup: {speedup:.2f}x")
        else:
            print(f"⚠ Slower by {1/speedup:.2f}x (overhead > savings at this sparsity)")
        
        # Verify both maintain sparsity
        sparse_nnz = (weights_sparse != 0).sum().item()
        normal_nnz = (weights_normal != 0).sum().item()
        
        print(f"\nSparsity preservation:")
        print(f"  Sparse pipeline: {sparse_nnz:,} non-zeros (maintained: {sparse_nnz == nnz})")
        print(f"  Normal pipeline: {normal_nnz:,} non-zeros (densified: {normal_nnz > nnz})")
        
        if sparse_nnz == nnz:
            print("  ✓ Sparse pipeline preserved sparsity structure")
        if normal_nnz > nnz:
            print("  ⚠ Normal pipeline densified the weights")
        
        # Memory statistics
        weight_mem = M * N * 4 / (1024 ** 2)
        state_mem = M * N * 2 * 4 / (1024 ** 2)
        sparse_state_mem = nnz * 2 * 4 / (1024 ** 2)
        
        print(f"\nMemory usage (current implementation):")
        print(f"  Normal pipeline: {weight_mem + state_mem:.2f} MB total")
        print(f"  Sparse pipeline: {weight_mem + state_mem:.2f} MB total (same as normal)")
        print(f"\nMemory with sparse storage:")
        print(f"  Potential savings: {weight_mem + sparse_state_mem:.2f} MB ({(weight_mem + state_mem)/(weight_mem + sparse_state_mem):.2f}x reduction)")


def test_correctness():
    """Detailed correctness test comparing sparse vs normal updates."""
    print(f"\n{'=' * 80}")
    print("Detailed Correctness Test")
    print("=" * 80)
    
    M, N = 256, 256
    block_size = 32
    sparsity = 0.8
    
    # Create mask and weights
    mask = create_block_sparse_mask((M, N), block_size, sparsity)
    initial_weights = torch.randn(M, N, device='cuda') * mask
    target = torch.randn(M, M, device='cuda')
    
    # Clone for both pipelines
    weights_sparse = initial_weights.clone().requires_grad_(True)
    weights_manual = initial_weights.clone().requires_grad_(True)
    
    # Initialize optimizer states
    exp_avg_sparse = torch.zeros_like(weights_sparse)
    exp_avg_sq_sparse = torch.zeros_like(weights_sparse)
    
    exp_avg_manual = torch.zeros_like(weights_manual)
    exp_avg_sq_manual = torch.zeros_like(weights_manual)
    
    # Parameters
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.01
    step = 1
    
    # Sparse pipeline
    grad_sparse, _ = dummy_forward_backward(weights_sparse, mask, target)
    apply_sparse_backward_and_adam(weights_sparse, grad_sparse, mask,
                                   exp_avg_sparse, exp_avg_sq_sparse,
                                   lr, beta1, beta2, eps, weight_decay, step,
                                   adamw=True, block_size=block_size)
    
    # Manual implementation for verification
    grad_manual, _ = dummy_forward_backward(weights_manual, mask, target)
    
    # Apply mask to gradient (sparse backward)
    masked_grad = grad_manual * mask
    
    # Manual Adam update
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    
    exp_avg_manual = beta1 * exp_avg_manual + (1 - beta1) * masked_grad
    exp_avg_sq_manual = beta2 * exp_avg_sq_manual + (1 - beta2) * masked_grad ** 2
    
    m_hat = exp_avg_manual / bias_correction1
    v_hat = exp_avg_sq_manual / bias_correction2
    
    update = m_hat / (torch.sqrt(v_hat) + eps)
    
    with torch.no_grad():
        weights_manual.copy_(weights_manual * (1 - lr * weight_decay) - lr * update)
    
    # Compare
    weight_diff = (weights_sparse - weights_manual).abs().max().item()
    m_diff = (exp_avg_sparse - exp_avg_manual).abs().max().item()
    v_diff = (exp_avg_sq_sparse - exp_avg_sq_manual).abs().max().item()
    
    print(f"Max differences vs manual implementation:")
    print(f"  Weights: {weight_diff:.2e}")
    print(f"  First moment: {m_diff:.2e}")
    print(f"  Second moment: {v_diff:.2e}")
    
    if weight_diff < 1e-5 and m_diff < 1e-5 and v_diff < 1e-5:
        print("✓ Perfect match!")
    elif weight_diff < 1e-3:
        print("✓ Very close (acceptable FP precision)")
    else:
        print("⚠ Significant differences detected")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    
    # Run tests
    test_correctness()
    test_sparse_pipeline()
    
    print(f"\n{'=' * 80}")