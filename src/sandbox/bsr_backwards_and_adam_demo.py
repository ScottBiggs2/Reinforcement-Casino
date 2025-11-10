"""
Block Sparse Row (BSR) Training Pipeline - Toy Example
Combines sparse backward pass with sparse Adam optimizer for end-to-end training.
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np
from typing import Tuple, Optional

# ============================================================================
# BACKWARD PASS KERNEL
# ============================================================================

@triton.jit
def sparse_backward_kernel(
    # Input pointers
    weights_ptr,
    grad_ptr,
    mask_ptr,
    output_ptr,
    # Matrix dimensions
    M, N,
    stride_wm, stride_wn,
    stride_gm, stride_gn,
    stride_mm, stride_mn,
    stride_om, stride_on,
    # Learning rate (for demonstration - not used in backward)
    lr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Sparse backward pass kernel - only computes gradients for masked regions.
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
    
    # Create mask for valid elements
    mask_valid = (row_offsets < M) & (col_offsets < N)
    
    # Calculate memory offsets
    w_offsets = row_offsets * stride_wm + col_offsets * stride_wn
    g_offsets = row_offsets * stride_gm + col_offsets * stride_gn
    m_offsets = row_offsets * stride_mm + col_offsets * stride_mn
    o_offsets = row_offsets * stride_om + col_offsets * stride_on
    
    # Load data
    g_block = tl.load(grad_ptr + g_offsets, mask=mask_valid, other=0.0)
    m_block = tl.load(mask_ptr + m_offsets, mask=mask_valid, other=0.0)
    
    # Apply mask to gradients
    masked_grad = g_block * m_block
    
    # Store masked gradients
    tl.store(output_ptr + o_offsets, masked_grad, mask=mask_valid)


def sparse_backward(gradient, mask, block_size=32):
    """
    Apply sparse gradient masking using Triton kernel.
    
    Args:
        gradient: Gradient tensor (M x N)
        mask: Binary mask tensor (M x N)
        block_size: Block size for Triton kernel
        
    Returns:
        Masked gradient tensor
    """
    M, N = gradient.shape
    output = torch.empty_like(gradient)
    
    # Calculate grid size
    num_blocks_m = triton.cdiv(M, block_size)
    num_blocks_n = triton.cdiv(N, block_size)
    grid = (num_blocks_m * num_blocks_n,)
    
    # Launch kernel
    sparse_backward_kernel[grid](
        gradient, gradient, mask, output,
        M, N,
        gradient.stride(0), gradient.stride(1),
        gradient.stride(0), gradient.stride(1),
        mask.stride(0), mask.stride(1),
        output.stride(0), output.stride(1),
        0.0,  # lr not used
        BLOCK_SIZE=block_size,
    )
    
    return output


# ============================================================================
# ADAM OPTIMIZER KERNEL
# ============================================================================

@triton.jit
def block_sparse_adam_2d_kernel(
    # Pointers to tensors
    weights_ptr,
    grads_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    mask_ptr,
    # Matrix dimensions
    M, N,
    stride_wm, stride_wn,
    stride_gm, stride_gn,
    stride_em, stride_en,
    stride_vm, stride_vn,
    stride_mm, stride_mn,
    # Optimization parameters
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bias_correction1,
    bias_correction2,
    use_adamw: tl.constexpr,
    # Block info
    BLOCK_SIZE: tl.constexpr,
):
    """
    2D Block-sparse Adam kernel - only updates masked regions.
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
    
    # Create mask for valid elements
    mask_valid = (row_offsets < M) & (col_offsets < N)
    
    # Check if this block should be processed (check center element)
    center_row = row_start + BLOCK_SIZE // 2
    center_col = col_start + BLOCK_SIZE // 2
    if center_row < M and center_col < N:
        block_mask = tl.load(mask_ptr + center_row * stride_mm + center_col * stride_mn)
        if block_mask == 0.0:
            return  # Skip this block entirely
    
    # Calculate memory offsets
    w_offsets = row_offsets * stride_wm + col_offsets * stride_wn
    g_offsets = row_offsets * stride_gm + col_offsets * stride_gn
    e_offsets = row_offsets * stride_em + col_offsets * stride_en
    v_offsets = row_offsets * stride_vm + col_offsets * stride_vn
    
    # Load data
    w = tl.load(weights_ptr + w_offsets, mask=mask_valid, other=0.0)
    g = tl.load(grads_ptr + g_offsets, mask=mask_valid, other=0.0)
    m = tl.load(exp_avg_ptr + e_offsets, mask=mask_valid, other=0.0)
    v = tl.load(exp_avg_sq_ptr + v_offsets, mask=mask_valid, other=0.0)
    
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
    """
    Apply sparse Adam/AdamW update using Triton kernel.
    
    Args:
        weights: Parameter tensor (M x N)
        gradient: Gradient tensor (M x N)
        mask: Binary mask tensor (M x N)
        exp_avg: First moment estimate (M x N)
        exp_avg_sq: Second moment estimate (M x N)
        lr: Learning rate
        beta1, beta2: Adam betas
        eps: Epsilon
        weight_decay: Weight decay coefficient
        step: Current optimization step
        adamw: Whether to use AdamW
        block_size: Block size
    """
    M, N = weights.shape
    
    # Precompute bias corrections
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    
    # Calculate grid size
    num_blocks_m = triton.cdiv(M, block_size)
    num_blocks_n = triton.cdiv(N, block_size)
    grid = (num_blocks_m * num_blocks_n,)
    
    # Launch kernel
    block_sparse_adam_2d_kernel[grid](
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
# TOY PROBLEM: SPARSE MATRIX REGRESSION
# ============================================================================

class SparseMatrixRegression:
    """
    Toy problem: Learn a sparse weight matrix to minimize ||Y - W @ X||^2
    """
    
    def __init__(self, input_dim=1024, output_dim=1024, sparsity=0.9, device='cuda'):
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        
        # Create true sparse weight matrix (target)
        self.W_true = self._create_sparse_matrix(
            (output_dim, input_dim), sparsity
        ).to(device)
        
        # Create mask from true weights
        self.mask = (self.W_true != 0).float()
        
        # Create training data
        self.X_train = torch.randn(input_dim, 100, device=device)
        self.Y_train = self.W_true @ self.X_train
        
        # Create test data
        self.X_test = torch.randn(input_dim, 20, device=device)
        self.Y_test = self.W_true @ self.X_test
        
        print(f"Created toy problem:")
        print(f"  Weight matrix: {output_dim} x {input_dim}")
        print(f"  Sparsity: {100*sparsity:.1f}%")
        print(f"  Non-zero elements: {self.mask.sum().item():,.0f} / {self.mask.numel():,.0f}")
        print(f"  Training samples: {self.X_train.shape[1]}")
        print(f"  Test samples: {self.X_test.shape[1]}")
    
    def _create_sparse_matrix(self, shape, sparsity):
        """Create a block-sparse matrix."""
        matrix = torch.randn(shape)
        flat = matrix.flatten()
        num_zeros = int(flat.numel() * sparsity)
        zero_indices = torch.randperm(flat.numel())[:num_zeros]
        flat[zero_indices] = 0
        return matrix.reshape(shape)
    
    def compute_loss(self, W, X, Y):
        """Compute MSE loss."""
        pred = W @ X
        return ((pred - Y) ** 2).mean()
    
    def compute_gradient(self, W, X, Y):
        """Compute gradient of MSE loss."""
        pred = W @ X
        error = pred - Y
        grad = 2 * error @ X.T / X.shape[1]
        return grad


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_dense_baseline(problem, n_steps=100, lr=0.01):
    """Train with standard PyTorch (dense updates)."""
    W = torch.randn_like(problem.W_true, requires_grad=True)
    optimizer = torch.optim.AdamW([W], lr=lr)
    
    train_losses = []
    test_losses = []
    times = []
    
    print("\nTraining Dense Baseline...")
    start_total = time.time()
    
    for step in range(n_steps):
        start = time.time()
        
        optimizer.zero_grad()
        loss = problem.compute_loss(W, problem.X_train, problem.Y_train)
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        times.append(time.time() - start)
        
        with torch.no_grad():
            train_losses.append(loss.item())
            test_loss = problem.compute_loss(W, problem.X_test, problem.Y_test)
            test_losses.append(test_loss.item())
        
        if step % 20 == 0:
            print(f"  Step {step:3d}: Train Loss = {loss.item():.6f}, "
                  f"Test Loss = {test_loss.item():.6f}")
    
    total_time = time.time() - start_total
    avg_time = np.mean(times[10:])  # Skip warmup
    
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Avg step time: {avg_time*1000:.3f}ms")
    
    return {
        'weights': W.detach(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'avg_step_time': avg_time,
        'total_time': total_time
    }


def train_sparse_triton(problem, n_steps=100, lr=0.01, block_size=32):
    """Train with sparse Triton kernels (backward + optimizer)."""
    W = torch.randn_like(problem.W_true)
    
    # Initialize optimizer states
    exp_avg = torch.zeros_like(W)
    exp_avg_sq = torch.zeros_like(W)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    weight_decay = 0.01
    
    train_losses = []
    test_losses = []
    times = []
    
    print(f"\nTraining Sparse Triton (block_size={block_size})...")
    start_total = time.time()
    
    for step in range(1, n_steps + 1):
        start = time.time()
        
        # Forward pass
        loss = problem.compute_loss(W, problem.X_train, problem.Y_train)
        
        # Compute gradient
        grad = problem.compute_gradient(W, problem.X_train, problem.Y_train)
        
        # Sparse backward (apply mask to gradients)
        grad_masked = sparse_backward(grad, problem.mask, block_size=block_size)
        
        # Sparse Adam update
        sparse_adam_update(
            W, grad_masked, problem.mask, exp_avg, exp_avg_sq,
            lr, beta1, beta2, eps, weight_decay, step,
            adamw=True, block_size=block_size
        )
        
        torch.cuda.synchronize()
        times.append(time.time() - start)
        
        train_losses.append(loss.item())
        test_loss = problem.compute_loss(W, problem.X_test, problem.Y_test)
        test_losses.append(test_loss.item())
        
        if (step - 1) % 20 == 0:
            print(f"  Step {step-1:3d}: Train Loss = {loss.item():.6f}, "
                  f"Test Loss = {test_loss.item():.6f}")
    
    total_time = time.time() - start_total
    avg_time = np.mean(times[10:])  # Skip warmup
    
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Avg step time: {avg_time*1000:.3f}ms")
    
    return {
        'weights': W,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'avg_step_time': avg_time,
        'total_time': total_time
    }


def train_sparse_pytorch(problem, n_steps=100, lr=0.01):
    """Train with PyTorch but manually masking gradients."""
    W = torch.randn_like(problem.W_true, requires_grad=True)
    optimizer = torch.optim.AdamW([W], lr=lr)
    
    train_losses = []
    test_losses = []
    times = []
    
    print("\nTraining Sparse PyTorch (masked gradients)...")
    start_total = time.time()
    
    for step in range(n_steps):
        start = time.time()
        
        optimizer.zero_grad()
        loss = problem.compute_loss(W, problem.X_train, problem.Y_train)
        loss.backward()
        
        # Manually mask gradients
        with torch.no_grad():
            W.grad *= problem.mask
        
        optimizer.step()
        
        torch.cuda.synchronize()
        times.append(time.time() - start)
        
        with torch.no_grad():
            train_losses.append(loss.item())
            test_loss = problem.compute_loss(W, problem.X_test, problem.Y_test)
            test_losses.append(test_loss.item())
        
        if step % 20 == 0:
            print(f"  Step {step:3d}: Train Loss = {loss.item():.6f}, "
                  f"Test Loss = {test_loss.item():.6f}")
    
    total_time = time.time() - start_total
    avg_time = np.mean(times[10:])  # Skip warmup
    
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Avg step time: {avg_time*1000:.3f}ms")
    
    return {
        'weights': W.detach(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'avg_step_time': avg_time,
        'total_time': total_time
    }


# ============================================================================
# BENCHMARKING & COMPARISON
# ============================================================================

def compare_results(problem, results_dense, results_sparse_triton, results_sparse_pytorch):
    """Compare all three training methods."""
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    # Final losses
    print("\nFinal Losses:")
    print(f"  Dense Baseline:")
    print(f"    Train: {results_dense['train_losses'][-1]:.6f}")
    print(f"    Test:  {results_dense['test_losses'][-1]:.6f}")
    print(f"  Sparse Triton:")
    print(f"    Train: {results_sparse_triton['train_losses'][-1]:.6f}")
    print(f"    Test:  {results_sparse_triton['test_losses'][-1]:.6f}")
    print(f"  Sparse PyTorch:")
    print(f"    Train: {results_sparse_pytorch['train_losses'][-1]:.6f}")
    print(f"    Test:  {results_sparse_pytorch['test_losses'][-1]:.6f}")
    
    # Performance comparison
    print("\nPerformance (avg step time after warmup):")
    baseline_time = results_dense['avg_step_time']
    triton_time = results_sparse_triton['avg_step_time']
    pytorch_time = results_sparse_pytorch['avg_step_time']
    
    print(f"  Dense Baseline:    {baseline_time*1000:6.3f}ms  (1.00x)")
    print(f"  Sparse Triton:     {triton_time*1000:6.3f}ms  ({baseline_time/triton_time:.2f}x)")
    print(f"  Sparse PyTorch:    {pytorch_time*1000:6.3f}ms  ({baseline_time/pytorch_time:.2f}x)")
    
    # Speedup analysis
    triton_speedup = baseline_time / triton_time
    pytorch_speedup = baseline_time / pytorch_time
    
    print(f"\nSpeedup Analysis:")
    print(f"  Triton vs Dense:    {triton_speedup:.2f}x faster")
    print(f"  Triton vs PyTorch:  {triton_time/pytorch_time:.2f}x")
    
    if triton_speedup > 1.2:
        print(f"  ✓ Triton achieves {triton_speedup:.2f}x speedup!")
    elif triton_speedup > 1.0:
        print(f"  ~ Triton shows modest {triton_speedup:.2f}x speedup")
    else:
        print(f"  ⚠ Triton overhead dominates at this problem size")
    
    # Weight recovery
    print("\nWeight Recovery (distance to true weights):")
    with torch.no_grad():
        dense_error = ((results_dense['weights'] - problem.W_true) ** 2).mean().sqrt()
        triton_error = ((results_sparse_triton['weights'] - problem.W_true) ** 2).mean().sqrt()
        pytorch_error = ((results_sparse_pytorch['weights'] - problem.W_true) ** 2).mean().sqrt()
    
    print(f"  Dense Baseline:  {dense_error:.6f}")
    print(f"  Sparse Triton:   {triton_error:.6f}")
    print(f"  Sparse PyTorch:  {pytorch_error:.6f}")
    
    # Sparsity preservation
    print("\nSparsity Preservation:")
    mask_support = problem.mask.sum().item()
    with torch.no_grad():
        triton_support = (results_sparse_triton['weights'] != 0).float().sum().item()
        pytorch_support = (results_sparse_pytorch['weights'] != 0).float().sum().item()
    
    print(f"  Mask support:      {mask_support:,.0f}")
    print(f"  Triton non-zeros:  {triton_support:,.0f}")
    print(f"  PyTorch non-zeros: {pytorch_support:,.0f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("BSR SPARSE TRAINING PIPELINE - TOY EXAMPLE")
    print("="*80)
    
    # Check device
    if not torch.cuda.is_available():
        print("\n⚠ CUDA not available! This script requires a GPU.")
        return
    
    device = torch.device('cuda')
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"CUDA Compute Capability: {torch.cuda.get_device_capability()}")
    
    # Problem configurations to test
    configs = [
        {'size': 2560, 'sparsity': 0.9, 'block_size': 32, 'n_steps': 100},
        {'size': 2560, 'sparsity': 0.9, 'block_size': 64, 'n_steps': 100},
        {'size': 2560, 'sparsity': 0.9, 'block_size': 128, 'n_steps': 100},
        {'size': 2560, 'sparsity': 0.9, 'block_size': 256, 'n_steps': 100},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"CONFIGURATION {i+1}/{len(configs)}")
        print(f"{'='*80}")
        print(f"Matrix size: {config['size']} x {config['size']}")
        print(f"Sparsity: {100*config['sparsity']:.1f}%")
        print(f"Block size: {config['block_size']}")
        print(f"Training steps: {config['n_steps']}")
        
        # Create problem
        problem = SparseMatrixRegression(
            input_dim=config['size'],
            output_dim=config['size'],
            sparsity=config['sparsity'],
            device=device
        )
        
        # Train with all methods
        results_dense = train_dense_baseline(problem, n_steps=config['n_steps'])
        results_sparse_triton = train_sparse_triton(
            problem, n_steps=config['n_steps'], block_size=config['block_size']
        )
        results_sparse_pytorch = train_sparse_pytorch(problem, n_steps=config['n_steps'])
        
        # Compare
        compare_results(problem, results_dense, results_sparse_triton, results_sparse_pytorch)
        
        print("\n" + "-"*80)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Sparse Triton should show speedups at high sparsity (90%+)")
    print("2. Both sparse methods should converge to similar losses")
    print("3. Larger matrices benefit more from sparse kernels")
    print("4. Block size affects performance (32-64 often optimal)")


if __name__ == "__main__":
    main()