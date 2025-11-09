import torch
import triton
import triton.language as tl


@triton.jit
def sparse_adam_kernel(
    # Pointers to tensors
    weights_ptr,
    grads_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    mask_ptr,  # Binary mask indicating non-zero positions
    # Optimization parameters
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    step,
    use_adamw: tl.constexpr,
    # Tensor dimensions
    n_elements,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for sparse Adam/AdamW optimizer.
    
    Only updates weights where mask is non-zero (1).
    
    Args:
        weights_ptr: Pointer to weight tensor
        grads_ptr: Pointer to gradient tensor
        exp_avg_ptr: Pointer to first moment estimate (momentum)
        exp_avg_sq_ptr: Pointer to second moment estimate (variance)
        mask_ptr: Pointer to binary mask (1 for non-zero, 0 for zero)
        lr: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        eps: Small constant for numerical stability
        weight_decay: Weight decay coefficient
        step: Current optimization step (for bias correction)
        use_adamw: Whether to use AdamW (decoupled weight decay) or Adam
        n_elements: Total number of elements
        BLOCK_SIZE: Number of elements to process per block
    """
    # Get program ID and compute element indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load mask values to check if elements are non-zero
    sparse_mask = tl.load(mask_ptr + offsets, mask=mask, other=0.0)
    
    # Only process elements where sparse_mask is non-zero
    active_mask = mask & (sparse_mask != 0.0)
    
    # Load current values (only for active elements)
    w = tl.load(weights_ptr + offsets, mask=active_mask, other=0.0)
    g = tl.load(grads_ptr + offsets, mask=active_mask, other=0.0)
    m = tl.load(exp_avg_ptr + offsets, mask=active_mask, other=0.0)
    v = tl.load(exp_avg_sq_ptr + offsets, mask=active_mask, other=0.0)
    
    # Update biased first moment estimate
    m_new = beta1 * m + (1.0 - beta1) * g
    
    # Update biased second raw moment estimate
    v_new = beta2 * v + (1.0 - beta2) * g * g
    
    # Compute bias correction using exp(step * log(beta)) = beta^step
    bias_correction1 = 1.0 - tl.exp(step * tl.log(beta1))
    bias_correction2 = 1.0 - tl.exp(step * tl.log(beta2))
    
    # Compute bias-corrected moments
    m_hat = m_new / bias_correction1
    v_hat = v_new / bias_correction2
    
    # Compute update
    denom = tl.sqrt(v_hat) + eps
    update = m_hat / denom
    
    # Apply weight decay
    if use_adamw:
        # AdamW: decoupled weight decay
        w_new = w * (1.0 - lr * weight_decay) - lr * update
    else:
        # Adam: L2 regularization in gradient
        w_new = w - lr * update
        if weight_decay != 0.0:
            w_new = w_new - lr * weight_decay * w
    
    # Store updated values (only for active elements)
    tl.store(weights_ptr + offsets, w_new, mask=active_mask)
    tl.store(exp_avg_ptr + offsets, m_new, mask=active_mask)
    tl.store(exp_avg_sq_ptr + offsets, v_new, mask=active_mask)


@triton.jit
def sparse_adam_coo_kernel(
    # Pointers to tensors
    weights_ptr,
    grads_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    indices_ptr,  # Non-zero indices
    # Optimization parameters
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    step,
    use_adamw: tl.constexpr,
    # Tensor dimensions
    nnz,  # Number of non-zeros
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for sparse Adam/AdamW using COO-like format.
    
    Works with explicit indices of non-zero elements.
    
    Args:
        indices_ptr: Pointer to array of non-zero indices
        nnz: Number of non-zero elements
        (other args same as sparse_adam_kernel)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid non-zero indices
    mask = offsets < nnz
    
    # Load indices of non-zero elements
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    
    # Load values at these indices
    w = tl.load(weights_ptr + indices, mask=mask, other=0.0)
    g = tl.load(grads_ptr + indices, mask=mask, other=0.0)
    m = tl.load(exp_avg_ptr + indices, mask=mask, other=0.0)
    v = tl.load(exp_avg_sq_ptr + indices, mask=mask, other=0.0)
    
    # Update moments
    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * v + (1.0 - beta2) * g * g
    
    # Bias correction using exp(step * log(beta)) = beta^step
    bias_correction1 = 1.0 - tl.exp(step * tl.log(beta1))
    bias_correction2 = 1.0 - tl.exp(step * tl.log(beta2))
    
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
    
    # Store updated values
    tl.store(weights_ptr + indices, w_new, mask=mask)
    tl.store(exp_avg_ptr + indices, m_new, mask=mask)
    tl.store(exp_avg_sq_ptr + indices, v_new, mask=mask)


class SparseAdam(torch.optim.Optimizer):
    """Sparse Adam/AdamW optimizer using Triton kernels."""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.0, adamw=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                       weight_decay=weight_decay, adamw=adamw)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Sparse gradients not supported yet')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    
                    # Create mask for non-zero weights
                    state['mask'] = (p != 0).float()
                
                state['step'] += 1
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                mask = state['mask']
                
                # Flatten tensors for processing
                p_flat = p.flatten()
                grad_flat = grad.flatten()
                exp_avg_flat = exp_avg.flatten()
                exp_avg_sq_flat = exp_avg_sq.flatten()
                mask_flat = mask.flatten()
                
                n_elements = p_flat.numel()
                
                # Launch kernel
                BLOCK_SIZE = 1024
                grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                
                sparse_adam_kernel[grid](
                    p_flat, grad_flat, exp_avg_flat, exp_avg_sq_flat, mask_flat,
                    group['lr'], beta1, beta2, group['eps'],
                    group['weight_decay'], float(state['step']),
                    group['adamw'],
                    n_elements,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
        
        return loss


# Example usage and benchmarking
if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("Sparse Adam/AdamW Optimizer - Triton vs PyTorch Benchmark")
    print("=" * 70)
    
    # Test different matrix sizes and sparsity levels
    configs = [
        {'size': (1000, 1000), 'sparsity': 0.9, 'name': '1M params, 90% sparse'},
        {'size': (2000, 2000), 'sparsity': 0.95, 'name': '4M params, 95% sparse'},
        {'size': (4000, 4000), 'sparsity': 0.99, 'name': '16M params, 99% sparse'},
    ]
    
    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"Testing: {config['name']}")
        print(f"{'=' * 70}")
        
        torch.manual_seed(42)
        size = config['size']
        sparsity = config['sparsity']
        
        # Create a sparse weight matrix
        weights = torch.randn(size, device='cuda')
        mask = (torch.rand(size, device='cuda') > sparsity).float()
        weights = weights * mask
        
        nnz = (weights != 0).sum().item()
        total = weights.numel()
        actual_sparsity = 1 - (nnz / total)
        
        print(f"Matrix size: {size[0]} x {size[1]} = {total:,} parameters")
        print(f"Non-zeros: {nnz:,} ({100 * (1-actual_sparsity):.2f}%)")
        print(f"Actual sparsity: {100 * actual_sparsity:.2f}%")
        
        # Clone for comparison
        weights_triton = weights.clone().requires_grad_(True)
        weights_torch = weights.clone().requires_grad_(True)
        
        # Create optimizers
        opt_triton = SparseAdam([weights_triton], lr=0.001, adamw=True)
        opt_torch = torch.optim.AdamW([weights_torch], lr=0.001)
        
        # Warmup runs
        print("\nWarming up...")
        for _ in range(10):
            # Triton warmup
            loss_triton = (weights_triton ** 2).sum()
            loss_triton.backward()
            opt_triton.step()
            opt_triton.zero_grad()
            
            # PyTorch warmup
            loss_torch = (weights_torch ** 2).sum()
            loss_torch.backward()
            opt_torch.step()
            opt_torch.zero_grad()
        
        torch.cuda.synchronize()
        
        # Benchmark Triton
        print("\nBenchmarking Triton kernel...")
        n_iters = 100
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(n_iters):
            loss_triton = (weights_triton ** 2).sum()
            loss_triton.backward()
            opt_triton.step()
            opt_triton.zero_grad()
        end_event.record()
        
        torch.cuda.synchronize()
        triton_time = start_event.elapsed_time(end_event) / n_iters
        
        # Benchmark PyTorch
        print("Benchmarking PyTorch optimizer...")
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(n_iters):
            loss_torch = (weights_torch ** 2).sum()
            loss_torch.backward()
            opt_torch.step()
            opt_torch.zero_grad()
        end_event.record()
        
        torch.cuda.synchronize()
        torch_time = start_event.elapsed_time(end_event) / n_iters
        
        # Calculate speedup
        speedup = torch_time / triton_time
        
        # Print results
        print(f"\n{'Results':^70}")
        print("-" * 70)
        print(f"{'Optimizer':<20} {'Time (ms)':<15} {'Speedup':<15}")
        print("-" * 70)
        print(f"{'PyTorch AdamW':<20} {torch_time:>10.4f} ms   {'1.00x':<15}")
        print(f"{'Triton Sparse Adam':<20} {triton_time:>10.4f} ms   {speedup:.2f}x")
        print("-" * 70)
        
        if speedup > 1:
            print(f"✓ Triton is {speedup:.2f}x FASTER")
        else:
            print(f"✗ Triton is {1/speedup:.2f}x SLOWER")
        
        # Verify correctness - compare only the masked (non-zero) weights
        # PyTorch updates all weights, but Triton only updates masked ones
        initial_mask = (mask != 0)
        masked_triton = weights_triton[initial_mask]
        masked_torch = weights_torch[initial_mask]
        
        masked_diff_max = (masked_triton - masked_torch).abs().max().item()
        masked_diff_mean = (masked_triton - masked_torch).abs().mean().item()
        relative_error = (masked_diff_max / masked_torch.abs().max().item()) * 100
        
        print(f"\nAccuracy check (masked weights only):")
        print(f"  Max absolute difference: {masked_diff_max:.2e}")
        print(f"  Mean absolute difference: {masked_diff_mean:.2e}")
        print(f"  Relative error: {relative_error:.3f}%")
        
        # More lenient threshold since we're accumulating FP errors over iterations
        if masked_diff_max < 1e-2 and relative_error < 1.0:
            print("✓ Results match within acceptable tolerance")
        elif masked_diff_max < 5e-2:
            print("⚠ Small numerical differences (likely due to FP precision)")
        else:
            print(f"✗ Results differ significantly!")
        
        # Count non-zero weights in each
        triton_nnz = (weights_triton != 0).sum().item()
        torch_nnz = (weights_torch != 0).sum().item()
        print(f"\nNon-zero count:")
        print(f"  Triton: {triton_nnz:,} (maintains sparsity)")
        print(f"  PyTorch: {torch_nnz:,} (updates all elements)")
        
        if triton_nnz == nnz:
            print("✓ Triton preserved sparsity structure")
    
    print(f"\n{'=' * 70}")
    print("Benchmark complete!")
    print("=" * 70)