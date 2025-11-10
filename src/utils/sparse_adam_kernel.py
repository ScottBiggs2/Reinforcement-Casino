import torch
import triton
import triton.language as tl


@triton.jit
def block_sparse_adam_2d_kernel(
    # Pointers to tensors
    weights_ptr,
    grads_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    mask_ptr,  # Binary mask (1 for non-zero blocks, 0 for zero blocks)
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
    use_sparse_storage: tl.constexpr,
    # Block info
    BLOCK_SIZE: tl.constexpr,
):
    """
    2D Block-sparse Adam kernel compatible with sparse backward pass.
    
    Each program processes a BLOCK_SIZE × BLOCK_SIZE block.
    Only processes blocks where mask indicates non-zero elements.
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
    
    # Check if this block should be processed (check center element of block)
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


@triton.jit
def block_sparse_adam_kernel(
    # Pointers to tensors
    weights_ptr,
    grads_ptr,
    exp_avg_ptr,      # Can be sparse or dense
    exp_avg_sq_ptr,   # Can be sparse or dense
    block_indices_ptr,  # Indices of non-zero blocks
    # Optimization parameters
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bias_correction1,
    bias_correction2,
    use_adamw: tl.constexpr,
    use_sparse_storage: tl.constexpr,  # Whether optimizer states are sparse
    # Dimensions
    num_blocks,
    block_size: tl.constexpr,
):
    """
    Block-sparse Adam kernel with support for both dense and sparse storage.
    
    Key insight: With sparse storage, optimizer states are stored contiguously
    for all active blocks, not scattered across the full tensor.
    
    Args:
        use_sparse_storage: If True, exp_avg/exp_avg_sq are size (num_blocks * block_size)
                           If False, they're full size (indexed by block_indices)
    """
    # Each program processes one block
    block_id = tl.program_id(0)
    
    if block_id >= num_blocks:
        return
    
    # Get the starting index of this block in the full weight tensor
    weight_block_start = tl.load(block_indices_ptr + block_id)
    
    # Compute offsets for this block
    offsets = tl.arange(0, block_size)
    weight_offsets = weight_block_start + offsets
    
    # Load weights and gradients from full tensors (scattered access)
    w = tl.load(weights_ptr + weight_offsets)
    g = tl.load(grads_ptr + weight_offsets)
    
    # Load optimizer states - different indexing based on storage mode
    if use_sparse_storage:
        # Sparse storage: states are stored contiguously by block_id
        state_block_start = block_id * block_size
        state_offsets = state_block_start + offsets
        m = tl.load(exp_avg_ptr + state_offsets)
        v = tl.load(exp_avg_sq_ptr + state_offsets)
    else:
        # Dense storage: states indexed same as weights
        m = tl.load(exp_avg_ptr + weight_offsets)
        v = tl.load(exp_avg_sq_ptr + weight_offsets)
    
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
    tl.store(weights_ptr + weight_offsets, w_new)
    
    if use_sparse_storage:
        # Sparse storage: write to contiguous location
        tl.store(exp_avg_ptr + state_offsets, m_new)
        tl.store(exp_avg_sq_ptr + state_offsets, v_new)
    else:
        # Dense storage: write back to same location as weights
        tl.store(exp_avg_ptr + weight_offsets, m_new)
        tl.store(exp_avg_sq_ptr + weight_offsets, v_new)


class BlockSparseAdam(torch.optim.Optimizer):
    """
    Block-sparse Adam/AdamW optimizer.
    
    Uses block-structured sparsity: divides tensor into fixed-size blocks
    and only processes blocks that contain at least one non-zero element.
    
    Key advantages over unstructured sparsity:
    - Better memory coalescing (contiguous access patterns)
    - Fewer kernel launches (one per block, not one per element)
    - Achieves speedups at 70-90% sparsity (not just 95%+)
    
    Block sizes:
    - Small blocks (64-128): Better for high sparsity, more granular
    - Large blocks (256-512): Better for moderate sparsity, fewer launches
    
    Note: Uses dense storage (stores optimizer states for all elements) but
    only PROCESSES non-zero blocks, giving compute speedup.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, adamw=False, block_size=128, storage='sparse'):
        """
        Args:
            block_size: Size of each block (64, 128, 256, or 512 recommended)
            storage: 'sparse' (memory efficient) or 'dense' (same memory as PyTorch)
        """
        if block_size not in [32, 64, 128, 256, 512]:
            print(f"Warning: block_size={block_size} may not be optimal. Recommended: 64, 128, 256, or 512")
        
        if storage not in ['sparse', 'dense']:
            raise ValueError(f"storage must be 'sparse' or 'dense', got {storage}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       adamw=adamw, block_size=block_size, storage=storage)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            block_size = group['block_size']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                    # Flatten and identify non-zero blocks
                    p_flat = p.flatten()
                    total_elements = p_flat.numel()
                    num_total_blocks = (total_elements + block_size - 1) // block_size
                    
                    # Find which blocks contain non-zero elements
                    nonzero_blocks = []
                    for i in range(num_total_blocks):
                        block_start = i * block_size
                        block_end = min(block_start + block_size, total_elements)
                        block_data = p_flat[block_start:block_end]
                        
                        # Block is "non-zero" if it contains any non-zero element
                        if (block_data != 0).any():
                            nonzero_blocks.append(block_start)
                    
                    # Convert to tensor
                    if len(nonzero_blocks) == 0:
                        nonzero_blocks = [0]  # At least one block
                    
                    state['block_indices'] = torch.tensor(nonzero_blocks, dtype=torch.int64, device=p.device)
                    state['num_blocks'] = len(nonzero_blocks)
                    state['block_size'] = block_size
                    
                    # Calculate sparsity
                    active_elements = state['num_blocks'] * block_size
                    state['block_sparsity'] = 1.0 - (active_elements / total_elements)
                    
                    # Storage mode
                    storage_mode = group['storage']
                    state['storage_mode'] = storage_mode
                    
                    if storage_mode == 'sparse':
                        # Sparse: only store optimizer states for active blocks (contiguous!)
                        sparse_size = state['num_blocks'] * block_size
                        state['exp_avg'] = torch.zeros(sparse_size, dtype=p.dtype, device=p.device)
                        state['exp_avg_sq'] = torch.zeros(sparse_size, dtype=p.dtype, device=p.device)
                    else:
                        # Dense: store optimizer states for all elements
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                
                state['step'] += 1
                step = state['step']
                
                # Precompute bias corrections
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                
                # Flatten tensors
                p_flat = p.flatten()
                grad_flat = grad.flatten()
                
                # Get optimizer states (already flattened for sparse, need flattening for dense)
                storage_mode = state['storage_mode']
                if storage_mode == 'sparse':
                    exp_avg = state['exp_avg']  # Already 1D, size = num_blocks * block_size
                    exp_avg_sq = state['exp_avg_sq']
                    use_sparse_storage = True
                else:
                    exp_avg = state['exp_avg'].flatten()  # Flatten to match weights
                    exp_avg_sq = state['exp_avg_sq'].flatten()
                    use_sparse_storage = False
                
                block_indices = state['block_indices']
                num_blocks = state['num_blocks']
                
                # Launch one kernel per block (only process non-zero blocks!)
                grid = (num_blocks,)
                
                block_sparse_adam_kernel[grid](
                    p_flat, grad_flat, exp_avg, exp_avg_sq,
                    block_indices,
                    group['lr'], beta1, beta2, group['eps'],
                    group['weight_decay'],
                    bias_correction1, bias_correction2,
                    group['adamw'],
                    use_sparse_storage,
                    num_blocks, block_size,
                )
        
        return loss
    
    def get_memory_stats(self):
        """Return memory statistics."""
        stats = {
            'total_params': 0,
            'total_active_elements': 0,
            'total_blocks': 0,
            'total_state_memory_saved_mb': 0,
            'per_param': []
        }
        
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state:
                    state = self.state[p]
                    param_size = p.numel()
                    num_blocks = state['num_blocks']
                    block_size = state['block_size']
                    active_elements = num_blocks * block_size
                    block_sparsity = state['block_sparsity']
                    storage_mode = state['storage_mode']
                    
                    # Memory calculations (4 bytes per float32)
                    standard_memory = param_size * 2 * 4  # 2 state tensors
                    
                    if storage_mode == 'sparse':
                        actual_memory = active_elements * 2 * 4
                    else:
                        actual_memory = param_size * 2 * 4
                    
                    saved_memory = (standard_memory - actual_memory) / (1024 ** 2)  # MB
                    
                    stats['total_params'] += param_size
                    stats['total_active_elements'] += active_elements
                    stats['total_blocks'] += num_blocks
                    stats['total_state_memory_saved_mb'] += saved_memory
                    
                    stats['per_param'].append({
                        'shape': tuple(p.shape),
                        'params': param_size,
                        'num_blocks': num_blocks,
                        'block_size': block_size,
                        'active_elements': active_elements,
                        'block_sparsity': block_sparsity,
                        'storage_mode': storage_mode,
                        'memory_saved_mb': saved_memory
                    })
        
        stats['avg_block_sparsity'] = 1.0 - (stats['total_active_elements'] / stats['total_params']) if stats['total_params'] > 0 else 0
        
        return stats


def create_block_sparse_tensor(shape, block_size, block_sparsity, device='cuda'):
    """
    Create a block-sparse tensor.
    
    Args:
        shape: Tensor shape
        block_size: Size of each block
        block_sparsity: Fraction of blocks to zero out (0.7 = 70% of blocks are zero)
        device: Device to create tensor on
    
    Returns:
        Block-sparse tensor
    """
    tensor = torch.randn(shape, device=device)
    tensor_flat = tensor.flatten()
    total_elements = tensor_flat.numel()
    num_blocks = (total_elements + block_size - 1) // block_size
    
    # Randomly select blocks to zero out
    num_zero_blocks = int(num_blocks * block_sparsity)
    zero_block_indices = torch.randperm(num_blocks)[:num_zero_blocks]
    
    # Zero out selected blocks
    for block_idx in zero_block_indices:
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, total_elements)
        tensor_flat[block_start:block_end] = 0
    
    return tensor.reshape(shape)


def triton_sparse_adam_update(weights, gradient, mask, exp_avg, exp_avg_sq, 
                               lr, beta1, beta2, eps, weight_decay, step, 
                               adamw=True, block_size=32):
    """
    Apply sparse Adam/AdamW update using Triton kernel.
    
    Compatible with the sparse backward pass implementation.
    
    Args:
        weights: Parameter tensor (M x N)
        gradient: Gradient tensor (M x N) 
        mask: Binary mask tensor (M x N) indicating non-zero blocks
        exp_avg: First moment estimate (M x N)
        exp_avg_sq: Second moment estimate (M x N)
        lr: Learning rate
        beta1, beta2: Adam betas
        eps: Epsilon for numerical stability
        weight_decay: Weight decay coefficient
        step: Current optimization step
        adamw: Whether to use AdamW or Adam
        block_size: Block size for Triton kernel
        
    Returns:
        None (updates weights, exp_avg, exp_avg_sq in-place)
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
        False,  # use_sparse_storage (not used in this version)
        BLOCK_SIZE=block_size,
    )


# Utility function to create block-sparse tensors
def create_block_sparse_tensor(shape, block_size, block_sparsity, device='cuda'):
    """
    Create a block-sparse tensor.
    
    Args:
        shape: Tensor shape
        block_size: Size of each block
        block_sparsity: Fraction of blocks to zero out (0.7 = 70% of blocks are zero)
        device: Device to create tensor on
    
    Returns:
        Block-sparse tensor
    """
    tensor = torch.randn(shape, device=device)
    tensor_flat = tensor.flatten()
    total_elements = tensor_flat.numel()
    num_blocks = (total_elements + block_size - 1) // block_size
    
    # Randomly select blocks to zero out
    num_zero_blocks = int(num_blocks * block_sparsity)
    zero_block_indices = torch.randperm(num_blocks)[:num_zero_blocks]
    
    # Zero out selected blocks
    for block_idx in zero_block_indices:
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, total_elements)
        tensor_flat[block_start:block_end] = 0
    
    return tensor.reshape(shape)


# Benchmark
if __name__ == "__main__":
    print("=" * 80)
    print("Block-Sparse Adam/AdamW - Achieving Speedups at 70-90% Sparsity")
    print("=" * 80)
    
    # Test configurations
    configs = [
        {'size': (1000, 1000), 'sparsity': 0.7, 'block_size': 128, 'name': '1M params, 70% sparse, BS=128'},
        {'size': (1000, 1000), 'sparsity': 0.8, 'block_size': 128, 'name': '1M params, 80% sparse, BS=128'},
        {'size': (1000, 1000), 'sparsity': 0.9, 'block_size': 128, 'name': '1M params, 90% sparse, BS=128'},
        {'size': (2000, 2000), 'sparsity': 0.7, 'block_size': 256, 'name': '4M params, 70% sparse, BS=256'},
        {'size': (2000, 2000), 'sparsity': 0.9, 'block_size': 256, 'name': '4M params, 90% sparse, BS=256'},
    ]
    
    for config in configs:
        print(f"\n{'=' * 80}")
        print(f"Testing: {config['name']}")
        print(f"{'=' * 80}")
        
        torch.manual_seed(42)
        size = config['size']
        sparsity = config['sparsity']
        block_size = config['block_size']
        
        # Create block-sparse weight matrix
        weights = create_block_sparse_tensor(size, block_size, sparsity, device='cuda')
        
        total = weights.numel()
        nnz = (weights != 0).sum().item()
        actual_sparsity = 1 - (nnz / total)
        
        print(f"Matrix size: {size[0]} x {size[1]} = {total:,} parameters")
        print(f"Block size: {block_size}")
        print(f"Non-zeros: {nnz:,} ({100 * (1-actual_sparsity):.2f}%)")
        print(f"Actual sparsity: {100 * actual_sparsity:.2f}%")
        
        # Clone for comparison
        weights_sparse = weights.clone().requires_grad_(True)
        weights_dense = weights.clone().requires_grad_(True)
        weights_torch = weights.clone().requires_grad_(True)
        
        # Create optimizers
        opt_sparse = BlockSparseAdam([weights_sparse], lr=0.001, adamw=True, 
                                     block_size=block_size, storage='sparse')
        opt_dense = BlockSparseAdam([weights_dense], lr=0.001, adamw=True,
                                    block_size=block_size, storage='dense')
        opt_torch = torch.optim.AdamW([weights_torch], lr=0.001)
        
        # Initialize
        for opt, w in [(opt_sparse, weights_sparse),
                       (opt_dense, weights_dense),
                       (opt_torch, weights_torch)]:
            loss = (w ** 2).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        # Memory statistics
        stats_sparse = opt_sparse.get_memory_stats()
        stats_dense = opt_dense.get_memory_stats()
        
        num_blocks = stats_sparse['per_param'][0]['num_blocks']
        active_elements = stats_sparse['per_param'][0]['active_elements']
        block_sparsity = stats_sparse['per_param'][0]['block_sparsity']
        total_blocks = (total + block_size - 1) // block_size
        
        print(f"\nBlock structure:")
        print(f"  Total blocks: {total_blocks:,}")
        print(f"  Active blocks: {num_blocks:,} (processing {100 * (num_blocks/total_blocks):.1f}%)")
        print(f"  Block sparsity: {100 * block_sparsity:.1f}%")
        print(f"  Elements processed: {active_elements:,} / {total:,} ({100 * (active_elements/total):.1f}%)")
        
        # Memory comparison
        weight_mem = total * 4 / (1024 ** 2)
        torch_state_mem = total * 2 * 4 / (1024 ** 2)
        sparse_state_mem = active_elements * 2 * 4 / (1024 ** 2)
        
        print(f"\n{'Memory Usage':^80}")
        print("-" * 80)
        print(f"{'Optimizer':<30} {'Weights':<15} {'States':<15} {'Total':<15}")
        print("-" * 80)
        print(f"{'PyTorch AdamW':<30} {weight_mem:>10.2f} MB   {torch_state_mem:>10.2f} MB   {weight_mem + torch_state_mem:>10.2f} MB")
        print(f"{'Block-Sparse (sparse)':<30} {weight_mem:>10.2f} MB   {sparse_state_mem:>10.2f} MB   {weight_mem + sparse_state_mem:>10.2f} MB")
        print(f"{'Block-Sparse (dense)':<30} {weight_mem:>10.2f} MB   {torch_state_mem:>10.2f} MB   {weight_mem + torch_state_mem:>10.2f} MB")
        print("-" * 80)
        
        mem_savings = torch_state_mem - sparse_state_mem
        mem_ratio = torch_state_mem / sparse_state_mem if sparse_state_mem > 0 else 1
        print(f"Sparse mode saves: {mem_savings:.2f} MB ({mem_ratio:.2f}x reduction in state memory)")
        
        # Warmup
        print(f"\n{'Performance Benchmark':^80}")
        print("Warming up...")
        for _ in range(10):
            for opt, w in [(opt_sparse, weights_sparse),
                           (opt_dense, weights_dense),
                           (opt_torch, weights_torch)]:
                loss = (w ** 2).sum()
                loss.backward()
                opt.step()
                opt.zero_grad()
        
        torch.cuda.synchronize()
        
        # Benchmark
        def benchmark(opt, w, n_iters=100):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(n_iters):
                loss = (w ** 2).sum()
                loss.backward()
                opt.step()
                opt.zero_grad()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / n_iters
        
        print("Benchmarking...")
        time_sparse = benchmark(opt_sparse, weights_sparse)
        time_dense = benchmark(opt_dense, weights_dense)
        time_torch = benchmark(opt_torch, weights_torch)
        
        speedup_sparse = time_torch / time_sparse
        speedup_dense = time_torch / time_dense
        
        print("-" * 80)
        print(f"{'Optimizer':<30} {'Time (ms)':<20} {'Speedup':<20}")
        print("-" * 80)
        print(f"{'PyTorch AdamW':<30} {time_torch:>10.4f} ms       {'1.00x':<20}")
        print(f"{'Block-Sparse (sparse)':<30} {time_sparse:>10.4f} ms       {speedup_sparse:.2f}x")
        print(f"{'Block-Sparse (dense)':<30} {time_dense:>10.4f} ms       {speedup_dense:.2f}x")
        print("-" * 80)
        
        # Analysis
        theoretical_reduction = total_blocks / num_blocks
        
        print(f"\nAnalysis:")
        print(f"  • Block sparsity: {100 * block_sparsity:.1f}%")
        print(f"  • Processing {num_blocks:,} blocks instead of {total_blocks:,}")
        print(f"  • Theoretical: {theoretical_reduction:.2f}x fewer blocks to process")
        print(f"  • Actual speedup:")
        print(f"    - Sparse storage: {speedup_sparse:.2f}x ({mem_ratio:.1f}x memory savings)")
        print(f"    - Dense storage: {speedup_dense:.2f}x (no memory savings)")
        
        efficiency_sparse = (speedup_sparse / theoretical_reduction) * 100
        efficiency_dense = (speedup_dense / theoretical_reduction) * 100
        print(f"  • Efficiency:")
        print(f"    - Sparse: {efficiency_sparse:.1f}% (actual vs theoretical)")
        print(f"    - Dense: {efficiency_dense:.1f}% (actual vs theoretical)")
        
        best_mode = "sparse" if speedup_sparse > speedup_dense else "dense"
        best_speedup = max(speedup_sparse, speedup_dense)
        
        if best_speedup > 1.3:
            print(f"  ✓ Significant speedup ({best_speedup:.2f}x with {best_mode} mode)!")
        elif best_speedup > 1.1:
            print(f"  ~ Modest speedup ({best_speedup:.2f}x with {best_mode} mode)")
        else:
            print(f"  ⚠ Limited speedup - overhead dominates at this sparsity/block size")
        
        if speedup_sparse > 1.0 and mem_ratio > 2.0:
            print(f"  🎉 Sparse mode: BOTH faster ({speedup_sparse:.2f}x) AND saves memory ({mem_ratio:.1f}x)!")
        
        # Accuracy
        initial_mask = (weights != 0)
        diff = (weights_sparse[initial_mask] - weights_torch[initial_mask]).abs().max().item()
        rel_err = (diff / weights_torch[initial_mask].abs().max().item()) * 100
        
        print(f"\nAccuracy: {rel_err:.3f}% relative error")
        if rel_err < 1.0:
            print("✓ Excellent agreement")
    
    print(f"\n{'=' * 80}")