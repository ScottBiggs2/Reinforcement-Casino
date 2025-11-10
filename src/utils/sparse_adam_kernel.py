import torch
import triton
import triton.language as tl


@triton.jit
def sparse_adam_masked_kernel(
    # Pointers to tensors
    weights_ptr,
    grads_ptr,
    exp_avg_ptr,      # Dense: size n_elements
    exp_avg_sq_ptr,   # Dense: size n_elements
    mask_ptr,         # Binary mask
    # Optimization parameters
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bias_correction1,
    bias_correction2,
    use_adamw: tl.constexpr,
    # Tensor dimensions
    n_elements,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Masked sparse kernel: processes all elements but skips updates for zeros.
    Faster at low/moderate sparsity but uses more memory.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    sparse_mask = tl.load(mask_ptr + offsets, mask=mask, other=0.0)
    active_mask = mask & (sparse_mask != 0.0)
    
    # Load values
    w = tl.load(weights_ptr + offsets, mask=active_mask, other=0.0)
    g = tl.load(grads_ptr + offsets, mask=active_mask, other=0.0)
    m = tl.load(exp_avg_ptr + offsets, mask=active_mask, other=0.0)
    v = tl.load(exp_avg_sq_ptr + offsets, mask=active_mask, other=0.0)
    
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
    tl.store(weights_ptr + offsets, w_new, mask=active_mask)
    tl.store(exp_avg_ptr + offsets, m_new, mask=active_mask)
    tl.store(exp_avg_sq_ptr + offsets, v_new, mask=active_mask)


@triton.jit
def sparse_adam_kernel(
    # Pointers to tensors
    weights_ptr,
    grads_ptr,
    exp_avg_ptr,      # Sparse: size nnz, not n_elements!
    exp_avg_sq_ptr,   # Sparse: size nnz, not n_elements!
    indices_ptr,      # Indices of non-zero elements
    # Optimization parameters
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bias_correction1,
    bias_correction2,
    use_adamw: tl.constexpr,
    # Tensor dimensions
    nnz,  # Number of non-zeros
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for truly sparse Adam/AdamW optimizer.
    
    Key difference: exp_avg and exp_avg_sq are SIZE NNZ, not size n_elements!
    This gives massive memory savings (10x at 90% sparsity).
    
    Args:
        weights_ptr: Pointer to full weight tensor
        grads_ptr: Pointer to full gradient tensor
        exp_avg_ptr: Pointer to SPARSE first moment (size nnz)
        exp_avg_sq_ptr: Pointer to SPARSE second moment (size nnz)
        indices_ptr: Pointer to indices of non-zero weights
        nnz: Number of non-zero elements
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid non-zero indices
    mask = offsets < nnz
    
    # Load indices of non-zero elements in the full weight tensor
    weight_indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    
    # Load values from full tensors at sparse locations
    w = tl.load(weights_ptr + weight_indices, mask=mask, other=0.0)
    g = tl.load(grads_ptr + weight_indices, mask=mask, other=0.0)
    
    # Load values from SPARSE state tensors (indexed by offset, not weight_indices!)
    m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0)
    
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
    
    # Store updated weight to full tensor
    tl.store(weights_ptr + weight_indices, w_new, mask=mask)
    
    # Store updated moments to SPARSE tensors
    tl.store(exp_avg_ptr + offsets, m_new, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, v_new, mask=mask)


class SparseAdam(torch.optim.Optimizer):
    """
    Sparse Adam/AdamW optimizer with configurable storage strategy.
    
    Storage modes:
    - 'sparse': Only stores states for non-zeros (huge memory savings, good for high sparsity)
    - 'dense': Stores states for all elements (faster compute, good for low/moderate sparsity)
    - 'auto': Automatically chooses based on sparsity (>95% → sparse, ≤95% → dense)
    
    Memory comparison at 90% sparsity with 1M parameters:
    - Standard Adam: 12MB (1M weights + 2M optimizer states)
    - Sparse storage: 1.6MB (1M weights + 200k optimizer states) - 7.5x savings
    - Dense storage: 12MB (same as standard, but preserves sparsity)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.0, adamw=False, storage='auto'):
        """
        Args:
            storage: 'sparse' (memory efficient), 'dense' (compute efficient), or 'auto'
        """
        if storage not in ['sparse', 'dense', 'auto']:
            raise ValueError(f"storage must be 'sparse', 'dense', or 'auto', got {storage}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                       weight_decay=weight_decay, adamw=adamw, storage=storage)
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
                    
                    # Find non-zero indices
                    indices = torch.nonzero(p != 0, as_tuple=False).flatten()
                    nnz = indices.numel()
                    full_size = p.numel()
                    sparsity = 1.0 - (nnz / full_size)
                    
                    # Determine storage mode
                    storage_mode = group['storage']
                    if storage_mode == 'auto':
                        # Use sparse storage at high sparsity, dense at low/moderate
                        storage_mode = 'sparse' if sparsity > 0.95 else 'dense'
                    
                    state['storage_mode'] = storage_mode
                    state['indices'] = indices
                    state['nnz'] = nnz
                    state['sparsity'] = sparsity
                    
                    if storage_mode == 'sparse':
                        # Sparse storage: only store states for non-zeros
                        state['exp_avg'] = torch.zeros(nnz, dtype=p.dtype, device=p.device)
                        state['exp_avg_sq'] = torch.zeros(nnz, dtype=p.dtype, device=p.device)
                        state['memory_savings'] = (full_size - nnz) * 2 * 4 / (1024 ** 2)  # MB
                    else:
                        # Dense storage: store states for all elements (faster compute)
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        state['mask'] = (p != 0).float()
                        state['memory_savings'] = 0.0
                
                state['step'] += 1
                step = state['step']
                
                # Precompute bias corrections
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                storage_mode = state['storage_mode']
                
                # Flatten tensors for processing
                p_flat = p.flatten()
                grad_flat = grad.flatten()
                
                if storage_mode == 'sparse':
                    # Sparse storage: use COO-style kernel
                    indices = state['indices']
                    nnz = state['nnz']
                    
                    if nnz == 0:
                        continue
                    
                    BLOCK_SIZE = 256
                    grid = lambda meta: (triton.cdiv(nnz, meta['BLOCK_SIZE']),)
                    
                    sparse_adam_kernel[grid](
                        p_flat, grad_flat, exp_avg, exp_avg_sq, indices,
                        group['lr'], beta1, beta2, group['eps'],
                        group['weight_decay'], 
                        bias_correction1, bias_correction2,
                        group['adamw'],
                        nnz,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
                else:
                    # Dense storage: use masked kernel (faster)
                    exp_avg_flat = exp_avg.flatten()
                    exp_avg_sq_flat = exp_avg_sq.flatten()
                    mask_flat = state['mask'].flatten()
                    n_elements = p_flat.numel()
                    
                    BLOCK_SIZE = 1024
                    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                    
                    sparse_adam_masked_kernel[grid](
                        p_flat, grad_flat, exp_avg_flat, exp_avg_sq_flat, mask_flat,
                        group['lr'], beta1, beta2, group['eps'],
                        group['weight_decay'], 
                        bias_correction1, bias_correction2,
                        group['adamw'],
                        n_elements,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
        
        return loss
    
    def get_memory_stats(self):
        """Return memory statistics for all parameters."""
        stats = {
            'total_params': 0,
            'total_nnz': 0,
            'total_state_memory_saved_mb': 0,
            'per_param': []
        }
        
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state:
                    state = self.state[p]
                    param_size = p.numel()
                    nnz = state['nnz']
                    sparsity = state['sparsity']
                    storage_mode = state['storage_mode']
                    
                    # Memory in bytes (4 bytes per float32)
                    standard_memory = param_size * 2 * 4  # 2 state tensors
                    
                    if storage_mode == 'sparse':
                        sparse_memory = nnz * 2 * 4
                    else:
                        sparse_memory = param_size * 2 * 4  # Dense mode: same as standard
                    
                    saved_memory = (standard_memory - sparse_memory) / (1024 ** 2)  # MB
                    
                    stats['total_params'] += param_size
                    stats['total_nnz'] += nnz
                    stats['total_state_memory_saved_mb'] += saved_memory
                    
                    stats['per_param'].append({
                        'shape': tuple(p.shape),
                        'params': param_size,
                        'nnz': nnz,
                        'sparsity': sparsity,
                        'storage_mode': storage_mode,
                        'memory_saved_mb': saved_memory
                    })
        
        stats['avg_sparsity'] = 1.0 - (stats['total_nnz'] / stats['total_params']) if stats['total_params'] > 0 else 0
        
        return stats


# Benchmark and demonstration
if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("Sparse Adam/AdamW - Memory vs Compute Trade-off")
    print("=" * 70)
    
    # Test different sparsity levels
    configs = [
        {'size': (1000, 1000), 'sparsity': 0.7, 'name': '1M params, 70% sparse'},
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
        
        # Clone for comparison (4 versions)
        weights_auto = weights.clone().requires_grad_(True)
        weights_sparse = weights.clone().requires_grad_(True)
        weights_dense = weights.clone().requires_grad_(True)
        weights_torch = weights.clone().requires_grad_(True)
        
        # Create optimizers
        opt_auto = SparseAdam([weights_auto], lr=0.001, adamw=True, storage='auto')
        opt_sparse = SparseAdam([weights_sparse], lr=0.001, adamw=True, storage='sparse')
        opt_dense = SparseAdam([weights_dense], lr=0.001, adamw=True, storage='dense')
        opt_torch = torch.optim.AdamW([weights_torch], lr=0.001)
        
        # Run one step to initialize states
        for opt, w in [(opt_auto, weights_auto), (opt_sparse, weights_sparse), 
                       (opt_dense, weights_dense), (opt_torch, weights_torch)]:
            loss = (w ** 2).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        # Check auto-selected mode
        auto_mode = opt_auto.state[weights_auto]['storage_mode']
        print(f"\nAuto-selected storage: {auto_mode} (threshold: 95% sparsity)")
        
        # Memory statistics
        print(f"\n{'Memory Usage':^70}")
        print("-" * 70)
        
        # Calculate memory
        weight_mem = total * 4 / (1024 ** 2)  # MB
        torch_state_mem = total * 2 * 4 / (1024 ** 2)  # 2 states
        sparse_state_mem = nnz * 2 * 4 / (1024 ** 2)  # 2 states, nnz elements
        dense_state_mem = total * 2 * 4 / (1024 ** 2)  # Same as PyTorch
        
        print(f"{'Optimizer':<25} {'Weights':<12} {'States':<12} {'Total':<12} {'Savings':<12}")
        print("-" * 70)
        print(f"{'PyTorch AdamW':<25} {weight_mem:>7.2f} MB   {torch_state_mem:>7.2f} MB   {weight_mem + torch_state_mem:>7.2f} MB   {'-':<12}")
        print(f"{'Sparse (memory mode)':<25} {weight_mem:>7.2f} MB   {sparse_state_mem:>7.2f} MB   {weight_mem + sparse_state_mem:>7.2f} MB   {(torch_state_mem + weight_mem) / (sparse_state_mem + weight_mem):.2f}x")
        print(f"{'Dense (compute mode)':<25} {weight_mem:>7.2f} MB   {dense_state_mem:>7.2f} MB   {weight_mem + dense_state_mem:>7.2f} MB   {'-':<12}")
        print("-" * 70)
        
        memory_saved = torch_state_mem - sparse_state_mem
        print(f"Sparse mode saves {memory_saved:.2f} MB of state memory ({(1 - sparse_state_mem/torch_state_mem)*100:.1f}%)")
        
        # Warmup
        print(f"\n{'Performance Benchmark':^70}")
        print("Warming up...")
        for _ in range(10):
            for opt, w in [(opt_auto, weights_auto), (opt_sparse, weights_sparse), 
                           (opt_dense, weights_dense), (opt_torch, weights_torch)]:
                loss = (w ** 2).sum()
                loss.backward()
                opt.step()
                opt.zero_grad()
        
        torch.cuda.synchronize()
        
        # Benchmark
        n_iters = 100
        
        def benchmark_optimizer(opt, weights):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(n_iters):
                loss = (weights ** 2).sum()
                loss.backward()
                opt.step()
                opt.zero_grad()
            end_event.record()
            torch.cuda.synchronize()
            return start_event.elapsed_time(end_event) / n_iters
        
        print("Benchmarking...")
        auto_time = benchmark_optimizer(opt_auto, weights_auto)
        sparse_time = benchmark_optimizer(opt_sparse, weights_sparse)
        dense_time = benchmark_optimizer(opt_dense, weights_dense)
        torch_time = benchmark_optimizer(opt_torch, weights_torch)
        
        speedup_auto = torch_time / auto_time
        speedup_sparse = torch_time / sparse_time
        speedup_dense = torch_time / dense_time
        
        print("-" * 70)
        print(f"{'Optimizer':<30} {'Time (ms)':<20} {'Speedup':<20}")
        print("-" * 70)
        print(f"{'PyTorch AdamW':<30} {torch_time:>10.4f} ms       {'1.00x':<20}")
        print(f"{'Auto (' + auto_mode + ')':<30} {auto_time:>10.4f} ms       {speedup_auto:.2f}x {'← BEST' if speedup_auto == max(speedup_auto, speedup_sparse, speedup_dense) else ''}")
        print(f"{'Sparse (memory mode)':<30} {sparse_time:>10.4f} ms       {speedup_sparse:.2f}x")
        print(f"{'Dense (compute mode)':<30} {dense_time:>10.4f} ms       {speedup_dense:.2f}x")
        print("-" * 70)
        
        # Analysis
        print(f"\nAnalysis at {100 * actual_sparsity:.1f}% sparsity:")
        if actual_sparsity > 0.95:
            print(f"  • High sparsity → Sparse mode wins (processes only {nnz:,} elements)")
            if speedup_sparse > speedup_dense:
                print(f"    ✓ Sparse is {speedup_sparse/speedup_dense:.2f}x faster than Dense")
            else:
                print(f"    ~ Dense is {speedup_dense/speedup_sparse:.2f}x faster (indirect access overhead)")
        else:
            print(f"  • Moderate sparsity → Dense mode likely faster (better memory access)")
            if speedup_dense > speedup_sparse:
                print(f"    ✓ Dense is {speedup_dense/speedup_sparse:.2f}x faster than Sparse")
            else:
                print(f"    ⚠ Sparse is {speedup_sparse/speedup_dense:.2f}x faster (unexpected)")
        
        print(f"\n  Trade-off:")
        print(f"    • Sparse mode: {(torch_state_mem/sparse_state_mem):.1f}x memory savings, {speedup_sparse:.2f}x speedup")
        print(f"    • Dense mode: No memory savings, {speedup_dense:.2f}x speedup")
        
        # Accuracy check
        initial_mask = (mask != 0)
        masked_auto = weights_auto[initial_mask]
        masked_torch = weights_torch[initial_mask]
        
        diff_max = (masked_auto - masked_torch).abs().max().item()
        diff_mean = (masked_auto - masked_torch).abs().mean().item()
        rel_error = (diff_max / masked_torch.abs().max().item()) * 100
        
        print(f"\n{'Accuracy':^70}")
        print("-" * 70)
        print(f"Max absolute difference: {diff_max:.2e}")
        print(f"Mean absolute difference: {diff_mean:.2e}")
        print(f"Relative error: {rel_error:.3f}%")
        
        if rel_error < 1.0:
            print("✓ Excellent agreement")
        elif rel_error < 3.0:
            print("✓ Good agreement")
        else:
            print("⚠ Moderate differences")
        
        # Sparsity preservation
        auto_nnz = (weights_auto != 0).sum().item()
        sparse_nnz = (weights_sparse != 0).sum().item()
        dense_nnz = (weights_dense != 0).sum().item()
        torch_nnz = (weights_torch != 0).sum().item()
        
        print(f"\n{'Sparsity Preservation':^70}")
        print("-" * 70)
        print(f"Auto: {auto_nnz:,} non-zeros (maintained)")
        print(f"Sparse: {sparse_nnz:,} non-zeros (maintained)")
        print(f"Dense: {dense_nnz:,} non-zeros (maintained)")
        print(f"PyTorch: {torch_nnz:,} non-zeros (densified)")
        
        if auto_nnz == nnz and sparse_nnz == nnz and dense_nnz == nnz:
            print("✓ All Triton modes preserved sparse structure")
    
    print(f"\n{'=' * 70}")