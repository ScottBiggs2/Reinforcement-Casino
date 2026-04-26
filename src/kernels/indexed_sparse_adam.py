
import triton
import triton.language as tl
import os

def get_env_int(name, default):
    return int(os.environ.get(name, default))

@triton.jit
def indexed_sparse_adamw_kernel(
    # Pointers to 2D tensors
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    # Pointer to 1D non-zero linear indices
    indices_ptr,
    n_indices,
    # Tensor shape and strides
    M, N,
    stride_p_m, stride_p_n,
    stride_g_m, stride_g_n,
    stride_m_m, stride_m_n,
    stride_v_m, stride_v_n,
    # Hyperparameters
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    # Precomputed bias corrections (not constexpr to avoid recompilation)
    bias_correction1_val,
    bias_correction2_val,
    BLOCK_SIZE: tl.constexpr,
    USE_SPARSE_STATES: tl.constexpr,
):
    """
    Indexed sparse AdamW without kernel recompilation.
    """
    pid = tl.program_id(0)
    
    # Calculate which non-zero indices this block handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_valid = offsets < n_indices
    
    # Load the actual flattened index we need to update
    linear_idx = tl.load(indices_ptr + offsets, mask=mask_valid, other=0)
    
    # Convert linear index to 2D coordinates
    row = linear_idx // N
    col = linear_idx % N
    
    # Compute memory offsets
    p_off = row * stride_p_m + col * stride_p_n
    g_off = row * stride_g_m + col * stride_g_n
    
    # Gather operation - only load values at non-zero indices
    param = tl.load(param_ptr + p_off, mask=mask_valid, other=0.0)
    grad = tl.load(grad_ptr + g_off, mask=mask_valid, other=0.0)
    
    if USE_SPARSE_STATES:
        # Sparse storage: exp_avg and exp_avg_sq are 1D tensors of size n_indices
        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask_valid, other=0.0)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask_valid, other=0.0)
    else:
        # Dense storage: exp_avg and exp_avg_sq match parameter shape
        m_off = row * stride_m_m + col * stride_m_n
        v_off = row * stride_v_m + col * stride_v_n
        exp_avg = tl.load(exp_avg_ptr + m_off, mask=mask_valid, other=0.0)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + v_off, mask=mask_valid, other=0.0)
    
    # Standard AdamW update computations
    decay_factor = 1.0 - lr * weight_decay
    param_decayed = param * decay_factor
    
    beta1_complement = 1.0 - beta1
    exp_avg_new = beta1 * exp_avg + beta1_complement * grad
    
    beta2_complement = 1.0 - beta2
    grad_squared = grad * grad
    exp_avg_sq_new = beta2 * exp_avg_sq + beta2_complement * grad_squared
    
    exp_avg_corrected = exp_avg_new / bias_correction1_val
    exp_avg_sq_corrected = exp_avg_sq_new / bias_correction2_val
    
    denom = tl.sqrt(exp_avg_sq_corrected) + eps
    step_size = lr / denom
    param_new = param_decayed - step_size * exp_avg_corrected
    
    # Scatter operation - only store at non-zero indices
    tl.store(param_ptr + p_off, param_new, mask=mask_valid)
    
    if USE_SPARSE_STATES:
        tl.store(exp_avg_ptr + offsets, exp_avg_new, mask=mask_valid)
        tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq_new, mask=mask_valid)
    else:
        tl.store(exp_avg_ptr + m_off, exp_avg_new, mask=mask_valid)
        tl.store(exp_avg_sq_ptr + v_off, exp_avg_sq_new, mask=mask_valid)


def triton_indexed_sparse_adamw_step(
    param, grad, nonzero_indices, exp_avg, exp_avg_sq,
    lr, beta1, beta2, eps, weight_decay, step, block_size=128
):
    """
    Wrapper for indexed sparse AdamW kernel.
    """
    n_indices = nonzero_indices.shape[0]
    
    if n_indices == 0:
        return
    
    bias_correction1 = 1.0 - (beta1 ** step)
    bias_correction2 = 1.0 - (beta2 ** step)
    
    M, N = param.shape
    grid = (triton.cdiv(n_indices, block_size),)
    
    # Detect sparse states
    use_sparse_states = (exp_avg.dim() == 1)
    
    # Tuneable parameters from environment
    num_warps = get_env_int("BSR_NUM_WARPS", 4)
    num_stages = get_env_int("BSR_NUM_STAGES", 2)
    
    # Handle strides for 1D vs 2D
    if use_sparse_states:
        stride_m_m = stride_m_n = 0
        stride_v_m = stride_v_n = 0
    else:
        stride_m_m, stride_m_n = exp_avg.stride(0), exp_avg.stride(1)
        stride_v_m, stride_v_n = exp_avg_sq.stride(0), exp_avg_sq.stride(1)

    indexed_sparse_adamw_kernel[grid](
        param, grad, exp_avg, exp_avg_sq,
        nonzero_indices,
        n_indices,
        M, N,
        param.stride(0), param.stride(1),
        grad.stride(0), grad.stride(1),
        stride_m_m, stride_m_n,
        stride_v_m, stride_v_n,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        bias_correction1_val=bias_correction1,
        bias_correction2_val=bias_correction2,
        BLOCK_SIZE=block_size,
        USE_SPARSE_STATES=use_sparse_states,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@triton.jit
def bsr_sparse_adamw_kernel(
    param_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
    active_blocks_ptr, n_active_blocks,
    M, N,
    stride_p_m, stride_p_n,
    stride_g_m, stride_g_n,
    stride_m_m, stride_m_n,
    stride_v_m, stride_v_n,
    lr, beta1, beta2, eps, weight_decay,
    bias_correction1_val, bias_correction2_val,
    BLOCK_SIZE: tl.constexpr,
    USE_SPARSE_STATES: tl.constexpr,
):
    """
    Block-aware AdamW for BSR-structured masks.
    Updates 16x16 blocks at a time for maximum coalescing.
    """
    pid = tl.program_id(0)
    if pid >= n_active_blocks:
        return
        
    block_idx = tl.load(active_blocks_ptr + pid)
    
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE)
    block_m = block_idx // num_blocks_n
    block_n = block_idx % num_blocks_n
    
    rows = block_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols = block_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]
    
    # Pointers for the 16x16 block
    p_offs = rows[:, None] * stride_p_m + cols[None, :] * stride_p_n
    g_offs = rows[:, None] * stride_g_m + cols[None, :] * stride_g_n
    
    param = tl.load(param_ptr + p_offs, mask=mask, other=0.0)
    grad = tl.load(grad_ptr + g_offs, mask=mask, other=0.0)
    
    if USE_SPARSE_STATES:
        # For BSR, we assume states are also stored as blocks contiguously
        # Each block is BLOCK_SIZE * BLOCK_SIZE elements
        state_block_start = pid * BLOCK_SIZE * BLOCK_SIZE
        state_offs = state_block_start + tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
        exp_avg = tl.load(exp_avg_ptr + state_offs, mask=mask, other=0.0)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + state_offs, mask=mask, other=0.0)
    else:
        m_offs = rows[:, None] * stride_m_m + cols[None, :] * stride_m_n
        v_offs = rows[:, None] * stride_v_m + cols[None, :] * stride_v_n
        exp_avg = tl.load(exp_avg_ptr + m_offs, mask=mask, other=0.0)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + v_offs, mask=mask, other=0.0)
        
    # AdamW math (same as unstructured)
    decay_factor = 1.0 - lr * weight_decay
    param_decayed = param * decay_factor
    
    beta1_complement = 1.0 - beta1
    exp_avg_new = beta1 * exp_avg + beta1_complement * grad
    
    beta2_complement = 1.0 - beta2
    grad_squared = grad * grad
    exp_avg_sq_new = beta2 * exp_avg_sq + beta2_complement * grad_squared
    
    exp_avg_corrected = exp_avg_new / bias_correction1_val
    exp_avg_sq_corrected = exp_avg_sq_new / bias_correction2_val
    
    denom = tl.sqrt(exp_avg_sq_corrected) + eps
    step_size = lr / denom
    param_new = param_decayed - step_size * exp_avg_corrected
    
    # Store results
    tl.store(param_ptr + p_offs, param_new, mask=mask)
    if USE_SPARSE_STATES:
        tl.store(exp_avg_ptr + state_offs, exp_avg_new, mask=mask)
        tl.store(exp_avg_sq_ptr + state_offs, exp_avg_sq_new, mask=mask)
    else:
        tl.store(exp_avg_ptr + m_offs, exp_avg_new, mask=mask)
        tl.store(exp_avg_sq_ptr + v_offs, exp_avg_sq_new, mask=mask)

def triton_bsr_sparse_adamw_step(
    param, grad, active_blocks, exp_avg, exp_avg_sq,
    lr, beta1, beta2, eps, weight_decay, step, block_size=16
):
    """
    Wrapper for BSR-aware sparse AdamW kernel.
    """
    n_active_blocks = active_blocks.shape[0]
    if n_active_blocks == 0:
        return
        
    bias_correction1 = 1.0 - (beta1 ** step)
    bias_correction2 = 1.0 - (beta2 ** step)
    
    M, N = param.shape
    grid = (n_active_blocks,)
    use_sparse_states = (exp_avg.dim() == 1)
    
    num_warps = get_env_int("BSR_NUM_WARPS", 4)
    num_stages = get_env_int("BSR_NUM_STAGES", 2)
    
    if use_sparse_states:
        stride_m_m = stride_m_n = 0
        stride_v_m = stride_v_n = 0
    else:
        stride_m_m, stride_m_n = exp_avg.stride(0), exp_avg.stride(1)
        stride_v_m, stride_v_n = exp_avg_sq.stride(0), exp_avg_sq.stride(1)

    bsr_sparse_adamw_kernel[grid](
        param, grad, exp_avg, exp_avg_sq,
        active_blocks, n_active_blocks,
        M, N,
        param.stride(0), param.stride(1),
        grad.stride(0), grad.stride(1),
        stride_m_m, stride_m_n,
        stride_v_m, stride_v_n,
        lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay,
        bias_correction1_val=bias_correction1,
        bias_correction2_val=bias_correction2,
        BLOCK_SIZE=block_size,
        USE_SPARSE_STATES=use_sparse_states,
        num_warps=num_warps,
        num_stages=num_stages,
    )
