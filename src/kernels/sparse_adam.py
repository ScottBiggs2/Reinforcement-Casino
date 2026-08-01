
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


def triton_sparse_adam_update(weights, gradient, mask, exp_avg, exp_avg_sq, 
                               lr, beta1, beta2, eps, weight_decay, step, 
                               adamw=True, block_size=32):
    """
    Apply sparse Adam/AdamW update using Triton kernel.
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
