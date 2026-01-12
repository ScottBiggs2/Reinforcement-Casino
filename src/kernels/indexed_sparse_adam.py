
import triton
import triton.language as tl

@triton.jit
def indexed_sparse_adamw_kernel(
    # Pointers to FULL tensors (flattened)
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    # Pointer to non-zero indices
    indices_ptr,
    n_indices,
    # Hyperparameters
    lr: tl.constexpr,
    beta1: tl.constexpr,
    beta2: tl.constexpr,
    eps: tl.constexpr,
    weight_decay: tl.constexpr,
    # Precomputed bias corrections (not constexpr to avoid recompilation)
    bias_correction1_val,
    bias_correction2_val,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Indexed sparse AdamW without kernel recompilation.
    """
    pid = tl.program_id(0)
    
    # Calculate which non-zero indices this block handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_valid = offsets < n_indices
    
    # Load the actual flattened indices we need to update
    idx = tl.load(indices_ptr + offsets, mask=mask_valid, other=0)
    
    # Gather operation - only load values at non-zero indices
    param = tl.load(param_ptr + idx, mask=mask_valid, other=0.0)
    grad = tl.load(grad_ptr + idx, mask=mask_valid, other=0.0)
    exp_avg = tl.load(exp_avg_ptr + idx, mask=mask_valid, other=0.0)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + idx, mask=mask_valid, other=0.0)
    
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
    tl.store(param_ptr + idx, param_new, mask=mask_valid)
    tl.store(exp_avg_ptr + idx, exp_avg_new, mask=mask_valid)
    tl.store(exp_avg_sq_ptr + idx, exp_avg_sq_new, mask=mask_valid)


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
    
    # Flatten all tensors for indexed access (creates views, not copies)
    param_flat = param.flatten()
    grad_flat = grad.flatten()
    exp_avg_flat = exp_avg.flatten()
    exp_avg_sq_flat = exp_avg_sq.flatten()
    
    # Only call .contiguous() if needed
    if not param_flat.is_contiguous():
        param_flat = param_flat.contiguous()
    if not grad_flat.is_contiguous():
        grad_flat = grad_flat.contiguous()
    if not exp_avg_flat.is_contiguous():
        exp_avg_flat = exp_avg_flat.contiguous()
    if not exp_avg_sq_flat.is_contiguous():
        exp_avg_sq_flat = exp_avg_sq_flat.contiguous()
    
    grid = (triton.cdiv(n_indices, block_size),)
    
    indexed_sparse_adamw_kernel[grid](
        param_flat, grad_flat, exp_avg_flat, exp_avg_sq_flat,
        nonzero_indices,
        n_indices,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        bias_correction1_val=bias_correction1,
        bias_correction2_val=bias_correction2,
        BLOCK_SIZE=block_size,
    )
    
    # Update .data directly to avoid overhead (modifications to views are reflected if contiguous)
    # If we made copies with .contiguous(), we might need to copy back?
    # Ideally, we assume they are contiguous or the layout matches.
    # The original script did: 
    # param.data = param_flat.reshape(param.shape)
    # Let's keep that for safety.
    
    if param_flat.data_ptr() != param.data_ptr():
         # If we copied, we must copy back. 
         # But wait, triton writes to the pointer we gave it. 
         # If param_flat is a copy, we wrote to the copy.
         # So we must copy back to param.
         param.copy_(param_flat.reshape(param.shape))
    else:
         # If it was a view, modifications are in place. nothing to do.
         pass
         
    # To be safe and identical to v3 logic:
    param.data = param_flat.reshape(param.shape)
    exp_avg.data = exp_avg_flat.reshape(exp_avg.shape)
    exp_avg_sq.data = exp_avg_sq_flat.reshape(exp_avg_sq.shape)
