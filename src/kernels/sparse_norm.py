
import torch
import triton
import triton.language as tl

@triton.jit
def bsr_grad_norm_kernel(
    grad_ptr, active_blocks_ptr, n_active_blocks,
    M, N,
    stride_g_m, stride_g_n,
    partial_norms_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Computes the sum of squares of a 16x16 BSR block."""
    pid = tl.program_id(0)
    if pid >= n_active_blocks:
        return
        
    block_idx = tl.load(active_blocks_ptr + pid)
    
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE)
    block_m = block_idx // num_blocks_n
    block_n = block_idx % num_blocks_n
    
    rows = block_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols = block_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    
    g_offs = rows[:, None] * stride_g_m + cols[None, :] * stride_g_n
    grad = tl.load(grad_ptr + g_offs, mask=mask, other=0.0).to(tl.float32)
    
    sq_sum = tl.sum(grad * grad)
    tl.store(partial_norms_ptr + pid, sq_sum)

def compute_bsr_grad_norm_sq(grad, active_blocks, block_size=16):
    """Returns the sum of squares of the active blocks in a BSR gradient."""
    n_active_blocks = active_blocks.shape[0]
    if n_active_blocks == 0:
        return torch.tensor(0.0, device=grad.device)
        
    M, N = grad.shape
    # Partial sums for each block
    partial_norms = torch.empty(n_active_blocks, device=grad.device, dtype=torch.float32)
    
    grid = (n_active_blocks,)
    bsr_grad_norm_kernel[grid](
        grad, active_blocks, n_active_blocks,
        M, N,
        grad.stride(0), grad.stride(1),
        partial_norms,
        BLOCK_SIZE=block_size,
    )
    
    return partial_norms.sum()

@triton.jit
def unstructured_grad_norm_kernel(
    grad_ptr, indices_ptr, n_indices,
    partial_norms_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Computes sum of squares for unstructured indexed gradients."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_indices
    
    # For unstructured, indices are already flattened
    idx = tl.load(indices_ptr + offs, mask=mask, other=0)
    grad = tl.load(grad_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    
    sq_sum = tl.sum(grad * grad)
    tl.store(partial_norms_ptr + pid, sq_sum)

def compute_unstructured_grad_norm_sq(grad, indices, block_size=128):
    """Returns the sum of squares of the indexed elements in a gradient."""
    n_indices = indices.shape[0]
    if n_indices == 0:
        return torch.tensor(0.0, device=grad.device)
        
    num_programs = (n_indices + block_size - 1) // block_size
    partial_norms = torch.empty(num_programs, device=grad.device, dtype=torch.float32)
    
    unstructured_grad_norm_kernel[(num_programs,)](
        grad.flatten(), indices, n_indices,
        partial_norms,
        BLOCK_SIZE=block_size,
    )
    
    return partial_norms.sum()
