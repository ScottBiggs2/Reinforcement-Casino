
import torch
import triton
import triton.language as tl

import os

def get_env_int(name, default):
    return int(os.environ.get(name, default))

@triton.jit
def sparse_grad_weight_kernel(
    grad_weight_ptr, grad_output_ptr, input_ptr, mask_ptr, active_blocks_ptr,
    batch_size, output_dim, input_dim,
    stride_go_batch, stride_go_out, stride_in_batch, stride_in_in,
    stride_gw_out, stride_gw_in, stride_m_out, stride_m_in,
    BLOCK_SIZE: tl.constexpr,
    BATCH_BLOCK_SIZE: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """Computes grad_W = grad_output.T @ input ONLY for non-masked blocks."""
    pid = tl.program_id(0)
    
    # Load the block index from the active blocks array
    block_idx = tl.load(active_blocks_ptr + pid)
    
    num_blocks_in = tl.cdiv(input_dim, BLOCK_SIZE)
    block_out = block_idx // num_blocks_in
    block_in = block_idx % num_blocks_in
    
    out_start = block_out * BLOCK_SIZE
    in_start = block_in * BLOCK_SIZE
    out_offsets = out_start + tl.arange(0, BLOCK_SIZE)
    in_offsets = in_start + tl.arange(0, BLOCK_SIZE)
    
    # Check mask for this block
    out_valid = out_offsets < output_dim
    in_valid = in_offsets < input_dim
    valid = out_valid[:, None] & in_valid[None, :]
    
    m_offsets = out_offsets[:, None] * stride_m_out + in_offsets[None, :] * stride_m_in
    mask_block = tl.load(mask_ptr + m_offsets, mask=valid, other=0.0)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for b_start in range(0, batch_size, BATCH_BLOCK_SIZE):
        b_offsets = b_start + tl.arange(0, BATCH_BLOCK_SIZE)
        b_mask = b_offsets < batch_size
        
        # Load grad_out (BATCH_BLOCK_SIZE, BLOCK_SIZE)
        go_offs = b_offsets[:, None] * stride_go_batch + out_offsets[None, :] * stride_go_out
        go_mask = b_mask[:, None] & out_valid[None, :]
        go = tl.load(grad_output_ptr + go_offs, mask=go_mask, other=0.0)
        
        # Load input (BATCH_BLOCK_SIZE, BLOCK_SIZE)
        in_offs = b_offsets[:, None] * stride_in_batch + in_offsets[None, :] * stride_in_in
        in_mask = b_mask[:, None] & in_valid[None, :]
        inp = tl.load(input_ptr + in_offs, mask=in_mask, other=0.0)
        
        if USE_TF32:
            go_f32 = go.to(tl.float32)
            inp_f32 = inp.to(tl.float32)
            acc += tl.dot(tl.trans(go_f32), inp_f32, allow_tf32=True)
        else:
            acc += tl.dot(tl.trans(go), inp)
        
    acc = acc * mask_block
    gw_offsets = out_offsets[:, None] * stride_gw_out + in_offsets[None, :] * stride_gw_in
    tl.store(grad_weight_ptr + gw_offsets, acc.to(grad_weight_ptr.dtype.element_ty), mask=valid)

def sparse_weight_gradient_triton(grad_output, input_tensor, mask, active_blocks=None, block_size=16, use_tf32=False):
    batch_size, output_dim = grad_output.shape
    _, input_dim = input_tensor.shape
    grad_weight = torch.zeros((output_dim, input_dim), device=grad_output.device, dtype=grad_output.dtype)
    
    if active_blocks is None:
        # Fallback if active_blocks is not precomputed
        num_blocks_m = (output_dim + block_size - 1) // block_size
        num_blocks_n = (input_dim + block_size - 1) // block_size
        pad_m = num_blocks_m * block_size - output_dim
        pad_n = num_blocks_n * block_size - input_dim
        
        # Safe float conversion for pooling
        mask_f = mask.float() if mask.dtype == torch.bool else mask
        padded_mask = torch.nn.functional.pad(mask_f, (0, pad_n, 0, pad_m), value=0)
        blocks = padded_mask.view(num_blocks_m, block_size, num_blocks_n, block_size)
        block_active = blocks.any(dim=1).any(dim=2)
        active_blocks = torch.nonzero(block_active.flatten(), as_tuple=True)[0].to(torch.int32)
        
    num_active_blocks = active_blocks.shape[0]
    if num_active_blocks == 0:
        return grad_weight
        
    grid = (num_active_blocks,)
    
    # Tuneable parameters from environment
    batch_block_size = get_env_int("BSR_BATCH_BLOCK_SIZE", 64)
    num_warps = get_env_int("BSR_NUM_WARPS", 4)
    num_stages = get_env_int("BSR_NUM_STAGES", 2)
    
    sparse_grad_weight_kernel[grid](
        grad_weight, grad_output, input_tensor, mask, active_blocks,
        batch_size, output_dim, input_dim,
        grad_output.stride(0), grad_output.stride(1),
        input_tensor.stride(0), input_tensor.stride(1),
        grad_weight.stride(0), grad_weight.stride(1),
        mask.stride(0), mask.stride(1),
        BLOCK_SIZE=block_size,
        BATCH_BLOCK_SIZE=batch_block_size,
        USE_TF32=use_tf32,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return grad_weight
