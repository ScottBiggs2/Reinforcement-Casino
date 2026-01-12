
import torch
import triton
import triton.language as tl

@triton.jit
def sparse_grad_weight_kernel(
    grad_weight_ptr, grad_output_ptr, input_ptr, mask_ptr,
    batch_size, output_dim, input_dim,
    stride_go_batch, stride_go_out, stride_in_batch, stride_in_in,
    stride_gw_out, stride_gw_in, stride_m_out, stride_m_in,
    BLOCK_SIZE: tl.constexpr,
):
    """Computes grad_W = grad_output.T @ input ONLY for non-masked blocks."""
    pid = tl.program_id(0)
    num_blocks_in = tl.cdiv(input_dim, BLOCK_SIZE)
    block_out = pid // num_blocks_in
    block_in = pid % num_blocks_in
    
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
    
    if tl.max(mask_block) == 0.0:
        gw_offsets = out_offsets[:, None] * stride_gw_out + in_offsets[None, :] * stride_gw_in
        tl.store(grad_weight_ptr + gw_offsets, 0.0, mask=valid)
        return

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for b_start in range(0, batch_size, 32):
        b_offsets = b_start + tl.arange(0, 32)
        b_mask = b_offsets < batch_size
        
        # Load grad_out (32, BLOCK_SIZE)
        go_offs = b_offsets[:, None] * stride_go_batch + out_offsets[None, :] * stride_go_out
        go_mask = b_mask[:, None] & out_valid[None, :]
        go = tl.load(grad_output_ptr + go_offs, mask=go_mask, other=0.0)
        
        # Load input (32, BLOCK_SIZE)
        in_offs = b_offsets[:, None] * stride_in_batch + in_offsets[None, :] * stride_in_in
        in_mask = b_mask[:, None] & in_valid[None, :]
        inp = tl.load(input_ptr + in_offs, mask=in_mask, other=0.0)
        
        acc += tl.dot(tl.trans(go), inp)
        
    acc = acc * mask_block
    gw_offsets = out_offsets[:, None] * stride_gw_out + in_offsets[None, :] * stride_gw_in
    tl.store(grad_weight_ptr + gw_offsets, acc.to(grad_weight_ptr.dtype.element_ty), mask=valid)

def sparse_weight_gradient_triton(grad_output, input_tensor, mask, block_size=16):
    batch_size, output_dim = grad_output.shape
    _, input_dim = input_tensor.shape
    grad_weight = torch.empty(output_dim, input_dim, device=grad_output.device, dtype=grad_output.dtype)
    
    grid = (triton.cdiv(output_dim, block_size) * triton.cdiv(input_dim, block_size),)
    sparse_grad_weight_kernel[grid](
        grad_weight, grad_output, input_tensor, mask,
        batch_size, output_dim, input_dim,
        grad_output.stride(0), grad_output.stride(1),
        input_tensor.stride(0), input_tensor.stride(1),
        grad_weight.stride(0), grad_weight.stride(1),
        mask.stride(0), mask.stride(1),
        BLOCK_SIZE=block_size,
    )
    return grad_weight
