
import os
from typing import Optional

import torch
import triton
import triton.language as tl


def _get_env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# Autotune configs kept small (5) to bound first-call tuning cost: 226 layers ×
# up to a few unique (output_dim, input_dim) pairs × 5 trials each is still
# seconds, not minutes. Keys on output_dim + input_dim so each layer shape
# picks its best config once and caches it.
_BSR_BACKWARD_CONFIGS = [
    triton.Config({'BLOCK_SIZE': 16}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_SIZE': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE': 64}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_SIZE': 64}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_BSR_BACKWARD_CONFIGS, key=['output_dim', 'input_dim'])
@triton.jit
def sparse_grad_weight_kernel(
    grad_weight_ptr, grad_output_ptr, input_ptr, mask_ptr,
    batch_size, output_dim, input_dim,
    stride_go_batch, stride_go_out, stride_in_batch, stride_in_in,
    stride_gw_out, stride_gw_in, stride_m_out, stride_m_in,
    BLOCK_SIZE: tl.constexpr,
    USE_TF32: tl.constexpr,
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
        
        if USE_TF32:
            go_f32 = go.to(tl.float32)
            inp_f32 = inp.to(tl.float32)
            acc += tl.dot(tl.trans(go_f32), inp_f32, allow_tf32=True)
        else:
            acc += tl.dot(tl.trans(go), inp)
        
    acc = acc * mask_block
    gw_offsets = out_offsets[:, None] * stride_gw_out + in_offsets[None, :] * stride_gw_in
    tl.store(grad_weight_ptr + gw_offsets, acc.to(grad_weight_ptr.dtype.element_ty), mask=valid)

def sparse_weight_gradient_triton(grad_output, input_tensor, mask, block_size=16, use_tf32=False):
    # `block_size` kept for API compat with callers in mlps/bsr_sparse_mlp.py; the
    # actual BLOCK_SIZE is picked by @triton.autotune at call time from the
    # configs above and cached per (output_dim, input_dim) key.
    del block_size
    batch_size, output_dim = grad_output.shape
    _, input_dim = input_tensor.shape
    grad_weight = torch.empty(output_dim, input_dim, device=grad_output.device, dtype=grad_output.dtype)

    grid = lambda META: (triton.cdiv(output_dim, META['BLOCK_SIZE']) * triton.cdiv(input_dim, META['BLOCK_SIZE']),)
    sparse_grad_weight_kernel[grid](
        grad_weight, grad_output, input_tensor, mask,
        batch_size, output_dim, input_dim,
        grad_output.stride(0), grad_output.stride(1),
        input_tensor.stride(0), input_tensor.stride(1),
        grad_weight.stride(0), grad_weight.stride(1),
        mask.stride(0), mask.stride(1),
        USE_TF32=use_tf32,
    )
    return grad_weight


# ============================================================================
# Sparse grad-input kernel (B1 baseline, ported from cav_fixes b9fec55).
# Computes grad_input = grad_output @ W over active 16x16 blocks only,
# accumulating via fp32 atomic_add.
# ============================================================================

@triton.jit
def sparse_grad_input_kernel(
    grad_input_ptr,
    grad_output_ptr,
    weight_ptr,
    mask_ptr,
    active_blocks_ptr,
    batch_size,
    output_dim,
    input_dim,
    stride_go_batch, stride_go_out,
    stride_w_out, stride_w_in,
    stride_gi_batch, stride_gi_in,
    stride_m_out, stride_m_in,
    BLOCK_SIZE: tl.constexpr,
    BATCH_BLOCK_SIZE: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """
    grad_input = grad_output @ W over active blocks only. fp32 atomic_add into grad_input.
    Grid is 1-D over active blocks (one program per block, full batch in inner loop) to
    avoid duplicate atomic_add when batch is split.
    """
    pid_block = tl.program_id(0)
    block_idx = tl.load(active_blocks_ptr + pid_block).to(tl.int32)

    num_blocks_in = tl.cdiv(input_dim, BLOCK_SIZE)
    block_out = block_idx // num_blocks_in
    block_in = block_idx % num_blocks_in

    out_start = block_out * BLOCK_SIZE
    in_start = block_in * BLOCK_SIZE
    out_offsets = out_start + tl.arange(0, BLOCK_SIZE)
    in_offsets = in_start + tl.arange(0, BLOCK_SIZE)

    out_valid = out_offsets < output_dim
    in_valid = in_offsets < input_dim
    valid_w = out_valid[:, None] & in_valid[None, :]

    w_offs = out_offsets[:, None] * stride_w_out + in_offsets[None, :] * stride_w_in
    Wblk = tl.load(weight_ptr + w_offs, mask=valid_w, other=0.0)

    m_offs = out_offsets[:, None] * stride_m_out + in_offsets[None, :] * stride_m_in
    mask_block = tl.load(mask_ptr + m_offs, mask=valid_w, other=0.0)
    Wblk = Wblk * mask_block

    for b_start in range(0, batch_size, BATCH_BLOCK_SIZE):
        b_offsets = b_start + tl.arange(0, BATCH_BLOCK_SIZE)
        b_mask = b_offsets < batch_size

        go_offs = b_offsets[:, None] * stride_go_batch + out_offsets[None, :] * stride_go_out
        go_mask = b_mask[:, None] & out_valid[None, :]
        go = tl.load(grad_output_ptr + go_offs, mask=go_mask, other=0.0)

        if USE_TF32:
            contrib = tl.dot(go.to(tl.float32), Wblk.to(tl.float32), allow_tf32=True)
        else:
            contrib = tl.dot(go, Wblk)

        gi_offs = b_offsets[:, None] * stride_gi_batch + in_offsets[None, :] * stride_gi_in
        gi_mask = b_mask[:, None] & in_valid[None, :]
        tl.atomic_add(grad_input_ptr + gi_offs, contrib, mask=gi_mask)


def sparse_grad_input_triton(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor,
    active_blocks: Optional[torch.Tensor] = None,
    block_size: int = 16,
    use_tf32: bool = False,
) -> torch.Tensor:
    """
    grad_input = grad_output @ weight, sparse over 16x16 active blocks of `mask`.
    Returns dtype matching grad_output (internal accumulation in fp32 for atomic stability).
    """
    batch_size, output_dim = grad_output.shape
    out_w, input_dim = weight.shape
    if out_w != output_dim:
        raise ValueError(f"grad_output out_dim {output_dim} != weight rows {out_w}")

    out_dtype = grad_output.dtype
    device = grad_output.device
    grad_input = torch.zeros((batch_size, input_dim), device=device, dtype=torch.float32)

    if active_blocks is None:
        num_blocks_m = (output_dim + block_size - 1) // block_size
        num_blocks_n = (input_dim + block_size - 1) // block_size
        pad_m = num_blocks_m * block_size - output_dim
        pad_n = num_blocks_n * block_size - input_dim
        mask_f = mask.float() if mask.dtype == torch.bool else mask
        padded = torch.nn.functional.pad(mask_f, (0, pad_n, 0, pad_m), value=0)
        blocks = padded.view(num_blocks_m, block_size, num_blocks_n, block_size)
        block_active = blocks.any(dim=1).any(dim=2)
        active_blocks = torch.nonzero(block_active.flatten(), as_tuple=True)[0].to(torch.int32)

    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(torch.int32)
    if not active_blocks.is_contiguous():
        active_blocks = active_blocks.contiguous()

    n_active = active_blocks.shape[0]
    if n_active == 0:
        return torch.zeros((batch_size, input_dim), device=device, dtype=out_dtype)

    grad_out = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    w = weight if weight.is_contiguous() else weight.contiguous()
    m = mask if mask.is_contiguous() else mask.contiguous()

    sparse_grad_input_kernel[(n_active,)](
        grad_input, grad_out, w, m, active_blocks,
        batch_size, output_dim, input_dim,
        grad_out.stride(0), grad_out.stride(1),
        w.stride(0), w.stride(1),
        grad_input.stride(0), grad_input.stride(1),
        m.stride(0), m.stride(1),
        BLOCK_SIZE=block_size,
        BATCH_BLOCK_SIZE=_get_env_int("BSR_BATCH_BLOCK_SIZE", 64),
        USE_TF32=use_tf32,
        num_warps=_get_env_int("BSR_NUM_WARPS", 4),
        num_stages=_get_env_int("BSR_NUM_STAGES", 2),
    )
    return grad_input.to(out_dtype)
