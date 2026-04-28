
import torch
import triton
import triton.language as tl

import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

def get_env_int(name, default):
    return int(os.environ.get(name, default))

def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class _BsrRecompileSigStats:
    first_seen_s: float
    launches: int = 0
    first_ms: Optional[float] = None
    last_ms: Optional[float] = None
    total_ms: float = 0.0
    compile_suspected: bool = False
    first_step: Optional[int] = None
    last_step: Optional[int] = None


# Global (per-process) diagnostics store. Off by default.
_BSR_DIAG: Dict[Tuple[Any, ...], _BsrRecompileSigStats] = {}
_BSR_DIAG_USE_ATOMIC_SEEN: Dict[bool, int] = {}


def bsr_recompile_diag_summary(reset: bool = False) -> str:
    """
    Return a concise summary string. Safe to call on CPU at end of a phase.
    """
    global _BSR_DIAG, _BSR_DIAG_USE_ATOMIC_SEEN
    if not _BSR_DIAG:
        out = "[bsr_diag] no kernel launches recorded"
        if reset:
            _BSR_DIAG_USE_ATOMIC_SEEN = {}
        return out

    suspects = [
        (k, v) for k, v in _BSR_DIAG.items() if v.first_ms is not None and v.compile_suspected
    ]
    suspects.sort(key=lambda kv: kv[1].first_ms or 0.0, reverse=True)
    top = suspects[:5]

    unique = len(_BSR_DIAG)
    flips = len(_BSR_DIAG_USE_ATOMIC_SEEN)
    out_lines = [
        f"[bsr_diag] unique_signatures={unique} use_atomic_values_seen={flips} {sorted(_BSR_DIAG_USE_ATOMIC_SEEN.keys())}",
    ]
    if top:
        out_lines.append("[bsr_diag] top_compile_suspects(first_ms):")
        for sig, st in top:
            out_lines.append(
                f"  first_ms={st.first_ms:.3f} launches={st.launches} step={st.first_step}->{st.last_step} sig={sig}"
            )
    if reset:
        _BSR_DIAG = {}
        _BSR_DIAG_USE_ATOMIC_SEEN = {}
    return "\n".join(out_lines)

@triton.jit
def sparse_grad_weight_kernel(
    grad_weight_ptr, grad_output_ptr, input_ptr, mask_ptr, active_blocks_ptr,
    batch_size, output_dim, input_dim,
    stride_go_batch, stride_go_out, stride_in_batch, stride_in_in,
    stride_gw_out, stride_gw_in, stride_m_out, stride_m_in,
    BLOCK_SIZE: tl.constexpr,
    BATCH_BLOCK_SIZE: tl.constexpr,
    USE_TF32: tl.constexpr,
    USE_ATOMIC: tl.constexpr,
):
    """Computes grad_W = grad_output.T @ input ONLY for non-masked blocks."""
    pid_block = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    # Load the block index from the active blocks array
    block_idx = tl.load(active_blocks_ptr + pid_block)
    
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
    
    # Calculate batch range for this program
    if USE_ATOMIC:
        num_batch_chunks = tl.num_programs(1)
        batch_chunk_size = tl.cdiv(batch_size, num_batch_chunks)
        batch_start = pid_batch * batch_chunk_size
        batch_end = tl.minimum(batch_start + batch_chunk_size, batch_size)
    else:
        batch_start = 0
        batch_end = batch_size

    for b_start in range(batch_start, batch_end, BATCH_BLOCK_SIZE):
        b_offsets = b_start + tl.arange(0, BATCH_BLOCK_SIZE)
        b_mask = b_offsets < batch_end
        
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
    
    if USE_ATOMIC:
        tl.atomic_add(grad_weight_ptr + gw_offsets, acc.to(grad_weight_ptr.dtype.element_ty), mask=valid)
    else:
        tl.store(grad_weight_ptr + gw_offsets, acc.to(grad_weight_ptr.dtype.element_ty), mask=valid)


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
    stride_go_batch,
    stride_go_out,
    stride_w_out,
    stride_w_in,
    stride_gi_batch,
    stride_gi_in,
    stride_m_out,
    stride_m_in,
    BLOCK_SIZE: tl.constexpr,
    BATCH_BLOCK_SIZE: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """
    grad_input = grad_output @ W over active blocks only.
    Accumulates into grad_input via atomic_add (fp32) so overlapping active blocks are correct.

    Grid is 1-D over active blocks only (one program per block, full batch in inner loops)
    to avoid duplicate atomic_add when batch dimension would be split without atomics.
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

    m_offsets = out_offsets[:, None] * stride_m_out + in_offsets[None, :] * stride_m_in
    mask_block = tl.load(mask_ptr + m_offsets, mask=valid_w, other=0.0)
    Wblk = Wblk * mask_block

    batch_start = 0
    batch_end = batch_size

    for b_start in range(batch_start, batch_end, BATCH_BLOCK_SIZE):
        b_offsets = b_start + tl.arange(0, BATCH_BLOCK_SIZE)
        b_mask = b_offsets < batch_end

        go_offs = b_offsets[:, None] * stride_go_batch + out_offsets[None, :] * stride_go_out
        go_mask = b_mask[:, None] & out_valid[None, :]
        go = tl.load(grad_output_ptr + go_offs, mask=go_mask, other=0.0)

        if USE_TF32:
            go_f = go.to(tl.float32)
            W_f = Wblk.to(tl.float32)
            contrib = tl.dot(go_f, W_f, allow_tf32=True)
        else:
            contrib = tl.dot(go, Wblk)

        gi_offs = b_offsets[:, None] * stride_gi_batch + in_offsets[None, :] * stride_gi_in
        gi_mask = b_mask[:, None] & in_valid[None, :]
        tl.atomic_add(grad_input_ptr + gi_offs, contrib, mask=gi_mask)


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
        
    # Parallelize over batch dimension for high sparsity (fewer active blocks)
    # H200 has ~132 SMs; we want enough programs to saturate them (target 1024+)
    num_batch_chunks = get_env_int("BSR_BATCH_CHUNKS", 1)
    if num_batch_chunks == 1:
        target_total_programs = 1024
        num_batch_chunks = (target_total_programs + num_active_blocks - 1) // num_active_blocks
        
    grid = (num_active_blocks, num_batch_chunks)
    # Phase-constant atomic choice (prevents layer-dependent constexpr flips / recompiles).
    # Default off unless explicitly enabled.
    use_atomic = bool(get_env_int("BSR_USE_ATOMIC", 0))
    
    # Tuneable parameters from environment
    batch_block_size = get_env_int("BSR_BATCH_BLOCK_SIZE", 64)
    num_warps = get_env_int("BSR_NUM_WARPS", 4)
    num_stages = get_env_int("BSR_NUM_STAGES", 2)

    # Optional diagnostics (off by default). Uses CUDA events.
    diag = _env_flag("RL_CASINO_BSR_RECOMPILE_DIAG", "0")
    diag_sync = _env_flag("RL_CASINO_BSR_RECOMPILE_DIAG_SYNC", "0")
    diag_step = int(os.environ.get("RL_CASINO_BSR_RECOMPILE_DIAG_STEP", "-1"))
    step_for_sig = diag_step if diag_step >= 0 else None

    sig = None
    start_evt = end_evt = None
    if diag and grad_output.is_cuda:
        sig = (
            int(block_size),
            int(batch_block_size),
            bool(use_tf32),
            bool(use_atomic),
            str(grad_output.dtype),
            str(input_tensor.dtype),
            str(mask.dtype),
            int(output_dim),
            int(input_dim),
        )
        _BSR_DIAG_USE_ATOMIC_SEEN[bool(use_atomic)] = _BSR_DIAG_USE_ATOMIC_SEEN.get(bool(use_atomic), 0) + 1
        if sig not in _BSR_DIAG:
            _BSR_DIAG[sig] = _BsrRecompileSigStats(first_seen_s=time.time(), first_step=step_for_sig)
        st = _BSR_DIAG[sig]
        st.launches += 1
        st.last_step = step_for_sig
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
    
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
        USE_ATOMIC=use_atomic,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if diag and diag_sync and start_evt is not None and end_evt is not None and sig is not None:
        end_evt.record()
        torch.cuda.synchronize()
        ms = float(start_evt.elapsed_time(end_evt))
        st = _BSR_DIAG[sig]
        st.last_ms = ms
        st.total_ms += ms
        if st.first_ms is None:
            st.first_ms = ms
        # Mark compile suspected if the first launch is much slower than the most recent.
        if st.launches >= 2 and st.first_ms is not None and st.last_ms is not None:
            if st.first_ms > max(50.0, 5.0 * st.last_ms):
                st.compile_suspected = True
    return grad_weight


def sparse_grad_input_triton(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor,
    active_blocks: Optional[torch.Tensor] = None,
    block_size: int = 16,
    use_tf32: bool = False,
) -> torch.Tensor:
    """
    grad_input = grad_output @ weight with block sparsity (active blocks only).

    Returns a tensor with the same dtype as ``grad_output``. Internally accumulates
    in fp32 for stable ``atomic_add``, then casts.
    """
    batch_size, output_dim = grad_output.shape
    out_w, input_dim = weight.shape
    if out_w != output_dim:
        raise ValueError(f"grad_output out_dim {output_dim} != weight shape[0] {out_w}")

    out_dtype = grad_output.dtype
    device = grad_output.device
    # fp32 accumulation for atomic_add stability
    grad_input = torch.zeros((batch_size, input_dim), device=device, dtype=torch.float32)

    if active_blocks is None:
        num_blocks_m = (output_dim + block_size - 1) // block_size
        num_blocks_n = (input_dim + block_size - 1) // block_size
        pad_m = num_blocks_m * block_size - output_dim
        pad_n = num_blocks_n * block_size - input_dim
        mask_f = mask.float() if mask.dtype == torch.bool else mask
        padded_mask = torch.nn.functional.pad(mask_f, (0, pad_n, 0, pad_m), value=0)
        blocks = padded_mask.view(num_blocks_m, block_size, num_blocks_n, block_size)
        block_active = blocks.any(dim=1).any(dim=2)
        active_blocks = torch.nonzero(block_active.flatten(), as_tuple=True)[0].to(torch.int32)

    if active_blocks.dtype != torch.int32:
        active_blocks = active_blocks.to(torch.int32)
    if not active_blocks.is_contiguous():
        active_blocks = active_blocks.contiguous()

    num_active_blocks = active_blocks.shape[0]
    if num_active_blocks == 0:
        return torch.zeros((batch_size, input_dim), device=device, dtype=out_dtype)

    # One program per active block (full batch inner loop). Avoids duplicate atomic_add across batch shards.
    grid = (num_active_blocks,)

    batch_block_size = get_env_int("BSR_BATCH_BLOCK_SIZE", 64)
    num_warps = get_env_int("BSR_NUM_WARPS", 4)
    num_stages = get_env_int("BSR_NUM_STAGES", 2)

    grad_out = grad_output
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()
    w = weight
    if not w.is_contiguous():
        w = w.contiguous()
    m = mask
    if not m.is_contiguous():
        m = m.contiguous()

    sparse_grad_input_kernel[grid](
        grad_input,
        grad_out,
        w,
        m,
        active_blocks,
        batch_size,
        output_dim,
        input_dim,
        grad_out.stride(0),
        grad_out.stride(1),
        w.stride(0),
        w.stride(1),
        grad_input.stride(0),
        grad_input.stride(1),
        m.stride(0),
        m.stride(1),
        BLOCK_SIZE=block_size,
        BATCH_BLOCK_SIZE=batch_block_size,
        USE_TF32=use_tf32,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return grad_input.to(out_dtype)
