
import os
import json
import time
import torch
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import sys
import wandb
from transformers import TrainerCallback

# ============================================================================
# TRITON KERNEL (from working notebook)
# ============================================================================

@triton.jit
def sparse_update_kernel(
    # Input pointers
    weights_ptr,      # Pointer to weight matrix
    grad_ptr,         # Pointer to gradient matrix
    mask_ptr,         # Pointer to mask
    output_ptr,       # Pointer to output matrix
    # Matrix dimensions
    M, N,             # Matrix is M×N
    stride_wm, stride_wn,  # Strides for weights
    stride_gm, stride_gn,  # Strides for gradients
    stride_mm, stride_mn,  # Strides for mask
    stride_om, stride_on,  # Strides for output
    # Learning rate
    lr,
    # Block size (compile-time constant)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for sparse weight updates.
    
    Each program instance processes a BLOCK_SIZE × BLOCK_SIZE block.
    """
    # Get this program's block ID
    pid = tl.program_id(0)  # 0 because we have a 1D grid
    
    # Calculate which row and column block this program handles
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE)  # Number of block columns
    block_row = pid // num_blocks_n
    block_col = pid % num_blocks_n
    
    # Calculate starting position in the matrix
    row_start = block_row * BLOCK_SIZE
    col_start = block_col * BLOCK_SIZE
    
    # Create offset ranges for this block
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    
    # Create 2D offset grid (BLOCK_SIZE × BLOCK_SIZE)
    row_offsets = row_offsets[:, None]  # Shape: [BLOCK_SIZE, 1]
    col_offsets = col_offsets[None, :]  # Shape: [1, BLOCK_SIZE]
    
    # Calculate memory offsets
    w_offsets = row_offsets * stride_wm + col_offsets * stride_wn
    g_offsets = row_offsets * stride_gm + col_offsets * stride_gn
    m_offsets = row_offsets * stride_mm + col_offsets * stride_mn
    o_offsets = row_offsets * stride_om + col_offsets * stride_on
    
    # Create mask for valid elements (handle edge cases)
    mask_valid = (row_offsets < M) & (col_offsets < N)
    
    # Load data from global memory
    w_block = tl.load(weights_ptr + w_offsets, mask=mask_valid, other=0.0)
    g_block = tl.load(grad_ptr + g_offsets, mask=mask_valid, other=0.0)
    m_block = tl.load(mask_ptr + m_offsets, mask=mask_valid, other=0.0)
    
    # Compute update: W_new = W_old + lr * (grad * mask)
    masked_grad = g_block * m_block
    w_new = w_block + lr * masked_grad
    
    # Store result back to global memory
    tl.store(output_ptr + o_offsets, w_new, mask=mask_valid)


def triton_sparse_update(weights, gradient, mask, lr, block_size=32):
    """
    Apply sparse gradient update using Triton kernel.
    
    Args:
        weights: Parameter tensor (M x N)
        gradient: Gradient tensor (M x N)
        mask: Binary mask tensor (M x N)
        lr: Learning rate
        block_size: Block size for Triton kernel
        
    Returns:
        Updated weights tensor
    """
    M, N = weights.shape
    
    # Allocate output tensor
    output = torch.empty_like(weights)
    
    # Calculate grid size (how many blocks to process)
    num_blocks_m = triton.cdiv(M, block_size)
    num_blocks_n = triton.cdiv(N, block_size)
    grid = (num_blocks_m * num_blocks_n,)
    
    # Launch kernel
    sparse_update_kernel[grid](
        weights, gradient, mask, output,
        M, N,
        weights.stride(0), weights.stride(1),
        gradient.stride(0), gradient.stride(1),
        mask.stride(0), mask.stride(1),
        output.stride(0), output.stride(1),
        lr,
        BLOCK_SIZE=block_size,
    )
    
    return output
