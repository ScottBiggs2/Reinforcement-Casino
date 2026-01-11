#!/usr/bin/env python3
"""
Sparse DPO Training V4 - Integrated BSR Sparse MLP (Master Script)

This script integrates the BSR (Block Sparse Row) backward pass for MLPs
and provides flexible optimizer options (SGD, AdamW, and SparseAdamW).

Key Features:
1. CUSTOM BSR KERNEL: Sparse gradient computation that respects element-wise masks.
2. DYNAMIC OPTIMIZER: Toggle between SGD, AdamW, and indexed SparseAdamW.
3. IN-PLACE REPLACEMENT: Injects SparseLinearLayers into the model.

Run: python src/magic/sparse_DPO_v4.py \
  --checkpoint checkpoints_gemma3_dpo/checkpoint-100 \
  --mask masks/top_10.0pct_momentum_w25_step25.pt \
  --n_steps 10 \
  --optimizer sgd
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import argparse
from typing import List, Dict, Any
from datetime import datetime
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME_DEFAULT = "google/gemma-3-270m-it"
DATASET_NAME = "qihoo360/Light-R1-DPOData"
SUBSET_SIZE = 10
MASK_PATH = "masks/top_10.0pct_momentum_w25_step25.pt"
BLOCK_SIZE_KERNELS = 128  # For AdamW kernels
BLOCK_SIZE_BSR = 16       # For BSR backward kernels (finer grain = better accuracy)

def sanitize_model_name(model_name: str) -> str:
    """Convert HuggingFace model name to filesystem-safe string."""
    sanitized = model_name.replace("/", "_").replace("-", "_").lower()
    return "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized).strip("_")

# ============================================================================
# TRITON KERNELS - BSR SPARSE MLP BACKWARD
# ============================================================================

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

# ============================================================================
# SPARE MLP AUTOGRAD & LAYER
# ============================================================================

class SparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, mask, block_size=16):
        ctx.save_for_backward(input, weight, mask)
        ctx.has_bias = bias is not None
        ctx.block_size = block_size
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight, mask = ctx.saved_tensors
        block_size = ctx.block_size
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight
        if ctx.needs_input_grad[1]:
            # BSR Sparse Weights Gradient
            grad_weight = sparse_weight_gradient_triton(
                grad_output.contiguous(), input_tensor.contiguous(), mask, block_size=block_size
            )
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
            
        return grad_input, grad_weight, grad_bias, None, None

class SparseLinearLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, mask=None, block_size=16):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)
        self.block_size = block_size
        
    def forward(self, input):
        if self.mask is not None:
             return SparseLinearFunction.apply(input, self.weight, self.bias, self.mask, self.block_size)
        return F.linear(input, self.weight, self.bias)

def replace_linear_modules(model, mask_dict, block_size=16):
    """Recursively replaces nn.Linear with SparseLinearLayer."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_name = f"{name}.weight"
            mask = mask_dict.get(weight_name)
            if mask is None: # Try fuzzy match
                for k, v in mask_dict.items():
                    if weight_name.endswith(k) or k.endswith(weight_name):
                        mask = v; break
            
            if mask is not None:
                sparse_layer = SparseLinearLayer(
                    module.in_features, module.out_features, module.bias is not None,
                    mask=mask, block_size=block_size
                )
                sparse_layer.weight.data = module.weight.data
                if module.bias is not None: sparse_layer.bias.data = module.bias.data
                
                # Update parent
                parts = name.split('.')
                parent = model
                for part in parts[:-1]: parent = getattr(parent, part)
                setattr(parent, parts[-1], sparse_layer)

# ============================================================================
# TRITON KERNELS - INDEXED SPARSE ADAMW (from V3)
# ============================================================================

@triton.jit
def indexed_sparse_adamw_kernel(
    param_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
    indices_ptr, n_indices,
    lr: tl.constexpr, beta1: tl.constexpr, beta2: tl.constexpr, eps: tl.constexpr, weight_decay: tl.constexpr,
    bias_correction1_val, bias_correction2_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_valid = offsets < n_indices
    
    idx = tl.load(indices_ptr + offsets, mask=mask_valid, other=0)
    
    param = tl.load(param_ptr + idx, mask=mask_valid, other=0.0)
    grad = tl.load(grad_ptr + idx, mask=mask_valid, other=0.0)
    exp_avg = tl.load(exp_avg_ptr + idx, mask=mask_valid, other=0.0)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + idx, mask=mask_valid, other=0.0)
    
    # Update
    decay_factor = 1.0 - lr * weight_decay
    param_decayed = param * decay_factor
    
    exp_avg_new = beta1 * exp_avg + (1.0 - beta1) * grad
    grad_squared = grad * grad
    exp_avg_sq_new = beta2 * exp_avg_sq + (1.0 - beta2) * grad_squared
    
    exp_avg_corrected = exp_avg_new / bias_correction1_val
    exp_avg_sq_corrected = exp_avg_sq_new / bias_correction2_val
    
    denom = tl.sqrt(exp_avg_sq_corrected) + eps
    step_size = lr / denom
    param_new = param_decayed - step_size * exp_avg_corrected
    
    tl.store(param_ptr + idx, param_new, mask=mask_valid)
    tl.store(exp_avg_ptr + idx, exp_avg_new, mask=mask_valid)
    tl.store(exp_avg_sq_ptr + idx, exp_avg_sq_new, mask=mask_valid)

def triton_indexed_sparse_adamw_step(
    param, grad, nonzero_indices, exp_avg, exp_avg_sq,
    lr, beta1, beta2, eps, weight_decay, step, block_size=128
):
    n_indices = nonzero_indices.shape[0]
    if n_indices == 0: return
    
    bias_correction1 = 1.0 - (beta1 ** step)
    bias_correction2 = 1.0 - (beta2 ** step)
    
    param_flat = param.flatten()
    grad_flat = grad.flatten()
    exp_avg_flat = exp_avg.flatten()
    exp_avg_sq_flat = exp_avg_sq.flatten()
    
    grid = (triton.cdiv(n_indices, block_size),)
    indexed_sparse_adamw_kernel[grid](
        param_flat, grad_flat, exp_avg_flat, exp_avg_sq_flat,
        nonzero_indices, n_indices,
        lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay,
        bias_correction1_val=bias_correction1,
        bias_correction2_val=bias_correction2,
        BLOCK_SIZE=block_size,
    )

# ============================================================================
# SPARSE MASK MANAGER
# ============================================================================

class SparseMaskManager:
    """Manages loading and applying sparse masks to model parameters."""
    
    def __init__(self, mask_path, device='cuda'):
        print(f"Loading masks from {mask_path}...")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        loaded = torch.load(mask_path, map_location='cpu')
        raw_masks = loaded['masks'] if isinstance(loaded, dict) and 'masks' in loaded else loaded
        
        self.masks = {}
        self.nonzero_indices = {}
        self.device = device
        
        print(f"Processing {len(raw_masks)} masks...")
        for name, mask in raw_masks.items():
            mask = mask.to(device)
            self.masks[name] = mask
            
            # Pre-compute indices for SparseAdamW
            flat_mask = mask.flatten()
            indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1).contiguous()
            self.nonzero_indices[name] = indices
            
        self._print_mask_summary()
    
    def _print_mask_summary(self):
        total_params = sum(m.numel() for m in self.masks.values())
        total_active = sum((m != 0).sum().item() for m in self.masks.values())
        total_sparsity = 1.0 - (total_active / total_params) if total_params > 0 else 0
        print(f"  Total parameters covered: {total_params:,}")
        print(f"  Total active parameters: {total_active:,}")
        print(f"  Overall sparsity: {total_sparsity*100:.2f}%")

    def get_mask(self, param_name):
        """Get mask with fuzzy matching."""
        for key in [param_name, param_name.replace('.', '_'), param_name.replace('_', '.')]:
            if key in self.masks: return self.masks[key]
        return None

    def get_nonzero_indices(self, param_name):
        """Get pre-computed indices with fuzzy matching."""
        for key in [param_name, param_name.replace('.', '_'), param_name.replace('_', '.')]:
            if key in self.nonzero_indices: return self.nonzero_indices[key]
        return None

    def has_mask(self, param_name):
        return self.get_mask(param_name) is not None

# ============================================================================
# SPARSE ADAMW OPTIMIZER
# ============================================================================

class SparseAdamW(torch.optim.Optimizer):
    def __init__(self, named_params, mask_manager, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, block_size=128):
        params = []
        self.param_to_name = {}
        for name, p in named_params:
            params.append(p)
            self.param_to_name[id(p)] = name
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.mask_manager = mask_manager
        self.block_size = block_size
        
        # Pre-init states
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                name = self.param_to_name.get(id(p))
                indices = self.mask_manager.get_nonzero_indices(name) if name else None
                
                if indices is not None:
                    state = self.state[p]
                    state['step'] += 1
                    triton_indexed_sparse_adamw_step(
                        p, p.grad, indices, state['exp_avg'], state['exp_avg_sq'],
                        group['lr'], group['betas'][0], group['betas'][1], group['eps'], group['weight_decay'],
                        state['step'], self.block_size
                    )
                else:
                    self._dense_step(p, group)
        return loss

    def _dense_step(self, p, group):
        grad = p.grad
        state = self.state[p]
        state['step'] += 1
        beta1, beta2 = group['betas']
        p.mul_(1 - group['lr'] * group['weight_decay'])
        state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
        state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bc1 = 1 - beta1 ** state['step']
        bc2 = 1 - beta2 ** state['step']
        step_size = group['lr'] / bc1
        denom = (state['exp_avg_sq'].sqrt() / bc2**0.5).add_(group['eps'])
        p.addcdiv_(state['exp_avg'], denom, value=-step_size)

# ============================================================================
# DATASET & COLLATOR
# ============================================================================

def load_dpo_dataset(subset_size=None):
    raw_ds = load_dataset(DATASET_NAME, split="train")
    def normalize_record(rec):
        def msg_to_text(x):
            if isinstance(x, str): return x
            if isinstance(x, dict): return x.get("value", "")
            if isinstance(x, list): return "\n".join(m.get("value", "") for m in x if isinstance(m, dict))
            return str(x)
        return {"prompt": msg_to_text(rec.get("prompt", "")).strip(),
                "chosen": msg_to_text(rec.get("chosen", "")).strip(),
                "rejected": msg_to_text(rec.get("rejected", "")).strip()}
    norm_ds = raw_ds.map(normalize_record, remove_columns=raw_ds.column_names)
    if subset_size is not None: norm_ds = norm_ds.select(range(min(subset_size, len(norm_ds))))
    return norm_ds

def dpo_collator_fn(examples: List[Dict[str, Any]], tokenizer) -> Dict[str, torch.Tensor]:
    prompts = [ex["prompt"] for ex in examples]
    chosens = [ex["chosen"] for ex in examples]
    rejects = [ex["rejected"] for ex in examples]
    model_inputs = tokenizer(prompts, max_length=512, truncation=True, padding=True, return_tensors="pt")
    chosen_inputs = tokenizer(chosens, max_length=1024, truncation=True, padding=True, return_tensors="pt")
    rejected_inputs = tokenizer(rejects, max_length=1024, truncation=True, padding=True, return_tensors="pt")
    return {"prompt_input_ids": model_inputs["input_ids"], "prompt_attention_mask": model_inputs["attention_mask"],
            "chosen_input_ids": chosen_inputs["input_ids"], "chosen_attention_mask": chosen_inputs["attention_mask"],
            "rejected_input_ids": rejected_inputs["input_ids"], "rejected_attention_mask": rejected_inputs["attention_mask"]}

# ============================================================================
# MAIN TRAINING
# ============================================================================

def train(
    model_name=MODEL_NAME_DEFAULT, checkpoint_path=None, mask_path=MASK_PATH,
    n_steps=5, batch_size=1, learning_rate=5e-5, subset_size=10, run_name=None,
    mlp_only=True, block_size=BLOCK_SIZE_BSR, optimizer_type="sgd", save_model=False,
):
    if checkpoint_path is None or checkpoint_path.lower() == "none": checkpoint_path = model_name
    if run_name is None: run_name = f"sparse_dpo_v4_{optimizer_type}_{sanitize_model_name(model_name)}"
    run_dir = os.path.join("results", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'='*60}\nSPARSE DPO V4 - FULLY INTEGRATED MASTER\n{'='*60}")
    print(f"Optimizer: {optimizer_type}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    dpo_dataset = load_dpo_dataset(subset_size=subset_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None
    )
    if hasattr(model, "to") and device.type == "cuda": model.to(device)
    model.config.use_cache = False
    
    # Integrated Mask Manager & Layer Replacement
    mask_manager = SparseMaskManager(mask_path, device=device)
    mask_dict = {n: mask_manager.get_mask(n) for n, _ in model.named_parameters() 
                 if ('mlp' in n.lower() or not mlp_only) and 'weight' in n and mask_manager.has_mask(n)}
                 
    print(f"Injecting Sparse MLP BSR backward for {len(mask_dict)} layers...")
    replace_linear_modules(model, mask_dict, block_size=block_size)
    
    # Optimizer selection
    print(f"Initializing {optimizer_type}...")
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_type == "sparse_adamw":
        optimizer = SparseAdamW(list(model.named_parameters()), mask_manager, lr=learning_rate, block_size=BLOCK_SIZE_KERNELS)
        
    dpo_config = DPOConfig(output_dir=os.path.join(run_dir, "checkpoints"), per_device_train_batch_size=batch_size,
                          learning_rate=learning_rate, max_steps=n_steps, logging_steps=1, report_to="none",
                          gradient_checkpointing=True, remove_unused_columns=False, bf16=True)
    
    trainer = DPOTrainer(model=model, args=dpo_config, train_dataset=dpo_dataset, 
                        data_collator=lambda x: dpo_collator_fn(x, tokenizer), optimizers=(optimizer, None))
    
    trainer.train()
    if save_model: model.save_pretrained(os.path.join(run_dir, "final_model"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME_DEFAULT)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mask", type=str, default=MASK_PATH)
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw", "sparse_adamw"], default="sgd")
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()
    train(**vars(args))