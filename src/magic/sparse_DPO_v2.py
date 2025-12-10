#!/usr/bin/env python3
"""
Triton-Accelerated Sparse DPO Training - OPTIMIZED VERSION

Key Optimizations:
1. Pre-initialized optimizer states (eliminates lazy init overhead)
2. Reduced block size from 32 to 16 (better granularity)
3. Pre-computed non-zero indices for true sparse operations
4. Indexed sparse kernel (only processes ~5% of elements vs 100%)
5. Gradient masking with indexed updates (true sparse computation)

Run: python src/sandbox/Triton_DPO_training_optimized.py \
  --checkpoint checkpoints_gemma3_dpo/checkpoint-100 \
  --mask masks/top_10.0pct_momentum_w25_step25.pt \
  --n_steps 10
"""

import os
import json
import time
import torch
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import wandb
import argparse
from typing import List, Dict, Any
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

def sanitize_model_name(model_name: str) -> str:
    """
    Convert HuggingFace model name to filesystem-safe string.
    
    Examples:
        "google/gemma-3-270m-it" -> "google_gemma_3_270m_it"
        "meta-llama/Llama-3.1-8B" -> "meta_llama_llama_3_1_8b"
    """
    # Replace "/" with "_", replace "-" with "_", convert to lowercase
    sanitized = model_name.replace("/", "_").replace("-", "_").lower()
    # Remove any remaining special characters that might cause issues
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
    # Collapse multiple underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


WANDB_PROJECT = "rl-casino-triton"
MODEL_NAME_DEFAULT = "google/gemma-3-270m-it"
DATASET_NAME = "qihoo360/Light-R1-DPOData"
SUBSET_SIZE = 10
MASK_PATH = "masks/top_10.0pct_momentum_w25_step25.pt"
BLOCK_SIZE = 16  # OPTIMIZATION: Reduced from 32 for better granularity


# ============================================================================
# TRITON KERNELS
# ============================================================================

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
    step: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    OPTIMIZED: Indexed sparse AdamW that only processes non-zero mask elements.
    
    KEY IMPROVEMENT: With 94% sparsity, this processes only ~6% of elements
    instead of 100%, directly addressing the memory bandwidth bottleneck.
    
    Algorithm:
    1. Each thread block processes BLOCK_SIZE non-zero indices
    2. Load parameters/gradients ONLY at these indices (gather operation)
    3. Perform AdamW update
    4. Store back ONLY at these indices (scatter operation)
    
    Expected speedup: ~15-20x for layers with 94% sparsity
    """
    pid = tl.program_id(0)
    
    # Calculate which non-zero indices this block handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_valid = offsets < n_indices
    
    # Load the actual flattened indices we need to update
    idx = tl.load(indices_ptr + offsets, mask=mask_valid, other=0)
    
    # OPTIMIZATION: Gather operation - only load values at non-zero indices
    param = tl.load(param_ptr + idx, mask=mask_valid, other=0.0)
    grad = tl.load(grad_ptr + idx, mask=mask_valid, other=0.0)
    exp_avg = tl.load(exp_avg_ptr + idx, mask=mask_valid, other=0.0)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + idx, mask=mask_valid, other=0.0)
    
    # Standard AdamW update computations
    # Weight decay (applied to parameters)
    decay_factor = 1.0 - lr * weight_decay
    param_decayed = param * decay_factor
    
    # Update biased first moment estimate (momentum)
    beta1_complement = 1.0 - beta1
    exp_avg_new = beta1 * exp_avg + beta1_complement * grad
    
    # Update biased second moment estimate (RMSprop)
    beta2_complement = 1.0 - beta2
    grad_squared = grad * grad
    exp_avg_sq_new = beta2 * exp_avg_sq + beta2_complement * grad_squared
    
    # Bias correction
    bias_correction1 = 1.0 - (beta1 ** step)
    bias_correction2 = 1.0 - (beta2 ** step)
    
    exp_avg_corrected = exp_avg_new / bias_correction1
    exp_avg_sq_corrected = exp_avg_sq_new / bias_correction2
    
    # Compute parameter update
    denom = tl.sqrt(exp_avg_sq_corrected) + eps
    step_size = lr / denom
    param_new = param_decayed - step_size * exp_avg_corrected
    
    # OPTIMIZATION: Scatter operation - only store at non-zero indices
    tl.store(param_ptr + idx, param_new, mask=mask_valid)
    tl.store(exp_avg_ptr + idx, exp_avg_new, mask=mask_valid)
    tl.store(exp_avg_sq_ptr + idx, exp_avg_sq_new, mask=mask_valid)


def triton_indexed_sparse_adamw_step(
    param, grad, nonzero_indices, exp_avg, exp_avg_sq,
    lr, beta1, beta2, eps, weight_decay, step, block_size=16
):
    """
    Wrapper for indexed sparse AdamW kernel.
    
    IMPORTANT: This function flattens all tensors for indexed access,
    then uses gather/scatter operations via the kernel.
    
    Args:
        param: Parameter tensor (any shape)
        grad: Gradient tensor (same shape as param)
        nonzero_indices: 1D tensor of flattened indices where mask is non-zero
        exp_avg, exp_avg_sq: Optimizer state tensors
        lr, beta1, beta2, eps, weight_decay: AdamW hyperparameters
        step: Current optimizer step (for bias correction)
        block_size: Triton kernel block size
    """
    n_indices = nonzero_indices.shape[0]
    
    if n_indices == 0:
        # Edge case: completely masked layer (shouldn't happen but be safe)
        return
    
    # Flatten all tensors for indexed access
    # Note: .flatten() creates a view, not a copy (efficient)
    param_flat = param.flatten()
    grad_flat = grad.flatten()
    exp_avg_flat = exp_avg.flatten()
    exp_avg_sq_flat = exp_avg_sq.flatten()
    
    # Calculate grid size based on number of non-zero indices
    grid = (triton.cdiv(n_indices, block_size),)
    
    # Launch kernel
    indexed_sparse_adamw_kernel[grid](
        param_flat, grad_flat, exp_avg_flat, exp_avg_sq_flat,
        nonzero_indices,
        n_indices,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        step=step,
        BLOCK_SIZE=block_size,
    )
    
    # Reshape back to original shape
    # Note: Since we modified in-place via flatten(), we need to update .data
    param.data = param_flat.reshape(param.shape)
    exp_avg.data = exp_avg_flat.reshape(exp_avg.shape)
    exp_avg_sq.data = exp_avg_sq_flat.reshape(exp_avg_sq.shape)


# ============================================================================
# SPARSE MASK MANAGER
# ============================================================================

class SparseMaskManager:
    """
    Manages loading and applying sparse masks to model parameters.
    
    OPTIMIZATION: Pre-computes non-zero indices for each mask to enable
    true sparse operations (gather/scatter) instead of dense masking.
    """
    
    def __init__(self, mask_path, device='cuda'):
        print(f"Loading masks from {mask_path}...")
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        self.masks = torch.load(mask_path, map_location='cpu')
        self.device = device
        
        if isinstance(self.masks, dict) and 'masks' in self.masks:
            self.masks = self.masks['masks']
        
        print(f"\nDEBUG - First 3 mask keys:")
        for i, key in enumerate(list(self.masks.keys())[:3]):
            print(f"  {key}")
        
        # OPTIMIZATION: Pre-compute non-zero indices for each mask
        print("\nPre-computing non-zero indices for sparse operations...")
        self.nonzero_indices = {}
        self.mask_stats = {}
        
        for name, mask in self.masks.items():
            # Move mask to device FIRST
            mask = mask.to(device, non_blocking=True)
            self.masks[name] = mask
            
            # Compute statistics
            sparsity = (mask == 0.0).sum().item() / mask.numel()  # % of zeros
            nonzero_count = (mask != 0.0).sum().item()
            
            self.mask_stats[name] = {
                'shape': tuple(mask.shape),
                'sparsity': sparsity,  # % zeros
                'nonzero': nonzero_count,  # count of non-zeros
            }
            
            # CRITICAL FIX: Pre-compute flattened indices of non-zero elements ON GPU
            # The mask is already on device, so nonzero() will return GPU indices
            if nonzero_count > 0:
                # nonzero() returns indices; we flatten and store as 1D tensor
                flat_mask = mask.flatten()
                indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1).contiguous()
                # Ensure indices are on the correct device (should already be, but verify)
                self.nonzero_indices[name] = indices.to(device, non_blocking=True)
            else:
                # Edge case: completely sparse mask
                self.nonzero_indices[name] = torch.empty(0, dtype=torch.long, device=device)
        
        print(f"✓ Loaded {len(self.masks)} masks with pre-computed indices")
        self._print_mask_summary()
    
    def _print_mask_summary(self):
        """Print summary statistics of loaded masks."""
        total_params = sum(stats['shape'][0] * stats['shape'][1] 
                          for stats in self.mask_stats.values() 
                          if len(stats['shape']) == 2)
        total_active = sum(stats['nonzero'] 
                          for stats in self.mask_stats.values())
        
        # Calculate true sparsity (% of zeros)
        total_sparsity = 1.0 - (total_active / total_params) if total_params > 0 else 0
        
        print(f"  Total parameters covered: {total_params:,}")
        print(f"  Total active (non-zero) parameters: {total_active:,}")
        print(f"  Overall sparsity (% zeros): {total_sparsity*100:.2f}%")
        print(f"  Active parameters (% non-zero): {(1-total_sparsity)*100:.2f}%")
    
    def get_mask(self, param_name):
        """Get mask for a parameter - try direct match first, then with conversions."""
        if param_name in self.masks:
            return self.masks[param_name]
        
        mask_key = param_name.replace('.', '_')
        if mask_key in self.masks:
            return self.masks[mask_key]
        
        mask_key_dots = param_name.replace('_', '.')
        if mask_key_dots in self.masks:
            return self.masks[mask_key_dots]
        
        return None
    
    def get_nonzero_indices(self, param_name):
        """Get pre-computed non-zero indices for a parameter."""
        if param_name in self.nonzero_indices:
            return self.nonzero_indices[param_name]
        
        idx_key = param_name.replace('.', '_')
        if idx_key in self.nonzero_indices:
            return self.nonzero_indices[idx_key]
        
        idx_key_dots = param_name.replace('_', '.')
        if idx_key_dots in self.nonzero_indices:
            return self.nonzero_indices[idx_key_dots]
        
        return None
    
    def is_mlp_layer(self, param_name):
        """Check if parameter belongs to an MLP layer."""
        return 'mlp' in param_name.lower()
    
    def has_mask(self, param_name):
        """Check if a mask exists for this parameter."""
        return (
            param_name in self.masks or
            param_name.replace('.', '_') in self.masks or
            param_name.replace('_', '.') in self.masks
        )


# ============================================================================
# SPARSE ADAMW OPTIMIZER WITH TRITON KERNELS
# ============================================================================

class SparseAdamW(torch.optim.Optimizer):
    """
    Custom AdamW optimizer with Triton-accelerated sparse updates.
    
    OPTIMIZATIONS:
    1. Pre-initialized optimizer states (no lazy init overhead)
    2. Indexed sparse kernel using gather/scatter operations
    3. Only processes non-zero mask elements (~5% with 94% sparsity)
    """
    
    def __init__(
        self,
        named_params,
        mask_manager: SparseMaskManager,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        block_size=16,
        mlp_only=True,
    ):
        self.param_to_name = {}
        params = []
        for name, param in named_params:
            params.append(param)
            self.param_to_name[id(param)] = name
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        self.mask_manager = mask_manager
        self.block_size = block_size
        self.mlp_only = mlp_only
        
        self.stats = {
            'sparse_steps': 0,
            'dense_steps': 0,
            'nan_warnings': [],
        }
        
        # OPTIMIZATION: Pre-initialize all optimizer states upfront
        # This eliminates the lazy initialization overhead during training
        print(f"\nPre-initializing optimizer states for all parameters...")
        init_start = time.time()
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    state['step'] = 0
                    # Initialize momentum buffers on same device as parameters
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        
        init_time = time.time() - init_start
        print(f"✓ Optimizer states pre-initialized in {init_time:.2f}s")
        print(f"✓ SparseAdamW optimizer ready")
        print(f"  MLP-only: {mlp_only}")
        print(f"  Block size: {block_size}")
        print(f"  Using indexed sparse kernels: TRUE")
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step using Triton kernels."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_name = self.param_to_name.get(id(p), None)
                
                if param_name is None:
                    self._dense_step(p, group)
                    continue
                
                should_use_sparse = (
                    self.mask_manager.has_mask(param_name) and
                    len(p.shape) == 2 and
                    (not self.mlp_only or self.mask_manager.is_mlp_layer(param_name))
                )
                
                if should_use_sparse:
                    self._sparse_step(p, param_name, group)
                else:
                    self._dense_step(p, group)
        
        return loss
    
    def _sparse_step(self, param, param_name, group):
        """
        OPTIMIZED: Sparse update using indexed Triton kernel.
        
        Key improvement: Uses pre-computed non-zero indices to only
        process active weights (gather/scatter operations).
        
        NOTE: Timing removed to avoid CPU-GPU synchronization overhead
        """
        grad = param.grad
        mask = self.mask_manager.get_mask(param_name)
        nonzero_indices = self.mask_manager.get_nonzero_indices(param_name)
        
        if mask is None or nonzero_indices is None:
            print(f"WARNING: Missing mask or indices for {param_name}, falling back to dense")
            self._dense_step(param, group)
            return
        
        if mask.shape != param.shape:
            print(f"WARNING: Shape mismatch for {param_name}")
            print(f"  Param: {param.shape}, Mask: {mask.shape}")
            print(f"  Falling back to dense update")
            self._dense_step(param, group)
            return
        
        state = self.state[param]
        
        # OPTIMIZATION: No more lazy init check - states pre-initialized
        state['step'] += 1
        
        # CRITICAL: Apply mask to gradient first
        # This ensures gradients at masked locations are zero before sparse update
        grad.mul_(mask)
        
        try:
            # Use optimized indexed sparse kernel
            triton_indexed_sparse_adamw_step(
                param=param,
                grad=grad,
                nonzero_indices=nonzero_indices,
                exp_avg=state['exp_avg'],
                exp_avg_sq=state['exp_avg_sq'],
                lr=group['lr'],
                beta1=group['betas'][0],
                beta2=group['betas'][1],
                eps=group['eps'],
                weight_decay=group['weight_decay'],
                step=state['step'],
                block_size=self.block_size,
            )
        except Exception as e:
            print(f"ERROR in indexed Triton kernel for {param_name}: {e}")
            print(f"  Falling back to dense update")
            self._dense_step(param, group)
            return
        
        self.stats['sparse_steps'] += 1
    
    def _dense_step(self, param, group):
        """
        Standard dense AdamW update (fallback for non-masked params).
        
        OPTIMIZATION: No more lazy init check - states pre-initialized.
        NOTE: Timing removed to avoid CPU-GPU synchronization overhead
        """
        grad = param.grad
        state = self.state[param]
        
        # OPTIMIZATION: No lazy init needed
        state['step'] += 1
        
        beta1, beta2 = group['betas']
        
        # AdamW weight decay
        param.mul_(1 - group['lr'] * group['weight_decay'])
        
        # Update biased first and second moment estimates
        state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
        state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # Bias correction
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        
        # Compute step
        step_size = group['lr'] / bias_correction1
        denom = (state['exp_avg_sq'].sqrt() / bias_correction2 ** 0.5).add_(group['eps'])
        param.addcdiv_(state['exp_avg'], denom, value=-step_size)
        
        self.stats['dense_steps'] += 1
    
    def check_for_nans(self, model):
        """Check all parameters for NaN/Inf after training completes."""
        nan_params = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                nan_count = torch.isnan(param).sum().item()
                inf_count = torch.isinf(param).sum().item()
                nan_params.append((name, nan_count, inf_count))
        
        if nan_params:
            print(f"\n{'='*60}")
            print("WARNING: NaN/Inf DETECTED IN PARAMETERS")
            print(f"{'='*60}")
            for name, nan_count, inf_count in nan_params:
                print(f"  {name}: NaN={nan_count}, Inf={inf_count}")
            print(f"{'='*60}\n")
            self.stats['nan_warnings'] = nan_params
        else:
            print("\n✓ No NaN/Inf detected in parameters\n")
    
    def print_stats(self):
        """Print optimizer statistics."""
        total = self.stats['sparse_steps'] + self.stats['dense_steps']
        sparse = self.stats['sparse_steps']
        dense = self.stats['dense_steps']
        
        print(f"\n{'='*60}")
        print(f"SPARSE ADAMW OPTIMIZER STATISTICS")
        print(f"{'='*60}")
        print(f"Total steps:      {total:,}")
        print(f"Sparse steps:     {sparse:,} ({sparse/total*100 if total > 0 else 0:.1f}%)")
        print(f"Dense steps:      {dense:,} ({dense/total*100 if total > 0 else 0:.1f}%)")
        print(f"\nNOTE: Per-step timing removed to avoid GPU synchronization overhead.")
        print(f"      Use PyTorch profiler or nsys for accurate performance measurement.")
        print(f"{'='*60}\n")


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dpo_dataset(subset_size=None):
    """Load and normalize DPO dataset from HuggingFace."""
    print(f"Loading dataset: {DATASET_NAME}")
    raw_ds = load_dataset(DATASET_NAME, split="train")
    
    def msg_to_text(x):
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return x.get("value", "")
        if isinstance(x, list):
            return "\n".join(m.get("value", "") for m in x if isinstance(m, dict))
        return str(x)

    def normalize_record(rec):
        prompt_raw   = rec.get("prompt", "")
        chosen_raw   = rec.get("chosen", "")
        rejected_raw = rec.get("rejected", "")

        if isinstance(prompt_raw, list):
            prompt_text = "\n".join(
                m.get("value","") for m in prompt_raw
                if isinstance(m, dict) and m.get("from","").lower() != "assistant"
            ).strip()
        else:
            prompt_text = msg_to_text(prompt_raw).strip()

        chosen_text   = msg_to_text(chosen_raw).strip()
        rejected_text = msg_to_text(rejected_raw).strip()

        return {"prompt": prompt_text, "chosen": chosen_text, "rejected": rejected_text}

    norm_ds = raw_ds.map(normalize_record, remove_columns=raw_ds.column_names)
    
    if subset_size is not None:
        norm_ds = norm_ds.select(range(min(subset_size, len(norm_ds))))
    
    print(f"✓ Loaded {len(norm_ds)} examples")
    return norm_ds


def dpo_collator_fn(examples: List[Dict[str, Any]], tokenizer) -> Dict[str, torch.Tensor]:
    """Data collator for DPO training."""
    if "prompt_input_ids" in examples[0]:
        def pad_stack(key):
            seqs = [torch.tensor(ex[key]) if not torch.is_tensor(ex[key]) else ex[key] for ex in examples]
            lens = [s.size(-1) for s in seqs]
            maxlen = max(lens)
            out = torch.full((len(seqs), maxlen), fill_value=0, dtype=torch.long)
            mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, : s.size(-1)] = s.to(torch.long)
                mask[i, : s.size(-1)] = 1
            return out, mask

        p_ids, p_mask = pad_stack("prompt_input_ids")
        c_ids, c_mask = pad_stack("chosen_input_ids")
        r_ids, r_mask = pad_stack("rejected_input_ids")
        return {
            "prompt_input_ids": p_ids, "prompt_attention_mask": p_mask,
            "chosen_input_ids": c_ids, "chosen_attention_mask": c_mask,
            "rejected_input_ids": r_ids, "rejected_attention_mask": r_mask,
        }

    prompts  = [ex.get("prompt", "")   for ex in examples]
    chosens  = [ex.get("chosen", "")   for ex in examples]
    rejects  = [ex.get("rejected", "") for ex in examples]

    enc_prompt = [tokenizer(p, truncation=True, max_length=512,  return_tensors="pt") for p in prompts]
    enc_chosen = [tokenizer(c, truncation=True, max_length=1024, return_tensors="pt") for c in chosens]
    enc_reject = [tokenizer(r, truncation=True, max_length=1024, return_tensors="pt") for r in rejects]

    batch_prompt = tokenizer.pad(enc_prompt, padding=True, return_tensors="pt", pad_to_multiple_of=8)
    batch_chosen = tokenizer.pad(enc_chosen, padding=True, return_tensors="pt", pad_to_multiple_of=8)
    batch_reject = tokenizer.pad(enc_reject, padding=True, return_tensors="pt", pad_to_multiple_of=8)

    for k in ("input_ids", "attention_mask"):
        batch_prompt[k] = batch_prompt[k].to(torch.long)
        batch_chosen[k] = batch_chosen[k].to(torch.long)
        batch_reject[k] = batch_reject[k].to(torch.long)

    return {
        "prompt_input_ids":        batch_prompt["input_ids"],
        "prompt_attention_mask":   batch_prompt["attention_mask"],
        "chosen_input_ids":        batch_chosen["input_ids"],
        "chosen_attention_mask":   batch_chosen["attention_mask"],
        "rejected_input_ids":      batch_reject["input_ids"],
        "rejected_attention_mask": batch_reject["attention_mask"],
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def make_run_dir(base_dir="results", run_name=None):
    """Create a timestamped run directory."""
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(
    model_name=MODEL_NAME_DEFAULT,
    checkpoint_path=None,
    mask_path=MASK_PATH,
    n_steps=5,
    batch_size=1, 
    learning_rate=5e-5,
    subset_size=10,
    run_name=None,
    mlp_only=True,
    block_size=BLOCK_SIZE,
    model=None,
    dpo_dataset=None,
    tokenizer=None,
):
    """Train with optimized Triton-accelerated sparse training."""
    
    # Determine model source - handle "None" string from argparse
    if checkpoint_path is None or (isinstance(checkpoint_path, str) and checkpoint_path.lower() == "none"):
        checkpoint_path = model_name
        print(f"No checkpoint specified, using base model from HF: {checkpoint_path}")
    else:
        print(f"Using checkpoint path: {checkpoint_path}")
    
    # Derive run_name from model_name if not provided
    if run_name is None:
        model_sanitized = sanitize_model_name(model_name)
        run_name = f"triton_sparse_dpo_{model_sanitized}_optimized"
    
    run_dir = make_run_dir(base_dir="results", run_name=run_name)
    print(f"\n{'='*60}")
    print(f"TRITON-ACCELERATED SPARSE DPO TRAINING (OPTIMIZED)")
    print(f"{'='*60}")
    print(f"Run directory: {run_dir}")
    print(f"Model source: {checkpoint_path}")
    print(f"Mask path: {mask_path}")
    print(f"Steps: {n_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Block size: {block_size}")
    print(f"{'='*60}\n")

    # Load or reuse tokenizer
    if tokenizer is None:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print("✓ Tokenizer loaded")
    else:
        print("✓ Using pre-loaded tokenizer")

    # Load or reuse dataset
    if dpo_dataset is None:
        print("Loading dataset...")
        dpo_dataset = load_dpo_dataset(subset_size=subset_size)
    else:
        print(f"✓ Using pre-loaded dataset ({len(dpo_dataset)} examples)")
    
    def collator(examples):
        return dpo_collator_fn(examples, tokenizer)

    # Load or reuse model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model is None:
        print(f"Loading model from: {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        if not torch.cuda.is_available() or model.device.type == 'cpu':
            model.to(device)
        
        print(f"✓ Model loaded on {device} with dtype: {model.dtype}")
    else:
        print(f"✓ Using pre-loaded model on {model.device} with dtype: {model.dtype}")
    
    model.config.use_cache = False

    # Clear cache before loading masks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load masks with pre-computed indices
    mask_manager = SparseMaskManager(mask_path, device=device)
    
    # Clear cache after mask loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # DEBUG: Check parameter name matching
    print("\n" + "="*60)
    print("PARAMETER NAME MATCHING DEBUG")
    print("="*60)
    model_mlp_params = [(name, p.shape) for name, p in model.named_parameters() 
                        if len(p.shape) == 2 and 'mlp' in name.lower()]
    print(f"\nModel MLP parameters (first 5):")
    for name, shape in model_mlp_params[:5]:
        has_it = mask_manager.has_mask(name)
        mask = mask_manager.get_mask(name) if has_it else None
        indices = mask_manager.get_nonzero_indices(name) if has_it else None
        mask_shape = mask.shape if mask is not None else "N/A"
        n_indices = len(indices) if indices is not None else 0
        match = "✓ MATCH" if (mask is not None and mask.shape == shape) else "✗ MISMATCH"
        print(f"  {name}")
        print(f"    Param shape: {shape}, Mask shape: {mask_shape}")
        print(f"    Non-zero indices: {n_indices} - {match}")

    print(f"\nTotal model MLP params: {len(model_mlp_params)}")
    matches = sum(1 for name, _ in model_mlp_params if mask_manager.has_mask(name))
    print(f"Matching masks: {matches}/{len(model_mlp_params)}")
    print("="*60 + "\n")

    # Set up DPOConfig
    dpo_config = DPOConfig(
        output_dir=os.path.join(run_dir, "checkpoints"),
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=n_steps,
        logging_steps=1,
        report_to="wandb",
        remove_unused_columns=False,
        run_name=run_name,
        gradient_accumulation_steps=1,
        beta=0.1,
        max_length=1024,
        max_prompt_length=512,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
    )

    # Initialize wandb
    wandb.init(
        project=WANDB_PROJECT, 
        name=run_name, 
        config={
            "model_name": model_name,
            "checkpoint": checkpoint_path,
            "mask_path": mask_path,
            "dataset": DATASET_NAME,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "subset_size": subset_size,
            "mlp_only": mlp_only,
            "block_size": block_size,
            "triton_enabled": True,
            "optimizer": "SparseAdamW_Indexed",
            "dtype": "bfloat16",
            "optimization": "pre_init_indexed_gather_scatter"
        }
    )

    # Snapshot initial params θ(0)
    base_state = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            base_state[name] = param.detach().float().cpu().clone()

    # Subnet logging callback
    class SubnetLoggingCallback(TrainerCallback):
        """Callback that logs subnet statistics to wandb."""
        
        def __init__(self, base_state, threshold=1e-3):
            self.base_state = base_state
            self.threshold = threshold

        def on_step_end(self, args, state, control, **kwargs):
            model = kwargs["model"]
            step = state.global_step

            layer_stats = {}

            with torch.no_grad():
                for name, param in model.named_parameters():
                    current = param.detach().float().cpu()
                    diff = current - self.base_state[name]

                    l2 = torch.norm(diff).item()
                    frac_big = (diff.abs() > self.threshold).float().mean().item()

                    layer_stats[name] = {
                        "l2_from_init": l2,
                        "frac_big_from_init": frac_big,
                    }

            # Aggregate summaries for wandb
            all_l2 = [v["l2_from_init"] for v in layer_stats.values()]
            all_frac = [v["frac_big_from_init"] for v in layer_stats.values()]
            mean_l2 = sum(all_l2) / len(all_l2)
            mean_frac = sum(all_frac) / len(all_frac)

            attn_l2 = []
            mlp_l2 = []
            for n, st in layer_stats.items():
                low = n.lower()
                if "attn" in low or "q_proj" in low or "k_proj" in low or "v_proj" in low or "o_proj" in low:
                    attn_l2.append(st["l2_from_init"])
                if "mlp" in low or "ffn" in low or "feed_forward" in low or "gate_proj" in low or "up_proj" in low or "down_proj" in low:
                    mlp_l2.append(st["l2_from_init"])

            wandb.log({
                "step": step,
                "subnet/mean_l2_from_init": mean_l2,
                "subnet/mean_frac_big_from_init": mean_frac,
                "subnet/attn_mean_l2": (sum(attn_l2)/len(attn_l2)) if attn_l2 else 0.0,
                "subnet/mlp_mean_l2": (sum(mlp_l2)/len(mlp_l2)) if mlp_l2 else 0.0,
            }, step=step)

            return control

    # Create optimized sparse optimizer
    sparse_optimizer = SparseAdamW(
        named_params=list(model.named_parameters()),
        mask_manager=mask_manager,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        block_size=block_size,
        mlp_only=mlp_only,
    )

    # Set up DPOTrainer - match baseline exactly
    print("\nInitializing DPOTrainer with optimized SparseAdamW...")
    dpo_trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dpo_dataset,
        eval_dataset=None,
        data_collator=collator,
    )
    
    # Set optimizer AFTER initialization
    print("Attaching optimized sparse optimizer to trainer...")
    dpo_trainer.optimizer = sparse_optimizer
    
    # Add subnet logging callback
    dpo_trainer.add_callback(SubnetLoggingCallback(base_state=base_state, threshold=1e-3))

    # Train
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")
    
    # Use CUDA events for accurate GPU timing (doesn't force synchronization)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Sync once before starting
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    
    training_start = time.time()
    dpo_trainer.train()
    
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()  # Sync once at the end
        training_time_gpu = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds
    
    training_time = time.time() - training_start
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Wall clock time: {training_time:.2f}s")
    if torch.cuda.is_available():
        print(f"GPU time: {training_time_gpu:.2f}s")
        print(f"Time per step (GPU): {training_time_gpu/n_steps:.2f}s")
    print(f"Time per step (wall): {training_time/n_steps:.2f}s")
    print(f"{'='*60}\n")
    
    # Check for NaNs after training
    sparse_optimizer.check_for_nans(model)
    
    # Print optimizer statistics
    sparse_optimizer.print_stats()
    
    print(f"Skipping final model save to conserve space.")
    
    # Save timing info
    timing_info = {
        'total_training_time': training_time,
        'time_per_step': training_time / n_steps,
        'n_steps': n_steps,
        'optimizer_stats': sparse_optimizer.stats
    }
    with open(os.path.join(run_dir, 'timing_info.json'), 'w') as f:
        json.dump(timing_info, f, indent=2)
    print(f"✓ Timing info saved to {run_dir}/timing_info.json")
    
    wandb.finish()
    print("\n✓ All done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimized Triton-accelerated sparse DPO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--model_name", type=str, default=MODEL_NAME_DEFAULT,
                       help=f"HuggingFace model name to load (default: {MODEL_NAME_DEFAULT})")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Path to model checkpoint (default: use base model from --model_name)")
    parser.add_argument("--mask", type=str, default=MASK_PATH, 
                       help=f"Path to sparse mask file (default: {MASK_PATH})")
    parser.add_argument("--n_steps", type=int, default=5, 
                       help="Number of training steps (default: 5)")
    parser.add_argument("--batch_size", type=int, default=1, 
                       help="Training batch size (default: 1)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                       help="Learning rate (default: 5e-5)")
    parser.add_argument("--subset_size", type=int, default=10, 
                       help="Dataset subset size (default: 10)")
    parser.add_argument("--run_name", type=str, default=None, 
                       help="Run name for wandb (default: auto-generated from model_name)")
    parser.add_argument("--mlp_only", action="store_true", default=False, 
                       help="Only apply sparse training to MLP layers (default: False - use all masked layers)")
    parser.add_argument("--block_size", type=int, default=BLOCK_SIZE, 
                       help=f"Triton kernel block size (default: {BLOCK_SIZE})")
    
    args = parser.parse_args()
    
    MODEL_NAME = args.model_name
    
    print(f"\n{'='*60}")
    print("INITIALIZING SHARED RESOURCES")
    print(f"{'='*60}\n")
    
    # Pre-load shared resources once
    print("Step 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✓ Tokenizer loaded and ready\n")
    
    print("Step 2: Loading dataset...")
    dataset = load_dpo_dataset(subset_size=args.subset_size)
    print("✓ Dataset loaded and ready\n")
    
    print("Step 3: Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # FIXED: Handle checkpoint argument properly
    if args.checkpoint is None or (isinstance(args.checkpoint, str) and args.checkpoint.lower() == "none"):
        model_path = MODEL_NAME
        print(f"Using base model: {MODEL_NAME}")
    else:
        model_path = args.checkpoint
        print(f"Using checkpoint: {args.checkpoint}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    if not torch.cuda.is_available() or model.device.type == 'cpu':
        model.to(device)
    
    print(f"✓ Model loaded on {device} with dtype: {model.dtype}\n")
    
    print(f"{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")
    
    # Now train with pre-loaded resources
    train(
        model_name=MODEL_NAME,
        run_name=args.run_name,
        checkpoint_path=args.checkpoint,
        mask_path=args.mask,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        subset_size=args.subset_size,
        mlp_only=args.mlp_only,
        block_size=args.block_size,
        model=model,
        dpo_dataset=dataset,
        tokenizer=tokenizer,
    )