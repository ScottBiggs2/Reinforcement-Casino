#!/usr/bin/env python3
"""
Triton-Accelerated Sparse DPO Training - FIXED VERSION

Fixes:
1. Unindented _dense_step method (was nested inside _sparse_step)
2. Changed to BF16 instead of FP16 for numerical stability
3. Removed per-step NaN checking to avoid synchronization overhead
4. Added summary NaN check at end instead

Run: python src/sandbox/Triton_DPO_training.py \
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

WANDB_PROJECT = "rl-casino-triton"
MODEL_NAME = "google/gemma-3-270m-it"
DATASET_NAME = "qihoo360/Light-R1-DPOData"
SUBSET_SIZE = 10
MASK_PATH = "masks/top_10.0pct_momentum_w25_step25.pt"
DEFAULT_CHECKPOINT = "google/gemma-3-270m-it"
BLOCK_SIZE = 32


# ============================================================================
# TRITON KERNELS
# ============================================================================

@triton.jit
def fused_sparse_adamw_kernel(
    # Pointers
    param_ptr,        # Input/Output: parameters
    grad_ptr,         # Input: gradients
    mask_ptr,         # Input: sparse mask
    exp_avg_ptr,      # Input/Output: first moment
    exp_avg_sq_ptr,   # Input/Output: second moment
    # Dimensions
    M, N,
    stride_pm, stride_pn,
    stride_gm, stride_gn,
    stride_mm, stride_mn,
    stride_m1m, stride_m1n,
    stride_m2m, stride_m2n,
    # Optimizer hyperparameters
    lr: tl.constexpr,
    beta1: tl.constexpr,
    beta2: tl.constexpr,
    eps: tl.constexpr,
    weight_decay: tl.constexpr,
    step: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused sparse AdamW optimizer kernel.
    
    Combines in a single kernel:
    1. Gradient masking (grad * mask)
    2. AdamW momentum updates (exp_avg, exp_avg_sq)
    3. Bias correction
    4. Parameter updates
    5. Weight decay
    """
    pid = tl.program_id(0)
    
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE)
    block_row = pid // num_blocks_n
    block_col = pid % num_blocks_n
    
    row_start = block_row * BLOCK_SIZE
    col_start = block_col * BLOCK_SIZE
    
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    
    row_offsets = row_offsets[:, None]
    col_offsets = col_offsets[None, :]
    
    p_offsets = row_offsets * stride_pm + col_offsets * stride_pn
    g_offsets = row_offsets * stride_gm + col_offsets * stride_gn
    m_offsets = row_offsets * stride_mm + col_offsets * stride_mn
    m1_offsets = row_offsets * stride_m1m + col_offsets * stride_m1n
    m2_offsets = row_offsets * stride_m2m + col_offsets * stride_m2n
    
    mask_valid = (row_offsets < M) & (col_offsets < N)
    
    # Load data
    param = tl.load(param_ptr + p_offsets, mask=mask_valid, other=0.0)
    grad = tl.load(grad_ptr + g_offsets, mask=mask_valid, other=0.0)
    sparse_mask = tl.load(mask_ptr + m_offsets, mask=mask_valid, other=0.0)
    exp_avg = tl.load(exp_avg_ptr + m1_offsets, mask=mask_valid, other=0.0)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + m2_offsets, mask=mask_valid, other=0.0)
    
    # Step 1: Apply sparse mask to gradient
    masked_grad = grad * sparse_mask
    
    # Step 2: AdamW weight decay (applied to all params, not just masked ones)
    decay_factor = 1.0 - lr * weight_decay
    param_decayed = param * decay_factor
    
    # Step 3: Update biased first moment estimate
    beta1_complement = 1.0 - beta1
    exp_avg_new = beta1 * exp_avg + beta1_complement * masked_grad
    
    # Step 4: Update biased second moment estimate
    beta2_complement = 1.0 - beta2
    grad_squared = masked_grad * masked_grad
    exp_avg_sq_new = beta2 * exp_avg_sq + beta2_complement * grad_squared
    
    # Step 5: Bias correction
    bias_correction1 = 1.0 - (beta1 ** step)
    bias_correction2 = 1.0 - (beta2 ** step)
    
    exp_avg_corrected = exp_avg_new / bias_correction1
    exp_avg_sq_corrected = exp_avg_sq_new / bias_correction2
    
    # Step 6: Compute parameter update
    denom = tl.sqrt(exp_avg_sq_corrected) + eps
    step_size = lr / denom
    param_new = param_decayed - step_size * exp_avg_corrected
    
    # Store results (in-place updates)
    tl.store(param_ptr + p_offsets, param_new, mask=mask_valid)
    tl.store(exp_avg_ptr + m1_offsets, exp_avg_new, mask=mask_valid)
    tl.store(exp_avg_sq_ptr + m2_offsets, exp_avg_sq_new, mask=mask_valid)


def triton_fused_sparse_adamw_step(
    param, grad, mask, exp_avg, exp_avg_sq,
    lr, beta1, beta2, eps, weight_decay, step, block_size=32
):
    """
    Fused sparse AdamW optimizer step using Triton kernel.
    Updates param, exp_avg, and exp_avg_sq in-place.
    """
    M, N = param.shape
    
    num_blocks_m = triton.cdiv(M, block_size)
    num_blocks_n = triton.cdiv(N, block_size)
    grid = (num_blocks_m * num_blocks_n,)
    
    fused_sparse_adamw_kernel[grid](
        param, grad, mask, exp_avg, exp_avg_sq,
        M, N,
        param.stride(0), param.stride(1),
        grad.stride(0), grad.stride(1),
        mask.stride(0), mask.stride(1),
        exp_avg.stride(0), exp_avg.stride(1),
        exp_avg_sq.stride(0), exp_avg_sq.stride(1),
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        step=step,
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
        
        self.masks = torch.load(mask_path, map_location='cpu')
        self.device = device
        
        if isinstance(self.masks, dict) and 'masks' in self.masks:
            self.masks = self.masks['masks']
        
        print(f"\nDEBUG - First 3 mask keys:")
        for i, key in enumerate(list(self.masks.keys())[:3]):
            print(f"  {key}")
        
        self.mask_stats = {}
        for name, mask in self.masks.items():
            self.masks[name] = mask.to(device)
            sparsity = (mask == 1.0).sum().item() / mask.numel()
            self.mask_stats[name] = {
                'shape': tuple(mask.shape),
                'sparsity': sparsity,
                'nonzero': (mask == 1.0).sum().item()
            }
        
        print(f"✓ Loaded {len(self.masks)} masks")
        self._print_mask_summary()
    
    def _print_mask_summary(self):
        """Print summary statistics of loaded masks."""
        total_params = sum(stats['shape'][0] * stats['shape'][1] 
                          for stats in self.mask_stats.values() 
                          if len(stats['shape']) == 2)
        total_active = sum(stats['nonzero'] 
                          for stats in self.mask_stats.values())
        
        print(f"  Total parameters covered: {total_params:,}")
        print(f"  Total active parameters: {total_active:,}")
        print(f"  Overall sparsity: {total_active/total_params*100:.2f}%")
    
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
    
    Uses fused Triton kernels for:
    - Gradient masking
    - Momentum updates
    - Parameter updates
    """
    
    def __init__(
        self,
        named_params,
        mask_manager: SparseMaskManager,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        block_size=32,
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
            'time_sparse': 0.0,
            'time_dense': 0.0,
            'nan_warnings': [],
        }
        
        print(f"✓ SparseAdamW optimizer initialized")
        print(f"  MLP-only: {mlp_only}")
        print(f"  Block size: {block_size}")
        print(f"  Using Triton fused kernels: TRUE")
    
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
        """Sparse update using Triton fused kernel."""
        start = time.time()
        
        grad = param.grad
        mask = self.mask_manager.get_mask(param_name)
        
        if mask.shape != param.shape:
            print(f"WARNING: Shape mismatch for {param_name}")
            print(f"  Param: {param.shape}, Mask: {mask.shape}")
            print(f"  Falling back to dense update")
            self._dense_step(param, group)
            return
        
        state = self.state[param]
        
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(param)
            state['exp_avg_sq'] = torch.zeros_like(param)
        
        state['step'] += 1
        
        try:
            triton_fused_sparse_adamw_step(
                param=param,
                grad=grad,
                mask=mask,
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
            print(f"ERROR in Triton kernel for {param_name}: {e}")
            print(f"  Falling back to dense update")
            self._dense_step(param, group)
            return
        
        self.stats['sparse_steps'] += 1
        self.stats['time_sparse'] += time.time() - start
    
    def _dense_step(self, param, group):
        """Standard dense AdamW update (fallback for non-masked params)."""
        start = time.time()
        
        grad = param.grad
        state = self.state[param]
        
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(param)
            state['exp_avg_sq'] = torch.zeros_like(param)
        
        state['step'] += 1
        
        beta1, beta2 = group['betas']
        
        param.mul_(1 - group['lr'] * group['weight_decay'])
        state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
        state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        
        step_size = group['lr'] / bias_correction1
        denom = (state['exp_avg_sq'].sqrt() / bias_correction2 ** 0.5).add_(group['eps'])
        param.addcdiv_(state['exp_avg'], denom, value=-step_size)
        
        self.stats['dense_steps'] += 1
        self.stats['time_dense'] += time.time() - start
    
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
        print(f"Time sparse:      {self.stats['time_sparse']:.3f}s")
        print(f"Time dense:       {self.stats['time_dense']:.3f}s")
        if sparse > 0:
            print(f"Avg sparse step:  {self.stats['time_sparse']/sparse*1000:.2f}ms")
        if dense > 0:
            print(f"Avg dense step:   {self.stats['time_dense']/dense*1000:.2f}ms")
        if sparse > 0 and dense > 0:
            speedup = (self.stats['time_dense']/dense) / (self.stats['time_sparse']/sparse)
            print(f"Sparse speedup:   {speedup:.2f}x")
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
    model_name=MODEL_NAME,
    checkpoint_path=None,
    mask_path=MASK_PATH,
    n_steps=5,
    batch_size=1, 
    learning_rate=5e-5,
    subset_size=10,
    run_name="triton_sparse_dpo",
    mlp_only=True,
    block_size=BLOCK_SIZE
):
    """Train with full Triton-accelerated sparse training."""
    
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT
        print(f"No checkpoint specified, using base model: {checkpoint_path}")
    
    run_dir = make_run_dir(base_dir="results", run_name=run_name)
    print(f"\n{'='*60}")
    print(f"TRITON-ACCELERATED SPARSE DPO TRAINING")
    print(f"{'='*60}")
    print(f"Run directory: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Mask path: {mask_path}")
    print(f"Steps: {n_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load dataset
    dpo_dataset = load_dpo_dataset(subset_size=subset_size)
    
    def collator(examples):
        return dpo_collator_fn(examples, tokenizer)

    # Load model with BF16 (more stable than FP16)
    print(f"Loading model from: {checkpoint_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16  # CHANGED: BF16 instead of FP16 for stability
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.config.use_cache = False
    print(f"✓ Model loaded on {device} with dtype: {model.dtype}")

    # Load masks
    mask_manager = SparseMaskManager(mask_path, device=device)

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
        mask_shape = mask.shape if mask is not None else "N/A"
        match = "✓ MATCH" if (mask is not None and mask.shape == shape) else "✗ MISMATCH"
        print(f"  {name}")
        print(f"    Param shape: {shape}, Mask shape: {mask_shape} - {match}")

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
        bf16=True,  # Match model dtype
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
            "optimizer": "SparseAdamW_Triton_Fused",
            "dtype": "bfloat16"
        }
    )

    # Create custom sparse optimizer
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

    # Set up DPOTrainer
    print("Initializing DPOTrainer with Triton-accelerated SparseAdamW...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        optimizers=(sparse_optimizer, None),
    )

    # Train
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")
    
    training_start = time.time()
    dpo_trainer.train()
    training_time = time.time() - training_start
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {training_time:.2f}s")
    print(f"Time per step: {training_time/n_steps:.2f}s")
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
        description="Triton-accelerated sparse DPO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--checkpoint", type=str, default=None, help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--mask", type=str, default=MASK_PATH, help=f"Path to sparse mask file (default: {MASK_PATH})")
    parser.add_argument("--n_steps", type=int, default=5, help="Number of training steps (default: 5)")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size (default: 1)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--subset_size", type=int, default=10, help="Dataset subset size (default: 10)")
    parser.add_argument("--run_name", type=str, default="triton_sparse_dpo", help="Run name for wandb (default: triton_sparse_dpo)")
    parser.add_argument("--mlp_only", action="store_true", default=True, help="Only apply sparse training to MLP layers (default: True)")
    parser.add_argument("--block_size", type=int, default=BLOCK_SIZE, help=f"Triton kernel block size (default: {BLOCK_SIZE})")
    
    args = parser.parse_args()
    
    train(
        checkpoint_path=args.checkpoint,
        mask_path=args.mask,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        subset_size=args.subset_size,
        run_name=args.run_name,
        mlp_only=args.mlp_only,
        block_size=args.block_size,
    )