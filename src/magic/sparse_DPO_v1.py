"""
BSR Sparse DPO Training with Fixed Precomputed Mask

This is a clean, focused implementation for validating BSR speedups.
Key simplifications:
- Uses a fixed mask (no updates during training)
- Only weights in the mask can change
- Focus on measuring actual speedup

Requirements:
- torch==2.9.0
- triton
- transformers==4.57.1
- trl==0.24.0
"""

import os
import argparse
import torch
import triton
import triton.language as tl
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOTrainer, DPOConfig
from typing import List, Dict, Any
import time
import numpy as np


#######################################
# TRITON KERNEL (Your Validated Implementation)
#######################################

@triton.jit
def sparse_adam_2d_kernel(
    # Pointers to tensors
    weights_ptr,
    grads_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    mask_ptr,
    # Matrix dimensions
    M, N,
    stride_wm, stride_wn,
    stride_gm, stride_gn,
    stride_em, stride_en,
    stride_vm, stride_vn,
    stride_mm, stride_mn,
    # Optimization parameters
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bias_correction1,
    bias_correction2,
    use_adamw: tl.constexpr,
    # Block info
    BLOCK_SIZE: tl.constexpr,
):
    """
    2D Block-sparse Adam kernel - assumes gradients are already masked.
    Uses early exit for masked blocks (KEY to performance).
    """
    pid = tl.program_id(0)
    
    # Calculate block position
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE)
    block_row = pid // num_blocks_n
    block_col = pid % num_blocks_n
    
    # Calculate starting position
    row_start = block_row * BLOCK_SIZE
    col_start = block_col * BLOCK_SIZE
    
    # Create offset ranges
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    row_offsets = row_offsets[:, None]
    col_offsets = col_offsets[None, :]
    
    # Create mask for valid elements
    mask_valid = (row_offsets < M) & (col_offsets < N)
    
    # Check if this block should be processed (sample center element)
    center_row = row_start + BLOCK_SIZE // 2
    center_col = col_start + BLOCK_SIZE // 2
    if center_row < M and center_col < N:
        block_mask = tl.load(mask_ptr + center_row * stride_mm + center_col * stride_mn)
        if block_mask == 0.0:
            return  # Skip this block entirely - KEY OPTIMIZATION
    
    # Calculate memory offsets
    w_offsets = row_offsets * stride_wm + col_offsets * stride_wn
    g_offsets = row_offsets * stride_gm + col_offsets * stride_gn
    e_offsets = row_offsets * stride_em + col_offsets * stride_en
    v_offsets = row_offsets * stride_vm + col_offsets * stride_vn
    
    # Load data (gradients should already be masked)
    w = tl.load(weights_ptr + w_offsets, mask=mask_valid, other=0.0)
    g = tl.load(grads_ptr + g_offsets, mask=mask_valid, other=0.0)
    m = tl.load(exp_avg_ptr + e_offsets, mask=mask_valid, other=0.0)
    v = tl.load(exp_avg_sq_ptr + v_offsets, mask=mask_valid, other=0.0)
    
    # Update moments
    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * v + (1.0 - beta2) * g * g
    
    # Bias-corrected moments
    m_hat = m_new / bias_correction1
    v_hat = v_new / bias_correction2
    
    # Compute update
    denom = tl.sqrt(v_hat) + eps
    update = m_hat / denom
    
    # Apply weight decay (AdamW style)
    if use_adamw:
        w_new = w * (1.0 - lr * weight_decay) - lr * update
    else:
        w_new = w - lr * update
        if weight_decay != 0.0:
            w_new = w_new - lr * weight_decay * w
    
    # Store updates
    tl.store(weights_ptr + w_offsets, w_new, mask=mask_valid)
    tl.store(exp_avg_ptr + e_offsets, m_new, mask=mask_valid)
    tl.store(exp_avg_sq_ptr + v_offsets, v_new, mask=mask_valid)


#######################################
# SPARSE OPTIMIZER
#######################################

class SparseAdamW(torch.optim.Optimizer):
    """
    Sparse AdamW optimizer using Triton kernels with FIXED masks.
    
    Key features:
    - Only updates parameters where mask = 1
    - Mask is set once at initialization and never changes
    - Falls back to dense PyTorch for 1D params (biases, norms)
    """
    
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        block_size=32,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            block_size=block_size,
        )
        super().__init__(params, defaults)
        
        # Store masks (set once, never changed)
        self.param_masks = {}
        self.param_name_mapping = {}
    
    def set_masks(self, masks: Dict[str, torch.Tensor]):
        """
        Set masks for all parameters at once.
        Call this ONCE before training starts.
        
        Args:
            masks: Dict mapping parameter names to binary mask tensors
        """
        self.param_masks = {k: v.cuda() for k, v in masks.items()}
        
        # Report statistics
        total_params = sum(m.numel() for m in masks.values())
        active_params = sum((m > 0).sum().item() for m in masks.values())
        sparsity = 1.0 - (active_params / total_params) if total_params > 0 else 0.0
        
        print(f"\nSparseAdamW: Loaded {len(masks)} masks")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Active parameters: {active_params:,} ({(1-sparsity)*100:.1f}%)")
        print(f"  Sparsity: {sparsity*100:.1f}%")
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            block_size = group['block_size']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Get parameter name for mask lookup
                param_name = None
                for name, param in self.param_name_mapping.items():
                    if param is p:
                        param_name = name
                        break
                
                # Initialize state if needed
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                state['step'] += 1
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                step = state['step']
                
                # Bias corrections
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                
                # Check if we have a mask for this parameter
                mask = self.param_masks.get(param_name)
                
                # Only use sparse kernel for 2D parameters with masks
                if mask is not None and p.dim() == 2:
                    self._sparse_update_2d(
                        p, grad, exp_avg, exp_avg_sq, mask,
                        lr, beta1, beta2, eps, weight_decay,
                        bias_correction1, bias_correction2,
                        block_size
                    )
                else:
                    # Fall back to dense PyTorch AdamW for 1D params or unmasked params
                    self._dense_update(
                        p, grad, exp_avg, exp_avg_sq,
                        lr, beta1, beta2, eps, weight_decay,
                        bias_correction1, bias_correction2
                    )
        
        return loss
    
    def _sparse_update_2d(
        self, param, grad, exp_avg, exp_avg_sq, mask,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2, block_size
    ):
        """Apply sparse Triton kernel for 2D parameters."""
        M, N = param.shape
        
        # Ensure all tensors are contiguous and on GPU
        param = param.contiguous()
        grad = grad.contiguous()
        exp_avg = exp_avg.contiguous()
        exp_avg_sq = exp_avg_sq.contiguous()
        mask = mask.contiguous()
        
        # Apply mask to gradient (sparse backward)
        grad = grad * mask
        
        # Calculate grid size
        num_blocks_m = triton.cdiv(M, block_size)
        num_blocks_n = triton.cdiv(N, block_size)
        grid = (num_blocks_m * num_blocks_n,)
        
        # Launch sparse Adam kernel
        sparse_adam_2d_kernel[grid](
            param, grad, exp_avg, exp_avg_sq, mask,
            M, N,
            param.stride(0), param.stride(1),
            grad.stride(0), grad.stride(1),
            exp_avg.stride(0), exp_avg.stride(1),
            exp_avg_sq.stride(0), exp_avg_sq.stride(1),
            mask.stride(0), mask.stride(1),
            lr, beta1, beta2, eps, weight_decay,
            bias_correction1, bias_correction2,
            True,  # use_adamw
            BLOCK_SIZE=block_size,
        )
    
    def _dense_update(
        self, param, grad, exp_avg, exp_avg_sq,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2
    ):
        """Standard PyTorch AdamW for 1D parameters."""
        # Update biased first moment estimate
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        # Update biased second raw moment estimate
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # Compute bias-corrected first and second moment estimates
        denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(eps)
        step_size = lr / bias_correction1
        
        # AdamW update
        param.mul_(1 - lr * weight_decay)
        param.addcdiv_(exp_avg, denom, value=-step_size)


#######################################
# MASK LOADING
#######################################

def load_precomputed_masks(
    mask_file: str,
    model_state_dict: Dict[str, torch.Tensor],
    device: torch.device = torch.device('cuda'),
) -> Dict[str, torch.Tensor]:
    """
    Load masks generated by mask_finder.py and convert from underscore to dot notation.
    
    mask_finder.py saves: 'model_layers_0_self_attn_q_proj_weight'
    PyTorch expects:      'model.layers.0.self_attn.q_proj.weight'
    
    Args:
        mask_file: Path to .pt file with masks (from mask_finder.py)
        model_state_dict: Model's state dict to match parameter names
        device: Device to load masks onto
    
    Returns:
        Dictionary of masks with dot notation keys
    """
    print(f"\nLoading precomputed masks from {mask_file}...")
    
    # Load masks with underscore keys
    masks_underscore = torch.load(mask_file, map_location='cpu')
    
    # Create mapping from underscore to dot notation
    param_names_dots = set(model_state_dict.keys())
    masks_dots = {}
    
    matched = 0
    unmatched = []
    
    for param_name_dot in param_names_dots:
        # Skip non-weight parameters
        if not param_name_dot.endswith('.weight') or model_state_dict[param_name_dot].dim() != 2:
            continue
        
        # Convert dot to underscore
        param_name_underscore = param_name_dot.replace(".", "_")
        
        if param_name_underscore in masks_underscore:
            mask = masks_underscore[param_name_underscore]
            
            # Verify shape matches
            expected_shape = model_state_dict[param_name_dot].shape
            if mask.shape != expected_shape:
                print(f"⚠️  Shape mismatch for {param_name_dot}")
                print(f"   Expected: {expected_shape}, Got: {mask.shape}")
                continue
            
            # Store with dot notation and move to device
            masks_dots[param_name_dot] = mask.to(device)
            matched += 1
        else:
            unmatched.append(param_name_dot)
    
    # Report statistics
    total_params = sum(m.numel() for m in masks_dots.values())
    active_params = sum((m > 0).sum().item() for m in masks_dots.values())
    actual_sparsity = 1.0 - (active_params / total_params) if total_params > 0 else 0.0
    
    print(f"\nMask Loading Summary:")
    print(f"  Matched: {matched} parameters")
    print(f"  Unmatched: {len(unmatched)} parameters")
    if len(unmatched) > 0 and len(unmatched) <= 5:
        for name in unmatched[:5]:
            print(f"    - {name}")
    print(f"  Total masked parameters: {total_params:,}")
    print(f"  Active parameters: {active_params:,}")
    print(f"  Sparsity: {actual_sparsity*100:.1f}%")
    
    return masks_dots


#######################################
# DATA COLLATOR
#######################################

def dpo_collator_fn(tokenizer, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collator for DPO data compatible with TRL 0.24.0."""
    
    # Check if already tokenized
    if "prompt_input_ids" in examples[0]:
        # Already tokenized - just pad and stack
        def pad_stack(key):
            seqs = [torch.tensor(ex[key]) if not isinstance(ex[key], torch.Tensor) else ex[key] 
                    for ex in examples]
            maxlen = max(s.size(-1) for s in seqs)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            out = torch.full((len(seqs), maxlen), fill_value=pad_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, :s.size(-1)] = s.to(torch.long)
                mask[i, :s.size(-1)] = 1
            return out, mask
        
        p_ids, p_mask = pad_stack("prompt_input_ids")
        c_ids, c_mask = pad_stack("chosen_input_ids")
        r_ids, r_mask = pad_stack("rejected_input_ids")
        
        return {
            "prompt_input_ids": p_ids,
            "prompt_attention_mask": p_mask,
            "chosen_input_ids": c_ids,
            "chosen_attention_mask": c_mask,
            "rejected_input_ids": r_ids,
            "rejected_attention_mask": r_mask,
        }
    
    # Tokenize on the fly
    prompts = [ex.get("prompt", "") for ex in examples]
    chosens = [ex.get("chosen", "") for ex in examples]
    rejects = [ex.get("rejected", "") for ex in examples]
    
    enc_prompt = tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    enc_chosen = tokenizer(
        chosens,
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors="pt"
    )
    enc_reject = tokenizer(
        rejects,
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors="pt"
    )
    
    return {
        "prompt_input_ids": enc_prompt["input_ids"],
        "prompt_attention_mask": enc_prompt["attention_mask"],
        "chosen_input_ids": enc_chosen["input_ids"],
        "chosen_attention_mask": enc_chosen["attention_mask"],
        "rejected_input_ids": enc_reject["input_ids"],
        "rejected_attention_mask": enc_reject["attention_mask"],
    }


#######################################
# TIMING CALLBACK
#######################################

class TimingCallback(TrainerCallback):
    """Callback to track step timing for speedup measurement."""
    
    def __init__(self):
        self.step_times = []
        self.start_time = None
    
    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        if self.start_time is not None:
            step_time = time.time() - self.start_time
            self.step_times.append(step_time)
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    "timing/step_time_ms": step_time * 1000,
                    "timing/steps_per_sec": 1.0 / step_time if step_time > 0 else 0,
                }, step=state.global_step)
    
    def get_avg_step_time(self, skip_first_n=10):
        """Get average step time, skipping warmup steps."""
        if len(self.step_times) <= skip_first_n:
            return 0.0
        return np.mean(self.step_times[skip_first_n:])


#######################################
# MAIN TRAINING FUNCTION
#######################################

def main(args):
    print("="*80)
    print("BSR SPARSE DPO TRAINING (FIXED MASK)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Mask file: {args.mask_file}")
    print(f"  Output: {args.output_dir}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Block size: {args.block_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print("="*80 + "\n")
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - GPU required for BSR training")
    
    print(f"Device: {torch.cuda.get_device_name()}")
    cc = torch.cuda.get_device_capability()
    print(f"Compute Capability: {cc}")
    if cc[0] < 7:
        print("⚠️  Warning: Compute capability < 7.0 may have limited Triton support\n")
    else:
        print("✓ GPU compatible with Triton kernels\n")
    
    #######################################
    # Load Dataset
    #######################################
    print("Loading dataset...")
    
    def normalize_record(rec):
        def msg_to_text(x):
            if isinstance(x, str):
                return x
            if isinstance(x, dict):
                return x.get("value", "")
            if isinstance(x, list):
                return "\n".join(m.get("value", "") for m in x if isinstance(m, dict))
            return str(x)
        
        prompt_raw = rec.get("prompt", "")
        if isinstance(prompt_raw, list):
            prompt_text = "\n".join(
                m.get("value", "") for m in prompt_raw
                if isinstance(m, dict) and m.get("from", "").lower() != "assistant"
            ).strip()
        else:
            prompt_text = msg_to_text(prompt_raw).strip()
        
        chosen_text = msg_to_text(rec.get("chosen", "")).strip()
        rejected_text = msg_to_text(rec.get("rejected", "")).strip()
        
        return {"prompt": prompt_text, "chosen": chosen_text, "rejected": rejected_text}
    
    raw_ds = load_dataset(args.dataset_name, split="train")
    norm_ds = raw_ds.map(normalize_record, remove_columns=raw_ds.column_names)
    
    if args.subset_size is not None:
        norm_ds = norm_ds.select(range(min(args.subset_size, len(norm_ds))))
    
    train_dataset = norm_ds
    print(f"Dataset size: {len(train_dataset)}\n")
    
    #######################################
    # Load Model & Tokenizer
    #######################################
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
    
    #######################################
    # Load Precomputed Masks
    #######################################
    if not os.path.exists(args.mask_file):
        raise FileNotFoundError(f"Mask file not found: {args.mask_file}")
    
    masks = load_precomputed_masks(
        mask_file=args.mask_file,
        model_state_dict=model.state_dict(),
        device=torch.device('cuda'),
    )
    
    if not masks:
        raise ValueError("No valid masks loaded - check mask file compatibility")
    
    #######################################
    # Create Sparse Optimizer
    #######################################
    print("\nCreating SparseAdamW optimizer...")
    optimizer = SparseAdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        block_size=args.block_size,
    )
    
    # Set up parameter name mapping
    optimizer.param_name_mapping = {
        name: param for name, param in model.named_parameters()
    }
    
    # Apply precomputed masks (ONCE, never changed)
    optimizer.set_masks(masks)
    
    #######################################
    # Configure Training
    #######################################
    run_name = args.run_name or f"sparse_bsr_{os.path.basename(args.mask_file)[:-3]}"
    
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        report_to=["wandb"] if args.wandb else [],
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.num_steps,
        num_train_epochs=1,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        warmup_steps=100,
        remove_unused_columns=False,
        # DPO parameters
        beta=0.1,
        max_length=1024,
        max_prompt_length=512,
        loss_type="sigmoid",
    )
    
    #######################################
    # Create Trainer
    #######################################
    print("\nCreating DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,  # TRL 0.24.0
        data_collator=lambda ex: dpo_collator_fn(tokenizer, ex),
        optimizers=(optimizer, None),  # Use sparse optimizer
    )
    
    # Add timing callback
    timing_callback = TimingCallback()
    trainer.add_callback(timing_callback)
    
    #######################################
    # Initialize W&B
    #######################################
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": args.model_name,
                "dataset": args.dataset_name,
                "mask_file": args.mask_file,
                "block_size": args.block_size,
                "num_steps": args.num_steps,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "sparsity": 1.0 - (sum((m > 0).sum().item() for m in masks.values()) / 
                                  sum(m.numel() for m in masks.values())),
            },
        )
    
    #######################################
    # Train
    #######################################
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    #######################################
    # Report Results
    #######################################
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    avg_step_time = timing_callback.get_avg_step_time()
    if avg_step_time > 0:
        print(f"\nPerformance:")
        print(f"  Average step time: {avg_step_time*1000:.2f}ms")
        print(f"  Steps per second: {1.0/avg_step_time:.3f}")
    
    # Save final model
    final_checkpoint = os.path.join(args.output_dir, "final")
    print(f"\nSaving final model to {final_checkpoint}...")
    trainer.save_model(final_checkpoint)
    
    if args.wandb:
        wandb.finish()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BSR Sparse DPO Training with Fixed Precomputed Mask"
    )
    
    # Model & Dataset
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--dataset_name", type=str, default="qihoo360/Light-R1-DPOData")
    parser.add_argument("--subset_size", type=int, default=None)
    
    # Mask & Output
    parser.add_argument("--mask_file", type=str, required=True,
                       help="Path to precomputed mask file from mask_finder.py")
    parser.add_argument("--output_dir", type=str, default="./results/sparse_bsr")
    
    # Training
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    # Sparse Training
    parser.add_argument("--block_size", type=int, default=32,
                       help="Triton kernel block size (16, 32, or 64)")
    
    # Logging
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="rl-casino-bsr")
    parser.add_argument("--run_name", type=str, default=None)
    
    args = parser.parse_args()
    main(args)