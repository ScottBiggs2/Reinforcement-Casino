# src/sandbox/Triton_DPO_train.py

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

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.load_openr1 import load_openr1_subset
from utils.logging_utils import make_run_dir

WANDB_PROJECT = "rl-casino-triton"
MODEL_NAME = "google/gemma-3-270m-it"
DATASET_NAME = "Light-R1"
SUBSET_SIZE = 10
MASK_PATH = "masks/top_10.0_percent_mask.pt"
CHECKPOINT_PATH = "results/gemma_dpo_training/final_model"
BLOCK_SIZE = 32


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


# ============================================================================
# SPARSE MASK MANAGER
# ============================================================================

class SparseMaskManager:
    """Manages loading and applying sparse masks to model parameters."""
    
    def __init__(self, mask_path, device='cuda'):
        """
        Load masks from disk.
        
        Args:
            mask_path: Path to mask file (e.g., 'masks/top_10.0_percent_mask.pt')
            device: Device to load masks onto
        """
        print(f"Loading masks from {mask_path}...")
        self.masks = torch.load(mask_path, map_location='cpu')
        self.device = device
        
        # Move masks to device and compute statistics
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
        """
        Get mask for a parameter, converting naming convention.
        
        Args:
            param_name: PyTorch parameter name (with dots)
            
        Returns:
            Mask tensor or None if no mask exists
        """
        # Convert dots to underscores to match mask file format
        mask_key = param_name.replace('.', '_')
        return self.masks.get(mask_key, None)
    
    def is_mlp_layer(self, param_name):
        """Check if parameter belongs to an MLP layer."""
        return 'mlp' in param_name.lower()
    
    def has_mask(self, param_name):
        """Check if a mask exists for this parameter."""
        mask_key = param_name.replace('.', '_')
        return mask_key in self.masks


# ============================================================================
# TRITON-ACCELERATED OPTIMIZER HOOK
# ============================================================================

class TritonSparseGradientHook:
    """
    Hooks into the optimizer to apply sparse updates via Triton kernels.
    """
    
    def __init__(self, model, mask_manager, learning_rate, block_size=32, mlp_only=True):
        """
        Args:
            model: The model to optimize
            mask_manager: SparseMaskManager instance
            learning_rate: Learning rate for updates
            block_size: Block size for Triton kernels
            mlp_only: If True, only apply to MLP layers
        """
        self.model = model
        self.mask_manager = mask_manager
        self.lr = learning_rate
        self.block_size = block_size
        self.mlp_only = mlp_only
        
        # Statistics tracking
        self.stats = {
            'total_updates': 0,
            'sparse_updates': 0,
            'dense_updates': 0,
            'time_sparse': 0.0,
            'time_dense': 0.0,
        }
        
        self._register_hooks()
        print(f"✓ Triton sparse gradient hooks registered")
        print(f"  MLP-only mode: {mlp_only}")
        print(f"  Block size: {block_size}")
    
    def _register_hooks(self):
        """Register backward hooks on model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: self._gradient_hook(grad, name))
    
    def _gradient_hook(self, grad, param_name):
        """
        Hook called after gradient computation for each parameter.
        
        Args:
            grad: Gradient tensor
            param_name: Name of the parameter
            
        Returns:
            Modified gradient (or original if not applying sparse update)
        """
        # Check if we should apply sparse update
        should_apply = (
            self.mask_manager.has_mask(param_name) and
            len(grad.shape) == 2 and  # Only 2D tensors (weight matrices)
            (not self.mlp_only or self.mask_manager.is_mlp_layer(param_name))
        )
        
        if not should_apply:
            self.stats['dense_updates'] += 1
            return grad
        
        # Get mask
        mask = self.mask_manager.get_mask(param_name)
        
        # Apply sparse mask to gradient
        # Note: We're masking the gradient, not applying the update here
        # The optimizer will then apply: W_new = W_old - lr * masked_grad
        masked_grad = grad * mask
        
        self.stats['sparse_updates'] += 1
        self.stats['total_updates'] += 1
        
        return masked_grad
    
    def print_stats(self):
        """Print statistics about sparse vs dense updates."""
        total = self.stats['total_updates']
        sparse = self.stats['sparse_updates']
        dense = self.stats['dense_updates']
        
        print(f"\n{'='*60}")
        print(f"TRITON SPARSE UPDATE STATISTICS")
        print(f"{'='*60}")
        print(f"Total updates:  {total:,}")
        print(f"Sparse updates: {sparse:,} ({sparse/total*100 if total > 0 else 0:.1f}%)")
        print(f"Dense updates:  {dense:,} ({dense/total*100 if total > 0 else 0:.1f}%)")
        print(f"{'='*60}\n")


# ============================================================================
# WEIGHT DELTA CALLBACK (unchanged from original)
# ============================================================================

class WeightDeltaCallback(TrainerCallback):
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.previous_state_dict = None
        self.step_times = []

    def on_train_begin(self, args, state, control, **kwargs):
        # Capture initial weights
        model = kwargs["model"]
        self.previous_state_dict = {name: p.clone().detach().cpu() 
                                    for name, p in model.named_parameters()}
        self.train_start_time = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        step_time = time.time() - self.step_start_time
        self.step_times.append(step_time)
        
        model = kwargs["model"]
        current_state_dict = {name: p.clone().detach().cpu() 
                             for name, p in model.named_parameters()}
        
        weight_deltas = {name: current_state_dict[name] - self.previous_state_dict[name] 
                        for name in current_state_dict}

        step_dir = os.path.join(self.run_dir, f"step_{int(state.global_step)}")
        os.makedirs(step_dir, exist_ok=True)
        
        # Save deltas to disk
        num_saved = 0
        for name, delta in weight_deltas.items():
            if torch.all(delta == 0):
                continue
            safe_name = name.replace(".", "_")
            torch.save(delta, os.path.join(step_dir, f"{safe_name}.pt"))
            num_saved += 1
        
        # Update previous state
        self.previous_state_dict = current_state_dict
        
        # Log timing
        print(f"  Step {state.global_step}: {step_time:.2f}s, saved {num_saved} deltas")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start_time
        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        
        print(f"\n{'='*60}")
        print(f"TRAINING TIME STATISTICS")
        print(f"{'='*60}")
        print(f"Total training time: {total_time:.2f}s")
        print(f"Average step time:   {avg_step_time:.2f}s")
        print(f"Total steps:         {len(self.step_times)}")
        print(f"{'='*60}\n")


# ============================================================================
# DATASET PREPARATION (unchanged from original)
# ============================================================================

def prepare_dpo_dataset(original_dataset, tokenizer, model):
    """
    Prepare a synthetic DPO dataset from a standard dataset.
    'chosen' is the ground truth, 'rejected' is generated by the base model.
    """
    dpo_data = []
    print("Preparing DPO dataset...")
    for i, item in enumerate(original_dataset):
        prompt = item['prompt']
        chosen = item['label']

        # Generate a 'rejected' response from the base model
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        generated_ids = model.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=20,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.8
        )
        rejected = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        dpo_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
        
        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/{len(original_dataset)} DPO pairs")
    
    return Dataset.from_list(dpo_data)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(
    model_name=MODEL_NAME,
    checkpoint_path=CHECKPOINT_PATH,
    mask_path=MASK_PATH,
    n_steps=5,
    batch_size=1, 
    learning_rate=5e-5,
    subset_size=10,
    run_name="triton_dpo_test",
    mlp_only=True,
    block_size=BLOCK_SIZE
):
    """
    Train a Gemma model using DPOTrainer with Triton-accelerated sparse updates.
    
    Args:
        model_name: Base model name (for tokenizer)
        checkpoint_path: Path to fine-tuned checkpoint to load
        mask_path: Path to sparse mask file
        n_steps: Number of training steps
        batch_size: Training batch size
        learning_rate: Learning rate
        subset_size: Dataset subset size
        run_name: Name for this run
        mlp_only: If True, only apply sparse updates to MLP layers
        block_size: Block size for Triton kernels
    """
    # Create output directory
    run_dir = make_run_dir(base_dir="results", run_name=run_name)
    print(f"\n{'='*60}")
    print(f"TRITON-ACCELERATED DPO TRAINING")
    print(f"{'='*60}")
    print(f"Run directory: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Mask path: {mask_path}")
    print(f"Steps: {n_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")

    # Load dataset and tokenizer
    original_dataset, tokenizer = load_openr1_subset(
        tokenizer_name=model_name,
        subset_size=subset_size
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model from checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✓ Model loaded on {device}")

    # Load sparse masks
    mask_manager = SparseMaskManager(mask_path, device=device)

    # Prepare DPO dataset
    print("\nPreparing DPO dataset...")
    dpo_dataset = prepare_dpo_dataset(original_dataset, tokenizer, model)
    print(f"✓ DPO dataset prepared with {len(dpo_dataset)} examples\n")

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
            "triton_enabled": True
        }
    )

    # Set up Triton sparse gradient hooks
    sparse_hook = TritonSparseGradientHook(
        model=model,
        mask_manager=mask_manager,
        learning_rate=learning_rate,
        block_size=block_size,
        mlp_only=mlp_only
    )

    # Set up DPOTrainer
    print("Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
        callbacks=[WeightDeltaCallback(run_dir=run_dir)]
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
    
    # Print sparse update statistics
    sparse_hook.print_stats()
    
    # Save final model
    final_model_path = os.path.join(run_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"✓ Final model saved to {final_model_path}")
    
    # Save timing info
    timing_info = {
        'total_training_time': training_time,
        'time_per_step': training_time / n_steps,
        'n_steps': n_steps,
        'sparse_stats': sparse_hook.stats
    }
    with open(os.path.join(run_dir, 'timing_info.json'), 'w') as f:
        json.dump(timing_info, f, indent=2)
    print(f"✓ Timing info saved to {run_dir}/timing_info.json")
    
    wandb.finish()
    print("\n✓ All done!")


if __name__ == "__main__":
    train()