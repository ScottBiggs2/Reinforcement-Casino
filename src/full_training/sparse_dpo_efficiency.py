
#!/usr/bin/env python3
"""
Triton-Accelerated Sparse DPO Training - EFFICIENCY FOCUSED

Refactored from sparse_DPO_v3.py.
Focus: Optimizer Ablations (Sparse AdamW vs Dense/AdamW/SGD)

Key Features:
1. Modular architecture using src.kernels, src.optimizers, src.utils
2. Flexible logging (CSV, WandB)
3. Optimized Triton kernels for Sparse AdamW
4. Flexible checkpointing
"""

import os
import sys
import argparse
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

# Add project root to sys.path to resolve 'src' imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.mask_manager import SparseMaskManager
from src.utils.data_utils import dpo_collator_fn
from src.utils.dataset_registry import get_dataset_config, load_dpo_dataset as registry_load_dpo
from src.utils.logging_utils import FlexibleCheckpointCallback, CSVLoggerCallback
from src.optimizers.sparse_adamw import SparseAdamW

def sanitize_model_name(model_name: str) -> str:
    sanitized = model_name.replace("/", "_").replace("-", "_").lower()
    return "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized).strip("_")

def train(
    model_name,
    checkpoint_path,
    mask_path,
    n_steps,
    batch_size,
    learning_rate,
    subset_size,
    run_name,
    mlp_only,
    block_size,
    optimizer_type,
    use_wandb,
    save_csv,
    grad_accum,
    save_model,
    dataset_key,
    output_base_dir,
    dataset_cache_dir,
):
    # Determine model path
    if checkpoint_path is None or str(checkpoint_path).lower() == "none":
        checkpoint_path = model_name
    
    # Set dataset cache directory
    os.environ["HF_DATASETS_CACHE"] = dataset_cache_dir
    
    # Resolve dataset via registry
    ds_config = get_dataset_config(dataset_key)
    dataset_name = ds_config["hf_id"]
    dataset_sanitized = ds_config["sanitized_name"]
    
    if run_name is None:
        run_name = f"sparse_dpo_efficiency_{optimizer_type}_{sanitize_model_name(model_name)}_{dataset_sanitized}"
    
    wandb_project = "huggingface"
    os.environ["WANDB_PROJECT"] = wandb_project
    run_dir = os.path.join(output_base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"SPARSE DPO EFFICIENCY TRAINING")
    print(f"{'='*60}")
    print(f"Run Directory: {run_dir}")
    print(f"Dataset: {dataset_key} ({dataset_name})")
    print(f"Optimizer: {optimizer_type}")
    print(f"WandB: {use_wandb}, CSV: {save_csv}")
    
    # Load Components
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    dpo_dataset = registry_load_dpo(dataset_key, subset_size=subset_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto"
    )
    if hasattr(model, "to") and device.type == "cuda": model.to(device)
    model.config.use_cache = False
    
    # Mask Manager
    mask_manager = SparseMaskManager(mask_path, device=device)
    
    # Optimizer Logic
    print(f"Initializing {optimizer_type}...")
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_type == "sparse_adamw":
        optimizer = SparseAdamW(
            list(model.named_parameters()), 
            mask_manager, 
            lr=learning_rate, 
            block_size=block_size,
            mlp_only=mlp_only
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
    # Checkpoint Schedule
    checkpoint_schedule = list(range(10, 50, 10)) + list(range(100, 250, 50))
    
    # Callbacks
    callbacks = []
    
    # Snapshot base state for flexible callback
    base_state = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
             base_state[name] = param.detach().float().cpu().clone()
             
    callbacks.append(FlexibleCheckpointCallback(
        base_state=base_state,
        delta_log_dir=os.path.join(run_dir, "deltas"),
        checkpoint_schedule=checkpoint_schedule,
        threshold=1e-3,
        model_name=model_name,
        dataset_name=dataset_name,
        subset_size=subset_size,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_accum=grad_accum,
        run_name=run_name,
        use_wandb=use_wandb,
        wandb_project=wandb_project
    ))
    
    if save_csv:
        callbacks.append(CSVLoggerCallback(output_dir=run_dir))
    
    # DPO Config
    dpo_config = DPOConfig(
        output_dir=os.path.join(run_dir, "checkpoints"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        max_steps=n_steps,
        logging_steps=1,
        report_to="wandb" if use_wandb else "none",
        run_name=run_name,
        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=True,
    )
    
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dpo_dataset,
        data_collator=lambda x: dpo_collator_fn(x, tokenizer),
        optimizers=(optimizer, None),
        callbacks=callbacks
    )
    
    trainer.train()

    # Final Saving
    if save_model:
        print(f"\nTraining complete. Saving final model to {run_dir}/final_model...")
        final_save_dir = os.path.join(run_dir, "final_model")
        os.makedirs(final_save_dir, exist_ok=True)
        
        trainer.save_model(final_save_dir)
        tokenizer.save_pretrained(final_save_dir)
        print(f"✓ Full checkpoint saved to {final_save_dir}")
    else:
        print("\nTraining complete. Skipping final model saving as requested.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mask", type=str, default="masks/top_10.0pct_momentum_w25_step25.pt")
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw", "sparse_adamw"], default="sparse_adamw")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--mlp_only", action="store_true", default=True)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--dataset", type=str, default="light-r1",
                       help="Dataset key (light-r1, tulu3, math-step-dpo, codepref) or HuggingFace ID")
    parser.add_argument("--output_base_dir", type=str, default="/scratch/biggs.s/rl_casino_outputs", help="Base directory for outputs")
    parser.add_argument("--dataset_cache_dir", type=str, default="/scratch/biggs.s/hf_cache/datasets", help="Cache directory for HuggingFace datasets")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name for WandB and results directory")
    
    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
        else: raise argparse.ArgumentTypeError('Boolean value expected.')
        
    parser.add_argument("--save_model", type=str2bool, default=True, help="Save final model checkpoint (default: True)")
    
    args = parser.parse_args()
    
    train(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        mask_path=args.mask,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        subset_size=args.subset_size,
        run_name=args.run_name,
        mlp_only=args.mlp_only,
        block_size=args.block_size,
        optimizer_type=args.optimizer,
        use_wandb=args.use_wandb,
        save_csv=args.save_csv,
        grad_accum=args.grad_accum,
        save_model=args.save_model,
        dataset_key=args.dataset,
        output_base_dir=args.output_base_dir,
        dataset_cache_dir=args.dataset_cache_dir,
    )
