
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
import argparse
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

from src.utils.mask_manager import SparseMaskManager
from src.utils.data_utils import load_dpo_dataset, dpo_collator_fn
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
):
    # Determine paths
    if checkpoint_path is None or str(checkpoint_path).lower() == "none":
        checkpoint_path = model_name
    
    if run_name is None:
        run_name = f"sparse_dpo_efficiency_{optimizer_type}_{sanitize_model_name(model_name)}"
    
    run_dir = os.path.join("results", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"SPARSE DPO EFFICIENCY TRAINING")
    print(f"{'='*60}")
    print(f"Run Directory: {run_dir}")
    print(f"Optimizer: {optimizer_type}")
    print(f"WandB: {use_wandb}, CSV: {save_csv}")
    
    # Load Components
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    dpo_dataset = load_dpo_dataset(argparse.Namespace(dataset_name="qihoo360/Light-R1-DPOData").dataset_name, subset_size=subset_size)
    
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
        dataset_name="qihoo360/Light-R1-DPOData",
        subset_size=subset_size,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_accum=grad_accum,
        run_name=run_name,
        use_wandb=use_wandb
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
        report_to="none", # We handle wandb manually in callback
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
    
    args = parser.parse_args()
    
    train(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        mask_path=args.mask,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        subset_size=args.subset_size,
        run_name=None,
        mlp_only=args.mlp_only,
        block_size=args.block_size,
        optimizer_type=args.optimizer,
        use_wandb=args.use_wandb,
        save_csv=args.save_csv,
        grad_accum=args.grad_accum,
    )
