
#!/usr/bin/env python3
"""
Triton-Accelerated Sparse DPO Training - BSR BACKPROP FOCUSED

Refactored from sparse_DPO_v4.py.
Focus: Backprop Ablations (BSR Sparse MLP vs Dense MLP)

Key Features:
1. Injects BSR Sparse MLP Layers (SparseLinearLayer)
2. Uses Custom Sparse Autograd Function
3. Modular Architecture
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
from src.utils.data_utils import load_dpo_dataset, dpo_collator_fn
from src.utils.logging_utils import FlexibleCheckpointCallback, CSVLoggerCallback
from src.optimizers.sparse_adamw import SparseAdamW
from src.mlps.bsr_sparse_mlp import replace_linear_modules

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
    block_size_bsr,
    block_size_adam,
    optimizer_type,
    use_wandb,
    save_csv,
    grad_accum,
    max_grad_norm,
    adam_beta1,
    adam_beta2,
    adam_eps,
    dpo_beta,
    warmup_steps,
    disable_tf32,
):
    # Determine paths
    if checkpoint_path is None or str(checkpoint_path).lower() == "none":
        checkpoint_path = model_name
    
    if run_name is None:
        # Construct a descriptive run name
        parts = ["sparse_dpo"]
        
        # Optimizer info
        if optimizer_type == "sparse_adamw":
            parts.append("bsr_adamw")
        else:
            parts.append(optimizer_type)
            
        # Backprop info (always BSR in this script, but good to label)
        parts.append("bsr_backprop")
        
        # Model info
        parts.append(sanitize_model_name(model_name))
        
        run_name = "_".join(parts)
    
    wandb_project = "huggingface"
    os.environ["WANDB_PROJECT"] = wandb_project
    run_dir = os.path.join("results", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"SPARSE DPO BSR TRAINING")
    print(f"{'='*60}")
    print(f"Run Directory: {run_dir}")
    print(f"Block Size BSR: {block_size_bsr}")
    
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
    
    # Inject Sparse Layers
    mask_dict = {n: mask_manager.get_mask(n) for n, _ in model.named_parameters() 
                 if ('mlp' in n.lower() or not mlp_only) and 'weight' in n and mask_manager.has_mask(n)}
                 
    print(f"Injecting Sparse MLP BSR backward for {len(mask_dict)} layers...")
    use_tf32_kernel = not disable_tf32
    print(f"BSR Kernel TF32 Precision Enabled: {use_tf32_kernel}")
    replace_linear_modules(model, mask_dict, block_size=block_size_bsr, use_tf32=use_tf32_kernel)
    
    # Optimizer Logic
    print(f"Initializing {optimizer_type}...")
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            betas=(adam_beta1, adam_beta2), 
            eps=adam_eps
        )
    elif optimizer_type == "sparse_adamw":
        optimizer = SparseAdamW(
            list(model.named_parameters()), 
            mask_manager, 
            lr=learning_rate, 
            betas=(adam_beta1, adam_beta2),
            eps=adam_eps,
            block_size=block_size_adam,
            mlp_only=mlp_only,
            max_grad_norm=max_grad_norm
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
    if disable_tf32:
        print("Disabling TF32 for strict fp32 accumulation precision.")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
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
        report_to=["wandb"] if use_wandb else [],
        run_name=run_name,
        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=max_grad_norm,
        beta=dpo_beta,
        warmup_steps=warmup_steps,
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
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw", "sparse_adamw"], default="sparse_adamw")
    parser.add_argument("--block_size_bsr", type=int, default=16)
    parser.add_argument("--block_size_adam", type=int, default=128)
    parser.add_argument("--mlp_only", action="store_true", default=True)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name for WandB and results directory")
    
    # Stability Tuning Parameters
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="Adam epsilon (increase to 1e-5 for stability)")
    
    # Advanced Noise Reduction
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO margin parameter (increase to 0.2-0.5 to bound updates)")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Linear warmup steps for LR scheduler")
    parser.add_argument("--disable_tf32", action="store_true", help="Disable TF32 for strict fp32 math (slow but precise)")
    
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
        block_size_bsr=args.block_size_bsr,
        block_size_adam=args.block_size_adam,
        optimizer_type=args.optimizer,
        use_wandb=args.use_wandb,
        save_csv=args.save_csv,
        grad_accum=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        dpo_beta=args.dpo_beta,
        warmup_steps=args.warmup_steps,
        disable_tf32=args.disable_tf32,
    )
