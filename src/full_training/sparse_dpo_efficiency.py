
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
from src.utils.scratch_paths import default_hf_datasets_cache, default_rl_casino_outputs
from src.utils.data_utils import dpo_collator_fn
from src.utils.dataset_registry import get_dataset_config, load_dpo_dataset as registry_load_dpo
from src.utils.logging_utils import FlexibleCheckpointCallback, CSVLoggerCallback
from src.utils.grpo_checkpoint_utils import (
    maybe_load_wandb_resume_env,
    resolve_resume_checkpoint,
    RunManifestCallback,
    WandbRunIdCallback,
)
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
    warmup_ratio,
    weight_decay,
    max_length,
    max_prompt_length,
    dpo_beta,
    gradient_checkpointing=True,
    save_steps=None,
    save_total_limit=None,
    resume_from_checkpoint=None,
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
    output_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    resume_ckpt = resolve_resume_checkpoint(output_dir, resume_from_checkpoint)
    if use_wandb:
        maybe_load_wandb_resume_env(run_dir, resume_ckpt)

    use_hf_rolling = save_steps is not None and save_steps > 0 and save_steps < 10**9
    hf_save_total_limit = save_total_limit if save_total_limit is not None else 3

    print(f"\n{'='*60}")
    print(f"SPARSE DPO EFFICIENCY TRAINING")
    print(f"{'='*60}")
    print(f"Run Directory: {run_dir}")
    print(f"Dataset: {dataset_key} ({dataset_name})")
    print(f"Optimizer: {optimizer_type}")
    print(f"WandB: {use_wandb}, CSV: {save_csv}")
    print(
        f"DPO training: max_steps={n_steps}, num_train_epochs=1, peak_lr={learning_rate}, "
        f"warmup_ratio={warmup_ratio}, lr_scheduler=linear (Trainer; align with DPO_train.py / pipeline NUM_STEPS_DPO)"
    )
    print(f"HF rolling checkpoints: {use_hf_rolling} resume={resume_ckpt!r}")

    # Callback only uses this for optional full-delta dumps; not tied to dense delta_logs.
    checkpoint_schedule = [n_steps] if n_steps > 0 else []

    # Load Components
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    dpo_dataset = registry_load_dpo(dataset_key, subset_size=subset_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map=None
    )
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
        
    # Callbacks
    callbacks = []

    manifest = {
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "mask_path": mask_path,
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "n_steps": n_steps,
        "learning_rate": learning_rate,
        "optimizer": optimizer_type,
        "mlp_only": mlp_only,
        "output_dir": output_dir,
        "resume_from_checkpoint": resume_ckpt,
        "hf_rolling_save_steps": save_steps,
        "hf_save_total_limit": hf_save_total_limit if use_hf_rolling else None,
    }
    callbacks.append(RunManifestCallback(run_dir, manifest))
    if use_wandb:
        callbacks.append(WandbRunIdCallback(run_dir))

    if not resume_ckpt:
        base_state = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                base_state[name] = param.detach().float().cpu().clone()
        callbacks.append(
            FlexibleCheckpointCallback(
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
                wandb_project=wandb_project,
            )
        )
    else:
        print(
            "Resume: skipping FlexibleCheckpointCallback weight-delta logging "
            "(base_state would not match a cold start)."
        )

    if save_csv:
        callbacks.append(CSVLoggerCallback(output_dir=run_dir))

    if use_hf_rolling:
        save_strategy = "steps"
        cfg_save_steps = save_steps
        cfg_save_total = hf_save_total_limit
    else:
        save_strategy = "no"
        cfg_save_steps = 500
        cfg_save_total = None

    # DPO Config — align with src/full_training/DPO_train.py (step-based run: max_steps + num_train_epochs=1)
    dpo_config = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_steps=n_steps,
        num_train_epochs=1,
        lr_scheduler_type="linear",
        logging_steps=1,
        save_strategy=save_strategy,
        save_steps=cfg_save_steps,
        save_total_limit=cfg_save_total,
        report_to="wandb" if use_wandb else "none",
        run_name=run_name,
        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=gradient_checkpointing,
        beta=dpo_beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dpo_dataset,
        data_collator=lambda x: dpo_collator_fn(x, tokenizer),
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=resume_ckpt)

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
    parser.add_argument("--n_steps", type=int, default=250, help="Must match dense --num_steps / pipeline NUM_STEPS_DPO")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-7,
        help="Peak LR (default 5e-7, same as DPO_train.py / pipeline DPO_LEARNING_RATE)",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw", "sparse_adamw"], default="sparse_adamw")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument(
        "--mlp_only",
        action="store_true",
        default=False,
        help="Restrict sparse updates to MLP layers only (default: full model where masks exist)",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--dataset", type=str, default="light-r1",
                       help="Dataset key (light-r1, tulu3, math-step-dpo, codepref) or HuggingFace ID")
    parser.add_argument("--output_base_dir", type=str, default=default_rl_casino_outputs(), help="Base directory for outputs")
    parser.add_argument("--dataset_cache_dir", type=str, default=default_hf_datasets_cache(), help="Cache directory for HuggingFace datasets")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name for WandB and results directory")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=None)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    
    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
        else: raise argparse.ArgumentTypeError('Boolean value expected.')
        
    parser.add_argument("--save_model", type=str2bool, default=True, help="Save final model checkpoint (default: True)")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="HF Trainer checkpoint interval (rolling). Omit for no intermediate HF checkpoints (legacy behavior).",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Keep only the newest K HF checkpoints when --save_steps is set (default: 3).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint-* dir, or 'auto' for latest under run_dir/checkpoints.",
    )

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
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        dpo_beta=args.dpo_beta,
        gradient_checkpointing=False if args.no_gradient_checkpointing else (True if args.gradient_checkpointing is True else True),
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
