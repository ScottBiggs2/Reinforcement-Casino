#!/usr/bin/env python3
"""
Triton-Accelerated Sparse GRPO Training - Math Focused

Refactored from GRPO_train.py.
Focus: GRPO with BSR Sparse MLP and Math Rewards.
"""

import os
import sys
import argparse
import json
import time
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOTrainer, GRPOConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.mask_manager import SparseMaskManager
from src.utils.dataset_registry import get_dataset_config, load_grpo_dataset
from src.utils.grpo_rewards import GRPO_REWARD_FUNCS
from src.utils.scratch_paths import default_hf_datasets_cache, default_rl_casino_outputs
from src.optimizers.sparse_adamw import SparseAdamW
from src.mlps.bsr_sparse_mlp import replace_linear_modules, restore_linear_modules

# =========================================================================
# Callbacks
# =========================================================================

class FlexibleCheckpointCallback(TrainerCallback):
    """Callback that saves deltas on a flexible schedule."""
    def __init__(self, base_state, delta_log_dir, checkpoint_schedule, threshold, run_name):
        self.base_state = base_state
        self.delta_log_dir = delta_log_dir
        self.checkpoint_schedule = set(checkpoint_schedule)
        self.threshold = threshold
        self.run_name = run_name
        os.makedirs(self.delta_log_dir, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        step = state.global_step

        layer_stats = {}
        full_deltas_to_save = {}

        with torch.no_grad():
            for name, param in model.named_parameters():
                current = param.detach().float().cpu()
                diff = current - self.base_state[name]
                l2 = torch.norm(diff).item()
                frac_big = (diff.abs() > self.threshold).float().mean().item()
                layer_stats[name] = {"l2_from_init": l2, "frac_big_from_init": frac_big}
                if step in self.checkpoint_schedule:
                    full_deltas_to_save[name] = diff.clone()

        if step in self.checkpoint_schedule:
            delta_file = os.path.join(self.delta_log_dir, f"deltas_step_{step}.pt")
            torch.save(full_deltas_to_save, delta_file)
            print(f"  ✓ Saved checkpoint at step {step}")
            
        return control

def sanitize_model_name(model_name: str) -> str:
    sanitized = model_name.replace("/", "_").replace("-", "_").lower()
    return "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized).strip("_")

# =========================================================================
# Main Trainer Let's Go
# =========================================================================

def train(
    model_name, checkpoint_path, mask_path, n_steps, batch_size,
    learning_rate, subset_size, run_name, mlp_only, block_size_bsr,
    block_size_adam, optimizer_type, use_wandb, max_grad_norm,
    adam_beta1, adam_beta2, adam_eps, grpo_beta, warmup_steps,
    disable_tf32, save_model, dataset_key, output_base_dir, dataset_cache_dir,
    num_generations, generation_batch_size, grad_accum,
    max_completion_length=2048, max_prompt_length=512,
):
    if checkpoint_path is None or str(checkpoint_path).lower() == "none":
        checkpoint_path = model_name
    
    os.environ["HF_DATASETS_CACHE"] = dataset_cache_dir
    ds_config = get_dataset_config(dataset_key)
    dataset_name = ds_config["hf_id"]
    dataset_sanitized = ds_config["sanitized_name"]
    
    if run_name is None:
        parts = ["sparse_grpo"]
        if optimizer_type == "sparse_adamw": parts.append("bsr_adamw")
        else: parts.append(optimizer_type)
        parts.append(sanitize_model_name(model_name))
        parts.append(dataset_sanitized)
        run_name = "_".join(parts)
    
    wandb_project = "huggingface"
    if use_wandb:
        os.environ["WANDB_PROJECT"] = wandb_project
        
    run_dir = os.path.join(output_base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'='*60}\nSPARSE GRPO MATH TRAINING\n{'='*60}")
    print(f"Dataset: {dataset_key} ({dataset_name})")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    train_dataset = load_grpo_dataset(dataset_key, subset_size=subset_size)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    multi_gpu = int(os.environ.get("WORLD_SIZE", 1)) > 1
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map=None if multi_gpu else "auto"
    )
    if multi_gpu or (hasattr(model, "to") and device.type == "cuda"):
        model.to(device)
    model.config.use_cache = False
    
    # -------------------------------------------------------------------------
    # Sparse Core Injections
    # -------------------------------------------------------------------------
    mask_manager = SparseMaskManager(mask_path, device=device)
    mask_dict = {n: mask_manager.get_mask(n) for n, _ in model.named_parameters() 
                 if ('mlp' in n.lower() or not mlp_only) and 'weight' in n and mask_manager.has_mask(n)}
                 
    print(f"Injecting Sparse BSR layers for {len(mask_dict)} layers...")
    use_tf32_kernel = not disable_tf32
    replace_linear_modules(model, mask_dict, block_size=block_size_bsr, use_tf32=use_tf32_kernel)
    
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2), eps=adam_eps)
    elif optimizer_type == "sparse_adamw":
        optimizer = SparseAdamW(
            list(model.named_parameters()), mask_manager, lr=learning_rate, 
            betas=(adam_beta1, adam_beta2), eps=adam_eps, block_size=block_size_adam,
            mlp_only=mlp_only, max_grad_norm=max_grad_norm
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
    if disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
    base_state = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
             base_state[name] = param.detach().float().cpu().clone()
             
    checkpoint_schedule = list(range(10, 50, 10)) + list(range(100, 250, 50))
    callbacks = [FlexibleCheckpointCallback(
        base_state=base_state,
        delta_log_dir=os.path.join(run_dir, "deltas"),
        checkpoint_schedule=checkpoint_schedule,
        threshold=1e-3,
        run_name=run_name
    )]
    
    cfg = GRPOConfig(
        output_dir=os.path.join(run_dir, "checkpoints"),
        run_name=run_name,
        report_to=["wandb"] if use_wandb else [],
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        max_steps=n_steps,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=999999,
        remove_unused_columns=False,
        num_generations=num_generations,
        generation_batch_size=generation_batch_size,
        max_completion_length=max_completion_length,
        max_prompt_length=max_prompt_length,
        beta=grpo_beta,
        warmup_steps=warmup_steps,
    )
    
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=train_dataset,
        reward_funcs=GRPO_REWARD_FUNCS,
        processing_class=tokenizer,
        optimizers=(optimizer, None),
        callbacks=callbacks
    )
    
    # Timing measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    wall_start = time.time()
    trainer.train()
    wall_time = time.time() - wall_start

    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        gpu_time = start_event.elapsed_time(end_event) / 1000.0

    timing_results = {
        "method": "sparse_bsr",
        "optimizer": optimizer_type,
        "model": model_name,
        "block_size_bsr": block_size_bsr,
        "block_size_adam": block_size_adam,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "wall_time": wall_time,
        "time_per_step_wall": wall_time / n_steps,
    }
    if torch.cuda.is_available():
        timing_results["gpu_time"] = gpu_time
        timing_results["time_per_step_gpu"] = gpu_time / n_steps

    timing_path = os.path.join(run_dir, "timing_results.json")
    with open(timing_path, "w") as f:
        json.dump(timing_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Wall time: {wall_time:.2f}s | Per step: {wall_time/n_steps:.2f}s")
    if torch.cuda.is_available():
        print(f"GPU time:  {gpu_time:.2f}s | Per step: {gpu_time/n_steps:.2f}s")
    print(f"Timing saved to {timing_path}")
    print(f"{'='*60}")

    if save_model:
        print(f"\nTraining complete. Saving final model to {run_dir}/final_model...")
        restore_linear_modules(model)
        final_save_dir = os.path.join(run_dir, "final_model")
        os.makedirs(final_save_dir, exist_ok=True)
        trainer.save_model(final_save_dir)
        tokenizer.save_pretrained(final_save_dir)
        print(f"✓ Full checkpoint saved to {final_save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mask", type=str, default="masks/top_10.0pct_momentum_w25_step25.pt")
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw", "sparse_adamw"], default="sparse_adamw")
    parser.add_argument("--block_size_bsr", type=int, default=16)
    parser.add_argument("--block_size_adam", type=int, default=128)
    parser.add_argument(
        "--mlp_only",
        action="store_true",
        default=False,
        help="Restrict sparse updates to MLP layers only (default: full model where masks exist)",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="math-220k")
    parser.add_argument("--output_base_dir", type=str, default=default_rl_casino_outputs())
    parser.add_argument("--dataset_cache_dir", type=str, default=default_hf_datasets_cache())
    
    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
        else: raise argparse.ArgumentTypeError('Boolean value expected.')
        
    parser.add_argument("--save_model", type=str2bool, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--grpo_beta", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--max_completion_length", type=int, default=2048,
                        help="Max generated completion length (default 2048). Raise if "
                             "'completions/clipped_ratio' is close to 1.0 in logs.")
    parser.add_argument("--max_prompt_length", type=int, default=512)
    
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
        max_grad_norm=args.max_grad_norm,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        grpo_beta=args.grpo_beta,
        warmup_steps=args.warmup_steps,
        disable_tf32=args.disable_tf32,
        save_model=args.save_model,
        dataset_key=args.dataset,
        output_base_dir=args.output_base_dir,
        dataset_cache_dir=args.dataset_cache_dir,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,
        grad_accum=args.grad_accum,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
    )
