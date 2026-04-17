#!/usr/bin/env python3
"""
Baseline Dense GRPO Training - Timing Comparison Version

Uses same dataset/reward setup as sparse_grpo_bsr.py for fair comparison.
No sparse layers — pure dense training with timing measurement.

Run: python GRPO_timing_baseline.py --n_steps 50 --optimizer adamw
"""

import os
import sys
import json
import time
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from src.utils.trl_vllm_import_guard import apply_trl_vllm_skip

apply_trl_vllm_skip()

from trl import GRPOTrainer, GRPOConfig

from src.utils.dataset_registry import get_dataset_config, load_grpo_dataset
from src.utils.grpo_rewards import GRPO_REWARD_FUNCS
from src.utils.scratch_paths import default_hf_datasets_cache, default_rl_casino_outputs


# =========================================================================
# Main
# =========================================================================
def train_baseline(
    model_name, n_steps, batch_size, learning_rate, subset_size,
    optimizer_type, use_wandb, dataset_key, output_base_dir,
    dataset_cache_dir, num_generations, generation_batch_size, grad_accum,
):
    os.environ["HF_DATASETS_CACHE"] = dataset_cache_dir
    ds_config = get_dataset_config(dataset_key)

    run_name = f"dense_grpo_{optimizer_type}"
    run_dir = os.path.join(output_base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    if use_wandb:
        os.environ["WANDB_PROJECT"] = "huggingface"

    print(f"\n{'='*60}")
    print(f"BASELINE DENSE GRPO TRAINING")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Optimizer: {optimizer_type}")
    print(f"Steps: {n_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Grad accum: {grad_accum}")
    print(f"Dataset: {dataset_key}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = load_grpo_dataset(dataset_key, subset_size=subset_size)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.config.use_cache = False

    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

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
        max_completion_length=1024,
        max_prompt_length=512,
        beta=0.1,
    )

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=train_dataset,
        reward_funcs=GRPO_REWARD_FUNCS,
        processing_class=tokenizer,
        optimizers=(optimizer, None),
    )

    # Timing
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
        "method": "dense",
        "optimizer": optimizer_type,
        "model": model_name,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw"], default="adamw")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--dataset", type=str, default="math-220k")
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=default_rl_casino_outputs(),
        help="Also set RL_CASINO_SCRATCH_ROOT to change defaults (default: /scratch/$USER/...).",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=default_hf_datasets_cache(),
        help="HF datasets cache dir (default under RL_CASINO_SCRATCH_ROOT or /scratch/$USER).",
    )

    args = parser.parse_args()

    train_baseline(
        model_name=args.model_name,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        subset_size=args.subset_size,
        optimizer_type=args.optimizer,
        use_wandb=args.use_wandb,
        dataset_key=args.dataset,
        output_base_dir=args.output_base_dir,
        dataset_cache_dir=args.dataset_cache_dir,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,
        grad_accum=args.grad_accum,
    )
