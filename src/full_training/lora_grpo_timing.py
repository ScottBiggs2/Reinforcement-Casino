#!/usr/bin/env python3
"""
LoRA GRPO Training - Timing Comparison Version

Baseline LoRA training for speedup comparison against dense and sparse methods.
Uses PEFT LoRA on the same model/dataset as other benchmarks.

Run: python lora_grpo_timing.py --n_steps 50 --lora_rank 16
"""

import os
import sys
import argparse
import re
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from src.utils.trl_vllm_import_guard import apply_trl_vllm_skip

apply_trl_vllm_skip()

from trl import GRPOTrainer, GRPOConfig

from src.utils.dataset_registry import get_dataset_config, load_grpo_dataset


# =========================================================================
# Math Reward Functions (same as sparse_grpo_bsr.py for fair comparison)
# =========================================================================

def parse_reasoning_response(text: str) -> dict:
    pattern = r"<think>\s*(.*?)\s*</think>\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return {"thinking_content": "", "response": text}
    return {"thinking_content": match.group(1).strip(), "response": match.group(2).strip()}

def get_completion_content(completion) -> str:
    if isinstance(completion, list):
        return " ".join(msg.get("content", "") if isinstance(msg, dict) else str(msg) for msg in completion)
    return str(completion)

def parse_responses(completions: list) -> list[dict]:
    return [parse_reasoning_response(get_completion_content(c)) for c in completions]

def accuracy_reward(completions, solution, **kwargs) -> list[float]:
    parsed_responses = parse_responses(completions)
    rewards = []
    for r, ans in zip(parsed_responses, solution):
        model_answer = r["response"].strip()
        ans = str(ans) if ans is not None else ""
        target_ans = ans.split("####")[1].strip() if "####" in ans else ans.strip()
        numbers = re.findall(r'-?\d+\.?\d*', model_answer.replace(',', ''))
        model_last_num = numbers[-1] if numbers else ""
        target_numbers = re.findall(r'-?\d+\.?\d*', target_ans.replace(',', ''))
        target_last_num = target_numbers[-1] if target_numbers else target_ans
        if model_answer == target_ans or (model_last_num and target_last_num and model_last_num == target_last_num):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def format_number_reward(completions, **kwargs) -> list[float]:
    return [0.5 if re.findall(r'-?\d+\.?\d*', r["response"].replace(',', '')) else 0.0 for r in parse_responses(completions)]

def format_reasoning_reward(completions, **kwargs) -> list[float]:
    return [0.5 if r["thinking_content"] and r["response"] else 0.0 for r in parse_responses(completions)]


# =========================================================================
# Main
# =========================================================================

def sanitize_model_name(model_name: str) -> str:
    sanitized = model_name.replace("/", "_").replace("-", "_").lower()
    return "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized).strip("_")


def train(
    model_name, n_steps, batch_size, learning_rate, subset_size,
    run_name, lora_rank, lora_alpha, lora_target_modules, lora_dropout,
    use_wandb, grpo_beta, warmup_steps, dataset_key,
    output_base_dir, dataset_cache_dir, num_generations,
    generation_batch_size, grad_accum, optimizer_type,
):
    os.environ["HF_DATASETS_CACHE"] = dataset_cache_dir
    ds_config = get_dataset_config(dataset_key)
    dataset_sanitized = ds_config["sanitized_name"]

    if run_name is None:
        run_name = f"lora_grpo_r{lora_rank}_{optimizer_type}_{sanitize_model_name(model_name)}_{dataset_sanitized}"

    if use_wandb:
        os.environ["WANDB_PROJECT"] = "huggingface"

    run_dir = os.path.join(output_base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*60}\nLoRA GRPO TRAINING (Timing Benchmark)\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print(f"Target modules: {lora_target_modules}")
    print(f"Optimizer: {optimizer_type}")
    print(f"Dataset: {dataset_key}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = load_grpo_dataset(dataset_key, subset_size=subset_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.config.use_cache = False

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
        beta=grpo_beta,
        warmup_steps=warmup_steps,
    )

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=train_dataset,
        reward_funcs=[accuracy_reward, format_number_reward, format_reasoning_reward],
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
        "method": "lora",
        "optimizer": optimizer_type,
        "model": model_name,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
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
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="math-220k")
    parser.add_argument("--output_base_dir", type=str, default="/scratch/xie.yiyi/rl_casino_outputs")
    parser.add_argument("--dataset_cache_dir", type=str, default="/scratch/xie.yiyi/hf_cache/datasets")
    parser.add_argument("--grpo_beta", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=0)

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        subset_size=args.subset_size,
        run_name=args.run_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        use_wandb=args.use_wandb,
        grpo_beta=args.grpo_beta,
        warmup_steps=args.warmup_steps,
        dataset_key=args.dataset,
        output_base_dir=args.output_base_dir,
        dataset_cache_dir=args.dataset_cache_dir,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,
        grad_accum=args.grad_accum,
        optimizer_type=args.optimizer,
    )
