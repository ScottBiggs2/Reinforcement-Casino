#!/usr/bin/env python3
"""
Baseline Dense GRPO Training - Timing Comparison Version

Stripped down version for clean performance comparison.
No logging, no checkpointing - just pure training time measurement.

Run: python GRPO_timing_baseline.py --n_steps 10
"""

import os
import sys
import json
import time
import torch
import argparse
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOTrainer, GRPOConfig
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.dataset_registry import get_dataset_config, load_grpo_dataset
from src.utils.grpo_rewards import GRPO_REWARD_FUNCS

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "google/gemma-3-270m-it"
DATASET_KEY_DEFAULT = "math-220k"
SUBSET_SIZE = 10


# ============================================================================
# DATASET COLLATOR
# ============================================================================

def grpo_collator_fn(examples: List[Dict[str, Any]], tokenizer) -> Dict[str, torch.Tensor]:
    """Data collator for GRPO training."""
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
        return {
            "prompt_input_ids": p_ids,
            "prompt_attention_mask": p_mask,
        }

    prompts = [ex.get("prompt", "") for ex in examples]

    enc_prompt = [tokenizer(p, truncation=True, max_length=512, return_tensors="pt") for p in prompts]
    batch_prompt = tokenizer.pad(enc_prompt, padding=True, return_tensors="pt", pad_to_multiple_of=8)

    for k in ("input_ids", "attention_mask"):
        batch_prompt[k] = batch_prompt[k].to(torch.long)

    return {
        "prompt_input_ids": batch_prompt["input_ids"],
        "prompt_attention_mask": batch_prompt["attention_mask"],
    }



# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_baseline(
    checkpoint_path=None,
    n_steps=10,
    batch_size=1,
    learning_rate=5e-5,
    subset_size=10,
    dataset_key=DATASET_KEY_DEFAULT,
    dataset_cache_dir=None,
):
    """Baseline dense GRPO training with timing measurement."""
    
    if dataset_cache_dir:
        os.environ["HF_DATASETS_CACHE"] = dataset_cache_dir
    ds_config = get_dataset_config(dataset_key)
    dataset_hf_id = ds_config["hf_id"]

    model_path = checkpoint_path if checkpoint_path else MODEL_NAME
    
    print(f"\n{'='*60}")
    print(f"BASELINE DENSE GRPO TRAINING")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_key} ({dataset_hf_id})")
    print(f"Steps: {n_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✓ Tokenizer loaded\n")

    # Load dataset
    print("Loading dataset...")
    dataset = load_grpo_dataset(dataset_key, subset_size=subset_size)
    print("✓ Dataset loaded\n")
    
    def collator(examples):
        return grpo_collator_fn(examples, tokenizer)

    # Load model
    print(f"Loading model from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    if not torch.cuda.is_available() or model.device.type == 'cpu':
        model.to(device)
    
    model.config.use_cache = False
    print(f"✓ Model loaded on {device} with dtype: {model.dtype}\n")

    # Configure GRPO
    grpo_config = GRPOConfig(
        output_dir="./baseline_temp",
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=n_steps,
        logging_steps=n_steps + 1,  # Disable logging
        report_to=["wandb"],
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        beta=0.1,  # KL penalty coefficient
        max_completion_length=1024,
        max_prompt_length=512,
        num_generations=8,  # Number of generations per prompt
        generation_batch_size=8,  # Batch size for generation
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        save_steps=n_steps + 1,  # Disable saving
    )

    # Snapshot initial params θ(0)
    base_state = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            base_state[name] = param.detach().float().cpu().clone()

    # Subnet logging callback
    class SubnetLoggingCallback(TrainerCallback):
        """Callback that logs subnet statistics to wandb."""
        
        def __init__(self, base_state, threshold=1e-3):
            self.base_state = base_state
            self.threshold = threshold
            self.wandb_initialized = False

        def on_train_begin(self, args, state, control, **kwargs):
            if not self.wandb_initialized:
                wandb.init(
                    project="grpo-timing-baseline",
                    name=args.run_name if hasattr(args, 'run_name') else "grpo_timing",
                    config={
                        "model_name": MODEL_NAME,
                        "dataset_key": dataset_key,
                        "dataset": dataset_hf_id,
                        "subset_size": subset_size,
                        "learning_rate": args.learning_rate,
                        "batch_size_per_device": args.per_device_train_batch_size,
                    },
                )
                self.wandb_initialized = True

        def on_step_end(self, args, state, control, **kwargs):
            model = kwargs["model"]
            step = state.global_step

            layer_stats = {}

            with torch.no_grad():
                for name, param in model.named_parameters():
                    current = param.detach().float().cpu()
                    diff = current - self.base_state[name]

                    l2 = torch.norm(diff).item()
                    frac_big = (diff.abs() > self.threshold).float().mean().item()

                    layer_stats[name] = {
                        "l2_from_init": l2,
                        "frac_big_from_init": frac_big,
                    }

            # Aggregate summaries for wandb
            all_l2 = [v["l2_from_init"] for v in layer_stats.values()]
            all_frac = [v["frac_big_from_init"] for v in layer_stats.values()]
            mean_l2 = sum(all_l2) / len(all_l2)
            mean_frac = sum(all_frac) / len(all_frac)

            attn_l2 = []
            mlp_l2 = []
            for n, st in layer_stats.items():
                low = n.lower()
                if "attn" in low or "q_proj" in low or "k_proj" in low or "v_proj" in low or "o_proj" in low:
                    attn_l2.append(st["l2_from_init"])
                if "mlp" in low or "ffn" in low or "feed_forward" in low or "gate_proj" in low or "up_proj" in low or "down_proj" in low:
                    mlp_l2.append(st["l2_from_init"])

            wandb.log({
                "step": step,
                "subnet/mean_l2_from_init": mean_l2,
                "subnet/mean_frac_big_from_init": mean_frac,
                "subnet/attn_mean_l2": (sum(attn_l2)/len(attn_l2)) if attn_l2 else 0.0,
                "subnet/mlp_mean_l2": (sum(mlp_l2)/len(mlp_l2)) if mlp_l2 else 0.0,
            }, step=step)

            return control

        def on_train_end(self, args, state, control, **kwargs):
            if self.wandb_initialized:
                wandb.finish()

    # Initialize trainer
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        eval_dataset=None,
        reward_funcs=GRPO_REWARD_FUNCS,
        data_collator=collator,
        processing_class=tokenizer,
    )
    
    # Add subnet logging callback
    trainer.add_callback(SubnetLoggingCallback(base_state=base_state, threshold=1e-3))
    
    print("✓ Trainer ready\n")

    # Train with timing
    print(f"{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")
    
    # Use CUDA events for GPU timing
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
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Wall clock time: {wall_time:.2f}s")
    if torch.cuda.is_available():
        print(f"GPU time: {gpu_time:.2f}s")
        print(f"Time per step (GPU): {gpu_time/n_steps:.2f}s")
    print(f"Time per step (wall): {wall_time/n_steps:.2f}s")
    print(f"{'='*60}\n")
    
    # Save timing results
    results = {
        'wall_time': wall_time,
        'time_per_step_wall': wall_time / n_steps,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }
    
    if torch.cuda.is_available():
        results['gpu_time'] = gpu_time
        results['time_per_step_gpu'] = gpu_time / n_steps
    
    with open('baseline_grpo_timing.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Timing results saved to baseline_grpo_timing.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline dense GRPO training for timing comparison")
    
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint (default: None = use base model)")
    parser.add_argument("--n_steps", type=int, default=10,
                       help="Number of training steps (default: 10)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (default: 1)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate (default: 5e-5)")
    parser.add_argument("--subset_size", type=int, default=10,
                       help="Dataset subset size (default: 10)")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_KEY_DEFAULT,
        help=f"Dataset registry key (default: {DATASET_KEY_DEFAULT})",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=None,
        help="If set, sets HF_DATASETS_CACHE before loading the dataset",
    )
    
    args = parser.parse_args()
    
    train_baseline(
        checkpoint_path=args.checkpoint,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        subset_size=args.subset_size,
        dataset_key=args.dataset,
        dataset_cache_dir=args.dataset_cache_dir,
    )
