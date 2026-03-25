#!/usr/bin/env python3
"""
Dense Baseline GRPO Training - Math Focused

Used to collect delta statistics for sparse mask generation.
"""

import os
import json
import argparse
import re
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOTrainer, GRPOConfig
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.dataset_registry import get_dataset_config, load_grpo_dataset

def sanitize_model_name(model_name: str) -> str:
    sanitized = model_name.replace("/", "_").replace("-", "_").lower()
    return "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized).strip("_")

parser = argparse.ArgumentParser(description="Dense Baseline GRPO Training Script")
parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
parser.add_argument("--dataset", type=str, default="open-r1/OpenR1-Math-220k")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--subset_size", type=int, default=None)
parser.add_argument("--output_base_dir", type=str, default="/scratch/biggs.s/rl_casino_outputs")
parser.add_argument("--dataset_cache_dir", type=str, default="/scratch/biggs.s/hf_cache/datasets")
parser.add_argument("--num_generations", type=int, default=8)
parser.add_argument("--generation_batch_size", type=int, default=8)
args = parser.parse_args()

os.environ["HF_DATASETS_CACHE"] = args.dataset_cache_dir

MODEL_NAME = args.model_name
MODEL_NAME_SANITIZED = sanitize_model_name(MODEL_NAME)

DATASET_KEY = args.dataset
DATASET_CONFIG = get_dataset_config(DATASET_KEY)
DATASET_NAME = DATASET_CONFIG["hf_id"]
DATASET_SANITIZED = DATASET_CONFIG["sanitized_name"]

BASE_DIR = args.output_base_dir
SUB_DIR = f"{MODEL_NAME_SANITIZED}_{DATASET_SANITIZED}_grpo_dense"

OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints", SUB_DIR)
DELTA_LOG_DIR = os.path.join(BASE_DIR, "deltas", SUB_DIR)

CHECKPOINT_SCHEDULE = (
    list(range(10, 51, 10)) +  # [10, 20, 30, 40, 50]
    list(range(100, 501, 100))  # [100, 200, 300, 400, 500]
)

THRESHOLD = 1e-5
NUM_STEPS = args.num_steps
SUBSET_SIZE = args.subset_size

WANDB_PROJECT = "huggingface"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
WANDB_RUN_NAME = args.run_name if args.run_name else f"{MODEL_NAME_SANITIZED}_{DATASET_SANITIZED}_grpo_{NUM_STEPS}steps"

# =========================================================================
# Math Reward Functions
# =========================================================================
def parse_reasoning_response(text: str) -> dict:
    pattern = r"<think>\s*(.*?)\s*</think>\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if not match: return {"thinking_content": "", "response": text}
    return {"thinking_content": match.group(1).strip(), "response": match.group(2).strip()}

def get_completion_content(completion) -> str:
    if isinstance(completion, list): return " ".join(msg.get("content", "") if isinstance(msg, dict) else str(msg) for msg in completion)
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
        rewards.append(1.0 if model_answer == target_ans or (model_last_num and target_last_num and model_last_num == target_last_num) else 0.0)
    return rewards

def format_number_reward(completions, **kwargs) -> list[float]:
    return [0.5 if re.findall(r'-?\d+\.?\d*', r["response"].replace(',', '')) else 0.0 for r in parse_responses(completions)]

def format_reasoning_reward(completions, **kwargs) -> list[float]:
    return [0.5 if r["thinking_content"] and r["response"] else 0.0 for r in parse_responses(completions)]

# =========================================================================
# Initialization
# =========================================================================
train_dataset = load_grpo_dataset(DATASET_KEY, subset_size=SUBSET_SIZE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
model.config.use_cache = False

cfg = GRPOConfig(
    output_dir=OUTPUT_DIR,
    run_name=WANDB_RUN_NAME,
    report_to=["wandb"] if args.use_wandb else [],
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    max_steps=NUM_STEPS,
    num_train_epochs=1,
    bf16=True, fp16=False,
    logging_steps=1,
    save_steps=999999,
    remove_unused_columns=False,
    num_generations=args.num_generations,
    generation_batch_size=args.generation_batch_size,
    max_length=1024,
    max_prompt_length=512,
    beta=0.1,
)

def grpo_collator_fn(examples):
    prompts = [ex.get("prompt", "") for ex in examples]
    enc_prompt = [tokenizer(p, truncation=True, max_length=512, return_tensors="pt") for p in prompts]
    batch_prompt = tokenizer.pad(enc_prompt, padding=True, return_tensors="pt", pad_to_multiple_of=8)
    for k in ("input_ids", "attention_mask"): batch_prompt[k] = batch_prompt[k].to(torch.long)
    return {"prompt_input_ids": batch_prompt["input_ids"], "prompt_attention_mask": batch_prompt["attention_mask"]}

trainer = GRPOTrainer(
    model=model,
    args=cfg,
    train_dataset=train_dataset,
    reward_funcs=[accuracy_reward, format_number_reward, format_reasoning_reward],
    data_collator=grpo_collator_fn,
    processing_class=tokenizer,
)

base_state = {}
with torch.no_grad():
    for name, param in trainer.model.named_parameters():
        base_state[name] = param.detach().float().cpu().clone()

os.makedirs(DELTA_LOG_DIR, exist_ok=True)
torch.save(base_state, os.path.join(DELTA_LOG_DIR, "base_state.pt"))

class FlexibleCheckpointCallback(TrainerCallback):
    def __init__(self, base_state, delta_log_dir, checkpoint_schedule, threshold):
        self.base_state = base_state
        self.delta_log_dir = delta_log_dir
        self.checkpoint_schedule = set(checkpoint_schedule)
        self.threshold = threshold
        os.makedirs(self.delta_log_dir, exist_ok=True)
        self.wandb_initialized = False

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.wandb_initialized and "wandb" in args.report_to:
            wandb.init(
                project=WANDB_PROJECT,
                name=args.run_name,
                config={
                    "model_name": MODEL_NAME,
                    "dataset": DATASET_NAME,
                    "subset_size": SUBSET_SIZE,
                    "learning_rate": args.learning_rate,
                    "batch_size_per_device": args.per_device_train_batch_size,
                    "grad_accum": args.gradient_accumulation_steps,
                    "checkpoint_schedule": sorted(list(self.checkpoint_schedule)),
                },
            )
            self.wandb_initialized = True

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

        stats_path = os.path.join(self.delta_log_dir, f"stats_step_{step}.json")
        with open(stats_path, "w") as f: json.dump(layer_stats, f)

        all_l2 = [v["l2_from_init"] for v in layer_stats.values()]
        all_frac = [v["frac_big_from_init"] for v in layer_stats.values()]
        mean_l2 = sum(all_l2) / len(all_l2) if all_l2 else 0.0
        mean_frac = sum(all_frac) / len(all_frac) if all_frac else 0.0

        attn_l2, mlp_l2 = [], []
        for n, st in layer_stats.items():
            low = n.lower()
            if any(x in low for x in ["attn", "q_proj", "k_proj", "v_proj", "o_proj"]): attn_l2.append(st["l2_from_init"])
            if any(x in low for x in ["mlp", "ffn", "feed_forward", "gate_proj", "up_proj", "down_proj"]): mlp_l2.append(st["l2_from_init"])

        wandb.log({
            "step": step,
            "subnet/mean_l2_from_init": mean_l2,
            "subnet/mean_frac_big_from_init": mean_frac,
            "subnet/attn_mean_l2": (sum(attn_l2)/len(attn_l2)) if attn_l2 else 0.0,
            "subnet/mlp_mean_l2": (sum(mlp_l2)/len(mlp_l2)) if mlp_l2 else 0.0,
        }, step=step)

        if step in self.checkpoint_schedule:
            delta_file = os.path.join(self.delta_log_dir, f"deltas_step_{step}.pt")
            torch.save(full_deltas_to_save, delta_file)
            print(f"  ✓ Saved checkpoint at step {step}")

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.wandb_initialized: wandb.finish()

trainer.add_callback(FlexibleCheckpointCallback(base_state, DELTA_LOG_DIR, CHECKPOINT_SCHEDULE, THRESHOLD))

print(f"\n{'='*60}\nStarting DENSE GRPO training with delta tracking\n{'='*60}")
trainer.train()
