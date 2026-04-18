#!/usr/bin/env python3
"""
Dense Baseline GRPO Training - Math Focused

Full-rank bf16 weights; checkpoint/resume for long Slurm jobs; optional delta logs for masks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils.trl_vllm_import_guard import apply_trl_vllm_skip

apply_trl_vllm_skip()

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from src.utils.dataset_registry import get_dataset_config, load_grpo_dataset
from src.utils.grpo_checkpoint_utils import (
    RunManifestCallback,
    WandbRunIdCallback,
    maybe_load_wandb_resume_env,
    resolve_resume_checkpoint,
)
from src.utils.grpo_rewards import GRPO_REWARD_FUNCS
from src.utils.model_slug import sanitize_model_name
from src.utils.scratch_paths import default_grpo_dense_outputs, default_hf_datasets_cache
from src.utils.training_precision import resolve_grpo_precision


class DeltaLoggingCallback(TrainerCallback):
    """Expensive: full-param delta logs for warm-start masks. Off by default."""

    def __init__(
        self,
        base_state: Dict[str, torch.Tensor],
        delta_log_dir: str,
        checkpoint_schedule: set,
        threshold: float,
    ) -> None:
        self.base_state = base_state
        self.delta_log_dir = delta_log_dir
        self.checkpoint_schedule = checkpoint_schedule
        self.threshold = threshold
        os.makedirs(self.delta_log_dir, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs) -> Any:
        if not getattr(state, "is_world_process_zero", True):
            return control
        model = kwargs["model"]
        step = state.global_step
        layer_stats: Dict[str, Any] = {}
        full_deltas_to_save: Dict[str, torch.Tensor] = {}

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
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(layer_stats, f)

        all_l2 = [v["l2_from_init"] for v in layer_stats.values()]
        all_frac = [v["frac_big_from_init"] for v in layer_stats.values()]
        mean_l2 = sum(all_l2) / len(all_l2) if all_l2 else 0.0
        mean_frac = sum(all_frac) / len(all_frac) if all_frac else 0.0

        attn_l2, mlp_l2 = [], []
        for n, st in layer_stats.items():
            low = n.lower()
            if any(x in low for x in ("attn", "q_proj", "k_proj", "v_proj", "o_proj")):
                attn_l2.append(st["l2_from_init"])
            if any(
                x in low
                for x in ("mlp", "ffn", "feed_forward", "gate_proj", "up_proj", "down_proj")
            ):
                mlp_l2.append(st["l2_from_init"])

        if wandb.run is not None:
            wandb.log(
                {
                    "step": step,
                    "subnet/mean_l2_from_init": mean_l2,
                    "subnet/mean_frac_big_from_init": mean_frac,
                    "subnet/attn_mean_l2": (sum(attn_l2) / len(attn_l2)) if attn_l2 else 0.0,
                    "subnet/mlp_mean_l2": (sum(mlp_l2) / len(mlp_l2)) if mlp_l2 else 0.0,
                },
                step=step,
            )

        if step in self.checkpoint_schedule:
            delta_file = os.path.join(self.delta_log_dir, f"deltas_step_{step}.pt")
            torch.save(full_deltas_to_save, delta_file)
            print(f"  ✓ Saved delta checkpoint at step {step}")

        return control


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense GRPO training (full-rank, checkpoint/resume)")
    p.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="HF id or local path to load weights from (default: same as --model_name).",
    )
    p.add_argument("--dataset", type=str, default="math-220k")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument(
        "--run_slug",
        type=str,
        default=None,
        help="Subdirectory under output_base_dir for this run (default: model_dataset_grpo_dense).",
    )
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--subset_size", type=int, default=None)
    p.add_argument(
        "--output_base_dir",
        type=str,
        default=default_grpo_dense_outputs(),
        help="Parent directory; each run uses a subfolder (see --run_slug).",
    )
    p.add_argument("--dataset_cache_dir", type=str, default=default_hf_datasets_cache())
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--generation_batch_size", type=int, default=8)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.025, help="GRPO KL-style beta.")
    p.add_argument(
        "--optim",
        type=str,
        default="adamw_8bit",
        help="Trainer optimizer name (e.g. adamw_8bit, adamw_torch).",
    )
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_completion_length", type=int, default=1024)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint dir, or 'auto' for latest under output_dir.",
    )
    p.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (default: enabled).",
    )
    p.add_argument(
        "--delta_log_interval",
        type=int,
        default=None,
        help="If set, log weight deltas vs init every N steps (expensive). Default: off.",
    )
    p.add_argument(
        "--delta_log_end_step",
        type=int,
        default=None,
        help="Last step (inclusive) for delta logs when interval is set.",
    )
    p.add_argument(
        "--precision",
        type=str,
        choices=["auto", "bf16", "fp16"],
        default="auto",
        help="Training AMP: auto prefers bf16 when supported, else fp16 (e.g. V100).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_CONSOLE", "off")
    os.environ["HF_DATASETS_CACHE"] = args.dataset_cache_dir

    model_id = args.checkpoint or args.model_name
    model_sanitized = sanitize_model_name(args.model_name)
    dataset_key = args.dataset
    dataset_config = get_dataset_config(dataset_key)
    dataset_name = dataset_config["hf_id"]
    dataset_sanitized = dataset_config["sanitized_name"]

    run_slug = args.run_slug or f"{model_sanitized}_{dataset_sanitized}_grpo_dense"
    base_dir = args.output_base_dir
    run_dir = os.path.join(base_dir, run_slug)
    output_dir = os.path.join(run_dir, "checkpoints")
    delta_log_dir = os.path.join(run_dir, "deltas")
    os.makedirs(output_dir, exist_ok=True)

    resume_ckpt = resolve_resume_checkpoint(output_dir, args.resume_from_checkpoint)
    if args.use_wandb:
        maybe_load_wandb_resume_env(run_dir, resume_ckpt)

    num_steps = args.num_steps
    wandb_project = os.environ.get("WANDB_PROJECT", "huggingface")
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = wandb_project
    run_display_name = args.run_name or f"{model_sanitized}_{dataset_sanitized}_grpo_{num_steps}steps"

    train_dataset = load_grpo_dataset(dataset_key, subset_size=args.subset_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    multi_gpu = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    use_bf16, use_fp16, model_dtype = resolve_grpo_precision(args.precision)
    print(
        f"Precision mode={args.precision} → Trainer bf16={use_bf16} fp16={use_fp16}, "
        f"model load dtype={model_dtype}"
    )

    load_kw: Dict[str, Any] = {
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True,
    }
    if multi_gpu:
        load_kw["device_map"] = None
    else:
        load_kw["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
    model.config.use_cache = False
    if multi_gpu and torch.cuda.is_available():
        model.to(device)

    cfg = GRPOConfig(
        output_dir=output_dir,
        run_name=run_display_name,
        report_to=["wandb"] if args.use_wandb else [],
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        max_steps=num_steps,
        num_train_epochs=1,
        bf16=use_bf16,
        fp16=use_fp16,
        optim=args.optim,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=train_dataset,
        reward_funcs=GRPO_REWARD_FUNCS,
        processing_class=tokenizer,
    )

    manifest = {
        "model_name": args.model_name,
        "checkpoint": model_id,
        "dataset": dataset_name,
        "dataset_key": dataset_key,
        "subset_size": args.subset_size,
        "num_steps": num_steps,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "optim": args.optim,
        "precision": args.precision,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "run_dir": run_dir,
        "output_dir": output_dir,
        "resume_from_checkpoint": resume_ckpt,
    }
    trainer.add_callback(RunManifestCallback(run_dir, manifest))
    trainer.add_callback(WandbRunIdCallback(run_dir))

    interval = args.delta_log_interval
    if interval is not None and interval > 0:
        end = args.delta_log_end_step
        if end is None:
            end = min(num_steps, max(interval, num_steps // 10))
        else:
            end = min(num_steps, end)
        schedule = list(range(interval, end + 1, interval))
        if not schedule and num_steps > 0:
            schedule = [num_steps]
        base_state: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in trainer.model.named_parameters():
                base_state[name] = param.detach().float().cpu().clone()
        os.makedirs(delta_log_dir, exist_ok=True)
        torch.save(base_state, os.path.join(delta_log_dir, "base_state.pt"))
        trainer.add_callback(
            DeltaLoggingCallback(
                base_state=base_state,
                delta_log_dir=delta_log_dir,
                checkpoint_schedule=set(schedule),
                threshold=1e-5,
            )
        )
        print(f"Delta logging enabled: schedule steps {schedule[:5]}{'...' if len(schedule) > 5 else ''}")

    print(f"\n{'=' * 60}\nDense GRPO | run_dir={run_dir}\nresume={resume_ckpt!r}\n{'=' * 60}\n")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    if args.use_wandb and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
