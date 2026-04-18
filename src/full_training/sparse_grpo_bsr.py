#!/usr/bin/env python3
"""
Triton-Accelerated Sparse GRPO Training - Math Focused

BSR sparse MLP + indexed SparseAdamW; checkpoint/resume; DDP-safe multi-GPU (torchrun).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils.trl_vllm_import_guard import apply_trl_vllm_skip

apply_trl_vllm_skip()

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from src.mlps.bsr_sparse_mlp import replace_linear_modules, restore_linear_modules
from src.optimizers.sparse_adamw import SparseAdamW
from src.utils.dataset_registry import get_dataset_config, load_grpo_dataset
from src.utils.grpo_checkpoint_utils import (
    RunManifestCallback,
    WandbRunIdCallback,
    maybe_load_wandb_resume_env,
    resolve_resume_checkpoint,
)
from src.utils.grpo_rewards import GRPO_REWARD_FUNCS
from src.utils.mask_manager import SparseMaskManager
from src.utils.model_slug import sanitize_model_name
from src.utils.scratch_paths import default_grpo_sparse_outputs, default_hf_datasets_cache
from src.utils.training_precision import resolve_grpo_precision


class SparseDeltaCheckpointCallback(TrainerCallback):
    """Optional weight-delta logs vs training start (for analysis)."""

    def __init__(
        self,
        base_state: Dict[str, torch.Tensor],
        delta_log_dir: str,
        checkpoint_schedule: Set[int],
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

        if step in self.checkpoint_schedule:
            delta_file = os.path.join(self.delta_log_dir, f"deltas_step_{step}.pt")
            torch.save(full_deltas_to_save, delta_file)
            print(f"  ✓ Saved sparse delta checkpoint at step {step}")

        return control


def train(
    model_name: str,
    checkpoint_path: Optional[str],
    mask_path: str,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    subset_size: Optional[int],
    run_name: Optional[str],
    mlp_only: bool,
    block_size_bsr: int,
    block_size_adam: int,
    optimizer_type: str,
    use_wandb: bool,
    max_grad_norm: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    grpo_beta: float,
    warmup_steps: int,
    disable_tf32: bool,
    save_model: bool,
    dataset_key: str,
    output_base_dir: str,
    dataset_cache_dir: str,
    num_generations: int,
    generation_batch_size: int,
    grad_accum: int,
    save_steps: int,
    save_total_limit: int,
    resume_from_checkpoint: Optional[str],
    max_prompt_length: int,
    max_completion_length: int,
    no_gradient_checkpointing: bool,
    delta_log_interval: Optional[int],
    delta_log_end_step: Optional[int],
    precision: str = "auto",
) -> None:
    if checkpoint_path is None or str(checkpoint_path).lower() == "none":
        checkpoint_path = model_name

    os.environ.setdefault("WANDB_CONSOLE", "off")
    os.environ["HF_DATASETS_CACHE"] = dataset_cache_dir
    ds_config = get_dataset_config(dataset_key)
    dataset_name = ds_config["hf_id"]
    dataset_sanitized = ds_config["sanitized_name"]

    if run_name is None:
        parts = ["sparse_grpo"]
        parts.append("bsr_adamw" if optimizer_type == "sparse_adamw" else optimizer_type)
        parts.append(sanitize_model_name(model_name))
        parts.append(dataset_sanitized)
        run_name = "_".join(parts)

    wandb_project = os.environ.get("WANDB_PROJECT", "huggingface")
    if use_wandb:
        os.environ["WANDB_PROJECT"] = wandb_project

    run_dir = os.path.join(output_base_dir, run_name)
    output_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    resume_ckpt = resolve_resume_checkpoint(output_dir, resume_from_checkpoint)
    if use_wandb:
        maybe_load_wandb_resume_env(run_dir, resume_ckpt)

    print(f"\n{'=' * 60}\nSPARSE GRPO MATH TRAINING\n{'=' * 60}")
    print(f"Dataset: {dataset_key} ({dataset_name})")
    print(f"run_dir={run_dir}\nresume={resume_ckpt!r}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = load_grpo_dataset(dataset_key, subset_size=subset_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    multi_gpu = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    pm = precision if precision in ("auto", "bf16", "fp16") else "auto"
    use_bf16, use_fp16, model_dtype = resolve_grpo_precision(pm)
    print(
        f"Precision mode={pm} → Trainer bf16={use_bf16} fp16={use_fp16}, "
        f"model load dtype={model_dtype}"
    )

    # Single-process training: avoid device_map="auto" (shards across devices). Trainer expects a
    # conventional single-device model; "already on multiple devices" + pin_memory warnings are common otherwise.
    load_kw: Dict[str, Any] = {
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True,
        "device_map": None,
    }

    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **load_kw)
    if torch.cuda.is_available():
        model.to(device)
    model.config.use_cache = False

    mask_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    mask_manager = SparseMaskManager(mask_path, device=mask_device)
    mask_dict = {
        n: mask_manager.get_mask(n)
        for n, _ in model.named_parameters()
        if ("mlp" in n.lower() or not mlp_only) and "weight" in n and mask_manager.has_mask(n)
    }

    print(f"Injecting Sparse BSR layers for {len(mask_dict)} layers...")
    use_tf32_kernel = not disable_tf32
    replace_linear_modules(model, mask_dict, block_size=block_size_bsr, use_tf32=use_tf32_kernel)

    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2), eps=adam_eps
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
            max_grad_norm=max_grad_norm,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    if disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    base_state: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            base_state[name] = param.detach().float().cpu().clone()

    manifest = {
        "model_name": model_name,
        "checkpoint": checkpoint_path,
        "mask_path": mask_path,
        "dataset": dataset_name,
        "dataset_key": dataset_key,
        "n_steps": n_steps,
        "learning_rate": learning_rate,
        "grpo_beta": grpo_beta,
        "optimizer": optimizer_type,
        "precision": pm,
        "run_dir": run_dir,
        "output_dir": output_dir,
        "resume_from_checkpoint": resume_ckpt,
    }

    callbacks: List[TrainerCallback] = [
        RunManifestCallback(run_dir, manifest),
        WandbRunIdCallback(run_dir),
    ]
    interval = delta_log_interval
    if interval is not None and interval > 0:
        if resume_ckpt:
            print(
                "WARNING: --delta_log_interval ignored when resuming (base_state would not match). "
                "Disable resume or omit delta logging."
            )
        else:
            end = delta_log_end_step
            if end is None:
                end = min(n_steps, max(interval, n_steps // 10))
            else:
                end = min(n_steps, end)
            sched_list = list(range(interval, end + 1, interval))
            if not sched_list and n_steps > 0:
                sched_list = [n_steps]
            delta_dir = os.path.join(run_dir, "deltas")
            os.makedirs(delta_dir, exist_ok=True)
            torch.save(base_state, os.path.join(delta_dir, "base_state.pt"))
            callbacks.append(
                SparseDeltaCheckpointCallback(
                    base_state=base_state,
                    delta_log_dir=delta_dir,
                    checkpoint_schedule=set(sched_list),
                    threshold=1e-3,
                )
            )

    cfg = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        report_to=["wandb"] if use_wandb else [],
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        max_steps=n_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=not no_gradient_checkpointing,
        dataloader_pin_memory=torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
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
        callbacks=callbacks,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = end_event = None

    wall_start = time.time()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    wall_time = time.time() - wall_start

    if torch.cuda.is_available() and start_event is not None:
        end_event.record()
        torch.cuda.synchronize()
        gpu_time = start_event.elapsed_time(end_event) / 1000.0
    else:
        gpu_time = None

    timing_results: Dict[str, Any] = {
        "method": "sparse_bsr",
        "precision": pm,
        "optimizer": optimizer_type,
        "model": model_name,
        "block_size_bsr": block_size_bsr,
        "block_size_adam": block_size_adam,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "wall_time": wall_time,
        "time_per_step_wall": wall_time / max(n_steps, 1),
    }
    if gpu_time is not None:
        timing_results["gpu_time"] = gpu_time
        timing_results["time_per_step_gpu"] = gpu_time / max(n_steps, 1)

    timing_path = os.path.join(run_dir, "timing_results.json")
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_results, f, indent=2)
    print(f"\n{'=' * 60}")
    print(f"Wall time: {wall_time:.2f}s | Per step: {wall_time / max(n_steps, 1):.2f}s")
    if gpu_time is not None:
        print(f"GPU time:  {gpu_time:.2f}s | Per step: {gpu_time / max(n_steps, 1):.2f}s")
    print(f"Timing saved to {timing_path}")
    print(f"{'=' * 60}")

    if save_model:
        print(f"\nTraining complete. Saving final model to {run_dir}/final_model...")
        restore_linear_modules(model)
        final_save_dir = os.path.join(run_dir, "final_model")
        os.makedirs(final_save_dir, exist_ok=True)
        trainer.save_model(final_save_dir)
        tokenizer.save_pretrained(final_save_dir)
        print(f"✓ Full checkpoint saved to {final_save_dir}")

    if use_wandb and wandb.run is not None:
        wandb.finish()


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
    parser.add_argument("--output_base_dir", type=str, default=default_grpo_sparse_outputs())
    parser.add_argument("--dataset_cache_dir", type=str, default=default_hf_datasets_cache())

    def str2bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        if v.lower() in ("no", "false", "f", "n", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    parser.add_argument("--save_model", type=str2bool, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--grpo_beta", type=float, default=0.025)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Checkpoint dir or 'auto' for latest under run_dir/checkpoints.",
    )
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--delta_log_interval",
        type=int,
        default=None,
        help="If set, save weight deltas vs init on this interval (skipped when resuming).",
    )
    parser.add_argument("--delta_log_end_step", type=int, default=None)
    parser.add_argument(
        "--precision",
        type=str,
        choices=["auto", "bf16", "fp16"],
        default="auto",
        help="Training AMP: auto uses bf16 when the GPU supports it (Transformers check), else fp16. "
        "Use fp16 on V100 / GPUs that raise 'Your setup doesn't support bf16/gpu'.",
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
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        no_gradient_checkpointing=args.no_gradient_checkpointing,
        delta_log_interval=args.delta_log_interval,
        delta_log_end_step=args.delta_log_end_step,
        precision=args.precision,
    )
