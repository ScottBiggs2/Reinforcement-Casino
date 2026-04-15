#!/usr/bin/env python3
"""
H200-oriented multi-phase BSR DPO benchmark: one dense + several random-mask sparse runs.

Dataset and tokenizer are loaded once; each phase reloads the base model from HF for fair timing.
CSV: <output_dir>/benchmark_training_log.csv (via BenchmarkRunLogSink + BenchmarkThroughputCallback).
"""

import argparse
import gc
import os
import sys
import tempfile
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.full_training.sparse_dpo_bsr import train as sparse_dpo_train
from src.utils.dataset_registry import load_dpo_dataset as registry_load_dpo
from src.utils.logging_utils import BenchmarkRunLogSink
from src.utils.mask_utils import (
    DEFAULT_MIN_LAYER_KEEP_RATIO,
    create_mask_from_scores_gpu_efficient,
    save_masks,
)


def _build_random_scores_for_masks(model: torch.nn.Module, mlp_only: bool) -> dict:
    scores = {}
    for name, p in model.named_parameters():
        if "weight" not in name or p.dim() != 2:
            continue
        if mlp_only and "mlp" not in name.lower():
            continue
        scores[name] = torch.rand(p.shape, dtype=torch.float32)
    return scores


def generate_random_masks_cpu(
    model_name: str,
    checkpoint_path: Optional[str],
    sparsity_percent: float,
    seed: int,
    mlp_only: bool,
    min_layer_keep_ratio: float,
) -> dict:
    """Load model on CPU, build global random masks, delete model."""
    ckpt = checkpoint_path if checkpoint_path and str(checkpoint_path).lower() != "none" else model_name
    torch.manual_seed(seed)
    print(f"  Generating random mask (target sparsity {sparsity_percent}%, seed={seed}) on CPU...")
    m = AutoModelForCausalLM.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    scores = _build_random_scores_for_masks(m, mlp_only)
    del m
    gc.collect()

    masks = create_mask_from_scores_gpu_efficient(
        scores,
        sparsity_percent,
        device="cpu",
        min_layer_keep_ratio=min_layer_keep_ratio,
        local_pool=False,
        add_tie_break_noise=True,
    )
    return masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--dataset", type=str, default="tulu3")
    p.add_argument("--subset_size", type=int, default=None)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--dpo_beta", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--block_size_bsr", type=int, default=16)
    p.add_argument("--block_size_adam", type=int, default=128)
    p.add_argument("--mlp_only", action="store_true")
    p.add_argument("--min_layer_keep_ratio", type=float, default=DEFAULT_MIN_LAYER_KEEP_RATIO)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--disable_tf32", action="store_true")
    p.add_argument("--device_map", type=str, default="none", help="HF device_map (none = single GPU .to)")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dataset_cache_dir", type=str, default=None)
    p.add_argument("--run_label", type=str, default="h200_bsr_bench")
    args = p.parse_args()

    print(f"Benchmark config: n_steps per phase={args.n_steps}, batch={args.batch_size}, grad_accum={args.grad_accum}")

    os.environ["HF_DATASETS_CACHE"] = args.dataset_cache_dir or os.environ.get(
        "HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets")
    )
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "benchmark_training_log.csv")
    if os.path.isfile(csv_path):
        os.remove(csv_path)

    sink = BenchmarkRunLogSink(csv_path)

    print("Loading tokenizer and dataset (once)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dpo_ds = registry_load_dpo(args.dataset, subset_size=args.subset_size)

    phases = [
        ("dense", None, "adamw"),
        ("sparse_90", 90.0, "sparse_adamw"),
        ("sparse_95", 95.0, "sparse_adamw"),
        ("sparse_97_5", 97.5, "sparse_adamw"),
        ("sparse_99", 99.0, "sparse_adamw"),
        ("sparse_99_75", 99.75, "sparse_adamw"),
    ]

    gc_flag = not args.no_gradient_checkpointing
    dm = args.device_map
    if dm.lower() in ("none", "null"):
        dm = "none"

    for idx, (phase_name, sparsity_pct, opt) in enumerate(phases):
        print(f"\n{'#'*60}\nPHASE {phase_name}  sparsity={sparsity_pct}  optimizer={opt}\n{'#'*60}\n")
        mask_path = None
        tmp_mask = None
        seed = 424200 + idx * 31 + (int(sparsity_pct * 10) if sparsity_pct is not None else 0)

        if sparsity_pct is not None:
            masks = generate_random_masks_cpu(
                args.model_name,
                args.checkpoint,
                sparsity_pct,
                seed=seed,
                mlp_only=args.mlp_only,
                min_layer_keep_ratio=args.min_layer_keep_ratio,
            )
            fd, tmp_mask = tempfile.mkstemp(suffix=".pt", prefix="mask_")
            os.close(fd)
            meta = {
                "method": "random_global_benchmark",
                "sparsity_percent": sparsity_pct,
                "seed": seed,
                "format": "torch_bool_binary",
            }
            bool_masks = {k: v.bool() if v.dtype != torch.bool else v for k, v in masks.items()}
            save_masks(bool_masks, tmp_mask, metadata=meta)
            mask_path = tmp_mask
            del masks
            gc.collect()

        run_name = f"{args.run_label}_{phase_name}"

        try:
            sparse_dpo_train(
                model_name=args.model_name,
                checkpoint_path=args.checkpoint,
                mask_path=mask_path,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                subset_size=args.subset_size,
                run_name=run_name,
                mlp_only=args.mlp_only,
                block_size_bsr=args.block_size_bsr,
                block_size_adam=args.block_size_adam,
                optimizer_type=opt,
                use_wandb=False,
                save_csv=False,
                grad_accum=args.grad_accum,
                max_grad_norm=args.max_grad_norm,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                dpo_beta=args.dpo_beta,
                warmup_steps=0,
                warmup_ratio=args.warmup_ratio,
                weight_decay=args.weight_decay,
                max_length=args.max_length,
                max_prompt_length=args.max_prompt_length,
                disable_tf32=args.disable_tf32,
                save_model=False,
                dataset_key=args.dataset,
                output_base_dir=args.output_dir,
                dataset_cache_dir=args.dataset_cache_dir or os.environ["HF_DATASETS_CACHE"],
                dense_baseline=(phase_name == "dense"),
                no_delta_callback=True,
                lr_scheduler_type="linear",
                num_train_epochs=1,
                gradient_checkpointing=gc_flag,
                save_strategy="no",
                report_to="none",
                device_map=dm,
                train_dataset=dpo_ds,
                tokenizer_obj=tokenizer,
                benchmark_log_sink=sink,
                benchmark_phase=phase_name,
                benchmark_sparsity_pct=sparsity_pct,
                benchmark_optimizer_label=opt,
            )
        finally:
            try:
                sink.flush()
            except OSError as e:
                print(f"WARNING: could not flush benchmark CSV (disk/NFS issue): {e}", file=sys.stderr)
            if tmp_mask and os.path.isfile(tmp_mask):
                try:
                    os.unlink(tmp_mask)
                except OSError:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        sink.close()
    except OSError as e:
        print(f"WARNING: final CSV flush failed: {e}", file=sys.stderr)
    print(f"\n✓ Benchmark complete. CSV: {csv_path}")


if __name__ == "__main__":
    main()
