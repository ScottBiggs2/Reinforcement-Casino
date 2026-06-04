"""
Gradient distribution diagnostic: single DPO backward pass, no optimizer step.

Reports the coefficient of variation (CV) of per-layer gradient magnitudes.

  High CV  → gradients concentrated in a few layers (sparse training viable)
  Low  CV  → gradients spread uniformly      (sparse training may not help)

Run via scripts/diag_h2_grad_distribution.slurm or directly:

  python src/diagnostics/grad_distribution_diag.py \
    --model_name Qwen/Qwen3-8B \
    --model_name2 meta-llama/Llama-3.1-8B-Instruct \
    --dataset light-r1 --n_samples 32

Outputs a JSON summary to --output_file (default: grad_diag_<model_sanitized>.json).
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from src.utils.dataset_registry import load_dpo_dataset
from src.utils.data_utils import dpo_collator_fn
from src.utils.scratch_paths import default_hf_datasets_cache


def _sanitize(s):
    return s.replace("/", "_").replace("-", "_").lower()


def run_single_model(model_name, dataset_key, n_samples, cache_dir, output_prefix):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_dpo_dataset(dataset_key, subset_size=n_samples)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    )
    model.config.use_cache = False

    # Use precompute_ref_log_probs so the reference model is freed before backward pass.
    cfg = DPOConfig(
        output_dir="/tmp/diag_grad",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=1,
        bf16=True,
        beta=0.1,
        max_length=1024,
        max_prompt_length=512,
        remove_unused_columns=False,
        report_to="none",
        logging_steps=1,
        precompute_ref_log_probs=True,
    )

    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        data_collator=lambda x: dpo_collator_fn(x, tokenizer),
    )

    # Get one batch and compute loss + gradients.
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))
    batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}

    model.zero_grad()
    loss = trainer.compute_loss(model, batch)
    loss.backward()

    # Collect per-layer gradient magnitudes (2D weight matrices only, deduplicated).
    seen_ptrs = set()
    layer_grads = {}
    for name, param in model.named_parameters():
        if param.grad is None or "weight" not in name or param.dim() != 2:
            continue
        ptr = param.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        layer_grads[name] = param.grad.detach().abs().mean().item()

    vals = np.array(list(layer_grads.values()), dtype=np.float64)
    cv = float(vals.std() / vals.mean()) if vals.mean() > 0 else 0.0

    # Split by layer type for budget analysis.
    buckets = {"embed_lmhead": 0.0, "attn": 0.0, "mlp": 0.0}
    counts  = {"embed_lmhead": 0,   "attn": 0,   "mlp": 0}
    for name, v in layer_grads.items():
        if "embed_tokens" in name or "lm_head" in name:
            buckets["embed_lmhead"] += v; counts["embed_lmhead"] += 1
        elif any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            buckets["attn"] += v; counts["attn"] += 1
        else:
            buckets["mlp"] += v; counts["mlp"] += 1

    print(f"\n  DPO loss (single batch): {loss.item():.6f}")
    print(f"  2D weight layers scored: {len(vals)}")
    print(f"  mean |∇w| across layers: {vals.mean():.3e}")
    print(f"  std  |∇w| across layers: {vals.std():.3e}")
    print(f"  CV   (higher=concentrated, lower=uniform): {cv:.4f}")
    print(f"\n  Mean |∇w| by group:")
    for g, s in buckets.items():
        c = counts[g]
        print(f"    {g:20s}: {s/c:.3e} over {c} layers" if c > 0 else f"    {g:20s}: (none)")

    top5 = sorted(layer_grads.items(), key=lambda x: -x[1])[:5]
    bot5 = sorted(layer_grads.items(), key=lambda x:  x[1])[:5]
    print(f"\n  Top-5 gradient layers:")
    for n, v in top5:
        print(f"    {v:.3e}  {n}")
    print(f"  Bot-5 gradient layers:")
    for n, v in bot5:
        print(f"    {v:.3e}  {n}")

    result = {
        "model": model_name,
        "dataset": dataset_key,
        "n_samples": n_samples,
        "dpo_loss": float(loss.item()),
        "n_layers": int(len(vals)),
        "mean_grad": float(vals.mean()),
        "std_grad":  float(vals.std()),
        "cv":        cv,
        "group_mean_grad": {g: float(s / counts[g]) if counts[g] > 0 else 0.0
                            for g, s in buckets.items()},
        "top5_layers": [(n, v) for n, v in top5],
        "bot5_layers": [(n, v) for n, v in bot5],
    }

    out_path = f"{output_prefix}_{_sanitize(model_name)}.json"
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Summary written to: {out_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  required=True,
                        help="Primary model to diagnose (e.g. Qwen/Qwen3-8B)")
    parser.add_argument("--model_name2", default=None,
                        help="Optional second model for comparison (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--dataset",     default="light-r1")
    parser.add_argument("--n_samples",   type=int, default=32,
                        help="Number of DPO examples for the backward pass (default 32)")
    parser.add_argument("--output_prefix", default="logs/grad_diag",
                        help="Prefix for output JSON files")
    parser.add_argument("--dataset_cache_dir", default=default_hf_datasets_cache())
    args = parser.parse_args()

    os.environ["HF_DATASETS_CACHE"] = args.dataset_cache_dir

    r1 = run_single_model(args.model_name,  args.dataset, args.n_samples,
                          args.dataset_cache_dir, args.output_prefix)

    if args.model_name2:
        r2 = run_single_model(args.model_name2, args.dataset, args.n_samples,
                              args.dataset_cache_dir, args.output_prefix)
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Model':<45} {'CV':>8}  {'mean|∇w|':>10}")
        print(f"  {args.model_name:<45} {r1['cv']:>8.4f}  {r1['mean_grad']:>10.3e}")
        print(f"  {args.model_name2:<45} {r2['cv']:>8.4f}  {r2['mean_grad']:>10.3e}")
        print(f"\n  Interpretation: higher CV = more concentrated gradients")
        print(f"  A model with higher CV is a better candidate for sparse DPO training.")


if __name__ == "__main__":
    main()
