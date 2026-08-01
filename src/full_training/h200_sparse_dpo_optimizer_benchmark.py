#!/usr/bin/env python3
"""
H200-oriented multi-phase DPO benchmark focused on **optimizer sparsity only**:

- Dense baseline phase: standard HF model + dense optimizer.
- Sparse phases: standard HF model + SparseAdamW (masked optimizer updates).

This intentionally does **not** inject BSR sparse backprop into nn.Linear modules.

Artifacts:
  - CSV: <output_dir>/benchmark_training_log.csv (BenchmarkRunLogSink + BenchmarkThroughputCallback)
  - Theory sidecar: <output_dir>/benchmark_theory.json (mask FLOP proxies; still useful for context)
  - Stdout: BENCH_JSON lines for mask gen/load timing and per-phase prepare vs trainer wall.
"""

import os
import sys
import argparse
import gc
import json
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.slurm_safe_log import slurm_safe_print
from src.full_training.sparse_dpo_efficiency import train as sparse_dpo_eff_train
from src.utils.dataset_registry import load_dpo_dataset as registry_load_dpo
from src.utils.logging_utils import BenchmarkRunLogSink
from src.utils.mask_utils import (
    DEFAULT_MIN_LAYER_KEEP_RATIO,
    create_mask_from_scores_gpu_efficient,
    save_masks,
)
from src.utils.bsr_theory_metrics import (
    compute_sparse_mask_theory_metrics,
    default_b_tokens_proxy,
    dense_phase_theory_stub,
)
from src.utils.block_profiler import print_block_sparsity_profile
from transformers import AutoModelForCausalLM, AutoTokenizer


def _bench_json(payload: Dict[str, Any]) -> None:
    slurm_safe_print("BENCH_JSON " + json.dumps(payload, separators=(",", ":"), sort_keys=True))


def _model_for_mask_shapes(ckpt: str):
    try:
        return AutoModelForCausalLM.from_pretrained(
            ckpt,
            torch_dtype=torch.bfloat16,
            device_map="meta",
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        slurm_safe_print(
            f"  WARNING: meta skeleton load failed ({exc}); using full CPU load for mask shapes (slow)."
        )
        return AutoModelForCausalLM.from_pretrained(
            ckpt,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )


def _build_random_scores_for_masks(model: torch.nn.Module, mlp_only: bool) -> dict:
    scores = {}
    for name, p in model.named_parameters():
        if "weight" not in name or p.dim() != 2:
            continue
        if mlp_only and "mlp" not in name.lower():
            continue
        scores[name] = torch.rand(tuple(p.shape), dtype=torch.float32, device="cpu")
    return scores


def generate_random_masks_cpu(
    model_name: str,
    checkpoint_path: Optional[str],
    sparsity_percent: float,
    seed: int,
    mlp_only: bool,
    min_layer_keep_ratio: float,
) -> dict:
    ckpt = checkpoint_path if checkpoint_path and str(checkpoint_path).lower() != "none" else model_name
    torch.manual_seed(seed)
    slurm_safe_print(
        f"  Generating random mask (target sparsity {sparsity_percent}%, seed={seed}) on CPU..."
    )
    m = _model_for_mask_shapes(ckpt)
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


def generate_block_random_masks_cpu(
    model_name: str,
    checkpoint_path: Optional[str],
    sparsity_percent: float,
    seed: int,
    mlp_only: bool,
    min_layer_keep_ratio: float,
    block_size: int = 16,
) -> dict:
    ckpt = checkpoint_path if checkpoint_path and str(checkpoint_path).lower() != "none" else model_name
    torch.manual_seed(seed)
    slurm_safe_print(
        f"  Generating BLOCK-STRUCTURED random mask (target sparsity {sparsity_percent}%, seed={seed}, block={block_size}) on CPU..."
    )
    m = _model_for_mask_shapes(ckpt)
    block_scores = {}
    original_shapes = {}
    for name, p in m.named_parameters():
        if "weight" not in name or p.dim() != 2:
            continue
        if mlp_only and "mlp" not in name.lower():
            continue
        M, N = p.shape
        original_shapes[name] = (M, N)
        blocks_m = (M + block_size - 1) // block_size
        blocks_n = (N + block_size - 1) // block_size
        block_scores[name] = torch.rand((blocks_m, blocks_n), dtype=torch.float32, device="cpu")
    del m
    gc.collect()
    block_masks = create_mask_from_scores_gpu_efficient(
        block_scores,
        sparsity_percent,
        device="cpu",
        min_layer_keep_ratio=min_layer_keep_ratio,
        local_pool=False,
        add_tie_break_noise=False,
    )
    final_masks = {}
    for name, b_mask in block_masks.items():
        expanded = b_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
        M, N = original_shapes[name]
        final_masks[name] = expanded[:M, :N]
    return final_masks


def _build_phases(
    sparsity_levels: List[float],
    *,
    dense_optimizers: Tuple[str, ...] = ("adamw_torch",),
    mask_types: Tuple[str, ...] = ("element",),
) -> List[Tuple]:
    phases: List[Tuple] = []
    for dopt in dense_optimizers:
        tag = "dense" if dopt == "adamw_torch" else f"dense_{dopt}"
        phases.append((tag, None, dopt, "none"))
    for sp in sparsity_levels:
        sp_tag = str(sp).replace(".", "p")
        for mask_type in mask_types:
            mt_short = "elem" if mask_type == "element" else "blk"
            phases.append((f"s{sp_tag}_{mt_short}", sp, "sparse_adamw", mask_type))
    return phases


def _stable_mask_seed(*, base: int, sparsity_pct: float, mask_type: str) -> int:
    h = hashlib.sha256(f"{sparsity_pct:.5f}|{mask_type}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) ^ int(base)


def _mask_cache_path(
    *,
    out_dir: str,
    sparsity_pct: float,
    mask_type: str,
    block_size: int,
    mlp_only: bool,
    min_layer_keep_ratio: float,
) -> str:
    tag = f"s{str(sparsity_pct).replace('.','p')}_{mask_type}_b{block_size}_mlp{int(mlp_only)}_floor{min_layer_keep_ratio:g}"
    return os.path.join(out_dir, "masks", f"{tag}.pt")


def _write_theory_sidecar(path: str, records: List[Dict[str, Any]]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as tf:
        json.dump(records, tf, indent=2)
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--dataset", type=str, default="tulu3")
    p.add_argument("--subset_size", type=int, default=None)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--dpo_beta", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--block_size_mask", type=int, default=16, help="Block size used for block masks only.")
    p.add_argument("--mlp_only", action="store_true")
    p.add_argument("--min_layer_keep_ratio", type=float, default=DEFAULT_MIN_LAYER_KEEP_RATIO)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dataset_cache_dir", type=str, default=None)
    p.add_argument("--run_label", type=str, default="h200_opt_bench")
    p.add_argument("--benchmark_sparsities", type=str, default="99.75,97.5,95,90")
    p.add_argument("--no_dense_baseline", action="store_true")
    p.add_argument(
        "--dense_optimizers",
        type=str,
        default="adamw_torch",
        help="Comma-separated dense baselines: adamw_torch,adamw_8bit",
    )
    p.add_argument("--phase_mask_types", type=str, default="element", help="Comma-separated: element,block")
    p.add_argument("--phase_start", type=int, default=0)
    p.add_argument("--phase_end", type=int, default=None)
    args = p.parse_args()

    sparsity_levels = [float(x.strip()) for x in str(args.benchmark_sparsities).split(",") if x.strip()]
    mask_types = tuple(x.strip() for x in str(args.phase_mask_types).split(",") if x.strip())
    for mt in mask_types:
        if mt not in ("element", "block"):
            raise ValueError(f"Invalid mask_type {mt!r}; expected element or block")

    dense_opts = tuple(x.strip() for x in str(args.dense_optimizers).split(",") if x.strip())
    if args.no_dense_baseline:
        dense_opts = tuple()
    for d in dense_opts:
        if d not in ("adamw_torch", "adamw_8bit"):
            raise ValueError(f"Invalid dense optimizer {d!r}; expected adamw_torch or adamw_8bit")

    phases = _build_phases(sparsity_levels, dense_optimizers=dense_opts, mask_types=mask_types)
    n_total = len(phases)
    ps = max(0, int(args.phase_start))
    pe = n_total if args.phase_end is None else int(args.phase_end)
    pe = max(ps, min(pe, n_total))
    phases = phases[ps:pe]
    slurm_safe_print(f"Phase slice: [{ps}:{pe}) of {n_total} expanded phases → running {len(phases)} phase(s)")

    slurm_safe_print(
        f"Benchmark config: n_steps per phase={args.n_steps}, batch={args.batch_size}, grad_accum={args.grad_accum}"
    )
    slurm_safe_print(
        f"Phase grid: {len(phases)} phases (dense={'no' if args.no_dense_baseline else 'yes'}) "
        f"sparsity levels={sparsity_levels} mask_types={list(mask_types)}"
    )
    if dense_opts:
        slurm_safe_print(f"Dense baselines: {list(dense_opts)}")

    os.environ["HF_DATASETS_CACHE"] = args.dataset_cache_dir or os.environ.get(
        "HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets")
    )
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "benchmark_training_log.csv")
    if os.path.isfile(csv_path):
        os.remove(csv_path)
    sink = BenchmarkRunLogSink(csv_path)

    theory_json_path = os.path.join(args.output_dir, "benchmark_theory.json")
    theory_records: List[Dict[str, Any]] = []
    b_tokens = default_b_tokens_proxy(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        chosen_rejected_pairs=True,
    )

    slurm_safe_print("Loading tokenizer and dataset (once)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dpo_ds = registry_load_dpo(args.dataset, subset_size=args.subset_size)

    gc_flag = not args.no_gradient_checkpointing

    for (phase_name, sparsity_pct, opt, mask_type) in phases:
        slurm_safe_print(
            f"\n{'#'*60}\nPHASE {phase_name}  sparsity={sparsity_pct}  optimizer={opt} mask_type={mask_type}\n{'#'*60}\n"
        )
        mask_path = None
        extra_log: Optional[Dict[str, Any]] = None

        if sparsity_pct is not None:
            os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True)
            mt = "block" if mask_type == "block" else "element"
            cache_path = _mask_cache_path(
                out_dir=args.output_dir,
                sparsity_pct=float(sparsity_pct),
                mask_type=mt,
                block_size=int(args.block_size_mask),
                mlp_only=bool(args.mlp_only),
                min_layer_keep_ratio=float(args.min_layer_keep_ratio),
            )
            seed = _stable_mask_seed(base=424200, sparsity_pct=float(sparsity_pct), mask_type=mt)

            if os.path.isfile(cache_path):
                slurm_safe_print(f"  Reusing cached mask: {cache_path}")
                _t = time.perf_counter()
                from src.utils.mask_manager import SparseMaskManager

                mm = SparseMaskManager(cache_path, device=torch.device("cpu"))
                bool_masks = {k: mm.get_mask(k).bool() for k in mm.masks.keys()}
                _bench_json(
                    {
                        "kind": "mask",
                        "phase": phase_name,
                        "seconds": round(time.perf_counter() - _t, 6),
                        "cached": True,
                        "cache_path": cache_path,
                        "mask_keys": int(len(bool_masks)),
                    }
                )
            else:
                _t = time.perf_counter()
                if mt == "block":
                    masks = generate_block_random_masks_cpu(
                        args.model_name,
                        args.checkpoint,
                        sparsity_pct,
                        seed=seed,
                        mlp_only=args.mlp_only,
                        min_layer_keep_ratio=args.min_layer_keep_ratio,
                        block_size=args.block_size_mask,
                    )
                else:
                    masks = generate_random_masks_cpu(
                        args.model_name,
                        args.checkpoint,
                        sparsity_pct,
                        seed=seed,
                        mlp_only=args.mlp_only,
                        min_layer_keep_ratio=args.min_layer_keep_ratio,
                    )
                bool_masks = {k: v.bool() if v.dtype != torch.bool else v for k, v in masks.items()}
                print_block_sparsity_profile(bool_masks, block_size=args.block_size_mask)
                meta = {
                    "method": "random_global_optimizer_benchmark",
                    "sparsity_percent": float(sparsity_pct),
                    "seed": int(seed),
                    "format": "torch_bool_binary",
                    "mask_type": mt,
                    "block_size_mask": int(args.block_size_mask),
                    "mlp_only": bool(args.mlp_only),
                    "min_layer_keep_ratio": float(args.min_layer_keep_ratio),
                }
                save_masks(bool_masks, cache_path, metadata=meta)
                slurm_safe_print(f"  Saved cached mask: {cache_path}")
                _bench_json(
                    {
                        "kind": "mask",
                        "phase": phase_name,
                        "seconds": round(time.perf_counter() - _t, 6),
                        "cached": False,
                        "cache_path": cache_path,
                        "mask_keys": int(len(bool_masks)),
                    }
                )
                del masks
                gc.collect()

            mask_path = cache_path
            base_log = dict(compute_sparse_mask_theory_metrics(bool_masks, b_tokens))
            base_log["mask_type"] = mask_type
            base_log["benchmark_phase_target_steps"] = int(args.n_steps)
            base_log["benchmark_mask_key_count"] = int(len(bool_masks))
            base_log["benchmark_mlp_only"] = int(bool(args.mlp_only))
            extra_log = base_log
            theory_records.append({"phase": phase_name, "sparsity_target_pct": float(sparsity_pct), **base_log})
            try:
                _write_theory_sidecar(theory_json_path, theory_records)
            except OSError as e:
                print(f"WARNING: could not write incremental benchmark_theory.json: {e}", file=sys.stderr)
        else:
            stub = dense_phase_theory_stub(b_tokens=b_tokens)
            stub["benchmark_phase_target_steps"] = int(args.n_steps)
            stub["benchmark_mask_key_count"] = 0
            stub["benchmark_mlp_only"] = int(bool(args.mlp_only))
            extra_log = stub
            theory_records.append({"phase": phase_name, "sparsity_target_pct": None, **stub})
            try:
                _write_theory_sidecar(theory_json_path, theory_records)
            except OSError as e:
                print(f"WARNING: could not write incremental benchmark_theory.json: {e}", file=sys.stderr)

        run_name = f"{args.run_label}_{phase_name}"
        sparse_dpo_eff_train(
            model_name=args.model_name,
            checkpoint_path=args.checkpoint,
            mask_path=mask_path if sparsity_pct is not None else None,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            subset_size=args.subset_size,
            run_name=run_name,
            mlp_only=args.mlp_only,
            block_size=32,
            optimizer_type=(opt if sparsity_pct is None else "sparse_adamw"),
            save_csv=False,
            grad_accum=args.grad_accum,
            save_model=False,
            dataset_key=args.dataset,
            output_base_dir=args.output_dir,
            dataset_cache_dir=args.dataset_cache_dir or os.environ["HF_DATASETS_CACHE"],
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
            dpo_beta=args.dpo_beta,
            gradient_checkpointing=gc_flag,
            save_steps=None,
            save_total_limit=None,
            resume_from_checkpoint=None,
            benchmark_log_sink=sink,
            benchmark_phase=phase_name,
            benchmark_sparsity_pct=sparsity_pct,
            benchmark_optimizer_label=("adamw" if sparsity_pct is None else "sparse_adamw"),
            benchmark_extra_log_fields=extra_log,
            use_wandb=False,
            train_dataset=dpo_ds,
            tokenizer_obj=tokenizer,
        )
        try:
            sink.flush()
        except OSError as e:
            print(f"WARNING: could not flush benchmark CSV (disk/NFS issue): {e}", file=sys.stderr)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        sink.close()
    except OSError as e:
        print(f"WARNING: final CSV flush failed: {e}", file=sys.stderr)
    try:
        _write_theory_sidecar(theory_json_path, theory_records)
    except OSError as e:
        print(f"WARNING: could not write benchmark_theory.json: {e}", file=sys.stderr)

    slurm_safe_print(f"\n✓ Optimizer-only benchmark complete. CSV: {csv_path}")
    slurm_safe_print(f"✓ Theory sidecar: {theory_json_path}")


if __name__ == "__main__":
    main()

