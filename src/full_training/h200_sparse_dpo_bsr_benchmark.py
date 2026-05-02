#!/usr/bin/env python3
"""
H200-oriented multi-phase BSR DPO benchmark: optional dense baseline (omit with ``--no_dense_baseline``)
plus a **full sparse grid**
(mask element vs block, grad_input dense vs Triton sparse, SparseAdamW block_1d vs block_2d)
per ``--benchmark_sparsities``.

- **Tokenizer + DPO dataset:** loaded once and reused across phases.
- **Training model:** each phase calls ``sparse_dpo_train``, which loads weights from the Hub/local
  cache again. Dense vs sparse need different module graphs (BSR replaces ``nn.Linear``), so we do
  not reuse one GPU model across those modes. Loads are from disk cache (fast) but still pay
  shard read + GPU alloc each phase.
- **Random masks:** built from parameter **shapes** only via ``device_map=\"meta\"`` (no full 8B CPU
  materialization per sparse phase). Falls back to a full CPU load if meta init fails.

CSV: <output_dir>/benchmark_training_log.csv (via BenchmarkRunLogSink + BenchmarkThroughputCallback).
**CUDA segment columns** (``t_step_total_ms``, ``t_forward_ms``, ``t_backward_ms``, etc.) are written only
when ``RL_CASINO_BSR_DETAILED_TIMING=1`` (see ``sparse_dpo_bsr.train``); default is **off** so sweeps stay fast.
Theory: <output_dir>/benchmark_theory.json (one object per phase) plus ``theory_*`` columns duplicated on
each CSV row from ``sparse_training_complexity.md``-style proxies (see ``src/utils/bsr_theory_metrics.py``).
"""

import os
import sys

# Before any library imports: force-disable W&B console wrapping (do not use setdefault — the login
# shell may export other values). wandb's stdout shim + NFS-backed Slurm .out can OSError errno 116.
def _force_wandb_inert_for_slurm_logs() -> None:
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_CONSOLE"] = "off"
    # Newer wandb: suppress service noise; harmless if ignored.
    os.environ.setdefault("WANDB_SILENT", "true")


_force_wandb_inert_for_slurm_logs()

import argparse
import gc
import json
import hashlib
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.slurm_safe_log import slurm_safe_print
from src.full_training.sparse_dpo_bsr import train as sparse_dpo_train
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


def _build_random_scores_for_masks(model: torch.nn.Module, mlp_only: bool) -> dict:
    scores = {}
    for name, p in model.named_parameters():
        if "weight" not in name or p.dim() != 2:
            continue
        if mlp_only and "mlp" not in name.lower():
            continue
        # CPU float scores; model may be meta (no materialized weights).
        scores[name] = torch.rand(tuple(p.shape), dtype=torch.float32, device="cpu")
    return scores


def _model_for_mask_shapes(ckpt: str):
    """
    Parameter shapes only — avoid materializing ~8B weights on CPU for every sparse phase.
    Prefer Hugging Face ``device_map='meta'`` (skeleton model). Fallback: one full CPU load.
    """
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


def generate_random_masks_cpu(
    model_name: str,
    checkpoint_path: Optional[str],
    sparsity_percent: float,
    seed: int,
    mlp_only: bool,
    min_layer_keep_ratio: float,
) -> dict:
    """Build global random masks from weight shapes (meta model) + CPU mask selector."""
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
    """Build global random masks that are explicitly block-structured (e.g., 16x16)."""
    ckpt = checkpoint_path if checkpoint_path and str(checkpoint_path).lower() != "none" else model_name
    torch.manual_seed(seed)
    slurm_safe_print(
        f"  Generating BLOCK-STRUCTURED random mask (target sparsity {sparsity_percent}%, seed={seed}, block={block_size}) on CPU..."
    )
    m = _model_for_mask_shapes(ckpt)
    
    # 1. Generate random scores for the block grid
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

    # 2. Prune the block grid! (This guarantees global sparsity constraints at the block level)
    # We turn OFF tie_break_noise so that if a block score is exactly on the threshold, 
    # it gets decided cleanly, though with random floats it's rare anyway.
    block_masks = create_mask_from_scores_gpu_efficient(
        block_scores,
        sparsity_percent,
        device="cpu",
        min_layer_keep_ratio=min_layer_keep_ratio,
        local_pool=False,
        add_tie_break_noise=False,
    )
    
    # 3. Expand block masks back to original shapes
    final_masks = {}
    for name, b_mask in block_masks.items():
        # Expand
        expanded = b_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
        # Crop padding
        M, N = original_shapes[name]
        final_masks[name] = expanded[:M, :N]
        
    return final_masks


def _build_benchmark_phases(
    sparsity_levels: List[float],
    *,
    include_dense_baseline: bool = True,
    mask_types: Tuple[str, ...] = ("element", "block"),
    grad_input_modes: Tuple[str, ...] = ("dense", "sparse"),
    adam_kernels: Tuple[str, ...] = ("block_1d", "block_2d"),
) -> List[Tuple]:
    """
    Full comparison grid (sparse phases):

    - **Mask:** element-wise random (`generate_random_masks_cpu`) vs block-random (`generate_block_random_masks_cpu`).
    - **Grad input:** dense ``grad_output @ weight`` vs Triton block-sparse grad_input
      (``RL_CASINO_BSR_GRAD_INPUT_MODE`` = dense vs sparse).
    - **SparseAdamW kernel:** ``RL_CASINO_ADAM_KERNEL`` = block_1d vs block_2d.

    Tuple: (phase_name, sparsity_pct, optimizer_type, mask_type, adam_kernel, grad_input_mode)
    ``grad_input_mode`` is None for the dense baseline; otherwise ``dense`` or ``sparse``.
    """
    phases: List[Tuple] = []
    if include_dense_baseline:
        phases.append(("dense", None, "adamw", "none", None, None))
    for sp in sparsity_levels:
        sp_tag = str(sp).replace(".", "p")
        for mask_type in mask_types:
            mt_short = "elem" if mask_type == "element" else "blk"
            for gi in grad_input_modes:
                for adam in adam_kernels:
                    phase_name = f"s{sp_tag}_{mt_short}_gi{gi}_{adam}"
                    phases.append((phase_name, sp, "sparse_adamw", mask_type, adam, gi))
    return phases


def _stable_mask_seed(*, base: int, sparsity_pct: float, mask_type: str) -> int:
    """
    Deterministic seed per (sparsity, mask_type), independent of phase order.
    This enables mask reuse across the 4 variants (gi dense/sparse × Adam 1d/2d).
    """
    h = hashlib.sha256(f"{sparsity_pct:.5f}|{mask_type}".encode("utf-8")).hexdigest()
    # take 31 bits to keep torch.manual_seed happy across platforms
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
    """Write theory JSON incrementally so partial runs still have a sidecar."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as tf:
        json.dump(records, tf, indent=2)
    os.replace(tmp, path)


def main():
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
    p.add_argument(
        "--benchmark_sparsities",
        type=str,
        default="99.75",
        help="Comma-separated target sparsity %% for sparse phases (e.g. 97.5,95,90). "
        "Each level expands to element/block mask × grad_input modes × Adam 1d/2d.",
    )
    p.add_argument(
        "--no_dense_baseline",
        action="store_true",
        help="Skip the dense AdamW baseline phase (only sparse BSR phases run).",
    )
    p.add_argument(
        "--phase_mask_types",
        type=str,
        default="element,block",
        help="Comma-separated mask types to benchmark: element,block.",
    )
    p.add_argument(
        "--phase_grad_input_modes",
        type=str,
        default="dense,sparse",
        help="Comma-separated grad_input modes: dense,sparse. Use 'dense' to focus on 1D/2D Adam kernels.",
    )
    p.add_argument(
        "--phase_adam_kernels",
        type=str,
        default="block_1d,block_2d",
        help="Comma-separated SparseAdamW kernels: block_1d,block_2d.",
    )
    args = p.parse_args()

    sparsity_levels = [
        float(x.strip())
        for x in str(args.benchmark_sparsities).split(",")
        if x.strip()
    ]

    mask_types = tuple(x.strip() for x in str(args.phase_mask_types).split(",") if x.strip())
    grad_input_modes = tuple(x.strip() for x in str(args.phase_grad_input_modes).split(",") if x.strip())
    adam_kernels = tuple(x.strip() for x in str(args.phase_adam_kernels).split(",") if x.strip())

    _valid_mask_types = {"element", "block"}
    _valid_gi = {"dense", "sparse"}
    _valid_adam = {"block_1d", "block_2d"}
    bad_mt = [x for x in mask_types if x not in _valid_mask_types]
    bad_gi = [x for x in grad_input_modes if x not in _valid_gi]
    bad_adam = [x for x in adam_kernels if x not in _valid_adam]
    if bad_mt or bad_gi or bad_adam:
        raise ValueError(
            "Invalid phase-grid filters: "
            f"mask_types bad={bad_mt} (valid={sorted(_valid_mask_types)}), "
            f"grad_input_modes bad={bad_gi} (valid={sorted(_valid_gi)}), "
            f"adam_kernels bad={bad_adam} (valid={sorted(_valid_adam)})"
        )

    phases = _build_benchmark_phases(
        sparsity_levels,
        include_dense_baseline=not args.no_dense_baseline,
        mask_types=mask_types,
        grad_input_modes=grad_input_modes,
        adam_kernels=adam_kernels,
    )

    dense_ct = (
        sum(1 for ph in phases if ph[1] is None)
        if phases
        else 0
    )
    sparse_ct = len(phases) - dense_ct

    slurm_safe_print(
        f"Benchmark config: n_steps per phase={args.n_steps}, batch={args.batch_size}, grad_accum={args.grad_accum}"
    )
    slurm_safe_print(
        f"Phase grid: {len(phases)} phases ({dense_ct} dense + {sparse_ct} sparse), "
        f"sparsity levels={sparsity_levels}  mask_types={list(mask_types)}  "
        f"grad_input_modes={list(grad_input_modes)}  adam_kernels={list(adam_kernels)}"
    )

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
    dm = args.device_map
    if dm.lower() in ("none", "null"):
        dm = "none"

    for idx, (phase_name, sparsity_pct, opt, mask_type, adam_kernel, grad_input_mode) in enumerate(
        phases
    ):
        slurm_safe_print(
            f"\n{'#'*60}\nPHASE {phase_name}  sparsity={sparsity_pct}  optimizer={opt} "
            f"mask_type={mask_type}  grad_input={grad_input_mode or 'n/a'}  adam_kernel={adam_kernel or 'n/a'}\n"
            f"{'#'*60}\n"
        )
        mask_path = None
        tmp_mask = None
        seed = None

        extra_log: Optional[Dict[str, Any]] = None
        if sparsity_pct is not None:
            # Mask reuse: cache one mask per (sparsity, mask_type, block_size, mlp_only, floor).
            os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True)
            mt = "block" if mask_type == "block" else "element"
            cache_path = _mask_cache_path(
                out_dir=args.output_dir,
                sparsity_pct=float(sparsity_pct),
                mask_type=mt,
                block_size=int(args.block_size_bsr),
                mlp_only=bool(args.mlp_only),
                min_layer_keep_ratio=float(args.min_layer_keep_ratio),
            )
            seed = _stable_mask_seed(base=424200, sparsity_pct=float(sparsity_pct), mask_type=mt)

            if os.path.isfile(cache_path):
                slurm_safe_print(f"  Reusing cached mask: {cache_path}")
                mask_path = cache_path
                # We still need theory metrics for the CSV/theory sidecar.
                from src.utils.mask_manager import SparseMaskManager
                mm = SparseMaskManager(cache_path, device=torch.device("cpu"))
                # SparseMaskManager stores the loaded mask tensors in `mm.masks` (no `list_masks()` helper).
                bool_masks = {k: mm.get_mask(k).bool() for k in mm.masks.keys()}
            else:
                if mt == "block":
                    masks = generate_block_random_masks_cpu(
                        args.model_name,
                        args.checkpoint,
                        sparsity_pct,
                        seed=seed,
                        mlp_only=args.mlp_only,
                        min_layer_keep_ratio=args.min_layer_keep_ratio,
                        block_size=args.block_size_bsr,
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

                # Profile and print block sparsity BEFORE saving
                print_block_sparsity_profile(bool_masks, block_size=args.block_size_bsr)

                meta = {
                    "method": "random_global_benchmark",
                    "sparsity_percent": float(sparsity_pct),
                    "seed": int(seed),
                    "format": "torch_bool_binary",
                    "mask_type": mt,
                    "block_size_bsr": int(args.block_size_bsr),
                    "mlp_only": bool(args.mlp_only),
                    "min_layer_keep_ratio": float(args.min_layer_keep_ratio),
                }
                save_masks(bool_masks, cache_path, metadata=meta)
                slurm_safe_print(f"  Saved cached mask: {cache_path}")
                mask_path = cache_path
                del masks
                gc.collect()

            base_log = dict(compute_sparse_mask_theory_metrics(bool_masks, b_tokens))
            base_log["mask_type"] = mask_type
            base_log["adam_kernel"] = adam_kernel or ""
            base_log["grad_input_mode"] = grad_input_mode or ""
            base_log["mask_seed"] = int(seed)
            extra_log = base_log
            theory_records.append(
                {"phase": phase_name, "sparsity_target_pct": float(sparsity_pct), **base_log}
            )
            try:
                _write_theory_sidecar(theory_json_path, theory_records)
            except OSError as e:
                print(f"WARNING: could not write incremental benchmark_theory.json: {e}", file=sys.stderr)
        else:
            stub = dense_phase_theory_stub(b_tokens=b_tokens)
            extra_log = stub
            theory_records.append({"phase": phase_name, "sparsity_target_pct": None, **stub})
            try:
                _write_theory_sidecar(theory_json_path, theory_records)
            except OSError as e:
                print(f"WARNING: could not write incremental benchmark_theory.json: {e}", file=sys.stderr)

        run_name = f"{args.run_label}_{phase_name}"

        try:
            prev_kernel = os.environ.get("RL_CASINO_ADAM_KERNEL", None)
            prev_gi = os.environ.get("RL_CASINO_BSR_GRAD_INPUT_MODE", None)
            if adam_kernel:
                os.environ["RL_CASINO_ADAM_KERNEL"] = adam_kernel
            if grad_input_mode:
                os.environ["RL_CASINO_BSR_GRAD_INPUT_MODE"] = grad_input_mode
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
                dense_baseline=(sparsity_pct is None),
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
                benchmark_extra_log_fields=extra_log,
            )
        finally:
            if adam_kernel:
                if prev_kernel is None:
                    os.environ.pop("RL_CASINO_ADAM_KERNEL", None)
                else:
                    os.environ["RL_CASINO_ADAM_KERNEL"] = prev_kernel
            if grad_input_mode:
                if prev_gi is None:
                    os.environ.pop("RL_CASINO_BSR_GRAD_INPUT_MODE", None)
                else:
                    os.environ["RL_CASINO_BSR_GRAD_INPUT_MODE"] = prev_gi
            try:
                sink.flush()
            except OSError as e:
                print(f"WARNING: could not flush benchmark CSV (disk/NFS issue): {e}", file=sys.stderr)
            # tmp_mask path no longer used; masks are cached under output_dir/masks/.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        sink.close()
    except OSError as e:
        print(f"WARNING: final CSV flush failed: {e}", file=sys.stderr)
    # Final theory write (incremental writes should already have succeeded).
    try:
        _write_theory_sidecar(theory_json_path, theory_records)
    except OSError as e:
        print(f"WARNING: could not write benchmark_theory.json: {e}", file=sys.stderr)
    slurm_safe_print(f"\n✓ Benchmark complete. CSV: {csv_path}")
    slurm_safe_print(f"✓ Theory sidecar: {theory_json_path}")


if __name__ == "__main__":
    main()
