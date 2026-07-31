
#!/usr/bin/env python3
"""
Triton-Accelerated Sparse DPO Training - EFFICIENCY FOCUSED

Refactored from sparse_DPO_v3.py.
Focus: Optimizer Ablations (Sparse AdamW vs Dense/AdamW/SGD)

Key Features:
1. Modular architecture using src.kernels, src.optimizers, src.utils
2. Flexible logging (CSV, WandB)
3. Optimized Triton kernels for Sparse AdamW
4. Flexible checkpointing
"""

import os
import sys

# Add project root to sys.path to resolve 'src' imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

os.environ.pop("WANDB_DISABLED", None)
os.environ["WANDB_MODE"] = "online"
os.environ.pop("WANDB_SILENT", None)
os.environ.setdefault("WANDB_CONSOLE", "off")

import argparse
import json
import time
import torch
import wandb
from typing import Callable
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from typing import Any, Dict, Optional, Union

from src.utils.mask_manager import SparseMaskManager
from src.utils.scratch_paths import default_hf_datasets_cache, default_rl_casino_outputs
from src.utils.data_utils import make_dpo_collator
from src.utils.dataset_registry import get_dataset_config, load_dpo_dataset as registry_load_dpo
from src.utils.logging_utils import (
    FlexibleCheckpointCallback,
    CSVLoggerCallback,
    BenchmarkThroughputCallback,
)
from src.utils.grpo_checkpoint_utils import (
    maybe_load_wandb_resume_env,
    resolve_resume_checkpoint,
    RunManifestCallback,
    WandbRunIdCallback,
)
from src.optimizers.sparse_adamw import SparseAdamW

def sanitize_model_name(model_name: str) -> str:
    sanitized = model_name.replace("/", "_").replace("-", "_").lower()
    return "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized).strip("_")


def train(
    model_name,
    checkpoint_path,
    mask_path,
    n_steps,
    batch_size,
    learning_rate,
    subset_size,
    run_name,
    mlp_only,
    block_size,
    optimizer_type,
    save_csv,
    grad_accum,
    save_model,
    dataset_key,
    output_base_dir,
    dataset_cache_dir,
    warmup_ratio,
    weight_decay,
    max_length,
    max_prompt_length,
    dpo_beta,
    gradient_checkpointing=True,
    save_steps=None,
    save_total_limit=None,
    resume_from_checkpoint=None,
    benchmark_log_sink=None,
    benchmark_phase: str = None,
    benchmark_sparsity_pct: float = None,
    benchmark_optimizer_label: str = None,
    benchmark_extra_log_fields: Optional[Dict[str, Any]] = None,
    use_wandb: bool = True,
    train_dataset=None,
    tokenizer_obj=None,
    load_in_8bit: bool = False,
    precompute_ref_log_probs: bool = False,
    max_grad_norm: float = 1.0,
    sparse_adamw_max_grad_norm: float = 0.0,
    delta_log_interval: Optional[int] = None,
    delta_log_end_step: Optional[int] = None,
):
    # Determine model path
    if checkpoint_path is None or str(checkpoint_path).lower() == "none":
        checkpoint_path = model_name

    if not use_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"
        os.environ.setdefault("WANDB_SILENT", "true")
    
    # Set dataset cache directory
    os.environ["HF_DATASETS_CACHE"] = dataset_cache_dir
    
    # Resolve dataset via registry
    ds_config = get_dataset_config(dataset_key)
    dataset_name = ds_config["hf_id"]
    dataset_sanitized = ds_config["sanitized_name"]
    
    if run_name is None:
        run_name = f"sparse_dpo_efficiency_{optimizer_type}_{sanitize_model_name(model_name)}_{dataset_sanitized}"
    
    wandb_project = "huggingface"
    os.environ["WANDB_PROJECT"] = wandb_project
    run_dir = os.path.join(output_base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    output_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    _prepare_t0 = time.perf_counter()

    resume_ckpt = resolve_resume_checkpoint(output_dir, resume_from_checkpoint)
    if use_wandb:
        maybe_load_wandb_resume_env(run_dir, resume_ckpt)

    use_hf_rolling = save_steps is not None and save_steps > 0 and save_steps < 10**9
    hf_save_total_limit = save_total_limit if save_total_limit is not None else 3

    print(f"\n{'='*60}")
    print(f"SPARSE DPO EFFICIENCY TRAINING")
    print(f"{'='*60}")
    print(f"Run Directory: {run_dir}")
    print(f"Dataset: {dataset_key} ({dataset_name})")
    print(f"Optimizer: {optimizer_type}")
    print(f"WandB: {'on' if use_wandb else 'off'}, CSV: {save_csv}")
    print(
        f"DPO training: max_steps={n_steps}, num_train_epochs=1, peak_lr={learning_rate}, "
        f"warmup_ratio={warmup_ratio}, lr_scheduler=linear (Trainer; align with DPO_train.py / pipeline NUM_STEPS_DPO)"
    )
    print(f"HF rolling checkpoints: {use_hf_rolling} resume={resume_ckpt!r}")

    # Load Components (optional reuse for multi-phase drivers that preload once)
    if tokenizer_obj is not None and train_dataset is not None:
        tokenizer = tokenizer_obj
        dpo_dataset = train_dataset
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dpo_dataset = registry_load_dpo(dataset_key, subset_size=subset_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if load_in_8bit:
        # int8 weight storage (bitsandbytes): policy ~32 GB for 32B, ref model freed via
        # precompute_ref_log_probs.  Trainable (unmasked) params will have requires_grad=True;
        # frozen (masked-out) params have requires_grad=False, so gradients + SparseAdamW
        # states only exist for the 2.5% active weights.
        # If SparseAdamW raises a dtype error at the optimizer step, fall back to
        # --load_in_8bit=false with 2 GPUs.
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            load_in_8bit=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        # device_map="auto" is an INFERENCE placement: it fills cuda:0 first and ignores
        # training memory (gradients ≈ weights; the live reference deep-copy ≈ weights;
        # dense SparseAdamW states ≈ 2× weights; activations).  On 32B it put the whole
        # 64 GB policy + 64 GB ref on cuda:0 → OOM at accelerator.prepare.  Cap per-GPU
        # weight placement so the policy (and the ref/grads/states that follow it) spread
        # across all GPUs.  cap × n_gpus must exceed the model size (else weights spill to
        # CPU and training crawls): 18 GiB × 4 = 72 GiB > 64 GB for Qwen3-32B in bf16.
        bf16_max_memory = None
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            cap_gib = int(os.environ.get("SPARSE_PER_GPU_WEIGHT_CAP_GIB", "18"))
            bf16_max_memory = {i: f"{cap_gib}GiB" for i in range(n_gpus)}
            print(f"bf16 device_map: capping weights to {cap_gib} GiB/GPU across {n_gpus} GPUs "
                  f"to leave headroom for ref + grads + optimizer state")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=bf16_max_memory,
        )
    model.config.use_cache = False

    mask_manager: Optional[SparseMaskManager] = None
    if optimizer_type == "sparse_adamw":
        if mask_path is None or not str(mask_path).strip():
            raise ValueError("sparse_adamw requires a non-empty mask_path")
        mask_path_resolved: Union[str, os.PathLike] = mask_path
        if not os.path.isfile(mask_path_resolved):
            raise FileNotFoundError(f"Mask file not found: {mask_path_resolved}")
        mask_manager = SparseMaskManager(mask_path_resolved, device=device)

    if load_in_8bit and mask_manager is not None:
        # Enable gradients only on unmasked (trainable) parameters so bitsandbytes'
        # autograd + SparseAdamW operate on the 2.5% active weight subset.
        _n_trainable = 0
        for name, param in model.named_parameters():
            is_active = name in mask_manager.masks and mask_manager.masks[name].any().item()
            if param.is_floating_point():
                param.requires_grad_(bool(is_active))
            if is_active:
                _n_trainable += 1
        print(f"int8 mode: {_n_trainable} trainable param tensors (requires_grad=True), "
              f"rest frozen in int8")
    
    # Optimizer Logic
    print(f"Initializing {optimizer_type}...")
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type in ("adamw", "adamw_torch"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_type == "adamw_8bit":
        try:
            from bitsandbytes.optim import AdamW8bit

            optimizer = AdamW8bit(model.parameters(), lr=learning_rate)
        except ImportError as e:
            raise ImportError(
                "optimizer_type=adamw_8bit requested but bitsandbytes is not importable"
            ) from e
    elif optimizer_type == "sparse_adamw":
        optimizer = SparseAdamW(
            list(model.named_parameters()),
            mask_manager,
            lr=learning_rate,
            block_size=block_size,
            mlp_only=mlp_only,
            # Single-clip regime (matches dense DPO_train.py): the Trainer's global-norm clip
            # (DPOConfig.max_grad_norm) is the only clip. SparseAdamW's per-param clip is OFF by
            # default (0.0) so sparse is not double-clipped with a differently-shaped clip — the
            # same fix as GRPO, and the gap the E2 config guard could not see.
            max_grad_norm=sparse_adamw_max_grad_norm,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # Callbacks
    callbacks = []

    manifest = {
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "mask_path": mask_path,
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "n_steps": n_steps,
        "learning_rate": learning_rate,
        "optimizer": optimizer_type,
        "mlp_only": mlp_only,
        "output_dir": output_dir,
        "resume_from_checkpoint": resume_ckpt,
        "hf_rolling_save_steps": save_steps,
        "hf_save_total_limit": hf_save_total_limit if use_hf_rolling else None,
        # Clipping provenance for the footing gate (verify_grpo_config_consistency.py works for
        # DPO arms too — DPOConfig-absent fields compare as None uniformly).
        "trainer_max_grad_norm": max_grad_norm,
        "sparse_adamw_max_grad_norm": sparse_adamw_max_grad_norm,
    }
    callbacks.append(RunManifestCallback(run_dir, manifest))
    if use_wandb:
        callbacks.append(WandbRunIdCallback(run_dir))

    # Weight-delta logging on a milestone schedule (source of the magnitude masks). Gated on
    # --delta_log_interval so ONLY the dense arm emits deltas; the sparse arms don't build masks.
    # Mirrors GRPO_train.py:344-370 exactly (same range(interval, end+1, interval) schedule, bf16
    # base_state, base_state.pt dump) so the DPO and GRPO dense arms produce magnitude masks on an
    # identical step schedule — a precondition for the mag k=50/100/250 masks being comparable
    # across objectives. This script runs under device_map="auto" (not FSDP), so a plain
    # model.named_parameters() pass gives full-shaped params — no accelerator.get_state_dict gather.
    interval = delta_log_interval
    if interval is not None and interval > 0 and not resume_ckpt:
        end = delta_log_end_step
        if end is None:
            end = min(n_steps, max(interval, n_steps // 10))
        else:
            end = min(n_steps, end)
        schedule = list(range(interval, end + 1, interval))
        if not schedule and n_steps > 0:
            schedule = [n_steps]
        delta_log_dir = os.path.join(run_dir, "deltas")
        os.makedirs(delta_log_dir, exist_ok=True)
        base_state = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                # bf16 base (matches GRPO_train.py:359 / DPO_train.py:480): deltas feed |Δθ|
                # mask selectors, so bf16 is sufficient and halves base_state.pt on disk.
                base_state[name] = param.detach().to(torch.bfloat16).cpu().clone()
        torch.save(base_state, os.path.join(delta_log_dir, "base_state.pt"))
        callbacks.append(
            FlexibleCheckpointCallback(
                base_state=base_state,
                delta_log_dir=delta_log_dir,
                checkpoint_schedule=schedule,
                threshold=1e-5,
                model_name=model_name,
                dataset_name=dataset_name,
                subset_size=subset_size,
                learning_rate=learning_rate,
                batch_size=batch_size,
                grad_accum=grad_accum,
                run_name=run_name,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
            )
        )
        print(f"Delta logging enabled: schedule steps {schedule[:5]}"
              f"{'...' if len(schedule) > 5 else ''}")
    elif resume_ckpt:
        print("Resume: skipping FlexibleCheckpointCallback weight-delta logging "
              "(base_state would not match a cold start).")
    else:
        print("Delta logging OFF (no --delta_log_interval): sparse arms don't emit masks.")

    if save_csv:
        callbacks.append(CSVLoggerCallback(output_dir=run_dir))

    if benchmark_log_sink is not None and benchmark_phase:
        label = benchmark_optimizer_label or optimizer_type
        _te = os.environ.get("RL_CASINO_THROUGHPUT_PRINT_EVERY", "").strip()
        _print_every = int(_te) if _te else 25
        callbacks.append(
            BenchmarkThroughputCallback(
                benchmark_log_sink,
                phase=str(benchmark_phase),
                sparsity_target_pct=benchmark_sparsity_pct,
                optimizer_label=str(label),
                print_every=_print_every,
                extra_log_fields=benchmark_extra_log_fields,
            )
        )

    if use_hf_rolling:
        save_strategy = "steps"
        cfg_save_steps = save_steps
        cfg_save_total = hf_save_total_limit
    else:
        save_strategy = "no"
        cfg_save_steps = 500
        cfg_save_total = None

    # DPO Config — align with src/full_training/DPO_train.py (step-based run: max_steps + num_train_epochs=1)
    dpo_config = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_steps=n_steps,
        num_train_epochs=1,
        lr_scheduler_type="linear",
        # Global-norm clip (HF Trainer). Set explicitly so it is pinned rather than left to the HF
        # default; MUST equal the dense DPO_train.py value. This is the only clip (SparseAdamW's
        # per-param clip is disabled above).
        max_grad_norm=max_grad_norm,
        logging_steps=1,
        save_strategy=save_strategy,
        save_steps=cfg_save_steps,
        save_total_limit=cfg_save_total,
        report_to="wandb" if use_wandb else "none",
        run_name=run_name,
        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=gradient_checkpointing,
        beta=dpo_beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        precompute_ref_log_probs=precompute_ref_log_probs,
    )

    if load_in_8bit:
        # transformers 4.57+ blocks direct int8 fine-tuning without PEFT by checking
        # is_quantized and not _hf_peft_config_loaded and not is_qat_trainable.
        # bitsandbytes int8 sets is_trainable=True but is_qat_trainable=False (base class
        # default), so the guard fires even though training is supported.  Setting
        # _hf_peft_config_loaded bypasses the check; it is only read at trainer init.
        model._hf_peft_config_loaded = True

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dpo_dataset,
        # Configurable, shared collator so dense (this script, --optimizer adamw_torch) and sparse
        # arms tokenize identically. Truncation from the script's args (prompt 1024) — the previous
        # data_utils.dpo_collator_fn hardcoded prompt=512, diverging from dense DPO_train.py's 1024.
        data_collator=make_dpo_collator(
            tokenizer,
            max_prompt_length=max_prompt_length,
            max_chosen_rejected_length=max_length,
        ),
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )

    _prepare_s = time.perf_counter() - _prepare_t0
    _train_t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    _trainer_s = time.perf_counter() - _train_t0

    if benchmark_log_sink is not None and benchmark_phase:
        try:
            payload = {
                "kind": "train_wall",
                "phase": str(benchmark_phase),
                "run_name": str(run_name),
                "prepare_s": round(_prepare_s, 6),
                "trainer_s": round(_trainer_s, 6),
            }
            print("BENCH_JSON " + json.dumps(payload, separators=(",", ":")), flush=True)
        except Exception:
            pass

    # Final Saving
    if save_model:
        print(f"\nTraining complete. Saving final model to {run_dir}/final_model...")
        final_save_dir = os.path.join(run_dir, "final_model")
        os.makedirs(final_save_dir, exist_ok=True)
        
        trainer.save_model(final_save_dir)
        tokenizer.save_pretrained(final_save_dir)
        print(f"✓ Full checkpoint saved to {final_save_dir}")
    else:
        print("\nTraining complete. Skipping final model saving as requested.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mask", type=str, default="masks/top_10.0pct_momentum_w25_step25.pt")
    parser.add_argument("--n_steps", type=int, default=250, help="Must match dense --num_steps / pipeline NUM_STEPS_DPO")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-7,
        help="Peak LR (default 5e-7, same as DPO_train.py / pipeline DPO_LEARNING_RATE)",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Global-norm gradient clip (HF Trainer). MUST equal the dense DPO_train.py value "
        "for equal footing (Table 7 = 1.0).",
    )
    parser.add_argument(
        "--sparse_adamw_max_grad_norm",
        type=float,
        default=0.0,
        help="Per-parameter clip inside SparseAdamW; 0 disables it (default). Kept off so the ONLY "
        "clip is the Trainer global clip above, matching the dense arm's single clip.",
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument(
        "--delta_log_interval",
        type=int,
        default=None,
        help="If set, log bf16 weight deltas vs init every N steps (source of magnitude masks). "
        "Pass ONLY on the dense arm; sparse arms don't build masks. Default: off. Match GRPO's value.",
    )
    parser.add_argument(
        "--delta_log_end_step",
        type=int,
        default=None,
        help="Last step (inclusive) for delta logs when --delta_log_interval is set (e.g. 250 for "
        "mag k=50/100/250 with interval 50). Match GRPO's value for cross-objective comparability.",
    )
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw", "adamw_torch", "sparse_adamw"], default="sparse_adamw",
                        help="Dense DPO arm uses adamw_torch (== adamw here); sparse arms use sparse_adamw.")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument(
        "--mlp_only",
        action="store_true",
        default=False,
        help="Restrict sparse updates to MLP layers only (default: full model where masks exist)",
    )
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--dataset", type=str, default="light-r1",
                       help="Dataset key (light-r1, tulu3, math-step-dpo, codepref) or HuggingFace ID")
    parser.add_argument("--output_base_dir", type=str, default=default_rl_casino_outputs(), help="Base directory for outputs")
    parser.add_argument("--dataset_cache_dir", type=str, default=default_hf_datasets_cache(), help="Cache directory for HuggingFace datasets")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name for WandB and results directory")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=None)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    
    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
        else: raise argparse.ArgumentTypeError('Boolean value expected.')
        
    parser.add_argument("--save_model", type=str2bool, default=True, help="Save final model checkpoint (default: True)")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="HF Trainer checkpoint interval (rolling). Omit for no intermediate HF checkpoints (legacy behavior).",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Keep only the newest K HF checkpoints when --save_steps is set (default: 3).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint-* dir, or 'auto' for latest under run_dir/checkpoints.",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model weights in int8 via bitsandbytes (~50%% VRAM reduction). "
             "Trainable (unmasked) params have requires_grad=True; frozen params stay in int8. "
             "Combine with --precompute_ref_log_probs to keep peak VRAM below 80 GB for 32B models.",
    )
    parser.add_argument(
        "--precompute_ref_log_probs",
        action="store_true",
        help="Precompute reference model log-probs before training; reference model is freed "
             "from VRAM before the training loop, matching the DPO_train.py option.",
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
        block_size=args.block_size,
        optimizer_type=args.optimizer,
        save_csv=args.save_csv,
        grad_accum=args.grad_accum,
        save_model=args.save_model,
        dataset_key=args.dataset,
        output_base_dir=args.output_base_dir,
        dataset_cache_dir=args.dataset_cache_dir,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        sparse_adamw_max_grad_norm=args.sparse_adamw_max_grad_norm,
        delta_log_interval=args.delta_log_interval,
        delta_log_end_step=args.delta_log_end_step,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        dpo_beta=args.dpo_beta,
        gradient_checkpointing=False if args.no_gradient_checkpointing else (True if args.gradient_checkpointing is True else True),
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
        load_in_8bit=args.load_in_8bit,
        precompute_ref_log_probs=args.precompute_ref_log_probs,
    )
