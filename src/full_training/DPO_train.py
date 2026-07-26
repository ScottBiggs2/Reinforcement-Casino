import os
import json
import argparse
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import DPOTrainer, DPOConfig
from typing import List, Dict, Any

from src.utils.scratch_paths import default_hf_datasets_cache, default_rl_casino_outputs
from src.utils.grpo_checkpoint_utils import (
    maybe_load_wandb_resume_env,
    resolve_resume_checkpoint,
    RunManifestCallback,
    WandbRunIdCallback,
)


#######################################
# 0. Config
#######################################

def sanitize_model_name(model_name: str) -> str:
    """
    Convert HuggingFace model name to filesystem-safe string.

    Examples:
        "google/gemma-3-270m-it" -> "google_gemma_3_270m_it"
        "meta-llama/Llama-3.1-8B" -> "meta_llama_llama_3_1_8b"
    """
    sanitized = model_name.replace("/", "_").replace("-", "_").lower()
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO Training Script")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-270m-it",
        help="HuggingFace model name to load (default: google/gemma-3-270m-it)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="light-r1",
        help="Dataset key from registry (light-r1, tulu3, math-step-dpo, codepref) or HuggingFace ID",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name for WandB")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=250,
        help="Number of training steps (default: 250; match pipeline NUM_STEPS_DPO)",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=None,
        help="Train for N epochs (overrides --num_steps when set).",
    )
    parser.add_argument("--subset_size", type=int, default=None, help="Limit dataset size (default: None = full)")
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=default_rl_casino_outputs(),
        help="Base directory for all outputs (checkpoints, deltas)",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=default_hf_datasets_cache(),
        help="Cache directory for HuggingFace datasets",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Per-device train batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-7,
        help="Peak learning rate (default: 5e-7; match pipeline DPO_LEARNING_RATE).",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio for LR schedule.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--max_length", type=int, default=1024, help="Max total sequence length.")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Max prompt length.")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta.")
    parser.add_argument("--optim", type=str, default="adamw_8bit", help="Optimizer name for Trainer (e.g. adamw_8bit).")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing.")
    parser.add_argument(
        "--delta_log_interval",
        type=int,
        default=50,
        help="Save full weight deltas (vs init) every N steps for warm-start masks (default: 50).",
    )
    parser.add_argument(
        "--delta_log_end_step",
        type=int,
        default=None,
        help="Last training step (inclusive) to save deltas. Default: min(num_steps, max(interval, num_steps//10)) "
        "e.g. 10%% of run with interval 50 → steps 50..200 for 2k steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="HF Trainer checkpoint interval (save_strategy=steps). Omit or use a very large value to disable "
        "(delta-only mode, legacy pipeline behavior).",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Keep only the newest K HF checkpoints on disk when --save_steps is set (rolling). Default: 3.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint-* directory, or 'auto' for latest under output_dir.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="HF device_map, e.g. 'auto' for multi-GPU model parallelism (single "
             "process, NOT torchrun) — required for models too big for one GPU "
             "(e.g. Qwen3-32B). Default None = single-GPU/DDP via torchrun (unchanged).",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Enable FSDP full_shard+auto_wrap (launch via torchrun --nproc_per_node=N). "
             "Keeps all GPUs busy (fast, no RC idle-cancel) — preferred for full-FT of "
             "big models like Qwen3-32B. Mutually exclusive with --device_map.",
    )
    parser.add_argument(
        "--fsdp_layer_cls",
        type=str,
        default="Qwen3DecoderLayer",
        help="Transformer decoder layer class to wrap under FSDP (model-specific).",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Train with LoRA adapters instead of full fine-tuning (PEFT baseline for the "
             "sparse-subnetwork comparison). Requires an explicit --learning_rate.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA rank. r=64 over all linear projections is ~168M trainable params on "
             "Llama-3.1-8B (2.1%%), the closest match to a 97.5%%-sparse subnetwork (2.5%%). "
             "r=16 is the standard-practice point (~42M, 0.5%%).",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA alpha (scaling = alpha/r). Default: 2*r.",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout.")
    parser.add_argument(
        "--lora_use_rslora",
        action="store_true",
        help="Rank-stabilized LoRA: scale by alpha/sqrt(r) instead of alpha/r "
             "(Kalajdzievski 2023). With the conventional alpha=2r the scaling alpha/r is "
             "constant across ranks, which under-scales HIGH ranks — i.e. exactly the r=64 "
             "matched-budget arm. Off by default so the reported config is the standard one; "
             "turn it on if the r=64 arm underperforms r=16 and record which was used.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names to adapt. Default: all linear projections, so "
             "the adapted parameter set spans the same tensors the sparse mask covers.",
    )
    return parser.parse_args()


def _resolve_lora(args):
    """Build the LoRA config and guard the failure modes that waste a whole run.

    Returns (peft_config, lora_manifest_fields) or (None, {}) when --lora is off.
    """
    if not args.lora:
        return None, {}

    if args.fsdp:
        raise ValueError(
            "--lora and --fsdp are not supported together here. The LoRA baseline is an "
            "8B-scale comparison that fits on one GPU; FSDP-wrapping an adapter model "
            "changes both the param naming and the ref-model handling."
        )

    # The full-FT default (5e-7) is ~200x too small to move a freshly-initialised
    # adapter: B is zero-init, so the run would train to a near-null delta and report
    # a meaningless "LoRA underperforms" number. Fail loudly instead of silently.
    if args.learning_rate <= 1e-6:
        raise ValueError(
            f"--lora with --learning_rate={args.learning_rate:g} will not train. LoRA needs "
            f"its own LR (typically 1e-4; 5e-5 is a reasonable second point) because the "
            f"adapter is zero-initialised, not a pretrained weight. Pass --learning_rate "
            f"explicitly. Note in any write-up that LoRA got a tuned LR and the sparse "
            f"runs did not."
        )

    try:
        from peft import LoraConfig
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "--lora requires `peft` (installed in the rl_casino env, not necessarily "
            "in a bare python). pip install peft"
        ) from exc

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=args.lora_use_rslora,
    )
    fields = {
        "lora": True,
        "lora_r": args.lora_r,
        "lora_alpha": alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": target_modules,
        "lora_use_rslora": args.lora_use_rslora,
        "lora_scaling": (alpha / (args.lora_r ** 0.5)) if args.lora_use_rslora else (alpha / args.lora_r),
    }
    return peft_config, fields


def main() -> None:
    args = parse_args()
    os.environ["HF_DATASETS_CACHE"] = args.dataset_cache_dir

    model_name = args.model_name
    model_name_sanitized = sanitize_model_name(model_name)

    from src.utils.dataset_registry import get_dataset_config

    dataset_key = args.dataset
    dataset_config = get_dataset_config(dataset_key)
    dataset_name = dataset_config["hf_id"]
    dataset_sanitized = dataset_config["sanitized_name"]

    peft_config, lora_fields = _resolve_lora(args)

    base_dir = args.output_base_dir
    sub_dir = f"{model_name_sanitized}_{dataset_sanitized}"
    if peft_config is not None:
        # Keep LoRA runs off the dense run's paths, AND off each other's. The tag must
        # include the learning rate: an LR sweep at fixed rank would otherwise collide,
        # and with --resume_from_checkpoint auto one arm would silently resume from
        # another arm's checkpoint at a different LR — invisible in the output.
        lr_tag = f"{args.learning_rate:g}".replace("-", "m").replace("+", "").replace(".", "p")
        sub_dir = f"{sub_dir}_lora_r{args.lora_r}_lr{lr_tag}"
        if args.lora_use_rslora:
            sub_dir = f"{sub_dir}_rs"
    output_dir = os.path.join(base_dir, "checkpoints", sub_dir)
    delta_log_dir = os.path.join(base_dir, "deltas", sub_dir)

    os.makedirs(output_dir, exist_ok=True)

    resume_ckpt = resolve_resume_checkpoint(output_dir, args.resume_from_checkpoint)
    if args.use_wandb:
        maybe_load_wandb_resume_env(base_dir, resume_ckpt)

    num_steps = args.num_steps
    subset_size = args.subset_size
    num_epochs = args.num_train_epochs

    interval = args.delta_log_interval
    end = args.delta_log_end_step
    if end is None:
        end = min(num_steps, max(interval, num_steps // 10))
    else:
        end = min(num_steps, end)
    checkpoint_schedule = list(range(interval, end + 1, interval))
    if not checkpoint_schedule and num_steps > 0:
        checkpoint_schedule = [num_steps]

    # setdefault, not assignment: the sbatches export WANDB_PROJECT and this used to
    # overwrite it, so runs landed in "huggingface" regardless of what the job asked for.
    # Unset -> unchanged behaviour.
    wandb_project = os.environ.setdefault("WANDB_PROJECT", "huggingface")
    wandb_run_name = args.run_name if args.run_name else f"{model_name_sanitized}_{dataset_sanitized}_dpo_{num_steps}steps"

    print(f"Delta (warm-mask) schedule: {checkpoint_schedule}")
    if resume_ckpt:
        print(f"Resume: {resume_ckpt!r} — skipping weight-delta callback (base_state would not match a cold start).")

    from src.utils.dataset_registry import load_dpo_dataset as registry_load_dpo

    raw_ds = registry_load_dpo(dataset_key, subset_size=subset_size)
    train_dataset = raw_ds
    eval_dataset = None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def dpo_collator_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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
            c_ids, c_mask = pad_stack("chosen_input_ids")
            r_ids, r_mask = pad_stack("rejected_input_ids")
            return {
                "prompt_input_ids": p_ids,
                "prompt_attention_mask": p_mask,
                "chosen_input_ids": c_ids,
                "chosen_attention_mask": c_mask,
                "rejected_input_ids": r_ids,
                "rejected_attention_mask": r_mask,
            }

        prompts = [ex.get("prompt", "") for ex in examples]
        chosens = [ex.get("chosen", "") for ex in examples]
        rejects = [ex.get("rejected", "") for ex in examples]

        _mpl = args.max_prompt_length
        _ml = args.max_length
        enc_prompt = [tokenizer(p, truncation=True, max_length=_mpl, return_tensors="pt") for p in prompts]
        enc_chosen = [tokenizer(c, truncation=True, max_length=_ml, return_tensors="pt") for c in chosens]
        enc_reject = [tokenizer(r, truncation=True, max_length=_ml, return_tensors="pt") for r in rejects]

        batch_prompt = tokenizer.pad(enc_prompt, padding=True, return_tensors="pt", pad_to_multiple_of=8)
        batch_chosen = tokenizer.pad(enc_chosen, padding=True, return_tensors="pt", pad_to_multiple_of=8)
        batch_reject = tokenizer.pad(enc_reject, padding=True, return_tensors="pt", pad_to_multiple_of=8)

        for k in ("input_ids", "attention_mask"):
            batch_prompt[k] = batch_prompt[k].to(torch.long)
            batch_chosen[k] = batch_chosen[k].to(torch.long)
            batch_reject[k] = batch_reject[k].to(torch.long)

        return {
            "prompt_input_ids": batch_prompt["input_ids"],
            "prompt_attention_mask": batch_prompt["attention_mask"],
            "chosen_input_ids": batch_chosen["input_ids"],
            "chosen_attention_mask": batch_chosen["attention_mask"],
            "rejected_input_ids": batch_reject["input_ids"],
            "rejected_attention_mask": batch_reject["attention_mask"],
        }

    if args.fsdp and args.device_map is not None:
        raise ValueError("--fsdp and --device_map are mutually exclusive: FSDP shards "
                         "params (needs torchrun), device_map pins whole layers per GPU.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
    )
    model.config.use_cache = False

    # device_map="auto": spread layers across GPUs (model parallelism, single
    # process). DPOTrainer won't auto-place its internal reference model under
    # device_map, so build one explicitly here. Default None path is unchanged.
    #
    # Under LoRA there is NO explicit ref model: TRL uses the base model with the
    # adapter disabled as the implicit reference. Building one anyway would both
    # double memory and give DPOTrainer a second, conflicting reference.
    ref_model = None
    if args.device_map is not None and peft_config is None:
        print(f"device_map={args.device_map}: building explicit reference model (model-parallel).")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
        )
        ref_model.config.use_cache = False
    elif args.device_map is not None and peft_config is not None:
        print("LoRA + device_map: no explicit ref model (TRL disables the adapter for the reference).")

    _grad_ckpt = args.gradient_checkpointing
    if args.no_gradient_checkpointing:
        _grad_ckpt = False
    if args.fsdp:
        # under FSDP use fsdp_config activation_checkpointing instead (set above);
        # leaving TrainingArguments gradient_checkpointing on double-checkpoints + warns.
        _grad_ckpt = False
    if peft_config is not None and _grad_ckpt:
        # With gradient checkpointing the base model's embedding output is produced
        # under no_grad and the frozen base weights require no grad, so nothing in the
        # checkpointed segment carries requires_grad and the adapter receives zero
        # gradient — the run trains to nothing without erroring. Forcing the embedding
        # output to require grad reconnects the graph.
        model.enable_input_require_grads()
        print("LoRA + gradient checkpointing: enabled input require_grads (adapter grad flow).")

    save_steps_arg = args.save_steps
    save_total_limit_arg = args.save_total_limit if args.save_total_limit is not None else 3
    use_hf_rolling = save_steps_arg is not None and save_steps_arg > 0 and save_steps_arg < 10**9

    if use_hf_rolling:
        save_strategy = "steps"
        hf_save_steps = save_steps_arg
        hf_save_total_limit = save_total_limit_arg
    else:
        save_strategy = "no"
        hf_save_steps = 500
        hf_save_total_limit = None

    # FSDP: full shard + wrap each transformer decoder layer. All GPUs stay busy
    # (fast, no RC idle-cancel). Launch this script via torchrun --nproc_per_node=N.
    _fsdp_kwargs = {}
    if args.fsdp:
        _fsdp_kwargs = dict(
            fsdp="full_shard auto_wrap",
            fsdp_config={
                "transformer_layer_cls_to_wrap": [args.fsdp_layer_cls],
                # save consolidated full checkpoints so checkpoint_diff_mask_finder
                # (oracle) can load them as a normal HF model dir.
                "state_dict_type": "FULL_STATE_DICT",
                # use FSDP-native activation checkpointing (NOT TrainingArguments
                # gradient_checkpointing, which adds a redundant backward AllGather
                # and spikes memory under full_shard).
                "activation_checkpointing": True,
            },
            # Save ONLY model weights, not optimizer state. (1) the oracle only needs
            # model weights; (2) FSDP gathering bitsandbytes 8-bit optimizer state for
            # a full state_dict crashes ("tensors on cuda:0 and cuda:1"). adamw_8bit
            # keeps optimizer memory low enough to fit; skipping its save dodges the bug.
            save_only_model=True,
            # Precompute reference log-probs once, then DROP the ref model. Under FSDP
            # the DPO ref model is NOT sharded — at 32B it sits full (~66GB) on every
            # GPU and OOMs. Precomputing frees it so training only holds the (sharded)
            # policy + optimizer.
            precompute_ref_log_probs=True,
        )
        print(f"FSDP enabled: full_shard auto_wrap, wrap={args.fsdp_layer_cls}")

    cfg = DPOConfig(
        **_fsdp_kwargs,
        output_dir=output_dir,
        run_name=wandb_run_name,
        report_to=["wandb"] if args.use_wandb else [],
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        max_steps=-1 if num_epochs is not None else num_steps,
        num_train_epochs=num_epochs if num_epochs is not None else 1,
        bf16=True,
        fp16=False,
        optim=args.optim,
        gradient_checkpointing=_grad_ckpt,
        logging_steps=1,
        save_strategy=save_strategy,
        save_steps=hf_save_steps,
        save_total_limit=hf_save_total_limit,
        remove_unused_columns=False,
        beta=args.dpo_beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    print(
        f"DPO training: max_steps={cfg.max_steps}, num_train_epochs={cfg.num_train_epochs}, "
        f"peak_lr={args.learning_rate}, warmup_ratio={args.warmup_ratio}, lr_scheduler=linear"
    )
    print(f"HF rolling checkpoints: {use_hf_rolling} (save_steps={save_steps_arg}, limit={hf_save_total_limit})")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=dpo_collator_fn,
        peft_config=peft_config,
    )

    # Trainable-parameter accounting. This is the axis the LoRA baseline is compared
    # on (LoRA r=64 ≈ 2.1% vs a 97.5%-sparse subnetwork's 2.5%), so record it rather
    # than quoting it from a formula.
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    trainable_pct = 100.0 * n_trainable / n_total if n_total else float("nan")
    print(f"\nTrainable parameters: {n_trainable:,} / {n_total:,} ({trainable_pct:.4f}%)")
    if peft_config is not None:
        print(f"LoRA: r={args.lora_r}, alpha={lora_fields['lora_alpha']}, "
              f"targets={lora_fields['lora_target_modules']}, lr={args.learning_rate:g}")

    manifest = {
        "model_name": model_name,
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "subset_size": subset_size,
        "num_steps": num_steps,
        "learning_rate": args.learning_rate,
        "dpo_beta": args.dpo_beta,
        "optim": args.optim,
        "output_dir": output_dir,
        "resume_from_checkpoint": resume_ckpt,
        "hf_rolling_save_steps": save_steps_arg,
        "hf_save_total_limit": hf_save_total_limit if use_hf_rolling else None,
        "trainable_params": n_trainable,
        "total_params": n_total,
        "trainable_pct": trainable_pct,
        **lora_fields,
    }
    trainer.add_callback(RunManifestCallback(base_dir, manifest))

    if args.use_wandb:
        trainer.add_callback(WandbRunIdCallback(base_dir))

    class FlexibleCheckpointCallback(TrainerCallback):
        """Saves weight deltas vs θ(0) for warm-start masks (omit when resuming from HF checkpoint)."""

        def __init__(
            self,
            base_state: Dict[str, torch.Tensor],
            delta_log_dir: str,
            checkpoint_schedule: List[int],
            wandb_project_name: str,
        ):
            self.base_state = base_state
            self.delta_log_dir = delta_log_dir
            self.checkpoint_schedule = set(checkpoint_schedule)
            self.wandb_project_name = wandb_project_name
            os.makedirs(self.delta_log_dir, exist_ok=True)
            self.wandb_initialized = False

        def on_train_begin(self, train_args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return
            if not self.wandb_initialized and "wandb" in train_args.report_to:
                wandb.init(
                    project=self.wandb_project_name,
                    name=train_args.run_name,
                    config={
                        "model_name": model_name,
                        "dataset": dataset_name,
                        "subset_size": subset_size,
                        "learning_rate": train_args.learning_rate,
                        "batch_size_per_device": train_args.per_device_train_batch_size,
                        "grad_accum": train_args.gradient_accumulation_steps,
                        "checkpoint_schedule": sorted(list(self.checkpoint_schedule)),
                    },
                )
                self.wandb_initialized = True

        def on_step_end(self, train_args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return control
            train_model = kwargs["model"]
            step = state.global_step
            if step in self.checkpoint_schedule:
                full_deltas_to_save = {}
                with torch.no_grad():
                    for name, param in train_model.named_parameters():
                        current = param.detach().float().cpu()
                        diff = current - self.base_state[name]
                        full_deltas_to_save[name] = diff
                delta_file = os.path.join(self.delta_log_dir, f"deltas_step_{step}.pt")
                torch.save(full_deltas_to_save, delta_file)
                print(f"  ✓ Saved weight deltas at step {step}")
            return control

        def on_train_end(self, train_args, state, control, **kwargs):
            if state.is_world_process_zero and self.wandb_initialized:
                wandb.finish()

    # FSDP renames params ('_fsdp_wrapped_module.…') and shards them, so the
    # per-param weight-delta callback (built from pre-wrap names) KeyErrors. The
    # oracle mask is built from checkpoint-diffs (initial vs ckpt-500), not these
    # delta logs, so just skip delta logging under FSDP.
    # LoRA: the delta callback exists to build warm-start magnitude masks from |θ^k − θ^0|
    # over the FULL parameter vector. Under PEFT the base weights never move (only the
    # adapter does), so every delta would be zero for the tensors the mask cares about,
    # and named_parameters() carries the PEFT wrapper prefixes anyway. Skip it.
    if not resume_ckpt and not args.fsdp and peft_config is None:
        base_state: Dict[str, torch.Tensor] = {}
        if trainer.is_world_process_zero():
            with torch.no_grad():
                for name, param in trainer.model.named_parameters():
                    base_state[name] = param.detach().float().cpu().clone()
            os.makedirs(delta_log_dir, exist_ok=True)
            torch.save(base_state, os.path.join(delta_log_dir, "base_state.pt"))
        trainer.add_callback(
            FlexibleCheckpointCallback(
                base_state=base_state,
                delta_log_dir=delta_log_dir,
                checkpoint_schedule=checkpoint_schedule,
                wandb_project_name=wandb_project,
            )
        )
    elif args.fsdp:
        print("FSDP: skipping weight-delta callback (oracle uses checkpoint-diff, not deltas).")
    elif peft_config is not None:
        print("LoRA: skipping weight-delta callback (base weights are frozen; no warm-start mask).")

    print(f"\n{'=' * 60}")
    print("Starting DPO training")
    print(f"{'=' * 60}")
    if not resume_ckpt:
        print(f"Delta checkpoints (warm masks) at steps: {checkpoint_schedule}")
    print(f"Total steps target: {num_steps}")
    print(f"{'=' * 60}\n")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    class SegmentStepCounter(TrainerCallback):
        """Counts steps executed in THIS process only.

        train_runtime covers only the resumed segment while global_step is cumulative,
        so train_runtime/global_step understates s/step on any requeued run. Since the
        LoRA arm exists to make a wall-clock claim, that error would land in our favour
        and nobody would question it. Count the segment explicitly instead.
        """

        def __init__(self):
            self.steps = 0

        def on_step_end(self, train_args, state, control, **kwargs):
            self.steps += 1
            return control

    seg_counter = SegmentStepCounter()
    trainer.add_callback(seg_counter)

    train_output = trainer.train(resume_from_checkpoint=resume_ckpt)

    # Efficiency summary. The PEFT-vs-sparse comparison is reported on trainable
    # params, wall-clock per step, and peak VRAM, so emit all three from the run
    # itself instead of reconstructing them from logs afterwards.
    metrics = dict(getattr(train_output, "metrics", {}) or {})
    runtime_s = metrics.get("train_runtime")
    steps_done = trainer.state.global_step or 0
    seg_steps = seg_counter.steps

    # DPO at an aggressive LR can drive chosen AND rejected log-probs down together while
    # the margin still grows (likelihood displacement — Razin et al., ICLR 2025). The
    # margin is our headline metric, so a blown-up arm could post the best number in the
    # table and generate garbage. Record the two log-probs so that is visible.
    last_logps = {}
    for entry in reversed(trainer.state.log_history or []):
        if "logps/chosen" in entry:
            last_logps = {
                "logps_chosen": entry.get("logps/chosen"),
                "logps_rejected": entry.get("logps/rejected"),
                "rewards_chosen": entry.get("rewards/chosen"),
                "rewards_rejected": entry.get("rewards/rejected"),
                "rewards_margin": entry.get("rewards/margins"),
                "rewards_accuracy": entry.get("rewards/accuracies"),
                "logged_at_step": entry.get("step"),
            }
            break

    summary = {
        "run_name": wandb_run_name,
        "model_name": model_name,
        "dataset_key": dataset_key,
        "lora": peft_config is not None,
        "trainable_params": n_trainable,
        "total_params": n_total,
        "trainable_pct": trainable_pct,
        "learning_rate": args.learning_rate,
        "steps_completed": steps_done,
        "steps_this_segment": seg_steps,
        "resumed": bool(resume_ckpt),
        "train_runtime_s": runtime_s,
        # runtime covers THIS segment only -> divide by this segment's steps, not global_step
        "sec_per_step": (runtime_s / seg_steps) if runtime_s and seg_steps else None,
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None,
        "final_loss": metrics.get("train_loss"),
        # NOTE: these are TRAINING-batch metrics logged by TRL, not held-out. Use
        # src/evaluation/preference_eval.py for the held-out numbers.
        "train_batch_metrics": last_logps,
        **lora_fields,
    }
    if trainer.is_world_process_zero():
        summary_path = os.path.join(output_dir, "efficiency_summary.json")
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nEfficiency summary → {summary_path}")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}")
    print(f"Deltas dir: {delta_log_dir}")
    print(f"HF checkpoints: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
