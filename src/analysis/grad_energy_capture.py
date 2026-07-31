"""
E1: how much DPO gradient energy sits inside a sparse subnetwork at the pretrained init?

For a boolean mask M over 2-D weight tensors, this measures

    phi(M) = || g ⊙ M ||_2 / || g ||_2

where g is one DPO gradient evaluated at the base weights θ⁰, and both norms are restricted to
the tensor set the mask covers. A mask that selects coordinates carrying no gradient energy lands
at the chance baseline; a mask aligned with gradient magnitude lands far above it.

Why not read train/grad_norm from W&B: HF Trainer logs clip_grad_norm_(model.parameters(), ...),
a single global dense norm over every parameter. Every arm of a masked run logs the same number
(observed: step-1 grad_norm = 70.0 for the Olmo3 dense, random, magnitude and oracle-deltalog
arms alike), so that column carries no per-mask information. phi has to be measured directly.

g is the gradient of the mean loss over `--microbatches × --per_device_train_batch_size` examples
— by default 64 × 2 = 128, i.e. the same effective batch the compared training runs take their
first optimizer step on. A single 2-sample microbatch is a far noisier estimate of g, and noise
biases phi toward chance, which is exactly the branch point of the pre-registered interpretation.

Two chance baselines are reported, because they are not the same number:

  chance_uniform_global        = sqrt( Σ_t k_t / Σ_t n_t )
      The naive null. Assumes gradient energy is spread in proportion to element count.

  chance_energy_weighted      = sqrt( Σ_t (k_t/n_t)·||g_t||² / Σ_t ||g_t||² )
      E[phi²] under uniform random selection *within each tensor* at that tensor's realized keep
      rate. This is the null the validity gate should be read against: masks built with a
      per-layer floor (min_layer_keep_ratio) have per-tensor keep rates that are not uniformly
      1-ρ, and gradient energy is very unevenly distributed across tensors.

Usage:

  python src/analysis/grad_energy_capture.py \
      --model_name meta-llama/Llama-3.1-8B-Instruct --dataset tulu3 \
      --mask oracle_dpo_tulu3=/scratch/$USER/rl_casino_masks/.../checkpoint_diff_...pt \
      --mask random_dpo_tulu3=/scratch/$USER/rl_casino_masks/.../random_baseline_...pt \
      --out_dir /scratch/$USER/rl_casino_analysis/rebuttal_e1_e2/e1

Smoke test on a small model with an in-memory uniform random mask (expect phi ≈ sqrt(density),
since a synthetic mask has no per-layer floor):

  python src/analysis/grad_energy_capture.py \
      --model_name Qwen/Qwen3-0.6B --dataset tulu3 --microbatches 2 \
      --synthetic_mask_density 0.025 --out_dir /tmp/e1_smoke
"""

import argparse
import csv
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from src.utils.data_utils import dpo_collator_fn
from src.utils.dataset_registry import load_dpo_dataset
from src.utils.mask_utils import load_masks_file
from src.utils.scratch_paths import default_hf_datasets_cache

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _layer_index(name):
    """Transformer block index for a parameter name, or None for embed/lm_head."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def build_gradient(args, device):
    """One DPO backward at the pretrained init. Returns (grads, meta).

    grads maps parameter name -> bf16 gradient tensor on `device`, for 2-D weight tensors only,
    deduplicated by storage pointer so tied embed_tokens/lm_head is not double counted.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    n_samples = args.microbatches * args.per_device_train_batch_size
    dataset = load_dpo_dataset(args.dataset, subset_size=n_samples)
    if len(dataset) < n_samples:
        raise SystemExit(
            f"dataset {args.dataset} yielded {len(dataset)} examples, need {n_samples}"
        )

    if args.dtype == "bfloat16" and device == "cpu":
        raise SystemExit(
            "bf16 requires a GPU (transformers rejects bf16 on CPU). For a CPU smoke test pass "
            "--dtype float32; the real measurement must run on GPU in bf16 so the numeric path "
            "matches the training runs."
        )
    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    # Move the policy before DPOTrainer.__init__: with ref_model=None and no PEFT, TRL calls
    # create_reference_model(model), a deep copy that inherits whatever device we are on.
    model.to(device)

    if args.gradient_checkpointing:
        # Every compared training run had gradient checkpointing on, so leaving it on keeps the
        # numeric path identical at no cost. use_reentrant=False per snip_scorer.py:381.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    cfg = DPOConfig(
        output_dir=os.path.join(args.out_dir, "_trainer"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=1,
        max_steps=1,
        bf16=(args.dtype == "bfloat16"),
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        remove_unused_columns=False,
        report_to="none",
        logging_steps=1,
        seed=args.seed,
        # The real reference term must be in the loss. precompute_ref_log_probs=True would give a
        # cheaper run but a different objective from the one the arms actually trained on.
        precompute_ref_log_probs=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        data_collator=lambda x: dpo_collator_fn(x, tokenizer),
    )

    dl = trainer.get_train_dataloader()
    model.zero_grad(set_to_none=True)

    # Accumulate the gradient of the *sum* over examples, then divide by the total count once, so
    # g is the gradient of the mean loss over all n_samples regardless of ragged batch sizes.
    seen, losses, t0 = 0, [], time.time()
    for i, batch in enumerate(dl):
        if i >= args.microbatches:
            break
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        bs = int(batch["chosen_input_ids"].shape[0])
        loss = trainer.compute_loss(model, batch)
        (loss * float(bs)).backward()
        losses.append(float(loss.item()))
        seen += bs
        if (i + 1) % 8 == 0 or i + 1 == args.microbatches:
            print(f"  microbatch {i+1}/{args.microbatches}  seen={seen}  "
                  f"loss={losses[-1]:.6f}  ({time.time()-t0:.0f}s)", flush=True)

    if seen == 0:
        raise SystemExit("no batches consumed")

    grads, seen_ptrs = {}, set()
    for name, param in model.named_parameters():
        if param.grad is None or "weight" not in name or param.dim() != 2:
            continue
        ptr = param.data_ptr()
        if ptr in seen_ptrs:
            print(f"  NOTE: skipping tied parameter {name} (shares storage)")
            continue
        seen_ptrs.add(ptr)
        grads[name] = param.grad.detach().div_(float(seen))

    meta = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "n_samples": seen,
        "microbatches": args.microbatches,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "beta": args.beta,
        "seed": args.seed,
        "dtype": args.dtype,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "precompute_ref_log_probs": False,
        "mean_microbatch_loss": sum(losses) / len(losses),
        "microbatch_losses": losses,
        "n_grad_tensors_2d": len(grads),
        "grad_dtype": str(next(iter(grads.values())).dtype) if grads else None,
    }
    return grads, meta, model


def tensor_energies(grads):
    """Per-tensor ||g_t||^2 in float64. Computed once and shared across masks."""
    den = {}
    for name, g in grads.items():
        g32 = g.float()
        den[name] = float(torch.linalg.vector_norm(g32).double() ** 2)
        del g32
    return den


def masked_energy(grads, mask_tensors, names):
    """Per-tensor ||g_t ⊙ M_t||^2 in float64, using boolean indexing.

    Boolean indexing rather than vector_norm(g * mask): the latter materializes a second
    full-size tensor per call, which on embed_tokens is another 2.1 GB of fp32.
    """
    num = {}
    for name in names:
        g32 = grads[name].float()
        m = mask_tensors[name]
        if m.device != g32.device:
            m = m.to(g32.device, non_blocking=True)
        num[name] = float(torch.linalg.vector_norm(g32[m]).double() ** 2)
        del g32, m
    return num


def synthetic_mask(grads, density, seed):
    """Uniform random mask over every 2-D weight tensor. Smoke-test instrument only."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    masks = {}
    for name, g in grads.items():
        masks[name] = (torch.rand(g.shape, generator=gen) < density)
    return masks, {"synthetic": True, "density": density, "seed": seed}


def analyze_mask(label, mask_path, grads, den_all, args):
    """phi for one mask, plus its per-tensor and per-layer profile."""
    if mask_path == "__synthetic__":
        mask_tensors, mask_meta = synthetic_mask(grads, args.synthetic_mask_density, args.seed)
    else:
        print(f"  loading {mask_path}", flush=True)
        mask_tensors, mask_meta, wrapped = load_masks_file(mask_path)
        mask_meta = dict(mask_meta or {})
        mask_meta["wrapped_format"] = wrapped

    # Coverage gate: every mask key must name a real 2-D weight with a matching shape. A silent
    # key or shape mismatch would land in the denominator as if the tensor were unmasked.
    missing = [k for k in mask_tensors if k not in grads]
    mism = [
        (k, tuple(mask_tensors[k].shape), tuple(grads[k].shape))
        for k in mask_tensors
        if k in grads and tuple(mask_tensors[k].shape) != tuple(grads[k].shape)
    ]
    if missing or mism:
        raise SystemExit(
            f"mask {label} does not match the model:\n"
            f"  {len(missing)} keys absent from the 2-D weight set: {missing[:5]}\n"
            f"  {len(mism)} shape mismatches: {mism[:5]}"
        )

    names = sorted(mask_tensors)
    num = masked_energy(grads, mask_tensors, names)

    rows, sum_num, sum_den, sum_k, sum_n, sum_rate_energy = [], 0.0, 0.0, 0, 0, 0.0
    for name in names:
        m = mask_tensors[name]
        n_t = int(m.numel())
        k_t = int(m.sum().item())
        d_t = den_all[name]
        n_num = num[name]
        sum_num += n_num
        sum_den += d_t
        sum_k += k_t
        sum_n += n_t
        sum_rate_energy += (k_t / n_t) * d_t
        rows.append({
            "mask": label,
            "tensor": name,
            "layer": _layer_index(name),
            "n_elements": n_t,
            "k_kept": k_t,
            "keep_rate": k_t / n_t,
            "grad_energy": d_t,
            "masked_grad_energy": n_num,
            "phi_tensor": (n_num / d_t) ** 0.5 if d_t > 0 else None,
        })

    # Per-layer profile: aggregate energies within a transformer block, then take the ratio.
    # Aggregating energies (not averaging per-tensor phi) keeps the layer number interpretable as
    # "fraction of this block's gradient energy that the mask covers".
    per_layer = {}
    for r in rows:
        key = "embed_lm_head" if r["layer"] is None else str(r["layer"])
        agg = per_layer.setdefault(key, {"num": 0.0, "den": 0.0, "k": 0, "n": 0})
        agg["num"] += r["masked_grad_energy"]
        agg["den"] += r["grad_energy"]
        agg["k"] += r["k_kept"]
        agg["n"] += r["n_elements"]
    per_layer_out = {
        k: {
            "phi": (v["num"] / v["den"]) ** 0.5 if v["den"] > 0 else None,
            "keep_rate": v["k"] / v["n"],
            "grad_energy_share": v["den"] / sum_den if sum_den > 0 else None,
        }
        for k, v in per_layer.items()
    }

    result = {
        "label": label,
        "mask_path": mask_path,
        "n_tensors_covered": len(names),
        "n_elements_covered": sum_n,
        "k_elements_kept": sum_k,
        "realized_keep_rate": sum_k / sum_n,
        "phi": (sum_num / sum_den) ** 0.5 if sum_den > 0 else None,
        "chance_uniform_global": (sum_k / sum_n) ** 0.5,
        "chance_energy_weighted": (sum_rate_energy / sum_den) ** 0.5 if sum_den > 0 else None,
        "masked_grad_energy": sum_num,
        "grad_energy_covered": sum_den,
        "per_layer": per_layer_out,
        "mask_metadata": {
            k: v for k, v in (mask_meta or {}).items()
            if isinstance(v, (str, int, float, bool, type(None)))
        },
    }

    del mask_tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset", default="tulu3", help="dataset_registry key")
    p.add_argument("--mask", action="append", default=[],
                   help="label=path, repeatable. All masks are scored against the same gradient.")
    p.add_argument("--synthetic_mask_density", type=float, default=None,
                   help="Smoke test: also score an in-memory uniform random mask at this density.")
    p.add_argument("--microbatches", type=int, default=64)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16",
                   help="bfloat16 matches the training runs and needs a GPU. float32 exists so "
                        "the instrument can be smoke-tested on a CPU node.")
    p.add_argument("--gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset_cache_dir", default=default_hf_datasets_cache())
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_DATASETS_CACHE", args.dataset_cache_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    masks = []
    for spec in args.mask:
        if "=" not in spec:
            raise SystemExit(f"--mask expects label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        if not os.path.exists(path):
            raise SystemExit(f"mask file not found: {path}")
        masks.append((label, path))
    if args.synthetic_mask_density is not None:
        masks.append((f"synthetic_random_d{args.synthetic_mask_density}", "__synthetic__"))
    if not masks:
        raise SystemExit("no masks given (--mask label=path, or --synthetic_mask_density)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== E1 gradient energy capture on {device} ===", flush=True)
    print(f"  model   : {args.model_name}")
    print(f"  dataset : {args.dataset}")
    print(f"  g       : mean over {args.microbatches} × {args.per_device_train_batch_size} "
          f"= {args.microbatches * args.per_device_train_batch_size} examples", flush=True)

    grads, meta, model = build_gradient(args, device)
    print(f"  2-D weight tensors with gradients: {len(grads)}", flush=True)

    den_all = tensor_energies(grads)
    total_energy = sum(den_all.values())
    total_elements = sum(int(g.numel()) for g in grads.values())
    print(f"  total 2-D elements: {total_elements:,}   ||g||² = {total_energy:.6e}", flush=True)

    results, all_rows = [], []
    for label, path in masks:
        print(f"\n--- {label}", flush=True)
        res, rows = analyze_mask(label, path, grads, den_all, args)
        results.append(res)
        all_rows.extend(rows)
        print(f"  tensors={res['n_tensors_covered']}  elements={res['n_elements_covered']:,}  "
              f"keep={res['realized_keep_rate']:.5f}")
        print(f"  phi={res['phi']:.6f}   chance(energy-weighted)={res['chance_energy_weighted']:.6f}"
              f"   chance(uniform-global)={res['chance_uniform_global']:.6f}")
        print(f"  phi / chance_energy_weighted = "
              f"{res['phi'] / res['chance_energy_weighted']:.4f}", flush=True)

    out = {
        "gradient": meta,
        "grad_totals": {
            "n_tensors_2d": len(grads),
            "n_elements_2d": total_elements,
            "grad_energy_2d": total_energy,
        },
        "masks": results,
    }
    json_path = os.path.join(args.out_dir, "e1_grad_energy.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    csv_path = os.path.join(args.out_dir, "e1_per_tensor.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n=== summary ===")
    print(f"{'mask':<28} {'phi':>10} {'chance_ew':>10} {'ratio':>8} {'tensors':>8} {'keep':>9}")
    for r in results:
        print(f"{r['label']:<28} {r['phi']:>10.6f} {r['chance_energy_weighted']:>10.6f} "
              f"{r['phi']/r['chance_energy_weighted']:>8.3f} {r['n_tensors_covered']:>8} "
              f"{r['realized_keep_rate']:>9.5f}")
    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
