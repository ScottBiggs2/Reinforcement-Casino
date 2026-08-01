"""
E1 for the GRPO cell: how much *GRPO* gradient energy sits inside a sparse subnetwork at θ⁰?

This is the true-objective sibling of grad_energy_capture.py. That instrument builds a DPO
gradient; scoring GRPO-derived masks against it answers the cross-objective question ("do
GRPO-selected coordinates carry DPO signal"). Here g is the gradient of the GRPO objective itself,
evaluated at the pretrained init, so phi(M) = ||g⊙M||₂ / ||g||₂ measures whether a mask sits where
the *GRPO* gradient energy is.

Mechanics differ from DPO in one place only: the GRPO loss is defined over generated completions
and their group-relative rewards, which GRPOTrainer produces internally during a training step. So
rather than hand-rolling a forward/backward (as the DPO path does over a dataloader), we let
GRPOTrainer run exactly one optimizer step and capture the accumulated gradient in
on_pre_optimizer_step — the moment after backward + gradient accumulation and before the optimizer
updates anything. Everything downstream (per-tensor energy, phi, both chance baselines, the
per-layer profile, the JSON/CSV) is imported unchanged from grad_energy_capture.py.

phi is a ratio of norms of the *same* g, so it is invariant to the overall scale of g. That means
(a) we do not need to normalize the accumulated gradient by sample count, and (b) the Trainer's
global-norm gradient clip — a single scalar rescale of all of g — cancels, so it is irrelevant
whether we capture before or after it.

Noise caveat, stated plainly: the DPO instrument averages g over 128 examples because gradient
noise biases phi toward chance (the branch point of the pre-registered interpretation). A GRPO
optimizer step sees far fewer unique prompts (num_generations completions per prompt), so we widen
the effective batch via --num_prompts (implemented by raising gradient_accumulation_steps) to pull
g away from a single-prompt estimate. Any residual noise is conservative for the "phi ≫ chance"
claim.

Usage:

  python src/analysis/grad_energy_capture_grpo.py \
      --model_name meta-llama/Llama-3.1-8B-Instruct --dataset math-220k \
      --grpo_reward_profile llama_cot --beta 0.025 --num_prompts 8 \
      --mask oracle_grpo=/scratch/$USER/rl_casino_masks_e1fixed/grpo_.../oracle_ckptdiff_sp97.5.pt \
      --mask random_grpo=/scratch/$USER/.../random_sp97.5_seed1234.pt \
      --out_dir /scratch/$USER/rl_casino_analysis/rebuttal_e1_e2/e1_grpo_llama31
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from src.utils.dataset_registry import load_grpo_dataset
from src.utils.grpo_rewards import get_grpo_reward_funcs, normalize_reward_profile
from src.utils.scratch_paths import default_hf_datasets_cache

# Reuse the whole scoring/reporting stack from the DPO instrument — identical semantics.
from src.analysis.grad_energy_capture import analyze_mask, tensor_energies


class _GradCapture(TrainerCallback):
    """Snapshot the accumulated gradient of 1-D/2-D weights at the first pre-optimizer point."""

    def __init__(self):
        self.grads = None
        self.two_d = None
        self.captured = False

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if self.captured:
            return control
        grads, two_d, seen_ptrs = {}, set(), set()
        for name, param in model.named_parameters():
            if param.grad is None or "weight" not in name or param.dim() not in (1, 2):
                continue
            ptr = param.data_ptr()
            if ptr in seen_ptrs:
                print(f"  NOTE: skipping tied parameter {name} (shares storage)")
                continue
            seen_ptrs.add(ptr)
            # Clone: the optimizer step / zero_grad is about to run right after this callback.
            grads[name] = param.grad.detach().clone()
            if param.dim() == 2:
                two_d.add(name)
        self.grads = grads
        self.two_d = two_d
        self.captured = True
        # One gradient is all we need; stop before the weights move off θ⁰.
        control.should_training_stop = True
        return control


def build_grpo_gradient(args, device):
    """One GRPO optimizer step at θ⁰, gradient captured pre-update. Returns (grads, two_d, meta, model)."""
    if args.dtype == "bfloat16" and device == "cpu":
        raise SystemExit(
            "bf16 requires a GPU. GRPO generation on CPU is not a realistic smoke test; run on GPU."
        )
    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    reward_prof = normalize_reward_profile(args.grpo_reward_profile)
    reward_funcs = get_grpo_reward_funcs(args.grpo_reward_profile)
    print(f"  GRPO reward profile = {reward_prof}", flush=True)

    # Effective batch: one optimizer step accumulates batch × grad_accum completions. With
    # num_generations completions per prompt, that is (batch × grad_accum / num_generations) unique
    # prompts. Solve grad_accum for the requested --num_prompts, keeping the validated recipe's
    # batch / num_generations / generation_batch_size (sparse_grpo_bsr.py defaults).
    seqs_needed = args.num_prompts * args.num_generations
    if seqs_needed % args.per_device_train_batch_size != 0:
        raise SystemExit(
            f"num_prompts×num_generations ({seqs_needed}) must be divisible by "
            f"per_device_train_batch_size ({args.per_device_train_batch_size})"
        )
    grad_accum = seqs_needed // args.per_device_train_batch_size
    print(f"  effective batch: {args.num_prompts} prompts × {args.num_generations} gens = "
          f"{seqs_needed} seqs  (batch={args.per_device_train_batch_size}, grad_accum={grad_accum})",
          flush=True)

    # Load enough prompts to cover one step with headroom for the sampler.
    subset = max(args.num_prompts * 8, 128)
    dataset = load_grpo_dataset(args.dataset, subset_size=subset)
    if len(dataset) < args.num_prompts:
        raise SystemExit(f"dataset {args.dataset} yielded {len(dataset)} prompts, need {args.num_prompts}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map=None
    )
    model.config.use_cache = False
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    cfg = GRPOConfig(
        output_dir=os.path.join(args.out_dir, "_trainer"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_steps=1,
        bf16=(args.dtype == "bfloat16"),
        gradient_checkpointing=args.gradient_checkpointing,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        remove_unused_columns=False,
        report_to="none",
        logging_steps=1,
        seed=args.seed,
        dataloader_num_workers=0,
    )

    cap = _GradCapture()
    # We capture g in on_pre_optimizer_step and stop, but HF still runs one optimizer.step() for the
    # step in flight. The Trainer's default AdamW would allocate exp_avg + exp_avg_sq — ~64 GB of
    # fp32 state for an 8B model — purely to be thrown away, which OOMs even a 141 GB H200 alongside
    # the model, activations and the cloned gradient. A stateless SGD allocates nothing; the update
    # itself is irrelevant since g was already snapshotted at θ⁰ before this step runs.
    dummy_opt = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
        callbacks=[cap],
        optimizers=(dummy_opt, None),
    )

    t0 = time.time()
    trainer.train()
    if not cap.captured or not cap.grads:
        raise SystemExit("no gradient captured (on_pre_optimizer_step never fired)")
    print(f"  captured GRPO gradient in {time.time()-t0:.0f}s", flush=True)

    grads, two_d = cap.grads, cap.two_d
    gnorm = float(torch.sqrt(sum(torch.linalg.vector_norm(g.float()).double() ** 2 for g in grads.values())))
    if not (gnorm > 0):
        raise SystemExit(f"captured gradient has non-positive norm ({gnorm}); reward/advantage likely degenerate")

    meta = {
        "objective": "grpo",
        "model_name": args.model_name,
        "dataset": args.dataset,
        "grpo_reward_profile": reward_prof,
        "num_prompts": args.num_prompts,
        "num_generations": args.num_generations,
        "generation_batch_size": args.generation_batch_size,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "max_completion_length": args.max_completion_length,
        "max_prompt_length": args.max_prompt_length,
        "beta": args.beta,
        "seed": args.seed,
        "dtype": args.dtype,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "n_grad_tensors": len(grads),
        "n_grad_tensors_2d": len(two_d),
        "grad_norm": gnorm,
        "grad_dtype": str(next(iter(grads.values())).dtype) if grads else None,
    }
    return grads, two_d, meta, model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset", default="math-220k", help="dataset_registry key (GRPO)")
    p.add_argument("--mask", action="append", default=[],
                   help="label=path, repeatable. All masks scored against the same GRPO gradient.")
    p.add_argument("--synthetic_mask_density", type=float, default=None,
                   help="Smoke test: also score an in-memory uniform random mask at this density.")
    p.add_argument("--grpo_reward_profile", default="llama_cot")
    p.add_argument("--num_prompts", type=int, default=8,
                   help="Unique prompts averaged into g. Widen to cut gradient noise (biases phi to chance).")
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--generation_batch_size", type=int, default=8)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--max_completion_length", type=int, default=2048)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--beta", type=float, default=0.025)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset_cache_dir", default=default_hf_datasets_cache())
    # analyze_mask() reads args.seed and args.synthetic_mask_density (synthetic branch only); both present.
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
    print(f"=== E1 GRPO gradient energy capture on {device} ===", flush=True)
    print(f"  model   : {args.model_name}")
    print(f"  dataset : {args.dataset}  (reward {args.grpo_reward_profile})", flush=True)

    grads, two_d, meta, model = build_grpo_gradient(args, device)
    print(f"  weight tensors with gradients: {len(grads)} ({len(two_d)} of them 2-D)", flush=True)

    den_all = tensor_energies(grads)
    total_energy = sum(den_all.values())
    total_elements = sum(int(g.numel()) for g in grads.values())
    energy_2d = sum(v for k, v in den_all.items() if k in two_d)
    elements_2d = sum(int(g.numel()) for k, g in grads.items() if k in two_d)
    print(f"  all weights : {total_elements:,} elements, ||g||² = {total_energy:.6e}")
    print(f"  2-D weights : {elements_2d:,} elements, ||g||² = {energy_2d:.6e} "
          f"({energy_2d / total_energy:.4f} of the total)", flush=True)

    results, all_rows = [], []
    for label, path in masks:
        print(f"\n--- {label}", flush=True)
        res, rows = analyze_mask(label, path, grads, two_d, den_all, args)
        results.append(res)
        all_rows.extend(rows)
        print(f"  tensors={res['n_tensors_covered']} ({res['n_tensors_2d']} 2-D)  "
              f"elements={res['n_elements_covered']:,}  keep={res['realized_keep_rate']:.5f}")
        print(f"  phi_2d  ={res['phi_2d']:.6f}   chance_2d(energy-weighted)="
              f"{res['chance_energy_weighted_2d']:.6f}"
              f"   ratio={res['phi_2d'] / res['chance_energy_weighted_2d']:.4f}", flush=True)

    out = {
        "gradient": meta,
        "grad_totals": {
            "n_tensors": len(grads),
            "n_tensors_2d": len(two_d),
            "n_elements": total_elements,
            "n_elements_2d": elements_2d,
            "grad_energy": total_energy,
            "grad_energy_2d": energy_2d,
        },
        "masks": results,
    }
    json_path = os.path.join(args.out_dir, "e1_grad_energy_grpo.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    csv_path = os.path.join(args.out_dir, "e1_per_tensor_grpo.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n=== summary (2-D scope: comparable across masks) ===")
    print(f"{'mask':<26} {'phi_2d':>10} {'chance_ew':>10} {'ratio':>8} {'tensors':>8} {'keep':>9}")
    for r in results:
        print(f"{r['label']:<26} {r['phi_2d']:>10.6f} {r['chance_energy_weighted_2d']:>10.6f} "
              f"{r['phi_2d']/r['chance_energy_weighted_2d']:>8.3f} {r['n_tensors_2d']:>8} "
              f"{r['realized_keep_rate_2d']:>9.5f}")
    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")
    print("\nVALIDITY GATE: phi_2d(random) must sit near chance_energy_weighted_2d. If it does not, "
          "the gradient is too noisy or the instrument is wrong — debug before reporting any phi.")


if __name__ == "__main__":
    main()
