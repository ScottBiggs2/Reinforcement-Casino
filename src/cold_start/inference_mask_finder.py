#!/usr/bin/env python3
"""Build cold-start masks from a small calibration set with CAV or SNIP."""

import os
import sys
import argparse
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.cold_start.utils.activation_hooks import FeatureExtractor, infer_model_input_device
from src.cold_start.utils.cav_probes import CAVProbeScorer
from src.cold_start.utils.snip_scorer import SNIPScorer

DPO_DATASET_NAME  = "qihoo360/Light-R1-DPOData"
GRPO_DATASET_NAME = "open-r1/OpenR1-Math-220k"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _msg_to_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("value", "")
    if isinstance(x, list):
        return "\n".join(m.get("value", "") for m in x if isinstance(m, dict))
    return str(x)


def load_calibration_samples_dpo(dataset_name, n_samples=64, seed=42):
    """Load chosen/rejected pairs from a DPO-format dataset.

    Returns (chosen_texts, rejected_texts) where each entry is prompt+response.
    """
    print(f"[DPO mode] Loading {n_samples} calibration samples from {dataset_name}...")
    raw = load_dataset(dataset_name, split="train")
    raw = raw.shuffle(seed=seed)

    chosen_texts   = []
    rejected_texts = []

    for rec in raw:
        if len(chosen_texts) >= n_samples:
            break

        prompt_raw   = rec.get("prompt", "")
        chosen_raw   = rec.get("chosen", "")
        rejected_raw = rec.get("rejected", "")

        # Handle nested conversation lists (e.g. Light-R1-DPOData uses 'conversations')
        if not prompt_raw:
            convs = rec.get("conversations", [])
            if isinstance(convs, list):
                human_parts = [
                    m.get("value", "") for m in convs
                    if isinstance(m, dict)
                    and m.get("from", m.get("role", "")).lower() in ("human", "user", "system")
                ]
                prompt_raw = "\n".join(human_parts).strip()

        if isinstance(prompt_raw, list):
            prompt = "\n".join(
                m.get("value", "") for m in prompt_raw
                if isinstance(m, dict) and m.get("from", "").lower() != "assistant"
            ).strip()
        else:
            prompt = _msg_to_text(prompt_raw).strip()

        chosen   = _msg_to_text(chosen_raw).strip()
        rejected = _msg_to_text(rejected_raw).strip()

        if not chosen or not rejected:
            continue

        chosen_texts.append(prompt + "\n" + chosen)
        rejected_texts.append(prompt + "\n" + rejected)

    print(f"  Loaded {len(chosen_texts)} chosen / {len(rejected_texts)} rejected samples.")
    return chosen_texts, rejected_texts


def load_calibration_samples_grpo(dataset_name, n_samples=64, seed=42):
    """Load problem/solution pairs from a GRPO-format dataset.

    Returns (positive_texts, negative_texts) where:
      - positive_texts: prompt_i + solution_i  (matched pair)
      - negative_texts: prompt_i + solution_{i+1 mod N}  (mismatched pair)

    Both classes have equal-length, full texts so the CAV probe learns
    reasoning-content differences rather than sequence-length differences.
    For SNIP/Fisher only positive_texts are used.
    """
    print(f"[GRPO mode] Loading {n_samples} calibration samples from {dataset_name}...")
    raw = load_dataset(dataset_name, split="train")
    raw = raw.shuffle(seed=seed)

    INPUT_CANDIDATES  = ["problem", "query", "prompt", "input", "text"]
    OUTPUT_CANDIDATES = ["solution", "response", "completion", "answer", "output"]

    cols = raw.column_names
    input_col  = next((c for c in INPUT_CANDIDATES  if c in cols), None)
    output_col = next((c for c in OUTPUT_CANDIDATES if c in cols), None)

    if input_col is None or output_col is None:
        raise KeyError(
            f"Could not find input/output columns in GRPO dataset '{dataset_name}'. "
            f"Available: {cols}. "
            f"Tried inputs={INPUT_CANDIDATES}, outputs={OUTPUT_CANDIDATES}."
        )

    prompts   = []
    solutions = []

    for rec in raw:
        if len(prompts) >= n_samples:
            break

        prompt   = str(rec.get(input_col,  "") or "").strip()
        solution = str(rec.get(output_col, "") or "").strip()

        if not prompt or not solution:
            continue

        prompts.append(prompt)
        solutions.append(solution)

    n = len(prompts)
    # Positive: prompt_i + matched solution_i
    positive_texts = [prompts[i] + "\n" + solutions[i] for i in range(n)]
    # Negative: prompt_i + mismatched solution_{i+1 mod n}
    # Both are full-length texts; CAV learns reasoning content, not length.
    negative_texts = [prompts[i] + "\n" + solutions[(i + 1) % n] for i in range(n)]

    print(f"  Loaded {n} positive (prompt+matched solution) / "
          f"{n} negative (prompt+mismatched solution) samples.")
    return positive_texts, negative_texts


def load_calibration_samples(n_samples=64, seed=42, mode="dpo", dataset_name=None):
    """Dispatch to the appropriate loader based on training mode."""
    if mode == "dpo":
        ds = dataset_name or DPO_DATASET_NAME
        return load_calibration_samples_dpo(ds, n_samples=n_samples, seed=seed)
    elif mode == "grpo":
        ds = dataset_name or GRPO_DATASET_NAME
        return load_calibration_samples_grpo(ds, n_samples=n_samples, seed=seed)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'dpo' or 'grpo'.")

def save_masks(masks, output_path, metadata=None):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_dict = {"masks": masks}
    if metadata:
        save_dict["metadata"] = metadata
    torch.save(save_dict, output_path)

    total  = sum(m.numel() for m in masks.values())
    kept   = sum(m.sum().item() for m in masks.values())
    actual = 100.0 * (1.0 - kept / total) if total > 0 else 0.0
    print(f"\n[save_masks] Saved {len(masks)} masks → {output_path}")
    print(f"  Parameters covered : {total:,}")
    print(f"  Active (kept)      : {int(kept):,}")
    print(f"  Sparsity           : {actual:.2f}%")

def main(args):
    set_seed(args.seed)

    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Cold-Start Mask Finder  [method={args.method}  mode={args.mode}]")
    print(f"{'='*60}")
    print(f"  Model      : {args.model_name}")
    print(f"  Mode       : {args.mode}")
    print(f"  N samples  : {args.n_samples}")
    print(f"  Sparsity   : {args.sparsity}%")
    print(f"  Output     : {args.output}")
    print(f"  Seed       : {args.seed}")
    print(f"  Device     : {compute_device}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model.to(compute_device)
    model.config.use_cache = False
    input_device = infer_model_input_device(model)
    print(f"  Model on {next(model.parameters()).device}, dtype={model.dtype}")
    print(f"  Input device: {input_device}")

    chosen_texts, rejected_texts = load_calibration_samples(
        args.n_samples, seed=args.seed, mode=args.mode, dataset_name=args.dataset_name,
    )
    effective_dataset = args.dataset_name or (DPO_DATASET_NAME if args.mode == "dpo" else GRPO_DATASET_NAME)

    if args.method == "cav":
        extractor = FeatureExtractor()
        extractor.register(model)

        print("\n[CAV] Collecting chosen activations...")
        pos_acts = extractor.collect(
            model, tokenizer, chosen_texts, input_device,
            max_length=args.max_length, batch_size=args.batch_size,
        )

        print("[CAV] Collecting rejected activations...")
        neg_acts = extractor.collect(
            model, tokenizer, rejected_texts, input_device,
            max_length=args.max_length, batch_size=args.batch_size,
        )

        extractor.remove()

        print("[CAV] Training linear probes...")
        scorer = CAVProbeScorer()
        neuron_scores = scorer.score(pos_acts, neg_acts, mag_weight=args.mag_weight)
        masks = scorer.scores_to_masks(
            neuron_scores, model,
            sparsity_percent=args.sparsity,
            local_pool=args.local_pool,
        )

        metadata = {
            "method": "cav",
            "model_name": args.model_name,
            "n_samples": args.n_samples,
            "sparsity_percent": args.sparsity,
            "dataset": effective_dataset,
            "seed": args.seed,
            "mag_weight": args.mag_weight,
            "local_pool": args.local_pool,
        }

    elif args.method == "snip":
        print("[SNIP] Computing gradient saliency scores...")
        scorer = SNIPScorer()
        snip_scores = scorer.score(
            model, tokenizer, chosen_texts, input_device,
            max_length=args.max_length, batch_size=args.batch_size,
        )
        masks = scorer.scores_to_masks(snip_scores, sparsity_percent=args.sparsity, local_pool=args.local_pool)

        metadata = {
            "method": "snip",
            "model_name": args.model_name,
            "n_samples": args.n_samples,
            "sparsity_percent": args.sparsity,
            "dataset": effective_dataset,
            "seed": args.seed,
        }

    else:
        raise ValueError(f"Unknown method: {args.method}. Choose 'cav' or 'snip'.")

    save_masks(masks, args.output, metadata)
    print("\n[Done] Cold-start mask generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cold-start inference-only mask finder (CAV / SNIP)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument(
        "--method", type=str, default="cav", choices=["cav", "snip"],
        help="Scoring method: cav (linear probes) or snip (gradient saliency)",
    )
    parser.add_argument(
        "--mode", type=str, default="dpo", choices=["dpo", "grpo"],
        help=(
            "Training mode: 'dpo' loads chosen/rejected pairs from a DPO dataset; "
            "'grpo' loads prompt+solution pairs from a GRPO dataset (prompt-only used as "
            "the CAV negative class)."
        ),
    )
    parser.add_argument(
        "--dataset_name", type=str, default=None,
        help=(
            "HuggingFace dataset to use for calibration. "
            f"Defaults to '{DPO_DATASET_NAME}' in dpo mode and "
            f"'{GRPO_DATASET_NAME}' in grpo mode."
        ),
    )
    parser.add_argument("--n_samples", type=int, default=64,
                        help="Number of calibration pairs")
    parser.add_argument("--sparsity", type=float, default=90.0,
                        help="Target sparsity %%: percentage of weights zeroed out")
    parser.add_argument("--output", type=str, default="masks/cold_start_cav_90pct.pt",
                        help="Output .pt file path")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset sampling and scoring reproducibility")
    parser.add_argument(
        "--mag_weight", type=float, default=1.0,
        help=(
            "[CAV only] Weight of task-presence signal (mean |act| on chosen) "
            "relative to the discriminative CAV signal. "
            "0.0 = pure discriminative (original); 1.0 = equal blend (default). "
            "Higher values retain more shared-subnetwork weights."
        ),
    )
    parser.add_argument(
        "--local-pool", dest="local_pool", action="store_true",
        help=(
            "[CAV only] Use per-layer neuron selection instead of global cross-layer ranking. "
            "Default (off): one global threshold — high-signal layers keep more neurons, "
            "low-signal layers keep fewer, sparsity varies per layer. "
            "With --local-pool: each layer independently keeps keep_frac * intermediate_size "
            "neurons, giving uniform sparsity across layers."
        ),
    )

    main(parser.parse_args())
