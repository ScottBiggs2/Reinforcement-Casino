import torch
import os
import json
import argparse
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.mask_utils import (
    DEFAULT_MIN_LAYER_KEEP_RATIO,
    create_mask_from_scores_gpu_efficient,
    compute_jaccard_similarity,
    pooling_metadata,
    save_masks,
)


def summarize_scores(scores):
    total_numel = 0
    total_sum = 0.0
    total_sq = 0.0
    global_min = None
    global_max = None
    nonzero = 0

    for score in scores.values():
        s = score.detach().float().reshape(-1).cpu()
        if s.numel() == 0:
            continue
        total_numel += s.numel()
        total_sum += float(s.sum().item())
        total_sq += float((s * s).sum().item())
        nonzero += int((s != 0).sum().item())
        local_min = float(s.min().item())
        local_max = float(s.max().item())
        global_min = local_min if global_min is None else min(global_min, local_min)
        global_max = local_max if global_max is None else max(global_max, local_max)

    if total_numel == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "nonzero_fraction": 0.0,
        }

    mean = total_sum / total_numel
    variance = max(total_sq / total_numel - mean * mean, 0.0)
    return {
        "min": global_min,
        "max": global_max,
        "mean": mean,
        "std": variance ** 0.5,
        "nonzero_fraction": nonzero / total_numel,
    }

# ============================================================
# Fisher Information Cold-Start Mask Finder
# ============================================================
# Goal: Compute a sparse mask over MLP parameters *before*
# any RL/DPO training, using the diagonal Fisher Information
# as a proxy for which parameters are task-relevant.
#
# Key insight: F_ii = E[(d log p(y|x,theta) / d theta_i)^2]
# Parameters with high Fisher scores are sensitive to the
# task distribution -- these are the ones likely to move
# during fine-tuning, so they form a good cold-start mask.
#
# Improvements over naive diagonal Fisher:
#   1. Empirical Fisher: use ground-truth DPO 'chosen' labels
#      instead of sampling from the model's own distribution.
#   2. Mini-batch gradient smoothing: average gradients over
#      a mini-batch before squaring, suppressing per-sample
#      noise spikes.
#   3. Full-sequence NLL: backward through the entire chosen
#      sequence rather than just the last token, using all
#      available gradient signal.
#   4. Per-layer normalization: z-score each layer's Fisher
#      scores before global thresholding, counteracting layer-
#      depth gradient attenuation (early layers are otherwise
#      systematically under-selected).
# ============================================================


def sanitize_model_name(model_name: str) -> str:
    """Mirrors the sanitizer in the training script for consistent naming."""
    sanitized = model_name.replace("/", "_").replace("-", "_").lower()
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


def load_calibration_data(dataset_name, tokenizer, n_samples=512, max_length=512, device="cuda"):
    """
    Load a small calibration set from the task DPO dataset.
    Returns a list of encoded dicts with input_ids, attention_mask, and labels
    (prompt tokens masked to -100, response tokens used for loss).

    Dataset schema (qihoo360/Light-R1-DPOData):
        'conversations': list of {"from": "human"|"gpt", "value": str}
        'chosen':        dict  {"from": "gpt", "value": str}
        'rejected':      dict  {"from": "gpt", "value": str}
    The prompt is the human turn(s) from 'conversations'; chosen is the response.
    """
    print(f"Loading calibration data from {dataset_name} ({n_samples} samples)...")
    ds = load_dataset(dataset_name, split="train")
    ds = ds.select(range(min(n_samples, len(ds))))

    def extract_prompt(rec):
        """Extract the human/user prompt from the conversations field."""
        convs = rec.get("conversations", None)
        if convs and isinstance(convs, list):
            human_parts = [
                m["value"] for m in convs
                if isinstance(m, dict)
                and m.get("from", m.get("role", "")).lower() in ("human", "user", "system")
                and "value" in m
            ]
            if human_parts:
                return "\n".join(human_parts).strip()
        # Fallback: try the 'prompt' column (other datasets)
        prompt_raw = rec.get("prompt", "")
        if isinstance(prompt_raw, str):
            return prompt_raw.strip()
        return ""

    def extract_chosen(rec):
        """Extract the preferred response text."""
        chosen = rec.get("chosen", "")
        if isinstance(chosen, dict):
            return chosen.get("value", "").strip()
        if isinstance(chosen, str):
            return chosen.strip()
        if isinstance(chosen, list):
            # list of message dicts — take any gpt/assistant turn
            parts = [m["value"] for m in chosen if isinstance(m, dict) and "value" in m]
            return "\n".join(parts).strip()
        return ""

    pairs = []
    for rec in ds:
        prompt_text = extract_prompt(rec)
        chosen_text = extract_chosen(rec)

        if not prompt_text or not chosen_text:
            continue

        # Encode the full prompt+response sequence with response-only labels
        full_text = prompt_text + " " + chosen_text
        full_enc = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=max_length,
        )
        prompt_enc = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=max_length,
        )

        # Mask prompt tokens so loss is computed only over the chosen response
        prompt_len = prompt_enc["input_ids"].shape[1]
        labels = full_enc["input_ids"].clone()
        labels[0, :prompt_len] = -100

        pairs.append({
            "input_ids": full_enc["input_ids"].to(device),
            "attention_mask": full_enc["attention_mask"].to(device),
            "labels": labels.to(device),
        })

    print(f"  Loaded {len(pairs)} calibration (prompt, chosen) pairs")
    return pairs





def compute_fisher_scores(
    model,
    calibration_data,
    device="cuda",
    mlp_only=False,
    mini_batch_size=4,
    normalize_per_layer=True,
):
    """
    Compute improved diagonal Fisher Information scores over model parameters.

    Improvements over naive diagonal Fisher:
      1. EMPIRICAL FISHER: uses actual 'chosen' tokens as labels (target distribution)
         rather than sampling from the model's own current distribution.
      2. FULL-SEQUENCE GRADIENTS: backward through the entire chosen response,
         not just the last token, utilizing all available gradient signal.
      3. MINI-BATCH GRADIENT SMOOTHING: accumulates gradients across a mini-batch,
         then squares the averaged gradient. This suppresses per-sample noise spikes
         (E[g]^2 is lower variance than E[g^2]).
      4. PER-LAYER NORMALIZATION: z-scores each layer's Fisher scores before
         global thresholding, counteracting gradient attenuation in earlier layers.

    Args:
        mini_batch_size: Number of examples to average gradients over before
            squaring. Larger=lower variance, but slower. 4-8 is usually optimal.
        normalize_per_layer: If True, z-score each tensor's scores before global
            thresholding. Strongly recommended to avoid late-layer bias.
    """
    print(f"\n=== Computing Improved Empirical Fisher Information ===")
    print(f"  Calibration samples:       {len(calibration_data)}")
    print(f"  MLP parameters only:       {mlp_only}")
    print(f"  Mini-batch size:           {mini_batch_size}")
    print(f"  Per-layer normalization:   {normalize_per_layer}")

    model.eval()

    MLP_KEYWORDS = ["gate_proj", "up_proj", "down_proj", "fc1", "fc2",
                    "feed_forward", "ffn", "mlp.c_fc", "mlp.c_proj"]

    def is_mlp_param(name):
        return any(kw in name.lower() for kw in MLP_KEYWORDS)

    # Initialize accumulator
    fisher_scores = {}
    scored_params = {}
    for name, param in model.named_parameters():
        if mlp_only and not is_mlp_param(name):
            continue
        if not param.requires_grad:
            continue
        fisher_scores[name] = torch.zeros_like(param, dtype=torch.float32)
        scored_params[name] = param

    print(f"  Scoring {len(fisher_scores)} parameter tensors")

    # Process in mini-batches for gradient smoothing
    n_batches = 0
    batch_start = 0
    total_samples = len(calibration_data)

    while batch_start < total_samples:
        batch_end = min(batch_start + mini_batch_size, total_samples)
        batch = calibration_data[batch_start:batch_end]
        batch_size = len(batch)

        if batch_start % (mini_batch_size * 10) == 0:
            print(f"  [{batch_start+1}/{total_samples}] Processing mini-batch...")

        model.zero_grad()

        # IMPROVEMENT 1 & 2: Full-sequence Empirical Fisher
        # Accumulate loss across the mini-batch, then do a single backward
        total_loss = None
        for enc in batch:
            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=enc["labels"],
            )
            # Loss is already averaged over non-masked tokens in the sequence
            loss = outputs.loss / batch_size
            if total_loss is None:
                total_loss = loss
            else:
                total_loss = total_loss + loss

        # IMPROVEMENT 3: Backward on averaged loss, then square
        # This gives E[g]^2 (low variance) rather than E[g^2] (high variance)
        total_loss.backward()

        with torch.no_grad():
            for name, param in scored_params.items():
                if param.grad is not None:
                    fisher_scores[name] += param.grad.float().pow(2)

        n_batches += 1
        batch_start = batch_end

    # Normalize by number of mini-batches
    print(f"\n  Normalizing by {n_batches} mini-batches ({total_samples} samples)...")
    for name in fisher_scores:
        fisher_scores[name] /= n_batches

    # IMPROVEMENT 4: Per-layer z-score normalization
    # This prevents later layers from dominating due to higher raw gradient magnitude
    if normalize_per_layer:
        print("  Applying per-layer z-score normalization...")
        for name in fisher_scores:
            s = fisher_scores[name]
            mean = s.mean()
            std = s.std()
            if std > 1e-12:
                fisher_scores[name] = (s - mean) / std
            else:
                # If std is negligible, the layer is uniformly sensitive — set to 0
                fisher_scores[name] = torch.zeros_like(s)

    # Summary stats
    stats = summarize_scores(fisher_scores)
    print(f"  Fisher score stats (after normalization):")
    print(f"    min:  {stats['min']:.6e}")
    print(f"    max:  {stats['max']:.6e}")
    print(f"    mean: {stats['mean']:.6e}")
    print(f"    std:  {stats['std']:.6e}")
    print(f"    nonzero fraction: {stats['nonzero_fraction']:.4f}")

    return fisher_scores


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print(f"Using device: {device}")

    print(f"\nLoading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    calibration_data = load_calibration_data(
        args.dataset_name,
        tokenizer,
        n_samples=args.n_calibration_samples,
        max_length=args.max_length,
        device=device,
    )

    fisher_scores = compute_fisher_scores(
        model,
        calibration_data,
        device=device,
        mlp_only=args.mlp_only,
        mini_batch_size=args.mini_batch_size,
        normalize_per_layer=not args.no_layer_norm,
    )

    masks = create_mask_from_scores_gpu_efficient(
        fisher_scores,
        args.sparsity_percent,
        device=device,
        min_layer_keep_ratio=args.min_layer_keep_ratio,
        local_pool=args.local_pool,
    )

    model_sanitized = sanitize_model_name(args.model_name)
    output_file = args.output_file or (
        f"masks/cold_fisher_{model_sanitized}_{args.dataset_name.replace('/', '_')}"
        f"_sparsity{args.sparsity_percent}pct"
        f"_n{args.n_calibration_samples}.pt"
    )

    jaccard_results = None
    if args.reference_mask:
        print(f"\nLoading reference mask from: {args.reference_mask}")
        ref = torch.load(args.reference_mask, map_location="cpu")
        reference_masks = ref["masks"]
        jaccard_results = compute_jaccard_similarity(masks, reference_masks)

    metadata = {
        "method": "fisher_cold_start_v2",
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "sparsity_percent": args.sparsity_percent,
        "n_calibration_samples": args.n_calibration_samples,
        "mini_batch_size": args.mini_batch_size,
        "layer_normalization": not args.no_layer_norm,
        "mlp_only": args.mlp_only,
        "device": device,
        **pooling_metadata(
            local_pool=args.local_pool,
            min_layer_keep_ratio=args.min_layer_keep_ratio,
        ),
    }

    if jaccard_results:
        metadata["jaccard_similarity"] = {
            "aggregate": jaccard_results["aggregate_jaccard"],
            "mean": jaccard_results["mean_jaccard"],
            "overlap_pred": jaccard_results.get("overlap_fraction_predicted"),
            "cosine": jaccard_results.get("cosine_similarity"),
        }

    save_masks(masks, output_file, metadata)

    if jaccard_results:
        jaccard_file = output_file.replace(".pt", "_jaccard.json")
        with open(jaccard_file, "w") as f:
            json.dump(jaccard_results, f, indent=2)
        print(f"Detailed similarity results saved to: {jaccard_file}")

    print("\n✓ Cold-start mask generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved Fisher Information cold-start mask generation")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--dataset_name", type=str, default="qihoo360/Light-R1-DPOData")
    parser.add_argument("--sparsity_percent", type=float, default=90.0)
    parser.add_argument("--n_calibration_samples", type=int, default=256,
                        help="Number of calibration examples. 128-512 is usually sufficient.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--mini_batch_size", type=int, default=4,
                        help="Mini-batch size for gradient smoothing before squaring. "
                             "Higher = lower variance but slower. Recommended: 4-8.")
    parser.add_argument("--no_layer_norm", action="store_true",
                        help="Disable per-layer z-score normalization. "
                             "Without this, later layers dominate due to larger gradients.")
    parser.add_argument(
        "--mlp_only",
        action="store_true",
        default=False,
        help="Only score MLP parameters (default: score all weights)",
    )
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--reference_mask", type=str, default=None,
                        help="Optional reference mask to compute similarity metrics against.")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument(
        "--min_layer_keep_ratio",
        type=float,
        default=DEFAULT_MIN_LAYER_KEEP_RATIO,
        help=(
            "Small per-tensor keep floor for hybrid global masking. "
            "Set to 0.0 for pure global selection."
        ),
    )
    parser.add_argument(
        "--local_pool", action="store_true",
        help=(
            "Use per-layer mask selection instead of global cross-layer ranking. "
            "Default (off): global masking with a small per-tensor keep floor. "
            "With --local_pool: each weight matrix independently keeps its top keep_frac elements."
        ),
    )
    args = parser.parse_args()
    main(args)

# Example usage:
# python src/cold_start/cold_mask_finder.py \
#   --model_name google/gemma-3-270m-it \
#   --dataset_name qihoo360/Light-R1-DPOData \
#   --sparsity_percent 97.5 \
#   --n_calibration_samples 256 \
#   --mini_batch_size 4 \
#   --reference_mask masks/warm_magnitude_google_gemma_3_270m_it_sparsity97.5pct.pt