import torch
import os
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.utils.mask_utils import (
    create_mask_from_scores_gpu_efficient,
    compute_jaccard_similarity,
    save_masks
)

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
# Output format mirrors gpu_mask_finder.py so masks slot
# directly into the existing sparse kernel infrastructure.
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
    Load a small calibration set from the task dataset.
    We only need prompts -- we'll sample completions from the model itself.
    
    Uses the same dataset format as the DPO training script (prompt/chosen/rejected),
    but only the prompt field is needed for Fisher computation.
    """
    print(f"Loading calibration data from {dataset_name} ({n_samples} samples)...")
    ds = load_dataset(dataset_name, split="train")
    ds = ds.select(range(min(n_samples, len(ds))))

    # Reuse the same normalization logic as the training script
    def msg_to_text(x):
        if isinstance(x, str): return x
        if isinstance(x, dict): return x.get("value", "")
        if isinstance(x, list): return "\n".join(m.get("value", "") for m in x if isinstance(m, dict))
        return str(x)

    prompts = []
    for rec in ds:
        prompt_raw = rec.get("prompt", "")
        if isinstance(prompt_raw, list):
            text = "\n".join(
                m.get("value", "") for m in prompt_raw
                if isinstance(m, dict) and m.get("from", "").lower() != "assistant"
            ).strip()
        else:
            text = msg_to_text(prompt_raw).strip()
        prompts.append(text)

    # Tokenize all prompts; we'll iterate as individual tensors (no batching needed,
    # Fisher accumulation is inherently a loop anyway)
    encodings = []
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=max_length)
        encodings.append({k: v.to(device) for k, v in enc.items()})

    print(f"  Loaded {len(encodings)} calibration prompts")
    return encodings


def compute_fisher_scores(model, calibration_data, device="cuda", mlp_only=True):
    """
    Compute diagonal Fisher Information scores over model parameters.

    For each calibration example:
      1. Forward pass to get logits
      2. Sample a completion token from the model's OWN distribution
         (this is the proper Fisher estimator -- NOT using ground truth labels)
      3. Backward pass to get gradients
      4. Accumulate squared gradients

    Fisher_ii = (1/N) * sum_n [ (d log p / d theta_i)^2 ]

    Args:
        mlp_only: If True, only compute Fisher for MLP layers (gate_proj, up_proj,
                  down_proj, fc1, fc2, ffn -- whatever naming the model uses).
                  This keeps memory manageable and is consistent with the rest of
                  the project targeting MLP sparse backprop.
    """
    print(f"\n=== Computing Diagonal Fisher Information ===")
    print(f"  Calibration samples: {len(calibration_data)}")
    print(f"  MLP parameters only: {mlp_only}")

    model.eval()

    # Identify which parameters to score
    # MLP layer name patterns -- covers LLaMA, Gemma, Mistral, Qwen naming conventions
    MLP_KEYWORDS = ["gate_proj", "up_proj", "down_proj", "fc1", "fc2",
                    "feed_forward", "ffn", "mlp.c_fc", "mlp.c_proj"]

    def is_mlp_param(name):
        return any(kw in name.lower() for kw in MLP_KEYWORDS)

    # Initialize accumulator dict -- only for params we care about
    fisher_scores = {}
    for name, param in model.named_parameters():
        if mlp_only and not is_mlp_param(name):
            continue
        if not param.requires_grad:
            continue
        fisher_scores[name] = torch.zeros_like(param, dtype=torch.float32)

    print(f"  Scoring {len(fisher_scores)} parameter tensors")

    # Accumulate squared gradients over calibration set
    n_processed = 0
    for idx, enc in enumerate(calibration_data):
        if idx % 50 == 0:
            print(f"  [{idx+1}/{len(calibration_data)}] Processing calibration sample...")

        model.zero_grad()

        with torch.no_grad():
            # Forward pass to get the output distribution
            outputs = model(**enc)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # Sample from the model's own distribution at the last token position
        # IMPORTANT: we sample, not argmax -- this gives the proper Fisher estimator
        # Using ground truth labels here would give the *empirical* Fisher, which
        # is a common but technically different quantity
        last_logits = logits[0, -1, :]  # (vocab_size,)
        probs = torch.softmax(last_logits.float(), dim=-1)
        sampled_token = torch.multinomial(probs, num_samples=1)  # (1,)

        # Now do a real forward+backward with the sampled target
        # We use NLL loss against the sampled token as a proxy for log p(y|x,theta)
        outputs = model(**enc)
        logits = outputs.logits[0, -1, :].float()  # (vocab_size,)
        loss = torch.nn.functional.cross_entropy(
            logits.unsqueeze(0), sampled_token
        )
        loss.backward()

        # Accumulate squared gradients (diagonal Fisher)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in fisher_scores:
                    continue
                if param.grad is not None:
                    fisher_scores[name] += param.grad.float().pow(2)

        n_processed += 1

    # Normalize by number of samples
    print(f"\n  Normalizing by {n_processed} samples...")
    for name in fisher_scores:
        fisher_scores[name] /= n_processed

    # Quick summary stats
    all_scores = torch.cat([s.flatten() for s in fisher_scores.values()])
    print(f"  Fisher score stats:")
    print(f"    min:  {all_scores.min().item():.6e}")
    print(f"    max:  {all_scores.max().item():.6e}")
    print(f"    mean: {all_scores.mean().item():.6e}")
    print(f"    nonzero fraction: {(all_scores > 0).float().mean().item():.4f}")

    return fisher_scores





def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
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

    # Load calibration data from the target task dataset
    # This is the *only* task-specific input -- the Fisher scores will
    # reflect which parameters matter for THIS dataset's distribution
    calibration_data = load_calibration_data(
        args.dataset_name,
        tokenizer,
        n_samples=args.n_calibration_samples,
        max_length=args.max_length,
        device=device,
    )

    # Compute Fisher scores -- this is the core cold-start signal
    fisher_scores = compute_fisher_scores(
        model,
        calibration_data,
        device=device,
        mlp_only=args.mlp_only,
    )

    # Pass Fisher scores into the same mask builder used for warm-start masks.
    # Fisher diagonal values ARE the importance scores -- no transformation needed.
    # High Fisher == parameter is sensitive to this task distribution == likely to update.
    masks = create_mask_from_scores_gpu_efficient(
        fisher_scores,
        args.sparsity_percent,
        device=device,
    )

    # Save with metadata -- note "method: fisher_cold_start" distinguishes
    # these from warm-start masks in downstream analysis
    model_sanitized = sanitize_model_name(args.model_name)
    output_file = args.output_file or (
        f"masks/cold_fisher_{model_sanitized}_{args.dataset_name.replace('/', '_')}"
        f"_sparsity{args.sparsity_percent}pct"
        f"_n{args.n_calibration_samples}.pt"
    )

    # Compute Jaccard if reference mask is provided
    jaccard_results = None
    if args.reference_mask:
        print(f"\nLoading reference mask from: {args.reference_mask}")
        ref = torch.load(args.reference_mask, map_location="cpu")
        reference_masks = ref["masks"]
        jaccard_results = compute_jaccard_similarity(masks, reference_masks)

    metadata = {
        "method": "fisher_cold_start",
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "sparsity_percent": args.sparsity_percent,
        "n_calibration_samples": args.n_calibration_samples,
        "mlp_only": args.mlp_only,
        "device": device,
    }

    if jaccard_results:
        metadata["jaccard_similarity"] = {
            "aggregate": jaccard_results["aggregate_jaccard"],
            "mean": jaccard_results["mean_jaccard"],
            "min": jaccard_results["min_jaccard"],
            "max": jaccard_results["max_jaccard"],
        }

    save_masks(masks, output_file, metadata)
    
    if jaccard_results:
        jaccard_file = output_file.replace(".pt", "_jaccard.json")
        with open(jaccard_file, "w") as f:
            json.dump(jaccard_results, f, indent=2)
        print(f"Detailed Jaccard results saved to: {jaccard_file}")
        
    print("\n✓ Cold-start mask generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fisher Information cold-start mask generation")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--dataset_name", type=str, default="qihoo360/Light-R1-DPOData")
    parser.add_argument("--sparsity_percent", type=float, default=90.0)
    parser.add_argument("--n_calibration_samples", type=int, default=512,
                        help="Number of calibration examples. 256-512 is usually sufficient.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--mlp_only", action="store_true", default=True,
                        help="Only score MLP parameters (recommended, consistent with sparse backprop target)")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--reference_mask", type=str, default=None,
                        help="Optional reference mask to compute Jaccard similarity against.")
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()
    main(args)

# Example usage:
# python fisher_cold_start_mask.py --model_name google/gemma-3-270m-it --dataset_name qihoo360/Light-R1-DPOData --sparsity_percent 90.0 --n_calibration_samples 512
# python fisher_cold_start_mask.py --model_name meta-llama/Llama-3.1-8B --dataset_name qihoo360/Light-R1-DPOData --sparsity_percent 95.0