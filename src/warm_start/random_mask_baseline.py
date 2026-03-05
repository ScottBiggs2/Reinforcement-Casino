import torch
import os
import argparse
import json

from src.utils.mask_utils import (
    create_mask_from_scores_gpu_efficient,
    compute_jaccard_similarity,
    save_masks
)

# ============================================================
# Random Baseline Mask Generator
# ============================================================
# Generates a random sparse mask at a target sparsity level,
# matched to the parameter shapes of a reference mask file.
#
# Purpose: null hypothesis baseline for subnetwork analysis.
# If Fisher/magnitude/momentum masks don't beat random on
# Jaccard similarity or training quality, the subnetwork
# story doesn't hold. This is the floor everything else
# needs to clear.
#
# Expected Jaccard between two independent random masks of
# sparsity s: J = s^2 / (2s - s^2)
# e.g. at 10% density (90% sparsity): J ≈ 0.053
#      at 5%  density (95% sparsity): J ≈ 0.026
# Any method with Jaccard substantially above this is
# finding real structure.
# ============================================================


def expected_random_jaccard(sparsity_percent: float) -> float:
    """
    Expected Jaccard similarity between two independent random masks
    of the same sparsity, as a closed-form sanity check.
    density = 1 - sparsity/100
    J = density^2 / (2*density - density^2)
    """
    density = 1.0 - sparsity_percent / 100.0
    if density <= 0:
        return 0.0
    return density ** 2 / (2 * density - density ** 2)


def generate_random_mask(reference_masks: dict, sparsity_percent: float, seed: int = None) -> dict:
    """
    Generate a random binary mask with the same parameter shapes and
    target sparsity as a reference mask dict.

    Uses global threshold on uniform random scores -- same pipeline as
    create_mask_from_scores_gpu_efficient -- so sparsity targeting
    behavior is identical to Fisher/magnitude masks.

    Args:
        reference_masks: dict of {param_name: tensor} (shapes only are used)
        sparsity_percent: target sparsity
        seed: random seed for reproducibility
    """
    if seed is not None:
        torch.manual_seed(seed)
        print(f"  Random seed: {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Draw uniform random scores -- no task signal whatsoever
    scores = {name: torch.rand_like(mask.float()) for name, mask in reference_masks.items()}

    # Utilize mask_utils directly
    masks = create_mask_from_scores_gpu_efficient(scores, sparsity_percent, device=device)

    return masks





def main(args):
    # Load reference mask to get parameter shapes and names.
    # This ensures the random mask is directly comparable to whatever
    # warm-start or cold-start mask you're validating against.
    print(f"Loading reference mask from: {args.reference_mask}")
    ref = torch.load(args.reference_mask, map_location="cpu")
    reference_masks = ref["masks"]
    ref_metadata = ref.get("metadata", {})

    # Use reference sparsity if not explicitly overridden
    sparsity = args.sparsity_percent
    if sparsity is None:
        sparsity = ref_metadata.get("sparsity_percent", None)
        if sparsity is None:
            raise ValueError("Could not infer sparsity from reference mask metadata. Pass --sparsity_percent explicitly.")
        print(f"  Using sparsity from reference metadata: {sparsity}%")

    print(f"\n=== Generating Random Baseline Mask ===")
    print(f"  Sparsity target: {sparsity}%")
    print(f"  Expected Jaccard vs another random mask at this sparsity: "
          f"{expected_random_jaccard(sparsity):.4f}")

    random_masks = generate_random_mask(reference_masks, sparsity, seed=args.seed)

    # Optionally compute Jaccard against the reference mask.
    # This tells you how much structure the reference mask has relative to chance --
    # a well-trained warm-start mask should score *much* higher than expected_random_jaccard.
    j = None
    if args.compare_to_reference:
        print(f"\n=== Jaccard vs Reference Mask ===")
        j = compute_jaccard_similarity(random_masks, reference_masks)
        expected = expected_random_jaccard(sparsity)
        if j:
            print(f"  Expected by chance:                      {expected:.4f}")
            print(f"  Ratio (signal/noise):                    {j['aggregate_jaccard']/expected:.2f}x")

    output_file = args.output_file or f"masks/random_sparsity{sparsity}pct_seed{args.seed}.pt"
    metadata = {
        "method": "random_baseline",
        "sparsity_percent": sparsity,
        "seed": args.seed,
        "reference_mask": args.reference_mask,
        "expected_jaccard_vs_random": expected_random_jaccard(sparsity),
    }
    save_masks(random_masks, output_file, metadata)
    
    if args.compare_to_reference and j:
        jaccard_file = output_file.replace(".pt", "_jaccard.json")
        with open(jaccard_file, "w") as f:
            json.dump(j, f, indent=2)
        print(f"Detailed Jaccard results saved to: {jaccard_file}")
        
    print("\n✓ Random baseline mask generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random baseline mask generator")
    parser.add_argument("--reference_mask", type=str, required=True,
                        help="Path to a .pt mask file (warm-start or cold-start). "
                             "Used to match parameter shapes. Sparsity inferred from metadata if not set.")
    parser.add_argument("--sparsity_percent", type=float, default=None,
                        help="Override sparsity level (default: infer from reference mask metadata)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed. Run with multiple seeds to get variance estimate on Jaccard.")
    parser.add_argument("--compare_to_reference", action="store_true",
                        help="Compute Jaccard between random mask and the reference mask.")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()
    main(args)

# Example usage:
# Basic random mask matched to a warm-start mask:
#   python random_mask.py --reference_mask masks/sparsity_90.0pct_magnitude.pt
#
# With Jaccard comparison against reference (key validation):
#   python random_mask.py --reference_mask masks/sparsity_90.0pct_magnitude.pt --compare_to_reference
#
# Multiple seeds for variance estimate on Jaccard:
#   for seed in 42 43 44 45 46; do
#     python random_mask.py --reference_mask masks/sparsity_90.0pct_magnitude.pt --seed $seed --compare_to_reference
#   done