#!/usr/bin/env python3
"""
Standalone script to run MATH evaluation on a saved model.
Based on the example in src/evaluation/README.md
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.math_evaluator import evaluate_math
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run MATH evaluation on a model checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python src/evaluation/run_math_eval.py \\
    --model_path results/triton_sparse_dpo_meta_llama_llama_3_1_8b_instruct_fixed/final_model \\
    --num_fewshot 4 \\
    --batch_size 8
        """
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to model (HuggingFace ID or local path with .safetensors)"
    )
    parser.add_argument(
        "--num_fewshot", 
        type=int, 
        default=4,
        help="Number of few-shot examples (default: 4)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Batch size for evaluation (default: 8)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of examples (default: all)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to run on (auto-detect if not specified)"
    )
    parser.add_argument(
        "--trust_remote_code", 
        action="store_true",
        help="Trust remote code in model config"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MATH EVALUATION")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Few-shot: {args.num_fewshot}")
    print(f"Batch size: {args.batch_size}")
    if args.limit:
        print(f"Limit: {args.limit} examples")
    print("=" * 80)
    print()
    
    results = evaluate_math(
        model_path=args.model_path,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        verbose=args.verbose,
    )
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
