#!/usr/bin/env python3
"""
Debug script to inspect the actual structure of MATH evaluation results.
"""

import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.math_evaluator import evaluate_math

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug MATH evaluation results structure")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (HuggingFace ID or local path)")
    parser.add_argument("--num_fewshot", type=int, default=4,
                        help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=10,
                        help="Limit number of examples (small for debugging)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DEBUGGING MATH EVALUATION RESULTS STRUCTURE")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Limit: {args.limit} examples (for quick debugging)")
    print("=" * 80)
    print()
    
    results = evaluate_math(
        model_path=args.model_path,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        verbose=True,
    )
    
    print("\n" + "=" * 80)
    print("FULL RESULTS STRUCTURE")
    print("=" * 80)
    print(json.dumps(results, indent=2, default=str))
    
    print("\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)
    
    if isinstance(results, dict):
        print(f"Top-level keys: {list(results.keys())}")
        
        if "results" in results:
            print(f"\nresults['results'] keys: {list(results['results'].keys())}")
            
            for key, value in results['results'].items():
                print(f"\n  Key: {key}")
                print(f"    Type: {type(value).__name__}")
                if isinstance(value, dict):
                    print(f"    Sub-keys: {list(value.keys())}")
                    for sub_key, sub_value in value.items():
                        print(f"      {sub_key}: {sub_value} (type: {type(sub_value).__name__})")
