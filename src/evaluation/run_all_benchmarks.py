"""
Unified evaluation runner for all LLM benchmarks.
Supports running individual benchmarks or all benchmarks at once.
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.mmlu_evaluator import evaluate_mmlu
from evaluation.math_evaluator import evaluate_math
from evaluation.squad_evaluator import evaluate_squad
from evaluation.gpqa_evaluator import evaluate_gpqa_diamond


BENCHMARKS = {
    "mmlu": evaluate_mmlu,
    "math": evaluate_math,
    "squad": evaluate_squad,
    "gpqa": evaluate_gpqa_diamond,
    "gpqa_diamond": evaluate_gpqa_diamond,
}


def run_benchmark(
    benchmark_name: str,
    model_path: str,
    output_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a single benchmark evaluation.
    
    Args:
        benchmark_name: Name of the benchmark to run
        model_path: Path to model (HuggingFace ID or local path)
        output_dir: Directory to save results (optional)
        **kwargs: Additional arguments for the benchmark evaluator
        
    Returns:
        Dictionary with evaluation results
    """
    if benchmark_name not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. "
            f"Available: {list(BENCHMARKS.keys())}"
        )
    
    evaluator = BENCHMARKS[benchmark_name]
    
    print(f"\n{'='*80}")
    print(f"Running {benchmark_name.upper()} benchmark")
    print(f"{'='*80}\n")
    
    try:
        results = evaluator(model_path=model_path, **kwargs)
        
        # Save results if output directory is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{benchmark_name}_results.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {output_file}")
        
        return results
    except Exception as e:
        print(f"\nError running {benchmark_name}: {e}")
        raise


def run_all_benchmarks(
    model_path: str,
    benchmarks: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Run all or specified benchmarks.
    
    Args:
        model_path: Path to model (HuggingFace ID or local path)
        benchmarks: List of benchmark names to run (None = all)
        output_dir: Directory to save results (optional)
        **kwargs: Additional arguments passed to all benchmark evaluators
        
    Returns:
        Dictionary mapping benchmark names to their results
    """
    if benchmarks is None:
        benchmarks = ["mmlu", "math", "squad", "gpqa_diamond"]
    
    all_results = {}
    
    for benchmark in benchmarks:
        try:
            results = run_benchmark(
                benchmark_name=benchmark,
                model_path=model_path,
                output_dir=output_dir,
                **kwargs
            )
            all_results[benchmark] = results
        except Exception as e:
            print(f"\nFailed to run {benchmark}: {e}")
            all_results[benchmark] = {"error": str(e)}
    
    # Save summary if output directory is specified
    if output_dir:
        summary_file = os.path.join(output_dir, "all_benchmarks_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nSummary saved to: {summary_file}")
    
    return all_results


def print_summary(results: Dict[str, Dict[str, Any]]):
    """Print a summary of all benchmark results."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    for benchmark_name, benchmark_results in results.items():
        print(f"\n{benchmark_name.upper()}:")
        if "error" in benchmark_results:
            print(f"  Error: {benchmark_results['error']}")
            continue
        
        # Extract key metrics based on benchmark
        if benchmark_name == "mmlu":
            if "results" in benchmark_results:
                mmlu_results = benchmark_results["results"]
                for key, value in mmlu_results.items():
                    if "mmlu" in key.lower():
                        if isinstance(value, dict) and "acc" in value:
                            print(f"  Accuracy: {value['acc']:.4f}")
        elif benchmark_name == "math":
            if "results" in benchmark_results:
                math_results = benchmark_results["results"]
                for key, value in math_results.items():
                    if "math" in key.lower():
                        if isinstance(value, dict) and "acc" in value:
                            print(f"  Accuracy: {value['acc']:.4f}")
        elif benchmark_name == "squad":
            if "exact_match" in benchmark_results:
                print(f"  Exact Match: {benchmark_results['exact_match']:.4f}")
                print(f"  F1 Score: {benchmark_results['f1']:.4f}")
            elif "results" in benchmark_results:
                squad_results = benchmark_results["results"]
                if "squad" in squad_results:
                    squad_data = squad_results["squad"]
                    exact_match = squad_data.get("exact_match,none", 0)
                    f1 = squad_data.get("f1,none", 0)
                    print(f"  Exact Match: {exact_match:.4f}")
                    print(f"  F1 Score: {f1:.4f}")
        elif benchmark_name in ["gpqa", "gpqa_diamond"]:
            if "results" in benchmark_results:
                gpqa_results = benchmark_results["results"]
                for key, value in gpqa_results.items():
                    if "gpqa" in key.lower() or "diamond" in key.lower():
                        if isinstance(value, dict):
                            acc = value.get("acc", value.get("acc,none", 0))
                            if acc:
                                print(f"  Accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM benchmark evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python run_all_benchmarks.py --model_path google/gemma-2-2b
  
  # Run specific benchmarks
  python run_all_benchmarks.py --model_path google/gemma-2-2b --benchmarks mmlu math
  
  # Run with custom settings
  python run_all_benchmarks.py --model_path ./models/my_model --batch_size 4 --limit 100
  
  # Save results to directory
  python run_all_benchmarks.py --model_path google/gemma-2-2b --output_dir ./results
        """
    )
    
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to model (HuggingFace ID or local path with .safetensors)"
    )
    parser.add_argument(
        "--benchmarks", type=str, nargs="+",
        choices=list(BENCHMARKS.keys()) + ["all"],
        default=["all"],
        help="Benchmarks to run (default: all)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save results (optional)"
    )
    parser.add_argument(
        "--num_fewshot", type=int, default=None,
        help="Number of few-shot examples (benchmark-specific defaults if not specified)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of examples per benchmark"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run on (auto-detect if not specified)"
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true",
        help="Trust remote code in model config"
    )
    parser.add_argument(
        "--squad_method", type=str, default="lm_eval",
        choices=["lm_eval", "hf_evaluate"],
        help="Method for SQuAD evaluation"
    )
    
    args = parser.parse_args()
    
    # Handle "all" benchmarks
    if "all" in args.benchmarks:
        benchmarks_to_run = ["mmlu", "math", "squad", "gpqa_diamond"]
    else:
        benchmarks_to_run = args.benchmarks
    
    # Prepare kwargs for evaluators
    eval_kwargs = {
        "limit": args.limit,
        "batch_size": args.batch_size,
        "device": args.device,
        "trust_remote_code": args.trust_remote_code,
    }
    
    # Add benchmark-specific kwargs
    if args.num_fewshot is not None:
        eval_kwargs["num_fewshot"] = args.num_fewshot
    
    if "squad" in benchmarks_to_run:
        eval_kwargs["method"] = args.squad_method
    
    # Run benchmarks
    results = run_all_benchmarks(
        model_path=args.model_path,
        benchmarks=benchmarks_to_run,
        output_dir=args.output_dir,
        **eval_kwargs
    )
    
    # Print summary
    print_summary(results)
