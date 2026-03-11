"""
Unified evaluation runner for all LLM benchmarks.
Supports running individual benchmarks or all benchmarks at once.
"""

import multiprocessing
try:
    # This MUST be the first thing that happens to fix the vLLM CUDA conflict
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
print(f"Multiprocessing start method: {multiprocessing.get_start_method()}")

import os
import sys
import argparse
import inspect
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Evaluators are imported lazily inside BENCHMARKS to prevent dependency issues 
# in one benchmark from crashing the entire suite.


def get_evaluator(name):
    """Lazy-load evaluators to avoid dependency crashes at startup."""
    if name == "mmlu":
        from evaluation.mmlu_evaluator import evaluate_mmlu
        return evaluate_mmlu
    elif name == "math":
        from evaluation.math_evaluator import evaluate_math
        return evaluate_math
    elif name == "gsm8k":
        from evaluation.gsm8k_evaluator import evaluate_gsm8k
        return evaluate_gsm8k
    elif name == "coding":
        from evaluation.coding_evaluator import evaluate_coding
        return evaluate_coding
    elif name == "ifeval":
        from evaluation.ifeval_evaluator import evaluate_ifeval
        return evaluate_ifeval
    elif name == "squad":
        from evaluation.squad_evaluator import evaluate_squad
        return evaluate_squad
    elif name in ["gpqa", "gpqa_diamond"]:
        from evaluation.gpqa_evaluator import evaluate_gpqa_diamond
        return evaluate_gpqa_diamond
    return None


BENCHMARKS = {
    "mmlu": "mmlu",
    "math": "math",
    "gsm8k": "gsm8k",
    "coding": "coding",
    "ifeval": "ifeval",
    "squad": "squad",
    "gpqa": "gpqa_diamond",
    "gpqa_diamond": "gpqa_diamond",
}


def run_benchmark(
    benchmark_name: str,
    model_path: str,
    output_dir: Optional[str] = None,
    verbose: bool = True,
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
    # Print versions for debugging
    try:
        import torch
        import vllm
        import transformers
        print(f"Versions: torch={torch.__version__}, vllm={vllm.__version__}, transformers={transformers.__version__}")
    except ImportError:
        pass

    evaluator = get_evaluator(benchmark_name)
    if evaluator is None:
        raise ValueError(f"Could not find or load evaluator for: {benchmark_name}")

    try:
        # Drop kwargs that are not accepted by the evaluator
        evaluator_signature = inspect.signature(evaluator)
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in evaluator_signature.parameters.values()
    )
    if accepts_var_kwargs:
        filtered_kwargs = kwargs
        ignored_kwargs = {}
    else:
        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in evaluator_signature.parameters
        }
        # Explicitly pass model type if accepted
        if "model" in evaluator_signature.parameters and "model" in kwargs:
            filtered_kwargs["model"] = kwargs["model"]
            
        ignored_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in evaluator_signature.parameters
        }

    if ignored_kwargs and verbose:
        print(
            f"Ignoring unsupported arguments for {benchmark_name}: "
            f"{list(ignored_kwargs.keys())}"
        )
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running {benchmark_name.upper()} benchmark")
        print(f"{'='*80}\n")
    
    # Pass verbose flag to evaluator
    if "verbose" not in filtered_kwargs:
        filtered_kwargs["verbose"] = verbose
    
    if verbose:
        print(f"DEBUG: Running {benchmark_name} with filtered_kwargs:")
        for k, v in filtered_kwargs.items():
            print(f"  - {k}: {v} (type: {type(v)})")
    
    try:
        results = evaluator(model_path=model_path, **filtered_kwargs)
        
        # Save results if output directory is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{benchmark_name}_results.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            if verbose:
                print(f"\nResults saved to: {output_file}")
        
        return results
    except Exception as e:
        print(f"\nError running {benchmark_name}: {e}")
        raise


def run_all_benchmarks(
    model_path: str,
    benchmarks: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    verbose: bool = False,
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
        benchmarks = ["mmlu", "math", "gsm8k", "coding", "ifeval", "squad", "gpqa_diamond"]

    # Ensure output directory exists before running any benchmarks or writing summary
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if not verbose:
        print(f"Evaluating on {len(benchmarks)} benchmarks: {', '.join(benchmarks)}")
        print(f"Model: {model_path}\n")
    
    all_results = {}
    
    for benchmark in benchmarks:
        try:
            results = run_benchmark(
                benchmark_name=benchmark,
                model_path=model_path,
                output_dir=output_dir,
                verbose=verbose,
                **kwargs
            )
            all_results[benchmark] = results
        except Exception as e:
            if verbose:
                print(f"\nFailed to run {benchmark}: {e}")
            else:
                print(f"{benchmark}: ERROR - {e}")
            all_results[benchmark] = {"error": str(e)}
    
    # Save summary if output directory is specified
    if output_dir:
        summary_file = os.path.join(output_dir, "all_benchmarks_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        if verbose:
            print(f"\nSummary saved to: {summary_file}")
    
    return all_results


def print_summary(results: Dict[str, Dict[str, Any]]):
    """Print a concise summary of all benchmark results."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    for benchmark_name, benchmark_results in results.items():
        if "error" in benchmark_results:
            print(f"{benchmark_name.upper()}: ERROR - {benchmark_results['error']}")
            continue
        
        # Extract key metrics based on benchmark
        if benchmark_name == "mmlu":
            if "results" in benchmark_results:
                mmlu_results = benchmark_results["results"]
                for key, value in mmlu_results.items():
                    if "mmlu" in key.lower():
                        if isinstance(value, dict):
                            acc = value.get("acc", value.get("acc,none"))
                            if acc is not None:
                                print(f"{benchmark_name.upper()}: Accuracy = {acc:.4f}")
                                break
                            else:
                                print(f"{benchmark_name.upper()}: Accuracy = N/A")
                                break
        elif benchmark_name == "math":
            if "results" in benchmark_results:
                math_results = benchmark_results["results"]
                acc = None
                for key, value in math_results.items():
                    if "math" in key.lower():
                        if isinstance(value, dict):
                            # Try multiple accuracy key formats (use explicit None check, not 'or' which treats 0 as falsy)
                            for acc_key in ["acc,none", "acc", "accuracy", "exact_match,none", "exact_match"]:
                                if acc_key in value:
                                    acc = value[acc_key]
                                    if isinstance(acc, (int, float)):
                                        print(f"{benchmark_name.upper()}: Accuracy = {acc:.4f}")
                                        break
                            if acc is not None:
                                break
                if acc is None:
                    print(f"{benchmark_name.upper()}: Could not extract accuracy (check verbose output)")
        elif benchmark_name == "gsm8k":
            if "results" in benchmark_results:
                gsm8k_results = benchmark_results["results"]
                # Look for gsm8k or any key containing gsm8k
                for key, data in gsm8k_results.items():
                    if "gsm8k" in key.lower() and isinstance(data, dict):
                        acc = data.get("exact_match,none", data.get("exact_match", data.get("acc,none", data.get("acc", 0))))
                        print(f"{benchmark_name.upper()}: Accuracy = {acc:.4f}")
                        break
        elif benchmark_name == "coding":
            if "results" in benchmark_results:
                for task, task_results in benchmark_results["results"].items():
                    pass_at_1 = task_results.get("pass@1", task_results.get("acc", 0))
                    print(f"CODING ({task}): pass@1 = {pass_at_1:.4f}")
        elif benchmark_name == "ifeval":
            if "results" in benchmark_results:
                ifeval_results = benchmark_results["results"]
                for key, data in ifeval_results.items():
                    if "ifeval" in key.lower() and isinstance(data, dict):
                        strict = data.get("prompt_level_strict_acc,none", data.get("prompt_level_strict_acc", 0))
                        loose = data.get("prompt_level_loose_acc,none", data.get("prompt_level_loose_acc", 0))
                        print(f"{benchmark_name.upper()}: Strict Acc = {strict:.4f}, Loose Acc = {loose:.4f}")
                        break
        elif benchmark_name == "squad":
            if "results" in benchmark_results:
                squad_results = benchmark_results["results"]
                for key, data in squad_results.items():
                    if "squad" in key.lower() and isinstance(data, dict):
                        exact_match = data.get("exact_match,none", data.get("exact_match", 0))
                        f1 = data.get("f1,none", data.get("f1", 0))
                        print(f"{benchmark_name.upper()}: Exact Match = {exact_match:.4f}, F1 = {f1:.4f}")
                        break
            elif "exact_match" in benchmark_results:
                em = benchmark_results['exact_match']
                f1 = benchmark_results['f1']
                print(f"{benchmark_name.upper()}: Exact Match = {em:.4f}, F1 = {f1:.4f}")
        elif benchmark_name in ["gpqa", "gpqa_diamond"]:
            if "results" in benchmark_results:
                gpqa_results = benchmark_results["results"]
                for key, data in gpqa_results.items():
                    if "gpqa" in key.lower() or "diamond" in key.lower():
                        if isinstance(data, dict):
                            acc = data.get("acc,none", data.get("acc", 0))
                            print(f"{benchmark_name.upper()}: Accuracy = {acc:.4f}")
                            break


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
        "--batch_size", type=str, default="auto",
        help="Batch size for evaluation (default: auto). Can be an integer or 'auto'."
    )
    parser.add_argument(
        "--use_vllm", action="store_true",
        help="Use vLLM as the backend for much faster evaluation (requires vllm package)"
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
        "--apply_chat_template", dest="apply_chat_template", action="store_true",
        help="Force applying the model's chat template when supported"
    )
    parser.add_argument(
        "--no-apply_chat_template", dest="apply_chat_template", action="store_false",
        help="Disable applying chat template (defaults to auto-detect for instruct/chat models)"
    )
    parser.set_defaults(apply_chat_template=None)
    parser.add_argument(
        "--squad_method", type=str, default="lm_eval",
        choices=["lm_eval", "hf_evaluate"],
        help="Method for SQuAD evaluation"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging (default: quiet mode for run_all)"
    )
    
    args = parser.parse_args()
    
    # Handle "all" benchmarks
    if "all" in args.benchmarks:
        benchmarks_to_run = ["mmlu", "math", "gsm8k", "coding", "ifeval", "squad", "gpqa_diamond"]
    else:
        benchmarks_to_run = args.benchmarks
    
    # Prepare kwargs for evaluators
    model_type = "vllm" if args.use_vllm else "hf"
    
    # Convert batch_size to int if possible
    try:
        batch_size = int(args.batch_size)
    except ValueError:
        batch_size = args.batch_size # Keep as "auto"
        
    eval_kwargs = {
        "model": model_type,
        "limit": args.limit,
        "batch_size": batch_size,
        "device": args.device,
        "trust_remote_code": args.trust_remote_code,
        "apply_chat_template": args.apply_chat_template,
    }
    
    # Add benchmark-specific kwargs
    if args.num_fewshot is not None:
        eval_kwargs["num_fewshot"] = args.num_fewshot
    
    if "squad" in benchmarks_to_run:
        eval_kwargs["method"] = args.squad_method
    
    # Run benchmarks (quiet by default unless --verbose is set)
    results = run_all_benchmarks(
        model_path=args.model_path,
        benchmarks=benchmarks_to_run,
        output_dir=args.output_dir,
        verbose=args.verbose,
        **eval_kwargs
    )
    
    # Print summary
    print_summary(results)
