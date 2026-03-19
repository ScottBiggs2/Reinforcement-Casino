import sys
import os
import argparse
import inspect
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import multiprocessing

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Evaluators are imported lazily inside BENCHMARKS to prevent dependency issues 
# in one benchmark from crashing the entire suite.





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
    Run a single benchmark evaluation inside an isolated subprocess.
    """
    import subprocess
    import json
    import tempfile
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running {benchmark_name.upper()} benchmark (Subprocess Isolated)")
        print(f"{'='*80}\n")
        
    # Set up temp file to intercept the results so we don't have to parse stdout
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
        tmp_results_path = tmp_file.name

    # Import target dynamically for the wrapper execution script
    import_module = f"evaluation.{benchmark_name}_evaluator"
    
    # Explicit mapping of benchmark names to their evaluator entry points
    BENCHMARK_FUNC_MAP = {
        "mmlu": "evaluate_mmlu",
        "math": "evaluate_math",
        "gsm8k": "evaluate_gsm8k",
        "coding": "evaluate_coding",
        "ifeval": "evaluate_ifeval",
        "squad": "evaluate_squad",
        "gpqa": "evaluate_gpqa_diamond",
        "gpqa_diamond": "evaluate_gpqa_diamond",
    }
    
    if benchmark_name in BENCHMARK_FUNC_MAP:
        func_name = BENCHMARK_FUNC_MAP[benchmark_name]
        if benchmark_name in ["gpqa", "gpqa_diamond"]:
            import_module = "evaluation.gpqa_evaluator"
    else:
        # Fallback to automated naming
        func_name = f"evaluate_{benchmark_name}"
        
    # Map 'method' if provided as an extra kwarg
    method = kwargs.pop("method", "lm_eval")
    
    if benchmark_name == "coding":
        kwargs["confirm_run_unsafe_code"] = True
        
    # Build subprocess injection
    script_code = f"""
import sys
import json
import os
import traceback
import inspect

sys.path.append({repr(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))})

try:
    from {import_module} import {func_name}
    
    # Introspect exactly what arguments this evaluator accepts so we don't throw an unexpected keyword arg
    raw_kwargs = {repr(kwargs)}
    evaluator_signature = inspect.signature({func_name})
    
    # Check if the evaluator accepts **kwargs
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in evaluator_signature.parameters.values()
    )
    
    # Defensive filtering: only pass arguments that the function actually accepts or if it takes **kwargs
    filtered_kwargs = {{}}
    for key, value in raw_kwargs.items():
        if accepts_var_kwargs or key in evaluator_signature.parameters:
            filtered_kwargs[key] = value
            
    # Always ensure 'method' is handled if accepted
    if "method" in evaluator_signature.parameters and "method" not in filtered_kwargs:
        filtered_kwargs["method"] = {repr(method)}
        
    if {repr(verbose)}:
        print(f"DEBUG: Running {benchmark_name} with filtered_kwargs:")
        for k, v in filtered_kwargs.items():
            print(f"  - {{k}}: {{v}} (type: {{type(v)}})")
    
    results = {func_name}(model_path={repr(model_path)}, **filtered_kwargs)
    
    # Write to transport file
    with open({repr(tmp_results_path)}, 'w') as f:
        json.dump(results, f, default=str)
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
"""
    
    try:
        # Run process and capture output for debugging
        result = subprocess.run(
            [sys.executable, "-c", script_code],
            capture_output=True,
            text=True,
            check=True
        )
        
        if verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
                
        # Load results from the transport file
        if not os.path.exists(tmp_results_path):
             results = {"error": f"Subprocess finished but results file {tmp_results_path} was not created."}
             if result.stderr:
                 results["error"] += f"\nStderr: {result.stderr}"
        else:
            with open(tmp_results_path, 'r') as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError:
                    results = {"error": "Subprocess script failed to dump valid JSON."}
                
        # Save formal results if output directory is specified
        if output_dir and "error" not in results:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{benchmark_name}_results.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            if verbose:
                print(f"\nResults saved to: {output_file}")
                
    except subprocess.CalledProcessError as e:
        print(f"\nError running {benchmark_name} subprocess: Returns non-zero exit code {e.returncode}")
        # Build more informative error with actual traceback if possible
        error_msg = f"Subprocess crashed: {str(e)}"
        if e.stderr:
            error_msg += f"\nStderr: {e.stderr}"
        if e.stdout:
            error_msg += f"\nStdout: {e.stdout}"
        results = {"error": error_msg}
    except Exception as e:
        print(f"\nError orchestrating {benchmark_name}: {e}")
        import traceback
        traceback.print_exc()
        results = {"error": str(e)}
    finally:
        # Cleanup
        if os.path.exists(tmp_results_path):
            os.remove(tmp_results_path)
            
    return results


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
                        acc = data.get("exact_match,strict-match", data.get("exact_match,flexible-extract", data.get("exact_match,none", data.get("exact_match", data.get("acc,none", data.get("acc", 0))))))
                        print(f"{benchmark_name.upper()}: Accuracy = {acc:.4f}")
                        break
        elif benchmark_name == "coding":
            if "results" in benchmark_results:
                coding_res = benchmark_results["results"]
                for task, task_results in coding_res.items():
                    if isinstance(task_results, dict):
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
                found_squad = False
                for key, data in squad_results.items():
                    if "squad" in key.lower() and isinstance(data, dict):
                        # EM and F1 are standard
                        exact_match = data.get("exact_match,none", data.get("exact_match", data.get("contains,none", 0)))
                        f1 = data.get("f1,none", data.get("f1", 0))
                        
                        # Defensive: if EM exists but F1 is 0, try to find ANY F1-like key
                        if exact_match > 0 and f1 == 0:
                            for k, v in data.items():
                                if "f1" in k.lower() and isinstance(v, (int, float)) and v > 0:
                                    f1 = v
                                    break
                                    
                        # Diagnostic: if both 0, print some keys to stdout
                        if exact_match == 0 and f1 == 0:
                             print(f"DEBUG: SQuAD Task '{key}' found but EM/F1 are 0. Available metrics: {list(data.keys())}")
                             
                        print(f"{benchmark_name.upper()}: Exact Match = {exact_match:.4f}, F1 = {f1:.4f}")
                        found_squad = True
                        break
                if not found_squad:
                    # Fallback for direct keys
                    em = benchmark_results.get("exact_match", 0)
                    f1 = benchmark_results.get("f1", 0)
                    if em > 0 or f1 > 0:
                         print(f"{benchmark_name.upper()}: Exact Match = {em:.4f}, F1 = {f1:.4f}")
            elif "exact_match" in benchmark_results:
                em = benchmark_results['exact_match']
                f1 = benchmark_results['f1']
                print(f"{benchmark_name.upper()}: Exact Match = {em:.4f}, F1 = {f1:.4f}")
        elif benchmark_name in ["gpqa", "gpqa_diamond"]:
            if "results" in benchmark_results:
                gpqa_results = benchmark_results["results"]
                found_gpqa = False
                for key, data in gpqa_results.items():
                    if ("gpqa" in key.lower() or "diamond" in key.lower()) and isinstance(data, dict):
                        acc = data.get("acc_norm,none", data.get("acc_norm", data.get("acc,none", data.get("acc", 0))))
                        
                        # Robustness: try to find ANYTHING accuracy-like if 0
                        if acc == 0:
                            for k, v in data.items():
                                if "acc" in k.lower() and isinstance(v, (int, float)) and v > acc:
                                    acc = v
                        
                        # Diagnostic: if 0, print some keys to stdout
                        if acc == 0:
                             print(f"DEBUG: GPQA Task '{key}' found but accuracy is 0. Available metrics: {list(data.keys())}")
                             
                        print(f"{benchmark_name.upper()}: Accuracy = {acc:.4f}")
                        found_gpqa = True
                        break
                
                if not found_gpqa and gpqa_results:
                     # If no 'gpqa' key found, check the first key if there's only one
                     if len(gpqa_results) == 1:
                         key = list(gpqa_results.keys())[0]
                         data = gpqa_results[key]
                         if isinstance(data, dict):
                             acc = data.get("acc_norm,none", data.get("acc_norm", data.get("acc,none", data.get("acc", 0))))
                             print(f"{benchmark_name.upper()}: Accuracy = {acc:.4f}")
                             found_gpqa = True
                
                if not found_gpqa:
                    print(f"{benchmark_name.upper()}: Accuracy = N/A")
                if not found_gpqa:
                     print(f"{benchmark_name.upper()}: Could not extract accuracy (check verbose output)")


if __name__ == "__main__":
    # This MUST be the first thing that happens in the main process
    # to fix the vLLM CUDA conflict when using isolated subprocesses.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    print(f"DEBUG: run_all_benchmarks.py starting")
    print(f"DEBUG: Python version: {sys.version}")
    print(f"DEBUG: Multiprocessing start method: {multiprocessing.get_start_method()}")
    
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
