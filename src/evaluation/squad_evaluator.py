"""
SQuAD (Stanford Question Answering Dataset) evaluation harness.
"""

import os
import sys
import inspect
from typing import Dict, Any, Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering
from datasets import load_dataset
import evaluate

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.model_loader import load_model_and_tokenizer

try:
    from lm_eval import simple_evaluate
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False


def evaluate_squad_with_hf_evaluate(
    model_path: str,
    split: str = "validation",
    limit: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    max_length: int = 384,
    stride: int = 128,
    trust_remote_code: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model on SQuAD using HuggingFace's evaluate library.
    This works best with question-answering models.
    
    Args:
        model_path: Path to model (HuggingFace ID or local path)
        split: Dataset split to use ("validation" or "train")
        limit: Limit number of examples (None = all)
        device: Device to run on (auto-detect if None)
        dtype: Model dtype (auto-detect if None)
        max_length: Maximum sequence length
        stride: Stride for sliding window
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Dictionary with evaluation results
    """
    if verbose:
        print("=" * 60)
        print("SQuAD EVALUATION (HuggingFace evaluate)")
        print("=" * 60)
    else:
        print("SQuAD: Running...", end=" ", flush=True)
    
    # Try to load as QA model first, fallback to causal LM
    try:
        if verbose:
            print("Attempting to load as QuestionAnswering model...")
        if dtype is None:
            if torch.cuda.is_available():
                dtype = torch.float16
            else:
                dtype = torch.float32
        
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda" or model.device.type == 'cpu':
            model.to(device)
        model.eval()
        is_qa_model = True
    except Exception as e:
        if verbose:
            print(f"Could not load as QA model: {e}")
            print("Falling back to CausalLM model...")
        model, tokenizer = load_model_and_tokenizer(
            model_path=model_path,
            dtype=dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        is_qa_model = False
    
    if is_qa_model:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Load SQuAD dataset
    if verbose:
        print(f"\nLoading SQuAD {split} dataset...")
    dataset = load_dataset("squad", split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    if verbose:
        print(f"Loaded {len(dataset)} examples")
    
    # Load SQuAD metric
    squad_metric = evaluate.load("squad")
    
    # Prepare predictions and references
    if verbose:
        print("\nRunning inference...")
    predictions = []
    references = []
    
    model.eval()
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if verbose and (i + 1) % 100 == 0:
                print(f"Processing {i+1}/{len(dataset)}...", end='\r')
            
            question = example["question"]
            context = example["context"]
            answers = example["answers"]
            
            # Format input
            if is_qa_model:
                inputs = tokenizer(
                    question,
                    context,
                    max_length=max_length,
                    truncation=True,
                    stride=stride,
                    return_tensors="pt",
                    padding=True,
                ).to(device)
                
                outputs = model(**inputs)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                
                # Get answer span
                start_idx = start_logits.argmax().item()
                end_idx = end_logits.argmax().item()
                
                # Decode answer
                input_ids = inputs["input_ids"][0]
                answer_tokens = input_ids[start_idx:end_idx+1]
                prediction_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            else:
                # For causal LM, use prompt-based approach
                prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract answer (text after "Answer:")
                if "Answer:" in full_text:
                    prediction_text = full_text.split("Answer:")[-1].strip()
                else:
                    prediction_text = full_text[len(prompt):].strip()
            
            predictions.append({
                "id": example["id"],
                "prediction_text": prediction_text,
            })
            references.append({
                "id": example["id"],
                "answers": answers,
            })
    
    if verbose:
        print(f"\nCompleted inference on {len(predictions)} examples")
        print("\nComputing metrics...")
    results = squad_metric.compute(predictions=predictions, references=references)
    
    if verbose:
        print("\n" + "=" * 60)
        print("SQuAD RESULTS")
        print("=" * 60)
        print(f"\nExact Match: {results.get('exact_match', 0):.4f}")
        print(f"F1 Score: {results.get('f1', 0):.4f}")
    else:
        em = results.get('exact_match', 0)
        f1 = results.get('f1', 0)
        print(f"Exact Match: {em:.4f}, F1: {f1:.4f}")
    
    return results


def evaluate_squad_with_lm_eval(
    model_path: str,
    num_fewshot: int = 0,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    batch_size: int = 8,
    trust_remote_code: bool = False,
    apply_chat_template: Optional[bool] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model on SQuAD using lm-evaluation-harness.
    
    Args:
        model_path: Path to model (HuggingFace ID or local path)
        num_fewshot: Number of few-shot examples
        limit: Limit number of examples (None = all)
        device: Device to run on (auto-detect if None)
        dtype: Model dtype (auto-detect if None)
        batch_size: Batch size for evaluation
        trust_remote_code: Whether to trust remote code
        apply_chat_template: Whether to apply the model's chat template (auto for instruct/chat if None)
        
    Returns:
        Dictionary with evaluation results
    """
    if not LM_EVAL_AVAILABLE:
        raise ImportError(
            "lm-evaluation-harness is required. Install with: pip install lm-eval"
        )
    
    if verbose:
        print("=" * 60)
        print("SQuAD EVALUATION (lm-evaluation-harness)")
        print("=" * 60)
    
    # Auto-detect dtype if not specified
    if dtype is None:
        if torch.cuda.is_available():
            dtype_str = "float16"
        elif torch.backends.mps.is_available():
            dtype_str = "float16"
        else:
            dtype_str = "float32"
    else:
        dtype_str = str(dtype).replace("torch.", "")
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Auto-apply chat templates for instruct/chat models if not explicitly set
    if apply_chat_template is None:
        lower_path = model_path.lower()
        apply_chat_template = any(keyword in lower_path for keyword in ["instruct", "chat", "-it", "-int"])

    # Convert to absolute path if it's a local path (for lm-eval compatibility)
    if os.path.exists(model_path):
        model_path = os.path.abspath(model_path)
    
    # Build model_args string for lm-eval
    # lm-eval's from_pretrained will automatically detect and use .safetensors files
    base_model_args_parts = [f"pretrained={model_path}", f"dtype={dtype_str}"]
    if trust_remote_code:
        base_model_args_parts.append("trust_remote_code=True")
    base_model_args_str = ",".join(base_model_args_parts)
    
    # Try different configurations for chat template
    configs_to_try = []
    if apply_chat_template:
        configs_to_try.append({"apply_chat_template": True, "fewshot_as_multiturn": True})
        configs_to_try.append({"apply_chat_template": True})
    configs_to_try.append({})  # Base config without chat template
    
    # Run evaluation
    # Note: simple_evaluate will load the model internally
    if verbose:
        print(f"\nRunning SQuAD evaluation...")
        print(f"Model: {model_path}")
        print(f"Device: {device}, Dtype: {dtype_str}")
        if limit:
            print(f"Limiting to {limit} examples")
    else:
        print("SQuAD: Running...", end=" ", flush=True)
    
    task_candidates = ["squad", "squad_v2", "squad_completion"]
    results = None
    task_errors = []
    
    import logging
    lm_eval_logger = logging.getLogger("lm_eval")
    old_level = lm_eval_logger.level
    if not verbose:
        lm_eval_logger.setLevel(logging.WARNING)

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*chat template.*")
        
        for task_name in task_candidates:
            for config in configs_to_try:
                try:
                    eval_kwargs = {
                        "model": "hf",
                        "model_args": base_model_args_str,
                        "tasks": task_name,
                        "num_fewshot": num_fewshot,
                        "limit": limit,
                        "batch_size": batch_size,
                        "device": device,
                    }
                    eval_kwargs.update(config)
                    results = simple_evaluate(**eval_kwargs)
                    break
                except TypeError as e:
                    if "apply_chat_template" in str(e) or "fewshot_as_multiturn" in str(e):
                        if verbose:
                            print(f"Chat template config not supported: {config}; retrying without it.")
                        continue
                    raise
                except Exception as e:
                    if isinstance(e, KeyError) or task_name in str(e) or "not found" in str(e).lower():
                        task_errors.append(str(e))
                        if verbose:
                            print(f"Task '{task_name}' not found, trying next alias if available...")
                        break
                    raise
            if results is not None:
                break
    
    if not verbose:
        lm_eval_logger.setLevel(old_level)
    
    if results is None:
        raise RuntimeError(
            f"Failed to run SQuAD; tried aliases {task_candidates}. Errors: {task_errors}"
        )
    
    # Extract and print key metrics
    if "results" in results:
        squad_results = results["results"]
        if "squad" in squad_results:
            squad_data = squad_results["squad"]
            exact_match = squad_data.get("exact_match,none", 0)
            f1 = squad_data.get("f1,none", 0)
            if verbose:
                print("\n" + "=" * 60)
                print("SQuAD RESULTS")
                print("=" * 60)
                print(f"\nExact Match: {exact_match:.4f}")
                print(f"F1 Score: {f1:.4f}")
            else:
                print(f"Exact Match: {exact_match:.4f}, F1: {f1:.4f}")
    
    return results


def evaluate_squad(
    model_path: str,
    method: str = "lm_eval",
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate a model on SQuAD benchmark.
    
    Args:
        model_path: Path to model (HuggingFace ID or local path)
        method: Evaluation method ("lm_eval" or "hf_evaluate")
        **kwargs: Additional arguments passed to the specific evaluator
        
    Returns:
        Dictionary with evaluation results
    """
    if method == "lm_eval":
        target = evaluate_squad_with_lm_eval
    elif method == "hf_evaluate":
        target = evaluate_squad_with_hf_evaluate
    else:
        raise ValueError(f"Unknown method: {method}. Use 'lm_eval' or 'hf_evaluate'")

    # Only forward arguments the target evaluator knows how to handle
    signature = inspect.signature(target)
    accepted_params = set(signature.parameters.keys()) - {"model_path"}
    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in accepted_params
    }
    # Always pass verbose if the target accepts it
    if "verbose" in signature.parameters:
        filtered_kwargs["verbose"] = verbose
    
    ignored_kwargs = {
        key: value for key, value in kwargs.items() if key not in accepted_params and key != "verbose"
    }

    if ignored_kwargs and verbose:
        print(
            f"Ignoring unsupported arguments for SQuAD ({method}): "
            f"{list(ignored_kwargs.keys())}"
        )

    return target(model_path=model_path, **filtered_kwargs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model on SQuAD benchmark")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (HuggingFace ID or local path)")
    parser.add_argument("--method", type=str, default="lm_eval",
                        choices=["lm_eval", "hf_evaluate"],
                        help="Evaluation method")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (auto-detect if not specified)")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Trust remote code in model config")
    
    args = parser.parse_args()
    
    results = evaluate_squad(
        model_path=args.model_path,
        method=args.method,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
