import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import sys
import evaluate

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def calculate_rouge(predictions, references):
    """
    Calculates ROUGE scores for a batch of predictions and references.
    Args:
        predictions (list of str): The generated responses from the model.
        references (list of str): The ground truth (chosen) responses.
    Returns:
        dict: A dictionary of ROUGE scores (e.g., rouge1, rouge2, rougeL).
    """
    rouge_metric = evaluate.load('rouge')
    scores = rouge_metric.compute(predictions=predictions, references=references)
    return scores


from src.utils.mask_manager import SparseMaskManager

def load_models(args):
    """
    Loads the models for evaluation.
    Optimized for 'single unit' checkpoints where weights are stored naturally.
    """
    # Determine device and dtype
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    # Load tokenizer
    print(f"Loading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load default (base) model
    print(f"Loading base model from {args.model_name_or_path}...")
    default_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype, trust_remote_code=True)

    # 2. Load DPO fine-tuned model (Dense / Single Unit)
    dpo_model = None
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading DPO/Sparse model from {args.checkpoint_path}...")
        # Natural loading: HF automatically handles .safetensors and full weights.
        dpo_model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, torch_dtype=dtype, trust_remote_code=True)
        print("✓ Model loaded successfully as single unit.")
    else:
        print(f"Note: No checkpoint path provided or valid. Skipping DPO model.")

    # 3. Load Masked Model (Legacy / Diagnostic)
    masked_model = None
    if args.mask_path and os.path.exists(args.mask_path) and dpo_model is not None:
        print(f"Diagnostic: Applying mask dynamically to DPO model to verify sparsity logic...")
        masked_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype, trust_remote_code=True)
        mask_manager = SparseMaskManager(args.mask_path, device='cpu')
        
        with torch.no_grad():
            dpo_params = dict(dpo_model.named_parameters())
            for name, param in masked_model.named_parameters():
                mask = mask_manager.get_mask(name)
                if mask is not None:
                    # Apply mask to delta (DPO - Base)
                    delta = dpo_params[name].data - param.data
                    param.data += (delta * mask.to(param.device)).to(param.dtype)
        print("✓ Dynamic masked model constructed for comparison.")
        
    return tokenizer, default_model, dpo_model, masked_model


def generate_responses(tokenizer, model, prompts, max_new_tokens=100):
    """
    Generates responses from a model for a batch of prompts.
    """
    if model is None:
        return ["Model not loaded."] * len(prompts)
        
    model.eval()
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model.to(device)

    responses = []
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            print(f"Generating response {i+1}/{len(prompts)}...", end='\r')
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False  # Deterministic for fair comparison
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
    print()  # New line after progress
    return responses


def main(args):
    # Load dataset (train split, no test split available)
    print(f"Loading dataset and sampling {args.num_samples} examples...")
    dataset = load_dataset("qihoo360/Light-R1-DPOData", split="train")
    
    # Check what columns exist
    print(f"Dataset columns: {dataset.column_names}")
    print(f"Dataset format: {dataset.features}")
    
    eval_dataset = dataset.shuffle(seed=42).select(range(args.num_samples))

    # Load models
    tokenizer, default_model, dpo_model, masked_model = load_models(args)

    # Extract prompts and references from the DPO dataset structure
    # Light-R1-DPOData has 'chosen' and 'rejected' dicts with 'from' and 'value' keys
    prompts = []
    references = []
    
    for example in eval_dataset:
        # The 'conversations' field contains the user query
        if 'conversations' in example and len(example['conversations']) > 0:
            # Get the human message from conversations
            user_msg = next((msg['value'] for msg in example['conversations'] if msg['from'] == 'human'), None)
            if user_msg:
                prompts.append(user_msg)
        
        # The 'chosen' field contains the preferred response
        if 'chosen' in example and isinstance(example['chosen'], dict):
            if 'value' in example['chosen']:
                references.append(example['chosen']['value'])
            elif 'content' in example['chosen']:
                references.append(example['chosen']['content'])
    
    if len(prompts) == 0:
        raise ValueError(f"Could not extract prompts from dataset. Please check dataset structure.")
    
    print(f"Extracted {len(prompts)} prompts and {len(references)} references")

    # Generate responses
    print("\n=== Generating responses from default model ===")
    default_responses = generate_responses(tokenizer, default_model, prompts, args.max_new_tokens)
    
    print("\n=== Generating responses from DPO model ===")
    dpo_responses = generate_responses(tokenizer, dpo_model, prompts, args.max_new_tokens)

    print("\n=== Generating responses from masked model ===")
    masked_responses = generate_responses(tokenizer, masked_model, prompts, args.max_new_tokens)

    # Evaluate responses
    print("\n" + "="*50)
    print("EVALUATION RESULTS (ROUGE scores)")
    print("="*50)
    
    if default_model:
        print("\n📊 Default Model:")
        default_scores = calculate_rouge(default_responses, references)
        for key, value in default_scores.items():
            if key in ['rouge1', 'rouge2', 'rougeL']:
                print(f"  {key}: {value:.4f}")

    if dpo_model:
        print("\n📊 DPO Fine-tuned Model:")
        dpo_scores = calculate_rouge(dpo_responses, references)
        for key, value in dpo_scores.items():
            if key in ['rouge1', 'rouge2', 'rougeL']:
                print(f"  {key}: {value:.4f}")

    if masked_model:
        print("\n📊 Masked Model (Base + Top-k% Deltas):")
        masked_scores = calculate_rouge(masked_responses, references)
        for key, value in masked_scores.items():
            if key in ['rouge1', 'rouge2', 'rougeL']:
                print(f"  {key}: {value:.4f}")

    # Print example outputs
    if args.show_examples:
        print("\n" + "="*50)
        print("EXAMPLE OUTPUTS")
        print("="*50)
        for i in range(min(3, len(prompts))):
            print(f"\n--- Example {i+1} ---")
            print(f"Prompt: {prompts[i][:100]}...")
            print(f"\nReference: {references[i][:150]}...")
            print(f"\nDefault: {default_responses[i][:150]}...")
            if dpo_model:
                print(f"\nDPO: {dpo_responses[i][:150]}...")
            if masked_model:
                print(f"\nMasked: {masked_responses[i][:150]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation harness for DPO models.")
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-3-270m-it", 
                        help="Base model name or path.")
    parser.add_argument("--checkpoint_path", type=str, default="results/gemma_dpo_training/final_model", 
                        help="Path to the DPO fine-tuned checkpoint.")
    parser.add_argument("--mask_path", type=str, default="masks/top_10.0_percent_mask.pt", 
                        help="Path to the mask file (for legacy delta reconstruction).")
    parser.add_argument("--masked_model_path", type=str, default=None, 
                        help="Path to a pre-saved masked model (single unit).")
    parser.add_argument("--num_samples", type=int, default=10, 
                        help="Number of samples to evaluate.")
    parser.add_argument("--max_new_tokens", type=int, default=100, 
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--show_examples", action="store_true", 
                        help="Show example outputs from each model.")
    
    args = parser.parse_args()
    main(args)

    # python src/evaluation/DPO_evaluation.py
    # git push <remote-name> <local-branch-name>:<remote-branch-name>