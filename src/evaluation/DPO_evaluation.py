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


def load_models(args):
    """
    Loads the three models for evaluation: default, fine-tuned, and masked.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine dtype based on available hardware
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load default (base) model
    print("Loading default model...")
    default_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype)

    # Load DPO fine-tuned model
    print("Loading fine-tuned model from checkpoint...")
    dpo_model = None
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        dpo_model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, torch_dtype=dtype)
    else:
        print(f"Warning: Checkpoint path '{args.checkpoint_path}' not found. Skipping DPO model.")

    # Load masked model (base + sparse deltas)
    print("Loading masked model...")
    masked_model = None
    
    if not (args.mask_path and os.path.exists(args.mask_path)):
        print(f"Warning: Mask path '{args.mask_path}' not found. Skipping masked model.")
        return tokenizer, default_model, dpo_model, masked_model
    
    if dpo_model is None:
        print("Warning: Cannot create masked model without DPO model. Skipping masked model.")
        return tokenizer, default_model, dpo_model, masked_model
    
    # Load base model for masked version
    masked_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype)
    
    # Load and apply masks
    print(f"Loading masks from {args.mask_path}...")
    masks = torch.load(args.mask_path, map_location='cpu')
    print(f"Loaded {len(masks)} mask tensors")
    
    # Convert mask keys to match model parameter names
    # Mask finder saves "model.norm.weight" as "model_norm_weight"
    model_param_names = {name for name, _ in masked_model.named_parameters()}
    converted_masks = {}
    
    for param_name in model_param_names:
        mask_key = param_name.replace(".", "_")
        if mask_key in masks:
            converted_masks[param_name] = masks[mask_key]
    
    print(f"Matched {len(converted_masks)}/{len(model_param_names)} parameters to masks")
    
    if len(converted_masks) == 0:
        print("ERROR: No masks matched model parameters!")
        print(f"Sample model param: {list(model_param_names)[0]}")
        print(f"Sample mask key: {list(masks.keys())[0]}")
        masked_model = None
        return tokenizer, default_model, dpo_model, masked_model
    
    # Apply masked deltas to the base model
    print("Applying masked weight deltas...")
    applied_count = 0
    total_masked_params = 0
    total_params = 0
    
    with torch.no_grad():
        dpo_params = dict(dpo_model.named_parameters())
        
        for name, param in masked_model.named_parameters():
            if name not in converted_masks:
                continue
            
            # Calculate delta between DPO and base
            delta = dpo_params[name].data - param.data
            
            # Apply mask to delta and update weights
            mask = converted_masks[name].to(param.device)
            masked_delta = delta * mask
            param.data += masked_delta
            
            # Track statistics
            applied_count += 1
            masked_count = mask.sum().item()
            total_count = mask.numel()
            total_masked_params += masked_count
            total_params += total_count
            
            # Print first few for debugging
            if applied_count <= 5:
                sparsity = 100 * masked_count / total_count
                print(f"  {name}: {masked_count:,}/{total_count:,} params ({sparsity:.2f}%)")
    
    # Print summary statistics
    overall_sparsity = 100 * total_masked_params / total_params if total_params > 0 else 0
    print(f"\nApplied masks to {applied_count} layers")
    print(f"Overall sparsity: {total_masked_params:,}/{total_params:,} params ({overall_sparsity:.2f}%)")
        
    return tokenizer, default_model, dpo_model, masked_model


def generate_responses(tokenizer, model, prompts, max_new_tokens=100):
    """
    Generates responses from a model for a batch of prompts.
    """
    if model is None:
        return ["Model not loaded."] * len(prompts)
        
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
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
                        help="Path to the mask file.")
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