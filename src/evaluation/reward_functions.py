import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.reward_functions import calculate_rouge


def load_models(args):
    """
    Loads the three models for evaluation: default, fine-tuned, and masked.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading default model...")
    default_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    print("Loading fine-tuned model from checkpoint...")
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        dpo_model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    else:
        print(f"Warning: Checkpoint path {args.checkpoint_path} not found. Skipping DPO model.")
        dpo_model = None

    print("Loading masked model...")
    masked_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if args.mask_path and os.path.exists(args.mask_path):
        print(f"Applying mask from {args.mask_path}...")
        masks = torch.load(args.mask_path, map_location='cpu')
        
        # Load the deltas from the DPO checkpoint
        if dpo_model is not None:
            print("Calculating weight deltas from DPO model...")
            with torch.no_grad():
                for name, param in masked_model.named_parameters():
                    # Get corresponding param from DPO model
                    dpo_param = dict(dpo_model.named_parameters())[name]
                    
                    # Calculate delta
                    delta = dpo_param.data - param.data
                    
                    # Apply mask to delta
                    if name in masks:
                        mask = masks[name].to(param.device)
                        masked_delta = delta * mask
                        param.data += masked_delta
                        print(f"Applied masked delta to {name}: {mask.sum().item()}/{mask.numel()} params")
                    else:
                        print(f"Warning: No mask found for {name}")
        else:
            print("Warning: Cannot apply masked deltas without DPO model. Using base model.")
            masked_model = None
    else:
        print(f"Warning: Mask path {args.mask_path} not found. Skipping masked model.")
        masked_model = None
        
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
    # Load dataset
    print(f"Loading dataset and sampling {args.num_samples} examples...")
    dataset = load_dataset("qihoo360/Light-R1-DPOData", split="test")
    eval_dataset = dataset.shuffle(seed=42).select(range(args.num_samples))

    # Load models
    tokenizer, default_model, dpo_model, masked_model = load_models(args)

    prompts = eval_dataset["prompt"]
    references = eval_dataset["chosen"]

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