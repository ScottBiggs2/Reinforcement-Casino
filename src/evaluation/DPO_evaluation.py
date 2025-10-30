import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
from reward_functions import calculate_rouge

def load_models(args):
    """
    Loads the three models for evaluation: default, fine-tuned, and masked.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading default model...")
    default_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    print("Loading fine-tuned model from checkpoint...")
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        dpo_model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path)
    else:
        print(f"Warning: Checkpoint path {args.checkpoint_path} not found. Skipping DPO model.")
        dpo_model = None

    print("Loading masked model...")
    masked_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    if args.mask_path and os.path.exists(args.mask_path):
        masks = torch.load(args.mask_path)
        with torch.no_grad():
            for name, param in masked_model.named_parameters():
                if name in masks:
                    param.data *= masks[name].to(param.device) # Ensure mask is on the same device
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
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)
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
    print("\nGenerating responses from default model...")
    default_responses = generate_responses(tokenizer, default_model, prompts)
    
    print("Generating responses from DPO model...")
    dpo_responses = generate_responses(tokenizer, dpo_model, prompts)

    print("Generating responses from masked model...")
    masked_responses = generate_responses(tokenizer, masked_model, prompts)

    # Evaluate responses
    print("\n--- Evaluation Results (ROUGE scores) ---")
    
    if default_model:
        default_scores = calculate_rouge(default_responses, references)
        print("\nDefault Model:")
        for key, value in default_scores.items():
            print(f"  {key}: {value.mid.fmeasure:.4f}")

    if dpo_model:
        dpo_scores = calculate_rouge(dpo_responses, references)
        print("\nDPO Model:")
        for key, value in dpo_scores.items():
            print(f"  {key}: {value.mid.fmeasure:.4f}")

    if masked_model:
        masked_scores = calculate_rouge(masked_responses, references)
        print("\nMasked Model:")
        for key, value in masked_scores.items():
            print(f"  {key}: {value.mid.fmeasure:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation harness for DPO models.")
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-3-270m-it", help="Base model name or path.")
    parser.add_argument("--checkpoint_path", type=str, default="results/gemma_dpo_training/checkpoints/checkpoint-5", help="Path to the DPO fine-tuned checkpoint.")
    parser.add_argument("--mask_path", type=str, default="masks/top_10_percent_mask.pt", help="Path to the mask file.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate.")
    
    args = parser.parse_args()
    main(args)
