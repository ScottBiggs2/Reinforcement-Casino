"""
Helper script to reconstruct full model checkpoints from deltas.

This script takes a base model from HuggingFace and applies saved deltas
to reconstruct full checkpoints at specific training steps.

Usage:
    python reconstruct_checkpoint.py \
        --model_name google/gemma-3-270m-it \
        --delta_log_dir ./delta_logs \
        --step 100 \
        --output_dir ./reconstructed_checkpoints/step_100
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional


def load_base_state(delta_log_dir: str) -> Optional[Dict[str, torch.Tensor]]:
    """Load the base state from delta_log_dir if it exists."""
    base_state_path = os.path.join(delta_log_dir, "base_state.pt")
    if os.path.exists(base_state_path):
        print(f"Loading base state from: {base_state_path}")
        return torch.load(base_state_path, map_location="cpu")
    return None


def load_deltas(delta_log_dir: str, step: int) -> Dict[str, torch.Tensor]:
    """Load deltas for a specific step."""
    delta_file = os.path.join(delta_log_dir, f"deltas_step_{step}.pt")
    if not os.path.exists(delta_file):
        raise FileNotFoundError(
            f"Delta file not found: {delta_file}\n"
            f"Available delta files in {delta_log_dir}: "
            f"{[f for f in os.listdir(delta_log_dir) if f.startswith('deltas_step_')]}"
        )
    print(f"Loading deltas from: {delta_file}")
    return torch.load(delta_file, map_location="cpu")


def validate_parameter_names(
    model_params: Dict[str, torch.Tensor],
    delta_params: Dict[str, torch.Tensor],
    base_state: Optional[Dict[str, torch.Tensor]] = None
):
    """Validate that parameter names match between model and deltas."""
    model_keys = set(model_params.keys())
    delta_keys = set(delta_params.keys())
    
    if model_keys != delta_keys:
        missing_in_model = delta_keys - model_keys
        missing_in_deltas = model_keys - delta_keys
        if missing_in_model:
            print(f"Warning: Parameters in deltas but not in model: {missing_in_model}")
        if missing_in_deltas:
            print(f"Warning: Parameters in model but not in deltas: {missing_in_deltas}")
    
    if base_state is not None:
        base_keys = set(base_state.keys())
        if base_keys != model_keys:
            missing_in_base = model_keys - base_keys
            missing_in_model = base_keys - model_keys
            if missing_in_base:
                print(f"Warning: Parameters in model but not in base_state: {missing_in_base}")
            if missing_in_model:
                print(f"Warning: Parameters in base_state but not in model: {missing_in_model}")


def reconstruct_checkpoint(
    model_name: str,
    delta_log_dir: str,
    step: int,
    output_dir: str,
    use_base_state_file: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
):
    """
    Reconstruct a full checkpoint by applying deltas to the base model.
    
    Args:
        model_name: HuggingFace model name to load as base
        delta_log_dir: Directory containing base_state.pt and deltas_step_*.pt files
        step: Training step to reconstruct
        output_dir: Directory to save the reconstructed checkpoint
        use_base_state_file: If True, use base_state.pt; if False, load fresh from HuggingFace
        torch_dtype: Data type for the reconstructed model
        device_map: Device mapping strategy for loading the model
    """
    print(f"\n{'='*60}")
    print(f"Reconstructing checkpoint for step {step}")
    print(f"{'='*60}")
    print(f"Base model: {model_name}")
    print(f"Delta log dir: {delta_log_dir}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load base state (if using saved base_state.pt)
    base_state = None
    if use_base_state_file:
        base_state = load_base_state(delta_log_dir)
        if base_state is None:
            print("Warning: base_state.pt not found. Will use model loaded from HuggingFace directly.")
    
    # Load deltas
    deltas = load_deltas(delta_log_dir, step)
    
    # Load base model from HuggingFace (needed for structure and config)
    print(f"Loading base model from HuggingFace: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    
    # Get model parameters
    model_state_dict = dict(model.named_parameters())
    
    # Validate parameter names
    print("Validating parameter names...")
    validate_parameter_names(model_state_dict, deltas, base_state)
    
    # Apply deltas to model parameters
    print(f"Applying deltas to reconstruct checkpoint at step {step}...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in deltas:
                print(f"Warning: No delta found for parameter {name}, keeping original value")
                continue
            
            delta = deltas[name]
            
            # If we have base_state, use it as the starting point (more accurate)
            # Otherwise, use the current parameter value from the loaded model
            if base_state is not None and name in base_state:
                # Deltas were computed as: current - base_state
                # So: current = base_state + delta
                base_param = base_state[name].to(param.device)
                # Work in float32 for precision, then convert to target dtype
                reconstructed = (base_param.float() + delta.float()).to(param.dtype)
                param.data = reconstructed
            else:
                # Fallback: add delta to current parameter (less accurate but works)
                delta = delta.to(param.device)
                if delta.dtype != param.dtype:
                    # Deltas are saved as float32, but model might be bfloat16
                    # We'll add in float32 then convert
                    param_float = param.float()
                    param_float = param_float + delta.float()
                    param.data = param_float.to(param.dtype)
                else:
                    param.data = param.data + delta
    
    # Save the reconstructed checkpoint
    print(f"Saving reconstructed checkpoint to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save using HuggingFace's save_pretrained for full compatibility
    model.save_pretrained(output_dir)
    
    # Also save the tokenizer for completeness
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n{'='*60}")
    print(f"Successfully reconstructed checkpoint at step {step}")
    print(f"Saved to: {output_dir}")
    print(f"{'='*60}\n")


def list_available_steps(delta_log_dir: str):
    """List all available delta steps in the delta_log_dir."""
    if not os.path.exists(delta_log_dir):
        print(f"Delta log directory does not exist: {delta_log_dir}")
        return []
    
    delta_files = [
        f for f in os.listdir(delta_log_dir)
        if f.startswith("deltas_step_") and f.endswith(".pt")
    ]
    
    steps = []
    for f in delta_files:
        try:
            step = int(f.replace("deltas_step_", "").replace(".pt", ""))
            steps.append(step)
        except ValueError:
            continue
    
    return sorted(steps)


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct full model checkpoints from deltas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Reconstruct checkpoint at step 100
    python reconstruct_checkpoint.py \\
        --model_name google/gemma-3-270m-it \\
        --delta_log_dir ./delta_logs \\
        --step 100 \\
        --output_dir ./reconstructed_checkpoints/step_100
    
    # List available steps
    python reconstruct_checkpoint.py \\
        --delta_log_dir ./delta_logs \\
        --list_steps
    
    # Reconstruct without using base_state.pt (load fresh from HuggingFace)
    python reconstruct_checkpoint.py \\
        --model_name google/gemma-3-270m-it \\
        --delta_log_dir ./delta_logs \\
        --step 100 \\
        --output_dir ./reconstructed_checkpoints/step_100 \\
        --no_use_base_state_file
        """
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-270m-it",
        help="HuggingFace model name to load as base (default: google/gemma-3-270m-it)"
    )
    
    parser.add_argument(
        "--delta_log_dir",
        type=str,
        required=True,
        help="Directory containing base_state.pt and deltas_step_*.pt files"
    )
    
    parser.add_argument(
        "--step",
        type=int,
        help="Training step to reconstruct (required unless --list_steps is used)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the reconstructed checkpoint (required unless --list_steps is used)"
    )
    
    parser.add_argument(
        "--list_steps",
        action="store_true",
        help="List all available delta steps and exit"
    )
    
    parser.add_argument(
        "--no_use_base_state_file",
        action="store_true",
        help="Don't use base_state.pt file, load fresh model from HuggingFace instead"
    )
    
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for the reconstructed model (default: bfloat16)"
    )
    
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device mapping strategy (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Handle list_steps mode
    if args.list_steps:
        steps = list_available_steps(args.delta_log_dir)
        if steps:
            print(f"\nAvailable delta steps in {args.delta_log_dir}:")
            for step in steps:
                print(f"  Step {step}")
            print(f"\nTotal: {len(steps)} checkpoints available")
        else:
            print(f"No delta files found in {args.delta_log_dir}")
        return
    
    # Validate required arguments
    if args.step is None:
        parser.error("--step is required unless --list_steps is used")
    if args.output_dir is None:
        parser.error("--output_dir is required unless --list_steps is used")
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]
    
    # Reconstruct checkpoint
    reconstruct_checkpoint(
        model_name=args.model_name,
        delta_log_dir=args.delta_log_dir,
        step=args.step,
        output_dir=args.output_dir,
        use_base_state_file=not args.no_use_base_state_file,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
    )


if __name__ == "__main__":
    main()
