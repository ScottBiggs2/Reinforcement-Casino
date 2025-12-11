"""
Model loading utility for evaluation harnesses.
Supports loading models from HuggingFace or local paths (including .safetensors files).
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Tuple


def load_model_and_tokenizer(
    model_path: str,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer from HuggingFace or local path.
    
    Args:
        model_path: Path to model (HuggingFace model ID or local directory)
        dtype: Data type for model weights (default: auto-detect based on hardware)
        device: Device to load model on (default: auto-detect)
        trust_remote_code: Whether to trust remote code in model config
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Auto-detect dtype if not specified
    if dtype is None:
        if torch.cuda.is_available():
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Convert to absolute path if it's a local path
    if os.path.exists(model_path):
        model_path = os.path.abspath(model_path)
    
    print(f"Loading model from: {model_path}")
    print(f"Device: {device}, Dtype: {dtype}")
    
    # Check if local path exists
    if os.path.exists(model_path):
        print(f"Loading from local path: {model_path}")
        # Check if it's a directory with model files
        if os.path.isdir(model_path):
            # Check for safetensors or pytorch_model.bin
            model_files = [f for f in os.listdir(model_path) 
                          if os.path.isfile(os.path.join(model_path, f))]
            has_safetensors = any(f.endswith('.safetensors') for f in model_files)
            has_pytorch = any(f.startswith('pytorch_model') for f in model_files)
            
            if has_safetensors:
                safetensor_files = [f for f in model_files if f.endswith('.safetensors')]
                print(f"Found {len(safetensor_files)} .safetensors file(s) in directory")
            elif has_pytorch:
                print("Found pytorch_model.bin in directory (using PyTorch format)")
            else:
                # Check if it might be a checkpoint directory structure
                if 'config.json' in model_files:
                    print("Found config.json, assuming model checkpoint directory")
                else:
                    print("Warning: No model weight files detected, but directory exists")
        else:
            raise ValueError(f"Model path {model_path} exists but is not a directory")
    else:
        print(f"Loading from HuggingFace: {model_path}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
    
    # Load model
    print("Loading model...")
    # from_pretrained automatically handles .safetensors files if present
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        # Prefer safetensors if available (default behavior)
    )
    
    # Move to device if not using device_map
    if device != "cuda" or model.device.type == 'cpu':
        model.to(device)
    
    model.eval()
    print(f"✓ Model loaded on {model.device} with dtype: {model.dtype}")
    
    return model, tokenizer
