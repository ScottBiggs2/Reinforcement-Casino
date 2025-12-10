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
    
    print(f"Loading model from: {model_path}")
    print(f"Device: {device}, Dtype: {dtype}")
    
    # Check if local path exists
    if os.path.exists(model_path):
        print(f"Loading from local path: {model_path}")
        # Check if it's a directory with model files
        if os.path.isdir(model_path):
            # Check for safetensors or pytorch_model.bin
            has_safetensors = any(
                f.endswith('.safetensors') 
                for f in os.listdir(model_path) 
                if os.path.isfile(os.path.join(model_path, f))
            )
            if has_safetensors:
                print("Found .safetensors files in directory")
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    
    # Move to device if not using device_map
    if device != "cuda" or model.device.type == 'cpu':
        model.to(device)
    
    model.eval()
    print(f"✓ Model loaded on {model.device} with dtype: {model.dtype}")
    
    return model, tokenizer
