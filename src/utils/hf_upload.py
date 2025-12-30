"""
Upload trained models to HuggingFace Hub
Uploads model files, config, and tokenizer files for Llama 3.1 8B Instruct models
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")  # HuggingFace username

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")
if not HF_USERNAME:
    raise ValueError("HF_USERNAME not set")

# Model configurations
MODELS = [
    {
        "local_path": "baseline_temp/checkpoint-100",
        "repo_name": "LLaMA-3.1-8B-Instruct-DPO-Baseline",
        "description": "LLaMA 3.1 8B fine tuned on Light R1 DPO dataset for 100 steps",
    },
    {
        "local_path": "results/triton_sparse_dpo_meta_llama_llama_3_1_8b_instruct_fixed/final_model",
        "repo_name": "LLaMA-3.1-8B-Instruct-DPO-Triton-Sparse",
        "description": "LLaMA 3.1 8B fine tuned on Light R1 DPO dataset for 100 steps with custom Triton-accelerated BSR-AdamW optimizer",
    },
]


def upload_model(api: HfApi, model_config: dict, username: str):
    """Upload a single model to HuggingFace Hub"""
    
    # Resolve path relative to script directory or current working directory
    local_path = Path(model_config["local_path"])
    if not local_path.is_absolute():
        # Try relative to current working directory first
        if not local_path.exists():
            # Try relative to script directory
            script_dir = Path(__file__).parent.parent.parent  # Go up to project root
            local_path = script_dir / local_path
    local_path = local_path.resolve()
    
    # Verify the path exists
    if not local_path.exists():
        print(f"❌ Error: Path does not exist: {local_path}")
        return False
    
    if not local_path.is_dir():
        print(f"❌ Error: Path is not a directory: {local_path}")
        return False
    
    repo_id = f"{username}/{model_config['repo_name']}"
    
    print(f"\n{'='*60}")
    print(f"Uploading: {repo_id}")
    print(f"From: {local_path}")
    print(f"{'='*60}\n")
    
    # Check if required files exist
    safetensors_path = local_path / "model.safetensors"
    config_path = local_path / "config.json"
    tokenizer_config_path = local_path / "tokenizer_config.json"
    tokenizer_json_path = local_path / "tokenizer.json"
    generation_config_path = local_path / "generation_config.json"
    
    # Check for model files (safetensors or pytorch_model.bin)
    # Also check for sharded models (model-00001-of-00002.safetensors, etc.)
    sharded_files = sorted(local_path.glob("model-*-of-*.safetensors"))
    model_index_path = local_path / "model.safetensors.index.json"
    is_sharded = len(sharded_files) > 0
    
    if not safetensors_path.exists() and not is_sharded:
        pytorch_model_path = local_path / "pytorch_model.bin"
        if not pytorch_model_path.exists():
            print(f"❌ Error: model.safetensors, sharded model files, or pytorch_model.bin not found in {local_path}")
            return False
        safetensors_path = pytorch_model_path
    
    if is_sharded:
        print(f"✓ Found {len(sharded_files)} sharded model files")
        if not model_index_path.exists():
            print(f"⚠️  Warning: model.safetensors.index.json not found (recommended for sharded models)")
    
    if not config_path.exists():
        print(f"⚠️  Warning: config.json not found in {local_path}")
    
    if not tokenizer_config_path.exists():
        print(f"⚠️  Warning: tokenizer_config.json not found in {local_path}")
    
    try:
        # Create repository (will skip if exists)
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            token=HF_TOKEN,
        )
        print(f"✓ Repository created/verified: {repo_id}")
        
        # Upload model file(s)
        if is_sharded:
            # Upload all sharded model files
            for shard_file in sharded_files:
                shard_filename = shard_file.name
                api.upload_file(
                    path_or_fileobj=str(shard_file),
                    path_in_repo=shard_filename,
                    repo_id=repo_id,
                    repo_type="model",
                    token=HF_TOKEN,
                )
                print(f"✓ Uploaded {shard_filename}")
            
            # Upload model index file if it exists
            if model_index_path.exists():
                api.upload_file(
                    path_or_fileobj=str(model_index_path),
                    path_in_repo="model.safetensors.index.json",
                    repo_id=repo_id,
                    repo_type="model",
                    token=HF_TOKEN,
                )
                print(f"✓ Uploaded model.safetensors.index.json")
        else:
            # Upload single model file
            # Check file extension to determine filename
            if safetensors_path.suffix == ".safetensors":
                model_filename = "model.safetensors"
            else:
                model_filename = "pytorch_model.bin"
            api.upload_file(
                path_or_fileobj=str(safetensors_path),
                path_in_repo=model_filename,
                repo_id=repo_id,
                repo_type="model",
                token=HF_TOKEN,
            )
            print(f"✓ Uploaded {model_filename}")
        
        # Upload config.json if it exists
        if config_path.exists():
            api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo="config.json",
                repo_id=repo_id,
                repo_type="model",
                token=HF_TOKEN,
            )
            print(f"✓ Uploaded config.json")
        
        # Upload generation_config.json if it exists
        if generation_config_path.exists():
            api.upload_file(
                path_or_fileobj=str(generation_config_path),
                path_in_repo="generation_config.json",
                repo_id=repo_id,
                repo_type="model",
                token=HF_TOKEN,
            )
            print(f"✓ Uploaded generation_config.json")
        
        # Upload tokenizer files
        tokenizer_files = [
            ("tokenizer_config.json", tokenizer_config_path),
            ("tokenizer.json", tokenizer_json_path),
        ]
        
        # Check for other common tokenizer files
        for filename in ["special_tokens_map.json", "vocab.json", "merges.txt"]:
            file_path = local_path / filename
            if file_path.exists():
                tokenizer_files.append((filename, file_path))
        
        # Check for chat template file (common in Llama models)
        chat_template_path = local_path / "chat_template.jinja"
        if chat_template_path.exists():
            tokenizer_files.append(("chat_template.jinja", chat_template_path))
        
        for filename, file_path in tokenizer_files:
            if file_path.exists():
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=filename,
                    repo_id=repo_id,
                    repo_type="model",
                    token=HF_TOKEN,
                )
                print(f"✓ Uploaded {filename}")
        
        # Create a basic README based on Llama 3.1 8B Instruct documentation
        readme_content = f"""---
license: llama3.1
base_model: meta-llama/Llama-3.1-8B-Instruct
tags:
- llama
- llama-3.1
- instruct
- dpo
- direct-preference-optimization
- text-generation
- conversational
---

# {model_config['repo_name']}

{model_config['description']}

## Model Details

- **Base Model**: [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- **Architecture**: Llama 3.1 8B Instruct
- **Training**: Direct Preference Optimization (DPO)
- **Task**: Text generation, instruction following, conversational AI

## Requirements

- `transformers >= 4.43.0` (required for full Llama 3.1 support)
- `torch` (recommended: `torch >= 2.0.0`)

## Usage

### Installation

```bash
pip install --upgrade transformers torch
```

### Basic Usage with Transformers

Starting with `transformers >= 4.43.0`, you can run conversational inference using the Transformers pipeline abstraction or by leveraging the Auto classes with the `generate()` function.

#### Using Pipeline

```python
import transformers
import torch

model_id = "{repo_id}"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={{"torch_dtype": torch.bfloat16}},
    device_map="auto",
)

messages = [
    {{"role": "system", "content": "You are a helpful assistant."}},
    {{"role": "user", "content": "Explain what machine learning is."}},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
```

#### Using AutoModelForCausalLM

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {{"role": "system", "content": "You are a helpful assistant."}},
    {{"role": "user", "content": "Explain what machine learning is."}},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
```

### Tool Use

Llama 3.1 supports tool use through chat templates in Transformers. See the [official documentation](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) for detailed examples.

## Model Information

This model is based on Meta's Llama 3.1 8B Instruct model, fine-tuned using Direct Preference Optimization (DPO). The model maintains compatibility with the original Llama 3.1 architecture and chat template format.

For more information about the base model, see:
- [Meta Llama 3.1 8B Instruct on Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Hugging Face Llama Recipes](https://github.com/huggingface/huggingface-llama-recipes)

## Citation

If you use this model, please cite the original Llama 3.1 paper:

```bibtex
@article{{{{meta2024llama,
  title={{{{Llama 3.1}}}},
  author={{{{Meta AI}}}},
  year={{{{2024}}}}
}}}}
```

"""
        
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN,
        )
        print(f"✓ Created README.md")
        
        print(f"\n✅ Successfully uploaded to: https://huggingface.co/{repo_id}\n")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading model: {str(e)}")
        return False


def main():
    """Main upload function"""
    
    # Initialize HF API
    api = HfApi()
    
    print(f"\n{'='*60}")
    print(f"HuggingFace Model Upload Script")
    print(f"User: {HF_USERNAME}")
    print(f"{'='*60}")
    
    # Upload each model
    results = []
    for model_config in MODELS:
        success = upload_model(api, model_config, HF_USERNAME)
        results.append((model_config["repo_name"], success))
    
    # Summary
    print(f"\n{'='*60}")
    print("Upload Summary")
    print(f"{'='*60}")
    for repo_name, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{status}: {repo_name}")
    print()


if __name__ == "__main__":
    main()