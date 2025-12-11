# LLM Evaluation Harnesses

This directory contains evaluation harnesses for common LLM benchmarks. All harnesses support loading models from HuggingFace or local paths (including `.safetensors` files).

## Supported Benchmarks

1. **MMLU** (Massive Multitask Language Understanding) - Multi-subject knowledge test
2. **MATH** - Mathematical problem-solving benchmark
3. **SQuAD** (Stanford Question Answering Dataset) - Reading comprehension
4. **GPQA Diamond** - Graduate-level science questions

## Installation

The evaluation harnesses require `lm-evaluation-harness` for most benchmarks.

### Quick Install (Recommended)

Use the provided installation script which handles Python 3.11 compatibility issues:

```bash
bash install_lm_eval.sh
```

### Manual Install

If the script doesn't work, try these steps:

```bash
# 1. Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# 2. Install rouge-score separately (often causes build issues)
pip install rouge-score

# 3. Install lm-eval
pip install lm-eval
```

### Install from Source (If PyPI install fails)

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

### Troubleshooting Python 3.11 Issues

If you encounter `AttributeError: module '_distutils_hack' has no attribute 'add_shim'`:
- This is usually a harmless warning from setuptools
- Try: `pip install --upgrade setuptools`
- If rouge-score fails to build, install it separately first: `pip install rouge-score --no-build-isolation`

For SQuAD, the HuggingFace `evaluate` library is also used (already in requirements.txt).

## Usage

### Running All Benchmarks

Use the unified runner to evaluate on all benchmarks:

```bash
python src/evaluation/run_all_benchmarks.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --output_dir ./evaluation_results

# Force applying chat templates for instruct/chat models
python src/evaluation/run_all_benchmarks.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --apply_chat_template
```

### Running Individual Benchmarks

#### MMLU

```bash
python src/evaluation/mmlu_evaluator.py \
    --model_path google/gemma-2-2b \
    --num_fewshot 5 \
    --batch_size 8
```

#### MATH

```bash
python src/evaluation/math_evaluator.py \
    --model_path google/gemma-2-2b \
    --num_fewshot 4 \
    --batch_size 8

python src/evaluation/math_evaluator.py \
    --model_path results/triton_sparse_dpo_meta_llama_llama_3_1_8b_instruct_fixed/final_model \
    --num_fewshot 4 \
    --batch_size 8

python src/evaluation/math_evaluator.py \
    --model_path results/baseline_dpo_timing/final_model \
    --num_fewshot 4 \
    --batch_size 8
```

#### SQuAD

```bash
# Using lm-evaluation-harness (default)
python src/evaluation/squad_evaluator.py \
    --model_path google/gemma-2-2b \
    --method lm_eval \
    --batch_size 8

# Using HuggingFace evaluate library
python src/evaluation/squad_evaluator.py \
    --model_path google/gemma-2-2b \
    --method hf_evaluate \
    --limit 100
```

#### GPQA Diamond

```bash
python src/evaluation/gpqa_evaluator.py \
    --model_path google/gemma-2-2b \
    --num_fewshot 0 \
    --batch_size 8
```

### Loading Models from Local Paths

All evaluators support loading models from local directories containing `.safetensors` files:

```bash
python src/evaluation/run_all_benchmarks.py \
    --model_path ./models/my_model \
    --output_dir ./results
```

```bash
python src/evaluation/run_all_benchmarks.py \
    --model_path results/triton_sparse_dpo_meta_llama_llama_3_1_8b_instruct_fixed/final_model \
    --output_dir ./llama_3.1_8B_97.5_eval_results
```


The model loader will automatically detect and use `.safetensors` files if present.

#### Loading Models Saved by sparse_DPO_v3.py

Models saved by `sparse_DPO_v3.py` with the `--save_model` flag are saved to `{run_dir}/final_model/` with `.safetensors` format. You can evaluate them directly:

```bash
# Example: evaluate a model saved from training
python src/evaluation/run_all_benchmarks.py \
    --model_path results/my_training_run/final_model \
    --output_dir ./evaluation_results
```

The evaluators automatically detect and load `.safetensors` files - no special configuration needed!

### Programmatic Usage

You can also use the evaluators programmatically:

```python
from evaluation import evaluate_mmlu, evaluate_math, evaluate_squad, evaluate_gpqa_diamond

# Evaluate on MMLU
mmlu_results = evaluate_mmlu(
    model_path="google/gemma-2-2b",
    num_fewshot=5,
    batch_size=8
)

# Evaluate on MATH
math_results = evaluate_math(
    model_path="./models/my_model",
    num_fewshot=4,
    limit=100
)

# Evaluate on SQuAD
squad_results = evaluate_squad(
    model_path="google/gemma-2-2b",
    method="lm_eval",
    batch_size=8
)

# Evaluate on GPQA Diamond
gpqa_results = evaluate_gpqa_diamond(
    model_path="google/gemma-2-2b",
    num_fewshot=0,
    batch_size=8
)
```

## Model Loading

The `model_loader.py` module provides a unified interface for loading models:

```python
from evaluation.model_loader import load_model_and_tokenizer

# Load from HuggingFace
model, tokenizer = load_model_and_tokenizer("google/gemma-2-2b")

# Load from local path with .safetensors
model, tokenizer = load_model_and_tokenizer("./models/my_model")

# With custom settings
model, tokenizer = load_model_and_tokenizer(
    model_path="./models/my_model",
    dtype=torch.float16,
    device="cuda",
    trust_remote_code=True
)
```

## Output

Results are returned as dictionaries containing:
- Benchmark-specific metrics (accuracy, F1, exact match, etc.)
- Detailed per-task breakdowns (for MMLU)
- Raw evaluation outputs

When using `--output_dir`, results are saved as JSON files:
- `{benchmark}_results.json` - Individual benchmark results
- `all_benchmarks_summary.json` - Summary of all benchmarks

## Notes

- **Device**: Models are automatically loaded on CUDA if available, otherwise CPU/MPS
- **Dtype**: Automatically uses float16 on GPU, float32 on CPU
- **Memory**: Large models may require significant GPU memory. Consider using `--limit` to reduce dataset size for testing
- **Few-shot**: Default few-shot settings vary by benchmark (MMLU: 5, MATH: 4, GPQA: 0)
- **Chat templates**: For chat/instruct models, pass `--apply_chat_template` (or rely on auto-detection) to apply the model's chat template; disable with `--no-apply_chat_template` if needed

## Troubleshooting

### Import Error: lm-evaluation-harness not found

Install the required package:
```bash
pip install lm-eval
```

### Out of Memory

- Reduce `--batch_size`
- Use `--limit` to evaluate on a subset
- Use CPU instead of GPU (slower but uses less memory)

### Model Loading Issues

- Ensure the model path is correct
- For local paths, ensure the directory contains model files (`.safetensors` or `pytorch_model.bin`)
- For HuggingFace models, ensure you have internet access and proper authentication if needed
