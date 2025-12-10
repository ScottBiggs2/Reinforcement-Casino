# RL Acceleration Project

Reinforcement Learning training infrastructure with Triton-accelerated sparse optimization for DPO and GRPO.

## Prerequisites

- **Python 3.11** (required for TRL compatibility)
- **H200 GPU** (required for optimal performance)
- **Wandb account** (for experiment tracking)
- **HuggingFace account** (for model access)

### Authentication

```bash
# Wandb login
wandb login [YOUR_KEY_HERE]

# HuggingFace login
hf auth login [YOUR_KEY_HERE]
```

## Project Structure

```
src/
├── cold_start/          # Cold start training scripts
├── full_training/      # Full training pipelines (DPO, GRPO)
├── lora_training/      # LoRA training scripts
├── magic/              # Triton-accelerated sparse training
├── utils/              # Utility scripts (checkpoint reconstruction, etc.)
└── warm_start/         # Mask finding and warm start utilities
```

## Full Training Scripts

### DPO Training

Train with Direct Preference Optimization on the Light-R1 dataset:

```bash
# Default model (google/gemma-3-270m-it)
python src/full_training/DPO_train.py

# Custom model
python src/full_training/DPO_train.py \
    --model_name "meta-llama/Llama-3.1-8B"
```

**Outputs:**
- Checkpoints: `./checkpoints_{model_name}_dpo/`
- Delta logs: `./delta_logs_{model_name}/`
- Wandb project: `{model_name}-dpo-subnetwork-emergence`

### GRPO Training

Train with Group Relative Policy Optimization on the OpenR1 dataset:

```bash
# Default model (google/gemma-3-270m-it)
python src/full_training/GRPO_train.py

# Custom model
python src/full_training/GRPO_train.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct"
```

**Outputs:**
- Checkpoints: `./checkpoints_{model_name}_grpo/`
- Delta logs: `./delta_logs_{model_name}_grpo/`
- Wandb project: `{model_name}-grpo-subnetwork-emergence`

## Sparse Training (Triton-Accelerated)

### Sparse DPO Training

Train with Triton-accelerated sparse optimization for DPO:

```bash
# From base model
python src/magic/sparse_DPO_v2.py \
    --model_name "google/gemma-3-270m-it" \
    --checkpoint None \
    --mask masks/top_10.0pct_momentum_w25_step25.pt \
    --n_steps 50

# From checkpoint [Try this!]
python src/magic/sparse_DPO_v3.py \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --checkpoint None \
    --mask masks/sparsity_97.5pct_fisher_step50.pt \
    --n_steps 100 \
    --batch_size 1 \
    --subset_size 100 \
    --learning_rate 5e-5
```

### Sparse GRPO Training

Train with Triton-accelerated sparse optimization for GRPO:

```bash
# From base model
python src/magic/sparse_GRPO_v2.py \
    --model_name "google/gemma-3-270m-it" \
    --checkpoint None \
    --mask masks/top_10.0pct_momentum_w25_step25.pt \
    --n_steps 50

# From checkpoint with custom settings
python src/magic/sparse_GRPO_v2.py \
    --model_name "google/gemma-3-270m-it" \
    --checkpoint checkpoints_google_gemma_3_270m_it_grpo/checkpoint-1000 \
    --mask masks/sparsity_90.0pct_magnitude_step100.pt \
    --n_steps 100 \
    --batch_size 1 \
    --learning_rate 5e-5 \
    --mlp_only
```

**Key Features:**
- Pre-initialized optimizer states
- Indexed sparse kernels (only processes non-zero mask elements)
- Gather/scatter operations for true sparse computation

## Timing Baselines

### DPO Baseline

```bash
python src/full_training/DPO_timing_baseline.py \
    --n_steps 100 \
    --batch_size 1 \
    --subset_size 100
```

### GRPO Baseline

```bash
python src/full_training/GRPO_timing_baseline.py \
    --checkpoint None \
    --n_steps 50 \
    --batch_size 1 \
    --subset_size 10
```

## Mask Finding

Find optimal sparse masks using various methods:

### Magnitude-based (simplest)

```bash
python src/warm_start/even_better_mask_finder.py \
    --delta_log_dir "delta_logs_meta_llama_llama_3_2_3b_instruct"
    --method magnitude \
    --sparsity_percent 90.0 \
    --target_step 100 \
    --compute_jaccard \
    --debug
```

### Momentum-based

```bash
python src/warm_start/even_better_mask_finder.py \
    --delta_log_dir "delta_logs_meta_llama_llama_3_2_3b_instruct" \
    --method momentum \
    --sparsity_percent 95.0 \
    --target_step 50 \
    --momentum_window 50 \
    --compute_jaccard \
    --debug
```

### Fisher Information-based

```bash
python src/warm_start/even_better_mask_finder.py \
    --delta_log_dir "delta_logs_meta_llama_llama_3_2_3b_instruct" \
    --method fisher \
    --sparsity_percent 97.5 \
    --target_step 100 \
    --compute_jaccard \
    --debug
```

## Checkpoint Reconstruction

Reconstruct full model checkpoints from delta files:

```bash
# Reconstruct checkpoint at step 100
python src/utils/reconstruct_checkpoint.py \
    --model_name "google/gemma-3-270m-it" \
    --delta_log_dir ./delta_logs_google_gemma_3_270m_it \
    --step 100 \
    --output_dir ./reconstructed_checkpoints/step_100

# List available delta steps
python src/utils/reconstruct_checkpoint.py \
    --delta_log_dir ./delta_logs_google_gemma_3_270m_it \
    --list_steps
```

## Model Name Sanitization

All scripts automatically sanitize HuggingFace model names for filesystem-safe paths:

- `"google/gemma-3-270m-it"` → `"google_gemma_3_270m_it"`
- `"meta-llama/Llama-3.1-8B"` → `"meta_llama_llama_3_1_8b"`

This ensures:
- Separate directories for different models
- No path conflicts
- Clear organization of checkpoints and deltas

## Notes

- **Delta Logging**: Training scripts save parameter deltas (changes from initial state) rather than full checkpoints, saving significant disk space
- **Checkpoint Schedules**: Flexible checkpoint schedules save deltas at specific training steps
- **Wandb Integration**: All training runs are automatically logged to Wandb with model-specific project names
- **Sparse Training**: Triton kernels only process non-zero mask elements, providing massive speedups for high sparsity
