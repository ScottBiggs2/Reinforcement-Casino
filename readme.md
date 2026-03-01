# RL Acceleration Project

Reinforcement Learning training infrastructure with Triton-accelerated sparse optimization for DPO and GRPO.

## Implementation status

- **GRPO rewards:** Not implemented. GRPO trainer and sparse GRPO scripts run, but reward computation is placeholder/stub.
- **Cold start:** Not implemented. The `cold_start/` directory is for future cold-start training; no runnable pipeline yet.
- **BSR backprop:** Not yet tested. `sparse_dpo_bsr.py` uses BSR sparse MLP layers and custom sparse autograd; correctness and performance are unvalidated.

A note here: AdamW decays **all** weights (BSR AdamW does not), so using it with the mask decays frozen weights which is incorrect. 

Junk, no learning evident. 
```bash
python src/full_training/sparse_dpo_bsr.py \
  --mask masks/sparsity_97.5pct_magnitude_step50.pt \
  --n_steps 50 \
  --optimizer sgd \
  --use_wandb
```

Good but extremely unstable (work ongoing)
```bash
python src/full_training/sparse_dpo_bsr.py \
  --mask masks/sparsity_97.5pct_magnitude_step50.pt \
  --n_steps 50 \
  --optimizer sparse_adamw \
  --use_wandb
```

Good but extremely unstable. Like BSR AdamW. Not tested at long range.
```bash
python src/full_training/sparse_dpo_bsr.py \
  --mask masks/sparsity_97.5pct_magnitude_step50.pt \
  --n_steps 50 \
  --optimizer adamw \
  --use_wandb
```

Gradient clipping bsr test: 

```bash 
python src/full_training/sparse_dpo_bsr.py \
  --mask masks/sparsity_97.5pct_magnitude_step50.pt \
  --n_steps 50 \
  --optimizer sparse_adamw \
  --use_wandb \
  --max_grad_norm 0.5 \
  --adam_eps 1e-6
```

DPO Beta noise reduction test:

```bash
python src/full_training/sparse_dpo_bsr.py \
  --mask masks/sparsity_97.5pct_magnitude_step50.pt \
  --n_steps 50 \
  --optimizer sparse_adamw \
  --use_wandb \
  --max_grad_norm 1.0 \
  --dpo_beta 0.5
```

LR and Warmup Stabilisation test:

```bash
python src/full_training/sparse_dpo_bsr.py \
  --mask masks/sparsity_97.5pct_magnitude_step50.pt \
  --n_steps 50 \
  --optimizer sparse_adamw \
  --use_wandb \
  --max_grad_norm 1.0 \
  --lr 5e-6 \
  --warmup_steps 10
```

---

## Prerequisites

- **Python 3.11** (required for TRL compatibility)
- **H100 GPU** (required for optimal performance - or H100, I always mix them up)
- **Wandb account** (for experiment tracking)
- **HuggingFace account** (for model access)

### Authentication

```bash
# Wandb login
wandb login [YOUR_KEY_HERE]

# HuggingFace login
hf auth login [YOUR_KEY_HERE]
```

## Project structure

```
src/
├── cold_start/          # Cold start (not implemented)
├── full_training/       # Full training pipelines (DPO, GRPO) + sparse DPO
├── lora_training/       # LoRA training scripts
├── magic/               # Triton-accelerated sparse training (legacy scripts)
├── utils/               # Utilities (checkpoint reconstruction, etc.)
└── warm_start/         # Mask finding and warm start utilities
```

## Full training (`src/full_training/`)

### DPO training

Train with Direct Preference Optimization on the Light-R1 dataset.

**Script:** `DPO_train.py`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | `google/gemma-3-270m-it` | HuggingFace model name |

```bash
# Default model
python src/full_training/DPO_train.py

# Custom model
python src/full_training/DPO_train.py --model_name "meta-llama/Llama-3.1-8B-Instruct"
```

**Output paths (determined by `--model_name` via sanitized name):**

| Output | Path pattern | Example (`google/gemma-3-270m-it`) |
|--------|----------------|-------------------------------------|
| Checkpoints | `./checkpoints_{sanitized}_dpo/` | `./checkpoints_google_gemma_3_270m_it_dpo/` |
| Delta logs | `./delta_logs_{sanitized}/` | `./delta_logs_google_gemma_3_270m_it/` |
| Wandb project | `{sanitized}-dpo-subnetwork-emergence` | `google_gemma_3_270m_it-dpo-subnetwork-emergence` |

**Checkpoint / delta schedule:** Defined in-script (not a CLI arg). Default schedule saves deltas at steps **10, 20, 30, 40, 100, 150, 200**. Delta files are `deltas_step_{step}.pt` and `base_state.pt` inside the delta log dir. To change which steps are saved, edit `CHECKPOINT_SCHEDULE` and `NUM_STEPS` in `DPO_train.py`.

**Mask finding:** The mask finder (`even_better_mask_finder.py`) reads from the **delta log directory** produced by this script. Use `--delta_log_dir` equal to that path (e.g. `./delta_logs_google_gemma_3_270m_it`). Use `--target_step` equal to a step where deltas were saved (one of the schedule steps above), or omit for the latest available step.

### GRPO training

Train with Group Relative Policy Optimization on the OpenR1 dataset. **GRPO rewards are not implemented**; reward logic is placeholder.

**Script:** `GRPO_train.py`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | `google/gemma-3-270m-it` | HuggingFace model name |

```bash
python src/full_training/GRPO_train.py
python src/full_training/GRPO_train.py --model_name "Qwen/Qwen2.5-0.5B-Instruct"
```

**Outputs:** `./checkpoints_{sanitized}_grpo/`, `./delta_logs_{sanitized}_grpo/`, Wandb project `{sanitized}-grpo-subnetwork-emergence`.

### Sparse DPO (efficiency)

Triton-accelerated sparse DPO with optimizer ablations (SGD, AdamW, Sparse AdamW).

**Script:** `sparse_dpo_efficiency.py`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | `google/gemma-3-270m-it` | Base model name |
| `--checkpoint` | str | None | Checkpoint path (None = use `model_name`) |
| `--mask` | str | `masks/top_10.0pct_momentum_w25_step25.pt` | Sparse mask file |
| `--n_steps` | int | 100 | Training steps |
| `--batch_size` | int | 4 | Batch size |
| `--grad_accum` | int | 4 | Gradient accumulation steps |
| `--lr` | float | 5e-5 | Learning rate |
| `--subset_size` | int | None | Dataset subset size (None = full) |
| `--optimizer` | str | `sparse_adamw` | `sgd`, `adamw`, or `sparse_adamw` |
| `--block_size` | int | 32 | Block size for sparse AdamW |
| `--mlp_only` | flag | True | Restrict sparsity to MLP layers only |
| `--use_wandb` | flag | False | Log to Wandb |
| `--save_csv` | flag | False | Save per-step CSV logs |

```bash
python src/full_training/sparse_dpo_efficiency.py \
  --model_name "google/gemma-3-270m-it" \
  --mask "masks/sparsity_95.0pct_magnitude_step50.pt" \
  --optimizer sparse_adamw \
  --n_steps 50 \
  --use_wandb \
  --save_csv
```

### Sparse DPO (BSR backprop)

BSR sparse MLP layers with custom sparse autograd. **BSR backprop is not yet tested**; use for experimentation only.

**Script:** `sparse_dpo_bsr.py`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | `google/gemma-3-270m-it` | Base model name |
| `--checkpoint` | str | None | Checkpoint path (None = use `model_name`) |
| `--mask` | str | `masks/top_10.0pct_momentum_w25_step25.pt` | Sparse mask file |
| `--n_steps` | int | 10 | Training steps |
| `--batch_size` | int | 1 | Batch size |
| `--grad_accum` | int | 8 | Gradient accumulation steps |
| `--lr` | float | 5e-5 | Learning rate |
| `--subset_size` | int | None | Dataset subset size (None = full) |
| `--optimizer` | str | `sparse_adamw` | `sgd`, `adamw`, or `sparse_adamw` |
| `--block_size_bsr` | int | 16 | BSR block size for sparse MLP |
| `--block_size_adam` | int | 128 | Block size for sparse AdamW |
| `--mlp_only` | flag | True | Restrict sparsity to MLP layers only |
| `--use_wandb` | flag | False | Log to Wandb |
| `--save_csv` | flag | False | Save per-step CSV logs |

```bash
python src/full_training/sparse_dpo_bsr.py \
  --model_name "google/gemma-3-270m-it" \
  --mask "masks/sparsity_95.0pct_magnitude_step50.pt" \
  --optimizer sparse_adamw \
  --block_size_bsr 16 \
  --use_wandb

python src/full_training/sparse_dpo_bsr.py \
  --model_name "google/gemma-3-270m-it" \
  --mask "masks/sparsity_97.5pct_magnitude_step50.pt" \
  --optimizer sgd \
  --block_size_bsr 16 \
  --n_steps 100 \
  --use_wandb
```

### Timing baselines (dense)

Minimal dense runs for timing comparison (no delta logging, minimal checkpointing).

**DPO baseline:** `DPO_timing_baseline.py`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | None | Checkpoint path (None = base model) |
| `--n_steps` | int | 10 | Training steps |
| `--batch_size` | int | 4 | Batch size |
| `--learning_rate` | float | 5e-5 | Learning rate |
| `--subset_size` | int | 10 | Dataset subset size |
| `--optimizer` | str | `sgd` | `sgd` or `adamw` |
| `--save_model` | flag | False | Save final model to safetensors |

```bash
python src/full_training/DPO_timing_baseline.py \
  --n_steps 100 \
  --batch_size 4 \
  --subset_size 1000 \
  --optimizer adamw \
  --save_model
```

**GRPO baseline:** `GRPO_timing_baseline.py` — **GRPO rewards not implemented.**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | None | Checkpoint path (None = base model) |
| `--n_steps` | int | 10 | Training steps |
| `--batch_size` | int | 1 | Batch size |
| `--learning_rate` | float | 5e-5 | Learning rate |
| `--subset_size` | int | 10 | Dataset subset size |

```bash
python src/full_training/GRPO_timing_baseline.py \
  --checkpoint None \
  --n_steps 50 \
  --batch_size 1 \
  --subset_size 10
```

## Sparse training (legacy, `src/magic/`)

Older Triton scripts; for new work prefer `src/full_training/sparse_dpo_efficiency.py` and `sparse_dpo_bsr.py`.

### Sparse DPO (v2 / v3)

```bash
python src/magic/sparse_DPO_v2.py --model_name "google/gemma-3-270m-it" --checkpoint None \
  --mask masks/top_10.0pct_momentum_w25_step25.pt --n_steps 50

python src/magic/sparse_DPO_v3.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --checkpoint None \
  --mask masks/sparsity_97.5pct_fisher_step50.pt --n_steps 100 --batch_size 1 --subset_size 100 --learning_rate 5e-5
```

### Sparse GRPO (v2)

**GRPO reward function is not implemented.**

```bash
python src/magic/sparse_GRPO_v2.py --model_name "google/gemma-3-270m-it" --checkpoint None \
  --mask masks/top_10.0pct_momentum_w25_step25.pt --n_steps 50
```

## Mask finding

**Uses output from `src/full_training/DPO_train.py`.** The mask finder reads the **delta log directory** that DPO training writes (see [DPO training](#dpo-training) for path format and checkpoint schedule).

**Argument setup:**

- **`--delta_log_dir`** — Must match the delta log dir produced by `DPO_train.py` for the same model. Format: `./delta_logs_{sanitized}/` where `{sanitized}` is the model name with slashes/dashes lowercased and replaced by underscores (e.g. `./delta_logs_google_gemma_3_270m_it`).
- **`--target_step`** — Step at which to compute the mask. Must be a step where DPO_train saved deltas (default schedule: 10, 20, 30, 40, 100, 150, 200). Omit to use all available steps up to the latest.

Script: `src/warm_start/even_better_mask_finder.py`.

### Magnitude-based

```bash
python src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "delta_logs_google_gemma_3_270m_it" \
  --method magnitude \
  --sparsity_percent 97.5 \
  --target_step 50 \
  --compute_jaccard \
  --debug
```

### Momentum-based

```bash
python src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "delta_logs_meta_llama_llama_3_1_8b_instruct" \
  --method momentum \
  --sparsity_percent 97.5 \
  --target_step 40 \
  --compute_jaccard \
  --debug
```

### Fisher-based

```bash
python src/warm_start/even_better_mask_finder.py \
  --delta_log_dir "delta_logs_meta_llama_llama_3_2_3b_instruct" \
  --method fisher \
  --sparsity_percent 97.5 \
  --target_step 100 \
  --compute_jaccard \
  --debug
```

## Checkpoint reconstruction

Reconstruct full checkpoints from delta files:

```bash
# Reconstruct at step 100
python src/utils/reconstruct_checkpoint.py \
  --model_name "google/gemma-3-270m-it" \
  --delta_log_dir ./delta_logs_google_gemma_3_270m_it \
  --step 100 \
  --output_dir ./reconstructed_checkpoints/step_100

# List available steps
python src/utils/reconstruct_checkpoint.py \
  --delta_log_dir ./delta_logs_google_gemma_3_270m_it \
  --list_steps
```

## Model name sanitization

Scripts sanitize HuggingFace model names for paths:

- `google/gemma-3-270m-it` → `google_gemma_3_270m_it`
- `meta-llama/Llama-3.1-8B` → `meta_llama_llama_3_1_8b`

## Notes

- **Delta logging:** Training scripts can save parameter deltas (vs initial state) instead of full checkpoints to save disk.
- **Checkpoint schedules:** Configurable step schedules control when deltas are written.
- **Wandb:** Runs log to Wandb with model-specific project names when enabled.
- **Sparse training:** Triton kernels only update non-zero mask elements for speed at high sparsity.
