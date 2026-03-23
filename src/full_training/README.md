# Sparse DPO Training Scripts

This directory contains the main training scripts for Sparse DPO, refactored for modularity and efficiency.

## Scripts

### 0. `DPO_train.py`
**Focus:** Original DPO training script with multi-dataset support.
Manually edit the delta logging frequency in the script. Current default is 250 steps w/ AdamW and effective batch size 16.

**Available datasets** (via `--dataset`):
| Key | Domain | HuggingFace ID |
|---|---|---|
| `light-r1` (default) | Math/Reasoning | `qihoo360/Light-R1-DPOData` |
| `tulu3` | Instruction Following | `allenai/llama-3.1-tulu-3-8b-preference-mixture` |
| `math-step-dpo` | Math | `xinlai/Math-Step-DPO-10K` |
| `codepref` | Coding | `Vezora/Code-Preference-Pairs` |

- **Usage:**
  ```bash
  # Default (Light-R1, backward compatible)
  python src/full_training/DPO_train.py \
    --model_name "google/gemma-3-270m-it"

  # Math domain
  python src/full_training/DPO_train.py \
    --model_name "google/gemma-3-270m-it" \
    --dataset math-step-dpo

  # Quick verification (10 steps, 64 examples)
  python src/full_training/DPO_train.py \
    --model_name "google/gemma-3-270m-it" \
    --dataset tulu3 \
    --num_steps 10 --subset_size 64
  ```

### 0.5 ``
**Focus** Get a mask.
Build masks. 
```bash
python src/warm_start/even_better_mask_finder.py \
    --delta_log_dir "delta_logs_google_gemma_3_270m_it" \
    --method magnitude \
    --sparsity_percent 95.0 \
    --target_step 50 \
    --compute_jaccard \
    --debug
```

### 1. `sparse_dpo_efficiency.py` (formerly v3)
**Focus:** Optimizer Ablations & Training Efficiency.
- Uses **Indexed Sparse AdamW** kernel for fast updates.
- Supports `sgd`, `adamw`, and `sparse_adamw` optimizers.
- **Usage:**
```bash
python src/full_training/sparse_dpo_efficiency.py \
--model_name "google/gemma-3-270m-it" \
--mask "masks/sparsity_95.0pct_magnitude_step50.pt" \
--optimizer sparse_adamw \
--use_wandb \
--save_csv \
--n_steps 50 
```

### 2. `sparse_dpo_bsr.py` (formerly v4)
**Focus:** Backprop Ablations & BSR Sparse MLP.
- Injects `SparseLinearLayer` (BSR Backward) into the model.
- Uses custom autograd function for sparse gradient computation.

- **Usage:**
  ```bash
  python src/full_training/sparse_dpo_bsr.py \
    --model_name "google/gemma-3-270m-it" \
    --mask "masks/sparsity_95.0pct_magnitude_step50.pt" \
    --optimizer sparse_adamw \
    --block_size_bsr 16 \
    --use_wandb
  ```

## Common Features
- **Scratch Storage:** Default output path is `/scratch/biggs.s/rl_casino_outputs` and dataset cache is `/scratch/biggs.s/hf_cache/datasets` to reduce home directory quota usage. Override with `--output_base_dir` and `--dataset_cache_dir`.
- **Flexible Checkpointing:** Saves deltas on a schedule (10, 20... 50, 100...).
- **Logging:** 
    - WandB: Metrics + Subnetwork statistics.
    - CSV: Per-step training logs (if `--save_csv` is set).
    - `deltas/`: JSON statistics and delta checkpoints.
- **Data Loading:** Unified utilities in `src/utils/data_utils.py`.

## Directory Structure
- `src/kernels/`: Extracted Triton kernels.
- `src/optimizers/`: Optimizer implementations (`SparseAdamW`).
- `src/mlps/`: Sparse layer implementations (`SparseLinearLayer`).
- `src/utils/`: Utilities for data, masks, and logging.
