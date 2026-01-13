# Sparse DPO Training Scripts

This directory contains the main training scripts for Sparse DPO, refactored for modularity and efficiency.

## Scripts

### 0. `DPO_train.py`
**Focus:** Original DPO training script.
Manually edit the delta logging frequency in the script. Current default is 250 steps w/ AdamW and effective batch size 16. 
- **Usage:**
  ```bash
  python src/full_training/DPO_train.py \
    --model_name "google/gemma-3-270m-it"
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
    --mask "masks/sparsity_95.0pct_magnitude_step50_jaccard.json" \
    --optimizer sparse_adamw \
    --use_wandb \
    --save_csv
  ```

### 2. `sparse_dpo_bsr.py` (formerly v4)
**Focus:** Backprop Ablations & BSR Sparse MLP.
- Injects `SparseLinearLayer` (BSR Backward) into the model.
- Uses custom autograd function for sparse gradient computation.

- **Usage:**
  ```bash
  python src/full_training/sparse_dpo_bsr.py \
    --model_name "google/gemma-3-270m-it" \
    --mask "masks/sparsity_95.0pct_magnitude_step50_jaccard.json" \
    --optimizer sparse_adamw \
    --block_size_bsr 16 \
    --use_wandb
  ```

## Common Features
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
