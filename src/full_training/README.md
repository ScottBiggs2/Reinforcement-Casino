# Sparse DPO Training Scripts

This directory contains the main training scripts for Sparse DPO, refactored for modularity and efficiency.

## Scripts

### 1. `sparse_dpo_efficiency.py` (formerly v3)
**Focus:** Optimizer Ablations & Training Efficiency.
- Uses **Indexed Sparse AdamW** kernel for fast updates.
- Supports `sgd`, `adamw`, and `sparse_adamw` optimizers.
- **Usage:**
  ```bash
  python src/full_training/sparse_dpo_efficiency.py \
    --model_name "google/gemma-3-270m-it" \
    --mask "masks/my_mask.pt" \
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
    --mask "masks/my_mask.pt" \
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
