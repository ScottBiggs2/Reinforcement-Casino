## Multi-GPU pipeline (parallel to existing single-GPU pipeline)

This folder contains **multi-GPU Slurm entrypoints** for the existing RL Casino pipeline.

- **Goal**: keep the original `scripts/pipeline_stage_*.sh` + `scripts/pipeline_common.sh` untouched, while providing multi-GPU alternatives that are explicit about world size and hyperparameters.
- **Default model scope**: **Llama 3.1 8B** (same as `scripts/pipeline_common.sh`).

### Explorer multi-GPU usage

Explorer’s multi-GPU access process and reservation workflow are described here:
- [Northeastern RC: Access to Multi-GPU Partition](https://rc-docs.northeastern.edu/en/explorer-main/gpus/multigpu-partition-access.html)

These scripts default to `#SBATCH --partition=multigpu` and allow optional reservation usage via env vars.

### Key differences vs single-GPU pipeline

- **Dense DPO stage** uses `torchrun` to launch multi-process training (1 process per GPU).
- `src/full_training/DPO_train.py` has been extended to accept paper-aligned hyperparameters and to write warm-start artifacts **only on rank 0** to avoid multi-process file corruption.

### Quick start (conceptual)

1. Set reservation vars (optional):
   - `export MULTIGPU_RESERVATION=<name>` (if you are running under a reservation)
   - `export MULTIGPU_GPU_TYPE=h200` (or whatever your reservation specifies)
   - `export MULTIGPU_NGPUS=4`

2. Submit stage 1:
   - `sbatch scripts/multigpu_pipeline/pipeline_stage_01_dense_dpo_multigpu.sh`

Stages 2–5 can be run from the single-GPU pipeline scripts, or adapted later (we keep them separate on purpose).

