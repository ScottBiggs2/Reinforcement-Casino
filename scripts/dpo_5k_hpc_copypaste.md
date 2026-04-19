# DPO 5k (dense + two sparse) — HPC copy-paste

Submit from **repo root** (`cd` to your clone, e.g. `$HOME/rl_casino`). Uses the same hyperparameters as `scratch.md` (sparse speed ablation block): LR `5e-7`, warmup `0.1`, lengths `1024`, batch `2`, grad accum `64`.

Use `sbatch --export=ALL ...` so `DPO_SAVE_STEPS`, `PIPELINE_RUN_ID`, and other exports reach the batch script (some sites default to a minimal environment).

**Stable run id:** set a fixed `DPO_RUN_ID` per experiment so outputs and resume paths stay the same across requeues.

**Rolling checkpoints:** `DPO_SAVE_STEPS=50`, `DPO_SAVE_TOTAL_LIMIT=3` (same idea as `scripts/grpo_openr1_llama31_slurm.sh`).

**Resume:** first submit leaves `DPO_RESUME` unset. If the job hits the wall before 5000 steps, resubmit the **same block** with `export DPO_RESUME=auto` (and the same `DPO_RUN_ID` / `PIPELINE_RUN_ID`).

HF checkpoints for dense live under:

`${TRAIN_OUT_BASE}/${DPO_RUN_ID}/checkpoints/<model>_<dataset>/checkpoint-*`

Sparse checkpoints live under:

`${SPARSE_OUT_BASE}/${DPO_RUN_ID}/<mask_stem>/checkpoints/checkpoint-*`

---

## 0) One-time: generate masks (GPU; can use `srun` or a short `sbatch`)

Set paths (example):

```bash
export REPO="${REPO:-$HOME/rl_casino}"
cd "$REPO" || exit 1

export SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER}}"
export MASK_RUN_ID="${MASK_RUN_ID:-dpo5k_masks_$(date +%Y%m%d)}"
export MASK_DIR="${MASK_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_masks}/${MASK_RUN_ID}"
mkdir -p "$MASK_DIR"

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export MIN_LAYER_KEEP_RATIO="${MIN_LAYER_KEEP_RATIO:-0.0025}"

# Random 97.5% mask (same selector style as the full pipeline)
export MASK_RANDOM="${MASK_DIR}/random_${MODEL//\//_}_sparsity97.5pct.pt"
python src/utils/generate_random_mask.py \
  --model_name "${MODEL}" \
  --sparsity_percent 97.5 \
  --output_file "${MASK_RANDOM}"

# CAV 97.5% cold mask (requires HF dataset id for calibration — tulu3 preference mixture)
export COLD_DATASET_HF="${COLD_DATASET_HF:-allenai/llama-3.1-tulu-3-8b-preference-mixture}"
export MASK_CAV="${MASK_DIR}/cold_cav_meta_llama_llama_3_1_8b_instruct_sparsity97.5pct.pt"
python src/cold_start/cav_cold_mask_finder.py \
  --model_name "${MODEL}" \
  --dataset_name "${COLD_DATASET_HF}" \
  --method cav \
  --sparsity_percent 97.5 \
  --subset_size "${COLD_CAV_SUBSET:-256}" \
  --num_batches "${COLD_CAV_NUM_BATCHES:-16}" \
  --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
  --output_file "${MASK_CAV}"
```

If the CAV output filename differs, set `MASK_CAV` to the actual path (see `MASK_DIR` after the run).

---

## 1) Dense DPO 5k (no pipeline stage 2+)

```bash
export REPO="${REPO:-$HOME/rl_casino}"
cd "$REPO" && mkdir -p logs

export SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER}}"
export TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
export TRAIN_OUT_BASE="${TRAIN_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_train}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"

export PIPELINE_CHAIN_NEXT_STAGE=0
export PIPELINE_RUN_ID="${DPO_RUN_ID:-dpo5k_dense_tulu3}"
export RUN_ID_OVERRIDE="${PIPELINE_RUN_ID}"

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
# DPO dataset defaults to tulu3 in pipeline_common.sh (omit DPO_DATASETS unless you change it in the repo).
export NUM_STEPS_DPO=5000
export DPO_LEARNING_RATE=5e-7
export DPO_WARMUP_RATIO=0.1
export DPO_MAX_LENGTH=1024
export DPO_MAX_PROMPT_LENGTH=1024
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE=2
export DPO_GRADIENT_ACCUMULATION_STEPS=64

export DPO_SAVE_STEPS=50
export DPO_SAVE_TOTAL_LIMIT=3
# export DPO_RESUME=auto   # uncomment on continuation jobs

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_CONSOLE=off

sbatch --export=ALL scripts/pipeline_stage_01_dense.sh
```

---

## 2) Sparse DPO 5k — random 97.5% mask (`SparseAdamW`, full model — do not pass `--mlp_only`)

Point `PIPELINE_MASK_FILE` at `MASK_RANDOM` from section 0.

```bash
export REPO="${REPO:-$HOME/rl_casino}"
cd "$REPO" && mkdir -p logs

export SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER}}"
export TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
export SPARSE_OUT_BASE="${SPARSE_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_sparse_train}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"

export PIPELINE_RUN_ID="${DPO_RUN_ID:-dpo5k_sparse_random_tulu3}"
export RUN_ID_OVERRIDE="${PIPELINE_RUN_ID}"
export PIPELINE_MASK_FILE="${MASK_RANDOM:?set MASK_RANDOM from section 0}"

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
# DPO dataset defaults to tulu3 in pipeline_common.sh (omit DPO_DATASETS unless you change it in the repo).
export NUM_STEPS_DPO=5000
export DPO_LEARNING_RATE=5e-7
export DPO_WARMUP_RATIO=0.1
export DPO_MAX_LENGTH=1024
export DPO_MAX_PROMPT_LENGTH=1024
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE=2
export DPO_GRADIENT_ACCUMULATION_STEPS=64

export DPO_SAVE_STEPS=50
export DPO_SAVE_TOTAL_LIMIT=3
# export DPO_RESUME=auto

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_CONSOLE=off

sbatch --export=ALL scripts/pipeline_sparse_one_mask.sh
```

---

## 3) Sparse DPO 5k — CAV 97.5% mask

```bash
export REPO="${REPO:-$HOME/rl_casino}"
cd "$REPO" && mkdir -p logs

export SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER}}"
export TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
export SPARSE_OUT_BASE="${SPARSE_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_sparse_train}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"

export PIPELINE_RUN_ID="${DPO_RUN_ID:-dpo5k_sparse_cav_tulu3}"
export RUN_ID_OVERRIDE="${PIPELINE_RUN_ID}"
export PIPELINE_MASK_FILE="${MASK_CAV:?set MASK_CAV from section 0}"

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
# DPO dataset defaults to tulu3 in pipeline_common.sh (omit DPO_DATASETS unless you change it in the repo).
export NUM_STEPS_DPO=5000
export DPO_LEARNING_RATE=5e-7
export DPO_WARMUP_RATIO=0.1
export DPO_MAX_LENGTH=1024
export DPO_MAX_PROMPT_LENGTH=1024
export DPO_PER_DEVICE_TRAIN_BATCH_SIZE=2
export DPO_GRADIENT_ACCUMULATION_STEPS=64

export DPO_SAVE_STEPS=50
export DPO_SAVE_TOTAL_LIMIT=3
# export DPO_RESUME=auto

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_CONSOLE=off

sbatch --export=ALL scripts/pipeline_sparse_one_mask.sh
```

---

## Notes

- `pipeline_stage_01_dense.sh` still passes `--use_wandb` to `DPO_train.py`; with `WANDB_MODE=disabled` training should stay offline. If your site requires no W&B init at all, set `WANDB_DISABLED=true` (above) or add a follow-up to make W&B optional in `run_dense_dpo`.
- Sparse runs use `src/full_training/sparse_dpo_efficiency.py` with `--optimizer sparse_adamw` (not BSR).
- For CAV, the default output filename matches `pipeline_common.sh` / `cav_cold_mask_finder.py` naming for `meta-llama/Llama-3.1-8B-Instruct`; adjust `MASK_CAV` if you change `MODEL`.
