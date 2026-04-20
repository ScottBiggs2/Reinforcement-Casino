# DPO 5k (dense + three sparse) — HPC copy-paste

Submit from **repo root** (`cd` to your clone, e.g. `$HOME/rl_casino`). Uses the same hyperparameters as `scratch.md` (sparse speed ablation block): LR `5e-7`, warmup `0.1`, lengths `1024`, batch `2`, grad accum `64`.

Sparse runs cover **random** global masks, **CAV** (`cav_cold_mask_finder.py`), and **SNIP** (`inference_mask_finder.py` + [`src/cold_start/utils/snip_scorer.py`](../src/cold_start/utils/snip_scorer.py)).

Use `sbatch --export=ALL ...` so `DPO_SAVE_STEPS`, `PIPELINE_RUN_ID`, and other exports reach the batch script (some sites default to a minimal environment).

**Stable run id:** set a fixed `DPO_RUN_ID` per experiment so outputs and resume paths stay the same across requeues.

**Rolling checkpoints:** `DPO_SAVE_STEPS=50`, `DPO_SAVE_TOTAL_LIMIT=3` (same idea as `scripts/grpo_openr1_llama31_slurm.sh`).

**Resume:** first submit leaves `DPO_RESUME` unset. If the job hits the wall before 5000 steps, resubmit the **same block** with `export DPO_RESUME=auto` (and the same `DPO_RUN_ID` / `PIPELINE_RUN_ID`).

HF checkpoints for dense live under:

`${TRAIN_OUT_BASE}/${DPO_RUN_ID}/checkpoints/<model>_<dataset>/checkpoint-*`

Sparse checkpoints live under:

`${SPARSE_OUT_BASE}/${DPO_RUN_ID}/<mask_stem>/checkpoints/checkpoint-*`

---

## One-command launcher (masks + queue DPO+GRPO)

If you want a **single Slurm job** that grabs **any GPU briefly** to generate the needed masks and then **queues all downstream DPO + GRPO jobs** with Slurm `afterok` dependencies, use:

```bash
cd /path/to/rl_casino
mkdir -p logs

# Required for gated models (e.g. Llama):
export HF_TOKEN="hf_xxxxxxxx"

sbatch scripts/orchestrate_masks_then_queue_dpo_grpo.slurm
```

Key overrides (optional; export before `sbatch`):

- **DPO run ids**: `DPO_DENSE_RUN_ID`, `DPO_SPARSE_RANDOM_RUN_ID`, `DPO_SPARSE_CAV_RUN_ID`, `DPO_SPARSE_SNIP_RUN_ID`
- **DPO mask config**: `DPO_DS_KEY` (default `tulu3`), `COLD_DATASET_HF` (default Tulu3 HF id), `DPO_SNIP_OBJECTIVE` (default `dpo_preference`)
- **DPO hyperparams** (same as below blocks): `NUM_STEPS_DPO`, `DPO_LEARNING_RATE`, `DPO_WARMUP_RATIO`, `DPO_MAX_LENGTH`, `DPO_MAX_PROMPT_LENGTH`, `DPO_PER_DEVICE_TRAIN_BATCH_SIZE`, `DPO_GRADIENT_ACCUMULATION_STEPS`

This doc’s manual blocks below remain useful if you want to submit only DPO runs without GRPO.

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
# Registry dataset key for tulu3 (sanitized name `tulu3`) — used in SNIP filenames below
export DPO_DS_KEY="${DPO_DS_KEY:-tulu3}"
export COLD_DATASET_HF="${COLD_DATASET_HF:-allenai/llama-3.1-tulu-3-8b-preference-mixture}"

# Random 97.5% mask (same selector style as the full pipeline)
export MASK_RANDOM="${MASK_DIR}/random_${MODEL//\//_}_sparsity97.5pct.pt"
python src/utils/generate_random_mask.py \
  --model_name "${MODEL}" \
  --sparsity_percent 97.5 \
  --min_layer_keep_ratio "${MIN_LAYER_KEEP_RATIO}" \
  --output_file "${MASK_RANDOM}"

# CAV 97.5% cold mask (standalone CAV script; HF dataset id for calibration)
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

### SNIP 97.5% (`inference_mask_finder.py`)

Unified entrypoint: [`src/cold_start/inference_mask_finder.py`](../src/cold_start/inference_mask_finder.py) with `--method snip`. Scoring lives in [`src/cold_start/utils/snip_scorer.py`](../src/cold_start/utils/snip_scorer.py):

| `--snip-objective` | What gets differentiated | Typical use |
|--------------------|--------------------------|-------------|
| `lm` | Causal LM loss on **chosen** sequences only (`SNIPScorer.score`: one backward over averaged CE). | Classic SNIP-style saliency; matches “score on positives only.” |
| `dpo_preference` | `dpo_style_preference_loss` on **chosen vs rejected** batches (`compute_snip_scores`: gradients of pairwise margin). | Closer to the **DPO preference signal**; uses the same DPO-style collator as other cold code when `--mode dpo`. |

Masks are built with `|grad ⊙ weight|` then [`build_snip_masks_from_scores`](../src/cold_start/utils/snip_scorer.py) → global top-`k` with `min_layer_keep_ratio` (same pooling story as other cold masks).

**Recommended for this DPO 5k study:** `dpo_preference` with `--snip-preference-beta` matching training `DPO_BETA` (default `0.1`). Use `lm` if you want a SNIP baseline comparable to generic LM pruning papers.

Filename matches [`pipeline_common.sh`](../scripts/pipeline_common.sh) when `PIPELINE_COLD_SNIP=1`: `cold_snip_<model_sanitized>_<ds_sanitized>_sparsity97.5pct_<objective>.pt`.

```bash
export COLD_SNIP_OBJECTIVE="${COLD_SNIP_OBJECTIVE:-dpo_preference}"
export MASK_SNIP="${MASK_DIR}/cold_snip_meta_llama_llama_3_1_8b_instruct_${DPO_DS_KEY}_sparsity97.5pct_${COLD_SNIP_OBJECTIVE}.pt"

python src/cold_start/inference_mask_finder.py \
  --model_name "${MODEL}" \
  --method snip \
  --mode dpo \
  --dataset_name "${DPO_DS_KEY}" \
  --snip-objective "${COLD_SNIP_OBJECTIVE}" \
  --n_samples "${COLD_CAV_SUBSET:-256}" \
  --sparsity 97.5 \
  --batch_size 4 \
  --max_length 1024 \
  --min-layer-keep-ratio "${MIN_LAYER_KEEP_RATIO}" \
  --snip-num-batches "${COLD_CAV_NUM_BATCHES:-16}" \
  --snip-preference-beta "${DPO_BETA:-0.1}" \
  --output "${MASK_SNIP}"
```

Notes:

- CLI uses **`--sparsity`** (percent weights zeroed), not `--sparsity_percent` (that name is for `cav_cold_mask_finder` / `generate_random_mask`).
- For `lm`, `--snip-num-batches` does not apply (`SNIPScorer.score` averages per-microbatch CE then runs **one** `backward()`).
- For `dpo_preference`, `compute_snip_scores` runs up to `--snip-num-batches` preference batches and calls `backward()` each time **without** `zero_grad` between them (gradients **accumulate**), then forms saliency from the accumulated grads—matching a multi-batch SNIP estimate, not independent runs per batch.
- If you switch `COLD_SNIP_OBJECTIVE=lm`, set `export COLD_SNIP_OBJECTIVE=lm` **before** defining `MASK_SNIP` so the path ends in `_lm.pt` instead of `_dpo_preference.pt`.

If any output path differs from the defaults above, point `MASK_CAV` / `MASK_SNIP` at the actual `.pt` under `MASK_DIR`.

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

## 4) Sparse DPO 5k — SNIP 97.5% mask (`MASK_SNIP` from section 0)

Use the same `SparseAdamW` / full-model path as sections 2–3. Point `PIPELINE_MASK_FILE` at `MASK_SNIP` (objective suffix `lm` or `dpo_preference` must match what you generated).

```bash
export REPO="${REPO:-$HOME/rl_casino}"
cd "$REPO" && mkdir -p logs

export SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/${USER}}"
export TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
export SPARSE_OUT_BASE="${SPARSE_OUT_BASE:-${SCRATCH_USER_ROOT}/rl_casino_sparse_train}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_USER_ROOT}/hf_cache/datasets}"

export PIPELINE_RUN_ID="${DPO_RUN_ID:-dpo5k_sparse_snip_tulu3}"
export RUN_ID_OVERRIDE="${PIPELINE_RUN_ID}"
export PIPELINE_MASK_FILE="${MASK_SNIP:?set MASK_SNIP from section 0 (SNIP block)}"

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
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
- **CAV:** default `MASK_CAV` filename matches `cav_cold_mask_finder.py` / [`pipeline_common.sh`](../scripts/pipeline_common.sh) (`cold_cav_<model_sanitized>_sparsity97.5pct.pt`). Change `MASK_CAV` if you change `MODEL` or use a different `--output_file`.
- **SNIP:** masks are standard `{"masks", "metadata"}` from `save_masks`. For **GRPO** calibration and `lm` / `dpo_preference` on math data, see [`docs/GRPO_HPC_COPYPASTE.md`](../docs/GRPO_HPC_COPYPASTE.md) (different `--mode` / `--dataset_name`). This doc uses `--mode dpo` and registry key `tulu3` to align with DPO training.
- If `MODEL` is not `meta-llama/Llama-3.1-8B-Instruct`, recompute sanitized names (`python -c "..."` using `sanitize_model_name` from `DPO_train.py`) or set `MASK_*` paths from `ls "$MASK_DIR"`.
