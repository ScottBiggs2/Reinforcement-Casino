# GRPO HPC — copy-paste launch blocks

Use these blocks on the cluster **from the repo root** (so `PYTHONPATH` and `scripts/` paths resolve). Paths follow [`scripts/pipeline_common.sh`](../scripts/pipeline_common.sh): fixed netid scratch (`SCRATCH_USER_ROOT`), GRPO artifacts **separate** from DPO (`rl_casino_grpo/...`).

**Replace** `YOUR_REPO_ROOT` if you `cd` elsewhere. **Set** `HF_TOKEN` for gated models (e.g. Llama).

---

## Shared environment (run once per shell session)

```bash
# Repo (Slurm sets SLURM_SUBMIT_DIR to cwd when you sbatch from repo root)
export REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "$REPO_ROOT"

# Scratch: match DPO pipeline style — fixed netid tree (see pipeline_common.sh)
export SCRATCH_USER_ROOT="${SCRATCH_USER_ROOT:-/scratch/biggs.s}"
export RL_CASINO_SCRATCH_ROOT="${RL_CASINO_SCRATCH_ROOT:-$SCRATCH_USER_ROOT}"

# Training env (adjust if your conda env lives elsewhere)
export TRAIN_ENV="${TRAIN_ENV:-${SCRATCH_USER_ROOT}/conda_envs/rl_casino}"
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# HF cache on scratch (default for grpo_openr1_llama31_slurm.sh)
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${RL_CASINO_SCRATCH_ROOT}/hf_cache/datasets}"

# GRPO mask directory and defaults (see scripts/grpo_openr1_llama31_slurm.sh)
export GRPO_MASK_DIR="${GRPO_MASK_DIR:-${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/masks}"
mkdir -p "$GRPO_MASK_DIR" logs

# Model / data (Open-R1 Math + Llama 3.1 8B Instruct)
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN for gated models}"
export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export GRPO_DATASET="${GRPO_DATASET:-math-220k}"

# Sequence caps — align dense and sparse (override 2048 if you use a different cap)
export GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-2048}"
export GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-512}"

# Reward profile (Llama Instruct: default llama_cot)
export GRPO_REWARD_PROFILE="${GRPO_REWARD_PROFILE:-llama_cot}"
```

**Artifact locations (defaults):**

| Kind | Path |
|------|------|
| Dense checkpoints | `${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/dense/<run_slug>/checkpoints/` |
| Sparse runs | `${RL_CASINO_SCRATCH_ROOT}/rl_casino_grpo/sparse/<GRPO_RUN_NAME>/` |
| Masks | `${GRPO_MASK_DIR}/` (default: `.../rl_casino_grpo/masks/`) |

---

## Deliverable 1 — Dense GRPO

```bash
# After "Shared environment" above
export GRPO_MODE=dense
export GRPO_TARGET_STEPS="${GRPO_TARGET_STEPS:-5000}"
# Optional: stable folder for resume — set on first job and reuse
# export GRPO_RUN_SLUG="llama31_math220k_grpo_dense_v1"

sbatch scripts/grpo_openr1_llama31_slurm.sh
```

Resume (same slug/model/dataset):

```bash
export GRPO_RESUME=auto
# export GRPO_RUN_SLUG="same_as_first_job"
sbatch scripts/grpo_openr1_llama31_slurm.sh
```

---

## Deliverable 2 — Random 97.5% mask + sparse GRPO

**Step A — generate random mask (GPU; one job or interactive)**

`MODEL_SLUG` is for filenames only (no `/`):

```bash
export MODEL_SLUG="${MODEL_SLUG:-llama31_8b_instruct}"
export RANDOM_SEED="${RANDOM_SEED:-42}"
export MASK_RANDOM="${GRPO_MASK_DIR}/grpo_random_sp975_${MODEL_SLUG}_math220k_seed${RANDOM_SEED}.pt"

python src/utils/generate_random_mask.py \
  --model_name "${MODEL}" \
  --sparsity_percent 97.5 \
  --seed "${RANDOM_SEED}" \
  --output_file "${MASK_RANDOM}"

python src/utils/generate_random_mask.py \
  --model_name "${MODEL}" \
  --sparsity_percent 97.5 \
  --output_file "${MASK_RANDOM}"
```

Do **not** pass `--mlp_only` (default: full linear coverage where the script targets weights).

**Step B — sparse GRPO**

```bash
export GRPO_MODE=sparse
export GRPO_MASK="${MASK_RANDOM}"
export GRPO_RUN_NAME="${GRPO_RUN_NAME:-grpo_sparse_random_sp975_${MODEL_SLUG}_v1}"
# Omit GRPO_MLP_ONLY — sparse_grpo_bsr defaults to full model where masks exist (do not pass --mlp_only)

sbatch scripts/grpo_openr1_llama31_slurm.sh
```

---

## Deliverable 3 — CAV (GRPO) 97.5% mask + sparse GRPO (v1)

CAV calibration uses **prompt + solution** vs **prompt-only** negatives (see plan: no DPO-style rejections). MLP-only mask broadcast in this repo; attention stays dense unless you extend the method later.

**Step A — CAV mask (GPU)**

```bash
export MODEL_SLUG="${MODEL_SLUG:-llama31_8b_instruct}"
export MASK_CAV="${GRPO_MASK_DIR}/grpo_cav_sp975_${MODEL_SLUG}_math220k.pt"

python src/cold_start/inference_mask_finder.py \
  --model_name "${MODEL}" \
  --mode grpo \
  --method cav \
  --dataset_name "open-r1/OpenR1-Math-220k" \
  --n_samples 64 \
  --sparsity 97.5 \
  --max_length "${GRPO_MAX_PROMPT_LENGTH:-512}" \
  --batch_size 4 \
  --output "${MASK_CAV}"
```

**Step B — sparse GRPO**

```bash
export GRPO_MODE=sparse
export GRPO_MASK="${MASK_CAV}"
export GRPO_RUN_NAME="${GRPO_RUN_NAME:-grpo_sparse_cav_sp975_${MODEL_SLUG}_v1}"

sbatch scripts/grpo_openr1_llama31_slurm.sh
```

---

## Appendix — deferred (SNIP / CAV+SNIP fusion)

Not required for v1. The main DPO pipeline and `gen_grpo_masks_and_plot.sh` produce **separate** SNIP and CAV masks and **compare** them (metrics/plots), not a single fused training mask. If you add a fusion script later, save the result under `GRPO_MASK_DIR` and point `GRPO_MASK` at it.

See the project plan document *GRPO HPC launch deliverables* for paper notes, MLP coverage caveats, and optional fusion rules.
