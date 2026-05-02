# GRPO HPC — copy-paste launch blocks

Use these blocks on the cluster **from the repo root** (so `PYTHONPATH` and `scripts/` paths resolve). Paths follow [`scripts/pipeline_common.sh`](../scripts/pipeline_common.sh): fixed netid scratch (`SCRATCH_USER_ROOT`), GRPO artifacts **separate** from DPO (`rl_casino_grpo/...`).

**Replace** `YOUR_REPO_ROOT` if you `cd` elsewhere. **Set** `HF_TOKEN` for gated models (e.g. Llama).

---

## One-command launcher (masks + queue DPO+GRPO)

If you want a **single Slurm job** that uses **one short generic GPU allocation** to compute the mask `.pt` files (both DPO + GRPO) and then **queues all downstream DPO + GRPO training jobs** with Slurm `afterok` dependencies, submit:

```bash
cd /path/to/rl_casino
mkdir -p logs

# Required for gated models (e.g. Llama):
export HF_TOKEN="hf_xxxxxxxx"

sbatch scripts/orchestrate_masks_then_queue_dpo_grpo.slurm
```

Key overrides (optional; export before `sbatch`):

- **GRPO run ids**: `GRPO_DENSE_RUN_SLUG`, `GRPO_SPARSE_RANDOM_RUN_NAME`, `GRPO_SPARSE_CAV_RUN_NAME`, `GRPO_SPARSE_SNIP_RUN_NAME`
- **GRPO mask config**: `GRPO_DATASET_HF` (default `open-r1/OpenR1-Math-220k`), `GRPO_SNIP_OBJECTIVE` (default `lm`)
- **GRPO training config**: `GRPO_DATASET` (default `math-220k`), `GRPO_TARGET_STEPS`, `GRPO_MAX_PROMPT_LENGTH`, `GRPO_MAX_COMPLETION_LENGTH`, `GRPO_REWARD_PROFILE`

The manual blocks below remain useful if you want to run only GRPO, or if you want to generate just one mask type and launch a single sparse run.

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

# Sequence caps — align dense and sparse (canonical: 512 / 1024 per open_r1_llama31.yaml)
export GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-1024}"
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
```

Do **not** pass `--mlp_only` (default: full linear coverage where the script targets weights). Large models use the chunked global mask selector; allow enough GPU RAM and wall time.

**Step B — sparse GRPO**

```bash
export GRPO_MODE=sparse
export GRPO_MASK="${MASK_RANDOM}"
export GRPO_RUN_NAME="${GRPO_RUN_NAME:-grpo_sparse_random_sp975_${MODEL_SLUG}_v1}"
# Omit --mlp_only on the trainer (default: full model where mask tensors exist)

sbatch scripts/grpo_openr1_llama31_slurm.sh
```

---

## Deliverable 3 — CAV (GRPO) 97.5% mask + sparse GRPO (v1)

CAV calibration uses **prompt + solution** vs **prompt-only** negatives. In this repo, CAV masks are **MLP-focused** (`scores_to_masks` broadcasts from down-proj neuron scores); attention stays dense unless you extend the method.

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
  --seed 42 \
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

## Deliverable 4 — SNIP (GRPO) 97.5% mask + sparse GRPO

SNIP ([Lee et al., 2019](https://arxiv.org/abs/1810.02340)) uses connection saliency \(|w \odot \nabla\mathcal{L}|\); this repo implements \(|w| \cdot |\nabla|\) on 2D weight matrices in [`SNIPScorer.score`](src/cold_start/utils/snip_scorer.py) and [`compute_snip_scores`](src/cold_start/utils/snip_scorer.py). Orchestration, masking, and metadata live in [`inference_mask_finder.py`](src/cold_start/inference_mask_finder.py): it loads the model, loads GRPO calibration via [`load_calibration_samples_grpo`](src/cold_start/inference_mask_finder.py) (prompt+solution vs prompt-only), then dispatches SNIP.

**Saved format:** `save_masks` writes `torch.bool` masks under `{"masks": {...}, "metadata": {...}}` (see [`src/utils/mask_utils.py`](src/utils/mask_utils.py)).

### Objectives (`--snip-objective`)

| Value | Meaning | GRPO calibration |
|--------|---------|-------------------|
| `lm` (default) | One backward on **mean causal LM loss** over sequences in `chosen_texts`. | `chosen_texts` = prompt+solution; **prompt-only rows are not used** for the loss (same as standard SNIP on “chosen” completion text). |
| `dpo_preference` | Gradients of a **pairwise preference** loss (`dpo_style_preference_loss` in `snip_scorer.py`) with separate forward passes on chosen vs rejected sequences. | Pairs **prompt+solution** vs **prompt-only** (aligned lists from the same GRPO loader). Uses [`build_preference_snip_dataloader`](src/cold_start/inference_mask_finder.py) for GRPO. Gradients accumulate over up to `--snip-num-batches` batches. |

**Coverage:** For `--method snip`, the inference entrypoint scores **all** 2D `nn.Linear` weights with non-null gradients (`mlp_only=False` in code paths). This is **broader** than CAV’s MLP-broadcast masks in Deliverable 3.

**Optional SNIP flags:** `--local-pool`, `--min-layer-keep-ratio` (default matches other cold masks), `--snip-preference-beta` (only for `dpo_preference`), `--snip-num-batches` (only for `dpo_preference`, default 32).

### Step A — SNIP mask (`lm` — recommended default)

```bash
export MODEL_SLUG="${MODEL_SLUG:-llama31_8b_instruct}"
export MASK_SNIP_LM="${GRPO_MASK_DIR}/grpo_snip_lm_sp975_${MODEL_SLUG}_math220k.pt"

python src/cold_start/inference_mask_finder.py \
  --model_name "${MODEL}" \
  --mode grpo \
  --method snip \
  --snip-objective lm \
  --dataset_name "open-r1/OpenR1-Math-220k" \
  --n_samples 64 \
  --sparsity 97.5 \
  --max_length "${GRPO_MAX_PROMPT_LENGTH:-512}" \
  --batch_size 4 \
  --seed 42 \
  --output "${MASK_SNIP_LM}"
```

### Step A — SNIP mask (`dpo_preference` — preference-gradient saliency)

```bash
export MODEL_SLUG="${MODEL_SLUG:-llama31_8b_instruct}"
export MASK_SNIP_PREF="${GRPO_MASK_DIR}/grpo_snip_dpo_preference_sp975_${MODEL_SLUG}_math220k.pt"

python src/cold_start/inference_mask_finder.py \
  --model_name "${MODEL}" \
  --mode grpo \
  --method snip \
  --snip-objective dpo_preference \
  --dataset_name "open-r1/OpenR1-Math-220k" \
  --n_samples 64 \
  --sparsity 97.5 \
  --max_length "${GRPO_MAX_PROMPT_LENGTH:-512}" \
  --batch_size 4 \
  --seed 42 \
  --snip-num-batches 32 \
  --snip-preference-beta 1.0 \
  --output "${MASK_SNIP_PREF}"
```

### Step B — sparse GRPO (pick one mask path)

```bash
export GRPO_MODE=sparse
# Example: LM SNIP mask
export GRPO_MASK="${MASK_SNIP_LM}"
export GRPO_RUN_NAME="${GRPO_RUN_NAME:-grpo_sparse_snip_lm_sp975_${MODEL_SLUG}_v1}"

sbatch scripts/grpo_openr1_llama31_slurm.sh
```

For a run using `MASK_SNIP_PREF`, set `GRPO_MASK` and e.g. `GRPO_RUN_NAME=grpo_sparse_snip_pref_sp975_${MODEL_SLUG}_v1`.

### Optional: batch SNIP + plots (GRPO vs DPO comparison)

[`scripts/gen_grpo_masks_and_plot.sh`](../scripts/gen_grpo_masks_and_plot.sh) runs SNIP (and CAV/Fisher) for both modes and compares masks. Set `MASK_DIR`, `MODEL`, `N_SAMPLES`, `SPARSITY`, and `COLD_SNIP_OBJECTIVE="${COLD_SNIP_OBJECTIVE:-lm}"` before submitting. Output filenames there are fixed (`snip_grpo.pt`, etc.); for the scratch layout in this doc, prefer explicit `--output` paths as above.

---

## Appendix — CAV+SNIP fusion (optional)

The main DPO pipeline and `gen_grpo_masks_and_plot.sh` produce **separate** SNIP and CAV masks and **compare** them (metrics/plots), not a single fused training mask. If you add a fusion script later, save the result under `GRPO_MASK_DIR` and point `GRPO_MASK` at it.

See the project plan document *GRPO HPC launch deliverables* for paper notes, MLP coverage caveats, and optional fusion rules.
