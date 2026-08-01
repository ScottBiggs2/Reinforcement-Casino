# Sparse Subnetwork Training for RL Post-Training

Training infrastructure for **dense → masked → sparse** experiments on LLM alignment, with
Triton-accelerated sparse optimization paths for **DPO** and **GRPO**.

## What this repo is for

We study **task-oriented subnetworks** in LLMs: binary masks over weights, built without full-scale
iterative pruning. The loop implemented here is:

**dense training → mask construction (warm / cold / random + complements) → sparse training → evaluation + interpretation**

The emphasis is on subnetworks that are **strong** at high sparsity, **transferable**,
**interpretable** (mask overlap + per-layer metrics + CKA), and **optimizable** (update only the
active set where appropriate).

## Prerequisites

- **Python 3.11**
- **CUDA GPU** — H100/H200 class intended for throughput; other GPUs work with reduced settings
- **Hugging Face access** — set `HF_TOKEN` for gated models (e.g. Llama 3.1)
- **Slurm** for the orchestrated pipeline (the Python entrypoints run standalone without it)
- **Weights & Biases** optional; every training script can run with tracking disabled

```bash
pip install -r requirements.txt        # training
pip install -r eval_requirements.txt   # evaluation harness
```

## Site configuration

The scripts are cluster-agnostic but need a few environment variables pointed at your site. Nothing
below has a working default outside a Slurm cluster with a `/scratch` filesystem.

| Variable | Purpose | Default |
|---|---|---|
| `RL_CASINO_SCRATCH_ROOT` | Root for outputs and HF caches | `/scratch/$USER` |
| `SCRATCH_USER_ROOT` | Same, for the shell pipeline | `/scratch/$USER` |
| `TRAIN_ENV` | Conda prefix for the training env | `/scratch/$USER/conda_envs/rl_casino` |
| `EVAL_ENV` | Conda prefix for the evaluation env | `/scratch/$USER/conda_envs/rl_casino_eval` |
| `CONDA_SH` | Path to your site's `profile.d/conda.sh` | *(must be set)* |
| `CPU_PARTITION` / `GPU_PARTITION` | Slurm partition names | `short` / `gpu` |
| `HF_TOKEN` | Hugging Face token for gated models | *(unset)* |

Path resolution is centralized in [`src/utils/scratch_paths.py`](src/utils/scratch_paths.py) — prefer
that helper over hardcoding.

Two conda environments are assumed so training and evaluation can run concurrently: a training env
and an evaluation env. Only the names differ; see `EVAL_ENV` above.

### Slurm: run `sbatch` from the repo root

Most Slurm scripts locate the repo via `SLURM_SUBMIT_DIR`. **Always submit from the repository root**
(the directory containing `src/` and `scripts/`). This is the most common failure mode.

## Start here

### 1) End-to-end DPO pipeline

- **Single job (≤8h wall)**: `scripts/run_full_pipeline.sh`
- **Chained jobs (`afterok`; recommended for longer runs)**: `scripts/submit_pipeline_chain.sh`

Runs dense DPO, generates warm/cold/random masks (and optional inverse masks), launches one sparse
job per mask, runs comparison/interpretation exports, then fans out evaluation jobs. Stage
breakdown, the resume helper, and output layout are documented in
[`scripts/README.md`](scripts/README.md).

Individual stages are `scripts/pipeline_stage_0*.sh`; resume mid-pipeline with
`scripts/resume_pipeline_from_stage.sh`.

### 2) GRPO path (Open-R1 Math, Llama 3.1 8B)

- **Slurm launcher**: `scripts/grpo_openr1_llama31_slurm.sh`
- **Runbook**: [`docs/GRPO_OPEN_R1_RUNBOOK.md`](docs/GRPO_OPEN_R1_RUNBOOK.md)
- **Canonical hyperparameters**: [`docs/hyperparams/open_r1_llama31.yaml`](docs/hyperparams/open_r1_llama31.yaml)

Two notes that cost the most debugging time:

- **Reward profiles** live in `src/utils/grpo_rewards.py`. The runbook documents `GRPO_REWARD_PROFILE`
  (`llama_cot` vs `openr1_tags`) and the evaluation-parity pitfalls.
- **vLLM import guard**: TRL imports vLLM opportunistically and fails if your vLLM build mismatches
  PyTorch. This repo skips that import unless `TRL_SKIP_VLLM_IMPORT=0`. See
  [`docs/TROUBLESHOOTING_GRPO.md`](docs/TROUBLESHOOTING_GRPO.md).

### 3) Mask interpretation and plots

Structural (Jaccard), functional (CKA), and per-layer metrics, plus plotting:
see [`src/cold_start/MASK_COMPARISON_GUIDE.md`](src/cold_start/MASK_COMPARISON_GUIDE.md).

- Pairwise Jaccard: `src/cold_start/mask_to_jaccard.py`
- Pairwise CKA: `src/cold_start/mask_to_cka.py`
- Many masks, all pairs: `src/cold_start/mask_interpretation_suite.py`

## Training entrypoints

**DPO**
- Dense: `src/full_training/DPO_train.py` — multiple dataset keys; writes optional delta logs for warm masks
- Sparse (efficiency): `src/full_training/sparse_dpo_efficiency.py` — SGD / AdamW / SparseAdamW ablations
- Sparse (BSR backprop, experimental): `src/full_training/sparse_dpo_bsr.py`

**GRPO**
- Dense: `src/full_training/GRPO_train.py`
- Sparse (BSR path, experimental): `src/full_training/sparse_grpo_bsr.py`

Dataset keys and further detail: [`src/full_training/README.md`](src/full_training/README.md).

To compare dense and sparse arms fairly, run **both** through `sparse_dpo_efficiency.py` (the dense
arm with `--optimizer adamw_torch`). The two entrypoints otherwise differ in collator truncation and
gradient-clipping regime; `scripts/verify_grpo_config_consistency.py` gates this.

## Mask construction

- **Warm-start** (DPO deltas → scores → masks): `src/warm_start/even_better_mask_finder.py`
- **Cold-start** (inference-time scoring; DPO or GRPO calibration): `src/cold_start/inference_mask_finder.py`
- **Random baselines**: `src/utils/generate_random_mask.py`
- **Checkpoint-diff oracle**: `src/warm_start/checkpoint_diff_mask_finder.py`

Cold-start scorers (SNIP, CAV probes, GRaSP, Wanda/OWL) live under `src/cold_start/utils/`.

Pooling defaults to **hybrid global with a per-tensor keep floor** (`min_layer_keep_ratio=0.0025`).
For pure global pooling pass `--min_layer_keep_ratio 0.0`; for per-tensor top-k pass `--local_pool`.

**Mask coverage matters.** Tensors with no mask entry receive a *dense* optimizer step — unmasked is
not frozen. Masks that cover fewer tensors therefore train *more* parameters, which silently
confounds capacity comparisons. `scripts/verify_masks_full_coverage.py` fails any mask that drops a
2-D tensor; run it before trusting a comparison.

## Evaluation

```bash
python src/evaluation/run_all_benchmarks.py \
  --model_path "meta-llama/Llama-3.1-8B-Instruct" \
  --benchmarks mmlu,math,gsm8k \
  --batch_size auto
```

Cluster launcher: `sbatch scripts/run_evals_slurm.sh --model_path <MODEL>`.
Harness overview and install guidance: [`src/evaluation/README.md`](src/evaluation/README.md).

## Datasets

Pointers in [`src/data/datasets.md`](src/data/datasets.md):

- Tulu3 DPO mixture (instruction following)
- Math-Step-DPO-10K (math)
- Light-R1-DPOData (math / reasoning; default in several scripts)
- CodePref (coding)

## Verification gates

Run before trusting any comparison:

| Script | Checks |
|---|---|
| `scripts/verify_masks_full_coverage.py` | No mask drops a 2-D weight tensor |
| `scripts/verify_grpo_config_consistency.py` | Dense and sparse arms share batch/clip/warmup footing |
| `scripts/verify_dpo_normalization_regression.py` | DPO text normalization is stable |
| `scripts/verify_mask_against_reference.py` | A mask matches a reference coverage set |
| `python -m pytest src/tests` | Kernel correctness, mask metrics, certifiability margins |

## Repository layout

```text
src/
  cold_start/     Cold-start mask methods + interpretation tools
  warm_start/     Warm-start mask finders over training deltas
  full_training/  Dense + sparse training entrypoints (DPO, GRPO)
  kernels/        Triton BSR backward + sparse Adam kernels
  optimizers/     SparseAdamW
  mlps/           BSR sparse MLP modules
  evaluation/     Benchmark harnesses
  analysis/       Certifiability margins, gradient energy, score-gap analysis
  utils/          Masks, data, checkpoints, logging, scratch paths
  tests/          Unit tests
scripts/          Slurm pipeline, benchmarks, verification gates, plotting
docs/             Runbooks, troubleshooting, hyperparameter canon
```

## License

MIT — see [LICENSE](LICENSE).
