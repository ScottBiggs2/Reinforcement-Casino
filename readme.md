# RL Acceleration Project (RL Casino)

Reinforcement Learning training infrastructure for **dense → masked → sparse** experiments, with Triton-accelerated sparse optimization paths for **DPO** and **GRPO**.

## What this repo is for

We study **task-oriented subnetworks** in LLMs: binary masks over weights, built without full-scale iterative pruning. The loop implemented here is:

**dense training → mask construction (warm/cold/random + complements) → sparse training → evaluation + interpretation**

Key emphasis: subnetworks that are **strong** at high sparsity, **transferable**, **interpretable** (mask overlap + layer metrics + CKA), and **optimizable** (update only the active set where appropriate).

If you’re joining midstream, read `docs/COLLABORATOR_GRPO_AND_AUTOMATION.md` and `scripts/README.md` first.

## Start here (common workflows)

### 1) End-to-end DPO pipeline (recommended default)

- **Single job (≤8h wall)**: `scripts/run_full_pipeline.sh`
- **Chained jobs (afterok; recommended for longer runs)**: `scripts/submit_pipeline_chain.sh`

This pipeline runs dense DPO, generates warm/cold/random masks (and optional inverse masks), launches sparse jobs per mask, runs comparison/interpretation exports, then fans out evaluation jobs.

Operational details (stage breakdown, resume helper, output layout, auto-resume wrapper): see `scripts/README.md`.

### 2) Production GRPO path (Open-R1 Math, Llama 3.1 8B)

- **Slurm launcher**: `scripts/grpo_openr1_llama31_slurm.sh`
- **Runbook**: `docs/GRPO_OPEN_R1_RUNBOOK.md`
- **Canonical hyperparameters**: `docs/hyperparams/open_r1_llama31.yaml`
- **HPC copy/paste blocks**: `docs/GRPO_HPC_COPYPASTE.md`

Important GRPO notes:

- **Reward profiles**: math rewards live in `src/utils/grpo_rewards.py`. The runbook documents `GRPO_REWARD_PROFILE` (`llama_cot` vs `openr1_tags`) and evaluation parity gotchas.
- **vLLM import guard**: TRL may import vLLM opportunistically and fail if your vLLM build mismatches PyTorch. This repo defaults to skipping vLLM import unless `TRL_SKIP_VLLM_IMPORT=0`. Details: `docs/TROUBLESHOOTING_GRPO.md`.

### 3) Mask interpretation + plots (Jaccard / CKA / CSV / figures)

Mask comparison tooling (structural + functional + per-layer metrics + plotting) is documented in:

- `src/cold_start/MASK_COMPARISON_GUIDE.md`

Common entrypoints:

- **Orchestrated GRPO vs DPO mask suite + plots**: `scripts/gen_grpo_masks_and_plot.sh`
- **Pairwise Jaccard**: `src/cold_start/mask_to_jaccard.py`
- **Pairwise CKA**: `src/cold_start/mask_to_cka.py`
- **Many masks, all pairs**: `src/cold_start/mask_interpretation_suite.py`

## Prerequisites

- **Python 3.11**
- **CUDA GPU** (H100/H200 intended for throughput; other GPUs may work with reduced settings)
- **Hugging Face** access (set `HF_TOKEN` for gated models like Llama 3.1)
- **Weights & Biases** optional (recommended for tracking; can disable per script)

## Environments (HPC)

The repo generally assumes two conda envs:

- **Training**: `rl_casino`
- **Evaluation**: `rl_casino_eval`

Explorer-style example (clone eval env to scratch):

```bash
conda create -n rl_casino_eval python=3.11.14 -y
conda activate rl_casino_eval

conda create --prefix /scratch/biggs.s/conda_envs/rl_casino_eval --clone /home/biggs.s/miniconda/envs/rl_casino_eval
conda activate /scratch/biggs.s/conda_envs/rl_casino_eval
```

Authentication:

```bash
wandb login [YOUR_KEY_HERE]
hf auth login [YOUR_KEY_HERE]
```

## “Repo root” requirement (Slurm)

Most Slurm scripts rely on `SLURM_SUBMIT_DIR` to find the repo. **Always run `sbatch` from the repository root** (the folder containing `src/` and `scripts/`). This is a common failure mode; see `scripts/README.md`.

## Project structure

```text
src/
  cold_start/        Cold-start mask methods + interpretation tools
  warm_start/        Warm-start mask finder over DPO deltas
  full_training/     Dense + sparse training entrypoints (DPO, GRPO)
  evaluation/        Benchmark harnesses (lm-eval, coding eval, etc.)
  utils/             Shared helpers (data, masks, checkpoint utils, etc.)
scripts/             Slurm + orchestration utilities (pipeline, eval, GRPO launch)
docs/                Runbooks + troubleshooting + hyperparameter canon
```

## Training entrypoints (overview)

### DPO

- **Dense DPO**: `src/full_training/DPO_train.py` (supports multiple dataset keys; writes optional delta logs for warm masks)
- **Sparse DPO (efficiency)**: `src/full_training/sparse_dpo_efficiency.py` (SGD/AdamW/SparseAdamW ablations)
- **Sparse DPO (BSR backprop; experimental)**: `src/full_training/sparse_dpo_bsr.py`

More details and dataset keys: `src/full_training/README.md`.

### GRPO

- **Dense GRPO (canonical)**: `src/full_training/GRPO_train.py`
- **Sparse GRPO (BSR path; experimental)**: `src/full_training/sparse_grpo_bsr.py`

Legacy / timing-only scripts with heuristic rewards exist (see `docs/COLLABORATOR_GRPO_AND_AUTOMATION.md` for the “what counts as comparable” split).

## Mask finding (overview)

- **Warm-start (DPO deltas → scores → masks)**: `src/warm_start/even_better_mask_finder.py`
- **Cold-start (inference-time scoring; DPO or GRPO calibration)**: `src/cold_start/inference_mask_finder.py`

Mask pooling defaults are **hybrid global with a per-tensor keep floor** (`min_layer_keep_ratio=0.0025`). For pure global pooling, use `--min_layer_keep_ratio 0.0`. For per-tensor top-k, use `--local_pool`. (See the mask scripts and pipeline defaults in `scripts/README.md`.)

## Evaluation

Evaluation harness overview and install guidance live in `src/evaluation/README.md`.

Common entrypoint:

```bash
python src/evaluation/run_all_benchmarks.py \
  --model_path "meta-llama/Llama-3.1-8B-Instruct" \
  --benchmarks mmlu,math,gsm8k \
  --batch_size auto
```

Cluster eval launcher:

```bash
sbatch scripts/run_evals_slurm.sh --model_path "meta-llama/Llama-3.1-8B-Instruct"
```

## Datasets (notes)

Short dataset pointers live in `src/data/datasets.md`:

- Tulu3 DPO mixture (instruction following)
- Math-Step-DPO-10K (math)
- Light-R1-DPOData (math/reasoning; default in several scripts)
- CodePref (coding)

## Troubleshooting (entry points)

- **GRPO vLLM import / bf16 setup / reward term stuck at zero**: `docs/TROUBLESHOOTING_GRPO.md`
- **GRPO launch + resume + checkpoint retention semantics**: `docs/GRPO_OPEN_R1_RUNBOOK.md`
- **Pipeline orchestration, resume-from-stage, auto-resume wrapper**: `scripts/README.md`

## Documentation index (high-signal)

- `docs/COLLABORATOR_GRPO_AND_AUTOMATION.md`
- `docs/GRPO_OPEN_R1_RUNBOOK.md`
- `docs/GRPO_HPC_COPYPASTE.md`
- `docs/TROUBLESHOOTING_GRPO.md`
- `src/full_training/README.md`
- `src/evaluation/README.md`
- `scripts/README.md`
- `src/cold_start/MASK_COMPARISON_GUIDE.md`
