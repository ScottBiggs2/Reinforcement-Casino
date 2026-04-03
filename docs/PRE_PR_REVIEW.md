# Pre-PR review checklist (`integrate/scott-irene-2026`)

## Branch in IDE

You are on **`integrate/scott-irene-2026`**. Sync with:

`git fetch origin && git checkout integrate/scott-irene-2026 && git pull`

## 1. GRPO consistency / correctness

| Component | Role |
|-----------|------|
| [`src/utils/grpo_rewards.py`](../src/utils/grpo_rewards.py) | **Single source of truth** for math rewards (accuracy, format_number, format_reasoning). |
| [`src/full_training/GRPO_train.py`](../src/full_training/GRPO_train.py) | Dense GRPO; uses `GRPO_REWARD_FUNCS`; `optim=adamw_torch`. |
| [`src/full_training/sparse_grpo_bsr.py`](../src/full_training/sparse_grpo_bsr.py) | Sparse GRPO; uses `GRPO_REWARD_FUNCS`; custom optimizers + BSR. |
| [`src/full_training/GRPO_timing_baseline.py`](../src/full_training/GRPO_timing_baseline.py) | Timing baseline; uses **same** `GRPO_REWARD_FUNCS` (no duplicate reward code). |
| [`src/magic/sparse_GRPO_v2.py`](../src/magic/sparse_GRPO_v2.py) | **Legacy** different heuristics — not comparable to the above without aligning rewards (see readme “Implementation status”). |

**Config deltas to remember:** `GRPO_train` uses fixed `GRPOConfig` (e.g. `adamw_torch`, lr 5e-6); `sparse_grpo_bsr` / timing baseline pass explicit optimizers and CLI LRs. That is intentional (different entrypoints), but **rewards are aligned** via `grpo_rewards.py`.

## 2. Scratch paths (`/scratch/$USER/...`)

- **Python:** Defaults use [`src/utils/scratch_paths.py`](../src/utils/scratch_paths.py): `RL_CASINO_SCRATCH_ROOT` or `/scratch/$USER`, then `rl_casino_outputs`, `hf_cache/datasets`.
- **Bash pipelines:** [`scripts/pipeline_common.sh`](../scripts/pipeline_common.sh) and [`scripts/multigpu_pipeline/pipeline_common_multigpu.sh`](../scripts/multigpu_pipeline/pipeline_common_multigpu.sh) use:
  - `SCRATCH_USER_ROOT` default **`/scratch/${USER}`**
  - Derived: `TRAIN_OUT_BASE`, `MASK_OUT_BASE`, `HF_DATASETS_CACHE_ROOT`, conda env paths, etc.

**Legacy layout (e.g. `/scratch/biggs.s/...`):** before `sbatch`, run:

```bash
export SCRATCH_USER_ROOT=/scratch/biggs.s
# optional full overrides:
# export TRAIN_ENV=/scratch/biggs.s/conda_envs/rl_casino
```

**Irene / collaborator scripts** under `scripts/` (`benchmark_speedup.sh`, `run_dpo_masks_grpo.sh`, …) may still hardcode `/scratch/xie.yiyi/...` — adjust or export the same variables when running those files.

## 3. `mlp_only` — comparability (**must stay default False**)

- All sparse trainers expose `--mlp_only` as **`action="store_true"`**, default **False**.
- **No** `scripts/*.sh` in the main pipeline passes `--mlp_only` (grep clean).
- Do **not** enable MLP-only masks or training for cross-method paper comparisons without labeling the run as an ablation.

## 4. General code review (manual)

- [ ] Run one cluster smoke: `verify_grpo_training.sh` or a short `pipeline_stage_*` with `SCRATCH_USER_ROOT` set for your account.
- [ ] Confirm conda exists at `${TRAIN_ENV}` after exporting `SCRATCH_USER_ROOT` if you changed it.
- [ ] Skim RL-irene-only scripts you use for hardcoded paths.
