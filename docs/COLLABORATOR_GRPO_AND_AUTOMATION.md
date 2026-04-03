# Collaborator note: research context, GRPO, automation

Assumes you already know **GRPO**; this file situates the **research** and points to where code and jobs live. Prerequisites, conda, and commands are in the [readme](../readme.md).

---

## What we’re studying

We care about **RL-oriented subnetworks** in LLMs: binary masks over weights, found without full-scale iterative pruning. The working hypotheses we stress in this codebase are that such subnetworks can be **strong** at high sparsity, **transfer** across related tasks or setups, **interpretable** through overlap analyses, scoring methods, and layer metrics, and **optimizable**—training and inference can target the active weights while dense (or controlled sparse) forwards keep experiments grounded. The repo implements the loop: **dense training → masks (warm/cold/random) → sparse training → evals**, with Slurm orchestration for long runs.

---

## Where your work plugs in

- **GRPO implementation:** Main paths are [`src/full_training/GRPO_train.py`](../src/full_training/GRPO_train.py) (dense) and [`src/full_training/sparse_grpo_bsr.py`](../src/full_training/sparse_grpo_bsr.py) (sparse). [`src/magic/sparse_GRPO_v2.py`](../src/magic/sparse_GRPO_v2.py) is a legacy Triton variant with **different** heuristic rewards—see readme “Implementation status” for the split.
- **Experiment automation:** The default chained pipeline is **DPO-first**: [`scripts/submit_pipeline_chain.sh`](../scripts/submit_pipeline_chain.sh) → `pipeline_stage_01_dense.sh` … `pipeline_stage_05_evals.sh`, with shared env and paths in [`scripts/pipeline_common.sh`](../scripts/pipeline_common.sh). **GRPO is not a stage there yet**; [`scripts/verify_grpo_training.sh`](../scripts/verify_grpo_training.sh) is the packaged dense → mask → sparse GRPO smoke test on Slurm. Resume failed chains with [`scripts/resume_pipeline_from_stage.sh`](../scripts/resume_pipeline_from_stage.sh).
- **Index of scripts:** [scripts/README.md](../scripts/README.md).

---

## Operational gotchas

- Submit **`sbatch` from the repo root** so `SLURM_SUBMIT_DIR` resolves (see `pipeline_common.sh`).
- Paths like `/scratch/biggs.s/...` and conda prefixes in `pipeline_common.sh` are **site-specific**—replace for your cluster.
