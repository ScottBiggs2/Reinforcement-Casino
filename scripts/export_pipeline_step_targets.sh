#!/usr/bin/env bash
# Pin DPO and GRPO trainer step targets so ad-hoc shell state or old defaults (e.g. 5000) cannot
# leak into sbatch --export. Source from repo root, then submit.
#
#   source scripts/export_pipeline_step_targets.sh
#   sbatch --export=ALL,HF_TOKEN="$HF_TOKEN" scripts/orchestrate_masks_then_queue_dpo_grpo.slurm
#
# ORCH_GRPO_TRAIN_STEPS must match GRPO_TARGET_STEPS when using ORCH_GRPO_THREEWAY_FOCUS=1, because
# the orchestrator overwrites GRPO_TARGET_STEPS from ORCH_GRPO_TRAIN_STEPS in that mode.

export NUM_STEPS_DPO=500
export GRPO_TARGET_STEPS=1000
export ORCH_GRPO_TRAIN_STEPS=1000

# Sparse training (orchestrator + pipeline stage 4): default A100 — override if your site uses different GRES syntax.
export ORCH_TRAIN_GRES_SPARSE="${ORCH_TRAIN_GRES_SPARSE:-gpu:a100:1}"
export SPARSE_SLURM_GRES="${SPARSE_SLURM_GRES:-gpu:a100:1}"
