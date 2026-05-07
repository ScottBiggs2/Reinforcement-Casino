# Run log — Warm-magnitude (step 200) sparse DPO on Tulu3 + Light-R1

**Date:** 2026-05-01
**Branch:** `irene-sparse-speed-ablation`
**Git SHA at launch:** `af0590e`
**Operator:** Irene (via Claude)

## Submitted job chain

All jobs queued via `sbatch --dependency=afterok:...`. No babysitting required.

| Stage | Job ID  | Script | Depends on |
|-------|---------|--------|------------|
| Light-R1 delta synth         | 6501264 | `scripts/synth_lightr1_deltas.sbatch`                  | (none) |
| Tulu3 dense re-run + dlog (v1, FAILED) | 6501265 | `scripts/transfer_p1_dense_dpo_tulu3_8b_deltalog.sbatch` | (none) |
| Tulu3 dense re-run + dlog (v2, CANCELLED — 4-GPU stuck in queue 8h+) | 6512576 | (same script, +`--gradient_checkpointing` +`--gres=gpu:h200:4`) | (none) |
| Tulu3 dense re-run + dlog (v3, 1-GPU)  | 6515782 | (same script, `--gres=gpu:h200:1` +`--gradient_accumulation_steps 64` for eff_batch 128) | (none) |
| Light-R1 warm-mag mask gen   | 6501268 | `scripts/warm_magnitude_mask_step200.sbatch`           | afterok:6501264 |
| Tulu3 warm-mag mask gen (v1, CANCELLED) | 6501269 | `scripts/warm_magnitude_mask_step200.sbatch` | afterok:6501265 |
| Tulu3 warm-mag mask gen (v2, CANCELLED) | 6512577 | (same script)                                | afterok:6512576 |
| Tulu3 warm-mag mask gen (v3)            | 6515783 | (same script)                                | afterok:6515782 |
| Light-R1 sparse DPO (Phase 3) | 6501270 | `scripts/transfer_p3_sparse_dpo_lightr1.sbatch`        | afterok:6501268 |
| Tulu3 sparse DPO (Phase 3, v1, CANCELLED) | 6501271 | `scripts/transfer_p3_sparse_dpo_tulu3.sbatch` | afterok:6501269 |
| Tulu3 sparse DPO (Phase 3, v2, CANCELLED) | 6512578 | (same script)                                 | afterok:6512577 |
| Tulu3 sparse DPO (Phase 3, v3)            | 6515784 | (same script)                                 | afterok:6515783 |

## What this is

Train sparse DPO on `meta-llama/Llama-3.1-8B-Instruct` for two datasets
(Tulu3, Light-R1) using **warm-start magnitude masks at target step 200**,
matching Scott's `cav_fixes` pipeline_common.sh defaults
(`TARGET_STEP_DPO=200`, `DELTA_LOG_INTERVAL=50`, `DELTA_LOG_END_STEP=200`).

Mask method: `src/warm_start/even_better_mask_finder.py --method magnitude`
i.e. score = `Σ_{t logged, t≤200} |W_t − W_0|`, then global top-K with a
small per-layer keep floor.

## Pre-launch state (cluster)

Existing dense DPO runs only logged step 50 (DPO_train default cap):

```
/scratch/xie.yiyi/transfer_v1/dense_dpo_lightr1_llama8b/deltas/.../
  base_state.pt
  deltas_step_50.pt   # only this — Scott spec wants 50/100/150/200

/scratch/xie.yiyi/transfer_v1/dense_dpo_tulu3_llama8b/deltas/.../
  base_state.pt
  deltas_step_50.pt   # same gap
```

HF checkpoints kept on disk (snapshot 2026-05-01):

| Dataset   | Checkpoints kept                                   | Step-100 ckpt? |
|-----------|----------------------------------------------------|----------------|
| Light-R1  | every 25 steps from 25 to 500                      | ✅ yes         |
| Tulu3     | every 50 steps from 150 to 500 (`save_total_limit=8`) | ❌ rolled out  |

Decision per Irene: synthesize Light-R1 missing deltas from saved
checkpoints; re-run dense DPO on Tulu3 with proper delta logging.

## Step 1 — Light-R1 deltas (synth from checkpoints)

**Script:** `scripts/synth_lightr1_deltas.sbatch` → `scripts/synth_deltas_from_ckpts.py`
**Job id:** _to fill in after submit_

What it does: loads `base_state.pt` (W_0, fp32 cpu), then for each step
∈ {100, 150, 200} loads HF `checkpoint-<N>` in bf16, casts each
`named_parameters()` tensor to fp32+cpu, subtracts base_state, saves to
`deltas_step_<N>.pt`. Mirrors `FlexibleCheckpointCallback.on_step_end` in
`src/full_training/DPO_train.py`.

After this job lands, Light-R1 delta dir will hold
`deltas_step_{50,100,150,200}.pt` plus `base_state.pt` — full Scott schedule.

### Light-R1 dense run that produced these ckpts (for provenance)

From `scripts/transfer_p1_dense_dpo_lightr1_8b.sbatch`:

| Param                       | Value |
|-----------------------------|-------|
| model                       | meta-llama/Llama-3.1-8B-Instruct |
| dataset                     | light-r1 |
| num_steps                   | 500 |
| learning_rate               | 5e-7 |
| warmup_ratio                | 0.1 (warmup ends at step 50) |
| dpo_beta                    | 0.1 |
| per_device_train_batch_size | 2 |
| gradient_accumulation_steps | 16 |
| num_gpus                    | 4 |
| effective_batch             | 128 |
| max_length                  | 1024 |
| max_prompt_length           | 1024 |
| save_steps                  | 25 |
| save_total_limit            | 30 |
| optimizer (HF Trainer)      | adamw_8bit (default `--optim`) |
| seed                        | HF Trainer default = 42 |

## Step 2 — Tulu3 dense re-run with delta logging

**Script:** `scripts/transfer_p1_dense_dpo_tulu3_8b_deltalog.sbatch`
**Job id:** _to fill in after submit_

Identical hparams to original `scripts/transfer_p1_dense_dpo_tulu3_8b.sbatch`,
plus `--delta_log_interval 50 --delta_log_end_step 200`. New output dir
`/scratch/xie.yiyi/transfer_v1/dense_dpo_tulu3_llama8b_deltalog/` so the
existing run's checkpoints aren't clobbered.

| Param                       | Value |
|-----------------------------|-------|
| model                       | meta-llama/Llama-3.1-8B-Instruct |
| dataset                     | tulu3 (allenai/llama-3.1-tulu-3-8b-preference-mixture) |
| num_steps                   | 500 |
| learning_rate               | 5e-7 |
| warmup_ratio                | 0.1 |
| dpo_beta                    | 0.1 |
| per_device_train_batch_size | 2 |
| gradient_accumulation_steps | 16 |
| num_gpus                    | 4 |
| effective_batch             | 128 |
| max_length                  | 1024 |
| max_prompt_length           | 1024 |
| save_steps                  | 50 |
| save_total_limit            | 8 |
| delta_log_interval          | 50 |
| delta_log_end_step          | 200 |
| optimizer                   | adamw_8bit |
| seed                        | 42 |
| walltime                    | 08:00:00 (school cap) + `--requeue` |

Why num_steps still 500 (not 200): the LR schedule is determined by
`warmup_ratio × num_steps`. Cutting to 200 would compress warmup to step 20,
producing a different trajectory than the original 500-step run. Keeping
num_steps=500 means W_50/W_100/W_150/W_200 in the new run match what the
original would have logged. Compute waste from steps 200→500 is intentional
fidelity tax.

## Step 3 — Warm magnitude mask generation (both datasets)

**Script:** `scripts/warm_magnitude_mask_step200.sbatch`
**Job ids:** _to fill in after submit_ (one per dataset)

| Param                  | Value |
|------------------------|-------|
| method                 | magnitude (`Σ_{t≤200} \|W_t − W_0\|`) |
| target_step            | 200 |
| sparsity_percent       | 97.5 |
| min_layer_keep_ratio   | 0.0025 (matches Phase-2 oracle masks) |
| local_pool             | false (global pooling, default) |
| mlp_only               | false (full model, default) |
| score_device           | cuda |

Outputs:
```
/scratch/xie.yiyi/transfer_v1/warm_masks_llama8b/
  warm_magnitude_dpo_lightr1_step200_sp97.5.pt
  warm_magnitude_dpo_tulu3_step200_sp97.5.pt
```

For Light-R1, magnitude aggregation will sum 4 files (steps 50/100/150/200).
For Tulu3, also 4 files (50/100/150/200) once the re-run lands. Symmetric
across datasets.

## Step 4 — Sparse DPO with the new masks

Driver scripts (already in repo, unchanged):
- Light-R1 → `scripts/transfer_p3_sparse_dpo_lightr1.sbatch`
- Tulu3    → `scripts/transfer_p3_sparse_dpo_tulu3.sbatch`

Submitted as:
```bash
MASK_PATH=/scratch/xie.yiyi/transfer_v1/warm_masks_llama8b/warm_magnitude_dpo_lightr1_step200_sp97.5.pt \
RUN_TAG=warm_magnitude_step200 \
sbatch scripts/transfer_p3_sparse_dpo_lightr1.sbatch

MASK_PATH=/scratch/xie.yiyi/transfer_v1/warm_masks_llama8b/warm_magnitude_dpo_tulu3_step200_sp97.5.pt \
RUN_TAG=warm_magnitude_step200 \
sbatch scripts/transfer_p3_sparse_dpo_tulu3.sbatch
```

| Param                       | Value |
|-----------------------------|-------|
| model                       | meta-llama/Llama-3.1-8B-Instruct |
| dataset                     | light-r1 / tulu3 |
| num_steps (n_steps)         | 500 |
| learning_rate (lr)          | 5e-7 |
| warmup_ratio                | 0.1 |
| weight_decay                | 0.0 |
| dpo_beta                    | 0.1 |
| per_device_train_batch_size | 2 |
| grad_accum                  | 64 |
| num_gpus                    | 1 (h200) |
| effective_batch             | 128 |
| max_length                  | 1024 |
| max_prompt_length           | 1024 |
| optimizer                   | sparse_adamw (Triton, mask-aware — does NOT decay frozen weights) |
| gradient_checkpointing      | true |
| save_steps                  | 50 |
| walltime                    | 08:00:00 + `--requeue --resume_from_checkpoint auto` |

Output dirs:
```
/scratch/xie.yiyi/transfer_v1/sparse_dpo_lightr1_warm_magnitude_step200/
/scratch/xie.yiyi/transfer_v1/sparse_dpo_tulu3_warm_magnitude_step200/
```

Wandb project: `rl_casino_transfer_v1`
Run names: `sparse_dpo_lightr1_warm_magnitude_step200_500steps`,
`sparse_dpo_tulu3_warm_magnitude_step200_500steps`.

## Log locations (post-launch reference)

**1. SLURM stdout/stderr** — `~/rc-sparse-speed/logs/` on cluster:
```
logs/synth_lightr1_deltas_6501264.{out,err}
logs/transfer_p1_dpo_tulu3_dlog_6501265.{out,err}
logs/warm_mag_mask_step200_6501268.{out,err}    # Light-R1
logs/warm_mag_mask_step200_6501269.{out,err}    # Tulu3
logs/transfer_p3_sparse_dpo_6501270.{out,err}   # Light-R1 sparse
logs/transfer_p3_sparse_dpo_tulu3_6501271.{out,err}  # Tulu3 sparse
```

**2. Wandb** — project `rl_casino_transfer_v1`, run names:
- `dense_dpo_llama8b_tulu3_scott_deltalog` (Tulu3 dense re-run, 6501265)
- `sparse_dpo_lightr1_warm_magnitude_step200_500steps` (6501270)
- `sparse_dpo_tulu3_warm_magnitude_step200_500steps` (6501271)

(synth + mask-gen jobs don't use wandb; their stdout is in the SLURM `.out` files.)

**3. Mask + delta artifacts on cluster:**
```
/scratch/xie.yiyi/transfer_v1/dense_dpo_lightr1_llama8b/deltas/.../
  base_state.pt, deltas_step_{50,100,150,200}.pt          # Light-R1 (50 original, 100/150/200 synth)

/scratch/xie.yiyi/transfer_v1/dense_dpo_tulu3_llama8b_deltalog/deltas/.../
  base_state.pt, deltas_step_{50,100,150,200}.pt          # Tulu3 (all from re-run)

/scratch/xie.yiyi/transfer_v1/warm_masks_llama8b/
  warm_magnitude_dpo_lightr1_step200_sp97.5.pt
  warm_magnitude_dpo_tulu3_step200_sp97.5.pt
```

## Post-mortem: Tulu3 dense v1 (6501265) crashed at step 0

**Symptom:** OOM after 5 min on node d1029 — "GPU 2 has a total capacity of 79.25 GiB of which 1.51 GiB is free. Including non-PyTorch memory, this process has 77.73 GiB memory in use." Cascading cancel of 6501269 + 6501271.

**Root cause:** Two issues combined:
1. d1029 is `gpu:a100:4`, not H200. SLURM landed there because the sbatch used bare `--gres=gpu:4` with no node-type constraint.
2. Sbatch was missing `--gradient_checkpointing`. Scott's `pipeline_common.sh` defaults dense DPO to `DPO_GRADIENT_CHECKPOINTING=1`. The original Tulu3 dense (6366611) ALSO missed this flag but happened to land on H200 (d4055) and just barely fit (~75 GB peak); the A100 fragmentation in our re-run pushed past 80 GB.

**v2 fix:** sbatch updated with `--gres=gpu:h200:4` AND `--gradient_checkpointing`. Resubmitted as 6512576 → 6512577 → 6512578.

Memory note added: `feedback_must_use_h200.md` — all training sbatches must constrain to H200 going forward.

## Deviations from Scott's exact spec

- **Light-R1 deltas at steps 100/150/200 are synthesized from saved
  checkpoints + base_state**, not produced live during training. The
  numerical content is identical (same fp32 subtraction, same params via
  `named_parameters()` order) but provenance is post-hoc.
- **Tulu3 step-50 delta** comes from the original (already-completed) run;
  steps 100/150/200 will come from the new `_deltalog` re-run. Both sources
  use the same seed (42) and same hparams, so the trajectory is identical
  modulo nondeterministic kernel ops; in practice expect tiny numerical
  drift (<<1e-3 relative).

If you want bit-for-bit consistency on Tulu3, take ALL four deltas
(50/100/150/200) from the new `_deltalog` run instead of mixing in the
original step-50 file. Easy fix: point `DELTA_LOG_DIR` of the magnitude
mask job at `dense_dpo_tulu3_llama8b_deltalog/deltas/.../` (it will already
contain step 50 from the new run). I'll wire this by default — see Step 3.
