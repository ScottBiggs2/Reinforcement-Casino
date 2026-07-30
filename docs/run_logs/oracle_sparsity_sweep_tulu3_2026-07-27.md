# Oracle mask sparsity sweep — Llama-3.1-8B-Instruct / Tulu3 / DPO

**Date:** 2026-07-27 · **Branch:** `irene-rebuttal-lora` · **Cluster:** Discovery (Explorer)

Sweep of the checkpoint-diff ("oracle") mask over ρ, holding everything else fixed.
ρ is the fraction of weights **zeroed**; keep rate is 1−ρ.

---

## 1. Why the source checkpoint had to change

The mask the paper reports at ρ=97.5 was built from
`transfer_v1/dense_dpo_tulu3_llama8b/checkpoints/.../checkpoint-500` (recorded in that
mask's own `metadata.final_model`). **That directory no longer exists.** The same is true
for the Light-R1 side (`dense_dpo_lightr1_llama8b`).

Deletion window, from job logs:

| When | Evidence |
|---|---|
| 2026-04-27 20:03–20:26 | job `6370756` read `checkpoint-500`/`checkpoint-150`, `Matched 291 parameters (out of 291)`, wrote both Light-R1 oracle masks (file mtimes 20:13 / 20:25 agree) |
| 2026-05-26 → 05-27 | `mask_seed_var` / `msv_mag_cpu` generated seed variants from `--delta_log_dir .../dense_dpo_lightr1_llama8b/deltas/...` — deltas still present |
| 2026-05-27 07:24 | job `7050891` loaded `.../checkpoint-500` three times and exited `[main] Done.` — dense checkpoint still present |
| 2026-07-25 | verified absent (noted in `preference_eval_existing_arms.sbatch` header) |

Cause not determined. Excluded: `wipe_pipeline_artifacts.sh` (only touches
`rl_casino_{train,masks,sparse_train,eval_runs}`, never `transfer_v1`); `~/.bash_history`
(runs through 2026-06-15, no `rm` of the path). `atime` is unusable — the VAST mount
reports `atime == mtime` on every directory, i.e. `noatime`.

Surviving oracle masks are **bool**, so they cannot be re-thresholded to another ρ. The
sweep therefore had to be regenerated from a different dense run.

**Source used:** `transfer_v1/dense_dpo_tulu3_llama8b_deltalog/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500`
(chosen over `dense_dpo_tulu3_long_llama8b` because its step-500 is the Scott-spec
convergence point; the `_long` run continues to step 2700 and is not the same quantity).

---

## 2. Mask generation hyperparameters

Identical across all four levels — ρ is the only knob.

| Parameter | Value |
|---|---|
| script | `src/warm_start/checkpoint_diff_mask_finder.py` |
| launcher | `scripts/transfer_p2_oracle_sweep_llama8b.sbatch` (array 0–3) |
| `--initial_model` | `meta-llama/Llama-3.1-8B-Instruct` |
| `--final_model` | `dense_dpo_tulu3_llama8b_deltalog/.../checkpoint-500` |
| `--min_layer_keep_ratio` | `0.0025` (held constant) |
| `--force_cpu` | yes |
| pooling | `global_with_layer_floor`, `local_pool=False` |
| `--mlp_only` | no (all 291 matched tensors scored) |
| seed / jitter | none (deterministic) |
| partition | `short`, **no GPU**, 16 CPU, 240 G |

No-GPU is deliberate: the scoring is CPU tensor work and the chunked selector already runs
on CPU at 8 B scale, so requesting no GPU avoids both the h200 queue and the RC 15-minute
idle-GPU auto-cancel.

---

## 3. Results

Total scored parameters: **8,030,261,248** (291 tensors).

| ρ | job | kept params | measured keep | error vs target | elapsed | file |
|---|---|---|---|---|---|---|
| 80 % | `8790708_0` | 1,606,052,249 | 20.0000 % | 0.000000 % | 2:57:40 | `oracle_dpo_tulu3_step500_sp80.pt` |
| 90 % | `8790708_1` | 803,026,124 | 10.0000 % | 0.000000 % | 1:31:29 | `oracle_dpo_tulu3_step500_sp90.pt` |
| 97.5 % | `8791301_2` | 200,756,531 | 2.5000 % | — | 0:24:28 | `oracle_dpo_tulu3_step500_sp97.5_src-deltalog.pt` |
| 99 % | `8790708_3` | 80,302,612 | 1.0000 % | 0.000000 % | 0:10:04 | `oracle_dpo_tulu3_step500_sp99.pt` |

All under `/scratch/$USER/transfer_v1/oracle_masks_llama8b/`, 7.5 G each.

Runtime scales with keep count (ρ=80 keeps 20× more than ρ=99 and took 18× longer),
consistent with the chunked top-k selector dominating.

The per-layer keep floor does **not** bind at these levels: at ρ=99 the floor could in
principle claim 25 % of the budget, but actual keep equals target exactly
(80,302,612), i.e. most layers' global top-k share already exceeds the floor.

---

## 4. Gate: is the new sweep comparable to the published ρ=97.5 point?

```
new   : oracle_dpo_tulu3_step500_sp97.5_src-deltalog.pt   (source: _deltalog run)
paper : oracle_dpo_tulu3_step500_sp97.5.pt                (source: deleted run)
Jaccard = 0.9763        chance level at ρ=97.5 ≈ 1.27e-2
```

**Gate passed.** Two independent dense DPO runs select 2.5 % subnetworks that overlap
97.63 %, ~77× above chance. The sweep can be read alongside the published ρ=97.5 number.

This is also a result in its own right: prior variance work (`mask_seed_variance`)
measured the same run under score jitter; this measures **run-to-run** reproducibility of
the oracle subnetwork, which is the stronger claim.

### Method note — a self-comparison bug was caught and fixed

The first ρ=97.5 attempt (`8790708_2`) wrote to the un-tagged filename, which already
existed from the paper run. It took the "already exists, skip" branch and then computed
Jaccard of that file **against itself**, reporting `1.0000`. That number is void.
Fix: `OUT_TAG` is now mandatory (outputs carry `_src-<tag>`) and the script exits non-zero
if `REF_MASK == OUT`. `8791301_2` is the valid run.

**Naming caveat:** the ρ=80/90/99 files were produced before the fix and still use the
un-tagged names. Their `metadata.final_model` correctly records the `_deltalog` source, so
the data is sound; only the directory listing is ambiguous. Rename pending.

---

## 5. Downstream (submitted, pending at time of writing)

Sparse DPO, `scripts/transfer_p3_sparse_dpo_tulu3.sbatch`, one 1-GPU job per ρ.
Model `meta-llama/Llama-3.1-8B-Instruct`, dataset `tulu3`, wandb project
`rl_casino_transfer_v1`, run names `sparse_dpo_tulu3_oracle_sp<ρ>_500steps`.

Hyperparameters identical across arms: 500 steps, effective batch 128
(per-device 2 × grad-accum 64 × 1 GPU), lr 5e-7, β 0.1, warmup-ratio 0.1,
max_length 1024, `--optimizer sparse_adamw`, save_steps 50.

| job | ρ | trainable params |
|---|---|---|
| `8794867` | 80 % | 1,606,052,249 |
| `8794868` | 90 % | 803,026,124 |
| `8794870` | 97.5 % | 200,756,531 |
| `8794871` | 99 % | 80,302,612 |

**Not yet submitted:** density-matched random controls at each ρ. Without them, an effect
at ρ=80 cannot be attributed to the mask rather than to parameter count. Held back pending
confirmation that the ρ=80 arm does not OOM — it is the only untested configuration
(its `torch.nonzero` index tensor is int64 × 1.6 B = 12.9 GB on GPU, vs 1.6 GB at ρ=97.5).

---

## 6. Scheduling constraint encountered

All five GPU jobs sat `PENDING (Priority)` for hours with **8 h200 GPUs idle**. The binding
resource is host memory, not GPUs:

| node | free h200 | Slurm-allocatable memory | request 128 G |
|---|---|---|---|
| d4055 | 4 | 44 G | no |
| d4052 | 3 | 72 G | no |
| d4053 | 1 | 80 G | no |
| d1026 / d1028 (a100) | 0 | 420 G / 452 G | GPUs full |

The 128 G request is justified, not padding: `FlexibleCheckpointCallback` holds a full
fp32 CPU copy of `base_state` (8.03 B × 4 B = 32 GB) for the whole run and builds a second
full copy when writing each delta. Measured peaks on prior equivalent runs: `6459076`
99.8 GB, `6459077` 101.5 GB — both at ReqMem 128 G. `sparse_dpo_efficiency.py` uses the
same callback (`:191–198`), so the sparse arms are not cheaper than the dense one.

Jobs were moved to `Partition=multigpu,gpu` in place via `scontrol update` (no cancel, job
IDs and queue age preserved). This changes the competing queue but not node memory, so it
is a cheap attempt rather than a fix.

**Open option (needs approval — requires cancelling pending jobs):** the sparse arms do not
consume their own deltas (the oracle comes from *dense* checkpoint diffs), so disabling
delta logging for them would cut peak RSS from ~100 GB to ~36 GB and let `--mem=64G` land
on d4052/d4053 immediately.

---

## 7. Lesson recorded

The original Light-R1 dense run passed no `--delta_log_end_step`, which defaults to
`min(num_steps, max(interval, num_steps//10))` = **50**. Only `deltas_step_50.pt` was
written, so when the checkpoints were deleted the step-500 |Δθ| scores went with them and
the surviving bool masks became un-rethresholdable.

The rebuild (`scripts/transfer_p1_dense_dpo_lightr1_8b_deltafull.sbatch` and its 1-GPU
variant) sets `--delta_log_interval 50 --delta_log_end_step 500`. Keeping `deltas/` +
`base_state.pt` — not the bool masks — is what makes any future ρ a CPU job on a 30 GB
file instead of a retrain.
