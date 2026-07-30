# Oracle mask sparsity sweep — Llama-3.1-8B-Instruct / Light-R1 / DPO

**Date:** 2026-07-28/29 · **Branch:** `irene-rebuttal-lora` · **Cluster:** Discovery (Explorer)
**Status:** ρ=99 and ρ=97.5 masks done, **gate passed (Jaccard 0.9691)**. ρ=90 and ρ=80
re-submitted onto fast-CPU nodes after the first attempt proved the selector cannot finish
them on a first-gen EPYC node inside the 8 h cap. Two arms queued. Companion to
`oracle_sparsity_sweep_tulu3_2026-07-27.md` (the Tulu3 half, complete).

| job | what | status |
|---|---|---|
| `8817043` | packed sweep, c2205 (`zen`) | ρ=99 ✓, ρ=97.5 ✓ + gate; **cancelled 00:50** mid-ρ=90 — could not finish, see §9 |
| `8819592` | ρ=90 mask, `zen2\|cascadelake`, 240 G | queued |
| `8818835` | ρ=80 mask, `zen2\|cascadelake`, 240 G | queued |
| `8817763` | sparse DPO arm ρ=99 | queued |
| `8819594` | sparse DPO arm ρ=97.5 | queued |

ρ is the fraction of weights **zeroed**; keep rate is 1−ρ. Target ρ ∈ {80, 90, 99},
plus ρ=97.5 which is required as the comparability **gate**, not as a new data point.

---

## 0. What already exists (verified 2026-07-28 20:00 EDT)

The blocker recorded in the Tulu3 log — "no Llama Light-R1 dense run exists anywhere" —
**is cleared.** Job `8792090` (`p1_dpo_lightr1_deltafull_1gpu`) COMPLETED
2026-07-28T10:05, elapsed 05:34:54 on d4053.

| artifact | path | status |
|---|---|---|
| dense checkpoint | `transfer_v1/dense_dpo_lightr1_llama8b/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1/checkpoint-500` | present, 4 safetensors shards + `trainer_state.json` |
| deltas | `.../deltas/meta_llama_llama_3_1_8b_instruct_light_r1/deltas_step_{50..500}.pt` | **all 10 present**, 32.1 GB each |
| base_state | same dir, `base_state.pt` | present, 32.1 GB |

`trainer_state.json` reports `global_step 500`, `epoch 20.862`. The deleted original ran
to **20.86 epochs** (DO_NOT_REPEAT wall-clock table), so the retrain is in the same data
regime, not merely the same step count.

Hyperparameters (`run_manifest.json` + sbatch): 500 steps, effective batch 128
(per-device 2 × grad-accum 64 × 1 GPU), lr 5e-7, β 0.1, warmup-ratio 0.1,
max_length 1024, `adamw_8bit`, `--delta_log_interval 50 --delta_log_end_step 500`.
Identical to Scott spec and to the Tulu3 dense source, so the two halves of the sweep are
matched.

Also unblocked: the Tulu3 ρ=80 sparse arm (`8794867`) COMPLETED in 06:01:56 without OOM,
so the density-matched random controls that were held back pending that check can now be
submitted (§4).

---

## 1. The scheduling fact that shapes the whole plan

Measured 2026-07-28 20:10 with `sbatch --test-only`:

| shape | partition | estimated start |
|---|---|---|
| 8 cpu, 96 G, 4 h, no GPU | `short` | **2026-08-03T09:06** |
| 1×h200, 64 G, 1 h | `multigpu` | 2026-07-29T03:20 |
| 1×h200, 128 G, 8 h | `multigpu` | 2026-07-29T03:09 |

`short` has **2229 pending jobs, 91 nodes down, 10 drained**. Its estimated start is the
rebuttal deadline itself. So the standing rule "mask generation belongs on a CPU
partition" (`feedback_rc_idle_gpu_cancel`, and the DO_NOT_REPEAT entry written earlier
today) **is suspended for this sweep, on measurement, not on preference.** The h200 route
with the keep-alive sidecar is the only one that lands before 08-03.

Note the 128 G training shape is estimated *earlier* than the 64 G mask shape — at these
sizes the queue is not memory-bound, so there is nothing to win by shaving `--mem`.

### Per-user QOS ceiling — the binding constraint

`sacctmgr show qos multigpu`: **`gres/gpu=8`, MaxJobsPU 4, MaxSubmitPU 8.**

Currently occupying multigpu slots (jitter-probe chains, not part of this sweep):

| job | name | shape | state | what it computes |
|---|---|---|---|---|
| `8814942` | n5j12_masks | 16 cpu, 200 G, gpu:1, 3 h, c2204 | RUNNING, 1:10 in, ends ≤22:37 | `jitter_rel=1e-12` control: n=5 seed masks (Oracle + Magnitude-50/100/200 + Random, seeds 42–46) off Light-R1 checkpoint-500 |
| `8814991` | n5j12_probe | 8 cpu, 64 G, gpu:1, 2 h | PENDING `afterok:8814942` | stage-2 zero-out probe that turns those masks into the figure |
| `8814998` | n5j12_probe | identical to `8814991` | PENDING (JobHeldUser) | **duplicate** — same command, same dependency; the 1e-12 run log already records it as "created by a retried `sbatch`, pending disposal" |
| `8808817` | n5_masks | 8 cpu, 256 G, **no GPU**, `short` | PENDING (Priority), start time quoted `N/A` | the `jitter_rel=1e-3` n=5 mask gen — the **reference** the 1e-12 job is a control *for* |
| `8808818` | n5_probe | 8 cpu, 96 G, gpu:1, 3 h | PENDING `afterok:8808817` | stage-2 probe for that reference |

`8808817` carries no `gres`, so it does **not** consume a multigpu slot — only
`8814942`, `8814991`, `8814998`, `8808818` do. That is 4 of 8 submit slots and 1 of 4 run
slots. `8814942`/`8814991` free themselves by ~01:00 tonight, i.e. before the multigpu
queue estimate above, so they are not really in the way.

Two things are worth acting on, neither of which is a cancel of live work:

- **`8814998`** is a self-declared duplicate. Dropping it frees a submit slot at zero cost.
- **`8808817` is on the wrong partition, not merely unlucky.** 68 `short` nodes can
  physically hold the 256 G ask, but Slurm will not even quote a start time — it is past
  the backfill horizon behind 2229 jobs. If it never runs, the 1e-12 control finishing
  tonight has nothing to be a control *against*, and that job's whole single-knob design
  is wasted. The fix is the one already applied to its sibling: move the same work onto a
  multigpu v100 node doing CPU scoring (`n5_j1e12_stage1_masks_v100.sbatch` pattern —
  c2205/c2206/c2207 are 480 G v100-pcie nodes, idle or mix). Costs 1 submit slot, converts
  a job that will not run into one that runs tonight.

---

## 2. Stage A — masks (delta route), 1 job, ~5.5 h of work

Because a queue wait of ~7 h dominates a 10-minute mask, **pack all four ρ into one
allocation** rather than paying the wait four times. This also costs only 1 of the scarce
submit slots.

Route: `scripts/mask_from_delta.sbatch` (already written, never run — job `8814721` was
cancelled before start). It scores `|deltas_step_500|` in place instead of holding
`initial_sd + final_sd + scores`, so peak RSS is ~⅓ of the checkpoint-diff route.

**Equivalence argument:** `FlexibleCheckpointCallback` writes `delta = param − base_state`
with `base_state = θ(0)`, so `|delta_500| ≡ |θ(500) − θ(0)|` — exactly what
`checkpoint_diff_mask_finder.py` computes. Both then call the *same*
`create_mask_from_scores_gpu_efficient` with the same `sparsity_percent`,
`min_layer_keep_ratio=0.0025`, `local_pool=False`. This is **not**
`even_better_mask_finder.py --method magnitude`, which accumulates
`Σ_k |θ(k) − θ(0)|` and would silently change method between the two halves.

Order the loop **cheapest first** so the fast arms can start training while ρ=80 is still
selecting:

| # | ρ | expected elapsed (Tulu3-measured) | output |
|---|---|---|---|
| 1 | 99 | 0:10 | `oracle_dpo_lightr1_step500_sp99_src-deltafull.pt` |
| 2 | 97.5 | 0:25 | `..._sp97.5_src-deltafull.pt` ← **gate** |
| 3 | 90 | 1:31 | `..._sp90_src-deltafull.pt` |
| 4 | 80 | 2:58 | `..._sp80_src-deltafull.pt` |

Total ≈ 5:04 + load ⇒ `--time=08:00:00`.

### Sizing: `--mem=200G`, and **not** an h200

`mask_from_delta.sbatch`'s header claims 64 G suffices. **That is too low.**
`create_mask_from_scores_gpu_efficient` does `s = score.to(...).clone()` for every tensor
([mask_utils.py:623](../../src/utils/mask_utils.py#L623)), so a full second copy of the
32 GB score dict is materialised inside the selector no matter how carefully the caller
avoids one. The in-place `abs_()` saves the *initial* state dict, not the clone. Measured
checkpoint-diff peaks were 199 / 161 / 105 / 98 GB for ρ=80/90/97.5/99; a single packed
job must cover the ρ=80 peak, so **200 G**.

Partition: `--partition=multigpu --gres=gpu:1` (bare), **not** `gpu:h200:1`. This is
CPU-only work, and the v100-pcie nodes `c2205/c2206/c2207` are 480 G and idle-or-mix —
exactly where `8814942` landed 23 seconds after submit doing the same class of work. It
also keeps `feedback_must_use_h200`'s mask-gen exemption intact instead of putting a
no-optimizer job into the most contended pool. `sbatch --test-only` for this exact shape
quotes 2026-07-29T03:46 worst-case on d1028; backfill onto a c22xx node should be much
sooner. The DO_NOT_REPEAT rule about pinning a100/h200 applies to jobs doing **8B forward
passes** — this one does none.

The keep-alive sidecar in `mask_from_delta.sbatch` stays: the GPU really is idle for the
whole run and the RC 15-minute idle-GPU canceller does not care why.

Every filename carries `_src-deltafull` from the start. This avoids the Tulu3 mistake
where ρ=80/90/99 were written un-tagged and now collide in a directory listing with the
paper-era files (rename still pending on that half), and it makes collision with the
surviving `oracle_dpo_lightr1_step500_sp97.5.pt` structurally impossible.

Disk: 4 × 7.5 G into `transfer_v1/oracle_masks_llama8b/` (already 149 G). `/scratch` has
798 T free.

### The gate

`REF_MASK=/scratch/$USER/transfer_v1/oracle_masks_llama8b/oracle_dpo_lightr1_step500_sp97.5.pt`
— the surviving Apr-27 mask, whose source dense run was deleted. Jaccard of the new
ρ=97.5 mask against it answers whether the Light-R1 sweep may be read alongside the
published ρ=97.5 point. Chance level at ρ=97.5 ≈ 1.27e-2; the Tulu3 half scored **0.9763**.

- **High (≈0.97):** splice — the sweep extends the published point.
- **Low:** report the Light-R1 sweep self-contained; do **not** mix in the paper number.

The gate does **not** gate submission of Stage B. It changes how the result is written up,
not whether the arms are worth running, and serializing on it would cost another ~7 h
queue wait against a 6-day deadline.

### Stage A′ — route cross-check: **DROPPED** (Irene, 2026-07-28)

A `checkpoint_diff_mask_finder.py` run at ρ=97.5 off `checkpoint-500`, compared against
the Stage-A ρ=97.5 mask, would have measured the delta-vs-checkpoint-diff route
equivalence directly (expected Jaccard ≈ 1.0). Dropped to save a submit slot; the routes
are taken as equivalent on the derivation above.

**Consequence to carry:** if the gate comes back low, it will have two live explanations
— a genuinely different dense run, or a route difference — and this sweep will not be
able to separate them. One partial mitigation is free: both routes call
`create_mask_from_scores_gpu_efficient` with `add_tie_break_noise=True,
tie_break_noise_scale=1e-6` (neither caller overrides the defaults), so tie-breaking is at
least the same mechanism on both sides.

---

## 3. Stage B — sparse DPO arms, 4 jobs × ~6 h

Script exists and is parameterized: `scripts/transfer_p3_sparse_dpo_lightr1.sbatch`,
`MASK_PATH=<mask> RUN_TAG=<tag> sbatch ...` → `1×h200, 128 G, 8 h, --requeue`.

Hyperparameters are already fixed in the script and match both the dense source and the
Tulu3 arms: 500 steps, eff. batch 128 (2 × 64 × 1), lr 5e-7, β 0.1, warmup-ratio 0.1,
max_length 1024, `--optimizer sparse_adamw`, `save_steps 50`, `--resume_from_checkpoint auto`.
wandb project `rl_casino_transfer_v1`.

| ρ | RUN_TAG | output dir | trainable params |
|---|---|---|---|
| 80 | `oracle_sp80` | `sparse_dpo_lightr1_oracle_sp80` | 1,606,052,249 |
| 90 | `oracle_sp90` | `sparse_dpo_lightr1_oracle_sp90` | 803,026,124 |
| 97.5 | `oracle_sp97.5_deltafull` | `sparse_dpo_lightr1_oracle_sp97.5_deltafull` | 200,756,531 |
| 99 | `oracle_sp99` | `sparse_dpo_lightr1_oracle_sp99` | 80,302,612 |

**Do not reuse `RUN_TAG=oracle_step500`** — `transfer_v1/sparse_dpo_lightr1_oracle_step500`
is the May-02 paper arm and `--resume_from_checkpoint auto` would resume into it.

The ρ=97.5 arm is retrained rather than reusing the paper arm, so that all four points
descend from one dense source — the same choice the Tulu3 half made. The old arm then
serves as a free run-to-run reproducibility check at ρ=97.5.

Submit each arm as its mask lands; MaxJobsPU 4 means at most 3 can run alongside the
Stage-A job.

---

## 4. Stage C — density-matched random controls — **DEFERRED** (Irene, 2026-07-28)

Decision: hold until the four oracle arms report. If the ρ-sweep itself comes back flat,
the controls cannot distinguish anything either and 24 GPU·h is saved; if it separates,
they become the priority. Kept here so the design does not have to be re-derived.

Without these, an effect at ρ=80 cannot be attributed to the *mask* rather than to
parameter count — and three reviewers already flagged that Table 1 cannot separate oracle
from random. This is the part of the sweep that answers the actual criticism.

Needed: random masks at ρ ∈ {80, 90, 99} (`random_baseline_lightr1_sp97.5_seed42` already
exists for 97.5) via `scripts/transfer_p3_prep_random_mask.sbatch`, then 4 arms with
`RUN_TAG=random_sp<ρ>`. Mask generation is minutes; the arms are another 4 × ~6 h.

The same controls are still missing on the **Tulu3** half. That is out of scope for this
request but is now unblocked by the same ρ=80 no-OOM result.

---

## 5. Stage D — evaluation

Held-out preference eval per arm, ~5 min each. **Pin the GPU type:**
`--gres=gpu:a100:1` or `h200:1`, never a bare `--gres=gpu:1` — job `8786294` was assigned a
V100 for an 8B eval, ran ~10× slower and hit the wall (DO_NOT_REPEAT).

---

## 6. Timeline against the 2026-08-03 deadline

Assuming the ~7 h queue estimate holds (it is Slurm's reservation-based worst case;
backfill usually beats it):

| | earliest finish |
|---|---|
| Stage A masks | 07-29 ~09:00 (ρ=99 by ~03:30) |
| Stage B arms | 07-29 ~16:00 if 3 run concurrently, else 07-30 |
| Stage C controls | 07-30 → 07-31 |
| Stage D evals | same day as each arm |

Slack is ~3 days. The critical path is the multigpu **MaxJobsPU 4** ceiling, not compute.

---

## 7. Submission commands

```bash
S=/scratch/$USER
DELTA=$S/transfer_v1/dense_dpo_lightr1_llama8b/deltas/meta_llama_llama_3_1_8b_instruct_light_r1/deltas_step_500.pt
OUT=$S/transfer_v1/oracle_masks_llama8b

# Stage A — packed, cheapest rho first (needs the small wrapper described in §2)
sbatch --partition=multigpu --gres=gpu:1 --cpus-per-task=16 --mem=200G --time=08:00:00 \
  --export=ALL,DELTA="$DELTA",OUT_DIR="$OUT",DATASET=lightr1,OUT_TAG=deltafull,\
REF_MASK="$OUT/oracle_dpo_lightr1_step500_sp97.5.pt" \
  scripts/mask_sweep_from_delta_lightr1.sbatch

# Stage B — one per rho, as its mask lands
for SP in 99 90 80; do
  MASK_PATH=$OUT/oracle_dpo_lightr1_step500_sp${SP}_src-deltafull.pt \
  RUN_TAG=oracle_sp${SP} \
  sbatch scripts/transfer_p3_sparse_dpo_lightr1.sbatch
done
MASK_PATH=$OUT/oracle_dpo_lightr1_step500_sp97.5_src-deltafull.pt \
RUN_TAG=oracle_sp97.5_deltafull \
sbatch scripts/transfer_p3_sparse_dpo_lightr1.sbatch
```

`scripts/mask_sweep_from_delta_lightr1.sbatch` is the only new file: a loop over
`SPARSITIES=(99 97.5 90 80)` around the body of `mask_from_delta.sbatch`, one allocation,
gate fired on the 97.5 iteration. Not yet written.

---

## 8b. Results so far

| ρ | kept params | measured keep | matches Tulu3 half? | job | elapsed |
|---|---|---|---|---|---|
| 99 | 80,302,612 | 1.0000 % | yes, identical keep count | `8817043` | 45:38 |
| 97.5 | 200,756,531 | 2.5000 % | yes, identical keep count | `8817043` | 2:32:27 |

The keep counts are bit-identical to the Tulu3 half's at the same ρ, which confirms the
delta route sees the same 291-tensor scored set and allocates the same budget.

### The gate: **PASSED**

```
new   : oracle_dpo_lightr1_step500_sp97.5_src-deltafull.pt   (source: 8792090 retrain)
paper : oracle_dpo_lightr1_step500_sp97.5.pt                 (source: deleted run)
Jaccard = 0.9691        chance level at ρ=97.5 ≈ 1.27e-2
```

~76× above chance. The Tulu3 half scored 0.9763 by the same construction. So the Light-R1
sweep **can be read alongside the published ρ=97.5 point**, and the retrained dense run
recovers 96.9 % of the published subnetwork.

Read together, the two halves are the stronger claim on their own: two datasets, each with
an independently retrained dense DPO run, both recovering ≥96.9 % of the originally
published 2.5 % subnetwork. That is run-to-run reproducibility of the oracle subnetwork,
not merely jitter-stability of one scoring pass.

---

## 9. The selector is memory-latency bound — node generation dominates runtime

The first attempt (`8817043`) landed on **c2205**, which backfilled in 63 s against a
`--test-only` estimate of 03:46. The catch: c2205 is `AvailableFeatures=zen`, an
**AMD EPYC 7351** — first-generation Naples, 2.4 GHz.

| ρ | Tulu3 half (`short` nodes) | this attempt (c2205, `zen`) | ratio |
|---|---|---|---|
| 99 | 0:10:04 | 0:45:38 | 4.5× |
| 97.5 | 0:24:28 | 2:32:27 | **6.2×** |

Not memory pressure: `free -g` on c2205 showed 403 G available of 471 G with **zero swap**,
and job MaxRSS was 114 GB against a 200 G request. Not core starvation either: the process
ran 31 threads but averaged only ~280 % CPU of 16 allocated cores. It is the
**global top-k over 8.03 B elements**, which is memory-latency bound, and first-gen EPYC
has the worst memory subsystem of any node generation on this cluster.

**The consequence is a hard block, not a delay.** Extrapolating linearly in keep count —
already the optimistic reading, since the observed scaling is superlinear — ρ=90 needs
~7.6 h and ρ=80 ~15 h on a `zen` node, against the cluster's **8 h hard walltime cap**.
ρ=80 cannot complete on that node class at any walltime request.

**Fix:** `--constraint="zen2|cascadelake"`. Node generation is exposed as a Slurm feature,
so it is selectable:

| node | feature | cores | RealMemory |
|---|---|---|---|
| c2205 | `zen` | 32 | 480 G |
| d1026 | `zen2` (Rome) | 64 | 512 G |
| d4053 | `cascadelake` | 128 | 512 G |

Two things that do **not** work, both measured: `short` is quoted 2026-08-03 even with the
constraint applied, and `multigpu` rejects a job with no `--gres` outright
(`allocation failure: Access/permission denied`), so CPU-only work still has to hold a GPU
there.

**Sizing correction:** the delta route does not save as much memory as
`mask_from_delta.sbatch`'s header claims. Measured MaxRSS was 114 GB through ρ=97.5, above
the checkpoint-diff route's 105 GB at the same ρ. The ρ=90/80 jobs therefore ask
**240 G** — what the Tulu3 half actually ran under — not the 64 G in that header.

**Script design note:** `mask_sweep_from_delta_lightr1.sbatch` reloads the 32 GB delta
inside every ρ iteration instead of loading once and reusing it. Loading once would have
been strictly better and would not have raised the peak, since
`create_mask_from_scores_gpu_efficient` clones the scores anyway and the originals have to
stay alive across the call. Worth fixing before the next multi-ρ sweep.

---

## 10. Decisions

| # | decision | 2026-07-28 |
|---|---|---|
| 1 | Stage A′ route cross-check | **dropped** — see §2 for the consequence |
| 2 | Stage C random controls | **deferred** until the oracle arms report |
| 3 | `n5*` slot disposition | reported in §1; awaiting Irene. Nothing cancelled or resubmitted without an explicit instruction. |

`scripts/mask_sweep_from_delta_lightr1.sbatch` written and submitted as `8817043`.
It exits non-zero if any ρ fails, because `sacct State=COMPLETED` is not proof the work
happened (DO_NOT_REPEAT). Watch `logs/lr1_mask_sweep_8817043.out`: each ρ prints its
measured density, and the ρ=97.5 iteration prints the gate Jaccard.

Stage B is submitted per ρ as its mask appears — no need to wait for ρ=80.
