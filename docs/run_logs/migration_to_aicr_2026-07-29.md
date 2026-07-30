# Migration of the Light-R1 ρ-sweep + n5 jitter study from Explorer to AICR

**Date:** 2026-07-29 · **Branch:** `irene-rebuttal-lora` · **Requested by:** Irene
**Outcome:** whole line running on AICR B200s; all 8 Explorer jobs cancelled at 17:36 EDT
after a 1:1 coverage check. Deadline context: NeurIPS #29841 rebuttal results due
2026-08-03.

## 0. Why

Explorer went into a maintenance-shaped hole: `multigpu` with 13 nodes drained, `short`
with 91 down and ~2200 pending, and **every h200 in `d4052-4055` unavailable** —
`8817763` (the ρ=99 arm) was sitting on `ReqNodeNotAvail` rather than on priority. Eight
rebuttal jobs were PENDING with no quoted start time inside the deadline.

The binding difference is the per-user ceiling, not the silicon:

| | Explorer `multigpu` | AICR `b200-batch` |
|---|---|---|
| per-user GPUs | `gres/gpu=8` | `gres/gpu=32` |
| concurrent jobs | **MaxJobsPU 4**, MaxSubmitPU 8 | none |
| walltime | 1 d (`gpu` is 8 h) | 24 h |
| GPU | H200 141 GB (all drained) | B200 sm_100 **178 GiB** |
| host RAM | 512 G | **2.26 TB** |

On Explorer four arms queued behind one surviving h200. On AICR they are four independent
jobs on four B200s.

## 1. What was verified on B200 before trusting it (smoke `234788`, 21 s)

- sm_100, 178 GiB, bf16 matmul fine.
- **The Triton `indexed_sparse_adamw` kernel compiles and runs correctly on sm_100**:
  419,430/419,430 masked weights updated, **0 strays**. No Hopper-only assumption.

That was the only real portability question; hyperparameters were copied verbatim
(500 steps, eff. batch 128 = 2 × 64 × 1, lr 5e-7, β 0.1, warmup-ratio 0.1, max_length 1024,
`--optimizer sparse_adamw`, `save_steps 50`, project `rl_casino_transfer_v1`).

## 2. Data transfer (Explorer login → AICR), completed 17:32:41

~250 GB. Two traps, both hit and both recorded in DO_NOT_REPEAT:

- The **AICR login node reaps `rsync --server` receivers on a ~5 min cycle.** With
  `--append-verify` each reconnect re-checksums the whole existing prefix, so throughput
  fell to ~2.3 MB/s and kept falling as the file grew. Replaced with **1 GiB chunks via
  `dd … | ssh … dd seek=`**: no transfer spans a reap, each chunk retries alone, and
  writing at absolute offsets makes the whole thing idempotent and resumable. Measured
  ~31 MB/s per stream afterwards.
- **`rsync --partial` renames its half-file to the *final* filename.** A stray retry loop
  (`xfer_d500.sh`) therefore clobbered an already-complete 32.1 GB `deltas_step_500.pt`
  down to 8.45 GB with an epoch-0 mtime, *after* the chunked writer had logged it done.
  Fixed by `xfer_d500_fix.sh`: `rm` then full resend, then prove it —
  `local md5 = remote md5 = 9816aae2b31bb94523b40e0858074cc7`, **MD5 MATCH**.

Readiness checks in `launch_all.sh` were changed from "size stopped changing" to an exact
byte match against the source (`file_ready`/`dir_ready`) — the stability check is what
submitted mask jobs `234836/234839/234847` against a 6.6 GB fragment of a 32.1 GB delta,
all three dying in <25 s on a truncated zip archive.

## 3. Coverage check before cancelling anything

| Explorer job (all PENDING, none ever started) | experiment | AICR | state at 17:36 |
|---|---|---|---|
| `8817763` (ReqNodeNotAvail) | ρ=99 arm | `234944` | RUNNING 48/500 |
| `8819594` | ρ=97.5 arm | `234945` | RUNNING 48/500 |
| `8819592` | ρ=90 mask | `234997` (`mask_sp90`) | PENDING `cpu` |
| `8818835` | ρ=80 mask | `234999` (`mask_sp80`) | PENDING `cpu` |
| `8824463` (afterok 8819592) | ρ=90 arm | `234998` (afterok 234997) | PENDING |
| `8824464` (afterok 8818835) | ρ=80 arm | `235000` (afterok 234999) | PENDING |
| `8808817` | n5 stage-1 masks | `235001` | PENDING `cpu` |
| `8808818` (afterok 8808817) | n5 stage-2 probe | `235002` (afterok 235001) | PENDING |

All eight cancelled: `CANCELLED by 103793`, elapsed 00:00:00 on every one — no compute was
thrown away, because none of them had ever been scheduled.

**A second reason to cancel, not just a neutral cleanup:** wandb is now wired up on AICR,
and both clusters write to the same project `rl_casino_transfer_v1`. An Explorer arm that
finally got scheduled would have written a *second* curve for an experiment already
running on AICR — the cross-cluster version of the `234952` duplicate below.

## 4. wandb

Arms `234928/234929` died in `on_train_begin` with `api_key not configured (no-tty)`:
AICR had no credential and a batch job cannot answer the login prompt. Sequence of fixes:

1. `WANDB_MODE=offline` in `aicr_env.sh` → arms relaunched as `234944/234945`, metrics to
   `/scratch/xie_yiyi_neu/wandb/`.
2. Irene supplied the key → `~/.netrc` (mode 600), verified against entity
   `xxiellan-northeastern-university`.
3. The three offline runs synced: `n56u594k` (ρ=99), `qbclyo1k` (ρ=97.5),
   `hx0d0obt` (the duplicate).
4. `aicr_env.sh` default flipped `offline` → `online`, so every job submitted from now on
   streams live. The two arms already running stay offline for their whole life; job
   `234980` (`--dependency=afterany:234944:234945`) re-syncs them with `--include-synced`
   when they exit, so a crash still yields curves.

Caveat worth knowing: **syncing a still-running offline run makes it show `finished` on the
dashboard.** Do not read that as done.

### 4b. The ρ-sweep landed in two different wandb projects (found + fixed 2026-07-30)

With all four arms COMPLETED at step 500, ρ=99 (`n56u594k`) and ρ=97.5 (`qbclyo1k`) were in
`rl_casino_transfer_v1` but ρ=90 (`vk25ioy4`, job 234998) and ρ=80 (`mescjqx2`, job 235056)
were in **`huggingface`**.

Root cause: `src/full_training/sparse_dpo_efficiency.py` hardcoded
`wandb_project = "huggingface"` and then *overwrote* `os.environ["WANDB_PROJECT"]` — so the
`export WANDB_PROJECT="rl_casino_transfer_v1"` in `aicr_p3_sparse_dpo.sbatch:29` was dead
code. The two offline arms escaped only because `wandb_sync_after.sbatch` re-uploads with an
explicit `wandb sync --project "$WANDB_PROJECT"`; the two online arms took the hardcoded
default. `sparse_dpo_bsr.py:175` had the same line; the GRPO scripts already used
`os.environ.get("WANDB_PROJECT", "huggingface")`, which is why the `rl_casino_rebuttal` arms
were never affected.

Fix: both DPO scripts now read the env var (patched in the repo **and** in the AICR working
copy `$W/rc-sparse-speed`, whose git metadata still points at the Explorer home path and so
cannot be updated by `git pull`).

Consolidation: the two stray runs were moved with the `moveRuns` GraphQL mutation
(`api.client.execute`, filter `{"name": {"$in": [...]}}`) — run IDs, history and runtimes are
preserved, no re-sync. All four now carry the tag `lightr1_oracle_rho_sweep`:

| ρ | run | job | runtime |
|---|---|---|---|
| 99 | `n56u594k` | 234944 | 3h40m27s (13227 s) |
| 97.5 | `qbclyo1k` | 234945 | 3h40m14s (13214 s) |
| 90 | `vk25ioy4` | 234998 | 5h00m01s (18001 s) — b0021 contention |
| 80 | `mescjqx2` | 235056 | 3h38m35s (13115 s) |

**Still split:** the Tulu3 half of the same sweep (`cj5c3606` sp99, `a0gb8auf` sp97.5,
`fhxyqy81` sp90, `rpo1kmdi` sp80, all 2026-07-28) is still in `huggingface` for the same
reason. That also corrects §6 below: those Tulu3 arms *do* have wandb runs — they were just
filed under the default project.

## 5. The duplicate: `234952`

`234952` and `234945` trained the **same ρ=97.5 arm from the same mask into the same
output directory**. Root cause, measured on the live queue:

```
squeue -h -O "Comment"  →  20 oracle_step500_sp99      20 oracle_step500_sp97.   ← truncated
squeue -h -o "%k"       →  19 oracle_step500_sp99      21 oracle_step500_sp97.5
```

`-O` pads and truncates to 20 chars, so the 21-char ρ=97.5 tag never matched the dedup
grep while the 19-char ρ=99 tag did — a guard that worked for one value of the loop
variable and silently failed for another. `autolaunch.sh` invoked the submitter **twice per
tick**, so the hole fired inside a single tick.

Cancelled at 15:04 elapsed / step 21, **before `save_steps=50` wrote anything**, so the
shared checkpoint dir was never corrupted. Cleanup: `wandb_run_id.txt` restored from
`hx0d0obt` to `qbclyo1k`; the duplicate offline dir moved out of the sync glob to
`wandb/_cancelled_dup_234952-hx0d0obt`; the wandb run renamed
`ZZ-CANCELLED-duplicate-234952-of-qbclyo1k` and tagged `cancelled/duplicate/do-not-analyse`
(kept, not deleted). Guards now use `tag_queued () { squeue -h -o "%k" | grep -Fxq "$1"; }`
at all three sites, and `autolaunch.sh` runs the submitter once per tick.

## 6. Still owed on Explorer — do NOT treat the cluster as finished with

**Deprioritized by Irene 2026-07-29:** the graft line below is parked. It is not to be
raised again unless she raises it — this section is a record so nothing has to be
re-derived if she does, not a to-do. The data must still not be deleted.

The graft experiments were cancelled at 03:47 today and never resubmitted:

- A `8818348` — necessity/sufficiency probes (`delta_only`, `anti_delta_only`)
- B `8818351` — held-out margin under θ_base + α·Δθ⊙M

Their prep `8818344` **COMPLETED** (03:37:45) and its outputs exist **only on Explorer**:
`/scratch/xie.yiyi/graft_ckpts_lightr1/{graft_oracle_a1,graft_anti_a1,graft_random42_a1,graft_oracle_a05}`
(~16 GB each) plus `dense_training_args.txt` and the three
`random_matched_lightr1_sp97.5_seed{42,43,44}.pt` masks. Rerunning them on AICR means
copying ~64 GB of grafted checkpoints first. See `graft_necsuff_lightr1_2026-07-28.md`.

Also Explorer-only: the 07-28 **Tulu3** sparse arms (`8794867-71`) have no wandb run and no
offline dir anywhere — wandb was off for them, so their metrics exist only inside each
`trainer_state.json`.

## 7. Where things stand at 17:36 EDT

```
AICR   234944  ρ=99    RUNNING  48/500  ETA ~2h31m   b0009
       234945  ρ=97.5  RUNNING  48/500  ETA ~2h31m   b0014
       234997  mask ρ=90  PENDING cpu  →  234998  arm ρ=90  (afterok)
       234999  mask ρ=80  PENDING cpu  →  235000  arm ρ=80  (afterok)
       235001  n5 masks   PENDING cpu  →  235002  n5 probe  (afterok)
       234980  wandb_sync_after  PENDING (afterany the two arms)
Explorer  queue empty; SSH, /scratch and the graft artifacts untouched
```
