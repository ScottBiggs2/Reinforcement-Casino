# Qwen3-32B chain — resumed on AICR from Explorer checkpoint-250

**Date:** 2026-07-29 (late) · **Branch:** `irene-rebuttal-lora` · **Cluster:** AICR `b200-batch`
**Requested by:** Irene — "把 qwen32b 的在新集群跑完吧 重新跑也行", for NeurIPS #29841
(reviewer/AC item 1: multiple base models; results postable until 2026-08-03).
**Scripts:** `scripts/aicr_qwen32b_p1_dense.sbatch`, `scripts/aicr_qwen32b_p2_oracle.sbatch`,
`scripts/aicr_qwen32b_p3_sparse.sbatch`, `scripts/aicr_qwen32b_chain.sh`
**Predecessor:** `docs/run_logs/qwen3_32b_autopipeline_2026-06-12.md` (Explorer autopipeline,
paused 2026-06-14 at p1 checkpoint-250/500, `PIPELINE_STALLED` sentinel left in place).

## Decision: resume, not rerun

Irene allowed either. Resume wins on the deadline: checkpoint-250 is 250 steps
(~1 slot-day) of paid-for compute, and p3 still needs a full 500 steps from scratch.
The chain must land before 08-03 for the number to be postable.

Verified before reuse: `trainer_state.json` at checkpoint-250 reads `global_step=250`,
`learning_rate=2.789e-07` — exactly the linear-schedule value at 250/500 with 50-step
warmup and peak 5e-7 (5e-7 × 250/450 = 2.7889e-7). The checkpoint is internally
consistent with the intended schedule.

## Provenance / disclosure

- **The p1 trajectory spans two clusters at step 250**: steps 1–250 on Explorer
  3×H200 (June 12–13), steps 251–500 on AICR 3×B200. bf16 numerics differ across
  GPU generations; env differs (Explorer June env vs AICR torch 2.9.0+cu128,
  transformers 4.57.1, trl 0.24.0). Optimizer state (adamw_8bit) is carried in
  `optimizer.pt` and is per-parameter, so the resume is exact up to numerics.
- checkpoint-250 (133 GB: 14 safetensors shards + 71 GB optimizer.pt) pushed
  Explorer→AICR via the chunked-dd route (`~/xfer_qwen32b_ckpt250.sh` on Explorer,
  4 workers). **Gate: per-file md5 must match on both sides before the chain is
  submitted**; the sentinel `CKPT250_VERIFIED` is written only after that check and
  `aicr_qwen32b_p1_dense.sbatch` refuses to start without it (migration lesson —
  a size-stability check once ran jobs against a 6.6 GB fragment of a 32.1 GB file).
- Qwen/Qwen3-32B weights pulled directly from HF on AICR (network + token verified
  2026-07-29); jobs run offline from the cache per `aicr_env.sh`.
- Explorer keeps the original checkpoints (150/200/250) untouched; nothing was
  deleted or cancelled on Explorer for this.

## Hyperparameters (unchanged from the Explorer run = Scott spec; only cluster differs)

| param | p1 dense | p3 sparse |
|---|---|---|
| backbone | Qwen/Qwen3-32B (instruct, NOT -Base) | same |
| dataset | light-r1 | same |
| steps | 500 | 500 |
| eff. batch | 128 (bs 2 × ga 64, single MP process) | same |
| lr / schedule | 5e-7, linear, warmup_ratio 0.1 | same |
| dpo_beta | 0.1 | same |
| max_length / max_prompt | 1024 / 1024 | same |
| optimizer | adamw_8bit | **sparse_adamw** |
| mask | — | p2 oracle @97.5%, `min_layer_keep_ratio` 0.0025 |
| parallelism | device_map=auto, **3×B200** | device_map=auto, **4×B200** |
| save_steps / limit | 10 / 3 | 10 / 3 |
| wall / timeout | 24 h / `timeout 82800s` | same |
| wandb | project `rl_casino_transfer_v1` | same |

p3 gets 4 GPUs, not 3: SparseAdamW holds dense fp32 moment buffers for every 2-D
param (~254 GB at 32.8B) on top of policy+ref+grads (~197 GB bf16) — ~480 GB total
against 3×178 GiB = 534 (tight, uneven layer placement) vs 4×178 = 712 (safe).
Naive-MP step time is roughly stage-count invariant, so the 4th GPU costs queue
share, not speed.

## Orchestration — afterok chain instead of advance.sh

Explorer needed the custom re-entrant advancer because its 8 h wall killed jobs on
TIME LIMIT (never requeued) and `MaxJobsPU 4` forbade pre-queuing slots. AICR has
24 h walls and no MaxJobsPU, so the whole chain is pre-submitted as plain `afterok`:

```
p1 ×3 → p2 → p3 ×3
```

Each training slot wraps python in `timeout 23h`, exits 0 on clean-or-timeout, and
the next slot resumes (`--resume_from_checkpoint auto`). Once a phase's
checkpoint-500 exists, its remaining slots exit 0 in seconds — extra slots are free
insurance. A hard crash (rc ∉ {0, 124}) propagates and freezes the rest of the
chain (DependencyNeverSatisfied), so nothing trains on top of a corrupt state.

Slot math against the Explorer-measured 177–422 s/it (B200 should be ≤ H200):
p1 needs 250 steps ≤ 2 slots even at the worst rate; 3 gives margin. p3 needs 500
steps ≤ 3 slots at ≤497 s/it. If B200 lands near 150–250 s/it the chain finishes
p1 in 1 slot and p3 in 2.

## Job IDs

Submitted 2026-07-30 01:27 EDT after the md5 gate passed (29/29 files match between
Explorer source and AICR dest; sentinel `CKPT250_VERIFIED` written 01:26, manifests
`Discovery:~/qwen32b_ckpt250_src.md5` / `$W/qwen32b_ckpt250_dest.md5`). Transfer
itself finished 01:07 EDT, `ALL_DONE`, 0 chunk failures.

GPU slots submitted with `--exclude=b0008,b0009,b0010,b0026,b0028` — every node
running one of our seven GRPO jobs at submit time (contention lesson from 235302:
156 s/it at 14% GPU util beside our own mask-gen job). chain.sh now derives this
exclude list from `squeue` automatically; override with `EXCLUDE_NODES=`.

| stage | job | state at submit | outcome |
|---|---|---|---|
| p1 slot A | 235953 | PD (Priority) | **FAILED 00:01:01** — base model missing (§ below) |
| p1 slot B | 235954 | afterok:235953 | CANCELLED (DependencyNeverSatisfied) |
| p1 slot C | 235955 | afterok:235954 | CANCELLED |
| p2 mask | 235956 | afterok:235955 | CANCELLED |
| p3 slot A | 235957 | afterok:235956 | CANCELLED |
| p3 slot B | 235958 | afterok:235957 | CANCELLED |
| p3 slot C | 235959 | afterok:235958 | CANCELLED |

## Attempt 1 failed: the base model was never on AICR

`235953` started 02:15, resolved the checkpoint-250 resume path correctly, loaded
the Light-R1 dataset from cache, and died 61 s in at
`AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")` with `LocalEntryNotFoundError` —
`HF_HUB_OFFLINE=1` and **no `models--Qwen--Qwen3-32B` in
`/scratch/xie_yiyi_neu/hf_cache/hub`**. The chain read rc=1 as a hard crash and
cancelled the six downstream jobs, which is the designed behaviour (never train on
top of a broken state), so nothing was corrupted and no GPU-hours were burned
beyond the minute.

**The provenance line in the "Decision" section above was wrong.** It recorded the
weights as "pulled directly from HF on AICR (network + token verified)". What was
actually verified was reachability of `huggingface.co`; the `hf download` that was
supposed to fetch the weights hangs with **zero bytes written** and leaves only a
urllib3 warning in its log. Reproduced on 2026-07-30 before switching routes —
process ALIVE for 5+ minutes, no cache directory ever created, while `curl -sI` on
the same node returns 307 in 0.33 s. Recorded in `DO_NOT_REPEAT.md`.

**Recovery route:** Explorer's cache already holds the model from the June run
(27 blobs, 65.5 GB). Moved with `~/xfer_qwen32b_base.sh` — the same chunked-dd
transport that carried checkpoint-250 — plus a replay of the `snapshots/<rev>/*`
symlinks, `refs/main`, and `.no_exist/<rev>/` entries, since blobs alone do not
make a resolvable HF repo.

**New gate:** `scripts/preflight_base_model.py` resolves config → tokenizer →
weight index → every shard named in the index, and both p1 and p3 now call it
before training. A config-only check would not have caught a missing 3.9 GB shard.

**Verification of the recovered cache (2026-07-30 14:34–15:10):** 27/27 blobs
transferred with zero chunk failures. Integrity was checked *against HF's own
published hashes* rather than against Explorer: the cache is content-addressed, so
the 64-hex blob filenames are the upstream sha256 — recomputing them locally gave
**18/18 match** (17 weight shards + `tokenizer.json`; the remaining 9 small files
are git-sha1-named config/vocab and are covered by the preflight's resolution
check). This is a stronger gate than the checkpoint-250 md5 comparison, which only
proved "same as Explorer". The Explorer↔AICR md5 comparison was also completed as
a secondary check and matches on **all 27 blobs**. Snapshot symlinks: 0 broken.
Preflight: PASS (qwen3, 64 layers; vocab 151669; 17 shards).

## Attempt 2 (2026-07-30 15:10)

| stage | job | dependency |
|---|---|---|
| p1 slot A | 238324 | — (PD Priority) |
| p1 slot B | 238325 | afterok:238324 |
| p1 slot C | 238326 | afterok:238325 |
| p2 mask | 238327 | afterok:238326 |
| p3 slot A | 238328 | afterok:238327 |
| p3 slot B | 238329 | afterok:238328 |
| p3 slot C | 238330 | afterok:238329 |

`--exclude` resolved to empty at submit because the seven GRPO arms had all
completed, so no node was hosting one of our jobs. Within the chain p1 and p3 are
`afterok`-sequential and never hold GPUs at the same time, so there is no
self-contention. Anything submitted alongside this chain must still exclude the
node p1 lands on.

## The cluster filled up — pivot to b200-devel for p1

By 16:00 on 07-30 `b200-batch` was saturated by other users (283 pending jobs from
one user, 131 from another, ~600 total). Measured free GPUs across the 28 nodes:
**4**, spread as b0010×2, b0013×1, b0026×1 — and b0007 has 8 free but is `drain`
(`gpu5_fault-needs_reset`). p1 wants **3 GPUs on one node**, which no node could
offer, so Slurm quoted a start of **2026-07-31T15:40** — a full day of idling
against an 08-03 deadline.

`b200-devel` was nearly empty at the same moment: 13 free GPUs over 3 nodes
(b0029×5, b0030×5, b0031×3) and 2 pending jobs. Its cost is a 4 h wall, which the
existing `timeout`+`--resume_from_checkpoint auto` design already absorbs.

**QOS limit found the hard way:** `b200-devel` carries `MaxTRESPU gres/gpu=2`, so a
3-GPU ask is rejected at submit with `QOSMaxGRESPerUser`. p1 fits in 2 because it
uses `adamw_8bit`: policy + ref + grads + 8-bit moments ≈ 244 GiB against
2×178 = 356 GiB (~70%). **p3 cannot use devel at all** — SparseAdamW's dense fp32
moments need 4 GPUs, twice the devel ceiling.

To avoid two jobs training into one output directory (the 234945/234952 collision
in `DO_NOT_REPEAT.md`), the three batch p1 slots were put on `scontrol hold` before
the devel chain was submitted. They must be released or cancelled once p1 reaches
checkpoint-500.

| devel slot | job | outcome |
|---|---|---|
| A | 238613 | started on b0030 within 13 s |
| B | 238614 | afterok:238613 |
| C | 238615 | afterok:238614 |
| D | 238673 | afterok:238615 (insurance) |
| E | 238674 | afterok:238673 (insurance) |

## B200 step time: 103 s/it — the chain is feasible after all

**Measured, not estimated.** Steps 252 and 253 of 238613 took 102 s and 103 s on
2×B200. The Explorer run was **177–422 s/it on 3×H200**, so B200 is 1.7–4× faster
and every earlier feasibility estimate (including the "cannot complete" verdict in
`DO_NOT_REPEAT.md`, which was written against the Explorer rate and an 8 h wall)
is superseded for this cluster.

Resume correctness confirmed from the same log: training restarted at **251/500**
with `learning_rate=2.7778e-07`, exactly 5e-7 × 250/450 for the linear schedule
with 50-step warmup. `rewards/accuracies` reads 1.0 and `margins` ≈ 3.8, which is
the ~21-epoch overfit regime the rebuttal already discloses for Light-R1 — the 32B
arm behaves like the 8B one.

Setup overhead per slot ≈ 10 min: base model 3.3 min, ref model 5 s (page cache),
tokenisation 70 s, and `optimizer.pt` (66 GiB) off NFS at **688 MiB/s** ≈ 100 s.
That is 4% of a 4 h devel slot.

**Budget at the measured rate**, including the ~200 s cost of writing a 136 GB
checkpoint every `save_steps=10`:

| stage | work | estimate |
|---|---|---|
| p1 remaining | 250 steps | ~93 steps per devel slot → **3 slots** (5 queued) |
| p2 mask | 32.8B ckpt-diff on cpu | ~2 h |
| p3 sparse | 500 steps | ~18 h — **fits one batch slot**, no split needed |

Against the 08-03 deadline: p1 lands ~07-31 early morning, p2 by mid-morning, and
p3 completes ~08-02 even if it waits 24–36 h for 4 B200s. There is real margin.

## p3 fallback: rtx-batch works (probe 238642)

`rtx-batch` is a **separate 32-GPU QOS pool** that does not consume the b200
ceiling, and 6× RTX PRO 6000 (95 GiB each, 570 GB) covers p3's ~480 GB. The open
question was the kernel: every sparse measurement we have is B200 `sm_100`, and
RTX PRO 6000 Blackwell is `sm_120`. Probe 238642 ran SparseAdamW + the Triton
kernel there for three steps **with the learning rate changing each step** — the
`tl.constexpr` recompile path real training takes — and passed. So p3 has a second
route if the B200 queue does not clear. Caveat: RTX is bandwidth-poorer than B200,
so assume a slower s/it and re-measure rather than reusing 103.

## Continuation chain (supersedes 238324–238330)

p2 and p3 are re-chained onto the devel p1 slots, with p3 split into 12 h slots
because a shorter `--time` backfills far better than a 24 h ask:

| stage | job | dependency |
|---|---|---|
| p2 mask | 238677 | afterok:238674 |
| p3 slot A | 238678 | afterok:238677 |
| p3 slot B | 238679 | afterok:238678 |

The original batch chain 238324–238330 is **inert**: its p1 slots are
`JobHeldUser`, so they cannot start and the p2/p3 jobs behind them can never fire.
Left in place rather than cancelled, per the standing "ask before scancel" rule.

## Read-out plan

Same as the 8B chain (`qwen3_8b_high_sparsity_dpo_2026-06-01.md`): dense vs sparse
reward margin on light-r1; the headline is what fraction of the dense margin the
2.5% subnetwork recovers at 32B. Held-out preference eval to follow if the window
allows. Any posted number carries the two-cluster disclosure above.
