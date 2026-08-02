# Qwen3 random-mask controls (32B + 8B) + dense-curve backfill

**Date:** 2026-08-01 · **Branch:** `irene-rebuttal-lora` · **Cluster:** AICR
**Requested by:** Irene — "补跑完 qwen3 32b 的 random run / dense run 为什么只显示半截 / 把 qwen3 8b 的 random run 也跑了"
**Scripts:** `scripts/aicr_qwen32b_mask_random.sbatch`, `scripts/aicr_qwen8b_mask_random.sbatch`,
`scripts/aicr_qwen8b_p3_sparse.sbatch`, `scripts/aicr_qwen32b_p3_sparse.sbatch` (WANDB_PROJECT now
overridable, default `qwen3`), `scripts/wandb_backfill_qwen32b_dense.py`

## 1. Dense 32B "half curve" — diagnosed and fixed

wandb `97idxxzx` showed only steps 251–500 because the AICR job resumed from the
Explorer checkpoint-250 as a **fresh** wandb run (see
`qwen3_32b_aicr_resume_2026-07-29.md`); the June Explorer attempts never logged
steps 1–250 online (all are `dead-attempt` crashes — verified across every
project in the entity). The full history exists in
`checkpoint-500/trainer_state.json` (500 entries, steps 1–500).

Fix (2026-08-01 18:31 ET): `wandb_backfill_qwen32b_dense.py` replayed the full
log_history into a new run with live-run metric names (`train/*`):

- **`40yoz8rd`** `dense_dpo_qwen3_32b_light_r1_scott_full` — full 1–500 curve,
  tags `backfilled_full_curve`, `two_cluster_provenance`. Verified: 500 rows,
  margins 0 → 4.243.
- `97idxxzx` renamed `(partial, live 251-500) …`, tag `partial-resume-segment`.
  Not deleted (it is the primary record of the AICR segment).
- Provenance carried in run notes: steps 1–250 Explorer 3×H200, 251–500 AICR
  3×B200 (bf16 numerics differ across the seam; disclosure identical to the
  resume log).

`FIGURE_RUN_REGISTRY.md` has a new "Qwen3 DPO 图" section anchoring all IDs.

## 2. Random-mask controls — design

Null-hypothesis arm for both Qwen3 backbones, completing the dense/oracle/random
triplet in project `qwen3`. Same recipe as the paper's Llama random baseline
(`random_mask_baseline.py`, uniform scores → identical selector path,
density-matched to the oracle reference, seed 42):

| param | 32B | 8B |
|---|---|---|
| reference mask | `oracle_masks_qwen3_32b/oracle_dpo_light_r1_step500_sp97.5.pt` | `oracle_masks_qwen3_8b/oracle_dpo_light_r1_step500_sp97.5.pt` (rsynced Explorer→AICR 2026-08-01, 8.19 GB, `XFER_OK`) |
| mask output | `oracle_masks_qwen3_32b/random_baseline_light_r1_sp97.5_seed42.pt` | `oracle_masks_qwen3_8b/random_baseline_light_r1_sp97.5_seed42.pt` |
| ρ / floor / seed | 97.5% / `min_layer_keep_ratio` 0.0025 / 42 | same |
| dataset | light-r1 | light-r1 (tulu3 arm not run: June log shows tulu3 = 0.23-epoch non-learning arm; a random control there would be uninformative) |
| training | 500 steps, bs 2 × ga 64 (eff. 128), lr 5e-7 linear, warmup_ratio 0.1, beta 0.1, max_len/prompt 1024, wd 0, sparse_adamw, grad ckpt | same |
| GPUs | 4×B200 (`device_map auto`; SparseAdamW dense fp32 moments ~254 GB) | 1×B200 (~130 GB total) |
| wandb | project `qwen3` (direct, no reorg needed), run `sparse_dpo_qwen3_32b_light_r1_random_sp97.5_seed42_500steps` | project `qwen3`, run `sparse_dpo_qwen3_8b_light_r1_random_sp97.5_seed42_500steps` |
| timeout/resume | `timeout 23h` + `--resume_from_checkpoint auto`, insurance slot afterok | same |

Qwen3-8B base weights were absent from the AICR cache; `snapshot_download` with
the IPv4-first `getaddrinfo` patch (same trick as
`aicr_prefetch_eval_datasets.py`) fetched all 15 files in ~12 s on the login
node — the 07-29/30 "hf download hangs" failure was the login IPv6 blackhole,
now with a working route. Cache: `hf_cache/hub/models--Qwen--Qwen3-8B`
(snapshot `b968826d`).

## 3. Job IDs (submitted 2026-08-01 18:30 ET)

| stage | job | dependency | outcome |
|---|---|---|---|
| 32B random mask (cpu, 600G, fp32) | 248909 | — | **FAILED** (selector, § below) |
| 32B p3 sparse slot A (4×B200) | 248910 | afterok:248909 | CANCELLED (dep) |
| 32B p3 sparse slot B (insurance) | 248911 | afterok:248910 | CANCELLED (dep) |
| 8B random mask (cpu, 250G) | 248912 | — | ✅ COMPLETED 4 min, density 2.5000%, 7.7 GB |
| 8B p3 sparse slot A (1×B200) | 248913 | afterok:248912 | queued |
| 8B p3 sparse slot B (insurance) | 248914 | afterok:248913 | queued |
| 32B random mask attempt 2 (cpu, 800G, **fp64**) | 248967 | — | queued |
| 32B p3 sparse slot A (4×B200) | 248968 | afterok:248967 | queued |
| 32B p3 sparse slot B (insurance) | 248969 | afterok:248968 | queued |

Reference rate: oracle 32B p3 completed 500 steps in a single 24 h slot on
4×B200 (07-31), so slot B should exit-0 instantly in the normal case.

## 3c. Queue pivot (2026-08-01 evening) — b200-batch effectively closed to us

After the masks completed, both 24 h training asks sat with no start estimate:
b200-batch had ~600 pending 1-GPU/24 h jobs from a single user with **8× our
fairshare** (sprio: 0.82 vs our 0.103 — Irene's usage is 89.5% of the account's
4.3M units, and Age weight is 10 vs FairShare 1000, so waiting doesn't help).
Hundreds of 1-GPU jobs also eat every backfill gap a 4-GPU ask could use.

Moves (skill: honest `--time` + separate pools):
- **8B → b200-devel** (13 GPUs free, our devel quota 0/2): 3× 4 h slots with
  `TRAIN_TIMEOUT=13200` + resume — **249086 started in seconds** on b0029;
  chain 249086 → 249087 → 249088.
- **32B → rtx-batch** (separate 32-GPU QOS pool, 27 pending small jobs, no
  fairshare flood): 6× RTX_PRO_6000 (573 GB ≥ ~480 GB need; sm_120 kernel path
  validated by probe 238642), 4× 12 h slots `TRAIN_TIMEOUT=41400`, chain
  249103 → 249104 → 249105 → 249106. Slurm start estimate at submit:
  2026-08-02T02:00 (backfill may pull it earlier). RTX step time is unmeasured
  for this workload — re-measure from the first slot's log; 4 slots budget
  48 h wall for an expected ~24-36 h of steps.
- All superseded/duplicate-output chains put on `scontrol hold`, NOT cancelled
  (outdir-collision guard; disposition is Irene's call): 248913/248914 (8B 24 h
  batch), 248968/248969 (32B 24 h batch), 249089/249090/249091 (32B 12 h batch).
- **Race (2026-08-01 ~21:00, Irene asked to accelerate 32B)**: the 12 h b200
  chain 249089-91 was RELEASED to race the rtx chain — B200 is 1.5–3× faster
  (103–130 s/it measured vs ~250 s/it assumed RTX), and at the rtx estimate
  (start 02:00, worst-case ~300 s/it) the finish brushes the 08-03 deadline.
  A 60 s arbitration watcher holds the losing chain the moment either primary
  (249089 / 249103) starts; both need ~25 min before the first checkpoint
  write (step 10), so the window is collision-safe.
- **249103 OOM'd on 6×RTX** (backfilled at 20:29, died 8 min in): OOM at
  SparseAdamW `state.to(device)` — `device_map=auto` packed GPU 0 to
  94.89/94.97 GiB before the optimizer states landed. The ~480 GB estimate
  ignored placement imbalance + activation headroom; 573 GB total is NOT
  enough at 6 GPUs. Probe 238642 only validated the kernel, not full-model
  memory. Chain froze cleanly (249104-06 CANCELLED, no checkpoint written).
  **RTX route for 32B p3 requires the full 8-GPU node (765 GB).**
- **Race round 2**: rtx 8-GPU chain 249169 → 249170 → 249171 (12 h slots) vs
  the re-released b200 4-GPU chain 249089-91; same 60 s arbitration guard.

## 3b. 248909 failure: float32 tie collisions in the chunked selector

`ValueError: Chunked selector only found 9920 boundary candidates for 11715
required positions` (`mask_utils.py:423`), 13 min in, MaxRSS 342 GiB (not OOM).

Mechanism: 32.76B uniform float32 scores put ~1950 exact duplicates on every
representable value near the keep threshold (ulp ≈ 6e-8 at 0.975). The
refinement pass bins scores with `torch.histc` while the gather pass compares
with `>= lo - 1e-9` — the epsilon vanishes when cast to float32, so the two
passes classify the boundary-tie mass differently and the exact-count invariant
breaks by ~1800 ≈ one tie group. The 8B mask (4× fewer params) and every |Δ|
oracle score (wide dynamic range, no uniform pile-up) never hit this.

Fix: `--float64_scores` flag added to `random_mask_baseline.py` (opt-in; the
proven float32 path is untouched for ≤8B reproducibility). Metadata now records
`score_dtype`. The shared selector in `mask_utils.py` (Scott's primitive) was
NOT modified. Mask job memory raised to 800G for the ~262 GB fp64 score dict.

## 4. Read-out

Same as the June 8B chain: dense vs oracle vs random `rewards/margins` on
light-r1. The rebuttal claim being closed: random @ρ=97.5% is the floor the
oracle must clear on **both** Qwen3 backbones. Wandb run IDs to be backfilled
into `FIGURE_RUN_REGISTRY.md` when the runs come up.
