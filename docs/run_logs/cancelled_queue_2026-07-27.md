# Queue cancelled 2026-07-27 — all six pending jobs

**Action:** `scancel 8734159_1 8761309 8761310 8761311 8761312 8769965`
**Time:** 2026-07-27 ~18:46 EDT · **Queue after:** empty (0 jobs)
**Instructed by:** Irene, explicitly, choosing "cancel all six" over three narrower options.
**Nothing was running** — all six were PENDING, so no partial work or GPU-hours were lost.

Every script survives in git. Each job below can be brought back with one `sbatch`.

---

## What was cancelled

| job | script | resources | what it was for |
|---|---|---|---|
| `8734159_1` | `scripts/grpo_matched_schedule.sbatch` (array elem 1) | gpu,multigpu · h200:1 · 4 h | sparse arm of the schedule-matched GRPO pair |
| `8761309` | `scripts/grpo_tsweep_dense.sbatch` | gpu,multigpu · h200:1 · 8 h | fresh dense GRPO trajectory keeping early checkpoints |
| `8761310` | `scripts/grpo_tsweep_masks.sbatch` | short · CPU · 6 h | build warm-start masks at T=50/100/150/200/250 + oracle T=500 |
| `8761311` | `scripts/grpo_tsweep_compare.sbatch` | gpu · gpu:1 · 5 h | Jaccard / CKA / tensor-class across those masks |
| `8761312` | `scripts/grpo_tsweep_launch_sparse.sbatch` | short · 1 cpu · 12 h | launcher that would have submitted the 4-arm sparse array |
| `8769965` | `scripts/grpo_random_control.sbatch` | multigpu,gpu · h200:1 · 8 h | random-mask sparse GRPO control for the 2026-04 arms |

Dependency wiring that was in force: `8761309 → 8761310 → {8761311, 8761312}` (`afterok`).
`8734159_1` and `8769965` were independent.

## To resubmit

```bash
cd /home/xie.yiyi/rc-sparse-speed
sbatch scripts/grpo_random_control.sbatch                      # 8769965 equivalent
sbatch --array=1 scripts/grpo_matched_schedule.sbatch          # 8734159_1 equivalent (arm 1 only; arm 0 is done)
# T-sweep chain, in order:
D=$(sbatch --parsable scripts/grpo_tsweep_dense.sbatch)
M=$(sbatch --parsable --dependency=afterok:$D scripts/grpo_tsweep_masks.sbatch)
sbatch --dependency=afterok:$M scripts/grpo_tsweep_compare.sbatch
sbatch --dependency=afterok:$M scripts/grpo_tsweep_launch_sparse.sbatch
```

**Before resubmitting `8734159_1`, note the 2026-07-27 clipping fix is now live** in
`src/full_training/sparse_grpo_bsr.py` (`max_grad_norm` is wired into `GRPOConfig`). Jobs
read `src` at launch, so a resubmission picks it up automatically. Gate on
`training_args.max_grad_norm == 0.1` before trusting the arm against dense `8734159_0`.

**Before resubmitting `8769965`,** its gate is unchanged: final `learning_rate` must be
**1.111e-08**, or the 2026-04 schedule did not replicate and it is not a valid control.

## What this costs the rebuttal

**1. The one outward promise had to be withdrawn.** bf6h's reply said a matched sparse-GRPO
rerun was in progress and committed to posting the number in-thread. That is now false, so
the paragraph was rewritten to report only the completed dense arm and to say plainly that
the sparse arm is not run. **Never leave a promise in a posted reply for work that has been
cancelled** — the reply text and the queue must agree.

**2. The random-mask GRPO control still does not exist.** The limitation already disclosed
in three places stands as written: the three 2026-04 sparse GRPO arms establish *that*
sparse GRPO trains at ρ=97.5% and can never establish *which mask is better*. No reply
needs editing for this — they were written honestly in the first place.

**3. AC item 3 keeps a hole.** The T-sweep was the answer to "how early is the GRPO
subnetwork decided", the GRPO analogue of Fig. 5's DPO sweep. The AC comment already says
"On scoring functions we state the coverage grid with its holes rather than imply
completeness", which remains accurate.

## What is unaffected

Everything already measured, which is most of the response:

- LoRA at both learning rates (8727274_0, 8727274_2) and both held-out evals (8763502,
  8787043) — the margin/accuracy/displacement table in W3 and the AC comment.
- The three 2026-04 sparse GRPO arms and their accuracy-reward rises.
- Dense arm `8734159_0` of the matched pair (final LR 6.769e-10, reward +0.1104).
- Update-mass concentration, stable rank, mask composition, k-sweep Jaccard, the
  optimizer-step microbench, and every number in `results.tsv`.

## Related

- `DO_NOT_REPEAT.md` — the clipping bug, the kernel fork, the cap-1024 truncation regime
- `docs/paper_drafts/REBUTTAL_REPLIES.md` — status header records this cancellation
- `docs/run_logs/grpo_random_control_2026-07-26.md` — full spec of the cancelled control
- `docs/run_logs/grpo_tsweep_2026-07-26.md` — full spec of the cancelled T-sweep chain
