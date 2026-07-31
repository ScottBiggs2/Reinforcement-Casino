# GRPO figure-extension set on AICR — Tülu3-RLVR dense + cross-task sparse arms + Fig 5 recompute

**Date:** 2026-07-30 · **Branch:** `irene-rebuttal-lora` · **Cluster:** AICR
**Jobs:** `239122` (dense GRPO, Tülu3-RLVR) · `239123` (sparse GRPO, DPO-LightR1 mask) ·
`239125` (Fig 5 mask-score-gap recompute, cpu) · third arm (DPO-Tülu3 mask) pending mask transfer.
**Requested by:** Irene — rebuttal figure revisions for NeurIPS #29841 (due 2026-08-03):
Figure 2 + a GRPO⇄GRPO cell, Figure 3 analogue with GRPO *training*, Figure 5 regeneration.

---

## 1. Why each job exists

1. **`239122` dense GRPO on Tülu3 RLVR.** Reviewer XD33: "the GRPO-OpenR1 ↔ GRPO-Tülu3
   cell does not exist because we never ran GRPO on Tülu3." This run creates the second
   GRPO subnetwork: checkpoint-500 → checkpoint-diff oracle mask (ρ=97.5) → Jaccard/CKA
   against `oracle_grpo_math220k_step500_sp97.5.pt` for the revised Figure 2.
2. **`239123` + pending arm — sparse GRPO under DPO-derived masks.** Figure 3 contains no
   GRPO training (XD33, conceded in the reply). These two arms complete the matched set
   {in-task oracle `235463`, random `235303`, dense `235271`} with cross-task mask origins,
   giving the true Figure-3 analogue: GRPO training, mask origin varies.
3. **`239125` mask-score-gap recompute.** Figure 5's source
   `mask_score_gap_histograms.npz` survives on neither cluster (2026-04-26 cleanup).
   Recomputed from the migrated Light-R1 dense retrain (`8792090`) under the submitted
   figure's settings; the replot then applies the reviewer-promised title + larger labels
   (reply item 15).

## 2. Hyperparameters

### 2a. Training arms (matched schedule, identical to grpo_matched_aicr_2026-07-29.md §2)

| field | value |
|---|---|
| model | `meta-llama/Llama-3.1-8B-Instruct` |
| steps / schedule | 500, cosine, warmup 50 (ratio 0.1), lr 5e-6 |
| β / clip / cap | 0.025 / global 0.1 / prompt 512, completion 2048 |
| batch | per-device 2 × accum 4, 8 generations |
| rewards | `llama_cot` (accuracy + format_number + format_reasoning) |
| optimizer | dense `adamw_8bit` (bnb preflight passed on B200) / sparse `sparse_adamw` |
| ρ (sparse arms) | 97.5% |
| wandb | project `rl_casino_rebuttal`, runs `grpo_matched_aicr_dense_tulu3rlvr_500steps_cap2048`, `grpo_matched_aicr_oracle_dpo_lightr1_500steps_cap2048` (+ tulu3-mask arm when submitted) |
| outputs | `/scratch/xie_yiyi_neu/rebuttal_analysis/grpo_matched_aicr/{dense_tulu3rlvr,oracle_dpo_lightr1,oracle_dpo_tulu3}` (30-day purge — pull `trainer_state.json` into repo on completion) |

**Dataset (dense arm only):** new registry key `tulu3-rlvr-math` =
`allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` filtered to `dataset ∈ {gsm8k, MATH}`:
29,946 → **14,973 rows** (IF rows dropped — their constraint verifiers are not
implemented; keeping them would zero their accuracy reward). `messages` → prompt,
`ground_truth` → solution; 2 empty solutions (ignored). Code:
`src/utils/dataset_registry.py` (row_filter support + role/content hardening in
`_extract_prompt`), sbatch `scripts/aicr_grpo_dense_tulu3rlvr.sbatch`.

**Disclosures that travel with the Tülu3-RLVR cell:**
- The DPO-side Tülu3 is the *preference mixture*; this cell is GRPO on **Tülu-3's RL
  (RLVR) data** — the faithful reading of "GRPO on Tülu3", but not the same rows.
- RLVR math overlaps in domain with OpenR1-Math (GSM/MATH-style vs competition math), so
  the "different data" contrast is weaker than DPO's Light-R1⇄Tülu3 pair. Caption must say so.
- RLVR prompts are 4-shot (p50 731 tokens, 100% > 512); TRL truncates **left**
  (`truncation_side="left"`, verified in trl 0.24.0), so exemplars are cut and the target
  question + CoT suffix survive — checked on decoded samples before launch.
- Step-≤6 telemetry: 23.3 s/it, `clipped_ratio` 0.875–1.0 (same truncation regime as the
  math arms).

**Masks (sparse arms):**
- `oracle_dpo_lightr1_step500_sp97.5_src-deltafull.pt` (AICR; gate vs paper mask Jaccard
  0.9691, oracle_sparsity_sweep_lightr1_2026-07-28.md §8b).
- `oracle_dpo_tulu3_step500_sp97.5.pt` — push from Explorer in flight, source MD5
  `9d0d8af824327778d1b8c88f0d6bebc2`; arm submits only after both-side MD5 match.

### 2b. Fig 5 recompute (`239125`, cpu partition, 32 cpu / 384 G)

`src/analysis/mask_score_gap_analysis.py` (restored from `origin/cav_fixes` onto this
branch; imports verified in the AICR env before submit) with: initial
`meta-llama/Llama-3.1-8B-Instruct`, final
`transfer_v1/dense_dpo_lightr1_llama8b/.../checkpoint-500`, milestones 50,100,150,200,
random seed 42, ρ=97.5, `CERT_TAU_RULE=global`, `CERT_GLOBAL_MODE=stream`, bf16,
verify deltas = same dir (`deltas_step_500.pt`). Out:
`/scratch/xie_yiyi_neu/rebuttal_analysis/mask_score_gap_lightr1/run_global_tau_239125`.
Provenance note: the dense source is the **07-28 retrain**, not the deleted paper run —
its ρ=97.5 mask recovers the published subnetwork at Jaccard 0.9691, disclosed if anyone
diffs the new Figure 5 against the submitted one.

## 2c. INCIDENT — two 8 GB masks vanished from AICR `/scratch` after arriving intact

**Unexplained. Recorded because a repeat would silently invalidate a figure.**

Timeline (2026-07-30, all EDT):

| time | event |
|---|---|
| ~18:55 | tulu3 mask push starts (`rsync --partial --inplace`) |
| ~19:05 | lightr1 paper mask push **completes**, 8,030,364,211 B verified on AICR |
| ~19:10 | tulu3 push dies at 82% — `rsync error code 12`, connection closed |
| 19:21 | restart with 8-attempt resume loop; every attempt ends in `Killed` (SIGKILL) after making partial progress — the login-node watchdog reaping the delta-algorithm resume, which re-reads and rolling-checksums the multi-GB partial each time |
| ~20:00 | switch to `rsync --append` (ships only the tail): `TRANSFER_DONE`, size verified **8,030,363,617 B = full** |
| ~20:01 | `md5sum` returns **`Stale file handle`**; seconds later `ls` reports **No such file or directory** for *both* today's files |

Established by measurement, not inference:

- Both files are gone from the whole filesystem (`find /scratch/xie_yiyi_neu` finds neither).
- **Not quota**: 3,308 G used of a 9,314 G limit; inode count 4,148 of 15 M.
- **Not the scripts**: the transfer and gate scripts contain `stat`, `md5sum`, `sbatch` and no `rm`.
- **Not a general storage failure**: a 400 MB file written to the same directory immediately
  afterwards survived (verified twice, minutes apart), then was cleaned up by hand.
- The entire 2026-07-29 mask set in the same directory is untouched. Only the two files
  created *today* disappeared.

Cause unknown. `Stale file handle` immediately before the disappearance points at the NFS
layer (`storage0001.nfs:/scratch`) rather than at anything user-space, so this is worth an
AICR support ticket if it recurs.

**Operational consequence — the rule this buys:** never let an md5 gate be the *first*
read of a freshly transferred multi-GB file, and never treat "rsync said DONE + size
matches" as arrival. The gate in `aicr_fig2_jcka.sbatch` did its job (it refused to run on
an unverified mask and the sparse arm was never submitted), which is the only reason this
cost time rather than a wrong figure.

**Recovery (same evening, both files now verified present).** Re-pushed with
`rsync -a --append` in a retry loop — `--append` ships only the missing tail instead of
re-checksumming the partial, which is what the watchdog kept killing. Then the new
procedure: warm read (`dd` of the first 8 MB) → 60 s settle → re-`stat` → `md5sum`.

| file | size | md5 | matches Explorer |
|---|---|---|---|
| `oracle_dpo_tulu3_step500_sp97.5.pt` | 8,030,363,617 | `9d0d8af824327778d1b8c88f0d6bebc2` | yes |
| `oracle_dpo_lightr1_step500_sp97.5_paper.pt` | 8,030,364,211 | `4d8b5cc255d8c31e0815766fa4121a3a` | yes |

Third sparse arm submitted on the strength of that check: **`239307`**
(`RUN_TAG=oracle_dpo_tulu3`). Transfer throughput when it is not being killed is
~88 MB/s, so a clean full push of one mask is ~90 s — the earlier hours were entirely
watchdog kills and the unexplained loss, not bandwidth.

## 2d. BUG — `linear_cka` silently reported 0.0 for small-magnitude activations

Found while checking the Figure 2 rerun (`240279`), which returned CKA exactly
`0.000000` on all 32 layers of GRPO-OpenR1 ⇄ GRPO-Tülu3RLVR while every other cell
looked normal.

**Cause.** `src/cold_start/mask_to_cka.py:linear_cka` guarded with
`if denom.abs().item() < 1e-30: return 0.0`. That denominator is
`sqrt(hsic_kk·hsic_ll)`, which scales as ‖X‖²‖Y‖². Applying a ρ=97.5% mask zeroes
97.5% of the weights, so pooled MLP activations come out at **1e-17 to 1e-10** and
the denominator lands near **1e-72** — six orders of magnitude past a threshold
meant to catch zero-variance inputs. Measured on the pair (diag job `240375`):
`X absmax 4.7e-17`, `Y absmax 6.1e-19`, `denom 3.1e-72`, all values finite and
distinct. Nothing was degenerate; the guard simply fired on valid data.

**Fix.** Centre the features rather than the Gram matrix (`Xc Xcᵀ ≡ H(XXᵀ)H`,
identical but without subtracting nearly-equal large numbers) and normalise by a
ratio of Frobenius norms, which is scale-free and needs no epsilon. A genuinely
zero-variance representation now returns `nan` (undefined) instead of `0.0`.
Verified before rerunning: over 300 well-scaled random pairs old and new agree to
**2e-15**, so previously published values are not restated by this change;
`CKA(X,X)=1.0`; and on a fixed pair scaled from 1e0 down to 1e-20 the old code
collapses to 0.000000 below ~1e-12 while the new code holds 0.902466 throughout.

**What it changed in practice** (same activations, `240279` buggy vs `240400` fixed):

| pair | layers the old code reported as 0.0 | their real values |
|---|---|---|
| DPO-LightR1 ⇄ DPO-Tülu3 | none | — |
| DPO-LightR1 ⇄ Random | none | — |
| DPO-Tülu3 ⇄ Random | none | — |
| DPO-LightR1 ⇄ GRPO-OpenR1 | 0,1,2,3,4,6 | 0.054, 0.049, 0.091, 0.194, 0.173 |
| DPO-Tülu3 ⇄ GRPO-OpenR1 | 0,1,2,3,4 | 0.045, 0.069, 0.080, 0.252 |
| GRPO-OpenR1 ⇄ Random | 0,1,2,3 | 0.108, 0.099, 0.129 |
| GRPO-OpenR1 ⇄ GRPO-Tülu3RLVR | **all 32** | 0.021 … 0.246 |

Aggregate CKA for GRPO-involving pairs rises ~0.02 (e.g. GRPO-OpenR1 ⇄ DPO-LightR1
0.114 → 0.136); pairs with no GRPO mask are bit-identical.

**This reaches the submitted paper.** Figure 2's CKA panel shows the DPO⇄GRPO and
GRPO⇄Random curves flat on zero across layers 0–4 — that is this bug, not a
measurement. The published claim ("substantially lower CKA across all layers") is
unaffected in direction, but the specific low-layer values are wrong and should be
restated from the corrected run. Layer 3 is the one genuinely undefined layer under
a GRPO mask and is now reported as such rather than as 0.0.

## 3. Downstream (pre-registered)

1. Dense `239122` completes → `aicr_mask_from_delta`-style checkpoint-diff mask
   `oracle_grpo_tulu3rlvr_step500_sp97.5.pt` (cpu job).
2. Jaccard/CKA job (`scripts/multi_mask_jaccard_cka.py`, b200-devel) over
   {DPO-LightR1, DPO-Tülu3, GRPO-OpenR1, GRPO-Tülu3RLVR, random s42} → per-layer lines →
   revised Figure 2 with the GRPO⇄GRPO pair added.
3. Sparse arms complete → `scripts/plot_grpo_transfer_curves.py` regenerated with all
   five curves (already produces the 3-arm draft).
4. `239125` completes → pull NPZ → `scripts/report_mask_score_gap_plots.py`-based replot
   with title + enlarged labels.

## 4. Related

- [grpo_matched_aicr_2026-07-29.md](grpo_matched_aicr_2026-07-29.md) — the matched set these arms extend
- [oracle_sparsity_sweep_lightr1_2026-07-28.md](oracle_sparsity_sweep_lightr1_2026-07-28.md) — mask provenance + gate
- [DO_NOT_REPEAT.md](DO_NOT_REPEAT.md) — why the 2026-04 transfer arms cannot be reused
