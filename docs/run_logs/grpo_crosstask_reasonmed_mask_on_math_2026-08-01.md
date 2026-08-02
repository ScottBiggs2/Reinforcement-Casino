# Cross-task GRPO arm: ReasonMed oracle mask → math-220k training

**Date:** 2026-08-01 · **Branch:** `irene-rebuttal-lora` · **Cluster:** AICR `b200-batch`
**Jobs:** `246716` (TIMEOUT at 12h, step ~270) → `248421` (dependency continuation,
resumed checkpoint-250, COMPLETED 500/500, 17:56 EDT)
**Script:** `scripts/aicr_grpo_matched_sparse.sbatch` (rc-sparse-speed worktree) with
`MASK_PATH=oracle_grpo_reasonmed_step500_sp97.5.pt RUN_TAG=oracle_reasonmed`
**wandb:** run `n1collz0` (`grpo_matched_aicr_oracle_reasonmed_500steps_cap2048`) —
continuation resumed the SAME run via the saved `wandb_run_id.txt`, one continuous
curve. Logged to `rl_casino_rebuttal` (the sbatch default), **moved to `g-rpo_group`
2026-08-01 evening** (same `moveRuns` mutation as `wandb_reorg.py`) to sit with every
other GRPO run; `rl_casino_rebuttal` is empty again.

---

## 1. What this arm is

The cross-task transfer test in the GRPO direction: the mask distilled from **medical**
GRPO training ([grpo_reasonmed_mask_fig2_2026-08-01.md](grpo_reasonmed_mask_fig2_2026-08-01.md)
§4, ρ=97.5, 2.5000% exact) applied to **math-220k** GRPO training. Complements the
Jaccard-level claim (cross-domain overlap ≈ rerun ceiling, §6b there) with a
training-dynamics measurement: does the medical subnetwork *train* on math?

## 2. Hyperparameters

Verbatim the matched-schedule sparse spec of
[grpo_matched_aicr_2026-07-29.md](grpo_matched_aicr_2026-07-29.md) §2 — Llama-3.1-8B-Instruct,
500 steps cosine warmup 0.1, lr 5e-6, β 0.025, clip 0.1, 8 gen, per-device 2 × accum 4,
prompt 512 / completion cap 2048, `llama_cot` rewards, adamw_8bit bf16. Only the mask
file differs from the `oracle_grpo_math` arm.

## 3. Timeline / provenance

| job | node | window | outcome |
|---|---|---|---|
| 246716 | b0021 | 08-01 03:15 → 15:15 | TIMEOUT at 12h, step ~270 of 500; ~156 s/it; last checkpoint 250 |
| 248421 | b0014 | 08-01 15:17 → 17:56 | `--dependency=afterany:246716`, `--resume_from_checkpoint auto` → checkpoint-250; 500/500 in 2h33 at ~36.5 s/it |

Checkpoints (400/450/500 retained) + `run_manifest.json` at
`/scratch/$USER/rebuttal_analysis/grpo_matched_aicr/oracle_reasonmed/grpo_matched_aicr_oracle_reasonmed_500steps_cap2048/`.

**Open anomaly — 4.3× step-time gap between the two segments** (156 vs 36.5 s/it, stable
within each segment from step 1, same script/config/dataset). Not diagnosed. Candidate:
node-level contention on b0021 (the nine 8h-TIMEOUT `eval_tulu3` jobs 246641–246649 ran
03:00–11:00 in the same window). Implication: do NOT use this run's wall-clock for any
speed/throughput claim; per-step timing is only trustworthy from the 248421 segment.

## 4. Result

Accuracy reward first-50 **0.0775** → last-50 **0.1125** (+0.035); both format rewards
at ceiling throughout (total reward last-50 ≈ 1.07); KL ~0.002, healthy grad norms.

Reference points: math **dense** arm last-50 accuracy 0.0925, within-run learning
+0.016 (fig2 log §3). The medical-mask arm therefore learns math at least as well as
dense trains it — the cross-domain mask costs nothing detectable in training dynamics,
matching the Jaccard prediction. Caveats: math-220k accuracy signal is weak everywhere
(dense +0.016, this +0.035 — both small vs step-to-step noise where per-step accuracy
bounces 0–0.25 on 8 samples); a real claim needs the eval-suite pass on checkpoint-500,
not train-reward deltas.

## 5. Figure (2026-08-01 evening)

Added as the fourth curve of `fig3_grpo_transfer_curves.{png,pdf}`
(`scripts/plot_grpo_transfer_curves.py`, arm key `oracle_reasonmed`, **black** — the
colour ReasonMed carries in the Figure-2 pair line). Train-reward EMA comparison,
mean of last 50 steps:

| arm | last-50 reward |
|---|---|
| oracle_reasonmed (GRPO cross-DOMAIN mask) | **1.0737** |
| oracle_grpo_math (in-task mask) | 1.0550 |
| oracle_dpo_tulu3 (cross-objective mask) | 1.0237 |
| random_seed42 | 0.9862 |

The medical-GRPO mask trains math *at least as well as* the in-task math mask —
train-reward corroboration of §6b's Jaccard claim (cross-domain ≈ rerun ceiling).
Run mapping recorded in `docs/paper_drafts/FIGURE_RUN_REGISTRY.md`. Caveat as ever:
train reward, not held-out eval; GSM8K held-out separates none of these arms
(grpo_matched_aicr_2026-07-29.md §4c).

## 6. Next

- Eval checkpoint-500 with the same suite as the other matched arms.
- If wall-clock ever matters, rerun the speed measurement on a quiet node (see §3 anomaly).
