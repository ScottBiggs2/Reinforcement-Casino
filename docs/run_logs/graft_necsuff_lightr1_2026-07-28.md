# Grafting sufficiency/necessity — Light-R1 (rebuttal experiments A + B)

**Date launched:** 2026-07-28 ~23:40 EDT
**Branch:** `irene-rebuttal-lora`
**Requested by:** Irene ("帮我把1 2做了我看看" — execute the top-2 ranked interp
experiments from the three-agent research sweep)
**Jobs:** P `8818344` (prep) → A `8818348` (probes, afterok:P) + B `8818351`
(margin, afterok:P)
**Scripts (committed to repo):** `scripts/graft_prep_lightr1.sbatch`,
`scripts/probe_necsuff_lightr1_v2.sbatch`, `scripts/graft_margin_eval_lightr1.sbatch`,
`src/evaluation/apply_delta_patch_and_save.py` (new materializer)

## What this is

Replace the destructive `zero_out` probe evidence with **grafting** (Panigrahi
et al., ICML 2023, "Task-Specific Skill Localization in Fine-tuned Language
Models"): transplant the DPO update restricted to the mask onto the untouched
base model and measure what it carries.

- **B (rank 1, behavioral):** held-out DPO implicit-reward margin under
  θ_base + α·Δθ⊙M for M ∈ {oracle, ¬oracle, matched-random} and α ∈ {1, 0.5},
  vs the dense ceiling. Sufficiency = graft_oracle recovers most of dense's
  margin; necessity = anti ≈ 0; specificity = matched-random ≈ 0.
- **A (rank 2, representational):** the 05-03 Tülu-3 three-leg probe protocol
  mirrored on Light-R1, `delta_only` + `anti_delta_only` only (`zero_out`
  dropped — its record exists and the operator is destructive).

Chosen over alternatives by a three-agent sweep (rc-researcher +
general-purpose web + rc-planner, 2026-07-28): least attackable because both
are direct interventions with built-in nulls, and B closes the validation gap
against our own citation [28] (Mukherjee et al. verify by parameter agreement +
recovery, not probes; they also record that layer-norms barely update — the
external justification for retiring `zero_out`).

## Provenance decisions (the traps this run avoids)

1. **Self-consistent mask.** ORACLE = `oracle_dpo_lightr1_step500_sp97.5_src-deltafull.pt`
   built by job `8817043` from the **07-28 dense retrain** — the same
   checkpoint being probed/grafted. The orphaned Apr-27 mask (dense source
   deleted) is NOT used; it appears only in 8817043's Jaccard gate.
2. **Matched random controls.** New files
   `random_matched_lightr1_sp97.5_seed{42,43,44}.pt` generated via
   `random_mask_baseline.py --reference_mask <oracle>` → 291 tensors including
   all 65 layer-norms. The legacy `random_baseline_lightr1_sp97.5_seed42.pt`
   (290 tensors, no layer-norms; mask_composition finding 3) is retired from
   comparisons. New filenames chosen specifically to dodge the
   skip-if-exists reuse bug in the v1 sbatch (line ~95).
3. **Trained-pair exclusion.** P dumps `training_args.bin` via
   `scripts/read_training_args.py` (run_manifest.json untrustworthy per
   DO_NOT_REPEAT); B passes `--trained_on light-r1 --holdout_tail 2000`
   matching the frozen arm0 protocol.
4. **Preference probes stay off** (`--no_preference`): below chance on the
   unmasked baseline in the tulu3 study (v1 sbatch header); frozen hyperparams
   leave no legal fix before 08-03.

## Job shapes

| Job | Partition/gres | Mem | Time | Contents |
|---|---|---|---|---|
| P `8818344` | multigpu, bare gpu:1 (idle v100 pool) + keep-alive | 200G | 5h | gate on 8817043's "rho=97.5% done" line → anti-mask → matched randoms ×3 (~25 min each) → 4 grafted ckpts (~16 GB each to `/scratch/xie.yiyi/graft_ckpts_lightr1/`) → training_args dump |
| A `8818348` | a100:1 | 72G | 3h | probe_pair_masks, delta_only + anti_delta_only, frozen 05-03 hyperparams (C=0.01, cv 3, holdout 0.2, seed 42, stride 4, `--no_preference`) |
| B `8818351` | a100:1 | 96G | 2.5h | 7 margin evals: dense/graft_oracle/graft_anti/graft_random/graft_α0.5 on light-r1 + dense/graft_oracle on tulu3 (protocol frozen at arm0: tail 2000, n 500, β 0.1, len 1024) |

a100 pinned in A/B on the 8786294 post-mortem (v100-pcie has no bf16, 10×
slower). `short` unschedulable (2200+ pending, 43% nodes down), hence P on the
idle v100 pool with the playbook-7b sidecar. QOS after submission: 7/8 submit
slots. The 1e-12 jitter chain died at 22:37 (8814942 TIMEOUT, 0 masks; its
conclusion stands analytically — see probe_jitter_1e12_control_n5 log).

## Success criteria (pre-registered)

**B:** graft_oracle margin CI overlaps dense (report % recovered);
graft_anti and graft_random CIs overlap 0; graft_α0.5 between 0 and
graft_oracle (dose-response). Rebuttal sentence: "Patching the top-2.5%
coordinates of the dense DPO update into the untouched base model — no
training — recovers X% of the dense held-out reward margin; the complementary
97.5% and a density-matched random 2.5% recover ~0."

**A (05-03 criteria verbatim):** Oracle(delta_only) ≈ unmasked baseline;
Anti < Random-band − ε; Random ×3 clustered between. Failure of any leg
downgrades the claim to sufficiency-only and is reported as such.

**Honest-outcome note:** if graft_oracle recovers substantially less than
dense, that is a real finding (the update is NOT localized at 97.5%) and gets
reported, not buried — it would align with the GRPO-resists-masking appendix.

## Outputs

```
/scratch/xie.yiyi/graft_ckpts_lightr1/{graft_oracle_a1,graft_anti_a1,graft_random42_a1,graft_oracle_a05}/
/scratch/xie.yiyi/rebuttal_analysis/graft_margin/*.json   (B)
/scratch/xie.yiyi/probe_necsuff_lightr1_v2/{delta_only,anti_delta_only}/probe_pair_results.json  (A)
```
