# GRPO on ReasonMed → oracle mask → Figure 2 cross-domain cell

**Date:** 2026-07-31 → 08-01 · **Branch:** `irene-rebuttal-lora` · **Cluster:** AICR (b200)
**Request:** Irene — swap the GRPO "RLVR" run's dataset for lingshu-medical-mllm/ReasonMed,
retrain, rebuild the mask-compare figure (Figure 2) against the OpenR1-Math mask.

**Figure:** `docs/paper_drafts/fig2_jaccard_cka_v3.{png,pdf}`
**Scripts:** `scripts/prep_reasonmed_mcq.py` · `scripts/aicr_grpo_dense_reasonmed.sbatch` ·
`scripts/aicr_mask_grpo_reasonmed.sbatch` · `scripts/plot_fig2_jaccard_cka.py`

---

## 1. Headline

**GRPO⇄GRPO subnetwork overlap does not care about domain.** The math⇄medical pair
(OpenR1-Math ⇄ ReasonMed) lands at Jaccard **0.2966** — statistically the same as the
math⇄math pair (OpenR1 ⇄ Tulu3RLVR, **0.3051**) and 23× above chance (0.0127):

| pair | Jaccard | reading |
|---|---|---|
| DPO ⇄ DPO (LightR1/Tulu3) | 0.7930 | same objective, ~unchanged subnetwork |
| **GRPO ⇄ GRPO, math ⇄ math** | **0.3051** | same objective, different dataset |
| **GRPO ⇄ GRPO, math ⇄ MEDICAL** | **0.2966** | same objective, different DOMAIN — barely moves |
| DPO ⇄ GRPO (any) | 0.094–0.129 | crossing the objective costs ~3× more overlap |
| anything ⇄ random | 0.013–0.018 | chance |

So the ordering is **objective ≫ dataset/domain**: which RL algorithm you train decides
most of which weights move; what data you train it on — even math vs medicine — barely
shifts the picked subnetwork. This is a stronger version of the Figure-2 story than the
Tulu3RLVR cell alone, because Tulu3RLVR was still math (GSM8K+MATH rows).

CKA for the new pair is 0.1318 — in the same band as the DPO⇄GRPO pairs (0.114–0.134).
Note the GRPO⇄GRPO-Tulu3 CKA remains **0.0000** (known unresolved artifact,
`aicr_diag_cka_zero.sbatch`); the new pair returning a normal nonzero value is further
evidence that 0.0000 is an artifact of that pair's computation, not a real signal.

---

## 2. What ReasonMed is and is NOT — required disclosures

ReasonMed (1.11 M rows) ships `instruction`/`input`/`output` and **no answer column**;
`input` is empty everywhere. It is an SFT CoT dataset, **not RLVR-ready**, and this run
must not be labelled RLVR.

`scripts/prep_reasonmed_mcq.py` recovers a GRPO target by parsing "the correct answer is
⟨text⟩" out of the dataset's own CoT and matching it against the lettered options in the
instruction; rows kept only on a unique match.

1. **The reward is agreement with the teacher CoT**, not a gold label — ReasonMed does
   not carry the MedQA/MedMCQA/PubMedQA answer keys. Distillation-flavoured signal.
2. **Keep rate 12.5 %** over a shuffled stream (20,000 kept / 160,151 scanned) — a biased,
   cleaner-question subset. (The unshuffled head of the dataset gives 40 %; never size
   from an offset-0 sample.)
3. Answers are emitted as **1-based option numbers**, not letters, so
   `format_number_reward` stays live and the reward composition matches the math arm.
4. Prompts end with an explicit `Final Answer: <number>` format instruction because
   `_split_cot_heuristic` does not recognise "the answer is" — without the marker,
   `format_reasoning_reward` pays 0 on every rollout and the arm optimises a differently
   shaped objective than the math arm (measured on math dense, last-50: accuracy 0.0925,
   format_number 0.4775, format_reasoning 0.4625).

Dataset artifact: `/scratch/$USER/datasets/reasonmed_mcq/data.parquet`, 20,000 rows,
`{prompt, solution}`, target distribution 1:5631 / 2:4918 / 3:4291 / 4:5093 / 5:67.

---

## 3. Training — job 243101, COMPLETED 07-31 20:42, 3h03m36s, b0027

Hyperparameters copied **verbatim** from the math-220k dense arm
(`aicr_grpo_matched_dense.sbatch`); dataset is the only variable:

| Parameter | Value |
|---|---|
| model | `meta-llama/Llama-3.1-8B-Instruct` |
| steps / schedule | 500, cosine, warmup_ratio 0.1 |
| lr / β / clip | 5e-6 / 0.025 / max_grad_norm 0.1 |
| generations / batch | 8 gen, per-device 2 × grad-accum 4 |
| lengths | prompt 512 (measured p99 = 181 tokens — no truncation), completion 2048 |
| rewards | `llama_cot` (accuracy + format_number + format_reasoning) |
| optimizer | `adamw_8bit`, bf16 |
| wandb | `rl_casino_rebuttal` / run `qxx9sk64` |

Training behaviour: accuracy 0.135 (first 50) → ~0.65–0.72 plateau from step 150; both
format rewards at ceiling; KL 0.007–0.013; saturated groups (`frac_reward_zero_std`)
14–20 % — 80 %+ of steps carried gradient. Learning magnitude (+0.5 accuracy) is ~30×
the math arm's (+0.016): medical MCQ is far more learnable, worth a caption note since
the two arms' Δθ come from different-strength signals.

## 4. Mask — job 243134, COMPLETED 07-31 21:37, 3m29s

`checkpoint_diff_mask_finder.py`, base vs checkpoint-500, ρ=97.5, global pooling,
`min_layer_keep_ratio 0.0025`, `--force_cpu` — identical construction to the OpenR1-Math
paper mask. Output `oracle_grpo_reasonmed_step500_sp97.5.pt`: kept 200,756,531 /
8,030,261,248 = 2.5000 % exactly; 291/291 tensors align with the math mask.

## 5. Figure build — and the missing paper-mask incident

The planned 6-mask recompute (`aicr_fig2_jcka_v2.sbatch`, job 245273) **failed its MD5
gate in 5 s**: `oracle_dpo_lightr1_step500_sp97.5_paper.pt` and
`oracle_dpo_tulu3_step500_sp97.5.pt` are **gone from AICR**
`transfer_v1/oracle_masks_llama8b/` (present for job 240279 on 07-30; dir mtime 07-31
22:28; several mask jobs from a parallel session ran that evening; cause not chased).
**Originals verified intact on Explorer** with md5 exactly matching the gate values
(`4d8b5cc2…` lightr1, `9d0d8af8…` tulu3). Explorer→AICR direct ssh is key-denied;
re-push needs a local relay (~16 GB) — left as a follow-up, not needed for the figure.

Resolution that avoids the loss entirely: job **245507** (3m17s) computed per-layer
Jaccard+CKA for **only the new pair** (both masks present), and its output was merged
into the intact `fig2_jcka_240279/multi_mask_jaccard_cka.json`. Every pre-existing line
in the figure is therefore **bit-identical to the 07-30 computation against the
paper-era masks** — provenance preserved; a full recompute would have silently swapped
the old lines onto regenerated masks. Merge sanity: same model, n_samples 64, same
`tulu3` calibration set; pair aggregate matches the mask job's independent global
Jaccard (0.2966).

Figure grammar: the new line is **black, densely dotted** — dotted because it sits
within 0.01 of the crimson GRPO⇄GRPO-Tulu3 line across all 32 layers and the two merge
into one dark band in grayscale/CVD; dotted ≠ dashed, so the "dashed = ⇄random" cue is
untouched.

## 6. Reproduce

```bash
# local
scp aicr:/scratch/$USER/rebuttal_analysis/fig2_jcka_240279/multi_mask_jaccard_cka.json base.json
scp aicr:/scratch/$USER/rebuttal_analysis/fig2_pair_reasonmed_245507/multi_mask_jaccard_cka.json pair.json
python - <<'PY'   # merge (see this log §5)
import json
base, pair = json.load(open('base.json')), json.load(open('pair.json'))
k = "GRPO-OpenR1 ⇄ GRPO-ReasonMed"
base["jaccard"][k], base["cka"][k] = pair["jaccard"][k], pair["cka"][k]
base["masks"] += [m for m in pair["masks"] if m["label"] == "GRPO-ReasonMed"]
json.dump(base, open('merged.json', 'w'), ensure_ascii=False)
PY
python scripts/plot_fig2_jaccard_cka.py merged.json docs/paper_drafts/fig2_jaccard_cka_v3
```

## 6b. The denominator — same-dataset rerun Jaccard (job 246449, 08-01)

Irene asked whether the two GRPO masks could be shown "as overlapping as the blue DPO
line". Raw top-2.5 % Jaccard says no (0.30 vs 0.79) — but the right question is what
GRPO's own reproducibility ceiling is. Probe: mask from the AICR matched-schedule math
dense run (235271) vs the paper math mask — same dataset, independent training run:

```
same-dataset rerun (math ⇄ math)      0.2989   [attn 0.3069  mlp 0.3152  norm 0.7358  other 0.1720]
cross-dataset      (math ⇄ T3RLVR)    0.3051   [attn 0.3119  mlp 0.3222  norm 0.7390  other 0.1658]
cross-domain       (math ⇄ medical)   0.2966   [attn 0.3034  mlp 0.3127  norm 0.7432  other 0.1740]
chance                                0.0127
```

All three coincide to ±0.009, bucket-by-bucket. **Changing the dataset — even the
domain — costs nothing beyond what rerunning already costs.** Normalized to the rerun
ceiling, cross-domain overlap is 0.2966/0.2989 ≈ 0.99; DPO by the same construction is
0.7930/0.9691 ≈ 0.82. So in normalized terms GRPO's subnetwork is MORE data-invariant
than DPO's — the defensible form of "the two GRPO lines coincide".

Contrast worth stating: DPO⇄DPO transfer is high in *absolute* overlap because DPO masks
are highly reproducible (0.97) AND transferable (0.79); GRPO masks are weakly
reproducible (0.30) but transfer at *zero marginal cost*. Two different regimes, both
pointing at objective ≫ data.

Confound note: the paper math run used the flat-LR schedule (cosine over 5000), 235271
used matched cosine-over-500, so 0.2989 is a **lower bound** on the same-config ceiling.
Closed by job 246456 (§6c): pairwise Jaccards among the three matched-schedule masks
(235271-math / Tulu3RLVR / ReasonMed), which carry {run ⊕ data} with no schedule term.

## 6c. Matched-schedule trio (job 246456) — confound CLOSED

Pairwise Jaccards among the three matched-schedule masks (cosine-500/cap-2048 runs:
235271-math, Tulu3RLVR, ReasonMed) — {run ⊕ data} with no schedule term:

```
math ⇄ Tulu3RLVR   (cross-dataset)  0.3099   [attn 0.3085  mlp 0.3190  norm 0.7220  other 0.2247]
math ⇄ ReasonMed   (cross-domain)   0.3028   [attn 0.3011  mlp 0.3104  norm 0.7261  other 0.2383]
T3RLVR ⇄ ReasonMed (cross-domain)   0.3079   [attn 0.3051  mlp 0.3169  norm 0.7277  other 0.2275]
```

With §6b's rerun (0.2989) and the two paper-anchored pairs (0.2966/0.3051), **all six
GRPO⇄GRPO Jaccards sit in [0.2966, 0.3099] — a 0.013 band** across every combination of
{schedule same/different, dataset same/different, domain same/different}. The 2×2 is
fully deconfounded: ~0.30 is the GRPO mask reproducibility constant at ρ=97.5 for this
model, invariant to everything varied.

Second-order detail, consistent in direction though tiny in size: the three
matched-schedule pairs (0.3028–0.3099) all exceed the two pairs against the flat-LR
paper mask (0.2966–0.3051), and cross-domain-same-schedule (0.3028) > same-data-
different-schedule (0.2989) — **a schedule change costs at least as much overlap as a
full domain change**. The embed/lm_head bucket moves the same way (0.17 vs paper → 0.23
matched).

## 6d. The task-universal core, measured directly (job 246466)

3-way intersection of the matched-schedule masks: **48,840,813 weights = 24.33 % of the
mask (0.61 % of the model) are selected by every GRPO run across math / math′ / medical**
— 389× the random 3-way rate (0.025² = 0.0625 %). Adding the flat-LR paper mask (4-way)
halves it again to 13.35 %, consistent with §6c's "schedule costs ≥ domain".

The binary core+churn model is **rejected**: it predicts 3-way ≈ pairwise-implied core
f = 0.46; measured 0.243 ≈ 0.46² instead. So membership is *graded* — a coordinate has a
stable selection propensity rather than a deterministic identity; a typical selected
weight re-appears in an independent run with p ≈ 0.5. Exception: the norm bucket is
near-deterministic (3-way 0.7535 vs pairwise ~0.73 — those coordinates barely churn).

Rebuttal-ready phrasing: "24 % of the GRPO subnetwork is a task-universal core selected
in every run regardless of dataset or domain; the remainder is run-stochastic with ≈50 %
re-selection probability; LayerNorm selections are 75 % persistent."

## 6e. CKA 0.0000 artifact RESOLVED — figure v4 (job 248502, 08-01 afternoon)

Root cause (diag job 240375 + fixed `mask_to_cka.linear_cka`, deployed in
rc-sparse-speed): the old implementation returned a hard 0.0 whenever
`sqrt(hsic_kk*hsic_ll) < 1e-30`. That denominator scales as ‖X‖²‖Y‖²; the
GRPO⇄GRPO activation deltas are ~1e-10, putting the HSIC terms near 1e-41, so
the guard fired on all 32 layers of the smallest-activation pair and on the
early layers of the other GRPO/random pairs. The fix centres features instead
of the Gram matrix and normalises by a scale-free Frobenius ratio (genuinely
zero-variance input → NaN, never a fabricated 0.0).

Job 248502 (7m34s, b0023) recomputed all six pairs among the four
non-DPO masks (OpenR1 / Tulu3RLVR / ReasonMed / Random-s42) with the fixed
code; Jaccard reproduced 240279/245507 bit-for-bit (merge sanity gate).
Fixed aggregates: OpenR1⇄Tulu3RLVR CKA **0.1370** (was 0.0000) — inside the
DPO⇄GRPO band (0.114–0.134) and next to OpenR1⇄ReasonMed 0.1318, exactly as
the artifact hypothesis predicted. Merged as
`$S/rebuttal_analysis/fig2_merged_v4.json` → figure
`docs/paper_drafts/fig2_jaccard_cka_v4.{png,pdf}`.

Caveat that must ride with v4: the five DPO-involving CKA lines are still the
07-30 old-code values (their recompute is blocked by the missing paper masks);
their occasional exact-zero early layers are the same guard artifact and will
shift slightly once recomputed. Per-layer Jaccard everywhere is unaffected.

## 7. Open

- Re-push the two paper masks Explorer→AICR (md5-gated) and record the disappearance in
  DO_NOT_REPEAT once the cause is known; then recompute the five DPO-involving
  CKA lines with the fixed `linear_cka` (§6e) so the whole panel shares one
  implementation.
- If the figure ships, caption must carry §2's disclosures (teacher-CoT reward, 12.5 %
  biased subset, not RLVR) and the learnability asymmetry from §3.
