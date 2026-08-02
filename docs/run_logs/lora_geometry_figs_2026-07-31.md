# LoRA geometry figures — full-rank vs low-rank metric + win plots

**Date:** 2026-07-31 · **Branch:** `irene-rebuttal-lora`
**Request:** Nadim, 2026-07-31 — "we need some metric to further demonstrate why having
low-rank updates is not great whilst full-rank updates is … the rewards plot won't help
with that. Can you create some plots with some of the results where our method wins?"

**Figures:** `docs/paper_drafts/lora_{stable_rank_spectrum,energy_capture_bound,margin_displacement}.{png,pdf}`
**Script:** `scripts/plot_lora_geometry_rebuttal.py` (no cluster access needed — reads repo-tracked JSONs)
**Data:** `docs/run_logs/delta_analysis_2026-07-25/` (Explorer job `8725606`, pulled into
repo 2026-07-31) and `docs/run_logs/pref_eval_2026-07-27/` (jobs `8727254`, `8763502`, `8787043`).

---

## 1. The metric

**Budget-matched energy capture of the dense update, per matrix.** For each dense-run
ΔW, put two numbers on one axis:

- **our prior (measured):** fraction of ‖ΔW‖² held by that matrix's top-2.5% coordinates
  — the per-layer concentration curve already computed by `src/analysis/delta_analysis.py`;
- **the low-rank ceiling (provable):** the best rank-r approximation captures
  Σ_{i≤r} σ_i² ≤ r·σ₁² = (r / stable rank)·‖ΔW‖²_F (Eckart–Young), so at the LoRA
  baseline's r=64 no adapter update — however trained — can exceed 64/srank.

Medians over the 47 measured linear matrices (embed excluded — the r=64 all-linear
baseline does not adapt it):

| dense checkpoint | sparse top-2.5% capture | rank-64 ceiling |
|---|---|---|
| DPO · Tülu-3 (Llama-8B) | **1.000** | 0.554 |
| GRPO · Math-220k (Llama-8B) | **0.915** | 0.309 |
| DPO · Light-R1 (Qwen3-8B) | **1.000** | 0.355 |

This survives the two methods tying on accuracy (they don't tie — see §3): it is a claim
about what each parameterisation *can represent* of the update that dense RL actually
makes, not about which trained arm scores higher.

## 2. The three figures

1. **`lora_stable_rank_spectrum`** — sorted per-matrix stable rank ‖ΔW‖²_F/‖ΔW‖²₂,
   three checkpoints, vs the r=64 line. Medians 132 / 229 / 180; 32, 42, 30 of 48
   matrices exceed the cap. GRPO is the *most* full-rank of the three.
2. **`lora_energy_capture_bound`** — the §1 metric: scatter of sparse capture vs stable
   rank, with the analytic rank-64 ceiling curve and the region above it shaded as
   unreachable. Sparse dots sit at 0.78–1.00 regardless of rank; the ceiling under the
   median matrix is 0.31–0.55.
3. **`lora_margin_displacement`** — held-out margin vs −r̄_chosen (Tülu-3 tail, n=500).
   LoRA LR ×20 ⇒ margin ×3.0 riding on displacement ×3.1 with accuracy −0.004
   (likelihood displacement, Razin et al. ICLR 2025); sparse arms sit at −r̄_chosen
   ≈ 0.31 vs LoRA's 1.12–3.48. This defuses "LoRA's margin is 3× larger", which the
   margin table alone invites.

## 3. Caveats that must travel with the figures

1. **LoRA wins held-out preference accuracy** (0.618/0.614 vs oracle 0.542) and the
   replies already commit to reporting that. None of these figures is a quality-win
   plot; §1–2 are representational-geometry claims and fig 3 is a metric-validity claim.
2. **The rank-64 ceiling is an upper bound that favours LoRA** — exact only when the top
   64 singular values are all equal; the true best-rank-64 capture is lower. The exact
   version is in flight (§4).
3. **Depth sample.** Job `8725606` ranked only the first 48 matched 2-D tensors
   (embed + layers 0–6). Full-depth (all 226) version in flight (§4).
4. **Budget matching is approximate at the model level:** sparse ρ=97.5 trains 200.76M
   (2.5%), LoRA r=64 all-linear 167.8M (2.05%) — sparse holds ~20% more trainable
   params. Per matrix the exact budget-matched rank is r≈51 (4096²), r≈20 (1024×4096),
   r≈80 (4096×14336); r=64 is what the baseline actually ran.
5. **Task mismatch in the delta trio:** none of the three measured checkpoints is
   Llama-8B DPO Light-R1 (deleted pre-retrain at the time), which is the LoRA arm's
   task. The matched-task delta is arm 0 of the in-flight job (§4).
6. Fig 3 is cross-dataset (trained Light-R1, evaluated Tülu-3 tail), n=500, n=1 seed per
   arm; contamination verdicts are recorded inside each pref_eval JSON.

## 4. COMPLETED same day — AICR job `243173` (cpu, array 0-1, ~9 min/arm)

Both arms COMPLETED 2026-07-31 ~17:11/17:21 EDT; JSONs pulled into
`docs/run_logs/delta_spectrum_2026-07-31/` and **`lora_energy_capture_exact`** generated
(now the headline figure — the bound version is kept as the 3-checkpoint view). Exact
numbers over all 224 linear matrices (embed + lm_head excluded):

| arm | stable rank median | above r=64 | sparse top-2.5% capture | **exact best-rank-64 capture** |
|---|---|---|---|---|
| DPO Light-R1 (the LoRA arm's task) | 186 | 210/224 | 1.000 | **0.101** |
| GRPO Math-220k | 252 | 218/224 | 0.921 | **0.074** |

So the Eckart–Young ceiling (~0.34/0.25 at these medians) was generous to LoRA by ~3×:
the best rank-64 approximation of the update dense RL actually makes holds **~10% (DPO)
/ ~7% (GRPO)** of its energy, vs 100%/92% for the same budget spent on coordinates.
This closes caveats 2, 3 and 5 of §3; caveats 1, 4 and 6 still stand. Note the exact
capture is the ceiling for *representing* this update — a LoRA trained from scratch is
not guaranteed to reach even that.

Original job description follows.

`scripts/aicr_delta_spectrum.sbatch`: `delta_analysis.py` with `--rank_max_tensors 226
--spectrum_topk 256` (new flag; records top-256 σ² per matrix via randomized SVD —
smoke-tested locally: planted rank-4 delta → capture 1.000, random delta → 0.265 vs
bound 0.303).

- arm 0: **llama_dpo_lightr1** — θ_T = the 07-28 dense retrain ckpt-500 (mask source of
  the whole ρ sweep). Closes caveats 3 and 5, and upgrades the bound to exact (caveat 2).
- arm 1: **llama_grpo_math220k** — same upgrade for the GRPO point.

Outputs `$S/rebuttal_analysis/delta_llama_{dpo_lightr1,grpo_math220k}_spec.json`; pull
into `docs/run_logs/` on completion and regenerate fig 2 with exact capture curves
(extend the plot script to read `sigma_sq_topk` when present).

## 4b. 2026-08-01 update — bound figure is now all-Llama

`lora_energy_capture_bound` regenerated: the DPO Light-R1 points are now the **Llama-8B**
full-depth delta from AICR job `243173` arm 0 (224 linear matrices, embed + lm_head
excluded) instead of the Qwen3-8B 48-tensor sample; "(Llama-8B)" moved from the legend
into the title; Eckart–Young formula annotation removed; retitled "The RL update fits in
2.5% of weights, but not in rank 64". New Light-R1 medians: sparse capture 1.000,
rank-64 ceiling 0.345 (was 0.355 on Qwen3). §1's table and caveat 5 of §3 describe the
pre-update version. The Qwen3 view survives in git history and in `delta_qwen3_dpo_lightr1.json`.

## 5. Related

- `docs/paper_drafts/REBUTTAL_REPLIES.md` — bf6h W3 / AC item 2 reply these figures back
- [dpo_rho_sweep_retention_2026-07-30.md](dpo_rho_sweep_retention_2026-07-30.md) — the ρ-sweep the LoRA arms are compared against
- [DO_NOT_REPEAT.md](DO_NOT_REPEAT.md) §"48.3 s/step" — wall-clock parity facts for any speed claim near these figures
