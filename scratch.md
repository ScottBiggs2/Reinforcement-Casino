# Revision Guide — "A Few Good Weights" (NeurIPS Submission 29841)

**Purpose:** Drive a rewriting/revisions agent through the rebuttal-window revision of this manuscript.
**Status:** Rebuttal window just opened; time is short; scores are 2 / 2 / 3 with a negative metareview.
**Audience for output:** Reviewers XD33, jzn5, bf6h, and the AC — plus a human co-author who will review every change.

---

## 0. READ THIS FIRST — Hard guardrails

These are non-negotiable. Violating any of them makes the rebuttal worse than doing nothing.

1. **Never fabricate, estimate, or "fill in" experimental numbers.** Qwen3 (8B/32B) and additional GRPO results are being supplied by the human authors. Where results are needed but not yet provided, insert a clearly marked placeholder (`<<HUMAN: insert Qwen3-8B Jaccard value>>`) and structure the surrounding prose/tables so the number drops in cleanly. Do not guess a plausible value.

2. **Do not upgrade a claim the evidence doesn't support.** The single biggest strategic error in the current draft is claiming *downstream performance parity* from a table where dense, base, random, and sparse are all within noise. We are fixing this by **rescoping the claim**, not by inventing a better table. See §2.

3. **Compute is genuinely limited.** Do NOT add tasks that require new large training runs (extra seeds, ρ sweeps, more model families, LoRA head-to-heads, longer schedules). If a reviewer asks for one, the response is a scoping/reframe in the *rebuttal text*, not a new experiment in the manuscript. Flag any such ask for the human, don't attempt to satisfy it.

4. **Preserve anonymity.** No author names, institutions, or identifying repo details beyond the existing anonymized link.

5. **Every substantive change gets a margin note** (or a tracked-changes comment / `<!-- CHANGE: ... -->`) stating which reviewer point it addresses, so the humans can audit coverage fast.

6. **When in doubt between "impressive" and "honest," choose honest.** This paper's credibility is currently leaking through overclaims (§3). Tightening claims is a feature, not a retreat.

---

## 1. Strategic context (so the agent understands *why*)

The paper's load-bearing contribution — agreed by all three reviewers and the metareview — is:

> **Subnetwork structure is determined primarily by the learning objective, not the training data.** (DPO↔DPO high Jaccard/CKA across datasets; DPO↔GRPO near-zero Jaccard even on related data.)

This finding does **not** depend on 97.5% recovery working for GRPO. Protect it. Everything else (recovery curves, the optimizer, the theory) is supporting cast.

The reviewers' many complaints reduce to three roots:
- **A — Generality:** single model; GRPO "fails" at 97.5%; no PEFT baseline; thin ablations.
- **B — Does the method do anything downstream?** Table 1 shows everything within noise. *This is the most dangerous critique and the current draft walks into it.*
- **C — Claim scope/honesty:** "performance retention" bounds loss not downstream; "speedup" is optimizer-step not wall-clock; abstract speedup numbers don't map to Table 2; several math mis-statements.

Resource reality: A is answered partly by in-hand Qwen3/GRPO results and partly by honest scoping. **B is answered entirely by reframing — zero compute.** C is answered entirely by honest edits — zero compute. So the bulk of the win is free.

---

## 2. THE CENTRAL REFRAME (highest leverage, zero compute) — Problem B

**Current failure:** Table 1 is presented as the main downstream-performance comparison, with claims like "sparse models match dense training performance across all benchmarks." But on that table base ≈ dense ≈ random ≈ sparse within noise (random even beats dense DPO on GSM8K at 0.8180). Reviewers XD33 and bf6h both flagged "random performs on-par" / "not much change before and after fine-tuning." As written, the headline reads as *nothing matches nothing*.

**Root cause (state this honestly, do not hide it):** 500 steps at LR 5e-7 under the cited Tülu3/Light-R1 DPO recipe produces small downstream movement for *every* method, dense included. That is a property of the standard low-LR preference-tuning regime, not a defect of the sparse method.

**The fix — route the "selection matters" claim to evidence that already separates the conditions:**
- **Loss dynamics (Fig 1):** oracle & warm-start track dense; random diverges. Selection demonstrably matters *for the optimization objective*.
- **Choose/reject accuracy (Fig 7, App A.5):** direct readout of the preference signal — promote this from appendix toward the main narrative if space allows.
- **Linear probing (Fig 4, 6):** oracle/warm-start degrade math & factual capability more than random → selected parameters *carry* those capabilities. This is positive evidence of meaningful structure.
- **Certifiability margins (Fig 5):** warm-start ≈ oracle threshold. Selection is principled, not arbitrary.

**Concrete actions:**
1. **Demote Table 1** from "main performance comparison" to a supporting "at high sparsity, sparse training does not *degrade* downstream metrics relative to dense" — a *non-degradation / preservation* claim, which the table honestly supports, NOT a parity-implies-benefit claim.
2. **Rewrite §4.1 framing** so the primary evidence for "selection matters" is the loss curves + probes + choose/reject, with Table 1 as "and downstream metrics are preserved."
3. **Add one honest sentence** near Table 1 explaining the small absolute deltas: standard low-LR/short-budget preference tuning moves benchmarks little for all methods; the claim is preservation under 97.5% sparsity, and the *discriminative* evidence lives in the training dynamics and probes.
4. **Add per-benchmark averages / an average column** to Table 1 (jzn5 asked; free) and **state which dataset the dense baseline uses** (jzn5 flagged it's unclear).
5. **Do NOT** claim sparse *beats* dense (drop/soften the "implicit regularization" speculation on IFEVAL/MATH, or mark it explicitly as speculative on within-noise deltas — it currently overreads noise).

> Agent: draft the reframed §4.1 opening paragraph and the Table 1 caption, then stop and surface both for human review before propagating the framing elsewhere.

---

## 3. HONEST SCOPING & MATH CORRECTIONS (zero compute) — Problem C

All cheap, all credibility-restoring.

### 3.1 "Performance retention" → "loss retention"
- jzn5 W1 & metareview: Theorem 1 bounds a **loss gap**, not downstream performance.
- Rename the concept throughout to **loss retention** (or "training-loss retention"). Keep the theorem, retitle its framing.
- Add one sentence scoping it: the bound is on optimization loss; downstream preservation is an *empirical* observation (Table 1), not a consequence of the theorem.

### 3.2 Speedup claims — make them traceable and correctly qualified
- Abstract says "reduces optimizer step time … ∼1.5× and ∼3.3× speedups over PyTorch and HuggingFace 8-bit AdamW at 97.5% sparsity." XD33 correctly notes the **3.3× is at 99.75%, not 97.5%** (Table 2/3).
- Fixes:
  - State it is an **optimizer-step** speedup, explicitly **not** end-to-end/wall-clock training speedup. (The BSR backward kernel is NOT in the training runs — see A.8/A.9. Do not let the abstract imply a training speedup.)
  - Reconcile the numbers with Table 2: at 97.5%, sparse vs PyTorch is ~1.69× (per Table 2), and ~1.98× is the 8-bit-vs-PyTorch reference. Restate the abstract so each multiplier is tied to its exact sparsity and baseline. `<<HUMAN: confirm the exact multipliers you want as headline; agent to make abstract match the table verbatim once confirmed.>>`
  - bf6h W5 / jzn5 Q5: acknowledge no end-to-end DPO wall-clock comparison; scope the contribution to optimizer-step memory-traffic reduction.

### 3.3 Proof corrections (App D.1) — XD33 caught these; they are correct
- **"By Cauchy-Schwarz"** for the ‖∇L‖₁‖δ‖∞ step is actually **Hölder's inequality** (1/∞ conjugate). Fix the name.
- Make the ℓ₁/ℓ∞ bookkeeping consistent (the ε‖δ‖₁ vs ‖δ‖∞ step): state ε = ‖∇L‖∞ explicitly and keep the norm pairing consistent. Final bound is right; the write-up is inconsistent — clean it.
- **D.3 false statement:** "high-curvature parameters receive larger gradient steps" is false as written (high curvature ≠ large gradient; you can sit at a basin floor with high curvature and zero gradient). Either delete this justification or replace with the intended assumption (e.g., an explicit magnitude–curvature alignment assumption already invoked for Theorem 3), and cross-reference it. Do NOT leave the causal claim standing.

### 3.4 Originality statement (jzn5 W5 / Q2)
Add a crisp sentence (intro or contributions) stating novelty precisely, since warm-start magnitude scoring itself is Panda et al. [32]:
- Novel here: (i) application to *preference optimization* (DPO/GRPO) subnetworks; (ii) the **objective-determines-structure** finding; (iii) the **optimal-scoring existence + certifiability** theory (Thms 2–3); (iv) the **BSR-aware sparse AdamW** optimizer.
- State plainly that the scoring function and τ-estimation build on prior techniques. Owning this reads as rigor, not weakness.

---

## 4. CLARITY DEBT (zero compute) — the reason XD33 gave Clarity = 1

Individually small; collectively decisive. XD33's reject is Confidence-3 and largely clarity/confusion-driven → the most *gettable* reviewer, and only via polish.

### 4.1 Undefined symbols — define at first use
- **T** (total training steps) — p3, "k ≪ T".
- **H** — the Hessian of the loss; define explicitly and motivate why the quadratic/diagonal approximation is relevant (XD33 asked "why is the quadratic approximation relevant").
- **L** — the loss (confirmed by XD33's guess); define at first appearance p4.
- **τ_ρ** — define locally at first use in §3.2 (currently only defined later in mask construction); at minimum refer to it by semantic meaning ("the score threshold retaining the top (1−ρ)|θ| parameters").
- Sweep the whole methods/theory section for any other symbol used-before-defined.

### 4.2 Figure/caption ↔ content mismatches (these erode trust fast)
- **Fig 1 / §4.1:** text says "random masks diverge" but the random trajectory appears not to be plotted. Either plot the random curve (no new compute — you already ran it) or add the initial random trajectory / explicitly point to where it's shown. XD33 flagged this directly.
- **Fig 3 caption / §4.2:** claims "GRPO masks show larger divergence" but **no GRPO curve is in Fig 3**. Either add the GRPO transfer curve (if you have it) or remove/relocate the claim so caption matches figure. XD33 flagged this directly.
- **Fig 2:** XD33 asks why GRPO-OpenR1 vs GRPO-Tülu3 similarity isn't shown. If you have it, add it (it would *strengthen* the objective-determines-structure story: same objective, different data → high similarity). If not, add a one-line note on why the shown pairs were chosen.
- **Fig 5:** add a title (XD33: "for most figures there are titles"). Fix the math typos in the caption/label: `mᵢ(s) = |, sᵢ − τ̂_ρ(s), |` has stray commas; clean to `mᵢ(s) = |sᵢ − τ̂_ρ(s)|`.
- **"as k increases" (p8):** XD33 asks if this should be **t**. Reconcile k vs t notation across §4.4 and Fig 5 — pick one symbol for the warm-start step count and use it everywhere.

### 4.3 Empty appendix sections — FILL THESE
- **App C.1 (GRPO on Open-R1)** and **C.2 (DPO on Light-R1 and Tülu3)** are headers with no body (XD33: "Missing content?"). Tables 6 and 7 already contain the hyperparameters — either move/anchor those tables under C.1/C.2 with a short paragraph each, or write the connective prose. **Do not ship empty sections.** This is one of the most damaging things currently in the PDF.

### 4.4 Formatting
- **Table 2 overflow** — flagged by XD33 AND bf6h. Fix the overflow (restructure columns, reduce precision, or split mean/median into stacked sub-rows). High-visibility, easy.
- Linear probe implementation detail (XD33, jzn5): add a short paragraph (methods or App) on how the pairwise-ranking probe is trained — features (per-layer mean-pooled hidden states, already described), objective (already have Lpair), train/test split, layers sampled. Mostly assembling text you already have.

---

## 5. IN-HAND RESULTS INTEGRATION (compute already spent) — Problem A

The humans have committed to: **additional GRPO results** and **Qwen3 8B & 32B results** in the appendix. Scaffold these; insert placeholders for numbers.

### 5.1 Second model family (Qwen3 8B/32B) — the strongest generality answer
- Present as a **replication of the central finding**, not a token plot. Priority order for what to show:
  1. Qwen3 Jaccard + CKA: DPO↔DPO (different data) high vs DPO↔GRPO low — i.e., does **objective-determines-structure replicate off the Llama family**? This is the headline.
  2. Recovery curve (warm-start vs dense vs random) at the operative sparsity.
- Be explicit about what is full-suite vs subset on Qwen3, so 32B isn't a one-figure cameo inviting cherry-pick suspicion. State the subset honestly.
- Add to Limitations: generality now shown on two families/scales, not exhaustively.
- `<<HUMAN: supply Qwen3 figures/tables and confirm which analyses ran at 8B vs 32B.>>`

### 5.2 GRPO — REFRAME, do not claim rescue
- **Do NOT claim 97.5% recovery for GRPO.** Your own App D.4 shows even the oracle fails there. Claiming otherwise is a self-inflicted credibility wound.
- **Reframe (this is jiu-jitsu):** the *tolerable sparsity threshold is itself objective-dependent.* GRPO's natural update sparsity is lower (~ρ≈70%, Mukherjee [29]); enforcing 97.5% is out-of-regime for GRPO. If in-hand GRPO results show recovery at its *natural* sparsity, that is **evidence FOR the objective-determines-structure thesis**, not against generality.
- Framing rules:
  - Present as **additive** ("the framework extends to GRPO when sparsity is matched to the objective; the tolerable threshold is objective-specific"), NOT as a mid-review goalpost move.
  - Keep the honest D.4 result; recontextualize it as "97.5% is beyond GRPO's natural sparsity" rather than "GRPO fails."
  - This directly answers bf6h W1 (converts a "lack of generalizability" into a finding) and the "why under preference learning" question (the objective is exactly what governs the structure).
- `<<HUMAN: supply GRPO-at-natural-sparsity results; confirm the natural ρ you're claiming and that recovery holds there.>>`

---

## 6. POSITIONING / FRAMING (zero compute)

### 6.1 LoRA (bf6h W3, jzn5, metareview) — reframe, don't run
- Abstract/intro currently position the method as an alternative to LoRA but include **no LoRA baseline**. This half-motivation is what drew fire.
- Fix: **foreground the sparse-not-low-rank argument you already cite (Mukherjee [29])** — RL updates are sparse and practically full-rank, a structure LoRA's low-rank assumption cannot capture. Position sparse selection as addressing a *different* structural regime than LoRA, and **soften/qualify the "alternative to LoRA" wording** so you're not implicitly promising a comparison you won't run.
- Add one sentence to Limitations acknowledging no direct LoRA/PEFT head-to-head, with the principled reason.

### 6.2 "Why under preference learning?" (bf6h)
- Answer directly in intro/discussion: the central finding is that the *objective* governs subnetwork structure; preference optimization is precisely the setting where this is demonstrable and consequential (DPO vs GRPO diverge). It is not an incidental framing — the objective is the independent variable.

---

## 7. REBUTTAL-TEXT ONLY (not manuscript changes)

Answer these in the response-to-reviewers, not by editing the paper:
- **n=1 / no error bars** (checklist Q7): already documented as a limitation; defend on compute cost. Do not attempt runs.
- **Sweeps over ρ, r, seeds, more families, LoRA** (jzn5 W4/Q1, metareview): decline on compute grounds, *confidently and without apology*, for the breadth asks specifically.
- **jzn5 Q3 (histogram chunking vs stochastic top-k [1,2]):** brief justification — memory-bounded exact-ish threshold at 8B scale on CPU in seconds; cite the tradeoff. If cheap, add one clarifying sentence to App A.2/A.3.
- **jzn5 Q5 / bf6h W5 (dense vs sparse training time & downstream):** give the optimizer-step timing you have; state clearly you don't have end-to-end wall-clock (scope, per §3.2).

> **Discipline rule for the rebuttal:** use "compute is limited" ONLY for breadth asks (more runs). NEVER use it to wave off the within-noise-table concern (Problem B) — that one is answered by the §2 reframe (evidence you already have), not by a budget excuse. Answering an evidence question with a cost excuse reads worse than the original problem.

---

## 8. EXECUTION ORDER (for the agent)

Do in this sequence; surface for human review at each ⏸️.

1. §3 + §4 — all zero-compute honesty + clarity fixes (definitions, figure/caption mismatches, empty appendices C.1/C.2, Table 2 overflow, math corrections, speedup traceability, loss-retention rename, originality sentence). These are independent and safe. ⏸️ human review.
2. §2 — draft reframed §4.1 + Table 1 caption + averages/dense-dataset note. ⏸️ **human review before propagating** the reframe elsewhere.
3. §5 — scaffold Qwen3 and GRPO-reframe sections with placeholders. ⏸️ human inserts numbers.
4. §6 — LoRA + preference-learning positioning edits. ⏸️ human review.
5. Draft the point-by-point rebuttal text (§7 + pointers to where each manuscript change lives). ⏸️ human review.

---

## 9. COVERAGE CHECKLIST (map every reviewer point → action)

| Reviewer point | Where addressed |
|---|---|
| XD33: T, H, L, τ_ρ undefined | §4.1 |
| XD33: Fig 1 random not shown | §4.2 |
| XD33: Fig 3 no GRPO curve | §4.2 |
| XD33: Fig 2 GRPO-GRPO pair missing | §4.2 |
| XD33: "as k increases" k/t | §4.2 |
| XD33: C.1/C.2 empty | §4.3 |
| XD33: abstract 3.3× at 99.75% not 97.5% | §3.2 |
| XD33: Fig 5 title + math typos | §4.2 |
| XD33: Cauchy-Schwarz → Hölder | §3.3 |
| XD33: ℓ1/ℓ∞ swap assumptions | §3.3 |
| XD33: high-curvature→large-step false | §3.3 |
| XD33: GSM8K random > dense; small deltas | §2 |
| XD33: probe implementation detail | §4.4 |
| XD33: cross-task warm-start transfer | §5 / rebuttal |
| XD33/bf6h: Table 2 overflow | §4.4 |
| jzn5 W1: "performance retention" overstated | §3.1 |
| jzn5 W2: too much loss emphasis, want downstream | §2 |
| jzn5 W3: Table 1 averages, dense dataset, missing combos | §2 / rebuttal |
| jzn5 W4: r, ρ ablation | rebuttal (§7) |
| jzn5 W5/Q2: originality | §3.4 |
| jzn5 Q3: histogram vs stochastic top-k | §7 |
| jzn5 Q5: dense vs sparse time/downstream | §3.2 / §7 |
| bf6h W1: GRPO fails at 97.5% | §5.2 |
| bf6h W2: single model | §5.1 |
| bf6h W3: no LoRA baseline | §6.1 |
| bf6h W4: lines 240–241 no cross-val | §2 (honest scoping) / rebuttal |
| bf6h W5: no pure-DPO training-speed comparison | §3.2 / §7 |
| bf6h W6: little change pre/post fine-tune | §2 |
| bf6h: "why preference learning" | §6.2 |
| Metareview: multiple base models | §5.1 |
| Metareview: PEFT comparison | §6.1 / §7 |
| Metareview: ablations | §7 |
| Metareview: theoretical scope | §3.1 |
| Metareview: presentation/inconsistencies | §4 |

Any reviewer point not appearing above = gap; flag it.

---

## 10. DPO training hyperparameters — Qwen3-8B & Olmo-3-7B (Light-R1)

Both model families reuse the Llama 3.1 base recipe (`dpo_5k_hpc_copypaste.md`); the values are identical across all three. Dense and sparse runs share the same optimizer settings so magnitudes match.

| Hyperparameter | Qwen3-8B | Olmo-3-7B-Instruct |
|---|---|---|
| HF model id | `Qwen/Qwen3-8B` | `allenai/Olmo-3-7B-Instruct` |
| Dataset | `light-r1` (`qihoo360/Light-R1-DPOData`) | `light-r1` (`qihoo360/Light-R1-DPOData`) |
| Steps | 500 | 500 |
| Learning rate | 5e-7 | 5e-7 |
| LR schedule | linear | linear |
| Warmup ratio | 0.1 | 0.1 |
| Per-device batch size | 2 | 2 |
| Grad accumulation | 64 | 64 |
| Effective batch | 128 | 128 |
| DPO β | 0.1 | 0.1 |
| Max length | 1024 | 1024 |
| Max prompt length | 1024 | 1024 |
| Weight decay | default (TRL) | default (TRL) |
| Precision | bf16 | bf16 |
| Gradient checkpointing | on | on |
| Sparsity (sparse runs) | 97.5% | 97.5% |
| Min layer keep ratio | 0.0025 | 0.0025 |
| Sparse optimizer | `sparse_adamw`, block_size 32 | `sparse_adamw`, block_size 32 |
| HF checkpoint | step 500 | step 500 |
| Delta logging | bf16, every 50 steps → step **250** | bf16, every 50 steps → step **500** |
| GPU / wall | 1× H200, 07:45:00 | 1× H200, 07:45:00 |

**Notes**
- Delta files and `base_state.pt` are stored in **bf16** (`DPO_train.py`, `diff.bfloat16()`), not fp32 — the earlier "float32 delta files" comment in the Qwen3 oracle script is inaccurate.
- **Delta-logging end step differs:** Qwen3 logged through step 250; Olmo logs through **step 500** so the final-step delta can feed a delta-log "ground_truth" oracle (`even_better_mask_finder.py --method ground_truth --target_step 500`) in addition to the step-250 magnitude/rewinding mask.
- **Qwen3 "oracle":** the shipped Qwen3 oracle run actually reused the step-250 magnitude mask; the true checkpoint-diff oracle hit bf16-cancellation precision issues at this low-LR/short-horizon regime.
- **Olmo masks (`gen_masks_olmo3_7b_lr1.slurm`):** builds three element-wise 97.5% masks — magnitude@250, delta-log oracle@500, and checkpoint-diff oracle@500 — plus the standalone random mask; keep whichever oracle is well-conditioned.

### Olmo 3 7B job scripts (mirror of the Qwen3 set)
| Condition | Script |
|---|---|
| Dense | `scripts/dpo_olmo3_7b_light_r1_500.slurm` |
| Sparse random | `scripts/sparse_dpo_olmo3_7b_random_lr1_500.slurm` |
| Mask generation | `scripts/gen_masks_olmo3_7b_lr1.slurm` (needs `DENSE_RUN_ID`) |
| Sparse oracle / magnitude | `scripts/sparse_dpo_olmo3_7b_masked_lr1_500.slurm` (needs `MASK_FILE`, `MASK_LABEL`) |

Env: `rl_casino` conda env (transformers 4.57.1, torch 2.9.0+cu128, trl 0.24.0) loads `olmo3` natively — no env changes needed.