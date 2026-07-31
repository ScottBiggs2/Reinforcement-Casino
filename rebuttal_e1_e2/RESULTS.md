# E1 and E2 results

Companion to [PREREGISTRATION.md](PREREGISTRATION.md), which was committed before any number
below was read. Run 2026-07-31 on Northeastern Explorer. Branch `rebuttal-e1-e2`.

Values marked **PENDING** are still in flight at the time of writing and will be filled in from
the artifacts named alongside them.

---

## 1. E1 — does mask selection sit where the DPO gradient energy is?

### What was measured

`phi(M) = ||g ⊙ M||₂ / ||g||₂` at the pretrained initialization of Llama-3.1-8B-Instruct, where
`g` is the gradient of the mean DPO loss over 128 Tülu3 examples (64 microbatches × 2, i.e. the
effective batch a real run's first optimizer step uses). Eight masks, all ρ=97.5%, all scored
against that one gradient. Code: `src/analysis/grad_energy_capture.py`; job
`scripts/e1_grad_energy_llama31_tulu3.slurm` (3 min 13 s on one H200).

This replaced a planned W&B `train/grad_norm` analysis, which cannot work: HF Trainer logs
`clip_grad_norm_(model.parameters(), ...)`, a single global *dense* norm, so every arm of a masked
run logs the same number. That is confirmed empirically in the Olmo3 `training_log.csv` files —
step-1 `grad_norm` is 70.0 for the dense, random, magnitude and oracle-deltalog arms alike. Keep
that observation: it independently confirms the BSR block-density argument, since at 2.5% element
density `P(16×16 block empty) = 0.975²⁵⁶ ≈ 0.0015`, so ~99.85% of blocks survive and the BSR
gradient is effectively dense.

### Instrument validation (both gates passed)

Two independent checks, run before the real measurement:

- **Synthetic-mask check.** On Qwen3-0.6B with an in-memory uniform 2.5% mask,
  `phi = 0.154766` against a computed chance of `0.158089` (ratio 0.979) — within the ~3% sampling
  noise a heavy-tailed gradient implies at this scale. Both chance baselines agreed to five
  decimals, as they should when a mask has no per-layer floor.
- **Loss-at-initialization check.** The DPO loss at θ⁰ came out to `0.693147 = ln 2` exactly.
  That is the signature of a correct reference term: at θ⁰ the policy equals the reference, the
  margin is zero, and `-log σ(0) = ln 2`. It rules out the common failure of measuring a
  model-only chosen-vs-rejected margin instead of the real DPO objective.
- **Pre-registered validity gate.** `phi(random)` must land near chance or nothing gets reported.
  Both random arms did: ratios **0.977** and **0.970**.

### Results

Scope note: masks differ in coverage. The DPO/Tülu3 masks cover all 291 weight tensors (including
the 65 RMSNorm vectors); the GRPO masks cover the 226 two-dimensional weights. The table uses the
common 2-D scope, which is also the only scope sparse training acts on. Full-coverage numbers are
within 0.0003 of these (`e1_grad_energy.json`), so the norms carry negligible gradient energy.

| mask | φ | chance (energy-weighted) | **φ / chance** |
|---|---|---|---|
| oracle DPO Tülu3 (checkpoint-diff) | 0.354692 | 0.174091 | **2.037** |
| warm DPO Tülu3 k=200 | 0.274987 | 0.164081 | **1.676** |
| random DPO Tülu3 | 0.154467 | 0.158143 | **0.977** ← gate |
| oracle GRPO Open-R1 | 0.438163 | 0.190895 | **2.295** |
| warm GRPO k=50 | 0.477897 | 0.191305 | **2.498** |
| warm GRPO k=100 | 0.458914 | 0.192244 | **2.387** |
| warm GRPO k=250 | 0.445780 | 0.192716 | **2.313** |
| random GRPO | 0.153347 | 0.158144 | **0.970** ← gate |

`||g||² = 3.413136e+03` over 226 tensors and 8,029,995,008 elements — matching the expected
`embed_tokens + lm_head + 32 × 7` decomposition exactly.

**The chance baseline had to be computed per mask, not assumed.** It ranges 0.158–0.193 across
these masks, against a naive `sqrt(1-ρ) = 0.158`. Using the naive value would have inflated the
GRPO ratios to 2.8–3.0. The spread comes from per-tensor keep rates being far from uniform
interacting with an unevenly distributed gradient energy; see `e1_per_tensor.csv`.

### Interpretation, against what was pre-registered

**Primary branch: φ ≫ chance.** Selection captures gradient energy — 2.04× chance for the DPO
oracle mask, 1.68× for the warm DPO mask. The §D.3 bridge from `s_warm` to `s*` survives on this
evidence, and the sign-persistence reserve measurement is *not* triggered.

**Secondary prediction: refuted, in the opposite direction.** We registered
`phi(oracle_GRPO) < phi(oracle_DPO)` on the reasoning that Figure 2 shows DPO and GRPO masks
selecting near-disjoint coordinates, so GRPO-selected coordinates should carry little DPO gradient
signal. Instead **every GRPO mask captures more DPO gradient energy than either DPO-derived
mask** (2.31–2.50× versus 1.68–2.04×), despite coming from a different objective *and* a different
dataset (Open-R1 rather than Tülu3).

Two consequences worth stating plainly:

1. The cross-objective transfer gap in Figure 3 is **not** explained by gradient-space
   misalignment at initialization. If anything it is anti-explained: the masks that transfer worse
   are better aligned with the target objective's initial gradient. Whatever makes a mask work, it
   is not initial gradient-magnitude alignment.
2. Within the GRPO family φ falls monotonically in k: 2.498 (k=50) → 2.387 (k=100) → 2.313
   (k=250) → 2.295 (oracle at k=T). This is coherent and mechanistic — a mask built from the
   earliest displacement tracks the *initial* gradient most closely, and drifts as accumulated
   displacement moves away from θ⁰. It also means "closer to the oracle" and "better aligned with
   the initial gradient" are opposing directions, which is itself an argument against reading φ as
   a proxy for mask quality.

---

## 2. E2 — does certifiability predict trajectory divergence?

### Gates, both of which fired

`src/analysis/trajectory_divergence.py`. Both gates exclude rather than caveat.

- **Config guard**, read from `training_args.bin` (not `run_manifest.json`, which carries only
  ~12 fields and lacks `warmup_ratio`, `seed`, `max_grad_norm`, `lr_scheduler_type`). All arms
  agree on the Table 7 values: `max_steps=500`, `learning_rate=5e-07`,
  `lr_scheduler_type=LINEAR`, `warmup_ratio=0.1`, `max_grad_norm=1.0`,
  `gradient_accumulation_steps=64`, `per_device_train_batch_size=2`, `beta=0.1`, `seed=42`,
  `data_seed=null`, `max_length=1024`, `accelerator_config.use_seedable_sampler=true`. `optim` is
  whitelisted: sparse arms log `adamw_torch_fused` but never consult it, since the real optimizer
  is `SparseAdamW` passed via `optimizers=`.
- **Step-1 `grad_norm` precondition.** It excluded the `oracle_ckptdiff` arm (70.5 against 70.0
  for every other arm). The gate is doing real work rather than decorating the analysis. Honest
  caveat: this scalar is bf16-quantized (70.0 and 70.5 are adjacent representable values near 70),
  so it is a cheap smoke test, not a precision check.
- **Sanity check.** Differencing the oracle arm against itself gives `d = 0.000000`.

### Measured divergence

`d(M) = mean_t |L_M(t) − L_oracle(t)|` over steps 401–500, joined on `step`, with a
block-bootstrap CI over contiguous 10-step blocks (10,000 draws).

| arm | coverage | d | 95% CI | V |
|---|---|---|---|---|
| oracle deltalog (reference) | 213 tensors | 0 | — | 0 (definitional) |
| warm k=250 | 216 tensors | **0.005851** | [0.005166, 0.006645] | PENDING |
| warm k=50 | 216 tensors | PENDING | | PENDING |
| warm k=100 | 216 tensors | PENDING | | PENDING |
| random, as originally built | 226 tensors | **0.168163** | [0.166386, 0.169921] | PENDING |
| random, coverage-matched to 216 | 216 tensors | PENDING | | (same mask, subset) |

V values come from `mask_score_gap_gap_diagnostics.json` under the hybrid τ rule
(`cert_tau_rule=hybrid_global_phase`, `cert_hybrid_min_layer_keep_ratio=0.0025`) so the estimator
matches how production masks are actually built — the surviving Llama artifact used global τ with
`min_layer_keep_ratio=0`, a mismatch §4.4 currently glosses.

The paired Δ(t) traces (`delta_traces.png`) are the more informative panel: the random arm
diverges steadily from the oracle trajectory through the first ~250 steps and plateaus near 0.17,
while the warm k=250 arm tracks it within ±0.01 for the entire run. Because all arms see an
identical data order (seeded sampler, `seed=42`, `data_seed=null`, verified by the step-1
`grad_norm` equality), `L_M(t)` and `L_oracle(t)` are evaluated on the *same* batch at step t, so
the pairing cancels the large batch-to-batch swings visible in the raw losses. That is what makes
a per-step training loss usable here at all.

---

## 3. Three instrumentation findings that bear on the paper

These were not the object of either experiment. They emerged from the verification gates and they
affect how existing results should be read.

### 3.1 Most of a warm-start mask is chosen by RNG, not by signal

`scripts/measure_mask_tiebreak_share.py`, restricted to each mask's own 216 tensors:

| | k=50 | k=250 |
|---|---|---|
| coordinates with score > 0 | 59,036,882 (0.83% of coverage) | 80,500,376 (1.13%) |
| keep budget | 178,242,764 | 178,242,764 |
| **selected with score == 0** | **134,866,352 → 75.7% of the mask** | **119,135,340 → 66.8%** |
| fraction of nonzero coords captured | 0.735 | 0.734 |

Two separate mechanisms:

1. **The keep budget exceeds the signal support.** At ρ=97.5% the mask must select 178.2 M
   coordinates, but only 59–81 M have any resolvable accumulated `|Δθ|` after 500 steps at
   lr=5e-7. The remainder is filled by tie-break noise (`torch.manual_seed(42)`, scale
   `max|score| × 1e-6`, `mask_utils.py:169`/`:513`).
2. **The tie-break noise outranks real signal.** Selection captures only ~73% of nonzero
   coordinates: it leaves 15.7 M coordinates that *do* have signal unselected while taking
   134.9 M zero-score ones. The noise scale (1.03e-10 at k=50, against a max score of 2.06e-4)
   exceeds many genuine scores, so it is not merely breaking ties — it is randomizing the bottom
   of the ranking.

Interpretive caveat: delta logs store bf16, so "score exactly zero" means "no displacement
resolvable at bf16 in the logged deltas", not necessarily no true displacement. Either way it is
zero as far as the mask builder can see, so the tie-broken share is real.

**This weakens the flatness argument.** Jaccard(k=50, k=250) = 0.8399, i.e. the two masks share
~163 M of 178 M coordinates — but at most ~59 M of that can be signal, so most of the agreement
across k is *shared tie-break noise*: the same seed, the same stream, the same per-coordinate noise
tensor, therefore the same arbitrary coordinates chosen at every k. Flat `d` across k is
consequently expected largely by construction, and should not be presented as evidence for a V→d
mechanism.

### 3.2 The delta-log mask builder silently drops weight tensors

Olmo-3-7B-Instruct has **226** two-dimensional weight tensors (32 × 7 projections +
`embed_tokens` + `lm_head`, untied). The masks actually contain:

| mask | tensors | covered elements |
|---|---|---|
| oracle deltalog (E2 reference) | **213** | 6,994,444,288 |
| warm magnitude k=50 | **216** | 7,129,710,592 |
| warm magnitude k=250 | **216** | 7,129,710,592 |
| warm magnitude k=100, built on a CPU node | **219** | 7,151,730,688 |
| oracle checkpoint-diff | 226 | 7,297,482,752 |
| random, seed 42 | 226 | 7,297,482,752 |

Cause: `even_better_mask_finder.py:105-116` de-duplicates tied weights by a **content** hash
(shape + `delta.sum()`). With ~99% of deltas exactly zero in bf16, distinct same-shape tensors
produce identical sums and one gets dropped as a "tie". The dropped tensors are systematically
**attention** projections — `k_proj`/`v_proj` are the smallest tensors (4.2 M elements) and
collide most readily, `q_proj`/`o_proj` share the 4096×4096 shape. MLP tensors, being larger, never
collided. Because the hash depends on float accumulation order, a CPU build and a GPU build drop
*different* sets, which is how this was caught.

### 3.3 The dropped tensors are trained densely, so arms differ in capacity

This is the consequence that matters. `sparse_dpo_efficiency.py:234-241` enforces sparsity
**only** through `SparseAdamW` — there is no BSR layer replacement in the trainer that produced
these runs, so forward and backward are dense and the mask acts on the optimizer update alone. And
`sparse_adamw.py:119-131` routes any parameter without a mask entry to `_dense_step`, "Standard
dense AdamW update (fallback for non-masked params)".

So a tensor missing from the mask file is trained at full density:

| arm | sparse (2.5% of covered) | dense (unmasked) | **total trainable** |
|---|---|---|---|
| oracle deltalog (reference) | 174.9 M | **303.0 M** | **477.9 M** |
| warm k=250 | 178.2 M | **167.8 M** | **346.0 M** |
| random, as built | 182.4 M | 0 | **182.4 M** |

The warm arm has ~1.9× and the reference ~2.6× the trainable parameters of the random arm, and the
extra capacity is concentrated in attention. **The original warm-vs-random comparison therefore
conflates mask quality with trainable capacity**, and the 29× gap in `d` cannot be attributed to
mask quality alone.

Remedy applied: `scripts/subset_mask_to_reference_coverage.py` restricts the *same* seed-42 random
selection to the warm masks' exact 216-tensor coverage, giving 178,242,313 kept against the warm
masks' 178,242,764 — a match to within 451 coordinates. Comparing `d(random_226)` against
`d(random_216)` isolates the coverage effect by itself, and `random_216` restores a
coverage-matched high-V point for the V-vs-d question.

The k-cluster is unaffected: the k=50 mask's key set is **identical** to k=250's, so those arms
differ only in which coordinates within identical coverage were selected.

---

## 4. What this does and does not establish

**Established.**
- φ ≫ chance for every non-random mask, with a validated instrument and both random arms at
  chance. Selection is not arbitrary with respect to gradient energy.
- GRPO-derived masks capture *more* DPO gradient energy than DPO-derived ones, so the
  cross-objective transfer gap is not a gradient-space alignment effect.
- The warm k=250 arm tracks the oracle trajectory to within ±0.01 while a random mask diverges to
  ~0.17 — though see the capacity caveat above for how much of that is mask quality.

**Not established, and not claimed.**
- Whether V predicts d. The original point set cannot answer it: the only arm with materially
  different V is also the only arm with different trainable capacity. The coverage-matched arm
  addresses this, but with a handful of points we report the scatter and the qualitative
  relationship only — no regression coefficient, no p-value, as pre-registered.
- Flatness of d across k as support for a V→d mechanism, since the k-cluster masks share ~84% of
  their coordinates, mostly via shared tie-break noise.

---

## 5. Known gaps

- Llama-3.1-8B Light-R1 oracle and warm masks, and their source run, are deleted. The Figure 1 DPO
  Light-R1 curves are not retrievable; only the exported PNG survives.
- The Tülu3 dense run that produced the E1 masks is also gone from scratch, so its exact batch
  configuration is unrecoverable. `g` was therefore computed at bs 2 × 64, `max_length` 1024,
  `max_prompt_length` 512, β 0.1, seed 42 — the Olmo3 arms' effective batch and `DPO_train.py`'s
  defaults. φ is a ratio of norms of the same gradient, so it is insensitive to the overall scale
  of `g`; it is sensitive to *which* examples `g` averages over, which is why the batch is fixed
  by seed and reported.
- Figure 1 is labelled "Magnitude Rewinding Step200" but every surviving Light-R1 warm mask and
  both completed warm-start DPO runs are at k=250. The Tülu3 mask genuinely is k=200. Likely a
  labelling error; resolve before resubmission.
- The surviving Llama certifiability artifact used global τ, not the hybrid estimator §4.4
  describes. The Olmo3 re-run fixes this going forward but does not retroactively fix the Llama
  numbers.
- E1 runs on Llama-3.1-8B/Tülu3 and E2 on Olmo-3-7B/Light-R1 by necessity, not oversight. Olmo3 is
  the only family with surviving delta logs at every milestone, a completed oracle run, a
  completed dense run, and no config drift; the Llama Light-R1 arms mix `max_steps=5000` runs with
  a `max_steps=500` random arm, which puts them on different LR trajectories.
- §3.1–3.3 are measured on Olmo-3-7B/Light-R1 at ρ=97.5%, lr=5e-7, 500 steps. The GRPO and Tülu3
  masks used in E1 cover all 226 tensors, so §3.2 does not affect them; whether §3.1's tie-break
  share also applies to them cannot be checked, because their delta logs are deleted.
