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

**A competing explanation, which the per-block panel raises and we cannot fully exclude.** Per-block
keep rates (`e1_phi_by_layer.png`) are [0.0250, 0.0250] for both random masks, [0.0248, 0.0257] for
the warm DPO Tülu3 mask, [0.0240, 0.0280] for the oracle DPO mask, but [0.0103, 0.0303] for the
GRPO masks. Near-uniform allocation across blocks is precisely the signature of tie-break dilution
measured directly on Olmo3 in §3.1: when most of a mask is filled by uniform noise, every block
receives about the same share. So the DPO-vs-GRPO φ gap may reflect **how much of each mask is
noise** rather than anything about objectives — a mask that is 3/4 arbitrary is mechanically pulled
toward chance.

This is testable in principle but not with surviving data: computing the tie-break share requires
the source delta logs, and the Tülu3 run's are deleted. The Open-R1 GRPO delta logs do survive
(`rl_casino_grpo/dense/grpo_dense_openr1_steps500_evolstudy/deltas`), so the GRPO side could be
measured; without the Tülu3 side there is nothing to compare it against. **Until that is resolved,
the ordering below should be read as a difference between mask *constructions*, not established as a
difference between objectives.**

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

| arm | coverage | Jaccard vs k=250 | d | 95% CI | V |
|---|---|---|---|---|---|
| oracle deltalog (reference) | 213 tensors | — | 0 | — | 0 (definitional) |
| warm k=250 | 216 tensors | 1.0 | **0.005851** | [0.005166, 0.006645] | 0.143982 |
| warm k=50 | 216 tensors | 0.8399 | PENDING | | 0.157780 |
| warm k=100 | 216 tensors | 0.9502 | PENDING | | 0.149917 |
| random, as originally built | 226 tensors | — | **0.168163** | [0.166386, 0.169921] | 0.511664 |
| random, coverage-matched to 216 | 216 tensors | 0.0400 | PENDING | | (same mask, subset) |

All four masks passed the gate in `scripts/verify_mask_against_reference.py` (identical key sets and
shapes, keep rate 0.025000, `pooling_mode=global_with_layer_floor`,
`min_layer_keep_ratio=0.0025`, no all-zero tensors). Jaccard rises with k as it should —
0.8399 at k=50 and 0.9502 at k=100 against the k=250 reference — which is the check that caught the
CPU-built mask described in §3.2.

### V under hybrid τ

From `cert_olmo3_hybrid_mat/mask_score_gap_gap_diagnostics.json`, with
`cert_tau_rule=hybrid_global_phase` and `cert_hybrid_min_layer_keep_ratio=0.0025` so the estimator
matches how production masks are actually built — the surviving Llama artifact used global τ with
`min_layer_keep_ratio=0`, a mismatch §4.4 currently glosses.

| arm | V | certifiability_strict_fraction | τ̂ |
|---|---|---|---|
| k=50 | 0.157780 | 0.8422197 | 4.2746e-10 |
| k=100 | 0.149917 | 0.8500832 | 1.3970e-09 |
| k=150 | 0.146467 | 0.8535334 | 2.9395e-09 |
| k=200 | 0.144732 | 0.8552682 | 5.0932e-09 |
| k=250 | 0.143982 | 0.8560182 | 7.8580e-09 |
| random, seed 42 | 0.511664 | 0.4883365 | 0.97266 |

`k_keep = 182,450,278` and `N = 7,298,011,136` throughout; oracle τ̂ = 4.2201e-09.

**Two independent checks on the τ machinery.** The random arm's τ̂ = 0.97266, and uniform[0,1]
scores cut at the top 2.5% must give ≈0.975. And V(random) = 0.511664 against the surviving Llama
figure of 0.51164 — two different models agreeing to four decimals.

**V is monotone in k and saturating**, with successive differences −0.0079, −0.0035, −0.0017,
−0.0008, i.e. roughly halving each step. Total spread 0.0138 across k ∈ [50, 250], about 3× the
Llama global-τ spread over a comparable range. So the pre-registered flatness caveat fired in the
useful direction: Olmo3 under hybrid τ is not as flat as Llama under global τ, which gives the
k-cluster slightly more lever arm than the plan expected — though 0.0138 is still a narrow x-axis.

**What V is actually measuring at this sparsity.** τ̂ grows ~18× across the k range
(4.27e-10 → 7.86e-09) while V falls. Since ~98.8% of coordinates have score exactly zero (§3.1),
their margin is `|0 − τ̂| = τ̂`, so for the overwhelming majority of weights the certifiability
condition reduces to "is the oracle's displacement at this coordinate below τ̂". V is therefore
largely a statement about how many coordinates the oracle barely moved, mediated by a threshold
that is itself only ~3–4× the tie-break noise scale (1.03e-10 at k=50, 4.52e-10 at k=100). That is
worth saying plainly: at ρ=97.5% on this run, V is not primarily a measure of selection quality.

**A degeneracy that limits what the k-cluster can show.** Within the warm family V is a monotone
function of k, and so is mask distance to the oracle (Jaccard vs k=250: 0.8399, 0.9502, 1.0 for
k=50, 100, 250). Any V–d correlation restricted to the warm family therefore cannot distinguish
"V predicts d" from "mask distance to the oracle predicts d". Only the random arm breaks that
degeneracy, which is why the coverage-matched random point matters so much.

**Scope mismatch to note.** V is computed over all 355 parameter tensors (N = 7,298,011,136),
while the trained masks cover 216 two-dimensional weights (7,129,710,592). V and d are thus
measured over slightly different tensor sets.

**τ̂ is an approximation, by the pipeline's own admission.** The artifact records
`cert_tau_note: "hybrid_global_phase: τ is the global-phase cutoff after per-layer floors
(mask_utils-style), not the pure-global Theorem-3 τ"`, and `cert_mode` adds that hybrid masks
"are not equivalent to a single scalar τ". So hybrid τ matches mask *construction* better than
global τ does, but it is not literally the τ of Theorem 3.

### `d = mean|ΔL|` measures noise for arms that track the oracle closely

Reporting `d` exactly as pre-registered, and alongside it the signed mean deviation, because the
absolute value turns out to matter a great deal:

| arm | V | d = mean\|ΔL\| | signed mean(L_M − L_oracle) | 95% CI on signed |
|---|---|---|---|---|
| warm k=250 | 0.14398 | 0.005851 | **−0.001171** | [−0.002019, +0.000130] |
| warm k=50 | 0.15778 | 0.006381 | **+0.003445** | [+0.002172, +0.004839] |
| random, 226 coverage | 0.51166 | 0.168163 | **+0.168163** | [+0.166386, +0.169921] |

For the random arm the sign never flips, so `d` and the signed mean agree to six decimals. For
warm k=250 the systematic gap is 0.0012 while `d` is 0.0059, i.e. **roughly 80% of `d` is per-step
noise**: symmetric fluctuation around zero becomes a positive floor once you take absolute values.
So `d` is a poor discriminator exactly where the interesting arms live — close to the oracle. The
signed mean is the better primary statistic, and `d` should be read as an upper bound inflated by
noise.

Read on the signed statistic, the three points are **monotone in V with every gap resolved**:
k=250 is statistically indistinguishable from the oracle (its CI contains zero), k=50 sits
significantly above it, and random sits far above. The arm-vs-arm difference confirms the
within-family gap that the overlapping `d` intervals could not:

| A − B | mean | 95% CI | excludes 0 |
|---|---|---|---|
| warm_k50 − warm_k250 | +0.004616 | [+0.003005, +0.005782] | yes |
| warm_k50 − random_226 | −0.164718 | [−0.166091, −0.163033] | yes |
| warm_k250 − random_226 | −0.169334 | [−0.170770, −0.167216] | yes |

Differencing two arms cancels the reference *and* the shared per-step batch effect, which is why it
resolves a 0.0046 gap that individual `d` intervals of width ~0.0015 could not.

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

### 3.0 Root cause: at lr=5e-7 in bf16, most weights never move at all

This is causally upstream of everything in §3.1–3.3, so it comes first.

Training runs with `bf16=True`, so the weights themselves are bfloat16 — confirmed directly:
`base_state.pt` has `dtype=torch.bfloat16`. bf16 carries 8 significand bits, so the representable
spacing at a weight `w` is `ulp(w) = 2^floor(log2|w|) · 2⁻⁷`. At lr=5e-7 a single AdamW step moves
a weight by roughly `lr`, which for `|w| ~ 1e-2` (ulp ~ 6e-5) is about two orders of magnitude
below the spacing. **The update rounds away and the weight does not change.**

The decisive test is that nonzero deltas land on exact integer multiples of `ulp(base)`:

| tensor | numel | nonzero frac | median \|Δ\|/ulp | median distance to integer multiple | frac within 1e-3 of an integer | frac exactly 1 ulp |
|---|---|---|---|---|---|---|
| layers.10.q_proj | 16,777,216 | 0.082471 | 29.0 | **0.000000** | 0.9615 | 0.0618 |
| layers.10.gate_proj | 45,088,768 | 0.002771 | 13.0 | **0.000000** | 0.9623 | 0.0904 |
| embed_tokens | 410,738,688 | 0.000089 | 9.0 | **0.000000** | 0.9673 | 0.1090 |

An fp32 weight differenced and then cast to bf16 *for storage* would land on the delta's own bf16
grid, which is far finer than `ulp(base)` because `|Δ| ≪ |w|`. Landing on integer multiples of the
**base weight's** ulp is only possible if the weight itself lives on that grid. So the zeros are
not a logging artifact and not a storage-precision choice; they are weights that genuinely never
moved.

**Corollary, and it inverts a standard intuition.** `ulp ∝ |w|`, so larger weights are quantized
more coarsely and are *less* able to move. The three sampled tensors order perfectly inversely:

| tensor | median \|w\| | median ulp | nonzero fraction |
|---|---|---|---|
| layers.10.q_proj | 1.08e-02 | 6.10e-05 | 8.25% |
| layers.10.gate_proj | 1.60e-02 | 1.22e-04 | 0.28% |
| embed_tokens | 7.91e-02 | 4.88e-04 | 0.009% |

In this regime the warm-start signal is therefore concentrated in **small-magnitude** weights,
because only they can register a 5e-7 update in bf16 — the opposite of what magnitude-based pruning
intuition would suggest. Any claim that warm-start displacement identifies "important" weights has
to contend with the fact that, at this learning rate and precision, it substantially identifies
*representably movable* weights.

A sanity note on the arithmetic: predicting the zero fraction as `P(ulp(base) > lr × steps)` gives
0.709 / 0.869 / 0.972 for the three tensors, against observed 0.918 / 0.997 / 0.99991. Predicted
sits below observed in every case, as it must — `lr × steps` is an upper bound on how far a
coordinate could travel, and real AdamW updates partially cancel. The ordering matches exactly.

**What would change this.** Delta storage precision is not the lever (bf16 stores a 2.5e-5 delta
with ~1e-7 relative error). The levers are fp32 master weights or a larger learning rate. This is
worth knowing before drawing conclusions about *which* weights matter from any run in this regime.

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

### 3.4 The random baseline is not fully independent of the warm masks

`generate_random_mask.py:70-95` seeds `torch.manual_seed(42)` and draws `torch.rand` per tensor in
parameter order. The warm masks' tie-break noise does the same thing with the same seed
(`mask_utils.py:169`/`:513`). Where the two RNG streams coincide, a "random" mask and the
tie-broken portion of a warm mask select the *same* coordinates.

Measured against the independence expectation computed from each mask's own per-tensor keep rates
(`E|A ∩ B| = Σ_t r_At · r_Bt · n_t`), for random_216 against warm k=250:

| | value |
|---|---|
| observed \|A ∩ B\| | 13,715,206 |
| independence expectation | 4,455,948 |
| excess factor | **3.08** |
| observed / independence Jaccard | 0.040013 / 0.012658 |
| **median excess across the 216 tensors** | **1.0200** |

The excess is concentrated, not diffuse: `embed_tokens` at 7.36× (1,287,728 observed against
174,869 expected) and the early MLP `gate_proj` layers 2/5/6/7 at ~6.28×, while the bottom tensors
sit at 0.979–0.983 (slightly under 1, as a fixed keep budget implies). That is the signature of
RNG-stream coincidence on the first tensors drawn, with the streams diverging as per-tensor draw
counts accumulate.

Scale of the problem: ~9.3 M coordinates of excess overlap, 5.2% of the mask. The median tensor is
independent, so the control is *approximately* valid. The direction is conservative — shared
coordinates pull `d(random)` toward `d(warm)`, so the warm-vs-random gap is if anything understated.
Worth fixing by seeding the random baseline differently from the tie-break, but it does not
invalidate the comparison.

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
