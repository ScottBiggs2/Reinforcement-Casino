# Pre-registration: E1 (gradient energy capture) and E2 (certifiability → divergence)

Written **before** any result from either experiment was read. Committed at the same time as the
measurement code and before the jobs that produce the numbers completed. The point is to commit to
what each outcome would mean, so neither result can be retrofitted into whichever story is more
convenient.

Date: 2026-07-31. Repo branch: `rebuttal-e1-e2`.

---

## E1 — does mask selection sit where the DPO gradient energy is?

### Quantity

For a boolean mask `M` over 2-D weight tensors, at the pretrained initialization θ⁰:

```
phi(M) = || g ⊙ M ||₂ / || g ||₂
```

`g` is the gradient of the mean DPO loss over 128 Tülu3 examples (64 microbatches × 2), i.e. the
same effective batch a real run's first optimizer step uses. Both norms are restricted to the
tensor set that mask covers. Model: Llama-3.1-8B-Instruct. All eight masks (three DPO/Tülu3, five
GRPO/Open-R1) are scored against this one gradient.

### Baselines

Two nulls are computed from the mask artifacts themselves, never assumed:

- `chance_uniform_global = sqrt(Σ_t k_t / Σ_t n_t)` — the naive null, which assumes gradient
  energy is spread in proportion to element count.
- `chance_energy_weighted = sqrt(Σ_t (k_t/n_t)·||g_t||² / Σ_t ||g_t||²)` — `E[phi²]` under uniform
  random selection *within each tensor* at that tensor's realized keep rate.

**The second is the null of record.** Production masks are built with a per-layer floor
(`min_layer_keep_ratio = 0.0025`, reserving ~20.1 M of the ~200.76 M keep budget), so per-tensor
keep rates are *not* uniformly 2.5 % and the flat 0.158 figure is not the right comparison.

### Validity gate (blocking, decided in advance)

`phi(random_dpo_tulu3)` and `phi(random_grpo)` must land near `chance_energy_weighted`. If they do
not, **the instrument is broken and no φ gets reported at all** — not even for the oracle arms.
Suspects, in order: a denominator not restricted to the mask's own covered tensors; bf16
accumulation error; a mask/parameter key or shape mismatch silently dropping tensors.

### The two branches, and what each would mean

There is a known gap in the theory that makes both outcomes informative rather than one being a
failure. Under AdamW the per-coordinate step is ≈ `η·sign(g_i)`, so warm-start displacement
measures `η|Σ_t sign(g_{i,t})|` — gradient **sign persistence**, not gradient **magnitude**. φ
measures alignment with initial gradient magnitude. These two can come apart.

- **φ ≫ chance.** Selection captures gradient energy. Direct support for non-arbitrary selection;
  the simple story, and the §D.3 bridge from `s_warm` to `s*` stands roughly as written.
- **φ ≈ chance.** Selection is *not* gradient-magnitude driven. Because these masks demonstrably
  work — the sparse arms train and the oracle/warm arms track the dense run — this is **positive
  evidence for the sign-persistence mechanism**, not a negative result. It also localizes the
  repair: §D.3 must bridge `s_warm` to `s*` through sign persistence rather than through
  magnitude.

Neither branch will be reported as a surprise or reframed after the fact.

### Secondary prediction, registered now

Figure 2 shows DPO and GRPO masks select near-disjoint coordinates. If objective-determines-
structure holds in gradient space, then against a **DPO** gradient:

```
phi(oracle_grpo_openr1) < phi(oracle_dpo_tulu3)
```

and the GRPO arms should sit closer to chance than the DPO arms do. If instead the GRPO masks
capture DPO gradient energy just as well as the DPO masks, the cross-objective transfer gap in
Figure 3 is *not* explained by gradient-space misalignment, and we say so.

### Reserve measurement — conditional, not run upfront

Only if φ ≈ chance: the Olmo3 delta logs carry snapshots every 50 steps, so per-coordinate sign
agreement across the ten intervals is computable on CPU with no new training. That tests mask
alignment against sign persistence directly, instead of against magnitude. Held in reserve
deliberately — running it regardless would make it look like a fishing expedition for whichever
metric happened to work.

### Stated assumption

The Tülu3 dense run that produced these masks has been deleted from scratch, so its exact
batch-shape configuration is not recoverable. `g` is therefore computed at bs 2 × 64,
`max_length` 1024, `max_prompt_length` 512, β 0.1, seed 42 — the Olmo3 arms' effective batch and
`DPO_train.py`'s defaults. φ is a ratio of norms of the same gradient, so it is insensitive to the
overall scale of `g`; it is sensitive to *which* examples `g` averages over, which is why the
batch is fixed by seed and reported.

---

## E2 — does certifiability predict trajectory divergence?

### Quantities

- `V(M) = 1 − certifiability_strict_fraction` — the fraction of weights violating the Theorem 2
  per-weight condition, from `mask_score_gap_gap_diagnostics.json` under the **hybrid** τ rule
  (`cert_tau_rule=hybrid_global_phase`, `cert_hybrid_min_layer_keep_ratio=0.0025`), so the
  estimator matches how production masks are actually built.
- `d(M) = mean_t | L_M(t) − L_oracle(t) |` over the final 100 steps (401–500), joined on `step`,
  against the oracle-deltalog arm's trajectory. Reported alongside the paired trace
  `Δ(t) = L_M(t) − L_oracle(t)` with a block-bootstrap CI over contiguous step blocks.

Model/dataset: Olmo-3-7B-Instruct / Light-R1, ρ = 97.5 %, five arms identical on every training
field including `max_steps=500`.

### Point set, fixed in advance

| arm | V | d |
|---|---|---|
| oracle (deltalog) | 0 | 0 — **definitional anchor, not a data point** |
| warm k=50 | measure | measure |
| warm k=100 | measure | measure |
| warm k=250 | measure | measure |
| random (seed 42) | measure | measure |

Four measured points. V will additionally be available at k ∈ {150, 200} from the same streaming
pass at no extra cost; those are reported as **V-only** points, marked distinctly, supporting the
flatness check but carrying no paired `d`.

At k = T, `s_warm ≡ s_oracle`, so the oracle's V = 0 and d = 0 hold **by construction**. It is a
legitimate anchor and not evidence, and will be labelled that way in every figure and table.

### The prediction is flatness, not a slope

The surviving Llama V(k) values (global τ) are 0.16985, 0.16829, 0.16614, 0.16443 at
k = 50, 100, 150, 200, with random at 0.51164. A **fourfold** change in trajectory length moves V
by 0.0054 — about 3 % relative. So:

- **Registered prediction:** if V governs trajectory behaviour, `d` should be similarly flat
  across k ∈ {50, 100, 250} and separated sharply for random. Line 287 of the paper already claims
  warm-start is a strong oracle estimator even at low k; flat V is the proposed *mechanism* for
  that claim.
- The informative variance in V is **between scoring families** (~0.166 warm vs ~0.51 random), not
  across k. That is where the lever arm is, and it is why the point set is a tight k-cluster
  against a random anchor rather than an even sweep.
- **This is why we are not training k ∈ {150, 200}.** Four runs spanning V ∈ [0.164, 0.170] would
  be a regression on a constant; any fitted slope would be noise. If `d` *did* vary across those k
  while V did not, that would be evidence *against* the V→d link, which is the outcome the tight
  cluster is designed to be able to detect.
- **Caveat registered in advance:** those numbers are Llama/Light-R1 under global τ. Olmo3 under
  hybrid τ may differ. Flatness is a hypothesis to test here, not an assumption. If Olmo3 V(k)
  shows real spread, the k-cluster becomes *more* informative than expected — good either way.

### Statistics, decided before seeing the numbers

With n = 4 measured points we report the scatter and the qualitative relationship. **No regression
coefficient and no p-value will be reported on n = 4**, in either direction.

### Null result is publishable

If V does not predict d, the honest finding is that certifiability margins do not translate into
trajectory behaviour at this sparsity — a real statement about the reach of Theorem 2. That gets
written up as the result. No fishing for a slope, no post-hoc arm exclusion beyond the two gates
below.

### Validity gates (blocking, decided in advance)

1. **Config-drift guard.** Arms must agree on `max_steps, learning_rate, lr_scheduler_type,
   warmup_ratio, max_grad_norm, gradient_accumulation_steps, per_device_train_batch_size, beta,
   seed, data_seed, max_length` and on `accelerator_config.use_seedable_sampler`, read from
   `training_args.bin`. One whitelisted difference: `optim` (`adamw_torch_fused` logged for sparse
   arms vs `adamw_8bit` dense) — for sparse arms the logged value is a red herring, since the real
   optimizer is `SparseAdamW` passed via `optimizers=`. Mismatched pairs are **refused**, not
   warned about.
2. **Step-1 `grad_norm` precondition.** Must match across a pair (observed 70.0 across dense,
   random, magnitude-k250 and oracle-deltalog). A mismatch means data order or init diverged, and
   that pair is **excluded, not caveated**. Noted honestly: this scalar is bf16-quantized (70.0 and
   70.5 are adjacent representable values near 70), so it is a cheap smoke test rather than a
   precision check.

### Excluded in advance

**No GRPO arm appears on the V-vs-d axes.** Sparse GRPO runs differ from dense in four
uncontrolled ways at once (`max_grad_norm` 1.0 vs 0.1, an extra per-parameter clip, linear vs
cosine schedule, `SparseAdamW` vs `adamw_8bit`). GRPO masks appear in E1 only, where none of that
applies because no training is involved.

The Olmo3 `oracle_ckptdiff` arm is also excluded: its step-1 `grad_norm` is 70.5 against 70.0 for
every other arm, so it fails gate 2. It was never in the point set.
