# Do Not Repeat

Failed, misleading, or disproved lines of work. Add an entry so nobody — human or
agent — proposes them again after the reasoning that killed them has scrolled away.

**Format:** start each entry with **the variable that was changed or the claim that was
made**, then the measured outcome, then why it must not be repeated. Never delete an
entry; historical failures are the point of this file.

Seeded 2026-07-26 from the NeurIPS #29841 rebuttal cycle. Verified findings live in
`docs/paper_drafts/VERIFIED_FINDINGS.md`; this file is only the negative space.

---

## Claims that were asserted and later disproved

### "The abstract's 3.3× speedup is measured at 99.75%, so the abstract overstates."
**False.** At ρ=97.5%: `adamw_8bit` 14.1286 ÷ `sparse_adamw` 4.2141 = **3.35×**. The
abstract says "∼1.5× and ∼3.3× over PyTorch and HuggingFace 8-bit AdamW respectively at
97.5% sparsity" and is exact; the ∼1.5× against PyTorch actually *understates* the
measured 1.69×. The 3.28× that looks like the source is the 99.75% row's
sparse-vs-`adamw_torch` figure — a numerical coincidence.
**Do not concede this.** Conceding would falsely confirm the overstatement charge
already in the meta-review.

### "Theorem 1's bounded-gradient assumption is used but never stated."
**False.** It is in the theorem's hypotheses (submitted PDF p.4 l.151, `‖∇L(θ)‖ ≤ ε`).
Only the `_inf` subscript is missing, supplied at l.694. Do not apologise for an
omission that did not occur.

### "Under Adam, parameter-space displacement scales as √n_active, so a sparse arm at matched LR is under-stepped and the comparison is conservative."
Premise correct, **inference wrong.** Under the same quadratic model, per-step descent
gives `η* ≈ Σ|g_i| / Σ H_ii`; both sums are extensive in |M|, so the optimal learning
rate is roughly **invariant to density**. Matched LR is the principled default, not a
self-imposed handicap. Asserting otherwise would also make any ρ-sweep a confounded
two-factor design (ρ=90% would "need" 3.16×, ρ=99% 10×). Use the pruning convention
instead (Frankle & Carbin ICLR 2019; Renda et al. ICLR 2020: retain the dense schedule).

### "~90% of the GRPO optimised signal is formatting."
**False**, because it was computed from reward *level*. GRPO learns from **within-group
variance**: a reward component that is constant across the 8 rollouts contributes
exactly zero gradient regardless of magnitude. By the honest measure
`accuracy_std / reward_std`, accuracy supplies **27% early and 42% by step 500**.

### "GRPO's accuracy_reward is structurally zero when clipped_ratio reaches 1.0."
**False.** It still registered 0.125 in exactly that step. The problem is its share of
the signal, not a hard zero.

### "The anti-oracle mask already exists, so the bottom-2.5% control is one training run away."
**False.** `anti_oracle_dpo_lightr1_step500_sp97.5.pt` has measured density **97.5%** —
it is the complement `1 − M` used by the probe necessity analysis. A bottom-k-by-|Δθ|
mask does not exist and **cannot be made by inverting the mask**; it requires the score
tensor.

### "MMLU is identical to four decimal places across all conditions" / "no condition moves on Tülu3."
Both **false**. MMLU spans 0.6871–0.6877 (four distinct values); IFEVAL moves
0.7375 → 0.7505 and MATH 0.4322 → 0.4376. The defensible claim is "no condition moves
by more than ~1.3 points and the between-condition spread is smaller than that."

---

## Experiments that must not be run as designed

### Changed variable: sparsity ρ, for GRPO, before fixing the schedule
A ρ sweep (95/90/70%) on GRPO **reproduces the Appendix D.4 artifact**, because the
confound is a scheduler mismatch that is *independent of ρ*: dense used cosine over
`max_steps=5000`, the sparse arms linear over 500, giving a **449× learning-rate gap at
step 500** with the two arms moving in opposite directions. Fix the schedule first
(`--lr_scheduler_type` now exists on `sparse_grpo_bsr.py`), then sweep.

### Changed variable: ρ = 70% for GRPO, taken from Mukherjee's natural-sparsity estimate
Our own measured concentration curve says GRPO's update mass saturates by keep-10%.
ρ=70% keeps 30% — three times past saturation. Success there demonstrates only that
training more parameters works. The informative points are the predicted knee,
ρ ∈ {95, 90}%.

### Changed variable: base model → Qwen3-32B
~7 min/step, ≈58 h per phase against an 8 h walltime cap. Stalled at dense 250/500 since
2026-06-14. Cannot complete. Do not report a partial run as a matched pair.

### Changed variable: base model → a third same-family dense decoder (e.g. Mistral-7B)
Three serialized stages for an architecture Llama and Qwen already represent, reporting
into an evaluation that does not yet discriminate. Adds a table row, not evidence.

### Changed variable: compute provider → rented cloud GPUs for GRPO
Investigated 2026-07-25, nothing provisioned. The hosted "Training" products train LoRA
adapters — the thing this work compares *against* — and raw rental needs the conda env
rebuilt plus ~23 GB moved on billed time. The binding constraint is a single contended
H200 node, and paying does not fix the parts that were already answerable by analysis.

### Changed variable: scoring function → static |θ| or SNIP with a pre-registered Theorem 2 prediction
No certifiability-margin code exists in the repo, and the prediction would come from a
theorem whose optimal family does not contain the deployed scorers. If one scorer is
added, make it **Fisher / FISH Mask** (Sung et al., NeurIPS 2021) — Theorem 2 already
invokes `H_ii ≈ F_i`, so omitting it is the most demandable gap. Gradient-based scorers
also need a real calibration set: the convention is 128 sequences (SparseGPT, Wanda),
not 2.

### "Scott's 5-way sweep already has GraSP arms, so item 3's scoring-function gap is over-conceded"
**False — checked 2026-07-26, no GraSP mask exists anywhere.** The infrastructure is
real and complete: `src/cold_start/utils/grasp_scorer.py`, five sbatch entry points
(`run_grpo_grasp_vanilla_vs_snr_abs.slurm`, `run_grpo_grasp_mask_only.slurm`,
`run_dpo_grasp_masks_only.slurm`, `orchestrate_light_r1_grasp_sweep.slurm`,
`run_light_r1_grasp_elem_base_mask.slurm`), and two locked arms in
`docs/hyperparams/grpo_500step_5way_sweep.yaml` (`grasp_abs_no_snr`,
`grasp_abs_snr_per_weight_log1p`). The **output** is not:

- `/scratch/biggs.s/rl_casino_masks/orch_lr1_grasp6_6376972` and `..._6377908` — despite
  the name, contain **only** `random_elem` / `random_block256_mean` masks.
- `/scratch/biggs.s/rl_casino_masks/tulu3_..._grasp_elem_base_..._20260503_164435` —
  **empty**, `total 0`.
- The only sparse training runs downstream of `grasp6` are named
  `dpo200_sparse_lr1_random_elem_rerun_*`, i.e. the random control.
- `find` over Scott's `rl_casino_{masks,sparse_train,train}` returns **zero** files with
  `grasp` in the name — only directories.
- An exhaustive `find /scratch/xie.yiyi -iname "*grasp*"` returns **zero hits of any
  kind**, so nothing was produced on Irene's side either.

The commit trail says why: `f50862b` "GRaSP and SNIP save me please bro" (04-27),
`dbbebbc` "testing GraSP implementations. Please work..." (04-29), `5f30ba3`
"relaunching GraSP jobs due to a bug in the torch autograd 2nd order implementation in
the backend" (04-30), `65febd5` "time boost for grasp calculation" (05-03). GraSP is
never mentioned again after 2026-05-04.

**Root cause is structural, not a passing bug.** GraSP scores `−w ⊙ (Hg)`, so it needs a
Hessian-vector product: `create_graph=True` then a second `autograd.grad` through an 8B
transformer. Flash and mem-efficient SDPA do not implement higher-order autograd (hence
`_sdp_force_math_backend_cuda` in the scorer), and reentrant gradient checkpointing is
incompatible with `autograd.grad` (hence the `use_reentrant=False` branch). Math-backend
attention plus a retained graph is a large VRAM multiple over a normal backward pass.

**Do not "just rerun Scott's GraSP arms" and do not soften item 3's concession on the
strength of them existing in a yaml.** If a scorer is to be added, the ledger entry above
still applies: make it Fisher / FISH Mask, which needs no second-order autograd.

---

## Infrastructure traps that have cost time more than once

### `--requeue` does not fire on TIME LIMIT
An 8 h wall hit is `CANCELLED`, not requeued. This is why the Qwen3-32B chain never
advanced past its first checkpoint. Wrap training in `timeout` and self-resubmit from
the job tail instead.

### `sacct State=COMPLETED` is not proof the work happened
Three `delta_analysis` array tasks reported `COMPLETED 0:0` having produced nothing,
because the sbatch did not propagate python's exit code. Always `exit $RC`, and verify
the artifact rather than the state.

### Requesting more memory than needed delays scheduling
A 256 GB ask on the `short` partition sat in the queue; the same job at 96 GB started in
under a minute. Right-size the request.

### `/tmp` on the cluster contains a stray `inspect.py` that shadows the stdlib
Running any python script from `/tmp` fails at `import torch`. Run from the repo.

### `run_manifest.json` does not record sequence caps, scheduler type, or max_steps
So the as-run configuration is **not** recoverable from the project's own records. Read
`training_args.bin` from the checkpoint instead — `scripts/read_training_args.py`. Three
Table 6 discrepancies were invisible until someone did.
