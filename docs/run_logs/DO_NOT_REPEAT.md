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

### "Figure 3 does contain GRPO results — the legend just mislabels them."
**False, and it is a tempting mistake.** Reviewer XD33's "there are no GRPO results in
the figure" is **correct**. Every curve in Figure 3 is a **DPO training run**; only the
origin of the mask varies. `DPO Oracle OpenR1` means *DPO training under a mask derived
from GRPO dense fine-tuning* — the label names the training objective and the mask's
dataset, never the mask's objective. Table 1's "Sparse · oracle GRPO Math" row is the
same thing (`transfer_v1/sparse_dpo_lightr1_oracle_grpo_math`). Answering the complaint
with "the curve is there, it is just mislabelled" is an evasion of a valid point and
would read as one.

The real answer is that sparse **GRPO training** runs exist and were never plotted in
that figure: `transfer_v1/sparse_grpo_math220k_oracle_{grpo_math,dpo_lightr1_500,
dpo_tulu3}`. See the entry below on what they show.

### "GRPO is resistant to masking" (Appendix B/D.4) — SETTLED FALSE 2026-07-30
Under a **matched schedule** (cosine over 500, warmup 50, lr 5e-6, β 0.025, global clip
0.1 on all arms, cap 2048, ρ=97.5%), three AICR arms of 500 steps each — dense `235271`,
sparse oracle `235463`, sparse random `235303`, all ending at `learning_rate` 6.0923e-11:

| last-50 total reward | gap | SE |
|---|---|---|
| dense − oracle | **−0.0225** | 0.6 |
| oracle − random | **+0.0687** | 1.8 |

Sparse oracle is **indistinguishable from dense** and consistently ahead of random (9 of
10 windows). KL displacement over training: oracle **6.1×** (0.00035→0.00214) vs random
**1.3×** (0.00026→0.00035) — the random subnetwork is nearly frozen while the oracle
tracks dense. **Do not repeat "GRPO resists masking" or cite Fig 8/9 as evidence for it**;
that figure's dense/sparse arms differed in scheduler, horizon, and global clip. Full
numbers: `grpo_matched_aicr_2026-07-29.md` §4a.

### Sparse GRPO at ρ=97.5% learns — do not repeat "the method fails for GRPO"
The three runs above are single-variable (same model, `open-r1/OpenR1-Math-220k`,
`n_steps=500`, lr 5e-6, β=0.025, reward profile, per `run_manifest.json`) and **all end
at LR 1.111e-08**, so they share a schedule and are comparable to each other. Mean
accuracy reward, steps 0–50 → 450–500:

| mask driving sparse GRPO | acc. reward | total reward Δ |
|---|---|---|
| in-task GRPO Open-R1 | 0.0434 → 0.0833 | +0.0720 |
| DPO Light-R1 | 0.0587 → 0.0809 | +0.0807 |
| DPO Tülu-3 | 0.0561 → 0.0956 | +0.0355 |

So the Appendix D.4 "failure at 97.5%" is the dense-vs-sparse **schedule confound**, not
a property of GRPO. **Caveat that must travel with these numbers:** no random-mask sparse
GRPO run exists anywhere (`find` over `transfer_v1` and Scott's `rl_casino_sparse_train`),
so they establish *that* sparse GRPO trains and can never establish *which mask is
better*. Do not use them to rank masks.

### "Sparse DPO runs at 48.3 s/step, so LoRA (45.8) is ~5% faster."
**False, and it was in two rebuttal replies until 2026-07-26.** 48.29 s/step is the
**LoRA r=16 smoke run** (job 8725068) — it was never a sparse measurement, and the
figure was carried into a wall-clock concession to LoRA that the data does not support.
The real numbers, all 500 steps of Light-R1 to the same 20.86 epochs on H200s:

| arm | train_runtime | s/step | GPUs | job |
|---|---|---|---|---|
| dense DPO | 6,092.6 s | 12.19 | 4 | 6366610 |
| sparse ρ=97.5% | 22,612.2 s | 45.22 | 1 | 6512341 (d4052) |
| LoRA r=64 | 22,916.2 s | 45.83 | 1 | 8727274_0 (d4053) |

Sparse and LoRA are at **parity** (1.3% apart) on identical hardware. Do not concede a
wall-clock loss to LoRA. Note dense's GPU-hours (6.77) include data-parallel overhead,
so the ~7% GPU-hour edge for sparse (6.28) is not a clean throughput win — the
defensible claim is that sparse fit on one GPU where dense needed four.
**Method note:** neither `run_manifest.json` nor `trainer_state.json` carries
`train_runtime`; it is printed in the job's stdout, and node GPU type comes from
`sacct -o NodeList` cross-referenced with `sinfo -N -o "%N %G"`.

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

**SUPERSEDED 2026-07-30 — the sweep was run anyway (Irene's call) and the plateau is the
result.** Seven matched-schedule levels, all ending at LR 6.0923e-11: ρ = 70/80/90/95/
97.5/99 are **all within 1.5 SE of dense**, and only ρ=99.75 breaks (reward −0.0237,
KL fold-change collapsing 6× → 2.0× toward random's 1.4×). So the GRPO oracle tolerates
a parameter budget **100× smaller** than Mukherjee's 70% anchor implies, and the "lower
bound" question is answered at ρ≈99.75, not below 97.5. Two lessons: (a) a saturated
level is uninformative *as a mechanism probe* but can be the strongest *rhetorical*
evidence — a measured plateau beats an argument about why the point is uninteresting;
(b) reward saturates across 70–99, so **KL displacement is the metric that resolves
sparsity tolerance**. Numbers: `grpo_rho_sweep_aicr_2026-07-29.md` §4a.

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

## `hf download` on the AICR login node hangs with zero bytes written

**Measured twice: 2026-07-29 and again 2026-07-30.** `hf download Qwen/Qwen3-32B
--max-workers 8` starts, the process stays ALIVE indefinitely, and **no cache
directory is ever created** — the only line in the log is an unrelated urllib3
version warning. `curl -sI https://huggingface.co/Qwen/Qwen3-32B/resolve/main/config.json`
returns **307 in 0.33 s** from the same login node, so a reachability check on
huggingface.co does **not** prove the transfer path works; the CDN/xet endpoint
that actually serves bytes is what stalls.

**Consequence that cost a full chain:** `qwen3_32b_aicr_resume_2026-07-29.md`
recorded "Qwen/Qwen3-32B weights pulled directly from HF on AICR (network + token
verified)" on the strength of that reachability check. The weights were never
there. Job `235953` queued, started, loaded the dataset, and died at
`AutoTokenizer.from_pretrained` 61 seconds in; the `afterok` chain read the
non-zero exit as a hard crash and cancelled all six downstream jobs.

**Do not** verify a model's availability with a HEAD against huggingface.co, and
do not record a model as "pulled" without listing the cache. The working route is
the **chunked-dd transfer from Explorer's cache**, which already holds the model
(`/scratch/xie.yiyi/hf_cache/hub/models--Qwen--Qwen3-32B`, 27 blobs, 65.5 GB):
`~/xfer_qwen32b_base.sh` moved it in ~8 minutes at ~200 MB/s with zero chunk
failures. The HF cache is content-addressed, so the transfer must also replay
`snapshots/<rev>/*` symlinks, `refs/main`, and the `.no_exist/<rev>/` negative
lookups — blobs alone do not make a resolvable repo.

**Preflight, now enforced:** `scripts/preflight_base_model.py <model_id>` resolves
config → tokenizer → weight index → **every shard named in the index**, and both
`aicr_qwen32b_p1_dense.sbatch` and `aicr_qwen32b_p3_sparse.sbatch` call it before
training. A config-only check is not sufficient: the small files can be present
while a 3.9 GB shard is missing, which fails ~30 min in rather than at second 5.

## A resumed HF run inherits `save_steps` from the checkpoint, silently ignoring the CLI

**Found 2026-07-30 on the Qwen3-32B p1 resume (job 238613).** The sbatch passes
`--save_steps 10`; the run trained past steps 260, 270 and 280 writing **no
checkpoint at all**. Not a broken save path — `DefaultFlowCallback.on_step_end`
tests `state.save_steps`, and `TrainerState.load_from_json` restores that field
from the checkpoint's `trainer_state.json`, where the original Explorer run had
recorded **50**. HF prints the mismatch and then proceeds with the checkpoint's
value:

```
Warning: The following arguments do not match the ones in the `trainer_state.json`
	save_steps: 10 (from args) != 50 (from trainer_state.json)
```

That warning is the only signal, and it scrolls past during startup.

**Why it matters for a `timeout`+resume chain:** the design assumes a slot that
hits its wall loses at most `save_steps` of work. Inheriting 50 instead of 10
raises the worst-case loss to 49 steps — at the 32B rate (105 s/it) that is 1.4 h
of compute per slot boundary. It also *reduces* total save overhead, so the effect
on a budget is mixed, not simply bad.

**Check `state.save_steps`, not the launcher, when planning slot boundaries for
any resumed run**, and read the mismatch warning rather than assuming the CLI won.

## `pkill -f "<pattern>"` matches its own ssh command line

`ssh host 'pkill -f "hf download"; ...'` kills the remote `bash -c` wrapper itself,
because the wrapper's command line contains the pattern. The ssh exits 255 and the
rest of the compound command never runs — so the check that was supposed to
confirm the kill never executes. Use a bracket pattern (`pgrep -f "[h]f download"`)
for the *verify* step, and prefer killing by PID captured beforehand.

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

### `--max_grad_norm` reached SparseAdamW but never TRL — dense/sparse GRPO were 10x apart
**Found 2026-07-27, same class of defect as the D.4 scheduler bug.** In
`sparse_grpo_bsr.py` the CLI `--max_grad_norm` was passed to `SparseAdamW` (which clips
**each tensor separately**) but was never put into `GRPOConfig`, so TRL's **global** clip
silently stayed at the HF default **1.0**. `GRPO_train.py` meanwhile passes its
`--max_grad_norm` (0.1) straight into `TrainingArguments`. So for any dense-vs-sparse GRPO
pair:

| | global clip | per-tensor clip |
|---|---|---|
| dense (`GRPO_train.py`) | **0.1** | — |
| sparse (`sparse_grpo_bsr.py`, before fix) | **1.0** (HF default) | 0.1 |

**It binds on nearly every step**, so this is not academic. Measured `grad_norm` from
`trainer_state.json`: sparse in-task GRPO median **1.21**, 92.2% of steps above 1.0;
dense matched arm 8734159_0 median **2.98**, 86.7% above 1.0.

**Fixed** by wiring `max_grad_norm=max_grad_norm` into `GRPOConfig`, driven by the same
CLI value, so a job passing 1.0 is unchanged and a job passing 0.1 becomes matched. After
a global clip to t, no tensor can exceed t, so SparseAdamW's per-tensor clip becomes a
no-op rather than a second, different clip.

**Do not compare a dense GRPO arm against any sparse GRPO arm that ran before
2026-07-27** without checking this. Sparse-vs-sparse comparisons are unaffected — every
sparse arm shares the same clipping — so the 2026-04 three-arm set and its random control
(8769965, which passes 1.0) remain internally valid.

### SparseAdamW's weight_decay default is 0.01 and no GRPO/DPO entry point overrides it
`SparseAdamW.__init__` defaults `weight_decay=0.01`; `sparse_grpo_bsr.py` never passes the
argument and exposes no flag, while `GRPO_train.py` defaults `--weight_decay 0.0`. So every
sparse arm carries decay 0.01 against dense's 0.0. **Left unfixed deliberately:** at
lr 5e-6 the per-step factor is `1 - lr*wd = 1 - 5e-8`, i.e. ~2.5e-5 relative shrinkage over
500 steps — far below bf16 training noise, and changing it would break comparability with
every existing sparse arm. Document it; do not "fix" it without rerunning the whole set.

### The optimizer kernel forked from Scott's, and ours recompiles every step (294x measured)
Same file, same kernel name, **different implementation**. `origin/cav_fixes`'s
`src/kernels/indexed_sparse_adam.py` carries four kernels (indexed + two BSR variants);
this branch has only the indexed one, and that one differs:

| | cav_fixes | this branch |
|---|---|---|
| addressing | `M, N` + full strides, decodes row/col | assumes contiguous flat, uses flat index |
| moment buffers | `USE_SPARSE_STATES` allows **packed** state | always full-size |
| lr / beta / eps / wd | **runtime args** | **`tl.constexpr`** |

`tl.constexpr` values are part of Triton's JIT cache key and **lr changes every step under
any scheduler**, so ours recompiles once per step. Measured on one H200 (job 8770992,
4096x4096, 2.5% kept): constant lr **0.93 ms/step**, varying lr **273.56 ms/step** —
**294.6x**. The file's own docstring says "without kernel recompilation" and the bias
corrections are deliberately non-constexpr for that reason; lr was missed.

**Do NOT swap the kernel mid-campaign.** Three reasons:
1. **Correctness is unaffected** — constexpr vs runtime arg changes compilation, not
   arithmetic. Every existing result stands.
2. **The cost is already inside every s/step number we have**, because every real run used
   a real scheduler: sparse DPO 45.22 s/step, sparse GRPO 22.6 s/step. One compile per
   step is shared across all ~290 param tensors (they share the step's lr), so it is
   ~0.27 s on a 22-45 s step, i.e. 0.6-1.2%. Nothing needs re-measuring.
3. Changing it would break time-comparability with the 2026-04 arms for a ~1% gain.

**But it does bite one claim.** `scripts/microbench_optimizer_step.py` — the source of
Table 2/3 and the abstract's 1.69x / 3.35x — runs at a **single fixed lr** (`lr: 5e-07` in
`microbench_optstep_scott_2026-06-21.md`) with no scheduler, so the kernel compiles once
there. The reported 4.2141 ms optimizer step is a regime that never occurs in training,
where the same step costs ~277 ms. The kernel's O((1-rho)P) asymptotics are unaffected;
the headline speedups do not describe deployed behaviour. The rebuttal's existing hedge
("the optimizer step is a small fraction of total step time, so end-to-end gains stay far
below the 1.69x/3.35x kernel figures") remains true and is now better grounded.

**Fix, when the campaign is over:** make lr a runtime arg, as cav_fixes already does.

### At cap 1024, ~90% of GRPO completions are truncated — that is why accuracy reward looks low
From `trainer_state.json` of the dense matched arm 8734159_0 (cap 1024):
`completions/clipped_ratio` mean **0.90**, `completions/mean_length` mean **973.9** of a
1024 cap, `completions/max_length` **1024.0 at every step**. Completions that terminate
naturally average only **413** tokens — so the 90% that hit the cap are cut off mid-reasoning
and can never emit a boxed answer.

**Consequences to keep straight:**
- The low absolute `accuracy_reward` in every cap-1024 GRPO run (~0.04–0.10) is largely a
  truncation artifact, not a statement about what the model can do. Do not describe those
  levels as the model's math ability.
- The *within-run* rise (e.g. 0.0434 → 0.0833 for the in-task sparse arm) is still valid:
  both endpoints sit in the same truncation regime, so the comparison is internally sound.
  Only the absolute level is suppressed.
- This is why Scott's locked `grpo_500step_5way_sweep.yaml` uses cap 2048 and why 39ce420
  moved the T-sweep to it. The T-sweep runs will NOT be comparable in absolute reward to
  the 2026-04 cap-1024 arms.
- Walltime is still fine: 8734159_0 ran 17.7 s/step at cap 1024, so even a 2x generation
  cost at cap 2048 gives ~35 s/step, ~4.9 h for 500 steps against the 8 h limit.
- The random-mask control 8769965 deliberately keeps cap 1024 to stay single-variable
  against the 2026-04 arms, and therefore inherits this truncation regime by design.

### `sparse_grpo_bsr.py` / `sparse_dpo_bsr.py` were ImportError-dead on this branch from 2026-04-30 to 2026-07-29
Commit `68dff0c` "Revert SparseAdamW + sparse kernels to origin/main" reverted
`bsr_backward.py` + three optimizer files but **not** `src/mlps/bsr_sparse_mlp.py`, which
kept importing the B1-era `sparse_grad_input_triton` — a symbol the reverted kernel no
longer defines. Both BSR entrypoints crashed at import for three months without anyone
noticing, because every sparse run in that window went through
`sparse_dpo_efficiency.py`. Found by AICR jobs 235272/235273; fixed by `39adbcb`
(restores the pre-B1 pair). The same partial revert had a **second face**: the
entrypoint passed the B1-era `eager_state_init=` kwarg to the reverted `SparseAdamW`,
a `TypeError` that only fires at optimizer construction — i.e. AFTER a clean import,
~3 min into the job (235292/235293; fixed by `f315ae4`). **Lesson: a revert must cover
every caller of the reverted API, not just the file that changed. The import check
alone is not enough — also assert call-site kwargs against
`inspect.signature(SparseAdamW.__init__)` — the preflight that gated the third
submission (235302/235303).**

### Mask generation on a first-gen EPYC (`zen`) node is 6x slower and ρ=80 cannot finish at all
**Measured 2026-07-28/29.** Job `8817043` backfilled onto **c2205** in 63 s against a
`--test-only` estimate of 03:46 — but c2205 is `AvailableFeatures=zen`, an **AMD EPYC
7351** (first-gen Naples, 2.4 GHz). The global top-k over 8.03 B elements is
**memory-latency bound**, and that is the worst node generation on the cluster for it:

| ρ | Tulu3 half (`short` nodes) | c2205 (`zen`) | ratio |
|---|---|---|---|
| 99 | 0:10:04 | 0:45:38 | 4.5x |
| 97.5 | 0:24:28 | 2:32:27 | **6.2x** |

Ruled out: memory pressure (403 G available of 471 G, **zero swap**, MaxRSS 114 G against
a 200 G ask) and core starvation (31 threads, ~280% CPU of 16 allocated cores).

**This is a hard block, not a delay.** Linear-in-keep extrapolation — optimistic, the real
scaling is superlinear — gives ρ=90 ~7.6 h and ρ=80 ~15 h against the **8 h hard cap**.
ρ=80 cannot complete on a `zen` node at any walltime request.

**Always pass `--constraint="zen2|cascadelake"` for 8B mask generation.** Node generation
is a Slurm feature and is selectable. Two escape routes that do NOT work, both measured
the same night: `short` is quoted 2026-08-03 *even with the constraint*, and `multigpu`
rejects a no-`--gres` job outright (`allocation failure: Access/permission denied`), so
CPU-only work still has to hold a GPU there.

Corollary: **`sbatch --test-only` is an upper bound, not a placement plan.** It quoted
03:46 and the job started in 63 s — onto exactly the node class that made the work
impossible. Fast backfill usually means an *unwanted* node was idle.

### The delta route does not save the memory its header claims
`mask_from_delta.sbatch` says scoring `|deltas_step_N|` in place needs ~64 G because it
avoids holding `initial_sd + final_sd + scores`. **Measured MaxRSS was 114 GB through
ρ=97.5** — *above* the checkpoint-diff route's 105 GB at the same ρ. The reason is
`create_mask_from_scores_gpu_efficient`, which does `s = score.to(...).clone()` per tensor
([mask_utils.py:623](../../src/utils/mask_utils.py#L623)): a second full 32 GB copy is
materialised inside the selector no matter how careful the caller is. Size these jobs from
the Tulu3 measurements (199/161/105/98 GB for ρ=80/90/97.5/99), not from that header.

### A bare `--gres=gpu:1` for an 8B eval can land on a V100 and blow the walltime
Job 8786294 (held-out preference eval for LoRA arm2) asked for `--partition=multigpu
--gres=gpu:1` and Slurm assigned **c2207 = `v100-pcie:2`**. V100 is Volta: no bf16, and
PCIe rather than SXM. The identical eval for arm0 (8763502) ran on **d1028 = `a100:4`** in
**3m12s**; on the V100 it finished pass 1 and reached 104/500 of pass 2 before a 30-minute
wall killed it — roughly **10x slower**. It was not stuck, just on the wrong silicon.

**Rule:** anything doing 8B forward passes — preference eval, probe extraction, mask
scoring on a loaded model — must pin the GPU type (`--gres=gpu:a100:1` or `h200:1`), not
just a count. The existing playbook note "mask-gen/eval can stay on bare `--gres=gpu:N`"
is too permissive and is superseded for any job that loads an 8B model.
Resubmitted as 8787043 with `a100:1` and a 1 h wall.

---

## Submitting mask-generation jobs as if they were training jobs (2026-07-28)

Four Light-R1 oracle-mask jobs were submitted with `--gres=gpu:h200:1 --time=06:00:00`
and per-rho `--mem` of 110/180/220 G. All four sat PENDING; a 34-hour estimate. Three
recorded rules were broken at once:

1. **`feedback_must_use_h200` exempts mask generation.** The rule locks H200 for anything
   calling `optimizer.step()`. `checkpoint_diff_mask_finder.py` is named in the exemption
   list and should use a bare `--gres` (or none) "and benefit from the broader queue".
   Locking it to H200 put a no-optimizer job into the most contended pool on the cluster.

2. **`feedback_rc_idle_gpu_cancel` says this job class belongs on a CPU partition.**
   "Run GPU-free steps (oracle ckpt-diff mask) on a CPU partition (`--partition=short`,
   no `--gres`)." Taking an H200 for CPU-bound work then required a keep-alive sidecar to
   dodge the 15-minute idle canceller — inventing a problem and then patching it. The
   sidecar is documented for *unavoidable naive-MP training*, not for this.

3. **`feedback_cluster_backfill` says walltime must track measured runtime.** "Set to
   actual estimated runtime + small buffer. Going too long forfeits the backfill
   advantage." Measured mask-gen times were already in hand from the Tulu3 half:
   rho=99 10 min, rho=97.5 25 min, rho=90 1 h 31, rho=80 2 h 58. Asking 6 h for a 10-min
   job discards backfill entirely. `--exclude=d1025` was also omitted.

**Measured mask-gen cost (use these, do not re-guess):**

| rho | keep | elapsed | MaxRSS |
|---|---|---|---|
| 80 | 1.61 B | 2:57:40 | 199 GB |
| 90 | 0.80 B | 1:31:29 | 161 GB |
| 97.5 | 0.20 B | 0:24:28 | 105 GB |
| 99 | 0.08 B | 0:10:04 | 98 GB |

Memory tracks the keep count, so a single blanket `--mem` for the whole array is wrong in
both directions: 240 G starves the small levels of backfill opportunities, and a "safe
looking" 128 G would have OOM'd rho=80 and rho=90.

**Cheaper route that was not taken:** a saved `deltas_step_N.pt` already IS
theta(N) - theta(0), so scoring is one in-place `abs()` over a single 32 GB dict instead
of holding initial + final + scores. Same `create_mask_from_scores_gpu_efficient` call,
same parameters, provably the same mask. Not to be confused with
`even_better_mask_finder.py --method magnitude`, which accumulates
sum_k |theta(k) - theta(0)| over every logged step — a different score.

---

## Deduplicating Slurm submissions with `squeue -O "Comment"` (2026-07-29, AICR)

`launch_all.sh` guarded against double-submitting an arm with

```bash
squeue -u "$USER" -h -O "Comment" | grep -q "$TAG"
```

**The long form `-O` pads and truncates every field to 20 characters.** `TAG` for the
ρ=97.5 arm is `oracle_step500_sp97.5` — **21 characters** — so squeue returned
`oracle_step500_sp97.` and the grep never matched. `oracle_step500_sp99` is 19 and fit,
which is why **only the 97.5 arm duplicated**: a partition-wide guard that works for one
value of a loop variable and silently fails for another.

Measured on the live queue:

```
$ squeue -h -O "Comment" | awk '{print length($0), $0}'
20 oracle_step500_sp99
20 oracle_step500_sp97.
$ squeue -h -o "%k"     | awk '{print length($0), $0}'
19 oracle_step500_sp99
21 oracle_step500_sp97.5
```

Consequence: jobs `234945` and `234952` both trained the ρ=97.5 arm from the same mask
into the **same output directory**, and `234952` overwrote `wandb_run_id.txt` so the
recorded run id no longer pointed at the run that was actually training. `save_steps=50`
would have had them writing `checkpoint-50` on top of each other ~8 min later, and with
`--resume_from_checkpoint auto` any requeue could resume into the other process's state.
`234952` was cancelled at 15:04 elapsed / step 21, before any checkpoint was written.

Two mechanisms conspired: `autolaunch.sh` also invoked `launch_all.sh` **twice per tick**
(once to act, once to test its output for `[wait]`), so a hole in the guard produced the
duplicate inside the same tick rather than never.

**Rules:**
- Read Slurm comments with `-o "%k"`, never `-O "Comment"`. Match with `grep -Fxq`, so a
  tag cannot match a longer tag that merely starts with it.
- Any `squeue -O` field used for a decision must be width-checked first — the truncation
  is silent and the failure mode is a duplicate run, not an error.
- An idempotent submitter must be invoked **once** per tick. Capture its output and test
  the capture.
