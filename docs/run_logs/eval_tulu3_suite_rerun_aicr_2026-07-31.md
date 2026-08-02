# Run log — Tulu3 eval suite RERUN on AICR + LoRA rows (Table 1 rebuttal)

**Date:** 2026-07-31 (submitted ~21:45 EDT)
**Branch:** `irene-rebuttal-lora`
**Operator:** Irene (via Claude)
**Cluster:** AICR (`b200-batch`, 1×B200 per job)
**Eval driver:** `scripts/aicr_eval_tulu3_suite.sbatch` (new AICR port) → `src/evaluation/run_all_benchmarks.py`
**Predecessor:** `eval_tulu3_suite_2026-05-03.md` (same 7 arms on Explorer A100; this rerun adds 2 LoRA arms and moves everything to one pipeline on AICR)

## Why rerun

Rebuttal (NeurIPS #29841, due 2026-08-03) needs a LoRA row in Table 1 — the one
AC-flagged gap. Decision (Irene, 2026-07-31): re-eval the **original Table-1
checkpoints** (not the in-flight e1-e2 reruns, which lack the two cross arms)
plus the two existing Light-R1 LoRA arms, so every number in the table comes from
the same cluster/backend/suite on the same day.

## Submitted jobs (all 2026-07-31, b200-batch, 8h cap)

| TAG | Job ID | Model |
|---|---|---|
| `base_llama31_8b_instruct` | 246429 | `meta-llama/Llama-3.1-8B-Instruct` (HF cache) |
| `dense_dpo_tulu3_ckpt500` | 246430 | `$T1/dense_dpo_tulu3_llama8b_deltalog/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500` |
| `sparse_warm_mag_step200` | 246431 | `$T1/sparse_dpo_tulu3_warm_magnitude_step200/…_500steps/final_model` |
| `sparse_oracle_dpo_tulu3` | 246432 | `$T1/sparse_dpo_tulu3_oracle_dpo_tulu3/…_500steps/final_model` |
| `sparse_oracle_dpo_lightr1` | 246433 | `$T1/sparse_dpo_tulu3_oracle_dpo_lightr1/…_500steps/final_model` |
| `sparse_oracle_grpo_math` | 246434 | `$T1/sparse_dpo_tulu3_oracle_grpo_math/…_500steps/final_model` |
| `sparse_random` | 246435 | `$T1/sparse_dpo_tulu3_random/…_500steps/final_model` |
| `lora_r64_lr1e-4_lightr1` | 246436 | adapter `$S/rebuttal_lora/arm2_r64_lr1e-4/checkpoints/…light_r1_lora_r64_lr0p0001/checkpoint-500`, peft-merged in-job |
| `lora_r64_lr5e-6_lightr1` | 246437 | adapter `$S/rebuttal_lora/arm0_r64_lr5e-6/checkpoints/…/checkpoint-500`, peft-merged in-job |

`$S = /scratch/xie_yiyi_neu`, `$T1 = $S/transfer_v1`. The 5 sparse `final_model`
dirs (15 G each) + both LoRA adapter trees (2.9 G each) were rsync'd off Explorer
2026-07-31 (~20 min, 5 streams, `--partial --append-verify` retry loops against the
AICR login reaper); every dir verified for `config.json` + ≥4 safetensors shards
before submission. Dense ckpt-500 was already on AICR.

## Environment (NEW — differs from 05-03 deliberately)

- env: `/work/neu/p2026_0038_neu/xie_yiyi/envs/rl_casino_eval` (fresh, py3.12)
- **vllm 0.26.0, torch 2.11.0+cu130, lm-eval 0.4.12** — `eval_requirements.txt`
  pins (vllm 0.6.3/torch 2.4) cannot run on B200 sm_100 and were NOT used
- suite: all 7 benchmarks (mmlu, math, gsm8k, coding, ifeval, squad, gpqa_diamond),
  lm-eval defaults, `--apply_chat_template --trust_remote_code --batch_size auto --use_vllm`
- HF offline flags from `aicr_env.sh` unset in-job; mmlu/gsm8k/ifeval pre-cached,
  rest download at runtime (HF reachable from AICR compute; token on disk)
- Known risk: vllm 0.26 × lm-eval 0.4.12 interface drift → fallback is resubmit
  with `FORCE_HF_BACKEND=1`

## LoRA provenance (from arm manifests)

Both arms: Llama-3.1-8B-Instruct, **qihoo360/Light-R1-DPOData** (NOT tulu3 —
label the table rows "(Light-R1)"), 500 steps, DPO beta 0.1, adamw_8bit, r=64,
alpha=128, dropout 0, all-proj targets, 167.8M trainable (2.05%); arm2 lr=1e-4,
arm0 lr=5e-6. Trained 2026-07-26/27 on Explorer H200 (`lora_baseline_8b.sbatch`
arms 0/2).

## Outputs

- results: `$S/rebuttal_analysis/eval_tulu3/<TAG>_<jobid>/{*_results.json, all_benchmarks_summary.json}`
- slurm logs: `/work/neu/p2026_0038_neu/xie_yiyi/logs/eval_tulu3_<jobid>.out`
- merged LoRA snapshots: `$S/rebuttal_analysis/eval_tulu3/merged_lora_*` (delete after eval if space tight)

## Round 1 FAILED — all 9 jobs (246429–246437), ~5 min each

Root cause: compute nodes' `/lib64/libstdc++.so.6` lacks `CXXABI_1.3.15`; vLLM's
spawned engine-core resolved it ahead of the conda env's own libstdc++ and died
importing `sqlite3` (via `libicui18n.so.78`). Every benchmark subprocess failed the
same way; the runner still wrote an `all_benchmarks_summary.json` shell, which is
why file-existence monitoring called it success. Login node does NOT reproduce
(newer system libstdc++ there). Fix: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:…"`
in the sbatch. Junk output dirs deleted; merged LoRA snapshots were complete and kept.

## Round 2 FAILED too — 246501–246509, ~10 min each

libstdc++ fix held; next layer broke: vLLM 0.26 on sm_100 JIT-compiles
flashinfer's trtllm-gen fmha at engine init, and **compute nodes have no
/usr/local/cuda** → `nvcc: No such file or directory` (login node has CUDA 13.1
locally, which is why nothing caught it earlier). Fix: export
CUDA_HOME/PATH/LD_LIBRARY_PATH from the shared module install
`/apps/aicr/packages/cuda/13.1.1/extrybz` in the sbatch (no `module` cmd
dependency) + clear the stale `~/.cache/flashinfer/0.6.14/100a/cached_ops`
whose build.ninja had the /usr/local/cuda path baked in.

## Round 3 — all 9 TIMEOUT at 8h (partial results kept)

Stack finally worked (engine init clean, mmlu suite 15 min/arm) but every job
died at the 8h wall. Cause: **not slowness — filelock deadlock.** All 9 jobs
share `HF_DATASETS_CACHE` on /scratch (autofs); math/ifeval/squad/gpqa datasets
download at first use, and concurrent jobs deadlocked on datasets' filelocks —
7 jobs hung 7h+ right after the "Running MATH benchmark" header (before engine
init), and the 2 that won the math download race (246642, 246648) hung the same
way at ifeval. Second, independent bug: `coding_evaluator.py` monkeypatched
`lm_eval.evaluator.get_task_dict`, which no longer exists in lm-eval 0.4.12 →
coding failed fast on every arm.

**Salvaged:** mmlu for all 9 arms (base 0.6872 vs 05-03's 0.6874 — sanity check
#1 PASSES, suites comparable); math+gsm8k for dense (246642) and lora lr1e-4
(246648). Dense gsm8k 0.818 strict.

## Round 4 — offline reruns of only the missing benchmarks (2026-08-01)

Prefetch forensics sharpened the round-3 diagnosis: even a SINGLE fresh process
hung at humaneval (poll-waiting in `do_poll`), because killed jobs leave their
datasets `*.lock` sentinels behind on /scratch (NFS locks from dead clients
don't clear). Remedy: with no eval running, deleted all stale `*.lock` files
under `hf_cache/datasets/` (85 top-level + subdir ones) before re-running the
prefetch. Corollary: any TIMEOUT/kill during a dataset download can poison the
shared cache for every later run — offline-by-default is the durable fix.

Fixes, all synced to `$W/rc-sparse-speed` (uncommitted, on `irene-rebuttal-lora`):

1. **Prefetch + offline:** `scripts/aicr_prefetch_eval_datasets.py` run once on
   the login node caches every suite dataset (incl. gated gpqa + the `evaluate`
   squad metric module); the sbatch now defaults to
   `HF_HUB/TRANSFORMERS/HF_DATASETS/HF_EVALUATE_OFFLINE=1` (`EVAL_ONLINE=1`
   restores old behavior). Evidence this works: 9 concurrent jobs ran the
   pre-cached mmlu fine.
2. **coding_evaluator ported to lm-eval 0.4.12:** modified Task objects now pass
   straight into `simple_evaluate(tasks=[...])` (TaskManager.load accepts Task
   instances); instruct-prompt injection rewrites `task.config.doc_to_text`
   (template string) and the markdown extractor is prepended to each
   `task._filters` FilterEnsemble. Both hooks dry-run-verified on the cluster env.
3. `HF_TOKEN` value is now redacted in every evaluator's env dump (round-1/2/3
   logs printed it in plaintext — consider rotating the token).
4. sbatch gained `BENCHMARKS` (subset selection) and `OUTPUT_DIR` (write into
   the arm's round-3 dir so each arm keeps ONE results dir).

Submission: `scripts/aicr_submit_eval_round4.sh` — 7 arms run
`math gsm8k coding ifeval squad gpqa_diamond`; dense + lora-lr1e-4 run only
`coding ifeval squad gpqa_diamond`. LoRA arms eval the existing merged
snapshots directly (no ADAPTER re-merge). Results land next to round-3 mmlu in
`$S/rebuttal_analysis/eval_tulu3/<TAG>_<round3-jobid>/`. NB:
`all_benchmarks_summary.json` in each dir is overwritten by the round-4 subset —
build Table 1 from the per-benchmark `*_results.json` files, not the summary.

Prefetch epilogue (all on 2026-08-01):
- Second hang traced by py-spy to the **login node's IPv6 routes to
  huggingface.co blackholing** (SYN timeout on all 8 v6 addresses; `evaluate`'s
  metric-script HEAD has no timeout → ~17 min apparent hang). Prefetch script now
  filters `getaddrinfo` to IPv4. curl was never affected (instant v4 fallback),
  which is why connectivity "checked out".
- gpqa NOT cached: `Idavidrein/gpqa` is **gated** and this HF account lacks
  access (likely why 05-03 base row has 6 numbers, and Explorer's cache has no
  gpqa either). → Irene: request access at
  https://huggingface.co/datasets/Idavidrein/gpqa, then run a gpqa-only follow-up.
- squad-direct (`load_dataset("squad")`) fails on modern huggingface_hub (bare
  legacy name rejected, needs `rajpurkar/squad`) — irrelevant: `evaluate_squad`
  defaults to the lm-eval path and `squad_completion` (2984 docs) is cached.
- Cached + verified: mmlu, hendrycks_math + minerva_math, gsm8k, humaneval,
  mbpp, `code_eval` metric, ifeval, squad_completion. (lm-eval 0.4.12 dropped
  the task names `math`, `squad`, `squad_v2`, `gpqa_diamond`, `gpqa_diamond_n-shot`
  from the registry; each evaluator's candidate list still finds a valid name.)

| TAG | Round-4 Job ID | Benchmarks |
|---|---|---|
| base_llama31_8b_instruct | 248621 | math gsm8k coding ifeval squad |
| dense_dpo_tulu3_ckpt500 | 248622 | coding ifeval squad |
| sparse_warm_mag_step200 | 248623 | math gsm8k coding ifeval squad |
| sparse_oracle_dpo_tulu3 | 248624 | math gsm8k coding ifeval squad |
| sparse_oracle_dpo_lightr1 | 248625 | math gsm8k coding ifeval squad |
| sparse_oracle_grpo_math | 248626 | math gsm8k coding ifeval squad |
| sparse_random | 248627 | math gsm8k coding ifeval squad |
| lora_r64_lr1e-4_lightr1 | 248628 | coding ifeval squad |
| lora_r64_lr5e-6_lightr1 | 248629 | math gsm8k coding ifeval squad |

## Round 4 addendum — the humaneval-zero bug and the coding redo (2026-08-01 evening)

First coding results came back **humaneval ≈ 0.006, mbpp healthy** on every arm
that ran. Root causes, established by a `log_samples` diagnostic (job 248721,
8 docs, pass@1 = 1.0 once fixed):

1. humaneval's completion-style stop strings (`"\ndef"`, `"\n#"`, …) truncate a
   chat answer on its first line; cleared to `[]` under chat mode (EOT stops).
2. humaneval's `create_test` filter builds `doc["prompt"] + resp`, so a chat
   model's full-function answer must be converted to a body-only continuation
   (strip the restated `def` signature; indent pre-def imports into the body).
3. The killer: `evaluate_coding` **stripped `apply_chat_template` from the
   `simple_evaluate` kwargs** (an 0.4.11-era leftover — it used to wrap prompts
   itself), so the instruct model saw raw completion prompts. Now passed through
   natively.

Validation: smoke 248742 (base, --limit 16): **humaneval 0.9375, mbpp 0.5625**.
NB: mbpp now also runs WITH chat template (previously effectively without), so
mbpp numbers move vs the first broken pass — the coding redo below re-runs both
tasks on every arm with the same fixed code, which is the comparable set.

Coding-only redo jobs, chained `--dependency=afterany:` on each arm's round-4
job so they can't race the same OUTPUT_DIR (they overwrite the arm's broken
`coding_results.json`):

| TAG | redo job | after |
|---|---|---|
| base | 248812 | 248621 |
| dense | 248813 | 248622 |
| warm_mag | 248814 | 248623 |
| oracle_dpo_tulu3 | 248815 | 248624 |
| oracle_dpo_lightr1 | 248816 | 248625 |
| oracle_grpo_math | 248817 | 248626 |
| random | 248818 | 248627 |
| lora lr1e-4 | 248819 | 248628 |
| lora lr5e-6 | 248820 | 248629 |

Fourth landmine (~18:30): **vLLM 0.26 engine-core children outlive the
evaluation**, so each benchmark subprocess finishes its work (transport JSON
fully written) then hangs forever in multiprocessing's atexit join — round-4
jobs sat wedged ~2h; the round-3 "hangs" were partly this too. Fixes: the
subprocess wrapper in `run_all_benchmarks.py` now `os._exit(0)`s after writing
the transport file, and a login-side "nudger" (srun --overlap, kills EngineCore
children of wrappers whose transport file is >90 s old) unstuck the live jobs.
Proof: base then produced math 0.403 / gsm8k 0.820 / **coding he=0.683
mbpp=0.584** (healthy humaneval at last) before hitting the 2:30 wall.

Round-4 mains all TIMEOUT at 2:30 with partial coverage (base has 4/7 suites;
the 5 sparse arms + lora lr5e-6 only mmlu). Mop-up jobs (patched wrapper, no
exit hangs), chained afterany on each arm's gpqa job: 249006 base / 249007
dense (ifeval+squad each), 249008-249012 the five sparse arms, 249014 lora
lr5e-6 (math+gsm8k+ifeval+squad each), 249013 lora lr1e-4 (ifeval+squad).
Chain per arm: round-4 main → coding redo → gpqa → mop-up.

Fifth landmine (~20:45), self-inflicted by the os._exit fix: the worker now
exits without shutting down its vLLM engine-core child, which is reparented to
init while still holding the wrapper's captured stdout/stderr **pipes** — so the
parent's `subprocess.run(capture_output=True)` blocks on pipe EOF forever even
though the worker is done and the transport file is written. All six running
mop-ups were wedged exactly this way (EngineCore with ppid 1). Fixes:
(1) wrapper pkill -9 -P's its own children before os._exit; (2) the parent now
redirects subprocess output to FILES instead of pipes (returns on child exit,
immune to lingering grandchildren); (3) nudger v2 also kills orphaned/
zombie-parent EngineCores — one sweep instantly unstuck all six jobs.
Casualties before the fix: 249006/249007/249013 TIMEOUT with ifeval+squad
unwritten → resubmitted as 249175 (base) / 249176 (dense) / 249177 (lora
lr1e-4), fixed code, no deps.

Sixth landmine (~20:40, the big one): **AICR's /scratch 30-day purge deleted
the model weights mid-eval** — all five sparse `final_model` dirs, dense
checkpoint-500, and the Qwen3-32B random dir vanished between benchmarks
(math loaded weights at ~19:55; gsm8k found the dir gone). Cause: `rsync -a`
preserved the 2026-04/05 training mtimes, so day-old copies were already
"90 days stale" to the mtime-based purge. Survivors prove it: merged-LoRA
snapshots, results, HF cache (all created ON AICR, fresh mtimes) untouched.
Recovery: 6 retry-loop rsync streams relaunched from Explorer (5 sparse arms +
dense ckpt-500) **with `find … -exec touch {} +` appended to each stream**, plus
a preemptive touch sweep over surviving transfer_v1 / rebuttal_analysis /
hf_cache. Casualties to resubmit after transfer: gsm8k+ifeval+squad × 5 sparse
arms, ifeval+squad × dense (249176 FAILED on the missing dir).

Meanwhile the fixed pipeline showed its true speed: 249175 (base ifeval+squad)
COMPLETED in **3.5 minutes** — ifeval strict 0.7394, squad contains 0.6883.
(NB squad_completion's metric is `contains`, the evaluator's "F1 = 0.0000"
printout is a mislabel; the figure axis should read "SQuAD (contains)".)

gpqa unblocked ~20:00: Irene accepted the Idavidrein/gpqa terms (account
`xxiellan`, verified via the hub auth-check endpoint — NB the `/tree/main`
endpoint returns 200 even without access and is useless as a grant probe);
gpqa_diamond_zeroshot (198 docs) prefetched. gpqa-only jobs, chained afterany
on each arm's coding redo: 248849–248857 (same arm order as the redo table).

Also: queue relief at ~17:00 — all round-4 jobs' TimeLimit lowered 8h→2:30
(backfill), 248628/248629 moved to b200-devel; everything was RUNNING within
~20 min. ⚠ Any coding_results.json timestamped before ~19:00 EDT 2026-08-01 is
from broken code — trust only the redo jobs' outputs (and per-benchmark files,
not all_benchmarks_summary.json, which the redo overwrites with a coding-only view).

## Round 3 (superseded) — resubmitted 2026-08-01 ~02:50 UTC

| TAG | Round-3 Job ID |
|---|---|
| base_llama31_8b_instruct | 246641 |
| dense_dpo_tulu3_ckpt500 | 246642 |
| sparse_warm_mag_step200 | 246643 |
| sparse_oracle_dpo_tulu3 | 246644 |
| sparse_oracle_dpo_lightr1 | 246645 |
| sparse_oracle_grpo_math | 246646 |
| sparse_random | 246647 |
| lora_r64_lr1e-4_lightr1 | 246648 |
| lora_r64_lr5e-6_lightr1 | 246649 |

First job to survive engine init >15 min validates the whole stack; expect ~5 min
of one-time flashinfer JIT (cached in $HOME thereafter). Fallback if this layer
also fails: resubmit everything with `FORCE_HF_BACKEND=1`.

## Sanity checks for readout

1. base row ≈ 05-03 base row (0.6874 / 0.7375 / 0.8203 / 0.4322 / 0.6887 / 0.6020) —
   if not, backend change moved the suite and ALL rows must be read as a new baseline,
   not mixed with 05-03 numbers.
2. ⚠ Open provenance question for Scott: Table 1 labels the warm-mag row
   "Light-R1" but the 05-03 log says the step-200 magnitude mask came from the
   dense **Tulu3** deltalog run. Resolve before the rebuttal quotes this row.
