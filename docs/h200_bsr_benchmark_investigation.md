# H200 BSR throughput benchmark — investigation notes

This memo documents why **logged** sparse phases in the **multi-phase H200 BSR–DPO benchmark** can show **much lower** `cumulative_steps_per_s` than dense (sometimes large factors, e.g. 2×–100× in the CSV) and how to interpret that **without** confusing it with “production sparse training is broken.” The benchmark driver and sparse GRPO/DPO paths share the same injection stack; differences are mostly **what is being measured** (cold per-phase throughput vs steady WandB training metrics) and **mask shape** (protocol random masks vs curated masks), not a fundamental software defect.

## 1. Metric definition: “cold” cumulative throughput

`BenchmarkThroughputCallback` (see `src/utils/logging_utils.py`) sets a timer at **`on_train_begin`**. Each logged row reports:

- `wall_time_s`: wall clock since that callback.
- `cumulative_steps_per_s`: `global_step / wall_time_s`.

So the denominator includes **model load from disk**, **mask I/O and index precomputation**, **`replace_linear_modules`**, **Triton compile**, **optimizer state init**, and **early dataloader / first-step** work—not only steady optimizer steps.

With **few steps per phase** (historically default 8), a fixed multi-second setup cost dominates the average and **suppresses** `cumulative_steps_per_s` for heavy sparse phases more than for a lighter dense phase.

**Mitigations (implemented or recommended):**

- Prefer a **larger** `H200_BSR_STEPS_PER_PHASE` (default **25** in `scripts/h200_sparse_dpo_bsr_benchmark.sh`; increase for heavier amortization) so setup is amortized.
- Read **`inst_steps_per_s`** (delta since the **previous** log row) when `RL_CASINO_LOGGING_STEPS=1`; tail mean/std of this column is closer to **steady-interval** throughput.
- Optionally enable `RL_CASINO_BSR_DETAILED_TIMING=1` only on short probes; it changes wall time (synchronize per micro-batch).

## 2. Mask scope: MLP-only vs full model (and why this is not a dig at production GRPO)

The Slurm wrapper **`h200_sparse_dpo_bsr_benchmark.sh` does not pass `--mlp_only`**, so random masks target **all** scored 2D weights the driver assigns; `replace_linear_modules` swaps **every** `nn.Linear` that has a mask—including **attention and embeddings**—for `SparseLinearLayer` where applicable. That matches the **same default** as sparse GRPO in this repo: [`scripts/grpo_openr1_llama31_slurm.sh`](../scripts/grpo_openr1_llama31_slurm.sh) drives the sparse GRPO entrypoint with **`--mlp_only` unset** (`store_true`, default **False**). So “no `mlp_only`” is normal, supported production behavior—not a misconfiguration.

**Why the benchmark CSV can still look scary while WandB GRPO looks fine**

1. **Metric:** `cumulative_steps_per_s` divides `global_step` by wall time since **`on_train_begin` for that phase**. Each benchmark phase **reloads the model** and pays setup again (I/O, injection, Triton compile, opt state). Dense and sparse phases **do not pay identical setup**, and short phases amplify that—so the **ratio of two CSV numbers** is not the same thing as “sparse GRPO is N× slower than dense GRPO” on a long steady run. Use **`inst_steps_per_s`** (and WandB’s training step metrics on real jobs) for steady-interval behavior.

2. **Workload:** This benchmark sweeps **random** masks at several sparsities (including very high sparsity) and alternates mask layouts (element vs block) **by design**. Production masks (GraSP, SNIP, CAV, checkpoint-diff, etc.) are **different distributions**—often more structured, sometimes implicitly MLP-heavy (e.g. CAV scoring), sometimes block-native—so **relative** sparse-vs-dense wall time is not required to match the benchmark grid.

3. **Entrypoint:** GRPO adds generation, rewards, and other costs; WandB curves reflect the **whole** job. The H200 DPO benchmark isolates **DPO-style BSR + SparseAdamW** steps for apples-to-apples **within** that script—not a universal speed ratio vs `grpo_openr1_llama31_slurm.sh`.

**MLP-only** remains an **optional** narrow ablation: run `h200_sparse_dpo_bsr_benchmark.py` manually with **`--mlp_only`** if you want masks restricted to MLP linears for a controlled comparison. It is not implied that full-model sparse is invalid or unusable.

## 3. Operational: Slurm time limits and partial CSVs

If the job hits **`TIME LIMIT`** mid-grid (e.g. during CPU mask generation for a new sparsity), later phases may be missing or truncated. Any report should flag **incomplete phases** (max logged `step` < `benchmark_phase_target_steps`). The Markdown/LaTeX generator `scripts/benchmark_training_log_to_report_md.py` does this when the CSV includes `benchmark_phase_target_steps` (written by `h200_sparse_dpo_bsr_benchmark.py`).

## 4. CSV columns for faithful reporting

| Column | Role |
|--------|------|
| `benchmark_phase_target_steps` | Expected optimizer steps per phase (for completeness checks). |
| `benchmark_mlp_only` | `1` if MLP-only mask scope. |
| `benchmark_mask_key_count` | Number of mask tensors in the phase (sanity check). |
| `wall_delta_s`, `inst_steps_per_s`, `inst_samples_per_s` | Interval metrics between consecutive logs (steady-state hint). |
| `cumulative_steps_per_s` | Legacy headline metric (includes cold start). |

## 5. Recommended workflow before paper numbers

1. Run the Slurm driver with defaults (full-model masks, dense + 99.75% grid, 25 steps/phase unless you override).
2. Confirm job **completed** (no `TIME LIMIT` in `.out`).
3. Run `python3 scripts/benchmark_training_log_to_report_md.py --csv .../benchmark_training_log.csv --emit-tex paper_tables/`.
4. Prefer **inst** tail statistics and **CV** in the generated tables when `cumulative` CV is high.
5. Include a **dense** phase in the same CSV or pass **`--baseline-csv`** for explicit “vs dense” ratios.
