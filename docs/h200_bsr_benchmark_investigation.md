# H200 BSR throughput benchmark — investigation notes

This memo documents why sparse phases in multi-phase BSR–DPO benchmarks can appear **much slower** than dense (e.g. 2×–100× lower `cumulative_steps_per_s`) and how we interpret metrics after the reporting retool (2026).

## 1. Metric definition: “cold” cumulative throughput

`BenchmarkThroughputCallback` (see `src/utils/logging_utils.py`) sets a timer at **`on_train_begin`**. Each logged row reports:

- `wall_time_s`: wall clock since that callback.
- `cumulative_steps_per_s`: `global_step / wall_time_s`.

So the denominator includes **model load from disk**, **mask I/O and index precomputation**, **`replace_linear_modules`**, **Triton compile**, **optimizer state init**, and **early dataloader / first-step** work—not only steady optimizer steps.

With **few steps per phase** (historically default 8), a fixed multi-second setup cost dominates the average and **suppresses** `cumulative_steps_per_s` for heavy sparse phases more than for a lighter dense phase.

**Mitigations (implemented or recommended):**

- Prefer a **larger** `H200_BSR_STEPS_PER_PHASE` (default raised to **50** in `scripts/h200_sparse_dpo_bsr_benchmark.sh`) so setup is amortized.
- Read **`inst_steps_per_s`** (delta since the **previous** log row) when `RL_CASINO_LOGGING_STEPS=1`; tail mean/std of this column is closer to **steady-interval** throughput.
- Optionally enable `RL_CASINO_BSR_DETAILED_TIMING=1` only on short probes; it changes wall time (synchronize per micro-batch).

## 2. Mask scope: MLP-only vs full model

By default the driver used **`--mlp_only` unset**, so random masks could target **all** scored 2D weights; `replace_linear_modules` swaps **every** `nn.Linear` that has a mask—including **attention and embeddings**—for `SparseLinearLayer`. That is a different experiment than “sparse MLP only” and can be **orders of magnitude** slower than dense for element-wise masks at extreme sparsity.

**Mitigation (implemented):** `scripts/h200_sparse_dpo_bsr_benchmark.sh` now defaults **`H200_BSR_MLP_ONLY=1`** and passes **`--mlp_only`**. Set `H200_BSR_MLP_ONLY=0` only when you intentionally benchmark full-model masks.

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

1. Run the Slurm driver with defaults (MLP-only, ≥50 steps/phase unless probing).
2. Confirm job **completed** (no `TIME LIMIT` in `.out`).
3. Run `python3 scripts/benchmark_training_log_to_report_md.py --csv .../benchmark_training_log.csv --emit-tex paper_tables/`.
4. Prefer **inst** tail statistics and **CV** in the generated tables when `cumulative` CV is high.
5. Include a **dense** phase in the same CSV or pass **`--baseline-csv`** for explicit “vs dense” ratios.
