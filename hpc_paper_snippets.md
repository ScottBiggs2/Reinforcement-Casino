# Paper snippets — H200 BSR DPO benchmark tables

Numeric tables in the **export block** below come from [`scripts/export_h200_bsr_paper_tables.py`](scripts/export_h200_bsr_paper_tables.py) (aggregating `benchmark_training_log.csv`). Use **`--compact`** into this file to avoid duplicated LaTeX blocks. Elsewhere, **NEG** / negative placeholders still mean “paste from env snapshot.” Driver: [`scripts/h200_sparse_dpo_bsr_benchmark.sh`](scripts/h200_sparse_dpo_bsr_benchmark.sh) (default: **dense baseline included**, **`BENCHMARK_SPARSITIES=99.75,97.5,95,90`**; Slurm script does **not** pass `--mlp_only`).

### CSV columns and aggregation (read before filling tables)

**Always present** (including default fast sweep with `RL_CASINO_BSR_DETAILED_TIMING=0`): `phase`, `step`, `wall_time_s`, `cumulative_steps_per_s`, `cumulative_samples_per_s`, trainer metrics (`loss`, etc.), `trainer_per_device_train_batch_size`, `trainer_grad_accum_steps`, and **theory** fields from the mask sidecar—especially **`theory_bsr_backward_flops_proxy`** (masked-linear backward FLOPs **per optimizer step**; includes gradient accumulation in the token proxy). The exporter fills the **“Theory BWD FLOP/step”** column from this—no detailed CUDA timing required.

**Timed TFLOP/s columns** (`TFLOP/s (bwd)`, `TFLOP/s (step)` in Markdown; `\textsubscript{bwd}` / `\textsubscript{step}` in LaTeX) come from **`eff_bsr_backward_tflops`** and **`eff_bsr_backward_tflops_over_e2e_step`**, which the logger computes **only when** CUDA segment timings exist (`t_backward_ms`, `t_step_total_ms`, etc.). Those appear **only** with `RL_CASINO_BSR_DETAILED_TIMING=1`. That mode wraps each micro-batch with `torch.cuda.synchronize()` and can **severely slow** training when gradient accumulation is large—use it only for short diagnostic runs, not full multi-phase grids, unless you accept the slowdown.

**Caveats when detailed timing is on:**

- `t_forward_ms` / `t_backward_ms` reflect the **last gradient-accumulation micro-batch only**, not the full optimizer step; `t_other_ms` can look huge if interpreted as “unexplained” time.
- **Bwd** effective TFLOP/s divides the theory proxy by backward time scaled by grad accumulation (see [`logging_utils.py`](../src/utils/logging_utils.py)). **Step** TFLOP/s divides the same proxy by the measured **full-step** wall slice. **Older CSVs** before the grad-accum fix may be wrong on `eff_*` — run [`scripts/recalc_benchmark_training_log_eff.py`](scripts/recalc_benchmark_training_log_eff.py) to rewrite without retraining.

**Quick per-phase summary on-cluster or locally:**

```bash
python scripts/analyze_benchmark_training_log.py /path/to/benchmark_training_log.csv --grad-accum 64
```

Table numbers below are **generated** from `benchmark_training_log.csv` (mean over the last `--tail-rows` logged rows per `phase`). Re-export whenever the CSV changes; do not hand-edit the export block.

**Fill / refresh (repo root):**

```bash
python scripts/export_h200_bsr_paper_tables.py \
  --csv /path/to/benchmark_training_log.csv \
  --inject hpc_paper_snippets.md \
  --tail-rows 8 \
  --compact
```

Example layout CSV (drafting only — **replace with your measured CSV** before submission): `scripts/fixtures/h200_bsr_table_export_example.csv`.

**Grid reference:** `h200_sparse_dpo_bsr_benchmark.sh` defaults to **`H200_BSR_SKIP_DENSE=0`** (one dense baseline) and **`BENCHMARK_SPARSITIES=99.75,97.5,95,90`**. Each sparsity level adds **4** phases (element vs block mask \(\times\) Adam block\_1d vs block\_2d; grad\_input=dense only) \(\Rightarrow\) **17** phases by default (1 dense + 16 sparse). Export `--compact` \(\Rightarrow\) Markdown + one throughput LaTeX block; omit `--compact` for timing + appendix LaTeX tables. Partial runs contribute fewer phases.

<!-- H200_BSR_PAPER_EXPORT_START -->

## Auto-filled from benchmark CSV (edit upstream CSV + re-export; do not hand-tune numbers here)

_Compact mode_ (`--compact`): Markdown summary and one throughput LaTeX block only. Omit `--compact` for timing-highlight + appendix LaTeX.

<!-- Generated from `benchmark_training_log.csv` — re-run export after updating the CSV. -->

### Aggregated throughput (mean of last logged rows per phase)

| Phase | Sparse % | Optimizer | Last step | microBS / accum | Steps/s | Samples/s | Wall (s) | Theory BWD FLOP/step | TFLOP/s (bwd) | TFLOP/s (step) |
|-------|----------|-----------|-----------|-----------------|---------|-----------|----------|----------------------|---------------|----------------|
| `dense` |  | adamw | 8 | --- | 0.024736 | 3.1663 | 217.00 | --- | --- | --- |
| `s99p75_elem_gidense_block_1d` | 99.75 | sparse_adamw | 8 | --- | 0.001233 | 0.1579 | 4356.14 | 5.2625e+12 | --- | --- |
| `s99p75_elem_gidense_block_2d` | 99.75 | sparse_adamw | 8 | --- | 0.001234 | 0.1579 | 4354.97 | 5.2625e+12 | --- | --- |
| `s99p75_blk_gidense_block_1d` | 99.75 | sparse_adamw | 8 | --- | 0.025233 | 3.2299 | 212.87 | 5.2625e+12 | --- | --- |
| `s99p75_blk_gidense_block_2d` | 99.75 | sparse_adamw | 8 | --- | 0.025275 | 3.2352 | 212.57 | 5.2625e+12 | --- | --- |
| `s97p5_elem_gidense_block_1d` | 97.5 | sparse_adamw | 8 | --- | 0.000598 | 0.0766 | 8980.26 | 5.2625e+13 | --- | --- |

```latex
\begin{table}[t]
  \centering
  \small
  \caption{H200 BSR --- throughput (tail-mean). \emph{Theory} $\tilde F_{\mathrm{BSR\,bwd}}$: masked-linear backward FLOPs per optimizer step (mask sidecar; sparse phases only). \emph{Timed} rates use CUDA segment timings when enabled: \textsubscript{bwd} scales the last micro-batch backward by grad accumulation; \textsubscript{step} divides the same theory proxy by the measured full-step wall slice. Otherwise \texttt{---}.}
  \label{tab:bsr-throughput-autogen}
  \begin{tabular}{@{}lcccccc@{}}
    \toprule
    Phase & Steps/s & Samples/s & Last step & $\tilde F_{\mathrm{BSR\,bwd}}$ / step & TFLOP/s\textsubscript{bwd} & TFLOP/s\textsubscript{step} \\
    \midrule
    \texttt{dense} & 0.024736 & 3.1663 & 8 & --- & --- & --- \\
    \texttt{s99p75\_elem\_gidense\_block\_1d} & 0.001233 & 0.1579 & 8 & 5.2625e+12 & --- & --- \\
    \texttt{s99p75\_elem\_gidense\_block\_2d} & 0.001234 & 0.1579 & 8 & 5.2625e+12 & --- & --- \\
    \texttt{s99p75\_blk\_gidense\_block\_1d} & 0.025233 & 3.2299 & 8 & 5.2625e+12 & --- & --- \\
    \texttt{s99p75\_blk\_gidense\_block\_2d} & 0.025275 & 3.2352 & 8 & 5.2625e+12 & --- & --- \\
    \texttt{s97p5\_elem\_gidense\_block\_1d} & 0.000598 & 0.0766 & 8 & 5.2625e+13 & --- & --- \\
    \bottomrule
  \end{tabular}
\end{table}
```

<!-- H200_BSR_PAPER_EXPORT_END -->

---

## Appendix — H200 BSR--DPO throughput benchmark (paper table)

Quantities only (no cluster paths or launcher names). Default configuration: Llama 3.1 8B Instruct, DPO on the **Tülu 3** preference mixture, random global pruning at three targets, elementwise **and** block-coarse layouts, dense grad-input contractions with sparse (BSR) weight backward, SparseAdamW with **two** blocked-parameter reductions crossed with those layouts, bf16 training, eight optimizer steps per timed phase for wall-clock summaries. Omit the dense-baseline narrative when publishing the default **sparse-only** grid.

```latex
% Requires: \usepackage{booktabs}
\begin{table}[t]
  \centering
  \small
  \caption{DPO throughput benchmark — H200 sparse BSR / SparseAdamW grid (Llama 3.1 8B Instruct, randomized global masks).}
  \label{tab:h200-bsr-dpo-benchmark-hparams}
  \begin{tabular}{@{}ll@{}}
    \toprule
    Hyperparameter & Value \\
    \midrule
    Hardware & 1 \(\times\) NVIDIA H200 \\
    Base model & \texttt{meta-llama/Llama-3.1-8B-Instruct} \\
    Preference dataset & Tülu 3 \\
    Epochs per phase & 1 \\
    Optimizer updates per phase & 8 \\
    Learning rate & $5 \times 10^{-7}$ \\
    LR schedule & Linear; warmup fraction 0.1 \\
    Weight decay & $0.0$ \\
    DPO $\beta$ & $0.1$ \\
    Max gradient norm & $1.0$ \\
    Per-device batch size & 2 \\
    Gradient accumulation steps & 64 \\
    Effective batch (1 GPU, per optimizer step) & 128 \\
    Max prompt length (tokens) & 1024 \\
    Max sequence length (tokens) & 1024 \\
    Precision & bf16 \\
    Dense baseline (optional) & Dense linear layers + AdamW 8-bit — omitted under default throughput grid \\
    Sparse linear maps & Sparse backward on weights (BSR-style); contraction w.r.t.\ activations remains dense \\
    Sparse optimizer & SparseAdamW \\
    Sparse factorial design & Twelve timed configurations: \(\{90,95,97.5\}\%\) sparsity \(\times\) unstructured vs.\ block masks \(\times\) two SparseAdamW block layouts \\
    Target parameter sparsity (random masks) & $90\%$, $95\%$, $97.5\%$ weights zeroed \\
    Mask construction & Global randomized scores with per-parameter retain floor ratio $2.5 \times 10^{-3}$ \\
    Block-structured pruning tile & BSR tiling width $16$; SparseAdamW tile width $128$ \\
    Adam $\beta_1$, $\beta_2$, $\epsilon$ & $0.9$, $0.999$, $10^{-8}$ \\
    Gradient checkpointing & Enabled \\
    Checkpoint / artifact saves & Disabled (timing-oriented runs) \\
    \bottomrule
  \end{tabular}
\end{table}
```

---

## Appendix — DPO/GRPO hyperparameter transparency tables (env vars)

The project primarily wires hyperparameters through **environment variables** in Slurm wrappers (then forwarded as CLI flags into Python entrypoints). For reviewer transparency, I recommend capturing **two artifacts** per run:

- **(A) “Declared defaults”**: the defaults in the repo scripts (tables below).
- **(B) “As-run snapshot”**: a sorted env dump from the Slurm `.out` *right before* launching Python.

### How to capture the “as-run snapshot” on HPC (copy/paste)

Put this near the top of your Slurm script (or run in an interactive allocation *before* `sbatch`), then paste the output into the Value column of the tables.

```bash
echo "=== ENV SNAPSHOT (DPO/GRPO/sparse) ==="
printenv | egrep '^(MODEL|HF_|SCRATCH_|TRAIN_|EVAL_|(DPO|GRPO|RL_CASINO|BSR|SPARSE|NUM_STEPS)_)[A-Z0-9_]*=' | sort
echo "=== /ENV SNAPSHOT ==="
```

---

### Mask Construction Stuff

This section matches the implementation in \texttt{src/warm\_start/checkpoint\_diff\_mask\_finder.py} (checkpoint-difference oracle scores) and \texttt{src/utils/mask\_utils.py} (mask selection). \textbf{Scores} are elementwise absolute weight movement,
\(
S_p = \bigl| W^{\mathrm{final}}_p - W^{\mathrm{initial}}_p \bigr|
\)
for every parameter $p$ present in both state dicts (optional restriction to MLP weight name patterns). \textbf{Masks} are boolean \emph{inclusion} tensors (same shape as $W_p$): train/update only where the mask is true.

The repository’s \textbf{default} selection mode is a \emph{hybrid}: \emph{global} competition for most of the keep budget, plus a small \emph{per-tensor keep floor} so no layer is fully starved at extreme sparsity. Concretely, \texttt{create\_mask\_from\_scores\_gpu\_efficient} uses \texttt{min\_layer\_keep\_ratio}\,$=$\,\texttt{DEFAULT\_MIN\_LAYER\_KEEP\_RATIO}\,$=$\,$0.0025$ unless overridden; the CLI flag \texttt{--local\_pool} switches to strict per-tensor ranking instead. For very large models, the same hybrid logic runs on CPU via a chunked threshold selector (histogram refinement plus a boundary $\mathrm{topk}$); smaller models use a single concatenated score vector and two-pass $\mathrm{topk}$ on the remainder after masking floor positions.

\textbf{Block-structured masks} (\texttt{--mask-granularity block} in the checkpoint script): two-dimensional weight scores are padded to a multiple of $B\times B$, each tile is reduced to one block score (mean or max over the $B^2$ elements), the same hybrid (or local) selector runs on the \emph{block grid}, and selected blocks are expanded to full weight masks by tiling; non-2D parameters get all-false masks so training focuses on masked matrix multiply paths. In \texttt{checkpoint\_diff\_mask\_finder.py}, block masks call the selector with tie-break noise disabled (\texttt{add\_tie\_break\_noise=False}); elementwise checkpoint masks use the API default (noise on).

\begin{algorithm}[h]
\caption{Checkpoint-difference scores (oracle)}
\label{alg:mask-scores-ckpt-diff}
\begin{algorithmic}[1]
\Require Initial weights $\{W^{\mathrm{init}}_p\}$, final weights $\{W^{\mathrm{fin}}_p\}$; optional MLP-only filter
\Ensure Score tensors $\{S_p\}$ aligned with trainable parameters
\For{each name $p$ in $\mathrm{keys}(W^{\mathrm{fin}}) \cap \mathrm{keys}(W^{\mathrm{init}})$}
    \If{MLP-only mode and $p$ does not match MLP name patterns}
        \State \textbf{skip} $p$
    \EndIf
    \State $S_p \gets \bigl| W^{\mathrm{fin}}_p - W^{\mathrm{init}}_p \bigr|$ (elementwise, promoted to float32 for subtraction)
\EndFor
\end{algorithmic}
\end{algorithm}

\begin{algorithm}[h]
\caption{Default hybrid mask: global budget with per-tensor keep floor (\texttt{local\_pool}$=$ false, \texttt{min\_layer\_keep\_ratio}$=r$)}
\label{alg:mask-hybrid-default}
\begin{algorithmic}[1]
\Require Score tensors $\{S_p\}$; target exclusion fraction $\rho \in [0,1]$; floor ratio $r \in [0,1]$; optional tie-break noise on scores (default on in API)
\Ensure Binary inclusion masks $\{m_p\}$ with total kept count $k_{\mathrm{keep}} = \lfloor (1-\rho)\,N \rfloor$, $N=\sum_p \mathrm{numel}(S_p)$
\State Sanitize $S_p$ (replace NaN/Inf; dtype float32)
\If{tie-break enabled}
    \State add i.i.d.\ Gaussian noise to each $S_p$ with scale $\propto$ global max $|S|$ (fixed RNG seed in code)
\EndIf
\State Initialize all $m_p \gets 0$
\State \textbf{Floor pass:} for each $p$, set $f_p \gets \lfloor r \cdot \mathrm{numel}(S_p) \rfloor$; if $\sum_p f_p > k_{\mathrm{keep}}$, multiply all $f_p$ by $k_{\mathrm{keep}}/\sum_q f_q$ and round down within each layer’s size
\For{each $p$ with $f_p>0$}
    \State $I_p^{\mathrm{floor}} \gets$ indices of the top-$f_p$ elements of $\mathrm{vec}(S_p)$
    \State set $m_p[i]=1$ for $i \in I_p^{\mathrm{floor}}$
\EndFor
\State $k_{\mathrm{rem}} \gets k_{\mathrm{keep}} - \sum_p |I_p^{\mathrm{floor}}|$
\State \textbf{Global pass:} among positions not fixed by the floor, select the $k_{\mathrm{rem}}$ largest scores (implementation: flat $\mathrm{topk}$ on masked concatenated scores if $N$ is below an internal threshold; otherwise chunked threshold search on CPU with histogram refinement, then $\mathrm{topk}$ within the final score bin)
\end{algorithmic}
\end{algorithm}

\begin{algorithm}[h]
\caption{Strict local mode (\texttt{--local\_pool}): uniform sparsity per tensor}
\label{alg:mask-local}
\begin{algorithmic}[1]
\Require Score tensors $\{S_p\}$; target exclusion fraction $\rho \in [0,1]$
\Ensure Binary masks $m_p \in \{0,1\}^{\mathrm{shape}(p)}$
\State $\alpha \gets 1 - \rho$ \Comment{fraction to keep}
\For{each tensor $p$}
    \State $k \gets \max(1,\lfloor \alpha \cdot \mathrm{numel}(S_p) \rfloor)$
    \State Optionally add tie-break noise to flattened $S_p$
    \State Let $I$ be indices of top-$k$ values of $\mathrm{vec}(S_p)$
    \State $m_p \gets 0$; set $m_p[i]=1$ for $i \in I$
\EndFor
\end{algorithmic}
\end{algorithm}

\begin{algorithm}[h]
\caption{Pure global selection ($r=0$): single ranking across all scored elements}
\label{alg:mask-global-pure}
\begin{algorithmic}[1]
\Require Valid score tensors $\{S_p\}$; target exclusion $\rho$; optional tie-break noise
\Ensure Binary masks $\{m_p\}$ with $\approx \rho$ fraction of zeros globally
\State Algorithm~\ref{alg:mask-hybrid-default} with $r=0$ (floor pass empty): keep top-$k_{\mathrm{keep}}$ scores across $\bigcup_p \mathrm{vec}(S_p)$
\end{algorithmic}
\end{algorithm}

\begin{algorithm}[h]
\caption{Block-pooled masks ($B\times B$ tiles on 2D weights only)}
\label{alg:mask-block}
\begin{algorithmic}[1]
\Require Element scores $\{S_p\}$ for 2D tensors; block size $B$; reduction $\in \{\mathrm{mean},\max\}$; same $\rho$, $r$, and \texttt{local\_pool} as above
\Ensure Expanded inclusion masks per 2D $p$; for non-2D parameters, all-false masks
\For{each 2D $S_p$ of shape $M\times N$}
    \State pad $S_p$ to $(M',N')$ multiples of $B$; reduce each $B\times B$ tile to one block score $\tilde{S}_{p,i,j}$ (mean or max over the tile)
\EndFor
\State Run Algorithm~\ref{alg:mask-hybrid-default} or \ref{alg:mask-local} on $\{\tilde{S}\}$ with the same $\rho$ (sparsity applies to \emph{blocks}; realized weight sparsity may differ slightly)
\State Expand each selected block to $B\times B$ ones on the padded grid, crop to $(M,N)$
\end{algorithmic}
\end{algorithm}


---

### DPO (dense + sparse) — pipeline-style env knobs

Defaults are primarily set/propagated by `scripts/pipeline_common.sh` (dense stage) and used again by sparse stages; for long-run launches, prefer the pipeline scripts in `scripts/`.

```latex
% Requires: \usepackage{booktabs}
\begin{table}[t]
  \centering
  \small
  \caption{DPO dense/sparse training hyperparameters and runtime knobs (environment variables). Negative numeric entries are placeholders if you haven’t copied the run-time snapshot yet.}
  \label{tab:appendix-dpo-env}
  \begin{tabular}{@{}lll@{}}
    \toprule
    Env var & Default (repo) & Value used (paste from snapshot) \\
    \midrule
    \texttt{MODEL} & \texttt{meta-llama/Llama-3.1-8B-Instruct} & \texttt{NEG} \\
    \texttt{DPO\_DATASET\_KEY} & \texttt{tulu3} & \texttt{NEG} \\
    \texttt{NUM\_STEPS\_DPO} & 250 (\texttt{pipeline\_common.sh}) / 5000 (\texttt{orchestrate\_*}) & -1 \\
    \texttt{DPO\_LEARNING\_RATE} & \texttt{5e-7} & \texttt{NEG} \\
    \texttt{DPO\_WARMUP\_RATIO} & \texttt{0.1} & \texttt{NEG} \\
    \texttt{DPO\_WEIGHT\_DECAY} & \textit{(often 0.0)} & \texttt{NEG} \\
    \texttt{DPO\_BETA} & \texttt{0.1} & \texttt{NEG} \\
    \texttt{DPO\_MAX\_LENGTH} & \texttt{1024} & -1 \\
    \texttt{DPO\_MAX\_PROMPT\_LENGTH} & \texttt{1024} & -1 \\
    \texttt{DPO\_PER\_DEVICE\_TRAIN\_BATCH\_SIZE} & 2 & -1 \\
    \texttt{DPO\_GRADIENT\_ACCUMULATION\_STEPS} & 64 & -1 \\
    \texttt{DPO\_GRADIENT\_CHECKPOINTING} & 1 (typical) & -1 \\
    \texttt{DPO\_OPTIM} & \texttt{adamw\_8bit} (dense baseline default) & \texttt{NEG} \\
    \texttt{DPO\_SAVE\_STEPS} & 50 & -1 \\
    \texttt{DPO\_SAVE\_TOTAL\_LIMIT} & 3 & -1 \\
    \texttt{HF\_DATASETS\_CACHE} & \textit{scratch path} & \texttt{NEG} \\
    \texttt{TRAIN\_ENV} & \textit{scratch path} & \texttt{NEG} \\
    \bottomrule
  \end{tabular}
\end{table}
```

---

### GRPO (Open-R1) — locked hyperparameters (see YAML; no duplicate table here)

Numeric defaults match **`docs/hyperparams/open_r1_llama31.yaml`** and **`scripts/grpo_training_env_defaults.sh`** (Llama 3.1 8B Instruct, Open-R1 Math-220k, 500-step reference, cosine + warmup, GRPO knobs, bf16). Dense training uses **`src/full_training/GRPO_train.py`**; sparse uses **`src/full_training/sparse_grpo_bsr.py`** with the same LR, $\beta$, batch, generation counts, and sequence caps. **Ablations:** set `GRPO_HPARAM_OVERRIDE=1` where the launcher permits overrides. Pull a standalone LaTeX “hyperparameter” table from that YAML when the paper needs a numeric caption distinct from **`tab:appendix-grpo-env`** below.

### GRPO (Open-R1) — dense + sparse env knobs

Canonical defaults live in `scripts/grpo_training_env_defaults.sh` (sourced by `scripts/grpo_openr1_llama31_slurm.sh`); they match `docs/hyperparams/open_r1_llama31.yaml`. **Verify each Slurm run** with the launcher log line `seq caps: prompt=… completion=…` and `STEPS=…` — older checkpoints may have been trained under a 1024 completion cap if `GRPO_MAX_COMPLETION_LENGTH` was not exported. Longer dense runs can override `GRPO_TARGET_STEPS` (e.g. `5000` in copy-paste blocks).

```latex
\begin{table}[t]
  \centering
  \small
  \caption{GRPO (Open-R1 style) training hyperparameters and runtime knobs (environment variables). Defaults match \texttt{grpo\_training\_env\_defaults.sh} / \texttt{open\_r1\_llama31.yaml}. Negative numeric entries are placeholders for an as-run snapshot.}
  \label{tab:appendix-grpo-env}
  \begin{tabular}{@{}lll@{}}
    \toprule
    Env var & Default (repo) & Value used (paste from snapshot) \\
    \midrule
    \texttt{MODEL} & \texttt{meta-llama/Llama-3.1-8B-Instruct} & \texttt{NEG} \\
    \texttt{GRPO\_DATASET} & \texttt{math-220k} (HF: \texttt{open-r1/OpenR1-Math-220k}) & \texttt{NEG} \\
    \texttt{GRPO\_MODE} & \texttt{dense} / \texttt{sparse} & \texttt{NEG} \\
    \texttt{GRPO\_NGPUS} & 1 & -1 \\
    \texttt{GRPO\_TARGET\_STEPS} & 500 & -1 \\
    \texttt{GRPO\_RESUME} & \textit{empty} (\texttt{auto} if resuming) & \texttt{NEG} \\
    \texttt{GRPO\_LR} & \texttt{5e-6} & \texttt{NEG} \\
    \texttt{GRPO\_BETA} & \texttt{0.025} & \texttt{NEG} \\
    \texttt{GRPO\_PER\_DEVICE\_BS} & 2 & -1 \\
    \texttt{GRPO\_GRAD\_ACCUM} & 4 & -1 \\
    \texttt{GRPO\_NUM\_GEN} & 8 & -1 \\
    \texttt{GRPO\_GEN\_BATCH} & 8 & -1 \\
    \texttt{GRPO\_MAX\_PROMPT\_LENGTH} & 512 & -1 \\
    \texttt{GRPO\_MAX\_COMPLETION\_LENGTH} & 2048 & -1 \\
    \texttt{GRPO\_REWARD\_PROFILE} & \texttt{llama\_cot} & \texttt{NEG} \\
    \texttt{GRPO\_OPTIM} & \texttt{adamw\_8bit} & \texttt{NEG} \\
    \texttt{GRPO\_PRECISION} & \texttt{bf16} & \texttt{NEG} \\
    \texttt{GRPO\_WARMUP\_RATIO} & \texttt{0.1} (sparse: launcher maps to integer warmup steps) & \texttt{NEG} \\
    \texttt{GRPO\_MAX\_GRAD\_NORM} & \texttt{0.1} & \texttt{NEG} \\
    \texttt{GRPO\_SAVE\_STEPS} & 50 & -1 \\
    \texttt{GRPO\_SAVE\_TOTAL\_LIMIT} & 3 & -1 \\
    \texttt{GRPO\_RUN\_SLUG} (dense) & \textit{unset} & \texttt{NEG} \\
    \texttt{GRPO\_RUN\_NAME} (sparse) & \textit{unset} & \texttt{NEG} \\
    \texttt{GRPO\_DELTA\_LOG\_INTERVAL} / \texttt{GRPO\_DELTA\_LOG\_END\_STEP} & \textit{unset} (set for magnitude-mask source runs) & \texttt{NEG} \\
    \texttt{GRPO\_MASK} (sparse) & \textit{unset} & \texttt{NEG} \\
    \texttt{GRPO\_SPARSE\_ADAMW\_LAZY} & 0 & -1 \\
    \texttt{HF\_DATASETS\_CACHE} & \texttt{\$\{RL\_CASINO\_SCRATCH\_ROOT\}/hf\_cache/datasets} & \texttt{NEG} \\
    \texttt{RL\_CASINO\_SCRATCH\_ROOT} & \textit{cluster scratch} & \texttt{NEG} \\
    \bottomrule
  \end{tabular}
\end{table}
```

---

### Sparse-kernel / speed-ablation knobs (BSR + SparseAdamW)

These apply when using BSR sparse backprop and/or SparseAdamW (DPO or GRPO), including the H200 benchmark scripts.

```latex
\begin{table}[t]
  \centering
  \small
  \caption{Sparse-kernel and logging knobs used in speed ablations (environment variables / flags). Negative numeric entries are placeholders.}
  \label{tab:appendix-sparse-kernel-knobs}
  \begin{tabular}{@{}lll@{}}
    \toprule
    Knob & Default (repo) & Value used \\
    \midrule
    \texttt{RL\_CASINO\_ADAM\_KERNEL} & \texttt{block\_1d} / \texttt{block\_2d} (H200 benchmark phase grid) & \texttt{NEG} \\
    \texttt{H200\_BSR\_SKIP\_DENSE} & 0 (= include one dense baseline; set 1 to omit) & -1 \\
    \texttt{BENCHMARK\_SPARSITIES} & \texttt{99.75,97.5,95,90} (\texttt{h200\_sparse\_dpo\_bsr\_benchmark.sh} driver default; comma-separated) & \texttt{NEG} \\
    \texttt{RL\_CASINO\_BSR\_GRAD\_INPUT\_MODE} & \texttt{dense} (\texttt{h200\_sparse\_dpo\_bsr\_benchmark.sh} default; sparse grad\_i only if overridden elsewhere) & \texttt{NEG} \\
    \texttt{RL\_CASINO\_BSR\_DETAILED\_TIMING} & \texttt{0} (off for sweeps; \texttt{1} enables \texttt{t\_*} CSV columns + per-micro-batch sync) & -1 \\
    \texttt{BSR\_USE\_ATOMIC} & 0 & -1 \\
    \texttt{BSR\_BATCH\_CHUNKS} & 8 & -1 \\
    \texttt{TRITON\_CACHE\_DIR} & \textit{scratch path} & \texttt{NEG} \\
    \texttt{RL\_CASINO\_LOGGING\_STEPS} & 1 (benchmark-style) & -1 \\
    \texttt{RL\_CASINO\_DISABLE\_TQDM} & 1 (Slurm-safe) & -1 \\
    \bottomrule
  \end{tabular}
\end{table}
```

---

### Note: some “speed ablation” scripts hard-code values

For example `scripts/benchmark_speedup.sh` and `scripts/run_ablation_warm_masks.sh` set several hyperparameters as local bash variables (not env vars). If you use those for a paper figure, consider adding a small table in the appendix that directly lists those script-local settings (model, subset size, steps, batch, grad accum, LR, sparsity, etc.) alongside the git commit hash.
