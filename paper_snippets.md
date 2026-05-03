# Paper snippets — H200 BSR DPO benchmark tables

Numeric tables in the **export block** below come from [`scripts/export_h200_bsr_paper_tables.py`](scripts/export_h200_bsr_paper_tables.py) (aggregating `benchmark_training_log.csv`). Elsewhere in this file, **NEG** / negative placeholders still mean “paste from env snapshot.” Driver: [`scripts/h200_sparse_dpo_bsr_benchmark.sh`](scripts/h200_sparse_dpo_bsr_benchmark.sh).

### CSV columns and aggregation (read before filling tables)

**Always present** (default sweep, `RL_CASINO_BSR_DETAILED_TIMING=0`): `phase`, `wall_time_s`, `cumulative_steps_per_s`, `cumulative_samples_per_s`, trainer metrics (`loss`, etc.), and theory-sidecar fields (`theory_*`, duplicated per row). Use throughput columns for end-to-end comparisons without CUDA instrumentation overhead.

**CUDA segment timings** (`t_step_total_ms`, `t_forward_ms`, `t_backward_ms`, `t_optim_ms`, `t_nonoptim_ms`, `t_other_ms`) appear **only** when `RL_CASINO_BSR_DETAILED_TIMING=1`. That mode wraps each micro-batch forward/backward/optimizer step with `torch.cuda.synchronize()` and can **severely slow** training when `DPO_GRADIENT_ACCUMULATION_STEPS` is large—use it only for short diagnostic phases, not full grids.

**Caveats when detailed timing is on:**

- `t_forward_ms` / `t_backward_ms` reflect the **last gradient-accumulation micro-batch only**, not the full optimizer step; `t_other_ms` can look huge if interpreted as “unexplained” time.
- `eff_bsr_backward_tflops` / `eff_bsr_backward_flops_per_s` divide theory (per optimizer step) by backward time; `t_backward_ms` is **one** micro-batch, so downstream code scales by ``trainer_grad_accum_steps``. **Older CSVs** before that fix may be inflated by ~`grad_accum` on those columns — run [`scripts/recalc_benchmark_training_log_eff.py`](scripts/recalc_benchmark_training_log_eff.py) to rewrite them without retraining.

**Quick per-phase summary on-cluster or locally:**

```bash
python scripts/analyze_benchmark_training_log.py /path/to/benchmark_training_log.csv --grad-accum 64
```

Table numbers below are **generated** from `benchmark_training_log.csv` (mean over the last `--tail-rows` logged rows per `phase`). Re-export whenever the CSV changes; do not hand-edit the export block.

**Fill / refresh (repo root):**

```bash
python scripts/export_h200_bsr_paper_tables.py \
  --csv /path/to/benchmark_training_log.csv \
  --inject paper_snippets.md \
  --tail-rows 8
```

Example layout CSV (drafting only — **replace with your measured CSV** before submission): `scripts/fixtures/h200_bsr_table_export_example.csv`.

**Grid reference:** default driver uses **1 dense** + **4 sparse phases per sparsity** (elem vs block \(\times\) block\_1d vs block\_2d, dense grad\_input). Full sweeps with four sparsities \(\{99.75,97.5,95,90\}\%\) yield **17** rows when the job completes; partial runs contribute fewer rows (export reflects whatever phases appear in the CSV).

<!-- H200_BSR_PAPER_EXPORT_START -->

## Auto-filled from benchmark CSV (edit upstream CSV + re-export; do not hand-tune numbers here)

<!-- Generated from `h200_bsr_table_export_example.csv` — re-run export after updating the CSV. -->

### Aggregated throughput (mean of last logged rows per phase)

| Phase | Sparse % | Optimizer | Steps/s | Samples/s | Wall (tail mean, s) |
|-------|----------|-----------|---------|-----------|---------------------|
| `dense` |  | adamw_8bit | 0.024800 | 3.1700 | 322.80 |
| `s99p75_elem_gidense_block_1d` | 99.75 | sparse_adamw | 0.001230 | 0.1580 | 6488.00 |
| `s99p75_elem_gidense_block_2d` | 99.75 | sparse_adamw | 0.001230 | 0.1580 | 6487.00 |
| `s99p75_blk_gidense_block_1d` | 99.75 | sparse_adamw | 0.001540 | 0.1970 | 5210.00 |
| `s99p75_blk_gidense_block_2d` | 99.75 | sparse_adamw | 0.001540 | 0.1970 | 5205.00 |
| `s97p5_elem_gidense_block_1d` | 97.5 | sparse_adamw | 0.001310 | 0.1680 | 6100.00 |
| `s97p5_elem_gidense_block_2d` | 97.5 | sparse_adamw | 0.001310 | 0.1680 | 6095.00 |
| `s97p5_blk_gidense_block_1d` | 97.5 | sparse_adamw | 0.001670 | 0.2140 | 4800.00 |
| `s97p5_blk_gidense_block_2d` | 97.5 | sparse_adamw | 0.001670 | 0.2140 | 4795.00 |

```latex
\begin{table}[t]
  \centering
  \small
  \caption{H200 BSR — mean step/component times (ms) per phase when \texttt{t\_*} columns exist in CSV; otherwise use throughput table only.}
  \label{tab:bsr-timing-highlights-autogen}
  \begin{tabular}{@{}lcccccc@{}}
    \toprule
    Phase & Sparse (\%) & Optimizer & $t_{\mathrm{step}}$ & $t_{\mathrm{fwd}}$ & $t_{\mathrm{bwd}}$ & $t_{\mathrm{opt}}$ \\
    \midrule
    \texttt{dense} & --- & adamw\_8bit & --- & --- & --- & --- \\
    \texttt{s99p75\_elem\_gidense\_block\_1d} & 99.75 & sparse\_adamw & --- & --- & --- & --- \\
    \texttt{s99p75\_elem\_gidense\_block\_2d} & 99.75 & sparse\_adamw & --- & --- & --- & --- \\
    \texttt{s97p5\_blk\_gidense\_block\_1d} & 97.5 & sparse\_adamw & --- & --- & --- & --- \\
    \texttt{s97p5\_blk\_gidense\_block\_2d} & 97.5 & sparse\_adamw & --- & --- & --- & --- \\
    \bottomrule
  \end{tabular}
\end{table}
```

```latex
\begin{table}[t]
  \centering
  \small
  \caption{H200 BSR — throughput (mean over tail rows per phase).}
  \label{tab:bsr-throughput-autogen}
  \begin{tabular}{@{}lccc@{}}
    \toprule
    Phase & Steps/s & Samples/s & BWD TFLOP/s (proxy) \\
    \midrule
    \texttt{dense} & 0.024800 & 3.1700 & --- \\
    \texttt{s99p75\_elem\_gidense\_block\_1d} & 0.001230 & 0.1580 & --- \\
    \texttt{s99p75\_elem\_gidense\_block\_2d} & 0.001230 & 0.1580 & --- \\
    \texttt{s99p75\_blk\_gidense\_block\_1d} & 0.001540 & 0.1970 & --- \\
    \texttt{s99p75\_blk\_gidense\_block\_2d} & 0.001540 & 0.1970 & --- \\
    \texttt{s97p5\_elem\_gidense\_block\_1d} & 0.001310 & 0.1680 & --- \\
    \texttt{s97p5\_elem\_gidense\_block\_2d} & 0.001310 & 0.1680 & --- \\
    \texttt{s97p5\_blk\_gidense\_block\_1d} & 0.001670 & 0.2140 & --- \\
    \texttt{s97p5\_blk\_gidense\_block\_2d} & 0.001670 & 0.2140 & --- \\
    \bottomrule
  \end{tabular}
\end{table}
```

```latex
\begin{table*}[t]
  \centering
  \footnotesize
  \caption{H200 BSR — all phases in CSV (partial runs omit unfinished sparsity levels).}
  \label{tab:bsr-appendix-autogen}
  \begin{tabular}{@{}lccccccccc@{}}
    \toprule
    Phase tag & Sparse (\%) & Mask & GI & Kernel & $\bar{t}_{\mathrm{step}}$ & $\bar{t}_{\mathrm{fwd}}$ & $\bar{t}_{\mathrm{bwd}}$ & $\bar{t}_{\mathrm{opt}}$ & BWD TFLOP/s \\
    \midrule
    \texttt{dense} & --- & --- & --- & adamw\textsubscript{dense} & --- & --- & --- & --- & --- \\
    \texttt{s99p75\_elem\_gidense\_block\_1d} & 99.75 & elem & dense & block\_1d & --- & --- & --- & --- & --- \\
    \texttt{s99p75\_elem\_gidense\_block\_2d} & 99.75 & elem & dense & block\_2d & --- & --- & --- & --- & --- \\
    \texttt{s99p75\_blk\_gidense\_block\_1d} & 99.75 & blk & dense & block\_1d & --- & --- & --- & --- & --- \\
    \texttt{s99p75\_blk\_gidense\_block\_2d} & 99.75 & blk & dense & block\_2d & --- & --- & --- & --- & --- \\
    \texttt{s97p5\_elem\_gidense\_block\_1d} & 97.5 & elem & dense & block\_1d & --- & --- & --- & --- & --- \\
    \texttt{s97p5\_elem\_gidense\_block\_2d} & 97.5 & elem & dense & block\_2d & --- & --- & --- & --- & --- \\
    \texttt{s97p5\_blk\_gidense\_block\_1d} & 97.5 & blk & dense & block\_1d & --- & --- & --- & --- & --- \\
    \texttt{s97p5\_blk\_gidense\_block\_2d} & 97.5 & blk & dense & block\_2d & --- & --- & --- & --- & --- \\
    \bottomrule
  \end{tabular}
\end{table*}
```

<!-- H200_BSR_PAPER_EXPORT_END -->

---

## Appendix — benchmark hyperparameters (static prose)

Hyperparameters / experimental setup notes (verbatim for appendix text).

```latex
\paragraph{Benchmark hyperparameters.}
Random global masks target sparsities $\{99.75, 97.5, 95, 90\}\%$ (\texttt{BENCHMARK\_SPARSITIES}).
Tokenizer and DPO preference dataset loaded once (\texttt{tulu3} registry / Allen AI mixture); each phase reloads model weights so dense and sparse graphs are not fused across phases.
Slurm driver prints \texttt{GIT\_SHA} and job id at job start for provenance.
Per-device batch size, gradient accumulation, learning rate, LR warmup fraction, sequence caps, optimizer choice for the dense baseline, gradient checkpointing, BSR block size, and SparseAdamW block size match the deployed Slurm driver defaults (replace every \textit{TBD} with the value from your Slurm env / job log).

\begin{itemize}\setlength\itemsep{0pt}
  \item \textbf{Hardware:} one NVIDIA H200 (cluster partition as submitted).
  \item \textbf{Model:} \texttt{meta-llama/Llama-3.1-8B-Instruct} (substitute actual checkpoint id if pinned).
  \item \textbf{Training steps per phase:} \textit{TBD numeric} (\texttt{H200\_BSR\_STEPS\_PER\_PHASE}; script default often small for throughput probe).
  \item \textbf{Batch / accum:} per-device BS \textit{TBD}, grad accum \textit{TBD} \(\Rightarrow\) effective samples per optimizer step per CSV.
  \item \textbf{LR / schedule:} base LR \textit{TBD}, warmup ratio \textit{TBD}, linear LR schedule (trainer); weight decay \textit{TBD}.
  \item \textbf{Sequence length:} max prompt \textit{TBD}, max length \textit{TBD} (DPO pairs).
  \item \textbf{Dense optimizer:} \texttt{DPO\_OPTIM} (\texttt{adamw\_8bit} vs full AdamW --- report which resolved at runtime).
  \item \textbf{Sparse phases:} BSR MLP substitution with \texttt{SparseAdamW}; \texttt{RL\_CASINO\_ADAM\_KERNEL} \(\in\) \texttt{\{block\_1d, block\_2d\}}; driver passes \texttt{--phase\_grad\_input\_modes dense} and defaults \texttt{RL\_CASINO\_BSR\_GRAD\_INPUT\_MODE=dense}.
  \item \textbf{BSR / Triton:} BSR block size \textit{TBD}, \texttt{BSR\_USE\_ATOMIC}, \texttt{BSR\_BATCH\_CHUNKS}; Triton cache on scratch (\texttt{TRITON\_CACHE\_DIR}).
  \item \textbf{Logging:} \texttt{benchmark\_training\_log.csv}: always \texttt{phase}, wall-clock throughput (\texttt{cumulative\_steps\_per\_s}, \texttt{cumulative\_samples\_per\_s}), \texttt{theory\_*}; CUDA segment columns only if \texttt{RL\_CASINO\_BSR\_DETAILED\_TIMING=1}. Proxy FLOPs: \texttt{eff\_bsr\_backward\_tflops} (masked-linear backward theory vs.\ measured backward slice—see caveats above).
\end{itemize}
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

Defaults are primarily set/propagated by `scripts/pipeline_common.sh` (dense stage) and used again by sparse stages; long-run copy/paste blocks also live in `scripts/dpo_5k_hpc_copypaste.md`.

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

### GRPO (Open-R1) — training hyperparameters (paper table)

Quantities only (no Slurm names). These match the **locked** production defaults: `scripts/grpo_training_env_defaults.sh` and `docs/hyperparams/open_r1_llama31.yaml`. Dense uses `src/full_training/GRPO_train.py`; sparse uses `src/full_training/sparse_grpo_bsr.py` with the same LR, $\beta$, batch, generation, and sequence caps. Ablations require `GRPO_HPARAM_OVERRIDE=1` in the launcher. With 500 total steps and warmup ratio 0.1, sparse training uses 50 linear warmup steps (integer match to dense).

```latex
% Requires: \usepackage{booktabs}
\begin{table}[t]
  \centering
  \small
  \caption{GRPO (Open-R1 Math) training hyperparameters — Llama 3.1 8B Instruct, locked defaults.}
  \label{tab:grpo-training-hparams}
  \begin{tabular}{@{}ll@{}}
    \toprule
    Hyperparameter & Value \\
    \midrule
    Base model & \texttt{meta-llama/Llama-3.1-8B-Instruct} \\
    Preference / prompt dataset & \texttt{math-220k} (\texttt{open-r1/OpenR1-Math-220k}) \\
    Training steps & 500 \\
    Learning rate & $5 \times 10^{-6}$ \\
    LR schedule & Cosine decay; warmup ratio 0.1 (dense) / 50 warmup steps (sparse) \\
    Weight decay & 0.0 \\
    Max gradient norm & 0.1 \\
    GRPO $\beta$ (KL-style coeff.) & 0.025 \\
    Per-device train batch size & 2 \\
    Gradient accumulation steps & 4 \\
    Effective batch (1 GPU, per optimizer step) & 8 \\
    \texttt{num\_generations} & 8 \\
    \texttt{generation\_batch\_size} & 8 \\
    Max prompt length (tokens) & 512 \\
    Max completion length (tokens) & 2048 \\
    Reward / parsing profile & \texttt{llama\_cot} \\
    Precision & bf16 \\
    Optimizer (dense) & AdamW 8-bit (\texttt{adamw\_8bit}) \\
    Optimizer (sparse BSR) & SparseAdamW (\texttt{sparse\_adamw}) \\
    Checkpoint saves (HF) & Every 50 steps; retain 3 \\
    Activation gradient checkpointing & Enabled \\
    \bottomrule
  \end{tabular}
\end{table}
```

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


python src/warm_start/checkpoint_diff_mask_finder.py \
  --initial_model "meta-llama/Llama-3.1-8B-Instruct" \
  --final_model "/scratch/biggs.s/rl_casino_grpo/dense/grpo_dense_openr1_steps800_with_deltas_v1/checkpoints/checkpoint-500" \
  --sparsity_percent 97.5

  sbatch --export=ALL,HF_TOKEN="$HF_TOKEN",WANDB_API_KEY="$WANDB_API_KEY",\
GRPO_MODE=sparse,\
GRPO_TARGET_STEPS=500,\
GRPO_MASK="/scratch/$USER/rl_casino_grpo/masks/checkpoint_diff_ground_truth_checkpoint-500_sparsity97.5pct.pt",\
GRPO_RUN_NAME="grpo_oracle_500", \
scripts/grpo_openr1_llama31_slurm.sh


# Magnitude 200 training on Light r1
(base) [biggs.s@explorer-02 rl_casino]$ sbatch --export=ALL,HF_TOKEN="$HF_TOKEN",WANDB_API_KEY="$WANDB_API_KEY",\
PIPELINE_MASK_FILE="/scratch/$USER/rl_casino_masks/manual_mag_lr1_step200_sp97.5.pt",\
DPO_DATASET_KEY="light-r1",\
NUM_STEPS_DPO=500,\
RUN_ID="dpo500_mag_lr1_step200_manual",\
PIPELINE_RUN_ID="dpo500_mag_lr1_step200_manual" \
scripts/pipeline_sparse_one_mask.sh
Submitted batch job 6497790

#magnitde 200 training on tulu3
(base) [biggs.s@explorer-02 rl_casino]$ sbatch --export=ALL,HF_TOKEN="$HF_TOKEN",WANDB_API_KEY="$WANDB_API_KEY",\
PIPELINE_MASK_FILE="/scratch/$USER/rl_casino_masks/manual_mag_tulu3_step200_sp97.5.pt",\
DPO_DATASET_KEY="tulu3",\
NUM_STEPS_DPO=500,\
RUN_ID="dpo500_mag_tulu3_step200_manual",\
PIPELINE_RUN_ID="dpo500_mag_tulu3_step200_manual" \
scripts/pipeline_sparse_one_mask.sh
Submitted batch job 6497793

# H200 BSR throughput sweep (dense grad_input default in script; optional explicit export)
# sbatch scripts/h200_sparse_dpo_bsr_benchmark.sh
Submitted batch job 6497854

# GraSP (vanilla) GRPO
(base) [biggs.s@explorer-02 rl_casino]$ sbatch --export=ALL,HF_TOKEN="$HF_TOKEN",MODEL="meta-llama/Llama-3.1-8B-Instruct",GRPO_DATASET_HF="open-r1/OpenR1-Math-220k",SPARSITY_PERCENT=97.5,MIN_LAYER_KEEP_RATIO=0.0025,GRPO_N_SAMPLES=256,ORCH_MASK_BATCH_SIZE=1,ORCH_SCORE_SNR_MODE=off,GRPO_TARGET_STEPS=800,EXTRA_GRASP_FLAGS="--also-emit-rank-variants" \
  scripts/run_grpo_grasp_mask_only.slurm
Submitted batch job 6497890

# GraSP DPO Job(s)
(base) [biggs.s@explorer-02 rl_casino]$ MASK_RUN_ID="dpo_grasp_abs_$(date +%Y%m%d_%H%M%S)"
MASK_OUT_BASE="/scratch/$USER/rl_casino_masks"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
SP=97.5

jid=$(sbatch --parsable --export=ALL,HF_TOKEN="$HF_TOKEN",MASK_RUN_ID="$MASK_RUN_ID",MASK_OUT_BASE="$MASK_OUT_BASE",MODEL="$MODEL",SPARSITY_PERCENT="$SP",MIN_LAYER_KEEP_RATIO=0.0025,COLD_CAV_SUBSET=256,ORCH_MASK_BATCH_SIZE=1,ORCH_SCORE_SNR_MODE=off,EXTRA_GRASP_FLAGS="--also-emit-rank-variants" \
  scripts/run_dpo_grasp_masks_only.slurm)

echo "Mask job id: $jid"
MASK_DIR="${MASK_OUT_BASE}/${MASK_RUN_ID}"

# Expected primary (GraSP-ABS) mask paths produced by run_dpo_grasp_masks_only.slurm
MODEL_SANITIZED="$(echo "$MODEL" | tr '/-' '__' | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]_' '_' | sed -e 's/__/_/g' -e 's/^_//' -e 's/_$//')"
MASK_TULU3="${MASK_DIR}/grasp_${MODEL_SANITIZED}_tulu3_sp${SP}pct_objdpo_preference_elem_snroff_log1p.pt"
MASK_LIGHTR1="${MASK_DIR}/grasp_${MODEL_SANITIZED}_light_r1_sp${SP}pct_objdpo_preference_elem_snroff_log1p.pt"

# Sparse DPO (500 steps) — chain mask paths into pipeline_sparse_one_mask.sh as needed.
Mask job id: 6497892
Submitted batch job 6497893
Submitted batch job 6497894

---

### BSR-aware SparseAdamW kernels (block\_1d vs block\_2d)

Sparse phases of the H200 BSR DPO benchmark ([`scripts/h200_sparse_dpo_bsr_benchmark.sh`](scripts/h200_sparse_dpo_bsr_benchmark.sh)) use **SparseAdamW** with Triton kernels in [`src/kernels/indexed_sparse_adam.py`](src/kernels/indexed_sparse_adam.py). When a 2D parameter has a block-structured mask, [`src/utils/mask_manager.py`](src/utils/mask_manager.py) pads the mask to multiples of $B{=}16$, marks each $B{\times}B$ tile active if any entry is non-zero, and stores flat block indices $b \in \{0,\ldots,\lceil M/B\rceil\lceil N/B\rceil-1\}$ with $b = b_m \cdot \lceil N/B\rceil + b_n$. [`src/optimizers/sparse_adamw.py`](src/optimizers/sparse_adamw.py) then chooses the launch via \texttt{RL\_CASINO\_ADAM\_KERNEL}: \texttt{block\_1d} calls \texttt{triton\_bsr\_sparse\_adamw\_step} (1D grid, one program per active block); \texttt{block\_2d} calls \texttt{triton\_bsr\_sparse\_adamw\_step\_2d} (2D grid with row tiling). Optimizer states in the block regime are packed as length $K B^2$ vectors ($K$ active blocks). Host code precomputes bias corrections $\texttt{bc}_1 = 1-\beta_1^t$ and $\texttt{bc}_2 = 1-\beta_2^t$ for training step $t$. Optional launch tuning (\texttt{BSR\_NUM\_WARPS}, \texttt{BSR\_NUM\_STAGES}, \texttt{BSR\_ADAM\_ROW\_TILE}) affects occupancy only, not numerics.

\begin{algorithm}[h]
\caption{BSR-aware AdamW update --- \texttt{block\_1d} (1D grid, \texttt{bsr\_sparse\_adamw\_kernel})}
\label{alg:bsr-adamw-1d}
\begin{algorithmic}[1]
\Require Weight matrix $P \in \mathbb{R}^{M \times N}$, gradient $G$ (same shape); first/second moments $m,v$ (dense $M\times N$ \textbf{or} packed length $K B^2$ if sparse storage); active block list $\texttt{active\_blocks} \in \mathbb{Z}^{K}$; block size $B$ (\texttt{BLOCK\_SIZE}${}=16$); $(\texttt{lr},\beta_1,\beta_2,\varepsilon,\texttt{wd})$; bias corrections $\texttt{bc}_1,\texttt{bc}_2$; flag \texttt{USE\_SPARSE\_STATES}
\Ensure In-place updates to $P$, $m$, $v$ on active tiles only
\State Launch one program per active block: $\texttt{pid} \in \{0,\ldots,K-1\}$
\If{$\texttt{pid} \geq K$} \textbf{return} \EndIf
\State $b \gets \texttt{active\_blocks}[\texttt{pid}]$ \Comment{flat block id}
\State $N_b \gets \lceil N / B \rceil$,\quad $b_m \gets \lfloor b / N_b \rfloor$,\quad $b_n \gets b \bmod N_b$
\State Form row indices $r_s \gets b_m B + s$ and column indices $c_t \gets b_n B + t$ for $s,t \in \{0,\ldots,B-1\}$
\State Validity mask $\mathcal{V}_{s,t} \gets (r_s < M) \land (c_t < N)$
\State Load $B\times B$ tiles $\mathbf{p},\mathbf{g}$ from $P,G$ at $(r_s,c_t)$, treating out-of-bounds as $0$
\If{\texttt{USE\_SPARSE\_STATES}}
    \State Base offset $\ell_0 \gets \texttt{pid} \cdot B^2$; load $\mathbf{m},\mathbf{v}$ from contiguous slices $m[\ell_0{:}\ell_0{+}B^2)$, $v[\ell_0{:}\ell_0{+}B^2)$ reshaped in row-major order aligned with $(s,t)$
\Else
    \State Load $\mathbf{m},\mathbf{v}$ from $m,v$ at the same $(r_s,c_t)$ coordinates as $\mathbf{p}$
\EndIf
\State \textbf{AdamW:} $\mathbf{p}' \gets \mathbf{p} \odot (1 - \texttt{lr}\cdot\texttt{wd})$ \Comment{elementwise}
\State $\mathbf{m}' \gets \beta_1 \mathbf{m} + (1-\beta_1)\mathbf{g}$,\quad $\mathbf{v}' \gets \beta_2 \mathbf{v} + (1-\beta_2)\mathbf{g}\odot\mathbf{g}$
\State $\widehat{\mathbf{m}} \gets \mathbf{m}' / \texttt{bc}_1$,\quad $\widehat{\mathbf{v}} \gets \mathbf{v}' / \texttt{bc}_2$
\State $\mathbf{p}_{\mathrm{new}} \gets \mathbf{p}' - \texttt{lr}\cdot (\widehat{\mathbf{m}} / (\sqrt{\widehat{\mathbf{v}}} + \varepsilon))$ \Comment{``$/$'' is elementwise (Hadamard) division}
\State Store $\mathbf{p}_{\mathrm{new}}$ into $P$ at valid $(r_s,c_t)$; store $\mathbf{m}',\mathbf{v}'$ to the same storage layout used for loads
\end{algorithmic}
\end{algorithm}

\begin{algorithm}[h]
\caption{BSR-aware AdamW update --- \texttt{block\_2d} (2D grid, \texttt{bsr\_sparse\_adamw\_kernel\_2d}; \texttt{block\_1d} is the $T{=}B$ special case)}
\label{alg:bsr-adamw-2d}
\begin{algorithmic}[1]
\Require Same tensors and hyperparameters as Algorithm~\ref{alg:bsr-adamw-1d}; row tile size $T{=}$\texttt{ROW\_TILE} with $B/T \in \mathbb{Z}_{>0}$ (default $T{=}8$ when $B{=}16$)
\Ensure In-place updates to $P$, $m$, $v$ on active row-tiles only
\State Launch 2D grid: $\texttt{pid}_b \in \{0,\ldots,K-1\}$, $\texttt{pid}_t \in \{0,\ldots,B/T-1\}$
\If{$\texttt{pid}_b \geq K$} \textbf{return} \EndIf
\State $b \gets \texttt{active\_blocks}[\texttt{pid}_b]$;\quad $N_b \gets \lceil N / B \rceil$;\quad decode $(b_m,b_n)$ as in Algorithm~\ref{alg:bsr-adamw-1d}
\State Row slice $r_s \gets b_m B + \texttt{pid}_t\cdot T + s$ for $s \in \{0,\ldots,T-1\}$;\quad $c_t \gets b_n B + t$ for $t \in \{0,\ldots,B-1\}$
\State $\mathcal{V}_{s,t} \gets (r_s < M) \land (c_t < N)$;\quad load $T\times B$ tiles $\mathbf{p},\mathbf{g}$ from $P,G$
\If{\texttt{USE\_SPARSE\_STATES}}
    \State $\ell_0 \gets \texttt{pid}_b\cdot B^2$;\quad $\ell_1 \gets \texttt{pid}_t\cdot T\cdot B$;\quad load $\mathbf{m},\mathbf{v}$ from $m,v$ at offsets $\ell_0+\ell_1$ in row-major layout over the full $B\times B$ block
\Else
    \State Load $\mathbf{m},\mathbf{v}$ at coordinates $(r_s,c_t)$
\EndIf
\State $\mathbf{p}' \gets \mathbf{p} \odot (1 - \texttt{lr}\cdot\texttt{wd})$;\quad $\mathbf{m}' \gets \beta_1 \mathbf{m} + (1-\beta_1)\mathbf{g}$;\quad $\mathbf{v}' \gets \beta_2 \mathbf{v} + (1-\beta_2)\mathbf{g}\odot\mathbf{g}$
\State $\widehat{\mathbf{m}} \gets \mathbf{m}' / \texttt{bc}_1$;\quad $\widehat{\mathbf{v}} \gets \mathbf{v}' / \texttt{bc}_2$;\quad $\mathbf{p}_{\mathrm{new}} \gets \mathbf{p}' - \texttt{lr}\cdot (\widehat{\mathbf{m}} / (\sqrt{\widehat{\mathbf{v}}} + \varepsilon))$ \Comment{``$/$'' elementwise}
\State Scatter results back to $P,m,v$ with the same indexing as the loads
\end{algorithmic}
\end{algorithm}

