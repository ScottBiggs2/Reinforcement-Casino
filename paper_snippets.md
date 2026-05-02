# Paper snippets — H200 BSR DPO benchmark tables

Placeholder convention: **any negative numeric cell** (e.g. `-9.876`) or the token `NEG` is **dummy data**. Replace after aggregating `benchmark_training_log.csv` (e.g., mean/median over last `k` steps per phase).

Suggested aggregation: pool rows by `phase` (and optionally last `steps` steps only, after warmup), then report mean \(\pm\) std for `t_step_total_ms`, `t_backward_ms`, `t_optim_ms`, `t_forward_ms`, and derived `eff_bsr_backward_tflops`.

---

## Main results — timing highlights (tabular)

Compact comparison: dense baseline vs sparse settings at representative sparsities. Tune row count to your narrative.

```latex
\begin{table}[t]
  \centering
  \small
  \caption{H200 sparse BSR vs dense baseline — end-to-end step and component times (milliseconds). Negative entries are placeholders pending experiment aggregation.}
  \label{tab:bsr-timing-highlights}
  \begin{tabular}{@{}lcccccc@{}}
    \toprule
    Setting & Sparse (\%) & Adam kernel & $t_{\mathrm{step}}$ (ms) & $t_{\mathrm{fwd}}$ (ms) & $t_{\mathrm{bwd}}$ (ms) & $t_{\mathrm{opt}}$ (ms) \\
    \midrule
    Dense baseline & --- & adamw\textsubscript{(dense)} & -123.456 & -123.456 & -123.456 & -123.456 \\
    Sparse (elem mask) & 99.75 & block\_1d & -123.456 & -123.456 & -123.456 & -123.456 \\
    Sparse (elem mask) & 99.75 & block\_2d & -123.456 & -123.456 & -123.456 & -123.456 \\
    Sparse (blk mask) & 97.5  & block\_1d & -123.456 & -123.456 & -123.456 & -123.456 \\
    Sparse (blk mask) & 97.5  & block\_2d & -123.456 & -123.456 & -123.456 & -123.456 \\
    \bottomrule
  \end{tabular}
\end{table}
```

Throughput / proxy-FLOPs companion (same runs; backward proxy from CSV `eff_bsr_backward_tflops` — omit or blank dense rows).

```latex
\begin{table}[t]
  \centering
  \small
  \caption{Throughput and backward proxy throughput (dense rows left blank — no BSR theory columns). Negative entries are placeholders.}
  \label{tab:bsr-throughput-highlights}
  \begin{tabular}{@{}lccc@{}}
    \toprule
    Setting & Steps/s & Samples/s & BWD proxy TFLOP/s \\
    \midrule
    Dense baseline & -9.876 & -987654 & --- \\
    Sparse (example) & -9.876 & -987654 & -9.876 \\
    \bottomrule
  \end{tabular}
\end{table}
```

---

## Appendix — full benchmark grid + hyperparameters

One wide table covering every phase in the scripted grid: **1 dense phase** plus, per sparsity \( \in \{90, 95, 97.5, 99.75\} \): **elem vs block mask** \(\times\) **block\_1d vs block\_2d** SparseAdamW, with **dense** `grad_input` (matching `benchmark_training_log.csv` `phase` names, e.g. \texttt{s99p75\_elem\_gidense\_block\_1d}).

```latex
\begin{table*}[t]
  \centering
  \footnotesize
  \caption{Complete H200 BSR DPO benchmark phases: timings and CSV-derived metrics (all negative numbers are placeholders).}
  \label{tab:bsr-appendix-full-grid}
  \begin{tabular}{@{}lccccccccc@{}}
    \toprule
    Phase tag & Sparse (\%) & Mask & GI & Kernel &
    $\bar{t}_{\mathrm{step}}$ (ms) & $\bar{t}_{\mathrm{fwd}}$ (ms) &
    $\bar{t}_{\mathrm{bwd}}$ (ms) & $\bar{t}_{\mathrm{opt}}$ (ms) &
    BWD TFLOP/s (proxy) \\
    \midrule
    \texttt{dense} & --- & --- & --- & adamw\textsubscript{dense}
      & -1.111 & -1.111 & -1.111 & -1.111 & --- \\
    \texttt{s99p75\_elem\_gidense\_block\_1d} & 99.75 & elem & dense & block\_1d
      & -2.222 & -2.222 & -2.222 & -2.222 & -9.876 \\
    \texttt{s99p75\_elem\_gidense\_block\_2d} & 99.75 & elem & dense & block\_2d
      & -2.222 & -2.222 & -2.222 & -2.222 & -9.876 \\
    \texttt{s99p75\_blk\_gidense\_block\_1d} & 99.75 & blk & dense & block\_1d
      & -3.333 & -3.333 & -3.333 & -3.333 & -9.876 \\
    \texttt{s99p75\_blk\_gidense\_block\_2d} & 99.75 & blk & dense & block\_2d
      & -3.333 & -3.333 & -3.333 & -3.333 & -9.876 \\
    \multicolumn{10}{c}{\(\vdots\) \quad replicate rows for 97.5, 95, 90~\% \quad \(\vdots\)} \\
    \bottomrule
  \end{tabular}
\end{table*}
```

Hyperparameters / experimental setup notes (verbatim for appendix text).

```latex
\paragraph{Benchmark hyperparameters.}
Random global masks target sparsities $\{99.75, 97.5, 95, 90\}\%$.
Tokenizer and DPO preference dataset loaded once (\texttt{tulu3} registry / Allen AI mixture); each phase reloads model weights so dense and sparse graphs are not fused across phases.
Per-device batch size, gradient accumulation, learning rate, LR warmup fraction, sequence caps, optimizer choice for the dense baseline, gradient checkpointing, BSR block size, and SparseAdamW block size match the deployed Slurm driver defaults (replace every \textit{TBD} with the value from your Slurm env / job log).

\begin{itemize}\setlength\itemsep{0pt}
  \item \textbf{Hardware:} one NVIDIA H200 (cluster partition as submitted).
  \item \textbf{Model:} \texttt{meta-llama/Llama-3.1-8B-Instruct} (substitute actual checkpoint id if pinned).
  \item \textbf{Training steps per phase:} \textit{TBD numeric} (\texttt{H200\_BSR\_STEPS\_PER\_PHASE}; script default often small for throughput probe).
  \item \textbf{Batch / accum:} per-device BS \textit{TBD}, grad accum \textit{TBD} \(\Rightarrow\) effective samples per optimizer step per CSV.
  \item \textbf{LR / schedule:} base LR \textit{TBD}, warmup ratio \textit{TBD}, linear LR schedule (trainer); weight decay \textit{TBD}.
  \item \textbf{Sequence length:} max prompt \textit{TBD}, max length \textit{TBD} (DPO pairs).
  \item \textbf{Dense optimizer:} \texttt{DPO\_OPTIM} (\texttt{adamw\_8bit} vs full AdamW --- report which resolved at runtime).
  \item \textbf{Sparse phases:} BSR MLP substitution with \texttt{SparseAdamW}; \texttt{RL\_CASINO\_ADAM\_KERNEL} \(\in\) \texttt{\{block\_1d, block\_2d\}}; \texttt{RL\_CASINO\_BSR\_GRAD\_INPUT\_MODE=dense} for this grid.
  \item \textbf{BSR / Triton:} BSR block size \textit{TBD}, \texttt{BSR\_USE\_ATOMIC}, \texttt{BSR\_BATCH\_CHUNKS}; Triton cache on scratch (\texttt{TRITON\_CACHE\_DIR}).
  \item \textbf{Logging:} \texttt{benchmark\_training\_log.csv}: \texttt{t\_step\_total\_ms}, \texttt{t\_forward\_ms}, \texttt{t\_backward\_ms}, \texttt{t\_optim\_ms}; throughput \texttt{cumulative\_steps\_per\_s}, \texttt{cumulative\_samples\_per\_s}; proxy FLOPs from \texttt{theory\_*} and \texttt{eff\_bsr\_backward\_tflops} (theory-bound for masked-linear backward, not full forward pass).
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

### GRPO (Open-R1) — dense + sparse env knobs

Defaults are set by `scripts/grpo_openr1_llama31_slurm.sh` (and optionally orchestrated via `scripts/orchestrate_masks_then_queue_dpo_grpo.slurm` for multi-job pipelines).

```latex
\begin{table}[t]
  \centering
  \small
  \caption{GRPO (Open-R1 style) training hyperparameters and runtime knobs (environment variables). Negative numeric entries are placeholders.}
  \label{tab:appendix-grpo-env}
  \begin{tabular}{@{}lll@{}}
    \toprule
    Env var & Default (repo) & Value used (paste from snapshot) \\
    \midrule
    \texttt{MODEL} & \texttt{meta-llama/Llama-3.1-8B-Instruct} & \texttt{NEG} \\
    \texttt{GRPO\_DATASET} & \texttt{math-220k} & \texttt{NEG} \\
    \texttt{GRPO\_MODE} & \texttt{dense} & \texttt{NEG} \\
    \texttt{GRPO\_TARGET\_STEPS} & 1000 (\texttt{grpo\_openr1\_*}) / 5000 (\texttt{orchestrate\_*}) & -1 \\
    \texttt{GRPO\_RESUME} & \textit{empty} & \texttt{NEG} \\
    \texttt{GRPO\_LR} & \texttt{5e-6} & \texttt{NEG} \\
    \texttt{GRPO\_BETA} & \texttt{0.025} & \texttt{NEG} \\
    \texttt{GRPO\_PER\_DEVICE\_BS} & 2 & -1 \\
    \texttt{GRPO\_GRAD\_ACCUM} & 4 & -1 \\
    \texttt{GRPO\_NUM\_GEN} & 8 & -1 \\
    \texttt{GRPO\_GEN\_BATCH} & 8 & -1 \\
    \texttt{GRPO\_MAX\_PROMPT\_LENGTH} & 512 & -1 \\
    \texttt{GRPO\_MAX\_COMPLETION\_LENGTH} & 1024 (\texttt{grpo\_openr1\_*}) / 2048 (\texttt{orchestrate\_*}) & -1 \\
    \texttt{GRPO\_REWARD\_PROFILE} & \texttt{llama\_cot} & \texttt{NEG} \\
    \texttt{GRPO\_OPTIM} & \texttt{adamw\_8bit} & \texttt{NEG} \\
    \texttt{GRPO\_PRECISION} & \texttt{bf16} & \texttt{NEG} \\
    \texttt{GRPO\_SAVE\_STEPS} & 50 & -1 \\
    \texttt{GRPO\_SAVE\_TOTAL\_LIMIT} & 3 & -1 \\
    \texttt{GRPO\_MASK} (sparse) & \textit{unset} & \texttt{NEG} \\
    \texttt{GRPO\_SPARSE\_ADAMW\_LAZY} & 0 & -1 \\
    \texttt{HF\_DATASETS\_CACHE} & \textit{scratch path} & \texttt{NEG} \\
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
    \texttt{RL\_CASINO\_ADAM\_KERNEL} & \texttt{block\_1d} / \texttt{block\_2d} (phase grid) & \texttt{NEG} \\
    \texttt{RL\_CASINO\_BSR\_GRAD\_INPUT\_MODE} & \texttt{dense} or \texttt{sparse} (phase grid) & \texttt{NEG} \\
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

# Speed benchmark (Final?)
(base) [biggs.s@explorer-02 rl_casino]$ export RL_CASINO_BSR_GRAD_INPUT_MODE=dense
sbatch scripts/h200_sparse_dpo_bsr_benchmark.sh
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

# Sparse DPO submit (500 steps) for each dataset, using pipeline_sparse_o  scripts/pipeline_sparse_one_mask.shELINE_MASK_FILE="$MASK_LIGHTR1" \SET
Mask job id: 6497892
Submitted batch job 6497893
Submitted batch job 6497894

