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
