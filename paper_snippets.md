\begin{table}[t]
  \centering
  \small
  \caption{DPO throughput benchmark --- H200 sparse BSR / SparseAdamW grid (Llama 3.1 8B Instruct, randomized global masks).}
  \label{tab:h200-bsr-dpo-benchmark-hparams}
  \begin{tabular}{@{}ll@{}}
    \toprule
    Hyperparameter & Value \\
    \midrule
    Hardware & 1 $\times$ NVIDIA H200 \\
    Launcher defaults & \texttt{scripts/h200\_sparse\_dpo\_bsr\_benchmark.sh} (pass-through to \texttt{h200\_sparse\_dpo\_bsr\_benchmark.py}) \\
    Base model & \texttt{meta-llama/Llama-3.1-8B-Instruct} \\
    Preference dataset & \texttt{tulu3} (\texttt{--dataset}) \\
    Device placement & Single-GPU HF load (\texttt{--device\_map none}) \\
    Epochs per phase & 1 \\
    Optimizer updates per phase (max steps) & 25 (\texttt{H200\_BSR\_STEPS\_PER\_PHASE}; \texttt{--n\_steps}) \\
    Learning rate & $5 \times 10^{-7}$ \\
    LR schedule & Linear (\texttt{lr\_scheduler\_type=linear}); warmup ratio $0.1$ (\texttt{--warmup\_ratio}) \\
    Weight decay & $0.0$ \\
    DPO $\beta$ & $0.1$ \\
    Max gradient norm & $1.0$ \\
    Per-device batch size & 2 \\
    Gradient accumulation steps & 64 \\
    Effective batch (1 GPU; pairs per optimizer step) & $2 \times 64 = 128$ chosen/rejected prompts \\
    Max prompt length (tokens) & 1024 \\
    Max sequence length (tokens) & 1024 \\
    Floating-point precision & BF16 (\texttt{torch.bfloat16} model load); DPO Trainer \texttt{bf16=True} \\
    TF32 (cuBLAS) / BSR kernels & TF32-backed path enabled unless \texttt{--disable\_tf32} is set (launcher does not enable the flag by default). \\
    Dense baseline phase & Included by default (\texttt{H200\_BSR\_SKIP\_DENSE=0}): dense \texttt{nn.Linear} stack; Trainer optimizer \textbf{AdamW 8-bit} (\texttt{DPO\_OPTIM=adamw\_8bit}, bitsandbytes) \\
    Sparse phases & SparseLinear BSR backward on weights (wrapper fixes \texttt{RL\_CASINO\_BSR\_GRAD\_INPUT\_MODE=dense}: dense grad-input w.r.t.\ activations) \\
    Sparse optimizer & SparseAdamW (\texttt{RL\_CASINO\_ADAM\_KERNEL} toggles block layout per phase below) \\
    Sparse factorial grid & 16 timed sparse configurations per job: \{90, 95, 97.5, $99.75$\}\,\% targeted global sparsity $\times$ unstructured vs.\ block mask $\times$ \{block\_1d, block\_2d\} SparseAdamW kernels \\
    Dense baseline counting & +1 dense phase when baseline not skipped (\texttt{17} phases total vs.\ 16 sparse only when \texttt{--no\_dense\_baseline}) \\
    Default sparsity list & \texttt{BENCHMARK\_SPARSITIES}= \texttt{99.75,97.5,95,90} (comma-separated grid; orchestrator shards may override per subjob; ``0'' maps to dense-only shard in \texttt{h200\_bsr\_orchestrate.sh}) \\
    Mask construction & Random global scores (\texttt{method=random\_global\_benchmark}); deterministic seed per (\%, mask type); per-parameter retain floor $2.5 \times 10^{-3}$ (\texttt{DEFAULT\_MIN\_LAYER\_KEEP\_RATIO}) \\
    Mask scope & Full model (all masked 2-D linear weights; launcher does \emph{not} pass \texttt{--mlp\_only}) \\
    BSR tile / SparseAdamW block tile & \texttt{--block\_size\_bsr} $= 16$; \texttt{--block\_size\_adam} $= 128$ \\
    Throughput instrumentation & Per-step Trainer logging (\texttt{RL\_CASINO\_LOGGING\_STEPS=1}); micro-batch CUDA profiler columns \emph{off} by default (\texttt{RL\_CASINO\_BSR\_DETAILED\_TIMING=0}) \\
    Stable BSR kernels (env) & \texttt{BSR\_USE\_ATOMIC=0}, \texttt{BSR\_BATCH\_CHUNKS=1} \\
    Adam $\beta_1$, $\beta_2$, $\epsilon$ & $0.9$, $0.999$, $10^{-8}$ \\
    Gradient checkpointing & Enabled (pass \texttt{--no\_gradient\_checkpointing} only if \texttt{DPO\_GRADIENT\_CHECKPOINTING=0}) \\
    Checkpoint / HF saves & Disabled (\texttt{save\_strategy=no}, no model checkpoints) \\
    \bottomrule
  \end{tabular}
\end{table}

% =============================================================================
% Optimizer-step microbench (SparseAdamW vs AdamW vs AdamW 8-bit) — AWAITING RERUN
%
% Do **not** populate this table until the matched-policy rerun is done. All rows must come from
% jobs whose `optimizer_step_microbench.md` reports:
%   max_total_numel = 525000000   max_tensors = 64
%   selection_order = model_order  cap_behavior = break
%   est_param_MB    ≈ 1050  (single ~525M element tensor: embed_tokens.weight or lm_head.weight)
%
% Source: scripts/microbench_optimizer_step.py via scripts/h200_sparse_adamw_optstep_microbench.slurm
%   STEPS=50 TRIM_FRAC=0.10 LR=5e-7 BLOCK_SIZE=32 dtype=bf16 sync_cuda=1
%
% Aggregator command (after every shard `optimizer_step_microbench.md` is verified):
%   python scripts/aggregate_optstep_microbench_sweep.py \
%       --root /scratch/$USER/rl_casino_optstep_microbench \
%       --out-dir /scratch/$USER/rl_casino_optstep_microbench/aggregate_matched_<timestamp> \
%       --jobid <0p25> --jobid <50> --jobid <75> --jobid <90> --jobid <97p5> --jobid <99p75>
%
% Drop matched-policy `mean_ms_mid` values (in milliseconds) into the rows below; speedups are
% computed against `adamw_torch` and `adamw_8bit` per row.
% =============================================================================
\begin{table}[t]
  \centering
  \small
  \caption{Optimizer-step microbench at matched tensor-subset policy (\textsc{model\_order} +
    \textsc{break}, \texttt{max\_total\_numel}=$5.25\times10^{8}$, $\sim$525M-element BF16 subset
    $\approx$ 1050\,MB; bf16, \texttt{steps}=50, \texttt{trim\_frac}=0.10). All rows reuse the same
    pre-generated random masks. \textbf{TODO: fill rows after matched-policy rerun.}}
  \label{tab:optstep-microbench-matched}
  \begin{tabular}{@{}lrrrrr@{}}
    \toprule
    Sparsity (\%) & \texttt{adamw\_torch} (ms) & \texttt{adamw\_8bit} (ms) & \texttt{sparse\_adamw} (ms) & vs.\ torch & vs.\ 8-bit \\
    \midrule
    0.25  & --- & --- & --- & --- & --- \\
    50    & --- & --- & --- & --- & --- \\
    75    & --- & --- & --- & --- & --- \\
    90    & --- & --- & --- & --- & --- \\
    97.5  & --- & --- & --- & --- & --- \\
    99.75 & --- & --- & --- & --- & --- \\
    \bottomrule
  \end{tabular}
\end{table}