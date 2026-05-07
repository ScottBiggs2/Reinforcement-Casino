# SparseAdamW \texttt{optimizer.step()} microbench

## Experimental setup

The microbench isolates the optimizer kernel: only \texttt{optimizer.step()} is timed
(no forward, backward, or data-loading paths), with full CUDA synchronization before
and after each timed iteration. Each configuration is timed for 50 consecutive steps;
we drop the first and last 10\% of samples and report both the trimmed mean and the
trimmed median across the middle window. Synthetic gradients are pre-allocated and
reused, so the only quantity the optimizer reads/writes is the parameter / state
buffers.

| Setting | Value |
|---|---|
| Hardware | 1 \(\times\) NVIDIA H200 |
| Parameter and gradient dtype | BF16 |
| First / second moment dtype | fp32 for \texttt{adamw\_torch} and \texttt{sparse\_adamw}; 8-bit block-quantized for \texttt{adamw\_8bit} |
| Timed steps per case (mid-window) | 50 (drop first / last 10\%) |
| CUDA synchronization | enabled before and after every timed step |
| Synthetic parameter subset | a single 2-D weight tensor of shape matching the largest model parameter (\(\approx 525\)M BF16 elements) held fixed across sparsity levels |
| SparseAdamW tile size | 32 |
| Mask construction | element-wise random masks at six target sparsities; one mask per sparsity, reused across all three optimizers |
| Sparsity grid (\%) | 0.25, 50, 75, 90, 97.5, 99.75 |
| Compared optimizers | PyTorch AdamW (\texttt{adamw\_torch}), bitsandbytes AdamW 8-bit (\texttt{adamw\_8bit}), and SparseAdamW (\texttt{sparse\_adamw}, this work) |

The fixed \(\approx 525\)M-element subset is used so that the dense baselines
(\texttt{adamw\_torch}, \texttt{adamw\_8bit}) see exactly the same workload at every
sparsity level, isolating the effect of the SparseAdamW kernel's mask-aware
load/store and update pattern.

## Step-time results

\begin{table}[H]
    \centering
    \caption{
        BSR AdamW step-time speedup multipliers relative to \texttt{adamw\_torch} (PyTorch)
        dense baseline at each sparsity level and \texttt{adamw\_8bit} (HuggingFace) ~\citep{NEURIPS2019_9015, dettmers2022llmint8}. \textbf{Bold} indicates best result per row, \slower{Red} indicates slowdown multipliers, and \faster{Green} indicates speedup multipliers.
    }
    \label{tab:bsr-timing-highlights}
    \small
    \begin{tabular}{l ccc ccc}
        \toprule
        & \multicolumn{3}{c}{\textbf{(mean) ms / step}} 
        & \multicolumn{3}{c}{\textbf{(median) ms / step}} \\
        \cmidrule(lr){2-4} \cmidrule(lr){5-7}
        \textbf{Sparsity} 
            & \texttt{adamw\_torch} 
            & \texttt{adamw\_8bit} 
            & \texttt{sparse\_adamw} 
            & \texttt{adamw\_torch} 
            & \texttt{adamw\_8bit} 
            & \texttt{sparse\_adamw} \\
        \midrule
        0.25\% 
            & $7.1200$ 
            & $14.1088$ \slower{1.98} 
            & $\mathbf{6.2083}$ \faster{1.15} 
            & $7.1183$ 
            & $14.1037$ \slower{1.98} 
            & $\mathbf{6.2149}$ \faster{1.15} \\
        50.0\% 
            & $7.1314$ 
            & $14.1201$ \slower{1.98} 
            & $\mathbf{6.1803}$ \faster{1.15} 
            & $7.1283$ 
            & $14.1173$ \slower{1.98} 
            & $\mathbf{6.1899}$ \faster{1.15} \\
        75.0\% 
            & $7.1385$ 
            & $14.1406$ \slower{1.98} 
            & $\mathbf{6.2071}$ \faster{1.15} 
            & $7.1372$ 
            & $14.1185$ \slower{1.98} 
            & $\mathbf{6.1908}$ \faster{1.15} \\
        90.0\% 
            & $7.1482$ 
            & $14.1186$ \slower{1.98} 
            & $\mathbf{6.1688}$ \faster{1.16} 
            & $7.1450$ 
            & $14.1140$ \slower{1.98} 
            & $\mathbf{6.1662}$ \faster{1.16} \\
        \midrule
        97.5\% 
            & $7.1321$ 
            & $14.1286$ \slower{1.98} 
            & $\mathbf{4.2141}$ \faster{1.69} 
            & $7.1307$ 
            & $14.1123$ \slower{1.98} 
            & $\mathbf{4.2172}$ \faster{1.69} \\
        \midrule
        99.75\% 
            & $7.1181$ 
            & $14.1167$ \slower{1.98} 
            & $\mathbf{2.1714}$ \faster{3.28} 
            & $7.1167$ 
            & $14.1063$ \slower{1.98} 
            & $\mathbf{2.1712}$ \faster{3.28} \\
        \bottomrule
    \end{tabular}
\end{table}

## Optimizer-state and traffic footprint (subset only)

The footprint table reports the static memory and per-step traffic charged to the
synthetic 525M-element subset. \emph{Param} and \emph{Grad} are BF16 buffers
(\(2\,\text{B} \times N\)). \emph{Dense state} is fp32 \(m + v\)
(\(8\,\text{B} \times N\)). \emph{Sparse state} is fp32 \(m + v\) restricted to
active elements only, scaling with the active fraction \(1 - s\). The
\emph{traffic proxy} charges \(112\) bytes per active element, modeling the
SparseAdamW kernel's per-element load/store footprint
(BF16 \(p, g\) read/write, fp32 \(m, v\) read/write, plus block-index metadata).

| Sparsity (\%) | Active fraction | Param (MB) | Grad (MB) | Dense state (MB) | Sparse state (MB) | Traffic proxy (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25  | 0.9975  | 1051 | 1051 | 4203 | 4192 | 58690 |
| 50.0  | 0.5000  | 1051 | 1051 | 4203 | 2101 | 29418 |
| 75.0  | 0.2500  | 1051 | 1051 | 4203 | 1051 | 14710 |
| 90.0  | 0.1000  | 1051 | 1051 | 4203 |  420 |  5885 |
| 97.5  | 0.0250  | 1051 | 1051 | 4203 |  105 |  1471 |
| 99.75 | 0.0025  | 1051 | 1051 | 4203 |   11 |   147 |

At the highest sparsity in the grid (\(99.75\%\)), the SparseAdamW state
footprint shrinks by approximately \(380\times\) relative to fp32 dense AdamW
(\(11\) MB vs.\ \(4203\) MB on this subset), and the per-step traffic proxy
shrinks by approximately \(400\times\) (\(147\) MB vs.\ \(58690\) MB), which is
consistent with the \(\sim 3.28\times\) measured step-time speedup over
\texttt{adamw\_torch} once kernel-launch and synchronization overhead are taken
into account.
