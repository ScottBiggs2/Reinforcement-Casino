# SparseAdamW optimizer.step() microbench

- **mask_label:** `elem`
- **mask_path:** `/scratch/biggs.s/rl_casino_h200_bsr/random_mask_blob/masks/s0p25_element_b16_mlp0_floor0.0025.pt`
- **device:** `cuda`  **dtype:** `bf16`
- **lr:** `5e-07`  **block_size:** `32`
- **steps_total:** `50`  **trim_frac:** `0.1` (excludes first/last 10%)
- **sync_cuda:** `True`
- **max_total_numel:** `525000000`  **max_tensors:** `64`  **selection_order:** `model_order`  **cap_behavior:** `break`

## Timing summary (trimmed mid-window)

| case | optimizer | tensors | total_numel | active_frac | mean_ms_mid | p50_ms_mid | note |
|---|---|---:|---:|---:|---:|---:|---|
| `dense_elem` | `adamw_torch` | 1 | 525336576 | 0.9975 | 7.12002 | 7.11832 |  |
| `dense8bit_elem` | `adamw_8bit` | 1 | 525336576 | 0.9975 | 14.1088 | 14.1037 |  |
| `sparse_elem` | `sparse_adamw` | 1 | 525336576 | 0.9975 | 6.20827 | 6.21492 |  |

## Key speedups (trimmed mean)

- **SparseAdamW vs torch AdamW:** x1.147 faster (lower is better).
- **SparseAdamW vs AdamW 8-bit:** x2.273 faster (lower is better).

## Memory / traffic estimates (subset only)

- `est_param_bytes` / `est_grad_bytes` use the chosen dtype bytes-per-element.
- AdamW state estimate assumes fp32 `m`+`v` (8 bytes/element).
- Sparse traffic proxy uses 112 bytes per active element (see `src/utils/bsr_theory_metrics.py`).

| case | est_param_MB | est_grad_MB | est_adam_state_MB_dense | est_adam_state_MB_sparse | traffic_proxy_MB |
|---|---:|---:|---:|---:|---:|
| `dense_elem` | 1050.7 | 1050.7 | 4202.7 | 4192.2 | 58690.4 |
| `dense8bit_elem` | 1050.7 | 1050.7 | 4202.7 | 4192.2 | 58690.4 |
| `sparse_elem` | 1050.7 | 1050.7 | 4202.7 | 4192.2 | 58690.4 |

# SparseAdamW optimizer.step() microbench

- **mask_label:** `elem`
- **mask_path:** `/scratch/biggs.s/rl_casino_h200_bsr/random_mask_blob/masks/s50p0_element_b16_mlp0_floor0.0025.pt`
- **device:** `cuda`  **dtype:** `bf16`
- **lr:** `5e-07`  **block_size:** `32`
- **steps_total:** `50`  **trim_frac:** `0.1` (excludes first/last 10%)
- **sync_cuda:** `True`
- **max_total_numel:** `525000000`  **max_tensors:** `64`  **selection_order:** `model_order`  **cap_behavior:** `break`

## Timing summary (trimmed mid-window)

| case | optimizer | tensors | total_numel | active_frac | mean_ms_mid | p50_ms_mid | note |
|---|---|---:|---:|---:|---:|---:|---|
| `dense_elem` | `adamw_torch` | 1 | 525336576 | 0.5 | 7.13145 | 7.1283 |  |
| `dense8bit_elem` | `adamw_8bit` | 1 | 525336576 | 0.5 | 14.1201 | 14.1173 |  |
| `sparse_elem` | `sparse_adamw` | 1 | 525336576 | 0.5 | 6.18028 | 6.18989 |  |

## Key speedups (trimmed mean)

- **SparseAdamW vs torch AdamW:** x1.154 faster (lower is better).
- **SparseAdamW vs AdamW 8-bit:** x2.285 faster (lower is better).

## Memory / traffic estimates (subset only)

- `est_param_bytes` / `est_grad_bytes` use the chosen dtype bytes-per-element.
- AdamW state estimate assumes fp32 `m`+`v` (8 bytes/element).
- Sparse traffic proxy uses 112 bytes per active element (see `src/utils/bsr_theory_metrics.py`).

| case | est_param_MB | est_grad_MB | est_adam_state_MB_dense | est_adam_state_MB_sparse | traffic_proxy_MB |
|---|---:|---:|---:|---:|---:|
| `dense_elem` | 1050.7 | 1050.7 | 4202.7 | 2101.3 | 29418.1 |
| `dense8bit_elem` | 1050.7 | 1050.7 | 4202.7 | 2101.3 | 29418.1 |
| `sparse_elem` | 1050.7 | 1050.7 | 4202.7 | 2101.3 | 29418.1 |

# SparseAdamW optimizer.step() microbench

- **mask_label:** `elem`
- **mask_path:** `/scratch/biggs.s/rl_casino_h200_bsr/random_mask_blob/masks/s75p0_element_b16_mlp0_floor0.0025.pt`
- **device:** `cuda`  **dtype:** `bf16`
- **lr:** `5e-07`  **block_size:** `32`
- **steps_total:** `50`  **trim_frac:** `0.1` (excludes first/last 10%)
- **sync_cuda:** `True`
- **max_total_numel:** `525000000`  **max_tensors:** `64`  **selection_order:** `model_order`  **cap_behavior:** `break`

## Timing summary (trimmed mid-window)

| case | optimizer | tensors | total_numel | active_frac | mean_ms_mid | p50_ms_mid | note |
|---|---|---:|---:|---:|---:|---:|---|
| `dense_elem` | `adamw_torch` | 1 | 525336576 | 0.25 | 7.13851 | 7.13721 |  |
| `dense8bit_elem` | `adamw_8bit` | 1 | 525336576 | 0.25 | 14.1406 | 14.1185 |  |
| `sparse_elem` | `sparse_adamw` | 1 | 525336576 | 0.25 | 6.20708 | 6.19083 |  |

## Key speedups (trimmed mean)

- **SparseAdamW vs torch AdamW:** x1.150 faster (lower is better).
- **SparseAdamW vs AdamW 8-bit:** x2.278 faster (lower is better).

## Memory / traffic estimates (subset only)

- `est_param_bytes` / `est_grad_bytes` use the chosen dtype bytes-per-element.
- AdamW state estimate assumes fp32 `m`+`v` (8 bytes/element).
- Sparse traffic proxy uses 112 bytes per active element (see `src/utils/bsr_theory_metrics.py`).

| case | est_param_MB | est_grad_MB | est_adam_state_MB_dense | est_adam_state_MB_sparse | traffic_proxy_MB |
|---|---:|---:|---:|---:|---:|
| `dense_elem` | 1050.7 | 1050.7 | 4202.7 | 1050.7 | 14709.6 |
| `dense8bit_elem` | 1050.7 | 1050.7 | 4202.7 | 1050.7 | 14709.6 |
| `sparse_elem` | 1050.7 | 1050.7 | 4202.7 | 1050.7 | 14709.6 |

# SparseAdamW optimizer.step() microbench

- **mask_label:** `elem`
- **mask_path:** `/scratch/biggs.s/rl_casino_h200_bsr/random_mask_blob/masks/s90p0_element_b16_mlp0_floor0.0025.pt`
- **device:** `cuda`  **dtype:** `bf16`
- **lr:** `5e-07`  **block_size:** `32`
- **steps_total:** `50`  **trim_frac:** `0.1` (excludes first/last 10%)
- **sync_cuda:** `True`
- **max_total_numel:** `525000000`  **max_tensors:** `64`  **selection_order:** `model_order`  **cap_behavior:** `break`

## Timing summary (trimmed mid-window)

| case | optimizer | tensors | total_numel | active_frac | mean_ms_mid | p50_ms_mid | note |
|---|---|---:|---:|---:|---:|---:|---|
| `dense_elem` | `adamw_torch` | 1 | 525336576 | 0.1 | 7.14816 | 7.14505 |  |
| `dense8bit_elem` | `adamw_8bit` | 1 | 525336576 | 0.1 | 14.1186 | 14.114 |  |
| `sparse_elem` | `sparse_adamw` | 1 | 525336576 | 0.1 | 6.16879 | 6.16622 |  |

## Key speedups (trimmed mean)

- **SparseAdamW vs torch AdamW:** x1.159 faster (lower is better).
- **SparseAdamW vs AdamW 8-bit:** x2.289 faster (lower is better).

## Memory / traffic estimates (subset only)

- `est_param_bytes` / `est_grad_bytes` use the chosen dtype bytes-per-element.
- AdamW state estimate assumes fp32 `m`+`v` (8 bytes/element).
- Sparse traffic proxy uses 112 bytes per active element (see `src/utils/bsr_theory_metrics.py`).

| case | est_param_MB | est_grad_MB | est_adam_state_MB_dense | est_adam_state_MB_sparse | traffic_proxy_MB |
|---|---:|---:|---:|---:|---:|
| `dense_elem` | 1050.7 | 1050.7 | 4202.7 | 420.4 | 5885.0 |
| `dense8bit_elem` | 1050.7 | 1050.7 | 4202.7 | 420.4 | 5885.0 |
| `sparse_elem` | 1050.7 | 1050.7 | 4202.7 | 420.4 | 5885.0 |

# SparseAdamW optimizer.step() microbench

- **mask_label:** `elem`
- **mask_path:** `/scratch/biggs.s/rl_casino_masks/orch_lr1_grasp6_6376972/random_elem_meta-llama_Llama-3.1-8B-Instruct_light_r1_sp97.5pct_seed42.pt`
- **device:** `cuda`  **dtype:** `bf16`
- **lr:** `5e-07`  **block_size:** `32`
- **steps_total:** `50`  **trim_frac:** `0.1` (excludes first/last 10%)
- **sync_cuda:** `True`
- **max_total_numel:** `525000000`  **max_tensors:** `64`  **selection_order:** `model_order`  **cap_behavior:** `break`

## Timing summary (trimmed mid-window)

| case | optimizer | tensors | total_numel | active_frac | mean_ms_mid | p50_ms_mid | note |
|---|---|---:|---:|---:|---:|---:|---|
| `dense_elem` | `adamw_torch` | 1 | 525336576 | 0.02499 | 7.13214 | 7.13068 |  |
| `dense8bit_elem` | `adamw_8bit` | 1 | 525336576 | 0.02499 | 14.1286 | 14.1123 |  |
| `sparse_elem` | `sparse_adamw` | 1 | 525336576 | 0.02499 | 4.21413 | 4.21716 |  |

## Key speedups (trimmed mean)

- **SparseAdamW vs torch AdamW:** x1.692 faster (lower is better).
- **SparseAdamW vs AdamW 8-bit:** x3.353 faster (lower is better).

## Memory / traffic estimates (subset only)

- `est_param_bytes` / `est_grad_bytes` use the chosen dtype bytes-per-element.
- AdamW state estimate assumes fp32 `m`+`v` (8 bytes/element).
- Sparse traffic proxy uses 112 bytes per active element (see `src/utils/bsr_theory_metrics.py`).

| case | est_param_MB | est_grad_MB | est_adam_state_MB_dense | est_adam_state_MB_sparse | traffic_proxy_MB |
|---|---:|---:|---:|---:|---:|
| `dense_elem` | 1050.7 | 1050.7 | 4202.7 | 105.0 | 1470.5 |
| `dense8bit_elem` | 1050.7 | 1050.7 | 4202.7 | 105.0 | 1470.5 |
| `sparse_elem` | 1050.7 | 1050.7 | 4202.7 | 105.0 | 1470.5 |

# SparseAdamW optimizer.step() microbench

- **mask_label:** `elem`
- **mask_path:** `/scratch/biggs.s/rl_casino_h200_bsr/random_mask_blob/masks/s99p75_element_b16_mlp0_floor0.0025.pt`
- **device:** `cuda`  **dtype:** `bf16`
- **lr:** `5e-07`  **block_size:** `32`
- **steps_total:** `50`  **trim_frac:** `0.1` (excludes first/last 10%)
- **sync_cuda:** `True`
- **max_total_numel:** `525000000`  **max_tensors:** `64`  **selection_order:** `model_order`  **cap_behavior:** `break`

## Timing summary (trimmed mid-window)

| case | optimizer | tensors | total_numel | active_frac | mean_ms_mid | p50_ms_mid | note |
|---|---|---:|---:|---:|---:|---:|---|
| `dense_elem` | `adamw_torch` | 1 | 525336576 | 0.0025 | 7.11813 | 7.11673 |  |
| `dense8bit_elem` | `adamw_8bit` | 1 | 525336576 | 0.0025 | 14.1167 | 14.1063 |  |
| `sparse_elem` | `sparse_adamw` | 1 | 525336576 | 0.0025 | 2.17141 | 2.17123 |  |

## Key speedups (trimmed mean)

- **SparseAdamW vs torch AdamW:** x3.278 faster (lower is better).
- **SparseAdamW vs AdamW 8-bit:** x6.501 faster (lower is better).

## Memory / traffic estimates (subset only)

- `est_param_bytes` / `est_grad_bytes` use the chosen dtype bytes-per-element.
- AdamW state estimate assumes fp32 `m`+`v` (8 bytes/element).
- Sparse traffic proxy uses 112 bytes per active element (see `src/utils/bsr_theory_metrics.py`).

| case | est_param_MB | est_grad_MB | est_adam_state_MB_dense | est_adam_state_MB_sparse | traffic_proxy_MB |
|---|---:|---:|---:|---:|---:|
| `dense_elem` | 1050.7 | 1050.7 | 4202.7 | 10.5 | 147.1 |
| `dense8bit_elem` | 1050.7 | 1050.7 | 4202.7 | 10.5 | 147.1 |
| `sparse_elem` | 1050.7 | 1050.7 | 4202.7 | 10.5 | 147.1 |

### Fill in the table: 

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