# Reference Table Format (incomplete): 

```latex
\begin{table}[H]
\centering
    \caption{
        BSR AdamW step-time speedup multipliers relative to \texttt{adamw\_torch} (PyTorch)
        dense baseline at each sparsity level and \texttt{adamw\_8bit} (HuggingFace) ~\citep{NEURIPS2019_9015, dettmers2022llmint8}. \textbf{Bold} indicates best result per row, \slower{Red} indicates slowdown multipliers, and \faster{Green} indicates speedup multipliers w.r.t the PyTorch step time. All measurements are in \textit{ms/step}.
    }
    \label{tab:bsr-timing-highlights}
    \small
    \begin{tabular}{l ccc ccc}
        \toprule
        & \multicolumn{3}{c}{\textbf{mean ms / step}} 
        & \multicolumn{3}{c}{\textbf{Memory Footprint}} \\
        \cmidrule(lr){2-4} \cmidrule(lr){5-7}
        \textbf{Sparsity} 
            & \texttt{adamw\_torch} 
            & \texttt{adamw\_8bit} 
            & \texttt{sparse\_adamw} 
            & \texttt{adamw\_torch} 
            & \texttt{adamw\_8bit} 
            & \texttt{sparse\_adamw} \\
        \midrule
        97.5\% 
            & $7.16021$ 
            & $14.1340$ \slower{~2} 
            & $\mathbf{2.57778}$ \faster{~3} 
            & $7.1307$ 
            & $14.1123$ \slower{X} 
            & $\mathbf{4.2172}$ \faster{X} \\
        \bottomrule
    \end{tabular}
\end{table}

```

# Summary Markdowns (individuals): 

(rl_casino) [biggs.s@d1027 rl_casino_optstep_microbench]$ cat 7759907/elem/optimizer_step_microbench.md
# SparseAdamW optimizer.step() microbench

- **mask_label:** `elem`
- **mask_path:** `/scratch/biggs.s/rl_casino_optstep_microbench/7759907/mask/random_elem_meta-llama_Llama-3.1-8B-Instruct_sp97.5pct_seed42.pt`
- **device:** `cuda`  **dtype:** `bf16`
- **lr:** `5e-07`  **block_size:** `32`
- **max_total_numel:** `525000000`  **max_tensors:** `64`  **selection_order:** `model_order`  **cap_behavior:** `break`

## Phase 1 — Speed (no memory instrumentation)

- **steps_total:** `50`  **trim_frac:** `0.1` (excludes first/last 10%)
- **sync_cuda:** `True`

| case | optimizer | tensors | total_numel | active_frac | mean_ms_mid | p50_ms_mid | note |
|---|---|---:|---:|---:|---:|---:|---|
| `dense_elem` | `adamw_torch` | 1 | 525336576 | 0.02499 | 7.16021 | 7.16001 |  |
| `dense8bit_elem` | `adamw_8bit` | 1 | 525336576 | 0.02499 | 14.134 | 14.1079 |  |
| `sparse_elem` | `sparse_adamw` | 1 | 525336576 | 0.02499 | 2.57778 | 2.57558 |  |

### Key speedups (trimmed mean)

- **SparseAdamW vs torch AdamW:** x2.778 faster
- **SparseAdamW vs AdamW 8-bit:** x5.483 faster

## Phase 2 — Memory (measured GPU footprint, isolated from speed phase)

- **bw_ref_steps:** `5` (short timing used only for bandwidth estimate, not the Phase 1 numbers)
- `params_grad_mb`: GPU bytes for params + grads, measured before optimizer is built.
- `mask_infra_mb`: GPU bytes for SparseMaskManager (all-layer bool masks + int64 nonzero indices). **0 for dense optimizers.** This is infrastructure shared across training, not per-step cost.
- `opt_state_mb`: GPU bytes added by the optimizer itself (exp_avg + exp_avg_sq via lazy first-step init). Measured after mask infrastructure, so SMM cost does not inflate this number.
- `peak_scratch_mb`: peak temp allocations above steady-state baseline during one step. Dense AdamW creates a full denom tensor; the Triton kernel is in-place.
- `bw_est_gb_s`: theoretical traffic proxy / bw_ref_mean_ms (dense uses total_numel × 112 B; sparse uses active_numel × 112 B).

| case | optimizer | active_frac | params_grad_MB | mask_infra_MB | opt_state_MB | total_footprint_MB | peak_scratch_MB | bw_est_GB_s | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `dense_elem` | `adamw_torch` | 0.02499 | 2627.73 | 0.00 | 2101.35 | 4729.08 | 1050.67 | 8231.9 |  |
| `dense8bit_elem` | `adamw_8bit` | 0.02499 | 2627.73 | 0.00 | 1069.19 | 3696.92 | 0.00 | 4166.9 |  |
| `sparse_elem` | `sparse_adamw` | 0.02499 | 2627.73 | 9640.24 | 2101.35 | 14369.32 | 0.03 | 570.3 |  |

### Memory profile

- **opt_state vs torch AdamW:** same-size (dense zeros_like buffers)
- **opt_state vs AdamW 8-bit:** x0.51 smaller
- **peak_scratch SparseAdamW vs torch AdamW:** x32064 less (1050.7 MB → 0.03 MB). Triton kernel is in-place; dense AdamW allocates a full denom tensor.

## Memory / traffic estimates (subset only)

- `est_param_bytes` / `est_grad_bytes` use the chosen dtype bytes-per-element.
- AdamW state estimate assumes fp32 `m`+`v` (8 bytes/element).
- Sparse traffic proxy uses 112 bytes per active element (see `src/utils/bsr_theory_metrics.py`).

| case | est_param_MB | est_grad_MB | est_adam_state_MB_dense | est_adam_state_MB_sparse | traffic_proxy_MB |
|---|---:|---:|---:|---:|---:|
| `dense_elem` | 1050.7 | 1050.7 | 4202.7 | 105.0 | 1470.5 |
| `dense8bit_elem` | 1050.7 | 1050.7 | 4202.7 | 105.0 | 1470.5 |
| `sparse_elem` | 1050.7 | 1050.7 | 4202.7 | 105.0 | 1470.5 |
(rl_casino) [biggs.s@d1027 rl_casino_optstep_microbench]$ 

(rl_casino) [biggs.s@d1027 rl_casino_optstep_microbench]$ cat 7759908/elem/optimizer_step_microbench.md
# SparseAdamW optimizer.step() microbench

- **mask_label:** `elem`
- **mask_path:** `/scratch/biggs.s/rl_casino_optstep_microbench/7759908/mask/random_elem_meta-llama_Llama-3.1-8B-Instruct_sp95.0pct_seed42.pt`
- **device:** `cuda`  **dtype:** `bf16`
- **lr:** `5e-07`  **block_size:** `32`
- **max_total_numel:** `525000000`  **max_tensors:** `64`  **selection_order:** `model_order`  **cap_behavior:** `break`

## Phase 1 — Speed (no memory instrumentation)

- **steps_total:** `50`  **trim_frac:** `0.1` (excludes first/last 10%)
- **sync_cuda:** `True`

| case | optimizer | tensors | total_numel | active_frac | mean_ms_mid | p50_ms_mid | note |
|---|---|---:|---:|---:|---:|---:|---|
| `dense_elem` | `adamw_torch` | 1 | 525336576 | 0.04998 | 7.12681 | 7.12721 |  |
| `dense8bit_elem` | `adamw_8bit` | 1 | 525336576 | 0.04998 | 14.1167 | 14.1159 |  |
| `sparse_elem` | `sparse_adamw` | 1 | 525336576 | 0.04998 | 2.851 | 2.85141 |  |

### Key speedups (trimmed mean)

- **SparseAdamW vs torch AdamW:** x2.500 faster
- **SparseAdamW vs AdamW 8-bit:** x4.951 faster

## Phase 2 — Memory (measured GPU footprint, isolated from speed phase)

- **bw_ref_steps:** `5` (short timing used only for bandwidth estimate, not the Phase 1 numbers)
- `params_grad_mb`: GPU bytes for params + grads, measured before optimizer is built.
- `mask_infra_mb`: GPU bytes for SparseMaskManager (all-layer bool masks + int64 nonzero indices). **0 for dense optimizers.** This is infrastructure shared across training, not per-step cost.
- `opt_state_mb`: GPU bytes added by the optimizer itself (exp_avg + exp_avg_sq via lazy first-step init). Measured after mask infrastructure, so SMM cost does not inflate this number.
- `peak_scratch_mb`: peak temp allocations above steady-state baseline during one step. Dense AdamW creates a full denom tensor; the Triton kernel is in-place.
- `bw_est_gb_s`: theoretical traffic proxy / bw_ref_mean_ms (dense uses total_numel × 112 B; sparse uses active_numel × 112 B).

| case | optimizer | active_frac | params_grad_MB | mask_infra_MB | opt_state_MB | total_footprint_MB | peak_scratch_MB | bw_est_GB_s | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `dense_elem` | `adamw_torch` | 0.04998 | 2627.73 | 0.00 | 2101.35 | 4729.08 | 1050.67 | 8256.3 |  |
| `dense8bit_elem` | `adamw_8bit` | 0.04998 | 2627.73 | 0.00 | 1069.19 | 3696.92 | 0.00 | 4165.7 |  |
| `sparse_elem` | `sparse_adamw` | 0.04998 | 2627.73 | 11248.17 | 2101.35 | 15977.24 | 0.03 | 1036.6 |  |

### Memory profile

- **opt_state vs torch AdamW:** same-size (dense zeros_like buffers)
- **opt_state vs AdamW 8-bit:** x0.51 smaller
- **peak_scratch SparseAdamW vs torch AdamW:** x32064 less (1050.7 MB → 0.03 MB). Triton kernel is in-place; dense AdamW allocates a full denom tensor.

## Memory / traffic estimates (subset only)

- `est_param_bytes` / `est_grad_bytes` use the chosen dtype bytes-per-element.
- AdamW state estimate assumes fp32 `m`+`v` (8 bytes/element).
- Sparse traffic proxy uses 112 bytes per active element (see `src/utils/bsr_theory_metrics.py`).

| case | est_param_MB | est_grad_MB | est_adam_state_MB_dense | est_adam_state_MB_sparse | traffic_proxy_MB |
|---|---:|---:|---:|---:|---:|
| `dense_elem` | 1050.7 | 1050.7 | 4202.7 | 210.0 | 2940.5 |
| `dense8bit_elem` | 1050.7 | 1050.7 | 4202.7 | 210.0 | 2940.5 |
| `sparse_elem` | 1050.7 | 1050.7 | 4202.7 | 210.0 | 2940.5 |
(rl_casino) [biggs.s@d1027 rl_casino_optstep_microbench]$ 

(rl_casino) [biggs.s@d1027 rl_casino_optstep_microbench]$ cat 7759909/elem/optimizer_step_microbench.md
# SparseAdamW optimizer.step() microbench

- **mask_label:** `elem`
- **mask_path:** `/scratch/biggs.s/rl_casino_optstep_microbench/7759909/mask/random_elem_meta-llama_Llama-3.1-8B-Instruct_sp90.0pct_seed42.pt`
- **device:** `cuda`  **dtype:** `bf16`
- **lr:** `5e-07`  **block_size:** `32`
- **max_total_numel:** `525000000`  **max_tensors:** `64`  **selection_order:** `model_order`  **cap_behavior:** `break`

## Phase 1 — Speed (no memory instrumentation)

- **steps_total:** `50`  **trim_frac:** `0.1` (excludes first/last 10%)
- **sync_cuda:** `True`

| case | optimizer | tensors | total_numel | active_frac | mean_ms_mid | p50_ms_mid | note |
|---|---|---:|---:|---:|---:|---:|---|
| `dense_elem` | `adamw_torch` | 1 | 525336576 | 0.09998 | 7.15236 | 7.15178 |  |
| `dense8bit_elem` | `adamw_8bit` | 1 | 525336576 | 0.09998 | 14.1284 | 14.1097 |  |
| `sparse_elem` | `sparse_adamw` | 1 | 525336576 | 0.09998 | 3.32087 | 3.32016 |  |

### Key speedups (trimmed mean)

- **SparseAdamW vs torch AdamW:** x2.154 faster
- **SparseAdamW vs AdamW 8-bit:** x4.254 faster

## Phase 2 — Memory (measured GPU footprint, isolated from speed phase)

- **bw_ref_steps:** `5` (short timing used only for bandwidth estimate, not the Phase 1 numbers)
- `params_grad_mb`: GPU bytes for params + grads, measured before optimizer is built.
- `mask_infra_mb`: GPU bytes for SparseMaskManager (all-layer bool masks + int64 nonzero indices). **0 for dense optimizers.** This is infrastructure shared across training, not per-step cost.
- `opt_state_mb`: GPU bytes added by the optimizer itself (exp_avg + exp_avg_sq via lazy first-step init). Measured after mask infrastructure, so SMM cost does not inflate this number.
- `peak_scratch_mb`: peak temp allocations above steady-state baseline during one step. Dense AdamW creates a full denom tensor; the Triton kernel is in-place.
- `bw_est_gb_s`: theoretical traffic proxy / bw_ref_mean_ms (dense uses total_numel × 112 B; sparse uses active_numel × 112 B).

| case | optimizer | active_frac | params_grad_MB | mask_infra_MB | opt_state_MB | total_footprint_MB | peak_scratch_MB | bw_est_GB_s | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `dense_elem` | `adamw_torch` | 0.09998 | 2627.73 | 0.00 | 2101.35 | 4729.08 | 1050.67 | 8223.8 |  |
| `dense8bit_elem` | `adamw_8bit` | 0.09998 | 2627.73 | 0.00 | 1069.19 | 3696.92 | 0.00 | 4165.7 |  |
| `sparse_elem` | `sparse_adamw` | 0.09998 | 2627.73 | 14457.83 | 2101.35 | 19186.91 | 0.03 | 1758.1 |  |

### Memory profile

- **opt_state vs torch AdamW:** same-size (dense zeros_like buffers)
- **opt_state vs AdamW 8-bit:** x0.51 smaller
- **peak_scratch SparseAdamW vs torch AdamW:** x32064 less (1050.7 MB → 0.03 MB). Triton kernel is in-place; dense AdamW allocates a full denom tensor.

## Memory / traffic estimates (subset only)

- `est_param_bytes` / `est_grad_bytes` use the chosen dtype bytes-per-element.
- AdamW state estimate assumes fp32 `m`+`v` (8 bytes/element).
- Sparse traffic proxy uses 112 bytes per active element (see `src/utils/bsr_theory_metrics.py`).

| case | est_param_MB | est_grad_MB | est_adam_state_MB_dense | est_adam_state_MB_sparse | traffic_proxy_MB |
|---|---:|---:|---:|---:|---:|
| `dense_elem` | 1050.7 | 1050.7 | 4202.7 | 420.2 | 5882.9 |
| `dense8bit_elem` | 1050.7 | 1050.7 | 4202.7 | 420.2 | 5882.9 |
| `sparse_elem` | 1050.7 | 1050.7 | 4202.7 | 420.2 | 5882.9 |
(rl_casino) [biggs.s@d1027 rl_casino_optstep_microbench]$ 
