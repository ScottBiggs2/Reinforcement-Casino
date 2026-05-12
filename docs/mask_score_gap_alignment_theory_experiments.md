# Mask score-gap: alignment-optimal scoring (theory confirmation experiments)

This document describes how we **estimate the appendix-optimal scoring function** \(s^*(\theta_i, B)\), run the **fixed-point iteration** for \(B^*\), and emit **sidecar** artifacts that extend mask-score-gap plots **without overwriting** the original analysis outputs.

It complements the parallel Slurm pipeline described in [`scripts/submit_mask_score_gap_parallel_light_r1.sh`](../scripts/submit_mask_score_gap_parallel_light_r1.sh) and the core analysis in [`src/analysis/mask_score_gap_analysis.py`](../src/analysis/mask_score_gap_analysis.py).

---

## What we are testing (short theory recap)

From the appendix (optimal scoring / performance retention):

- **Per-coordinate score** (with Hessian diagonal \(H_{ii}\), remainder bound \(C\), gradient bound \(\epsilon\), and \(\ell_1\) mass \(B\) on pruned weights):

  \[
  s^*(\theta_i, B) = \frac{H_{ii}}{2}\theta_i^2 + (CB + \epsilon)\,|\theta_i|
  \]

- **Keep mask** at sparsity \(\rho\): keep the top \((1-\rho)\) fraction of coordinates by \(s^*(\cdot, B)\) (same global keep count convention as certifiability: `keep_percent = 100 - sparsity_percent`).

- **Fixed point**:

  \[
  B^{(t+1)} = \sum_{i \notin \mathcal{M}^{(t)}} |\theta_i|
  \]

  where \(\mathcal{M}^{(t)}\) is the keep set at iteration \(t\). Stop when \(|B^{(t+1)} - B^{(t)}| < \eta\) or `max_iter`.

**Implementation note:** we use **empirical Fisher diagonal** \(F_{ii} \approx \mathbb{E}[g_i^2]\) as a **proxy for \(H_{ii}\)**. Caption results accordingly in any paper figure.

---

## What gets reused from prior mask-score-gap runs

| Artifact | Role |
|----------|------|
| `OUT_DIR/mask_score_gap_run.json` | Source of `initial_model`, `final_model`, `delta_log_dir`, `magnitude_milestones`, `sparsity_percent`, `histogram_bins`, etc. |
| `OUT_DIR/magnitude_caches/mag_aggregate_step_*.pt` | Warm-start magnitude scores per milestone (no delta re-stream). |
| `OUT_DIR/mask_score_gap_histograms.npz` (unchanged) | Original milestone / random / oracle gap histograms for baseline plots. |

**New outputs only** (sidecars; never replace the core files above):

| File | Contents |
|------|----------|
| `OUT_DIR/mask_score_gap_alignment_gap_diagnostics.json` | `B_trace`, `B_star`, `tau_star`, `C`, `epsilon_grad`, `eta`, convergence flags, key counts, etc. |
| `OUT_DIR/mask_score_gap_alignment_histograms.npz` | Log-binned histograms for \(\|s^* - s_{\mathrm{oracle}}\|\), \(\|s^* - s_{\mathrm{rand}}\|\), \(\|s^* - s_{\mathrm{mag},t}\|\) per milestone. |

---

## Code map

| Component | Path |
|-----------|------|
| Fixed-point + streaming top-\(k\) \(\tau\) for \(s^*\) | [`src/analysis/alignment_optimal_scores.py`](../src/analysis/alignment_optimal_scores.py) |
| Fisher diagonal export (raw, **no** per-layer z-score) | [`scripts/export_alignment_fisher_diag.py`](../scripts/export_alignment_fisher_diag.py) |
| Fisher accumulation on **CPU** (avoids GPU OOM for large models) | [`src/cold_start/cold_mask_finder.py`](../src/cold_start/cold_mask_finder.py) — `compute_fisher_scores` |
| Sidecar generator (loads checkpoints + Fisher + caches) | [`scripts/compute_mask_score_gap_alignment_reference.py`](../scripts/compute_mask_score_gap_alignment_reference.py) |
| Plots with optional alignment overlay | [`scripts/report_mask_score_gap_plots.py`](../scripts/report_mask_score_gap_plots.py) — auto-loads `mask_score_gap_alignment_histograms.npz` if present |
| GPU Fisher one-off (multigpu + 1 GPU) | [`scripts/slurm_export_alignment_fisher_diag_oneoff.slurm`](../scripts/slurm_export_alignment_fisher_diag_oneoff.slurm) |
| CPU alignment one-off (short partition, high RAM) | [`scripts/slurm_compute_mask_score_gap_alignment_reference_oneoff.slurm`](../scripts/slurm_compute_mask_score_gap_alignment_reference_oneoff.slurm) |
| Unit test (toy fixed point) | [`src/tests/test_alignment_optimal_scores.py`](../src/tests/test_alignment_optimal_scores.py) |

Both Python entrypoints insert the repo root into `sys.path` so `import src...` works from Slurm `sbatch` and arbitrary working directories.

---

## Step 1 — Export Fisher diagonal (GPU job)

**Why GPU:** forward + backward through the causal LM for empirical Fisher.

**Why CPU buffers:** Fisher accumulators are **`float32` on CPU** so a 32 GiB GPU is not asked to hold a second full-sized fp32 copy of every parameter on device.

**Practical constraint:** Llama‑3.1‑8B at long context on a **small GPU** can still OOM during attention (activations), not Fisher buffers. Use an **A100 / H200** (or reduce `max_length` / batch) if needed.

### Slurm (recommended): multigpu + A100

From repo root:

```bash
export CKPT500=/scratch/$USER/rl_casino_train/dpo5k_dense_light-r1/checkpoints/meta_llama_llama_3_1_8b_instruct_light_r1/checkpoint-500
export FISHER_OUT=/scratch/$USER/rl_casino_analysis/alignment/fisher_diag_raw_ckpt500_a100.pt

export FISHER_SAMPLES=256
export FISHER_MAX_LEN=512
export FISHER_MINI_BS=4
export FISHER_MLP_ONLY=1   # optional but strongly recommended for memory

sbatch \
  --partition=multigpu \
  --gres=gpu:A100:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=02:00:00 \
  scripts/slurm_export_alignment_fisher_diag_oneoff.slurm
```

**Outputs:**

- `FISHER_OUT` — `torch.save` dict `name -> tensor` (fp32), **raw** diagonal (no z-score).
- `${FISHER_OUT%.pt}.json` — metadata (dataset, sample counts, flags).

**Explorer-specific:** if `--gres=gpu:A100:1` is rejected, use your site’s constraint syntax (e.g. `--constraint=a100`) while keeping `partition=multigpu`.

---

## Step 2 — Compute alignment sidecars (CPU job)

This loads **both** full checkpoints on CPU (bf16 by default), intersects parameter names with the Fisher dict, runs the fixed-point loop, then walks tensors again to fill histograms.

**Do not run on the login node** for full models: the process can be **SIGKILL**’d for RSS. Use the short-partition wrapper:

```bash
export OUT_DIR=/scratch/$USER/rl_casino_analysis/mask_score_gap_parallel/run2_parallel
export FISHER=/scratch/$USER/rl_casino_analysis/alignment/fisher_diag_raw_ckpt500_a100.pt
export ALIGN_C=1.0
export ALIGN_EPS_GRAD=0.0

sbatch scripts/slurm_compute_mask_score_gap_alignment_reference_oneoff.slurm
```

**Optional env overrides** (see the Slurm file):

- `ALIGN_ETA` (default `1e-6`)
- `ALIGN_MAX_ITER` (default `25`)
- `HISTOGRAM_BINS` (default `1024`)
- `ALIGN_MAX_KEYS` — smoke subset

**Coverage caveat:** if Fisher was computed with **`FISHER_MLP_ONLY=1`**, only tensors present in the Fisher dict participate in \(s^*\) and gap histograms (typically **MLP projections**). For a **full-parameter** alignment reference, export Fisher for **all** scored parameters (and ensure RAM/time on the alignment job).

---

## Step 3 — Regenerate plots (login node or short CPU)

```bash
cd /path/to/repo
MPLBACKEND=Agg python scripts/report_mask_score_gap_plots.py --analysis-dir "$OUT_DIR"
```

If `mask_score_gap_alignment_histograms.npz` exists, the script extends the **raw** density + ECDF panel with an **alignment** series (label: \(|s^* - \text{oracle}|\)).

---

## Parameter reference

| Symbol | CLI / env | Meaning |
|--------|-----------|--------|
| \(H_{ii}\) | `--fisher-diag` | Empirical Fisher diagonal proxy (fp32 tensors). |
| \(C\) | `--C` / `ALIGN_C` | Remainder / curvature bound constant in \((CB+\epsilon)|\theta|\). |
| \(\epsilon\) | `--epsilon-grad` / `ALIGN_EPS_GRAD` | **Gradient bound** \(\|\nabla \mathcal{L}\|\le \epsilon\) from the appendix (not machine epsilon). |
| \(\eta\) | `--eta` / `ALIGN_ETA` | Convergence tolerance on \(B\). |
| \(\rho\) | from `mask_score_gap_run.json` | Same `sparsity_percent` as the mask-score-gap run. |
| \(\theta_i\) | `final_model` weights | Same checkpoint as the analysis run. |

---

## Troubleshooting

| Symptom | Likely cause | Mitigation |
|---------|--------------|------------|
| Fisher job CUDA OOM on first forward | Model + activations + 512 tokens on 32 GiB | A100/H200; or lower `FISHER_MAX_LEN` / `FISHER_MINI_BS`; or `FISHER_MLP_ONLY=1` |
| `ModuleNotFoundError: src` | Old script without `sys.path` fix | `git pull` to current `main`/branch |
| Alignment job `Killed` on login node | RSS (two 8B loads + histograms) | Use `slurm_compute_mask_score_gap_alignment_reference_oneoff.slurm` |
| Few “Eligible keys” vs 291 | Fisher dict subset (e.g. MLP-only) | Full Fisher export for full-model \(s^*\) |

---

## Copy-out for paper / home

Use [`scripts/copy_mask_score_gap_results_to_home.sh`](../scripts/copy_mask_score_gap_results_to_home.sh); it copies CSV/JSON/NPZ and small `parallel_shards/` files. Alignment sidecars are **small** and will be included in the top-level globs if present.

---

## Related documentation

- Mask score-gap Slurm (monolithic): [`scripts/slurm_mask_score_gap_light_r1.slurm`](../scripts/slurm_mask_score_gap_light_r1.slurm)
- Parallel DAG: [`scripts/submit_mask_score_gap_parallel_light_r1.sh`](../scripts/submit_mask_score_gap_parallel_light_r1.sh)
- Status helper: [`scripts/check_mask_score_gap_status.sh`](../scripts/check_mask_score_gap_status.sh)
