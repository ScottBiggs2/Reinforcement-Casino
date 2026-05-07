---
name: Alignment optimal scores + plots
overview: Implement fixed-point optimal scores s*(θ,B)=(H_ii/2)θ²+(CB+ε)|θ| with H_ii≈Fisher diagonal; emit sidecar artifacts and extend ECDF/cert-style plots without overwriting existing mask-score-gap outputs; add paper styling and optional KDE density curves.
todos:
  - id: core-phi
    content: "alignment_optimal_scores.py: s_star, Φ iteration, global top-(1-ρ), optional tie-break; unit tests"
    status: pending
  - id: fisher-pipeline
    content: "Export/load Fisher diagonal as H_ii proxy (raw diagonal; avoid z-score for absolute s* scale); CLI path"
    status: pending
  - id: sidecar-script
    content: "compute_mask_score_gap_alignment_reference.py → alignment_*.json/npz; gaps vs oracle/mag/random"
    status: pending
  - id: plots-overlay
    content: "report_mask_score_gap_plots: alignment overlays, KDE smoothed density, --paper-style / figures_paper/"
    status: pending
  - id: certifiability-bridge
    content: "Optional diagnostics/plots linking |s-s*(B*)| to existing margin τ framework (theorem certifiability)"
    status: pending
  - id: ops-docs
    content: "GPU Fisher job → CPU alignment; document C,ε,η; no overwrite of mask_score_gap_* core files"
    status: pending
isProject: false
---

# Alignment optimal scores + reference plots (updated with full appendix)

## Theoretical framework (what we implement)

### Performance retention (context)

Second-order expansion with \(H(\xi)=D_S+R_S(\xi)\) leads to a bound involving \(\sum_{i\notin\mathcal{M}} s_i\) plus remainder controlled by \(\|R_S\|_2\le C\) and gradient magnitude \(\|\nabla\mathcal{L}(\theta)\|\le \epsilon\). The \(\ell_1\) choice for \(\|\delta\|\) in the bound yields **diagonal curvature weight \(\kappa(\theta_i)=H_{ii}\)** in the diagonal approximation term.

**Implementation:** We do **not** re-implement the full bound estimator here; we implement the **optimal scoring / fixed-point** construction from Appendix optimal scoring, using **Fisher diagonal as \(H_{ii}\)**.

### Optimal scoring function (core formula)

**Theorem (Appendix optimal scoring, fixed point):** There exists \(B^*\) with \(\Phi(B^*)=B^*\), where

\[
\Phi(B)=\sum_{i\notin \mathcal{M}(B)}|\theta_i|
\]

and \(\mathcal{M}(B)\) is the **mask that keeps** the \((1-\rho)\) fraction of coordinates with **largest** scores \(s^*(\theta_i,B)\) (equivalently: **prune** the \(\rho\) fraction with **smallest** \(s^*\), minimizing \(\sum_{i\notin\mathcal{M}} s^*(\theta_i,B)\) at fixed \(|\mathcal{M}|=(1-\rho)|\theta|\)).

**Scoring function (coordinate-wise):**

\[
s^*(\theta_i,B)=\frac{H_{ii}}{2}\theta_i^2 + (CB+\epsilon)|\theta_i|
\]

- \(H_{ii}\): Hessian diagonal (implementation: **Fisher diagonal proxy**, see below).
- \(C\): spectral-norm style bound on remainder \(R_S=H-D_S\) from the performance-retention derivation (\(\|R_S\|_2\le C\)).
- \(\epsilon\): **gradient bound** \(\|\nabla\mathcal{L}(\theta)\|\le \epsilon\) from the same bound (not floating-point machine epsilon — expose as `--epsilon_grad` or env `ALIGN_EPSILON_GRAD` in code/docs).
- \(B\): current iterate for \(\|\delta\|_1\) mass on the **pruned** coordinates; enters only via \((CB+\epsilon)|\theta_i|\).
- \(\theta_i\): scalar parameter value at the evaluation point \(\theta\) (default: **final checkpoint** weights, same `final_model` as mask-score-gap).

Uniqueness of \(B^*\) is under the **magnitude–curvature alignment** condition (paper cites Fernández et al.); we only assume fast empirical convergence per appendix.

**Iterative algorithm (implementation target):** \(B^{(0)}=0\). For \(t=0,1,\ldots\): (1) form \(s^*(\theta_i,B^{(t)})\) for all coordinates **in fixed key order** (same iteration order as [`process_keys`](src/analysis/mask_score_gap_analysis.py)); (2) build \(\mathcal{M}^{(t)}\) = global **keep** set = top-\(k\) scores with \(k=\) `global_keep_count(N, sparsity_percent)`; (3) \(B^{(t+1)}=\sum_{i\notin\mathcal{M}^{(t)}}|\theta_i|\). Stop when \(|B^{(t+1)}-B^{(t)}|<\eta\) (`--eta` / convergence tol) or `max_iter`.

**Mask semantics vs hybrid:** **v1 = pure global** top-\(k\) by \(s^*\) (no per-layer floors). Matches the theorem statement the user quoted. Optional **v2**: hybrid floors analogous to `cert_tau_rule=hybrid_global_phase` only if the write-up extends the fixed-point map accordingly.

**Tie-breaking:** Optional same recipe as mask/cert pipeline (Gaussian noise scale \(\propto 10^{-6}\max|s^*|\), seed 42) when many scores tie at the \(k\)-th rank — keeps discrete stability consistent with [`mask_utils`](src/utils/mask_utils.py).

### Certifiability appendix (plot/diagnostic bridge)

Theorem certifiability: practical \(s(\theta_i)\) matches optimal mask if controlled by **deviation** \(|s(\theta_i)-s^*(\theta_i,B^*)|\) and **margin** \(m_i(s)=|s(\theta_i)-\tau_\rho(s)|\).

**Plan:** After estimating \(s^*\) at converged \(B^*\) (or final iterate):

- Emit histograms / ECDFs for **score gaps** \(|s_{\mathrm{oracle}}-s^*|\), \(|s_{\mathrm{mag},t}-s^*|\), \(|s_{\mathrm{rand}}-s^*|\) (already planned).
- **Optional v1.5:** scatter or binned summaries relating \(|s-s^*|\) vs \(m_i(s)\) for oracle/magnitude/random — connects directly to paper’s certifiability narrative without replacing existing \(\tau\) cert rows.

---

## Hessian proxy: Fisher diagonal

Appendix instructs using the diagonal curvature weight \(H_{ii}\); **implementation uses empirical Fisher**

\[
F_{ii} \approx \mathbb{E}\big[g_i^2\big]
\]

(per coordinate \(i\), same shape as \(\theta\)) as **\(H_{ii}\)** in \(s^*\). Existing codebase pointer:

- [`src/cold_start/cold_mask_finder.py`](src/cold_start/cold_mask_finder.py) — `compute_fisher_scores` and related accumulation.

**Critical:** Any per-layer **z-score or variance normalization** of Fisher scores **destroys absolute scale** required for \(\frac{H_{ii}}{2}\theta_i^2\) and for comparing \(s^*\) to oracle/magnitude scores. Alignment pipeline must consume **raw squared-gradient diagonal** aligned to parameter names (same keys/order as mask-score-gap). Add `--no_layer_normalize`, a dedicated **`export_alignment_fisher_diag.py`**, or a flag on an existing Fisher job; record `"fisher_normalized": false` and batch/token metadata in sidecar JSON.

**Operational split:** Full-model Fisher is **data- + GPU-heavy**; run as a **separate Slurm job** that writes `alignment_fisher_diag.pt` (dict name→tensor, FP32 CPU). CPU-only alignment fixed-point reads Fisher + `final_model` shards — **no recomputation** of magnitude caches.

---

## Reuse vs new compute

| Input | Source |
|--------|--------|
| \(\theta\) | Same **`final_model`** state dict as mask-score-gap (for \(|\theta_i|\) and \(\theta_i^2\) terms). |
| Oracle scores | \(|w_{\mathrm{final}}-w_{\mathrm{initial}}|\) — same as today. |
| Warm magnitude | Existing **`magnitude_caches/mag_aggregate_step_*.pt`** — no delta re-stream. |
| \(H_{ii}\) | **New:** Fisher job → `alignment_fisher_diag.pt` (or future Hessian diag if available). |
| \(\rho\) | From **`mask_score_gap_run.json`** `sparsity_percent`. |
| \(C,\epsilon,\eta\) | CLI/env; log in JSON; sensitivity sweeps optional later. |

---

## Artifacts (never overwrite core run files)

Write **only**:

- `mask_score_gap_alignment_gap_diagnostics.json` — `B_trace`, `iterations`, `converged`, `C`, `epsilon`, `eta`, `rho`, Fisher metadata, `s_star_formula_version`.
- `mask_score_gap_alignment_histograms.npz` — binned series for gaps vs \(s^*\), optional \(s^*\) marginal.

Leave **`mask_score_gap_histograms.npz`**, **`mask_score_gap_summary.csv`**, **`mask_score_gap_gap_diagnostics.json`** untouched unless an explicit **`--append-summary`** flag is added later.

---

## Plotting ([`scripts/report_mask_score_gap_plots.py`](scripts/report_mask_score_gap_plots.py))

- Auto-detect or `--alignment-npz` / `--alignment-json` for sidecar artifacts; **overlay** on existing milestone/raw/norm/margin panels:
  - ECDF of **gaps** \(|s_{\mathrm{oracle}}-s^*|\), \(|s_{\mathrm{mag},t}-s^*|\), \(|s_{\mathrm{rand}}-s^*|\) at converged \(B^*\) (and optionally marginal of \(s^*\) vs oracle for intuition).
  - **Cert / acceptance:** optional second row or inset summarizing fraction of weights where \(|s-s^*|< m_i(s)\) if we bin margins from existing NPZ (bridge to Appendix certifiability).
- **Terminology:** **ECDF** = **empirical** cumulative distribution function (standard statistics — the **E** is “empirical,” not “estimated” in the ML sense). For a smooth density companion curve, implement **kernel density estimate (KDE)** or **histogram + light smoothing** on log-spaced grids for raw gaps; label axes/caption **“KDE density”** or **“smoothed histogram”** — avoid undefined acronym **EPDF** unless the paper defines it explicitly.
- **`--paper-style`:** matplotlib rcParams (serif/sans, font size, line width, color cycle); write outputs under **`figures_paper/`** (and optional `.pdf`) while leaving default **`figures/`** untouched for iterative debugging.

---

## Implementation order

1. [`src/analysis/alignment_optimal_scores.py`](src/analysis/alignment_optimal_scores.py) — \(s^*\), \(\Phi\) loop, tests.
2. Fisher export script — raw diagonal `.pt` + JSON meta.
3. [`scripts/compute_mask_score_gap_alignment_reference.py`](scripts/compute_mask_score_gap_alignment_reference.py) — sidecar only.
4. Plot overlays + paper style + optional KDE.
5. Optional certifiability-bridge summaries/plots.

---

## Risks

- Fisher \(\neq\) true \(H_{ii}\); all plots/diagnostics must caption **Fisher-diagonal proxy for \(H_{ii}\)**.
- \(C\) and **gradient bound \(\epsilon\)** are theory constants — may need grid search or values from training stability logs; **log** \(B^*\), score quantiles, and chosen \((C,\epsilon)\) in JSON for reproducibility.
- **Scale mismatch:** if \(|(CB+\epsilon)|\theta||\) dominates or vanishes vs \(\frac{H_{ii}}{2}\theta_i^2\), fixed-point behavior changes — document ranges in diagnostics.
- Full-model Fisher remains **GPU + data** expensive — keep alignment reference compute as **CPU post-process** once Fisher `.pt` exists.
- **Order sensitivity:** \(s^*\) tie-breaking and global top-\(k\) must use **identical key order** as mask-score-gap for apples-to-apples cert comparisons.
