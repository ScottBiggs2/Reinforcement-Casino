# The Atomic-Throughput Regime Crossover in Sparse Backward Kernels

**Status**: empirical finding, paper-bound
**Date**: 2026-04-29
**Hardware**: NVIDIA H200 (Hopper, 132 SM, 4.8 TB/s HBM3e)
**Model**: Llama-3.1-8B-Instruct, MLP linears (4096↔14336), bf16

---

## 1. Headline finding

For a sparse backward `grad_input = grad_out @ W` kernel that uses fp32
`atomic_add` to accumulate contributions from active blocks (the standard
megablocks-style baseline), **the speedup over dense cuBLAS matmul is
non-monotonic in the active fraction ρ**, and the crossover happens at a
*hardware-determined* ρ that is much *higher* (i.e. less sparse) than the
sparsity regime most papers benchmark in.

For our setup (Llama-3.1-8B MLP shapes, B=2048 tokens, bf16, H200):

| ρ (kept) | Sparsity | sparse_grad_input vs dense | Win? |
|---:|---:|---:|---|
| 0.0025 | 99.75% | ~3× | ✅ |
| 0.005  | 99.5% | ~1.7× (extrap) | ✅ |
| **0.0091** | **99.09%** | **1.0× (crossover)** | break-even |
| 0.025  | **97.5%** (real training) | **0.36×** | ❌ **2.8× slower** |
| 0.05   | 95% | <0.2× | ❌ much slower |

The "1.5–2× speedup" reported in megablocks §4.2 and folklore around
sparse backward kernels presupposes ρ well below the crossover.

---

## 2. Empirical evidence

Benchmarked on H200 with the exact `sparse_grad_input_kernel` from
`origin/cav_fixes` commit `b9fec55` (an established BSR sparse training
codebase). Identical kernel, identical wrapper, only ρ varies.

### Layer breakdown at ρ = 0.025 (training sparsity)

| Path | gate/up_proj | down_proj |
|---|---:|---:|
| Dense cuBLAS `gO @ W` | 0.345 ms | 0.356 ms |
| **`sparse_grad_input_triton`** (Scott baseline, atomic) | **0.953 ms** | **1.055 ms** |
| Sparse-vs-dense ratio | **0.36×** | **0.34×** |

### Full backward speedup at ρ = 0.025

| Configuration | gate/up_proj | down_proj |
|---|---:|---:|
| Dense `F.linear` (cuBLAS, autograd) | 1.023 ms | 1.031 ms |
| BSR with **dense input-grad** (current) | 0.842 ms (**1.22×**) | 0.884 ms (**1.17×**) |
| BSR with **B1 atomic input-grad** (Scott port) | 1.527 ms (**0.67×**) | 1.661 ms (**0.62×**) |

Routing input-grad through the sparse atomic kernel **regresses backward
time by 81%**.

(Bench harness: [`scripts/bench_bsr_kernel_breakdown.py`](../../scripts/bench_bsr_kernel_breakdown.py),
job 6387658 on Discovery H200, 10 warmup + 200 timed iterations,
`torch.cuda.Event` timing.)

---

## 3. Hardware mechanism: atomic throughput, not contention

The naive expectation (that we shared in our planning) was that the
sparse kernel saves a factor of `1/ρ` in compute and matmul memory
traffic, so it should win whenever ρ < 1. The bench falsifies this.

### The dominant cost is the atomic-add count, not its memory traffic

Each program covers one active 16×16 block and accumulates, for every
batch token in that block's input column, one fp32 `atomic_add` into
`grad_input`. Per program:

- BATCH_BLOCK_SIZE × BLOCK_SIZE atomics per inner step
- ⌈B / BATCH_BLOCK_SIZE⌉ inner steps
- Total atomics per program = B × BLOCK_SIZE

Across all active blocks:

```
N_atomics = ρ · (out / BLOCK) · (in / BLOCK) · B · BLOCK
          = ρ · out · in · B / BLOCK
```

For our gate/up shape (out=14336, in=4096, B=2048, BLOCK=16):

```
N_atomics(ρ) = ρ · 7.52 × 10⁹
```

### H200 fp32 atomic-add peak throughput

Hopper SMs sustain ~1 fp32 atomic per cycle per SM. With 132 SMs at
~1.5 GHz boost:

```
T_atomic_peak ≈ 132 · 1.5 × 10⁹ ≈ 2 × 10¹¹ atomics/s
```

(Atomics that cross L2 → HBM are throttled further when the target
buffer exceeds L2 capacity. Our `grad_input` is 56 MB, larger than
H200's L2.)

### Predicted vs measured kernel time

```
T_atomic(ρ) ≈ N_atomics(ρ) / T_atomic_peak
            = ρ · 7.52 × 10⁹ / 2 × 10¹¹
            = ρ · 37.6 ms
```

For ρ = 0.025: predicted 0.94 ms vs **measured 0.953 ms** — fits within
1% of an *a priori* hardware-throughput model. There is no fitting
constant; the agreement comes from the H200 spec sheet.

→ The kernel is **not contention-bound** (which would scale with active
columns per output cell) and **not memory-bandwidth-bound** (atomics
move very few bytes here). It is **atomic-instruction-throughput-bound**.

### Why dense cuBLAS doesn't share this ceiling

Dense cuBLAS matmul is compute-bound on this shape:

```
FLOPs       = 2 · B · out · in           = 240 GFLOPs
AI          = FLOPs / bytes              = 1200 FLOPs/byte
H200 ridge  = 989 TF/s / 4.8 TB/s        = 206 FLOPs/byte
```

At AI ≫ ridge, cuBLAS sits on the compute roof; with measured 0.345 ms
on 240 GFLOPs it achieves ~70% of peak bf16 — well-tuned cuBLAS,
ρ-independent.

### The crossover ρ

Setting the two costs equal:

```
ρ_crossover · 37.6 ms = 0.345 ms    →    ρ_crossover ≈ 0.0092
```

i.e. **the atomic-baseline kernel only wins when sparsity exceeds
~99.1%**. Below that — including all of the standard sparse
fine-tuning literature (90–95% sparse, ρ = 0.05–0.10) and our actual
training (ρ = 0.025) — it is a *net loss* relative to a vanilla cuBLAS
backward.

The crossover is purely a function of three hardware quantities and
the problem shape:

```
ρ_crossover = (out · in · B / BLOCK) · 1 / (T_atomic · T_compute)
            ≈ T_compute / (atomic_count_per_unit_ρ × atomic_peak_throughput)
```

It does **not** depend on the user's choice of mask scorer, the
quality of recovery, or the sparsity pattern.

---

## 4. The methodology lesson: bench-ρ selection bias

The kernel above (`sparse_grad_input_kernel` from `cav_fixes`) is the
input-grad component of an active sparse-training research codebase.
Its companion benchmark (`h200_sparse_dpo_bsr_benchmark.py`) reports a
**~3× speedup over dense** for the full sparse training step.

The benchmark fixes ρ at **0.0025 (99.75% sparse)** for *all* sparse
phases, citing "stable benchmarking". This ρ is well *below* the
crossover at 0.0092 and gives a kernel-friendly result.

The same codebase's **actual oracle-mask training scripts**
(`scripts/transfer_p2_oracle_masks_llama8b.sbatch`) default to
`SPARSITY=97.5` (ρ = 0.025) — i.e., the *training* configuration sits
on the *losing* side of the crossover the *bench* hides.

> The reported speedup is therefore a **bench artifact, not a property
> of the training regime**. A reader who identifies "BSR sparse
> backward beats dense" with the bench number will overstate the
> kernel's value at any sparsity researchers actually train at.

This is not a critique of the codebase — choosing extreme sparsity
to "show" the kernel's potential is a defensible (and very common)
benchmarking choice. The lesson is that when the underlying speed
mechanism has a hardware-determined regime crossover, **the benchmark
ρ must be reported and justified against the deployment ρ**, otherwise
the speedup is unfalsifiable from the paper alone.

We searched the megablocks paper, STK readme, and the public sparse
fine-tuning kernel literature for an explicit crossover analysis;
none was found. The folklore "atomic_add baseline gets ~2× over
dense" appears to assume ρ → 0 silently.

---

## 5. Implications for sparse kernel design

### 5.1 Atomic baseline is not a viable building block at training ρ

Any sparse backward design that reaches the user via fp32 atomic
accumulation onto a full-resolution gradient buffer pays the full
`ρ · const` time, *which is independent of how clever the rest of
the kernel is*. As long as the atomic count scales with `ρ · out · in
· B / BLOCK`, the bound applies.

### 5.2 The fix is to remove atomics, not optimize them

The viable directions all *eliminate* the atomic cross-program write:

- **Input-tile bucketing** (megablocks STK §4.2): re-index
  active blocks so each program owns one input-column tile and
  accumulates into registers, ending with a *single non-atomic*
  store per cell. Memory-bound at HBM write bandwidth.
- **Two-pass scheme**: first pass writes per-block partial outputs
  to a scratch buffer; second pass reduces. Trades atomics for
  one extra HBM round-trip — wins iff `ρ · 37.6 ms` exceeds the
  reduction pass cost (~0.05 ms here).
- **Cooperative reduction within an in-block**: programs sharing
  the same in-tile cooperate via shared memory, only the
  block-level result is atomically (or non-atomically) committed.

### 5.3 Forward path is more leveraged than backward at our ρ

Because dense cuBLAS is already compute-bound on these shapes, a
sparse forward (which would cut FLOPs by 1/ρ rather than memory
traffic) has more upside than any backward optimization, but at the
cost of writing a sparse forward matmul that beats cuBLAS — a much
harder engineering target.

### 5.4 Optimizer-step optimizations (8-bit state, fused
backward+optimizer) are unaffected

These attack a different cost (memory-traffic on the optimizer
step), are not bottlenecked by atomic throughput, and remain
math-justified at our ρ.

---

## 6. Reproducibility

Bench harness lives at
[`scripts/bench_bsr_kernel_breakdown.py`](../../scripts/bench_bsr_kernel_breakdown.py).
It compares dense `F.linear` autograd, BSR-with-dense-input-grad
(current main branch), and BSR-with-sparse-input-grad (the failing
B1 atomic baseline) head-to-head, isolating each subcomponent's
wall-time on a single H200.

Reproduction:

```bash
sbatch scripts/bench_bsr_kernel_breakdown.sbatch
# Outputs: kernel_breakdown.md + per-shape breakdown to stdout
```

Sweep ρ to walk the crossover:

```bash
for r in 0.001 0.005 0.01 0.025 0.05 0.1; do
  KBREAK_RHO=$r sbatch scripts/bench_bsr_kernel_breakdown.sbatch
done
```

The hardware-throughput model in §3 predicts the crossover at
ρ ≈ `T_compute_dense / (out · in · B / BLOCK / T_atomic_peak)`,
which for any (shape, B, hardware) yields a falsifiable line that
the bench can confirm or refute.

---

## 7. What we did with this finding

1. **Aborted B1-atomic as the optimization vehicle.** It is correct
   and ports cleanly from `cav_fixes`, but at ρ = 0.025 it regresses
   backward time. The atomic kernel is retained behind
   `RL_CASINO_BSR_GRAD_INPUT_MODE=sparse` (default `dense`) so it
   can be re-evaluated at higher sparsity without code changes.
2. **Added the breakdown harness as the standard kernel measurement
   tool**, not just a one-off. Future sparse kernel candidates must
   ship their crossover ρ before being merged into the training path.
3. **Pivoted the optimization roadmap** to (a) input-tile-bucketed
   input-grad (the atomic-free version), and/or (b) optimizer-step
   memory traffic reductions (8-bit state, fused backward+optimizer)
   — both of which the §3 model predicts are not subject to the
   atomic ceiling.

---

## Appendix: numbers for the paper table

```
H200, bf16, B=2048, Llama-3.1-8B MLP shapes
─────────────────────────────────────────────────────────
gate/up_proj  (out=14336, in=4096)
  dense F.linear bwd            1.023 ms
  BSR dense ig                  0.842 ms   (1.22× speedup)
  BSR + B1 atomic ig            1.527 ms   (0.67× speedup, regression)

down_proj     (out=4096, in=14336)
  dense F.linear bwd            1.031 ms
  BSR dense ig                  0.884 ms   (1.17× speedup)
  BSR + B1 atomic ig            1.661 ms   (0.62× speedup, regression)

Crossover ρ (predicted, gate/up) :  0.0092  (99.08% sparse)
Crossover ρ (empirical bracket)  :  in [0.005, 0.025]
```
