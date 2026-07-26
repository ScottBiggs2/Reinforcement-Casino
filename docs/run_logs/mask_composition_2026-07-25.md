# Mask composition audit — Llama-3.1-8B @ 97.5% (job 8724440)

**Date:** 2026-07-25 · **Branch:** `irene-rebuttal-lora` · **Cost:** 1m37s CPU, `short` partition, 41.5 GB peak RSS
**Script:** `src/analysis/mask_composition.py` · **Driver:** `scripts/mask_composition_audit.sbatch`
**Report:** `/scratch/xie.yiyi/rebuttal_analysis/mask_composition_llama8b.json`
**Masks:** `/scratch/xie.yiyi/transfer_v1/oracle_masks_llama8b/`

Over-selection = (share of retained budget) / (share of model parameters). 1.00x means
the class is represented in the subnetwork exactly in proportion to its size.

## Over-selection by tensor class

| class | % of model | DPO light-r1 | random s42 | DPO tulu3 | GRPO math220k |
|---|---|---|---|---|---|
| mlp_gate_up | 46.80% | 1.03x | 1.00x | 0.98x | 0.97x |
| mlp_down | 23.40% | **1.08x** | 1.00x | 1.02x | **1.16x** |
| attn_qkv | 10.03% | 0.98x | 1.00x | 0.99x | **0.83x** |
| attn_o | 6.69% | **1.09x** | 1.00x | 1.06x | **1.22x** |
| lm_head | 6.54% | **0.76x** | 1.00x | 0.98x | 0.81x |
| embed | 6.54% | **0.69x** | 1.00x | 1.03x | 0.86x |
| layernorm | 0.0033% | 0.70x | *absent* | 0.78x | **0.14x** |

All masks hold exactly 200,756,531 / 8,030,261,248 = 2.5000%.

## Findings

### 1. The BitFit hypothesis is refuted — the subnetwork is not norms-and-biases

LayerNorm is 0.0033% of the model and receives 0.0023% of the DPO oracle's budget
(0.70x — *under*-selected). And **Llama-3.1 has no biases in its linear layers at all**:
no `bias` class appears in any mask. So the concern that the 2.5% subnetwork is
BitFit-plus-noise (Ben Zaken et al., ACL 2022) is not merely unsupported here, it is
structurally impossible for this architecture. This removes a threat rather than
creating one, and it is worth stating pre-emptively.

### 2. But every sparse run trains all LayerNorm parameters densely — including random

`SparseAdamW.step()` gates sparsity on `has_mask(name) and len(p.shape) == 2`
(`src/optimizers/sparse_adamw.py:113-119`). LayerNorm weights are 1-D, so they fall
through to `_dense_step` in **every** arm, whether or not the mask file contains an
entry for them. (`mlp_only` defaults to False and is never passed by any sbatch, so
attention and MLP are both genuinely masked — the 1-D guard is the only leak.)

Consequences:

- The precise claim is "2.5% of 2-D weight matrices, plus 100% of LayerNorm scales",
  i.e. 201.02M rather than 200.76M trainable. Numerically negligible: 2.5033% vs
  2.5000%. The headline number does not change.
- It is *not* negligible as a confound. Every arm gets fully-trained LayerNorm for
  free, which compresses the difference between arms. Given three reviewers
  independently observed that the evaluation cannot separate oracle from random, this
  is a candidate contributor and should be named rather than discovered by a reviewer.
- Cheap test if a slot is free: one oracle run with LayerNorm explicitly frozen. If
  nothing changes, the confound is dismissed with a number.

### 3. The random baseline's mask does not cover the same tensor set as the oracle

`random_baseline_lightr1_sp97.5_seed42.pt` has **no LayerNorm entries at all**
(8,029,995,008 params covered vs the oracle's 8,030,261,248 — the difference is exactly
the 266,240 LayerNorm parameters). Because of finding 2 the training behaviour is
identical either way, so this changes no result — but the two masks are objects over
different parameter sets, which is worth fixing before anyone diffs them.

### 4. The oracle's class-level signal is real but weak

Every over-selection factor sits in 0.69–1.09x. The pattern is interpretable: the DPO
oracle **avoids the embedding and output layers** (embed 0.69x, lm_head 0.76x) and
mildly prefers the **output projections** (attn_o 1.09x, mlp_down 1.08x). Random is
1.00x everywhere by construction.

The honest reading: at tensor-class granularity the oracle is only mildly non-uniform,
so whatever distinguishes it from random lives in *which coordinates* rather than
*which kinds of tensor*. A corollary worth noting — a globally-random mask is already
approximately class-matched to the oracle, so the Frankle et al. (ICML 2021)
layerwise-shuffle control is only informative at **depth** granularity, not class.

### 5. DPO and GRPO subnetworks differ in composition, not just in coordinates

This is the positive result. Relative to DPO light-r1, the GRPO oracle:

- concentrates harder in the **output projections**: attn_o 1.22x vs 1.09x,
  mlp_down 1.16x vs 1.08x
- pulls out of the **QKV projections**: 0.83x vs 0.98x
- nearly abandons **LayerNorm**: 0.14x vs 0.70x, a 5x difference

Figure 2 already shows the two objectives select different *coordinates* (Jaccard ~0.1).
This shows they select different *kinds of tensor*, which is an independent and more
interpretable axis for the same claim, obtained from masks that already existed at zero
training cost.

### 6. The Tülu3 oracle is closer to uniform than the Light-R1 oracle

Tülu3 over-selection factors are 0.98–1.06x versus Light-R1's 0.69–1.09x — a visibly
flatter profile, particularly on embed (1.03x vs 0.69x) and lm_head (0.98x vs 0.76x).
Consistent with Tülu3 being the weak-signal arm: 500 steps is 0.23 epoch there, so
|Δθ| has barely separated from noise and the resulting "oracle" is correspondingly
less structured. This gives a mechanism for the Table 1 reading rather than only an
assertion.

## Correction to an earlier note

`anti_oracle_dpo_lightr1_step500_sp97.5.pt` is **not** a bottom-2.5% mask. Measured
density is 97.5000% — it is the complement (1 − M), used by the probe
necessity/sufficiency analysis. The bottom-k-by-|Δθ| control does **not** exist and
still needs generating (it requires the score tensor, not an inversion of the mask).

## Not yet run

- `src/analysis/delta_analysis.py` (mass concentration + stable rank) — needs the dense
  checkpoints. `dense_grpo_*` returned no checkpoints on /scratch, so the DPO-vs-GRPO
  concentration curve may not be recoverable; the Qwen3-8B and Llama Tülu3 dense
  checkpoints are present.
