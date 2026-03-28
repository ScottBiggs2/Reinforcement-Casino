# Complexity Analysis: Mask Construction Methods

## Scope

This note analyzes the current mask-construction stack in:

- `src/utils/mask_utils.py`
- `src/warm_start/even_better_mask_finder.py`
- `src/cold_start/cold_mask_finder.py`
- `src/cold_start/cav_cold_mask_finder.py`
- `src/cold_start/utils/snip_scorer.py`
- `src/cold_start/utils/cav_probes.py`
- `src/warm_start/random_mask_baseline.py`
- `src/utils/generate_random_mask.py`

The goals are:

1. Clarify the current time and memory complexity of each method.
2. Identify where "streaming" is real versus where we still materialize full-model state.
3. Highlight correctness and robustness issues.
4. Propose concrete optimization directions, especially for exact global pooling.

This is a draft for internal research planning, not a polished paper section.

---

## Executive Summary

The most important structural fact is this:

- Most mask methods are only "streaming" in the **score accumulation** phase.
- The repo now has an **exact chunked global selection phase** for large models, but some callers can still be limited by how scores are accumulated before selection.
- The core methodological distinction remains: **exact global competition** versus **regional/blockwise approximations**.

So the dominant bottleneck for scale is no longer "how do we compute scores?" but:

**How do we do exact global top-k over very large score tensors without materializing multiple `O(P)` copies?**

Where:

- `P` = total number of scored weight elements across all selected matrices.

Current large-model global pooling is exact, but chunked:

- It reserves any per-tensor floor locally.
- It refines a global score interval with repeated histogram passes over score chunks.
- It materializes only the narrowed boundary candidate set exactly.
- It then writes per-layer masks directly.

This means the selector no longer needs to materialize one monolithic flattened score vector for large `P`, even though it still does multiple full passes over the scores.

The current code is therefore:

- conceptually correct for global pooling,
- operationally much safer at large scale,
- especially vulnerable for warm momentum and any full-model setting,
- and still inconsistent across some helper utilities.

---

## Notation

- `P`: total number of scored weight elements after any filtering such as `mlp_only`.
- `L`: number of scored parameter tensors.
- `T`: number of delta checkpoints used by warm-start methods.
- `W`: momentum window size.
- `N`: number of calibration examples for cold-start methods.
- `B`: number of calibration minibatches.
- `D_l`: hidden/intermediate width for layer `l`.
- `p_l`: number of scored parameters in tensor `l`.
- `K`: number of kept parameters, `K = (1 - sparsity) * P`.

Important implementation note:

- The exact complexity of `torch.topk` is backend-dependent. In practice it behaves like a specialized selection kernel, not a full `O(P log P)` sort.
- For this note, the important point is not the exact asymptotic constant of `topk`, but that the current implementation still requires **full materialization of the candidate score vector** before selection.

---

## Pooling Semantics Today

### Shared global pooling path

The main production paths call `create_mask_from_scores_gpu_efficient()` in `src/utils/mask_utils.py`.

Default behavior:

- `local_pool=False`
- repo-wide default `min_layer_keep_ratio=0.0025`
- one global ranking across all scored weights plus a small per-tensor keep floor
- exact target sparsity

Optional alternate behaviors:

- `local_pool=True`: per-weight-matrix top-k
- `min_layer_keep_ratio=0.0`: pure global top-k with no floor

### Methods currently using the shared global path

- Warm magnitude
- Warm momentum
- Warm Fisher
- Cold Fisher
- Cold CAV / activation / SNIP
- `warm_start/random_mask_baseline.py`
- `utils/generate_random_mask.py`

### Methods not using the shared path

- `src/cold_start/utils/cav_probes.py::scores_to_masks()`

These helper paths still implement their own thresholding logic.

---

## Shared Global Pooling Engine

### Current algorithm

In `create_mask_from_scores_gpu_efficient()`:

1. Convert score tensors to `float32`.
2. For small `P`, use the original flat exact path.
3. For large `P`, move scores to CPU and optionally add tiny tie-break noise.
4. Reserve any per-layer floor positions tensor-by-tensor.
5. Run an exact chunked threshold refinement pass over score chunks.
6. Gather only the final boundary candidates exactly.
7. Materialize per-layer masks directly and save them in the same float32 on-disk format.

### Time complexity

- Score normalization / device transfer: `O(P)`
- Flat path concatenate: `O(P)`
- Chunked path histogram refinement: `O(RP)` where `R` is the number of refinement passes
- Final exact boundary top-k: `O(C) + topk(C, K_b)` where `C` is the narrowed candidate set
- Reconstruct per-layer masks: `O(P)`

Overall:

- Flat path: `O(P) + topk(P, K)`
- Chunked exact path: `O(RP) + topk(C, K_b)`
- In practice, the dominant work is still memory traffic, but the large-model path trades one huge allocation spike for several sequential scans.

### Peak memory complexity

Ignoring Python container overhead, the flat path can require:

- existing score storage: `O(P)`
- `all_scores`: `O(P)`
- `global_mask_flat`: `O(P)`
- `candidate_scores`: `O(P)`
- output masks on CPU: `O(P)`

So the global selection stage alone can transiently cost roughly:

- **device**: about `3P` extra float32 elements beyond the stored score tensors
- **host**: `O(P)` once masks are copied out

This was the key systems bottleneck in the original design.

For the chunked exact path:

- score storage: `O(P)`
- per-layer floor indices: `O(F)` where `F` is the total floor reservation
- per-layer boolean masks on CPU: `O(P)`
- narrowed boundary candidates: `O(C)`

So the selector no longer needs extra `O(P)` flattened score buffers on top of the stored scores.

### Robustness status

The code now includes `_topk_indices_safe()` with a CPU fallback for very large CUDA tensors, and the large-model selector itself is CPU-first and chunked. That avoids both the old CUDA `topk` failure mode and the monolithic flat-vector allocation.

### Key limitation

This function is now "exact global" and "chunked global," but it is still not single-pass streaming because exactness requires multiple scans over the score tensors.

---

## Warm-Start Methods

All warm methods live in `src/warm_start/even_better_mask_finder.py`.

## Warm Magnitude

### Algorithm

For each delta checkpoint:

- load deltas on the score-accumulation device,
- accumulate `abs(delta)` into `aggregated[name]`,
- discard checkpoint tensor,
- after all checkpoints, run exact global pooling on `aggregated`.

### Time complexity

- Delta loading and accumulation: `O(TP)`
- Final global selection: `O(P) + topk(P, K)`

Overall:

- **Time**: `O(TP) + topk(P, K)`

### Memory complexity

Persistent during accumulation:

- `aggregated`: `O(P)`

Transient during each checkpoint:

- current delta tensor(s): approximately `O(P)` worst-case if full-model deltas are loaded at once

Transient during final global pooling:

- flat path: roughly `+3P` float32 elements
- chunked path: `O(C)` boundary candidates plus per-layer masks on CPU

So peak memory is effectively:

- **Score device**: `O(P)` during accumulation
- **Selector**: no extra flattened `O(P)` score vector on the chunked path

### Assessment

This is a reasonable score-computation design. It is still not fully streaming end-to-end, but the selector no longer needs a monolithic flattened score vector on large models.

---

## Warm Momentum

### Algorithm

For each checkpoint:

- load current deltas,
- compute velocity `delta_t - delta_{t-1}`,
- append to a per-parameter sliding window,
- after the pass, compute:
  - mean velocity,
  - std of velocity,
  - consistency,
  - momentum score = `abs(mean_velocity) * (1 + consistency)`,
- then run exact global pooling.

### Time complexity

- Reading checkpoints and building velocities: `O(TP)`
- Computing per-parameter statistics over the velocity window: `O(WP)`
- Final global selection: `O(P) + topk(P, K)`

Overall:

- **Time**: `O(TP + WP) + topk(P, K)`

### Memory complexity

Persistent state can include:

- `prev_deltas`: `O(P)`
- `velocity_window`: up to `O(WP)`
- `momentum_scores`: `O(P)`

Then add final global pooling overhead:

- roughly `+3P` device elements

So in the worst case:

- **Device**: roughly `O((W + 4)P)`

This is the most memory-dangerous warm method in the current stack.

### Assessment

The code describes this as streaming because it does not keep all checkpoints. That is only partially true. The sliding velocity window is still an `O(WP)` structure, which is large enough to explain the observed OOM failures.

### Best improvement opportunity

Replace explicit velocity windows with one of:

- exponential moving averages,
- online mean/variance recurrences,
- or chunked per-block moment accumulation.

Any of these can reduce memory from `O(WP)` toward `O(P)`.

---

## Warm Fisher

### Algorithm

For each checkpoint:

- accumulate `sum_delta[name] += delta`
- accumulate `sum_delta_sq[name] += delta ** 2`

After the pass:

- compute `mean_delta`
- compute `mean_delta_sq`
- compute `variance = mean_delta_sq - mean_delta ** 2`
- define score as `variance + abs(mean_delta)`
- run exact global pooling

### Time complexity

- Accumulation: `O(TP)`
- Final score formation: `O(P)`
- Final global selection: `O(P) + topk(P, K)`

Overall:

- **Time**: `O(TP) + topk(P, K)`

### Memory complexity

Persistent state:

- `sum_delta`: `O(P)`
- `sum_delta_sq`: `O(P)`

Post-pass:

- `fisher_scores`: `O(P)`

Selection overhead:

- about `+3P`

So peak memory is approximately:

- **Device**: `O(6P)` in the worst phase

### Assessment

This method is more stable than momentum but still very heavy. It keeps two full-model accumulators plus another full-model score tensor before selection.

### Important implementation note

The comment says "using Welford's algorithm," but the code is not actually Welford. It uses raw sums and sum-of-squares. That is mathematically fine in principle, but:

- it is not the numerically stable online Welford recurrence,
- and it costs more memory than a true running-mean/running-M2 formulation would.

---

## Cold-Start Fisher

Implemented in `src/cold_start/cold_mask_finder.py`.

### Algorithm

1. Load `N` calibration examples.
2. Encode and store them.
3. For each minibatch of size `m`:
   - run forward/backward on the chosen sequence loss,
   - accumulate squared gradients into `fisher_scores[name]`
4. Normalize by minibatch count.
5. Optionally z-score each layer.
6. Run exact global pooling.

### Time complexity

Let `C_fwd_bwd` be the cost of one calibration forward/backward pass through the model.

- Calibration scoring: about `O(B * C_fwd_bwd)`
- Gradient accumulation over all parameters: `O(BP)` memory traffic
- Per-layer normalization: `O(P)`
- Final global pooling: `O(P) + topk(P, K)`

Overall:

- **Time**: dominated by `O(B * C_fwd_bwd) + topk(P, K)`

### Memory complexity

Persistent:

- model weights and optimizer-free autograd state
- `fisher_scores`: `O(P)`

Additional issue:

- `load_calibration_data()` currently stores encoded examples in a Python list and moves tensors to the device early.
- That adds an avoidable `O(N * sequence_length)` device-memory footprint before scoring even begins.

### Assessment

The gradient-based cost is expected. The avoidable inefficiency is the early, eager device placement of calibration samples.

### Improvement directions

- Keep calibration examples on CPU until minibatch execution.
- Use a proper `DataLoader` with pinned memory.
- Optionally accumulate Fisher scores in chunks or per-block groups.

---

## Cold Activation / CAV / SNIP

Implemented in `src/cold_start/cav_cold_mask_finder.py`.

These methods differ in how they compute scores, but they all eventually call the same global pooling helper.

## Cold Activation

### Algorithm

- Register hooks on `down_proj`
- Run chosen pass and rejected pass
- Pool token activations to `[N, D]` per layer
- Compute per-neuron mean absolute activation
- Broadcast neuron scores back to dense weight scores
- Run exact global pooling

### Time complexity

- Two forward sweeps over selected batches
- Per-layer reduction over pooled activations
- Broadcast-to-weight expansion: `O(P)`
- Final global pooling: `O(P) + topk(P, K)`

Overall:

- **Time**: roughly `O(forward passes) + O(P) + topk(P, K)`

### Memory complexity

The dominant issue is not the forward passes, but activation retention:

- pooled activations are stored on CPU lists for each layer and later concatenated,
- so memory is roughly `O(total collected pooled activations)`,
- then weight-score expansion adds another `O(P)` CPU footprint.

## Cold CAV

### Algorithm

- Collect pooled chosen activations and pooled rejected activations
- For each layer:
  - fit an `sklearn` logistic regression probe
  - produce a neuron-level score vector
- Map neuron scores to dense weight scores
- Run exact global pooling

### Time complexity

There are two distinct phases:

1. Activation collection:
   - two forward passes over `num_batches`
2. Probe fitting:
   - per-layer logistic regression on CPU

If `X_l` is the pooled activation matrix for layer `l`, then probe fitting is roughly:

- `O(sum_l fit_cost(X_l))`

This is not negligible. It is the main extra cost beyond ordinary forward passes.

### Memory complexity

- chosen activations stored explicitly
- rejected activations stored explicitly
- intermediate labeled activation matrices
- broadcast dense weight scores: `O(P)` on CPU
- final global pooling adds another `O(P)` path

This makes CAV the most CPU-memory-intensive cold-start method.

### Important observation

The probe hyperparameters `epochs`, `lr`, and `weight_decay` are currently passed around, but `compute_cav_scores()` uses `sklearn.linear_model.LogisticRegression` directly. Those hyperparameters are effectively ignored there.

That is not a complexity bug, but it is an experimental reproducibility concern.

## Cold SNIP

### Algorithm

- For each batch:
  - run chosen forward pass
  - run rejected forward pass
  - compute DPO-style preference loss
  - accumulate gradients
- After the loop:
  - compute `abs(grad * weight)` per scored tensor
- Run exact global pooling

### Time complexity

- About one forward/backward pair per batch over chosen and rejected inputs
- Final score extraction over all parameters: `O(P)`
- Final global pooling: `O(P) + topk(P, K)`

Overall:

- **Time**: dominated by repeated full-model forward/backward passes

### Memory complexity

- model gradients across scored parameters: `O(P)`
- CPU score tensors after extraction: `O(P)`
- global pooling stage as usual

### Assessment

SNIP is computationally expensive but structurally simpler than CAV. It does not store long-lived activation datasets, but it does require full gradient computation.

---

## Random Baselines

There are two random-mask paths, and they are not equivalent operationally.

## `warm_start/random_mask_baseline.py`

### Algorithm

- Load a reference mask to obtain tensor shapes
- Draw one random score tensor per reference tensor
- Call `create_mask_from_scores_gpu_efficient()`

### Complexity

- Random score creation: `O(P)`
- Final exact selection: `O(P) + topk(P, K)`

Peak memory is similar to other score-based methods:

- random score tensors: `O(P)`
- global selection buffers: another several `O(P)` terms

### Assessment

This is the better random baseline for methodological parity because it uses the same exact selection engine as the learned masks.

## `utils/generate_random_mask.py`

### Algorithm

- Load the model on CPU
- Build one random score tensor per target weight matrix
- Call `create_mask_from_scores_gpu_efficient()` on CPU
- Save metadata that now matches the shared pooling semantics

### Time complexity

- Model inspection: `O(number of parameters)` just to enumerate shapes
- Score generation: `O(P)`
- Final exact selection: `O(P) + topk(P, K)`

### Memory complexity

This path is still memory-heavy on CPU:

- random score tensors: `O(P)`
- global selection buffers: another several `O(P)` terms
- masks: `O(P)`
- model weights are also resident on CPU

So peak host memory can easily exceed multiple copies of the scored parameter set.

### Assessment

This script is now aligned with the shared selector semantics, but it still pays the cost of loading the full model on CPU just to infer tensor shapes.

---

## Current Bugs, Inconsistencies, and Risks

## 1. Warm Fisher comment is inaccurate

`compute_fisher_mask_streaming()` says it uses Welford's algorithm, but it currently uses raw `sum` and `sum_sq`. That should be corrected in comments or refactored into a true running-mean/running-M2 implementation.

## 2. `sanitize_model_name()` bug in cold Fisher

In `src/cold_start/cold_mask_finder.py`, the loop:

```python
while "__" in sanitized:
    sanitized = sanitized.replace("__", "__")
```

does not collapse repeated underscores and appears to be a typo. It is harmless for some common model names, but it is a latent bug.

## 3. "Streaming" is overstated in several places

Warm magnitude and warm Fisher stream over checkpoints, but still hold full-model accumulators and then fully materialize the score vector for global selection. Warm momentum is even less streaming because it stores a sliding tensor window.

## 4. Some helper paths still bypass the shared robust top-k code

The following still use custom `torch.topk` logic rather than `_topk_indices_safe()`:

- `src/cold_start/utils/snip_scorer.py::scores_to_masks()`
- `src/cold_start/utils/cav_probes.py::scores_to_masks()`

Even if these are not the main pipeline path today, they increase maintenance risk.

## 5. CAV probe hyperparameters are misleading

`epochs`, `lr`, and `weight_decay` are passed through the cold CAV pipeline, but the actual probe training uses `sklearn` logistic regression directly. The parameters are effectively ignored in `compute_cav_scores()`.

## 6. Hybrid floors change the null random-mask interpretation

Once a small per-tensor floor is the default, the closed-form iid random Jaccard baseline is exact only for pure global masking (`min_layer_keep_ratio=0.0`). Random baselines remain useful, but analysis language should distinguish hybrid-global random masks from pure-global iid random masks.

---

## Improvement Opportunities

Below is a prioritized roadmap from highest expected impact to lower impact.

## Priority 1: Replace full `torch.cat` global selection

This is the most important systems improvement.

### Option A: Exact threshold search with chunked counting

Idea:

- Do not materialize one global vector.
- Sweep chunks of score tensors and count how many values exceed a candidate threshold.
- Use binary search or histogram refinement to find the threshold yielding exactly `K` kept parameters.
- Perform a second pass to emit the mask.

Properties:

- exact global pooling
- memory close to `O(chunk_size)` beyond existing score storage
- more passes over the data

Approximate complexity:

- `O(P * num_passes)`
- where `num_passes` is typically modest if threshold search is well initialized

This is the cleanest exact alternative.

### Option B: External / blockwise top-k merge

Idea:

- partition scores by tensor or block,
- take a local top-k or oversampled top-k per block,
- merge candidate sets globally,
- run final top-k on the reduced candidate pool.

Properties:

- lower memory than full concatenation
- can be exact only if oversampling is large enough or if the merge procedure preserves all true global winners
- otherwise becomes approximate

This is probably what the user-facing "chunk by transformer block, then pool block top-k's" idea maps to.

Recommendation:

- implement this only if approximate global selection is acceptable,
- otherwise prefer Option A.

## Priority 2: Fix warm momentum memory

Replace explicit velocity windows with:

- exponential moving averages,
- online mean/variance recurrences,
- or a fixed small-statistics summary instead of storing raw windows.

Goal:

- reduce memory from `O(WP)` to `O(P)`

## Priority 3: Make cold Fisher calibration truly streaming

Current issues:

- encoded samples are moved to device early and stored in a Python list

Improve by:

- keeping raw tokenized examples on CPU,
- using a `DataLoader`,
- moving minibatches to device only when needed.

## Priority 4: Stream CAV activations and probes

Current issues:

- chosen and rejected pooled activations are stored for every layer before fitting probes
- dense weight-score expansion creates another `O(P)` object

Possible refactors:

- use an online linear classifier such as `SGDClassifier.partial_fit`
- store running sufficient statistics instead of full activation matrices
- postpone dense score expansion until thresholding time

## Priority 5: Unify all thresholding through `mask_utils`

Any path that builds masks should share:

- the same global/local semantics,
- the same metadata,
- the same large-tensor safety logic.

That means migrating or deleting custom thresholding code in helper utilities.

## Priority 6: Use numerically stable running statistics for warm Fisher

True Welford-style updates would:

- reduce confusion,
- improve numerical stability,
- and possibly reduce the need for duplicated accumulator tensors depending on formulation.

---

## Recommended Experimental Order

If we want the highest-value next steps with the least methodological ambiguity:

1. Implement an **exact chunked global selector** for `mask_utils.py`.
2. Refactor warm momentum to `O(P)` state.
3. Stream cold Fisher calibration from CPU minibatches.
4. Unify random baseline generation on top of the same selector.
5. Revisit CAV activation/probe memory once the shared selector is no longer the dominant bottleneck.

---

## External References

These are relevant background references for future optimization work:

### Systems / large-array selection

- PyTorch CUDA top-k PR discussing multi-block behavior for large slices:
  [pytorch/pytorch#71081](https://github.com/pytorch/pytorch/pull/71081)
- Dask `topk` documentation, which explicitly notes that top-k works best when `k` is much smaller than chunk size and that reduction structure matters for memory:
  [dask.array.topk](https://docs.dask.org/en/latest/generated/dask.array.topk.html)

These references are not proofs for our code, but they do align with the practical issue seen here: large-array top-k is often more of a systems problem than a pure asymptotic one.

### Pruning / subnetwork context

- Wanda emphasizes practical LLM pruning with per-output selection rather than a single model-wide global threshold:
  [Wanda](https://arxiv.org/abs/2306.11695)
- SparseGPT is a canonical example of layer-wise / blockwise local reconstruction for LLM pruning at scale:
  [SparseGPT](https://proceedings.mlr.press/v202/frantar23a.html)
- Movement Pruning is a strong first-order saliency reference for pretrained language models and supports the general saliency-ranking framing used here:
  [Movement Pruning](https://papers.nips.cc/paper/2020/hash/eae15aabaa768ae4a5993a8a4f4fa6e4-Abstract.html)
- GraSP is a classic gradient-flow preservation reference when reasoning about subnetwork selection criteria:
  [GraSP](https://arxiv.org/abs/2002.07376)

---

## Bottom Line

Today, the mask methods are mostly correct in their **ranking semantics**:

- default behavior is hybrid global pooling with a small per-tensor floor,
- random and cold methods now follow the same default semantics,
- pure global remains available via `min_layer_keep_ratio=0.0`,
- and `--local_pool` is opt-in.

But they are not yet scalable in their **selection implementation**.

The core issue is not how we score weights; it is that exact global pooling still expands into several full-model buffers. That is the right next target for systems work.
