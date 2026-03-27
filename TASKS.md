# Reinforcement-Casino Task List

## Easy

- [x] Verify that `MLP_ONLY = False` everywhere — critical for level comparisons with prior work
- [x] Verify that all mask finding algorithms (especially cold starts) use global pooling, not local pooling
- [x] Write up docs for the mask comparison and charting suite (`mask_to_jaccard.py`, `mask_to_cka.py`)
- [ ] Create a separate `eval` environment so that training and eval jobs can run concurrently without errors
- [ ] Run full pipeline and verify that evals for sparse trained models look good

## Medium

- [ ] Figure out how to clearly compare several masks at once, potentially across several axes
- [ ] Verify that CAV and other cold start masks work for GRPO
- [ ] Related literature search
- [ ] Measure training speedups with sparse masks/kernels — turn off WandB reporting, use high step counts to warm up the GPU
- [ ] Find an oddball or very unusual DPO/GRPO setting to compare with — could be interesting

## Hard

- [ ] Complexity analysis of sparse kernels
- [ ] Design experiments
- [ ] Compare masks across tasks (warm vs cold, method variance, dataset, objective, etc.)
- [ ] Compare masks across models (LLM A vs B)
- [ ] Scale to larger models or MoE models (MoE may not be compatible with current mask naming scheme — stick to dense models; may require multi-GPU partition)
- [ ] Implement Code-GRPO environment (Math is already set, instruction-following GRPO is likely not useful)

---

**Notes**

- `main` branch contains a complete end-to-end DPO pipeline; GRPO version coming soon
- Make sure your environment and paths are set up properly before starting
- Discuss progress on Tuesday
