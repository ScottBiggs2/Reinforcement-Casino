# Historical mask catalogue

Archive of the 12-mask layout that drove the earlier
`src/analysis/probe_pair_12masks.py` + `scripts/run_probe_pair_12masks.sh`
pipeline. Both of those files are removed; the active pipeline is now
`src/analysis/probe_pair_masks.py` with the 6-config default
(Oracle / Cold-CAV / Random × {DPO, GRPO}).

Kept here in case you want to re-run any of the retired masks — the
paths below are valid on the cluster scratch filesystem as long as the
mask files themselves have not been garbage-collected.

## DPO side (step50, sp97.5)

| Label | Path |
|-------|------|
| Cold-SNIP-DPO    | `/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_snip_dpo.pt` |
| Cold-CAV-DPO     | `/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_cav_dpo.pt` |
| Cold-Fisher-DPO  | `/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_fisher_dpo.pt` |
| Warm-Fisher-DPO    | `/scratch/xie.yiyi/rl_casino_masks/llama8b/warm_fisher_step50_sp97.5.pt` |
| Warm-Magnitude-DPO | `/scratch/xie.yiyi/rl_casino_masks/llama8b/warm_magnitude_step50_sp97.5.pt` |
| Warm-Momentum-DPO  | `/scratch/xie.yiyi/rl_casino_masks/llama8b/warm_momentum_step50_sp97.5.pt` |

## GRPO side

| Label | Path |
|-------|------|
| Cold-SNIP-GRPO   | `/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_snip_grpo.pt` |
| Cold-CAV-GRPO    | `/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_cav_grpo.pt` |
| Cold-Fisher-GRPO | `/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_fisher_grpo.pt` |
| Warm-Fisher-GRPO    | `/scratch/xie.yiyi/rl_casino_masks/llama8b_warm_grpo/warm_fisher_grpo.pt` |
| Warm-Magnitude-GRPO | `/scratch/xie.yiyi/rl_casino_masks/llama8b_warm_grpo/warm_magnitude_grpo.pt` |
| Warm-Momentum-GRPO  | `/scratch/xie.yiyi/rl_casino_masks/llama8b_warm_grpo/warm_momentum_grpo.pt` |

## Active (kept) subset

The 6-mask default in `probe_pair_masks.py::DEFAULT_MASKS`:

| Label | Path |
|-------|------|
| Oracle-DPO       | `/scratch/xie.yiyi/rl_casino_masks/llama8b/warm_magnitude_step50_sp97.5.pt` |
| Cold-CAV-DPO     | `/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_cav_dpo.pt` |
| Random-DPO       | `/scratch/xie.yiyi/rl_casino_masks/llama8b/random_baseline_dpo_sp97.5_seed42.pt` |
| Oracle-GRPO      | `/scratch/xie.yiyi/rl_casino_masks/llama8b_warm_grpo/warm_magnitude_grpo.pt` |
| Cold-CAV-GRPO    | `/scratch/xie.yiyi/rl_casino_masks/llama8b_cold/cold_cav_grpo.pt` |
| Random-GRPO      | `/scratch/xie.yiyi/rl_casino_masks/llama8b_warm_grpo/random_baseline_grpo_sp97.5_seed42.pt` |

Here "Oracle" is the warm-start magnitude mask — not a true oracle, but
the strongest reference line available (seen a small amount of task data
during warm-start, so it is the practical upper bound of masks that do
not require ground-truth importance).
