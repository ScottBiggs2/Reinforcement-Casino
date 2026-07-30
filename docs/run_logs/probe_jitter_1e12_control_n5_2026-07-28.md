# Jitter-magnitude control at `jitter_rel=1e-12` — n=5 seeds, Light-R1

**Date launched:** 2026-07-28
**Branch:** `irene-rebuttal-lora`
**Requested by:** Irene ("别的不改,把 noise 改成 1e-12,再跑 5 个 seed 重新生成图二")
**Jobs (live):** stage1 masks `8814942` (RUNNING on c2204, v100-pcie node, CPU
scoring) → stage2 probe `8814991` (multigpu / h200, afterok). A duplicate stage2
`8814998` was created by a retried `sbatch` and is pending disposal.
**Superseded submissions:**
- `8810823`/`8810824` — short, 256G, 8h. Unschedulable: `short`'s available nodes
  are mostly 62 GB, and the sibling 1e-3 job at the same ask was quoted a start
  time of 2026-07-30T13:20.
- `8814204`/`8814205` — multigpu/h200, 80G, GPU scoring. Every H200 had all 8 GPUs
  allocated; only d4055 had a free card, with 84 GB allocatable against an 80 GB
  ask. Never started.

**Scripts:** `n5_j1e12_stage1_masks_v100.sbatch`, `n5_j1e12_stage2_probe_h200.sbatch`,
`n5_j1e12_stage3_plot.py` (staged in `~/diag_tmp` on Discovery)
**Output root:** `/scratch/xie.yiyi/mask_seed_variance_n5_j1e12/`

## Goal

Single-knob control against the n=5 run at `jitter_rel=1e-3`
(jobs `8808817`/`8808818`, [probe_jitter_variance_n5_2026-07-28.md](probe_jitter_variance_n5_2026-07-28.md)):
**is the error bar on the Oracle/Magnitude panels of `zero_out_heatmap_lightr1_v5.png`
an artifact of the jitter magnitude?**

Everything else is held fixed — same model, checkpoint, delta log, sparsity,
floor, seeds, pooling, mask script, probe script and probe hyperparameters.
Only `--jitter_rel` changes.

## The one changed parameter

| | 1e-3 run (8808817) | this run (8810823) |
|---|---|---|
| `jitter_rel` | `1e-3` | **`1e-12`** |

`eps = std(score_of_tensor) * jitter_rel`, applied per tensor in
[`multi_seed_mask_gen.py:_jitter_and_save`](../../src/warm_start/multi_seed_mask_gen.py#L52-L86),
followed by a global top-k.

## Prediction (recorded before results)

Scores are float32 (relative ulp ≈ 1.2e-7). At `jitter_rel=1e-12` the
perturbation sits roughly five orders of magnitude below the ulp at the score
scale, so it should be lost entirely to rounding for every score of non-trivial
magnitude. Expected:

1. The five seeds produce **bit-identical** Oracle and Magnitude masks
   (Jaccard 1.000000, 0 tensors differing).
2. Oracle / Magnitude panels show **sd exactly 0**; `n5_j1e12_stage3_plot.py`
   drops the `±` annotation automatically when `max(sd) < 1e-12`.
3. Panel centre values return to the un-jittered values from the May-6 v3 run
   — e.g. Oracle syntax `0.31 0.36 0.36 0.33 0.40 0.42 0.40 0.30 0.32`.
4. Random keeps a real `±` (it is a genuine mask resample via
   `torch.manual_seed(seed)`, unaffected by jitter), now over 5 seeds instead of 3.

Known edge case that could produce a non-zero sd: scores that are *exactly* 0
(unchanged weights in a bf16 checkpoint) can still absorb a 1e-15 perturbation
because denormals are representable. Those sit at the bottom of the global
ranking and should only matter where the `min_layer_keep_ratio` floor is
selecting inside an all-zero tensor. Stage 1 measures this directly.

If prediction (1) fails, the jitter is doing something other than what the
formula implies and the 1e-3 result needs re-examination rather than the figure.

## Hyperparameters

### Mask generation (`multi_seed_mask_gen.py`)
| Param | Value |
|---|---|
| Backbone | `meta-llama/Llama-3.1-8B-Instruct` |
| Final ckpt | `transfer_v1/dense_dpo_lightr1_llama8b/.../checkpoint-500` |
| Delta log | `transfer_v1/dense_dpo_lightr1_llama8b/deltas/meta_llama_llama_3_1_8b_instruct_light_r1` |
| Sparsity | 97.5% |
| `min_layer_keep_ratio` | 0.0025 |
| **`jitter_rel`** | **1e-12** |
| Pooling | `local_pool=False` (global top-k), hardcoded |
| Seeds | 42, 43, 44, 45, 46 |
| Magnitude `target_steps` | 50, 100, 200 |
| Score device | **cpu** (`--force_cpu` + `RL_CASINO_WARM_MASK_SCORE_DEVICE=cpu`) — identical to the 1e-3 run |
| Tie-break noise | `add_tie_break_noise=True`, scale `1e-6`, seed hardcoded 42 (unchanged, constant across seeds) |

**Placement note (scheduling, not science).** Three submissions were needed to
land this job; the science is unchanged across all of them, only the node is.

1. `short`, 256 GB, CPU scoring — correct config, unschedulable. `short`'s
   available nodes are mostly 62 GB; only d0142 had enough free.
2. `multigpu`/H200, 80 GB, **GPU scoring** — an H200 node's free host RAM was
   40–90 GB, below the ~104 GB CPU-scoring peak, so scoring was moved to the GPU
   to cut the peak to ~64 GB. Never started: all four H200 nodes had 8/8 GPUs
   allocated except d4055 (1 free card, 84 GB allocatable vs an 80 GB ask).
3. **`multigpu`/v100-pcie (c2204), 200 GB, CPU scoring** — the four V100 nodes
   were fully idle with 480 GB each. This job is pure CPU tensor work, so the GPU
   type is irrelevant; only the node's RAM matters. Started immediately.

Host-RAM peak by scoring device, for the record:

| Stage | CPU scoring | GPU scoring |
|---|---|---|
| load 2 state dicts + build fp32 scores | 64 GB host | 64 GB host |
| per-seed jitter copy | +32 GB host | in VRAM |
| chunked global top-k (`mask_utils.py:600-612` forces the selector to CPU at 8B params) + mask | +40 GB host | 40 GB host |
| **host peak** | **~104 GB** | **~64 GB** |

`--force_cpu` is **required** on a GPU node: `run_oracle()` ignores
`RL_CASINO_WARM_MASK_SCORE_DEVICE` and would otherwise move 32 GB of fp32 scores
onto a 16/32 GB V100. On the CPU-only `short` node the 1e-3 sibling targets,
`torch.cuda.is_available()` is False and it takes the same branch, so the two
runs are byte-identical on the mask path. A keep-alive sidecar
(AGENT_PLAYBOOK §7b) holds the idle GPU against RC's 15-minute reaper.

### Random baseline
| Param | Value |
|---|---|
| Seeds 42–44 | reused from `transfer_v1/oracle_masks_llama8b/` |
| Seeds 45–46 | generated into **this run's** `masks/` dir, not the canonical dir, to avoid racing the concurrently-queued 1e-3 stage1 (`8808817`) which writes the same filenames to `transfer_v1/oracle_masks_llama8b/` |
| Generator | `src/utils/generate_random_mask.py`, `random_global`, 97.5% / 0.0025 |

Stage 2 resolves the random path as "canonical if present, else this run's dir".

### Probe (`probe_pair_masks.py`) — byte-identical to May-6 v3, May-27 v5 and the 1e-3 n=5 run
| Param | Value |
|---|---|
| Probed model | dense Light-R1 DPO checkpoint-500 |
| `patch_mode` | `zero_out` (`w' = w_ft * M`) |
| `layer_stride` | 4 (layers 0,4,8,12,16,20,24,28,31) |
| `cv_folds` | 3 |
| `pairs_per_pos` | 2 |
| `holdout_frac` | 0.2 with `--use_holdout_as_test` |
| `probe_C` | 0.01 |
| `batch_size` / `max_length` | 8 / 256 |
| `--no_preference` | yes (4 benchmark probes) |
| Probe seed | 42, fixed internally |

## Reproduction gate

`n5_j1e12_stage3_plot.py` prints FAIL if Baseline is not identical across all
five seeds. Random must also match the v3/v5 values. If either fails the run is
not comparable to figure 2 and should be discarded.

## Defects inherited from the 1e-3 run (NOT fixed here — single knob only)

Carried over verbatim from [probe_jitter_variance_n5_2026-07-28.md](probe_jitter_variance_n5_2026-07-28.md)
so this figure is not over-read:

1. **The random control omits 65 tensors.** `generate_random_mask.py:60` skips
   `param.dim() != 2`, so all 64 layer-norms plus `model.norm` are absent from
   the random mask and are therefore left fully intact by `apply_mask`, while
   Oracle/Magnitude have theirs reduced to the keep floor. Measured activation
   scale gap: ~1400×. Random is not a matched control.
2. **`zero_out` is destructive** — it zeroes 97.5% of the base model too, so
   every masked panel measures survival of a lobotomy, not where the DPO update
   localises. `delta_only` / `anti_delta_only` were not run.
3. **Oracle and Magnitude-200 are effectively the same mask** (Jaccard 0.998 on
   the un-jittered set); their agreement is not independent corroboration.

## Outputs

```
/scratch/xie.yiyi/mask_seed_variance_n5_j1e12/
├── masks/                      # oracle + magnitude x 5 seeds, random 45/46
└── probes/
    ├── lightr1/seed{42..46}/probe_pair_results.json
    ├── zero_out_heatmap_lightr1_n5_j1e12.png|pdf
    └── n5_j1e12_summary.json
```
