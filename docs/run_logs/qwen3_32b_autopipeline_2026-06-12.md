# Qwen3-32B high-sparsity DPO — fully-automated requeue-chain run

**Created:** 2026-06-12 · **Owner:** Irene · **Branch:** `irene-sparse-speed-ablation` (on `cav_fixes`)
**Cluster worktree:** `~/rc-sparse-speed` · **Model:** `Qwen/Qwen3-32B` (instruct, NOT -Base)

## Goal
Run the dense → oracle-mask → sparse DPO transfer chain on Qwen3-32B to completion,
fully unattended, surviving the Discovery 8h walltime cap via self-resubmitting
checkpoint resume. Same algorithm + hparams as the 8B chain (Scott `cav_fixes`),
only backbone/parallelism/paths change.

## Why the previous attempts stalled
- `--requeue` does **not** fire on TIME LIMIT — an 8h wall hit is `CANCELLED`, not a
  requeue. So `sbatch --dependency=afterok` (old launcher) never advanced: p1 never
  exited 0, p2/p3 stayed PENDING forever.
- Throughput ~7 min/step (`device_map=auto` naive MP on 3×H200). 500 steps ≈ 58h →
  impossible in one 8h slot; needs ~10–11 slots per training phase.

## Orchestration (this run)
- **`scripts/qwen3_32b_advance.sh`** — stateless re-entrant advancer. Inspects
  `/scratch` checkpoints, submits the ONE next job, called from each job's tail.
  - squeue guard (no duplicate phase submits; excludes the calling job).
  - **progress/stall guard**: if a training phase makes no checkpoint progress over
    2 consecutive slots (resume silently broken / idle-killed during load), writes
    `/scratch/$USER/transfer_v1/PIPELINE_STALLED` and halts instead of burning days.
  - state: `/scratch/$USER/transfer_v1/pipeline_state.env`; human log:
    `docs/run_logs/qwen3_32b_autopipeline.log`; done sentinel: `PIPELINE_DONE`.
- Each training sbatch wraps python in `timeout 27000s` (7.5h of the 8h wall) so it
  exits cleanly ~30 min early and the tail can resubmit; `--resume_from_checkpoint auto`.
- Launcher `qwen3_32b_transfer_launch.sh` just kicks off via advance.sh.

## Verified before launch
- `checkpoint_diff_mask_finder.load_state_dict` loads a HF checkpoint **dir** (p1's
  checkpoint-500) via `AutoModelForCausalLM.from_pretrained`. ✓
- p1 (`DPO_train.py`) + p3 (`sparse_dpo_efficiency.py`) use plain `DPOConfig` (no
  `save_only_model` on the non-FSDP path) → `optimizer.pt` saved → checkpoints are
  truly resumable. p3 has an explicit `if not resume_ckpt:` branch. ✓
- **Untested (covered by stall guard, per decision to skip a separate probe):**
  resume of adamw_8bit (p1) / custom SparseAdamW (p3) optimizer state across the
  device_map MP layout.

## Hyperparameters (both p1 dense & p3 sparse, unless noted)
| param | value |
|---|---|
| backbone | Qwen/Qwen3-32B |
| dataset | light-r1 |
| target steps | 500 |
| eff. batch | 128 (bs 2 × grad_accum 64, single MP process) |
| learning_rate | 5e-7 |
| warmup_ratio | 0.1 |
| lr_scheduler | linear |
| dpo_beta | 0.1 |
| max_length / max_prompt_length | 1024 / 1024 |
| gradient_checkpointing | on |
| parallelism | device_map=auto across 3×H200 (`--gres=gpu:h200:3`) |
| optimizer | p1: adamw_8bit · p3: sparse_adamw |
| sparsity (p2 oracle, p3) | 97.5% (min_layer_keep_ratio 0.0025) |
| **save_steps** | **10** (was 50; smaller so each slot crosses save boundaries) |
| **save_total_limit** | **3** (was 12; ~390 GB on /scratch vs 1.5 TB) |
| timeout per slot | 27000 s (7.5h) |

## Expected cost
~45 net steps/slot after model load → **p1 ≈ 11 slots, p3 ≈ 11 slots + p2 mask ≈
~8–9 days wall-clock**, fully unattended.

## Outputs
- dense: `/scratch/xie.yiyi/transfer_v1/dense_dpo_light_r1_qwen3_32b/`
- oracle mask: `/scratch/xie.yiyi/transfer_v1/oracle_masks_qwen3_32b/oracle_dpo_light_r1_step500_sp97.5.pt`
- sparse: `/scratch/xie.yiyi/transfer_v1/sparse_dpo_light_r1_qwen3_32b_oracle_step500/`
- wandb project: `rl_casino_transfer_v1`

## How to monitor / stop
- `squeue -u $USER` ; `tail -f docs/run_logs/qwen3_32b_autopipeline.log`
- Stop: scancel the running job **and** `touch /scratch/$USER/transfer_v1/PIPELINE_STALLED`.
  Remove the sentinel to resume.
