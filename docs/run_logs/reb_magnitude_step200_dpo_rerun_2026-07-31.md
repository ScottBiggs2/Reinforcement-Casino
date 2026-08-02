# Rebuttal-e1-e2 magnitude-step200 DPO rerun — Light-R1 + Tulu3

**Date:** 2026-07-31 · **Branch (local):** `irene-rebuttal-lora` · **Cluster:** AICR (b200)
**Request:** Irene, 2026-07-31 — rerun the DPO magnitude-rewinding step-200 arms for
Light-R1 and Tulu3, with masks built **only** by the `rebuttal-e1-e2` branch's magnitude
method, logging to wandb **entity `xxiellan-northeastern-university`, project
`huggingface`** (https://wandb.ai/xxiellan-northeastern-university/huggingface).

## 1. What "rebuttal-e1-e2 method" means here

Masks are built by the staged copy of the `rebuttal-e1-e2` branch at
`$W/rebuttal_e1e2_mask` (commit `32f99dee49f09a4b0dd9e564b086e5ed49a66bcf`), via
`src/warm_start/even_better_mask_finder.py --method magnitude`. Differences vs the
submitted-figure masks (multi_seed_mask_gen on `irene-rebuttal-lora`):

1. Scoring restricted to 2-D weight tensors with exact-equality tie dedup
   (`_select_2d_weight_names`) — the bf16 fingerprint-collision fix.
2. No score jitter, no `--seed`: masks are deterministic by construction.

Unchanged: sparsity 97.5 %, `min_layer_keep_ratio 0.0025`, global top-k, CPU scoring,
tie-break noise seed 42. Score = Σ_{t logged, t≤200} |W_t − W_0| from the dense DPO
delta logs.

## 2. Mask stage

| side | delta source | job | script |
|---|---|---|---|
| Light-R1 | `$S/transfer_v1/dense_dpo_lightr1_llama8b/deltas/meta_llama_llama_3_1_8b_instruct_light_r1` (deltas 50/100/150/200/500 + base) | **243170** `reb_mag_lr1` (launched by the earlier session; builds step 50/100/200) | `$W/aicr_reb_mag_masks_lightr1.sbatch` |
| Tulu3 | `$S/transfer_v1/dense_dpo_tulu3_llama8b_deltalog/deltas/meta_llama_llama_3_1_8b_instruct_tulu3` — **streamed from Explorer 2026-07-31 ~17:10** via `~/xfer_tulu3_deltas.sh` (chunked-dd recipe, 5 files × ~30 GB: base + deltas 50/100/150/200) | **243209** `reb_mag_t3` (polls for exact source byte sizes, then builds step **200 first**, then 50/100) | `$W/aicr_reb_mag_masks_tulu3.sbatch` |

Mask outputs: `$S/rebuttal_e1e2_magnitude/masks/reb_magnitude_dpo_{lightr1,tulu3}_step200_sp97.5.pt`

Exact source sizes gating the Tulu3 mask job (sequential chunked-dd ⇒ exact size ==
complete): base 32121142092 · d50 32121143280 · d100/d150/d200 32121143577 bytes.

## 3. Training stage (the two runs requested)

Trainer: `src/full_training/sparse_dpo_efficiency.py` — hyperparameters verbatim from
`aicr_p3_sparse_dpo.sbatch` / Explorer `transfer_p3_sparse_dpo_*.sbatch` (Scott spec,
identical to the previous `warm_magnitude_step200` arms):

- model `meta-llama/Llama-3.1-8B-Instruct`, from base (magnitude **rewinding**: weights
  start at θ⁰, only mask-selected coords train)
- `n_steps 500`, `batch_size 2 × grad_accum 64` (eff. 128), `lr 5e-7`,
  `warmup_ratio 0.1`, `weight_decay 0.0`, `max_length 1024`, `max_prompt_length 1024`,
  `dpo_beta 0.1`, `optimizer sparse_adamw`, `save_steps 50`, gradient checkpointing,
  `resume_from_checkpoint auto`, 1× B200.

| run | job | dependency | dataset | wandb run name |
|---|---|---|---|---|
| Light-R1 | **243210** `reb_mag200_lr1` | afterok:243170 | `light-r1` | `sparse_dpo_lightr1_reb_magnitude_step200_500steps` |
| Tulu3 | **243211** `reb_mag200_t3` | afterok:243209 | `tulu3` | `sparse_dpo_tulu3_reb_magnitude_step200_500steps` |

Scripts: `$W/aicr_reb_dpo_mag200_{lightr1,tulu3}.sbatch`.
Outputs: `$S/rebuttal_e1e2_magnitude/sparse_dpo_{lightr1,tulu3}_reb_magnitude_step200/`.

Both jobs `export WANDB_PROJECT=huggingface` and
`WANDB_ENTITY=xxiellan-northeastern-university`; the AICR working copy already carries
the 2026-07-30 fix making `sparse_dpo_efficiency.py` respect `WANDB_PROJECT`
(migration log §4b), and `~/.netrc` on AICR has the wandb credential, so
`WANDB_MODE=online`.

## 3b. Supplementary arms for the 4-arm figure (added 2026-07-31 evening)

Irene's follow-up: all four arms (dense / magnitude / oracle / random, ρ=97.5) for both
datasets must live in wandb project `huggingface` and reproduce the submitted loss
figure. Inventory of usable **recent** runs found: light-r1 oracle `qbclyo1k` and dense
`2antafyb` were in `rl_casino_transfer_v1` (left untouched — reruns launched instead so
the ρ-sweep project stays intact); tulu3 oracle sp97.5 `a0gb8auf` is **already in
`huggingface`** (07-28, B200) and is reused. Missing entirely: both random arms and a
recent tulu3 dense. Launched (same Scott-spec hparams as §3; scripts
`$W/aicr_reb_dense.sbatch` + `$W/aicr_reb_sparse_arm.sbatch`):

| job | arm | mask | wandb run name |
|---|---|---|---|
| 244471 | dense light-r1 | — | `dense_dpo_lightr1_500steps_b200` |
| 244472 | dense tulu3 | — | `dense_dpo_tulu3_500steps_b200` |
| ~~244473~~ | ~~oracle light-r1~~ | — | **CANCELLED 07-31 night as a duplicate** of sweep run `qbclyo1k` (identical mask/hparams/hardware, both B200); its wandb run `qdugjd31` deleted, partial outputs removed. The figure reads this oracle curve cross-project from `rl_casino_transfer_v1`. |
| 245164 | oracle tulu3 (resubmit of cancelled 245142) | `oracle_dpo_tulu3_step500_sp97.5_src-deltalog.pt` | `sparse_dpo_tulu3_oracle_step500_sp97.5_500steps_b200` — hardware audit showed the sweep run `a0gb8auf` ran on Explorer **H200** (host d4052, 07-28, pre-migration), unlike the 7 other B200 arms, so this rerun is a hardware-consistency arm, not a duplicate. |
| 244474 | random light-r1 | `random_baseline_lightr1_sp97.5_seed42.pt` | `sparse_dpo_lightr1_random_sp97.5_seed42_500steps_b200` |
| 244475 | random tulu3 | `random_baseline_tulu3_sp97.5_seed42.pt` (copied from Explorer 2026-07-31, 8,030,364,741 B verified) | `sparse_dpo_tulu3_random_sp97.5_seed42_500steps_b200` |

Dense reruns use `DPO_train.py` (torchrun 1×B200) with default delta logging (single
step-50 snapshot); full delta sets already exist from the May runs. Figure rebuild:
`scripts/plot_reb_dpo_4arm_loss.py` (pulls the eight runs from wandb, original styling).
The May-era arms (`lamw769g`, `cwkto2y4`, `sahpme5c`, `ginyus8t`, `4i2u6d7p`,
`8ev25wv6`, …) remain in `huggingface` as the original-figure fallback.

## 4. Provenance / caveats

- Tulu3 dense deltalog run: Explorer
  `/scratch/xie.yiyi/transfer_v1/dense_dpo_tulu3_llama8b_deltalog` (May 2 dense run,
  Scott-spec convergence, canonical since the 2026-07-27 sweep log).
- Expected runtime ≈ 3.7 h per arm on B200 (matches the ρ-sweep arms).
- The earlier probe jobs (243171 `reb_probe_lr1`, and the not-yet-submitted
  `aicr_reb_mag_probe_tulu3.sbatch`) are a separate figure pipeline; the Tulu3 probe
  additionally needs the Tulu3 checkpoint-500, which was **not** transferred here.
