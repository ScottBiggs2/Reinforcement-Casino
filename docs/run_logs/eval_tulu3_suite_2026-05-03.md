# Run log — Tulu3 eval suite (5 sparse + 1 dense + 1 base)

**Date:** 2026-05-03
**Branch:** `irene-sparse-speed-ablation`
**Operator:** Irene (via Claude)
**Repo worktree on cluster:** `~/rc-sparse-speed`
**Eval driver:** `scripts/eval_tulu3_suite.sbatch` (new — parameterized wrapper around `src/evaluation/run_all_benchmarks.py`)

## What this is

End-to-end benchmark eval of every Tulu3-related Llama-3.1-8B-Instruct
checkpoint we've produced this week, plus the untrained base model as a
zero-training reference. Goal: see whether each sparse mask method recovers /
matches dense Tulu3 DPO on standard chat-LLM benchmarks.

## Submitted jobs

| TAG                          | Job ID  | Model path |
|------------------------------|---------|------------|
| `base_llama31_8b_instruct`   | 6525568 | `meta-llama/Llama-3.1-8B-Instruct` (HF hub, untrained reference) |
| `dense_dpo_tulu3_ckpt500`    | 6525569 | `/scratch/xie.yiyi/transfer_v1/dense_dpo_tulu3_llama8b_deltalog/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500` |
| `sparse_warm_mag_step200`    | 6525570 | `/scratch/xie.yiyi/transfer_v1/sparse_dpo_tulu3_warm_magnitude_step200/sparse_dpo_tulu3_warm_magnitude_step200_500steps/final_model` |
| `sparse_oracle_dpo_tulu3`    | 6525571 | `/scratch/xie.yiyi/transfer_v1/sparse_dpo_tulu3_oracle_dpo_tulu3/sparse_dpo_tulu3_oracle_dpo_tulu3_500steps/final_model` (in-task oracle) |
| `sparse_oracle_dpo_lightr1`  | 6525572 | `/scratch/xie.yiyi/transfer_v1/sparse_dpo_tulu3_oracle_dpo_lightr1/sparse_dpo_tulu3_oracle_dpo_lightr1_500steps/final_model` (cross-task DPO oracle) |
| `sparse_oracle_grpo_math`    | 6525573 | `/scratch/xie.yiyi/transfer_v1/sparse_dpo_tulu3_oracle_grpo_math/sparse_dpo_tulu3_oracle_grpo_math_500steps/final_model` (cross-objective oracle) |
| `sparse_random`              | 6525574 | `/scratch/xie.yiyi/transfer_v1/sparse_dpo_tulu3_random/sparse_dpo_tulu3_random_500steps/final_model` (random-mask baseline) |

## Pre-launch fixes

**chat_template patch (one-time):** All 6 saved checkpoints (5 sparse `final_model/`
+ dense `checkpoint-500`) ship the chat template as a separate `chat_template.jinja`
file. The eval-env transformers (4.46.3) does NOT auto-load the `.jinja` file —
it only reads `chat_template` from `tokenizer_config.json`. Without this, every
`--apply_chat_template` eval would crash with
`ValueError: Cannot use chat template functions because tokenizer.chat_template is not set`.

Patched by inlining `chat_template.jinja` content into each `tokenizer_config.json`
under the `chat_template` key, then verifying via `AutoTokenizer.apply_chat_template`.

This is the same bug Scott documented in `src/evaluation/apply_mask_and_save.py`
("Some transformers versions drop chat_template during save_pretrained").

## Eval pipeline (no code changes — same as `main` and `cav_fixes`)

- Driver: `src/evaluation/run_all_benchmarks.py` (byte-identical between `main` and `origin/cav_fixes`)
- Backend: vLLM (auto-detected; falls back to HF transformers if missing)
- Suite: **all 7 benchmarks** — `mmlu`, `math`, `gsm8k`, `coding`, `ifeval`, `squad`, `gpqa_diamond`
- Per-benchmark settings: lm-eval defaults, `--apply_chat_template`, `--trust_remote_code`, `--batch_size auto`
- No `--limit` (full eval set)

## Resource shape per job

| Param         | Value |
|---------------|-------|
| partition     | gpu |
| gres          | gpu:a100:1 |
| mem           | 128G |
| time          | 06:00:00 (under 8h school cap) |
| eval env      | `/scratch/biggs.s/conda_envs/rl_casino_eval` (Scott's, has lm-eval + vLLM) |

Total budget: 7 jobs × ≤6h on A100 = ≤42 GPU-hours upper bound; vLLM should
finish each in ~2–3h so realistic wall is ~15–20 GPU-hours.

## Outputs

- SLURM logs: `~/rc-sparse-speed/logs/eval_tulu3_<jobid>.{out,err}`
- Per-benchmark results: `~/rc-sparse-speed/results/eval_tulu3_<TAG>_<jobid>/{mmlu_results.json,math_results.json,...,all_benchmarks_summary.json}`

## Provenance — sparse training hparams (from earlier run logs)

- All 5 sparse runs: 97.5% sparsity, sparse_adamw, 500 steps, lr=5e-7, dpo_beta=0.1, eff batch 128, max_len 1024, gradient checkpointing on
- Mask sources:
  - `warm_mag_step200`: magnitude of cumulative deltas from dense Tulu3 deltalog re-run, steps 50/100/150/200
  - `oracle_dpo_tulu3`: ckpt-diff oracle from same dense Tulu3 run (in-task)
  - `oracle_dpo_lightr1`: ckpt-diff oracle from dense DPO Light-R1
  - `oracle_grpo_math`: ckpt-diff oracle from dense GRPO math220k
  - `random`: random binary mask at 97.5% sparsity, fixed seed

## Status (at submit)

```
$ squeue -u $USER  (subset)
JOBID    NAME        STATE    TIME LIMIT  REASON
6525568  eval_tulu3  PENDING  6:00:00     Priority
6525569  eval_tulu3  PENDING  6:00:00     Priority
6525570  eval_tulu3  PENDING  6:00:00     Priority
6525571  eval_tulu3  PENDING  6:00:00     Priority
6525572  eval_tulu3  PENDING  6:00:00     Priority
6525573  eval_tulu3  PENDING  6:00:00     Priority
6525574  eval_tulu3  PENDING  6:00:00     Priority
```

All 7 in queue at 2026-05-03 ~00:09 (cluster local). Reason: Priority — should
flow into the gpu partition as A100 nodes free up.

## Next session checklist

1. `squeue -u $USER` → confirm no PD/F states; check NODELIST for assignment
2. Tail one `out` to verify vLLM picked up successfully:
   `tail -50 ~/rc-sparse-speed/logs/eval_tulu3_<jobid>.out`
3. Aggregate across 7 result dirs into a comparison table:
   ```
   for d in ~/rc-sparse-speed/results/eval_tulu3_*; do
     echo "=== $d ==="; jq '. | to_entries[] | "\(.key): \(.value.results // .value)"' "$d/all_benchmarks_summary.json" 2>/dev/null
   done
   ```
4. Sanity check: `base_llama31_8b_instruct` numbers should roughly match
   published Llama-3.1-8B-Instruct numbers (e.g. MMLU ~69, ifeval prompt-strict ~76).
   If they don't, eval pipeline is mis-configured before reading sparse vs dense.
