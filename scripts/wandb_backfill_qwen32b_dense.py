#!/usr/bin/env python
"""Backfill the full step-1..500 dense Qwen3-32B DPO curve into wandb.

The live run 97idxxzx (project qwen3) starts at step 251 because the AICR job
resumed from the Explorer checkpoint-250 as a fresh wandb run; steps 1-250 were
never logged online (the June Explorer attempts all crashed pre-logging or are
tagged dead-attempt). checkpoint-500/trainer_state.json carries the complete
log_history (500 entries, steps 1-500, verified 2026-08-01), so this script
replays it into ONE new run with the same metric names the live runs use
(train/<key> + train/global_step).

Run on the AICR login node (wandb creds in ~/.netrc):
    python wandb_backfill_qwen32b_dense.py
"""

import json

import wandb

ENTITY = "xxiellan-northeastern-university"
PROJECT = "qwen3"
SRC_RUN = "97idxxzx"
TRAINER_STATE = (
    "/scratch/xie_yiyi_neu/transfer_v1/dense_dpo_light_r1_qwen3_32b/"
    "checkpoints/qwen_qwen3_32b_light_r1/checkpoint-500/trainer_state.json"
)

state = json.load(open(TRAINER_STATE))
entries = [e for e in state["log_history"] if "loss" in e and "step" in e]
entries.sort(key=lambda e: e["step"])
steps = [e["step"] for e in entries]
assert steps[0] == 1 and steps[-1] == 500, f"unexpected step range {steps[0]}..{steps[-1]}"
print(f"log_history: {len(entries)} entries, steps {steps[0]}..{steps[-1]}")

api = wandb.Api()
src = api.run(f"{ENTITY}/{PROJECT}/{SRC_RUN}")
cfg = {k: v for k, v in src.config.items() if not k.startswith("_")}
cfg["backfill_note"] = (
    "Full 1-500 curve replayed from checkpoint-500 trainer_state.json. "
    "Steps 1-250 trained on Explorer 3xH200 (June 2026), steps 251-500 on "
    f"AICR 3xB200 (resume run {SRC_RUN}). See "
    "docs/run_logs/qwen3_32b_aicr_resume_2026-07-29.md for provenance."
)

run = wandb.init(
    entity=ENTITY,
    project=PROJECT,
    name="dense_dpo_qwen3_32b_light_r1_scott_full",
    tags=["dense", "qwen3_32b", "qwen3_family", "backfilled_full_curve",
          "two_cluster_provenance"],
    config=cfg,
    notes=cfg["backfill_note"],
)
for e in entries:
    step = e["step"]
    payload = {f"train/{k}": v for k, v in e.items() if k != "step"}
    payload["train/global_step"] = step
    run.log(payload, step=step)
run.finish()
print(f"backfilled run: {run.id} ({run.name})")

# Mark the live resume run as the partial segment so nobody plots it as full.
src.name = "(partial, live 251-500) dense_dpo_qwen3_32b_light_r1_scott_full"
src.tags = list(set(src.tags) | {"partial-resume-segment", "qwen3_32b", "qwen3_family"})
src.update()
print(f"renamed {SRC_RUN} -> {src.name}")
