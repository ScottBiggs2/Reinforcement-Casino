"""Shared checkpoint / W&B helpers for dense and sparse GRPO entrypoints."""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Optional

import wandb
from transformers import TrainerCallback


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    pattern = os.path.join(output_dir, "checkpoint-*")
    dirs = [p for p in glob.glob(pattern) if os.path.isdir(p)]

    def step_key(path: str) -> int:
        name = os.path.basename(path)
        try:
            return int(name.split("-")[-1])
        except ValueError:
            return -1

    dirs.sort(key=step_key)
    return dirs[-1] if dirs else None


def resolve_resume_checkpoint(output_dir: str, resume: Optional[str]) -> Optional[str]:
    if not resume:
        return None
    r = resume.strip()
    if r.lower() == "auto":
        return find_latest_checkpoint(output_dir)
    if os.path.isdir(r):
        return r
    print(f"WARNING: resume path is not a directory: {r!r}; ignoring.")
    return None


def maybe_load_wandb_resume_env(run_dir: str, resume_ckpt: Optional[str]) -> None:
    if not resume_ckpt:
        return
    path = os.path.join(run_dir, "wandb_run_id.txt")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        run_id = f.read().strip()
    if run_id:
        os.environ.setdefault("WANDB_RESUME", "allow")
        os.environ["WANDB_RUN_ID"] = run_id


class WandbRunIdCallback(TrainerCallback):
    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir

    def on_train_begin(self, args, state, control, **kwargs) -> Any:
        if not getattr(state, "is_world_process_zero", True):
            return control
        if wandb.run is not None:
            path = os.path.join(self.run_dir, "wandb_run_id.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(wandb.run.id)
        return control


class RunManifestCallback(TrainerCallback):
    def __init__(self, run_dir: str, manifest: Dict[str, Any]) -> None:
        self.run_dir = run_dir
        self.manifest = manifest

    def on_train_begin(self, args, state, control, **kwargs) -> Any:
        if not getattr(state, "is_world_process_zero", True):
            return control
        os.makedirs(self.run_dir, exist_ok=True)
        path = os.path.join(self.run_dir, "run_manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)
        return control
