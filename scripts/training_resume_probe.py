#!/usr/bin/env python3
"""
Inspect HF Trainer checkpoint dirs for auto-resume orchestration.

Resolves checkpoints/ the same way as dense DPO, sparse DPO, dense GRPO, and sparse GRPO
when the same environment variables are set as the Slurm entrypoints.

Prints one JSON object to stdout with:
  global_step, target_steps, complete, resumable, checkpoints_dir, latest_checkpoint, error

Exit codes:
  0  Success (including "no checkpoints yet" — see JSON)
  1  Usage / resolution error
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Keep this script free of heavy imports (datasets, wandb): duplicate small helpers from
# grpo_checkpoint_utils / dataset_registry for CLI use on login nodes.

_DATASET_SANITIZED = {
    "tulu3": "tulu3",
    "light-r1": "light_r1",
    "math-step-dpo": "math_step_dpo",
    "math-220k": "math_220k",
    "codepref": "codepref",
}


def _dataset_sanitized(ds_key: str) -> str:
    if ds_key in _DATASET_SANITIZED:
        return _DATASET_SANITIZED[ds_key]
    s = ds_key.replace("/", "_").replace("-", "_").lower()
    s = "".join(c if c.isalnum() or c == "_" else "_" for c in s)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _find_latest_checkpoint(output_dir: str) -> Optional[str]:
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


def _sanitize_model_name(model_name: str) -> str:
    sanitized = model_name.replace("/", "_").replace("-", "_").lower()
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


def _resolve_checkpoints_dir(mode: str) -> Tuple[str, Optional[str]]:
    """Return (checkpoints_dir, error_message)."""
    override = os.environ.get("TRAINING_RESUME_CHECKPOINTS_DIR", "").strip()
    if override:
        return override, None

    if mode == "dense_dpo":
        train_base = os.environ.get("TRAIN_OUT_BASE", "")
        run_id = os.environ.get("PIPELINE_RUN_ID") or os.environ.get("RUN_ID", "")
        model = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        ds_key = os.environ.get("DPO_DATASET_KEY", "tulu3")
        if not train_base or not run_id:
            return "", "dense_dpo requires TRAIN_OUT_BASE and PIPELINE_RUN_ID or RUN_ID"
        sub = f"{_sanitize_model_name(model)}_{_dataset_sanitized(ds_key)}"
        return os.path.join(train_base, run_id, "checkpoints", sub), None

    if mode == "sparse_dpo":
        sparse_base = os.environ.get("SPARSE_OUT_BASE", "")
        run_id = os.environ.get("PIPELINE_RUN_ID") or os.environ.get("RUN_ID", "")
        mask_file = os.environ.get("PIPELINE_MASK_FILE", "")
        if not sparse_base or not run_id or not mask_file:
            return "", "sparse_dpo requires SPARSE_OUT_BASE, RUN_ID/PIPELINE_RUN_ID, PIPELINE_MASK_FILE"
        stem = os.path.splitext(os.path.basename(mask_file))[0]
        run_name = os.environ.get("SPARSE_DPO_RUN_NAME")
        if not run_name:
            run_name = f"fullpipe_sparse_{stem}_{run_id}"
        out_base = os.path.join(sparse_base, run_id, stem)
        return os.path.join(out_base, run_name, "checkpoints"), None

    if mode == "dense_grpo":
        base = os.environ.get("GRPO_DENSE_OUTPUT_BASE", "")
        model = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        ds_key = os.environ.get("GRPO_DATASET", "math-220k")
        slug = os.environ.get("GRPO_RUN_SLUG", "").strip()
        if not base:
            return "", "dense_grpo requires GRPO_DENSE_OUTPUT_BASE"
        if not slug:
            slug = f"{_sanitize_model_name(model)}_{_dataset_sanitized(ds_key)}_grpo_dense"
        return os.path.join(base, slug, "checkpoints"), None

    if mode == "sparse_grpo":
        base = os.environ.get("GRPO_SPARSE_OUTPUT_BASE", "")
        name = os.environ.get("GRPO_RUN_NAME", "")
        if not base or not name:
            return "", "sparse_grpo requires GRPO_SPARSE_OUTPUT_BASE and GRPO_RUN_NAME"
        return os.path.join(base, name, "checkpoints"), None

    return "", f"unknown mode: {mode}"


def _read_global_step(checkpoint_dir: str) -> Optional[int]:
    state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    if not os.path.isfile(state_path):
        return None
    try:
        with open(state_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and "global_step" in data:
        return int(data["global_step"])
    return None


def _manifest_target(run_dir: str) -> Optional[int]:
    path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("n_steps", "num_steps"):
        if key in m and m[key] is not None:
            try:
                return int(m[key])
            except (TypeError, ValueError):
                return None
    return None


def _resolve_target_steps(mode: str, checkpoints_dir: str, explicit: Optional[int]) -> int:
    if explicit is not None and explicit > 0:
        return explicit
    if mode in ("dense_dpo", "sparse_dpo"):
        v = os.environ.get("NUM_STEPS_DPO", "")
        if v.strip():
            return int(v)
    if mode in ("dense_grpo", "sparse_grpo"):
        v = os.environ.get("GRPO_TARGET_STEPS", "")
        if v.strip():
            return int(v)
    run_dir = os.path.dirname(checkpoints_dir)
    mt = _manifest_target(run_dir)
    if mt is not None:
        return mt
    raise ValueError(
        "Could not determine target steps: set --target-steps, or NUM_STEPS_DPO / GRPO_TARGET_STEPS, "
        "or ensure run_manifest.json exists under the run directory."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Probe HF checkpoint progress for auto-resume.")
    p.add_argument(
        "--mode",
        required=True,
        choices=("dense_dpo", "sparse_dpo", "dense_grpo", "sparse_grpo"),
    )
    p.add_argument("--target-steps", type=int, default=None, help="Override target max_steps.")
    p.add_argument(
        "--json-only",
        action="store_true",
        dest="json_only",
        help="Print compact JSON on one line.",
    )
    args = p.parse_args()

    err: Optional[str] = None
    checkpoints_dir, e = _resolve_checkpoints_dir(args.mode)
    if e:
        err = e
        out: Dict[str, Any] = {
            "mode": args.mode,
            "checkpoints_dir": checkpoints_dir or None,
            "global_step": None,
            "target_steps": None,
            "complete": False,
            "resumable": False,
            "latest_checkpoint": None,
            "error": err,
        }
        print(json.dumps(out, indent=2))
        sys.exit(1)

    try:
        target = _resolve_target_steps(args.mode, checkpoints_dir, args.target_steps)
    except ValueError as ex:
        out = {
            "mode": args.mode,
            "checkpoints_dir": checkpoints_dir,
            "global_step": None,
            "target_steps": None,
            "complete": False,
            "resumable": False,
            "latest_checkpoint": None,
            "error": str(ex),
        }
        print(json.dumps(out, indent=2))
        sys.exit(1)

    latest = _find_latest_checkpoint(checkpoints_dir)
    gs: Optional[int] = None
    if latest:
        gs = _read_global_step(latest)

    complete = gs is not None and gs >= target
    resumable = bool(latest) and not complete and gs is not None

    out = {
        "mode": args.mode,
        "checkpoints_dir": checkpoints_dir,
        "global_step": gs,
        "target_steps": target,
        "complete": complete,
        "resumable": resumable,
        "latest_checkpoint": latest,
        "error": None,
    }
    if args.json_only:
        print(json.dumps(out))
    else:
        print(json.dumps(out, indent=2))

    # Optional exit code contract for shell: TRAINING_RESUME_PROBE_EXIT_POLICY=1
    if int(os.environ.get("TRAINING_RESUME_PROBE_EXIT_POLICY", "0")) == 1:
        if complete:
            sys.exit(0)
        if resumable:
            sys.exit(10)
        if latest is None or gs is None:
            sys.exit(11)
        sys.exit(0)


if __name__ == "__main__":
    main()
