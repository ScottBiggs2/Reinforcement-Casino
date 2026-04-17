"""Default scratch layout for training outputs and HF caches.

Override with env var ``RL_CASINO_SCRATCH_ROOT`` (e.g. ``/scratch/netid`` on your cluster).
Falls back to ``/scratch/$USER`` when unset.
"""

from __future__ import annotations

import os


def scratch_root() -> str:
    return os.environ.get("RL_CASINO_SCRATCH_ROOT", f"/scratch/{os.environ.get('USER', 'unknown')}")


def default_rl_casino_outputs() -> str:
    return os.path.join(scratch_root(), "rl_casino_outputs")


def default_hf_datasets_cache() -> str:
    return os.path.join(scratch_root(), "hf_cache", "datasets")


def default_grpo_dense_outputs() -> str:
    """Default base directory for dense GRPO runs (per-run subdirs live underneath)."""
    return os.path.join(scratch_root(), "rl_casino_grpo", "dense")


def default_grpo_sparse_outputs() -> str:
    """Default base directory for sparse GRPO runs."""
    return os.path.join(scratch_root(), "rl_casino_grpo", "sparse")
