"""
Slurm / NFS: wandb may wrap ``sys.stdout`` (console_capture); writes can raise OSError errno 116.
Use this for training-side diagnostics so logs go to stderr by default.
"""

from __future__ import annotations

import sys
from typing import Any


def slurm_safe_print(*args: Any, **kwargs: Any) -> None:
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)
