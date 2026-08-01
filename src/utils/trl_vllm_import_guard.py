"""
TRL 0.24+ may import vLLM when the package is present (optional fast GRPO generation).

A vLLM wheel built for a different PyTorch than your env fails at import time
(`ScalarType`, undefined symbols in `vllm._core_C`) and breaks **all** GRPO imports,
even with `use_vllm=False`.

**Default:** behave as if vLLM were absent (HF generate for rollouts — our scripts).

**Use vLLM:** install a TRL-compatible build (see TRL docs; e.g. `vllm==0.10.2`) matching
your torch, then run with `TRL_SKIP_VLLM_IMPORT=0`.

**Alternative:** `pip uninstall vllm` in the training env.

See: https://github.com/huggingface/trl/issues (GRPO + vLLM import)
"""

from __future__ import annotations

import os


def apply_trl_vllm_skip() -> None:
    # Default "1" = skip broken/partial vLLM installs; set TRL_SKIP_VLLM_IMPORT=0 to use vLLM.
    v = os.environ.get("TRL_SKIP_VLLM_IMPORT", "1").lower()
    if v in ("0", "false", "no", "off"):
        return
    import trl.import_utils as trl_iu

    trl_iu._vllm_available = False
