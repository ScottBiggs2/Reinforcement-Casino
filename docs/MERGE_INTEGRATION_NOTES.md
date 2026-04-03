# Integration merge: scott-dev + RL-irene (2026-04-03)

## Git state

- **Integration branch:** `integrate/scott-irene-2026` (contains `origin/main` fast-forward to `origin/scott-dev`, then merge commit for `origin/RL-irene`).
- **Remote:** branch and tags pushed to `origin`.

## Rollback tags (on `origin`)

| Tag | Points to | Use |
|-----|-----------|-----|
| `pre-merge-main` | `origin/main` before integration | Reset `main` or compare baselines |
| `pre-merge-scott` | `origin/scott-dev` tip | Restore scott line |
| `pre-merge-irene` | `origin/RL-irene` tip | Restore Irene line |
| `pre-merge-before-RL-irene` | Integration branch after scott-dev FF, before RL-irene merge | `git reset --hard pre-merge-before-RL-irene` on integration branch to drop only the RL-irene merge |
| `post-merge-RL-irene-into-integration` | Tip of `integrate/scott-irene-2026` after merge | Anchor before merging to `main` |

### Rollback commands

- **Abandon integration branch only (keep main untouched):**  
  `git checkout main && git branch -D integrate/scott-irene-2026`  
  Recreate from origin if needed: `git fetch origin && git checkout -b integrate/scott-irene-2026 origin/integrate/scott-irene-2026`

- **Undo only the RL-irene merge on the integration branch:**  
  `git checkout integrate/scott-irene-2026 && git reset --hard pre-merge-before-RL-irene`

- **After `main` has been updated from integration and you need to revert the merge on main:**  
  `git revert -m 1 <merge_commit_sha_on_main>` then push.

## Resolution choices (research / ops)

- **Pipeline:** Kept scott-dev `scripts/run_full_pipeline.sh` + `pipeline_common.sh` (single-allocation pipeline). RL-irene’s older monolithic inline script was not merged.
- **Cold-start scripts:** Kept `cold_mask_finder.py`, `cav_cold_mask_finder.py`, and `generate_random_mask.py` so `pipeline_common.sh` and legacy Slurm paths keep working. `inference_mask_finder.py` is extended per RL-irene and documented as preferred in `readme.md`.
- **`inference_mask_finder` CLI:** Shell helpers use `--method`, `--sparsity`, `--n_samples` (not `--sparsity_percent` / `--min_layer_keep_ratio` on the unified script; the latter are not in that CLI).
- **`GRPO_timing_baseline.py`:** Took RL-irene version (alignment with `sparse_grpo_bsr` reward/dataset story).
- **`mlp_only`:** argparse kept with default **False** and help text on sparse training scripts.

## Follow-ups

- Optional: add `SCRATCH_USER_ROOT` indirection in `pipeline_common.sh` for multi-cluster paths.
- Run a full pipeline or `verify_grpo_training` smoke on the cluster before merging `integrate/scott-irene-2026` → `main`.
