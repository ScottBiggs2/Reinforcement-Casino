# Cold-start masks: `RL-irene` vs `scott-dev` (summary)

When comparing CAV/SNIP/Fisher masks with Irene’s branch, the **scoring code paths** (`SNIPScorer`, `CAVProbeScorer`) are largely shared; differences that change **inputs to those scorers** matter most.

## `src/cold_start/inference_mask_finder.py`

| Area | `origin/RL-irene` | Current branch (`scott-dev`) |
|------|-------------------|-------------------------------|
| DPO calibration data | Manual field parsing (`_msg_to_text`, conversation heuristics) | [`normalize_dpo_record`](src/utils/dpo_text_normalize.py) + [`resolve_hf_dataset_id`](src/utils/dataset_registry.py) — aligned with DPO training |
| Default DPO dataset constant | `qihoo360/Light-R1-DPOData` | Registry key `tulu3` (resolved via registry) |
| GRPO calibration | `open-r1/OpenR1-Math-220k` | Unchanged |

**Implication:** CAV/SNIP scores on **DPO-mode** cold start can differ between branches because **prompt/chosen/rejected text** can differ after normalization, not necessarily because the SNIP/CAV math changed.

## Other `src/cold_start/` files

`git diff origin/RL-irene..HEAD -- src/cold_start/` also shows large updates in plotting/export utilities (`plot_layer_metrics_csv.py`, `export_layer_metrics_csv.py`, `mask_to_cka.py`) on `scott-dev` — tooling only, not the core mask scores.

## Practical guidance

- For **GRPO-mode** masks (`--mode grpo`), prefer comparing runs that use the **same** `OpenR1-Math-220k` loading path; both branches target the same HF id.
- For **DPO-mode** masks, record whether masks were built with **Irene-style** raw parsing or **`normalize_dpo_record`** (`scott-dev`) in your experiment notes.
