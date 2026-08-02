# Table-1 eval suite — final numbers (AICR 2026-08-01)

Metrics: MMLU acc / MATH exact-match macro / GSM8K strict EM / HumanEval+MBPP pass@1 / IFEval prompt-strict / SQuAD contains / GPQA-Diamond acc. Source: docs/run_logs/eval_tulu3_suite_rerun_aicr_2026-07-31.md

| Arm | MMLU | MATH | GSM8K | HumanEval | MBPP | IFEval | SQuAD | GPQA-D |
|---|---|---|---|---|---|---|---|---|
| Base Llama-3.1-8B-Instruct | 0.687 | 0.403 | 0.820 | 0.683 | 0.584 | 0.739 | 0.688 | 0.263 |
| Dense DPO (Tulu3) | 0.687 | 0.407 | 0.818 | 0.671 | 0.586 | 0.749 | 0.690 | 0.273 |
| Sparse oracle (DPO Tulu3) | 0.687 | 0.409 | 0.817 | 0.671 | 0.584 | 0.749 | 0.689 | 0.273 |
| Sparse oracle (DPO Light-R1) | 0.687 | 0.407 | 0.817 | 0.671 | 0.584 | 0.750 | 0.688 | 0.273 |
| Sparse oracle (GRPO math) | 0.687 | 0.407 | 0.816 | 0.671 | 0.586 | 0.741 | 0.689 | 0.268 |
| Sparse warm magnitude@200 | 0.687 | 0.407 | 0.816 | 0.671 | 0.588 | 0.749 | 0.689 | 0.268 |
| Sparse random (control) | 0.687 | 0.405 | 0.821 | 0.683 | 0.586 | 0.743 | 0.688 | 0.263 |
| LoRA r64 lr1e-4 (Light-R1) | 0.686 | 0.085 | 0.810 | 0.591 | 0.596 | 0.743 | 0.668 | 0.278 |
| LoRA r64 lr5e-6 (Light-R1) | 0.686 | 0.388 | 0.831 | 0.671 | 0.578 | 0.763 | 0.681 | 0.273 |
