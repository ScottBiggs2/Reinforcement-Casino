#!/bin/bash
# Round-4 resubmission of the Tulu3 eval suite (NeurIPS #29841 rebuttal Table 1).
# Round 3 (246641-49) all hit the 8h TIMEOUT: concurrent dataset downloads into
# the shared /scratch hf_cache deadlocked on datasets' filelocks over autofs.
# Round 4 runs fully offline against the prefetched cache
# (scripts/aicr_prefetch_eval_datasets.py) and only the benchmarks each arm is
# still missing, writing into the arm's existing round-3 results dir.
#
# Run on login.aicr.ai from the repo root:  bash scripts/aicr_submit_eval_round4.sh
set -euo pipefail

W=/work/neu/p2026_0038_neu/xie_yiyi
cd "$W/rc-sparse-speed"
S=/scratch/xie_yiyi_neu
OUT=$S/rebuttal_analysis/eval_tulu3
T1=$S/transfer_v1

# gpqa_diamond excluded: Idavidrein/gpqa is gated and this HF account has no
# access yet (prefetch 2026-08-01); run it as a tiny follow-up once granted.
FULL="math gsm8k coding ifeval squad"      # mmlu done everywhere
PARTIAL="coding ifeval squad"              # arms that also have math+gsm8k

submit() {  # tag round3_dir benchmarks model_path
    local tag=$1 dir=$2 bm=$3 model=$4
    local jid
    jid=$(TAG="$tag" OUTPUT_DIR="$OUT/$dir" BENCHMARKS="$bm" MODEL_PATH="$model" \
        sbatch --parsable scripts/aicr_eval_tulu3_suite.sbatch)
    echo "$jid  $tag  [$bm]"
}

submit base_llama31_8b_instruct base_llama31_8b_instruct_246641 "$FULL" meta-llama/Llama-3.1-8B-Instruct
submit dense_dpo_tulu3_ckpt500 dense_dpo_tulu3_ckpt500_246642 "$PARTIAL" "$T1/dense_dpo_tulu3_llama8b_deltalog/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500"
submit sparse_warm_mag_step200 sparse_warm_mag_step200_246643 "$FULL" "$T1/sparse_dpo_tulu3_warm_magnitude_step200/sparse_dpo_tulu3_warm_magnitude_step200_500steps/final_model"
submit sparse_oracle_dpo_tulu3 sparse_oracle_dpo_tulu3_246644 "$FULL" "$T1/sparse_dpo_tulu3_oracle_dpo_tulu3/sparse_dpo_tulu3_oracle_dpo_tulu3_500steps/final_model"
submit sparse_oracle_dpo_lightr1 sparse_oracle_dpo_lightr1_246645 "$FULL" "$T1/sparse_dpo_tulu3_oracle_dpo_lightr1/sparse_dpo_tulu3_oracle_dpo_lightr1_500steps/final_model"
submit sparse_oracle_grpo_math sparse_oracle_grpo_math_246646 "$FULL" "$T1/sparse_dpo_tulu3_oracle_grpo_math/sparse_dpo_tulu3_oracle_grpo_math_500steps/final_model"
submit sparse_random sparse_random_246647 "$FULL" "$T1/sparse_dpo_tulu3_random/sparse_dpo_tulu3_random_500steps/final_model"
submit lora_r64_lr1e-4_lightr1 lora_r64_lr1e-4_lightr1_246648 "$PARTIAL" "$OUT/merged_lora_r64_lr1e-4_lightr1"
submit lora_r64_lr5e-6_lightr1 lora_r64_lr5e-6_lightr1_246649 "$FULL" "$OUT/merged_lora_r64_lr5e-6_lightr1"
