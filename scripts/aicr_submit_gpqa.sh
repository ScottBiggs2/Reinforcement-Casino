#!/bin/bash
# gpqa_diamond-only follow-up for all 9 Table-1 arms, once Idavidrein/gpqa
# access is granted and the dataset is prefetched into the shared cache.
# Chained afterany on each arm's coding-redo job (248812-20) so two jobs never
# write the same OUTPUT_DIR concurrently.
#
# Run on login.aicr.ai from the repo root: bash scripts/aicr_submit_gpqa.sh
set -euo pipefail

W=/work/neu/p2026_0038_neu/xie_yiyi
cd "$W/rc-sparse-speed"
S=/scratch/xie_yiyi_neu
OUT=$S/rebuttal_analysis/eval_tulu3
T1=$S/transfer_v1

submit() {  # tag results_dir dependency_jobid model_path
    local tag=$1 dir=$2 dep=$3 model=$4
    local jid
    jid=$(TAG="$tag" OUTPUT_DIR="$OUT/$dir" BENCHMARKS="gpqa_diamond" MODEL_PATH="$model" \
        sbatch --parsable --time=00:45:00 --dependency=afterany:$dep \
        scripts/aicr_eval_tulu3_suite.sbatch)
    echo "$jid  gpqa  $tag  (after $dep)"
}

submit base_llama31_8b_instruct base_llama31_8b_instruct_246641 248812 meta-llama/Llama-3.1-8B-Instruct
submit dense_dpo_tulu3_ckpt500 dense_dpo_tulu3_ckpt500_246642 248813 "$T1/dense_dpo_tulu3_llama8b_deltalog/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500"
submit sparse_warm_mag_step200 sparse_warm_mag_step200_246643 248814 "$T1/sparse_dpo_tulu3_warm_magnitude_step200/sparse_dpo_tulu3_warm_magnitude_step200_500steps/final_model"
submit sparse_oracle_dpo_tulu3 sparse_oracle_dpo_tulu3_246644 248815 "$T1/sparse_dpo_tulu3_oracle_dpo_tulu3/sparse_dpo_tulu3_oracle_dpo_tulu3_500steps/final_model"
submit sparse_oracle_dpo_lightr1 sparse_oracle_dpo_lightr1_246645 248816 "$T1/sparse_dpo_tulu3_oracle_dpo_lightr1/sparse_dpo_tulu3_oracle_dpo_lightr1_500steps/final_model"
submit sparse_oracle_grpo_math sparse_oracle_grpo_math_246646 248817 "$T1/sparse_dpo_tulu3_oracle_grpo_math/sparse_dpo_tulu3_oracle_grpo_math_500steps/final_model"
submit sparse_random sparse_random_246647 248818 "$T1/sparse_dpo_tulu3_random/sparse_dpo_tulu3_random_500steps/final_model"
submit lora_r64_lr1e-4_lightr1 lora_r64_lr1e-4_lightr1_246648 248819 "$OUT/merged_lora_r64_lr1e-4_lightr1"
submit lora_r64_lr5e-6_lightr1 lora_r64_lr5e-6_lightr1_246649 248820 "$OUT/merged_lora_r64_lr5e-6_lightr1"
