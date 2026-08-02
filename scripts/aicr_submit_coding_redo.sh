#!/bin/bash
# Coding-only redo for all 9 Table-1 arms with the fixed coding_evaluator
# (lm-eval 0.4.12 port + apply_chat_template passed through + humaneval
# continuation filter; validated by smoke 248742: he=0.94, mbpp=0.56).
# Each redo is chained --dependency=afterany:<round-4 job> so it cannot race
# the main job's own (broken) coding stage for the same OUTPUT_DIR.
#
# Run on login.aicr.ai from the repo root: bash scripts/aicr_submit_coding_redo.sh
set -euo pipefail

W=/work/neu/p2026_0038_neu/xie_yiyi
cd "$W/rc-sparse-speed"
S=/scratch/xie_yiyi_neu
OUT=$S/rebuttal_analysis/eval_tulu3
T1=$S/transfer_v1

submit() {  # tag results_dir dependency_jobid model_path
    local tag=$1 dir=$2 dep=$3 model=$4
    local jid
    jid=$(TAG="$tag" OUTPUT_DIR="$OUT/$dir" BENCHMARKS="coding" MODEL_PATH="$model" \
        sbatch --parsable --time=01:30:00 --dependency=afterany:$dep \
        scripts/aicr_eval_tulu3_suite.sbatch)
    echo "$jid  coding-redo  $tag  (after $dep)"
}

submit base_llama31_8b_instruct base_llama31_8b_instruct_246641 248621 meta-llama/Llama-3.1-8B-Instruct
submit dense_dpo_tulu3_ckpt500 dense_dpo_tulu3_ckpt500_246642 248622 "$T1/dense_dpo_tulu3_llama8b_deltalog/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500"
submit sparse_warm_mag_step200 sparse_warm_mag_step200_246643 248623 "$T1/sparse_dpo_tulu3_warm_magnitude_step200/sparse_dpo_tulu3_warm_magnitude_step200_500steps/final_model"
submit sparse_oracle_dpo_tulu3 sparse_oracle_dpo_tulu3_246644 248624 "$T1/sparse_dpo_tulu3_oracle_dpo_tulu3/sparse_dpo_tulu3_oracle_dpo_tulu3_500steps/final_model"
submit sparse_oracle_dpo_lightr1 sparse_oracle_dpo_lightr1_246645 248625 "$T1/sparse_dpo_tulu3_oracle_dpo_lightr1/sparse_dpo_tulu3_oracle_dpo_lightr1_500steps/final_model"
submit sparse_oracle_grpo_math sparse_oracle_grpo_math_246646 248626 "$T1/sparse_dpo_tulu3_oracle_grpo_math/sparse_dpo_tulu3_oracle_grpo_math_500steps/final_model"
submit sparse_random sparse_random_246647 248627 "$T1/sparse_dpo_tulu3_random/sparse_dpo_tulu3_random_500steps/final_model"
submit lora_r64_lr1e-4_lightr1 lora_r64_lr1e-4_lightr1_246648 248628 "$OUT/merged_lora_r64_lr1e-4_lightr1"
submit lora_r64_lr5e-6_lightr1 lora_r64_lr5e-6_lightr1_246649 248629 "$OUT/merged_lora_r64_lr5e-6_lightr1"
