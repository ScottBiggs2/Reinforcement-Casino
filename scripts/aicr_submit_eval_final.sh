#!/bin/bash
# Final gap-fill after the /scratch purge recovery: the 12 cells lost when the
# purge deleted the model weights mid-run. Run AFTER all 6 rsync streams report
# STREAM_DONE (weights re-landed + mtimes touched).
set -euo pipefail

W=/work/neu/p2026_0038_neu/xie_yiyi
cd "$W/rc-sparse-speed"
S=/scratch/xie_yiyi_neu
OUT=$S/rebuttal_analysis/eval_tulu3
T1=$S/transfer_v1

submit() {  # tag results_dir benchmarks model_path
    local tag=$1 dir=$2 bm=$3 model=$4
    test -f "$model/config.json" || { echo "SKIP $tag: $model/config.json missing"; return; }
    local jid
    jid=$(TAG="$tag" OUTPUT_DIR="$OUT/$dir" BENCHMARKS="$bm" MODEL_PATH="$model" \
        sbatch --parsable --time=00:50:00 scripts/aicr_eval_tulu3_suite.sbatch)
    echo "$jid  final  $tag  [$bm]"
}

SPARSE="gsm8k ifeval squad"
submit sparse_warm_mag_step200 sparse_warm_mag_step200_246643 "$SPARSE" "$T1/sparse_dpo_tulu3_warm_magnitude_step200/sparse_dpo_tulu3_warm_magnitude_step200_500steps/final_model"
submit sparse_oracle_dpo_tulu3 sparse_oracle_dpo_tulu3_246644 "$SPARSE" "$T1/sparse_dpo_tulu3_oracle_dpo_tulu3/sparse_dpo_tulu3_oracle_dpo_tulu3_500steps/final_model"
submit sparse_oracle_dpo_lightr1 sparse_oracle_dpo_lightr1_246645 "$SPARSE" "$T1/sparse_dpo_tulu3_oracle_dpo_lightr1/sparse_dpo_tulu3_oracle_dpo_lightr1_500steps/final_model"
submit sparse_oracle_grpo_math sparse_oracle_grpo_math_246646 "$SPARSE" "$T1/sparse_dpo_tulu3_oracle_grpo_math/sparse_dpo_tulu3_oracle_grpo_math_500steps/final_model"
submit sparse_random sparse_random_246647 "$SPARSE" "$T1/sparse_dpo_tulu3_random/sparse_dpo_tulu3_random_500steps/final_model"
submit dense_dpo_tulu3_ckpt500 dense_dpo_tulu3_ckpt500_246642 "ifeval squad" "$T1/dense_dpo_tulu3_llama8b_deltalog/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500"
