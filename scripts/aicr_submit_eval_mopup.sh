#!/bin/bash
# Mop-up round: per-arm benchmarks still missing after the round-4 TIMEOUTs
# (jobs spent ~2h wedged on vLLM engine-core exit-hangs before the nudger/
# os._exit fix; the lowered 2:30 walltime then cut them off mid-suite).
# Runs the patched wrapper (os._exit after transport write — no exit hangs).
# Chained afterany on each arm's gpqa job = the tail of its existing chain.
#
# Run on login.aicr.ai from the repo root: bash scripts/aicr_submit_eval_mopup.sh
set -euo pipefail

W=/work/neu/p2026_0038_neu/xie_yiyi
cd "$W/rc-sparse-speed"
S=/scratch/xie_yiyi_neu
OUT=$S/rebuttal_analysis/eval_tulu3
T1=$S/transfer_v1

TWO="ifeval squad"                 # arms that already have math+gsm8k
FOUR="math gsm8k ifeval squad"     # arms that got nothing past mmlu

submit() {  # tag results_dir dep benchmarks time model_path
    local tag=$1 dir=$2 dep=$3 bm=$4 tl=$5 model=$6
    local jid
    jid=$(TAG="$tag" OUTPUT_DIR="$OUT/$dir" BENCHMARKS="$bm" MODEL_PATH="$model" \
        sbatch --parsable --time=$tl --dependency=afterany:$dep \
        scripts/aicr_eval_tulu3_suite.sbatch)
    echo "$jid  mopup  $tag  [$bm]  (after $dep)"
}

submit base_llama31_8b_instruct base_llama31_8b_instruct_246641 248849 "$TWO" 01:15:00 meta-llama/Llama-3.1-8B-Instruct
submit dense_dpo_tulu3_ckpt500 dense_dpo_tulu3_ckpt500_246642 248850 "$TWO" 01:15:00 "$T1/dense_dpo_tulu3_llama8b_deltalog/checkpoints/meta_llama_llama_3_1_8b_instruct_tulu3/checkpoint-500"
submit sparse_warm_mag_step200 sparse_warm_mag_step200_246643 248851 "$FOUR" 02:00:00 "$T1/sparse_dpo_tulu3_warm_magnitude_step200/sparse_dpo_tulu3_warm_magnitude_step200_500steps/final_model"
submit sparse_oracle_dpo_tulu3 sparse_oracle_dpo_tulu3_246644 248852 "$FOUR" 02:00:00 "$T1/sparse_dpo_tulu3_oracle_dpo_tulu3/sparse_dpo_tulu3_oracle_dpo_tulu3_500steps/final_model"
submit sparse_oracle_dpo_lightr1 sparse_oracle_dpo_lightr1_246645 248853 "$FOUR" 02:00:00 "$T1/sparse_dpo_tulu3_oracle_dpo_lightr1/sparse_dpo_tulu3_oracle_dpo_lightr1_500steps/final_model"
submit sparse_oracle_grpo_math sparse_oracle_grpo_math_246646 248854 "$FOUR" 02:00:00 "$T1/sparse_dpo_tulu3_oracle_grpo_math/sparse_dpo_tulu3_oracle_grpo_math_500steps/final_model"
submit sparse_random sparse_random_246647 248855 "$FOUR" 02:00:00 "$T1/sparse_dpo_tulu3_random/sparse_dpo_tulu3_random_500steps/final_model"
submit lora_r64_lr1e-4_lightr1 lora_r64_lr1e-4_lightr1_246648 248856 "$TWO" 01:15:00 "$OUT/merged_lora_r64_lr1e-4_lightr1"
submit lora_r64_lr5e-6_lightr1 lora_r64_lr5e-6_lightr1_246649 248857 "$FOUR" 02:00:00 "$OUT/merged_lora_r64_lr5e-6_lightr1"
