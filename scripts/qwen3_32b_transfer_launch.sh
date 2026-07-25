#!/bin/bash
# Kickoff for the Qwen3-32B high-sparsity DPO chain (light-r1), device_map="auto"
# model parallelism across H200.
#   p1 dense DPO  ->  p2 oracle mask @97.5%  ->  p3 sparse DPO
#
# This NO LONGER uses `sbatch --dependency=afterok` (that chain was dead: p1 hits
# the 8h walltime and exits TIME LIMIT, never 0, so afterok p2/p3 stayed PENDING).
# Orchestration now lives in scripts/qwen3_32b_advance.sh, which each job re-invokes
# from its tail. This launcher just submits the FIRST job; the chain self-advances
# across as many 8h slots as it takes, with no human in the loop.
#
# Run on the cluster:  DATASET=light-r1 bash scripts/qwen3_32b_transfer_launch.sh
# Watch:   squeue -u $USER   |   tail -f docs/run_logs/qwen3_32b_autopipeline.log
# Halt:    scancel the running job AND `touch /scratch/$USER/transfer_v1/PIPELINE_STALLED`
#          (the sentinel stops the tail from resubmitting; remove it to resume).
set -euo pipefail
cd /home/xie.yiyi/rc-sparse-speed
export DATASET="${DATASET:-light-r1}"
echo "Kicking off Qwen3-32B auto-pipeline for DATASET=$DATASET via advance.sh"
bash scripts/qwen3_32b_advance.sh
echo "Done. The chain will self-advance. Watch: squeue -u \$USER"
