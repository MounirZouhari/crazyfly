#!/bin/bash
# Planner variants at test time (eval_planner_variants.py), one run per array
# task; RUNS_FILE lists one run folder per line. Nothing is trained.
#
#   cd $SCRATCH && sbatch --array=0-24 --export=ALL,RUNS_FILE=$SCRATCH/tdmpc2/batch2/retest_runs.txt \
#     ~/multi-agentCrazyflieLift/code/src/tdmpc2_training/cluster/variants_job.sh
#
# Results: planner_variants/ inside each run.
#SBATCH --account=def-cbelling-ab
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:30:00
#SBATCH --job-name=tdmpc2-variants
#SBATCH --output=%x-%A_%a.out
set -e

RUN_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RUNS_FILE")
if [ -z "$RUN_DIR" ]; then
  echo "no run on line $((SLURM_ARRAY_TASK_ID + 1)) of $RUNS_FILE"
  exit 1
fi

cd ~/multi-agentCrazyflieLift
SCRIPTS=code/src/tdmpc2_training
echo "planner variants of $RUN_DIR on $(hostname), $(date), commit $(git rev-parse --short HEAD)"
[ -f "$RUN_DIR/planner_variants/summary.csv" ] || bash $SCRIPTS/cluster/python.sh $SCRIPTS/eval_planner_variants.py "$RUN_DIR"
echo "done, $(date)"
