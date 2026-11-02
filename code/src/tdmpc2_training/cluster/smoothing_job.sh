#!/bin/bash
# Smoother ways of acting and slow motors (eval_smoothing.py), one run per array
# task; RUNS_FILE lists one run folder per line. Nothing is trained.
#
#   cd $SCRATCH && sbatch --array=0-10 --export=ALL,RUNS_FILE=$SCRATCH/tdmpc2/batch2/disturb_runs_R1.txt \
#     ~/multi-agentCrazyflieLift/code/src/tdmpc2_training/cluster/smoothing_job.sh
#
# Results: smoothing_test/ inside each run.
#SBATCH --account=def-cbelling-ab
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --job-name=tdmpc2-smoothing
#SBATCH --output=%x-%A_%a.out
set -e

RUN_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RUNS_FILE")
if [ -z "$RUN_DIR" ]; then
  echo "no run on line $((SLURM_ARRAY_TASK_ID + 1)) of $RUNS_FILE"
  exit 1
fi

cd ~/multi-agentCrazyflieLift
SCRIPTS=code/src/tdmpc2_training
echo "smoothing test of $RUN_DIR on $(hostname), $(date), commit $(git rev-parse --short HEAD)"
[ -f "$RUN_DIR/smoothing_test/summary.csv" ] || bash $SCRIPTS/cluster/python.sh $SCRIPTS/eval_smoothing.py "$RUN_DIR"
echo "done, $(date)"
