#!/bin/bash
# World-model prediction error along test flights (model_error.py), one run per
# array task; RUNS_FILE lists one run folder per line. Nothing is trained.
#
#   cd $SCRATCH && sbatch --array=0-2 --export=ALL,RUNS_FILE=$SCRATCH/tdmpc2/batch2/disturb_runs_batch1_1M.txt \
#     ~/multi-agentCrazyflieLift/code/src/tdmpc2_training/cluster/model_error_job.sh
#
# Results: model_error/ inside each run.
#SBATCH --account=def-cbelling-ab
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=tdmpc2-model-error
#SBATCH --output=%x-%A_%a.out
set -e

RUN_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RUNS_FILE")
if [ -z "$RUN_DIR" ]; then
  echo "no run on line $((SLURM_ARRAY_TASK_ID + 1)) of $RUNS_FILE"
  exit 1
fi

cd ~/multi-agentCrazyflieLift
SCRIPTS=code/src/tdmpc2_training
echo "model error of $RUN_DIR on $(hostname), $(date), commit $(git rev-parse --short HEAD)"
[ -f "$RUN_DIR/model_error/summary.csv" ] || bash $SCRIPTS/cluster/python.sh $SCRIPTS/model_error.py "$RUN_DIR"
echo "done, $(date)"
