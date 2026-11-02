#!/bin/bash
# Disturbance tests (eval_disturbances.py) of already-trained runs, one run per
# array task; RUNS_FILE lists one run folder per line. Nothing is trained.
# Each run is flown with planning and with the policy network alone.
#
# Batch 2, D (see ../EXPERIMENTS.md), e.g. for the Batch 1 1M-step agents:
#   ls -d $SCRATCH/tdmpc2/batch1_standard_1M/seed_* > $SCRATCH/tdmpc2/batch2/disturb_runs_batch1.txt
#   cd $SCRATCH && sbatch --array=0-2 --export=ALL,RUNS_FILE=$SCRATCH/tdmpc2/batch2/disturb_runs_batch1.txt \
#         ~/multi-agentCrazyflieLift/code/src/tdmpc2_training/cluster/disturb_job.sh
#
# Results: disturbances/ and disturbances_policy_only/ inside each run.
#SBATCH --account=def-cbelling-ab
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:30:00
#SBATCH --job-name=tdmpc2-disturb
#SBATCH --output=%x-%A_%a.out
set -e

RUN_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RUNS_FILE")
if [ -z "$RUN_DIR" ]; then
  echo "no run on line $((SLURM_ARRAY_TASK_ID + 1)) of $RUNS_FILE"
  exit 1
fi

cd ~/multi-agentCrazyflieLift
SCRIPTS=code/src/tdmpc2_training
PY="bash $SCRIPTS/cluster/python.sh"
echo "disturbance tests of $RUN_DIR on $(hostname), $(date), commit $(git rev-parse --short HEAD)"

[ -f "$RUN_DIR/disturbances/summary.csv" ] || $PY $SCRIPTS/eval_disturbances.py "$RUN_DIR"
[ -f "$RUN_DIR/disturbances_policy_only/summary.csv" ] || $PY $SCRIPTS/eval_disturbances.py "$RUN_DIR" --no-mpc
echo "done, $(date)"
