#!/bin/bash
# Snapshot evaluation (eval_snapshots.py) of finished runs, one run per array
# task; RUNS_FILE lists one run folder per line. Nothing is trained. Every
# snapshot flies 40 validation starts and the three Batch 2 test groups, with
# planning and with the policy network alone.
#
#   cd $SCRATCH && sbatch --array=0-10 --dependency=aftercorr:<training job> \
#     --export=ALL,RUNS_FILE=$SCRATCH/tdmpc2/batch2/disturb_runs_R1.txt \
#     ~/multi-agentCrazyflieLift/code/src/tdmpc2_training/cluster/snapshot_job.sh
#
# Results: snapshot_evaluation/ and snapshot_evaluation_policy_only/ inside each run.
#SBATCH --account=def-cbelling-ab
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --job-name=tdmpc2-snapshots
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
echo "snapshot evaluation of $RUN_DIR on $(hostname), $(date), commit $(git rev-parse --short HEAD)"

[ -f "$RUN_DIR/snapshot_evaluation/summary.csv" ] || $PY $SCRIPTS/eval_snapshots.py "$RUN_DIR"
[ -f "$RUN_DIR/snapshot_evaluation_policy_only/summary.csv" ] || $PY $SCRIPTS/eval_snapshots.py "$RUN_DIR" --no-mpc
echo "done, $(date)"
