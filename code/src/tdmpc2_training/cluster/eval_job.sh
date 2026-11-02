#!/bin/bash
# Test already-trained runs on the Batch 2 test starts. Nothing is trained.
# One run per array task; RUNS_FILE lists one run folder per line.
#
# Batch 2, R4 (Batch 1 models on the Batch 2 test starts, see ../EXPERIMENTS.md):
#   ls -d $SCRATCH/tdmpc2/batch1_standard_1M/seed_* $SCRATCH/tdmpc2/batch1_standard/seed_* \
#         $SCRATCH/tdmpc2/batch1_small_narval/seed_* > $SCRATCH/tdmpc2/batch2/retest_runs.txt
#   cd $SCRATCH && sbatch --array=0-24 --export=ALL,RUNS_FILE=$SCRATCH/tdmpc2/batch2/retest_runs.txt \
#         ~/multi-agentCrazyflieLift/code/src/tdmpc2_training/cluster/eval_job.sh
#
# Results go to new folders inside each run (the run's own evaluation/ is
# never touched): evaluation_batch2_tests/ (planning as trained) and
# evaluation_batch2_tests_policy_only/ (policy network alone).
#SBATCH --account=def-cbelling-ab
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=tdmpc2-retest
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
echo "retesting $RUN_DIR on $(hostname), $(date), commit $(git rev-parse --short HEAD)"

test_once() {  # $1: output folder in RUN_DIR, then extra eval_hover.py arguments
  local out=$1; shift
  [ -f "$RUN_DIR/$out/summary.csv" ] || \
    $PY $SCRIPTS/eval_hover.py "$RUN_DIR" --test-sets all --out-name "$out" "$@"
}
test_once evaluation_batch2_tests
test_once evaluation_batch2_tests_policy_only --no-mpc
echo "done, $(date)"
