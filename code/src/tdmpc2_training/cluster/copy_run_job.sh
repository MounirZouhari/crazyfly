#!/bin/bash
# Copy finished runs to a new batch folder so that training can continue there:
# seed_job.sh then resumes from the copied checkpoint, and the original run and
# its test results stay untouched. One seed per array task; no GPU.
#
#   cd $SCRATCH && sbatch --array=0-2 --dependency=aftercorr:<training job> \
#     --export=ALL,SRC_BATCH=batch2/random_start_1M,DST_BATCH=batch2/random_start_2M \
#     ~/multi-agentCrazyflieLift/code/src/tdmpc2_training/cluster/copy_run_job.sh
#SBATCH --account=def-cbelling-ab
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:30:00
#SBATCH --job-name=tdmpc2-copy-run
#SBATCH --output=%x-%A_%a.out
set -e

NAME=$(printf "seed_%02d" "$SLURM_ARRAY_TASK_ID")
SRC=$SCRATCH/tdmpc2/$SRC_BATCH/$NAME
DST=$SCRATCH/tdmpc2/$DST_BATCH/$NAME
[ -d "$SRC/checkpoint" ] || { echo "no checkpoint in $SRC"; exit 1; }
if [ ! -d "$DST/checkpoint" ]; then
  mkdir -p "$DST"
  # The agent and its replay buffer, the snapshots and the training history; not the tests.
  cp -r "$SRC/checkpoint" "$SRC/snapshots" "$SRC/.hydra" "$SRC/tensorboard" "$SRC/learning_curve.csv" "$DST/"
fi
echo "copied $SRC -> $DST, $(date)"
du -sh "$DST"
