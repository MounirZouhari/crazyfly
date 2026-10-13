#!/bin/bash
# Train and test PPO on the TD-MPC2 hover task, one seed per array task.
#
# Submit seeds 0-10 from $SCRATCH (logs land there):
#   cd $SCRATCH && sbatch --array=0-10 ~/multi-agentCrazyflieLift/code/src/tdmpc2_training/cluster/ppo_job.sh
#
# Defaults: 20M environment steps per seed (STEPS to change), results in
# $SCRATCH/tdmpc2/<BATCH>/seed_XX/ with the same files as a TD-MPC2 run.
#SBATCH --account=def-cbelling-ab
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --job-name=ppo-seed
#SBATCH --output=%x-%A_%a.out
set -e

BATCH=${BATCH:-ppo_fixed_start}
STEPS=${STEPS:-20000000}
SEED=$SLURM_ARRAY_TASK_ID
BATCH_DIR=$SCRATCH/tdmpc2/$BATCH
RUN_DIR=$BATCH_DIR/$(printf "seed_%02d" "$SEED")

cd ~/multi-agentCrazyflieLift
SCRIPTS=code/src/tdmpc2_training

if [ -f "$RUN_DIR/evaluation/summary.csv" ]; then
  echo "seed $SEED already finished"
  exit 0
fi

mkdir -p "$BATCH_DIR"
if [ ! -f "$BATCH_DIR/batch_info.json" ]; then
  COMMIT=$(git rev-parse HEAD)
  DIRTY=$([ -n "$(git status --porcelain)" ] && echo true || echo false)
  cat > "$BATCH_DIR/batch_info.json" <<EOF
{"steps": $STEPS, "overrides": ["PPO (gpu_training/ppo_training.py defaults, episode_length=500)"],
 "launches": [{"commit": "$COMMIT", "uncommitted_changes": $DIRTY, "cluster": "$CC_CLUSTER"}]}
EOF
fi

nvidia-smi --query-gpu=name --format=csv,noheader
bash $SCRIPTS/cluster/python.sh $SCRIPTS/ppo_hover.py --seed "$SEED" --steps "$STEPS" --out "$RUN_DIR"
