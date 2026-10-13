#!/bin/bash
# Train and test one seed per array task on an Alliance cluster. Same steps as
# run_seeds.py on the home PC, but the seeds run as separate GPU jobs.
#
# Submit seeds 0-10 from $SCRATCH (logs land there):
#   cd $SCRATCH && sbatch --array=0-10 ~/multi-agentCrazyflieLift/code/src/tdmpc2_training/cluster/seed_job.sh
#
# Defaults: standard TD-MPC2 settings, 50,000 steps. To change, e.g.
#   sbatch --array=0-10 --export=ALL,CONFIG=config_hover,BATCH=batch1_small_cluster .../seed_job.sh
# Extra Hydra settings go in EXTRA (set it in the shell before sbatch), e.g.
#   export EXTRA="buffer_size=1000000 save_interval_steps=50000"
#
# Results: $SCRATCH/tdmpc2/<BATCH>/seed_XX/  (same layout as outputs/ at home,
# so analyze_seeds.py works on it after copying it back).
#
# Runs on a 2/7 slice of an A100 (a100_2g.10gb): far shorter queue than a full
# GPU, and the standard settings train at ~49 steps/s on it (~30 min per seed).
# Jobs of 3 hours or less usually start sooner.
#SBATCH --account=def-cbelling-ab
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --job-name=tdmpc2-seed
#SBATCH --output=%x-%A_%a.out
set -e

CONFIG=${CONFIG:-config_hover_standard}
BATCH=${BATCH:-batch1_standard}
STEPS=${STEPS:-50000}
EXTRA=${EXTRA:-}
SEED=$SLURM_ARRAY_TASK_ID
BATCH_DIR=$SCRATCH/tdmpc2/$BATCH
RUN_DIR=$BATCH_DIR/$(printf "seed_%02d" "$SEED")

cd ~/multi-agentCrazyflieLift
SCRIPTS=code/src/tdmpc2_training
PY="bash $SCRIPTS/cluster/python.sh"

if [ -f "$RUN_DIR/evaluation/summary.csv" ]; then
  echo "seed $SEED already finished"
  exit 0
fi

# Record what this batch ran (first task to get here writes it).
mkdir -p "$BATCH_DIR"
if [ ! -f "$BATCH_DIR/batch_info.json" ]; then
  COMMIT=$(git rev-parse HEAD)
  DIRTY=$([ -n "$(git status --porcelain)" ] && echo true || echo false)
  OVERRIDES="\"--config-name=$CONFIG\""
  for setting in $EXTRA; do OVERRIDES="$OVERRIDES, \"$setting\""; done
  cat > "$BATCH_DIR/batch_info.json" <<EOF
{"steps": $STEPS, "overrides": [$OVERRIDES],
 "launches": [{"commit": "$COMMIT", "uncommitted_changes": $DIRTY, "cluster": "$CC_CLUSTER"}]}
EOF
fi

nvidia-smi --query-gpu=name --format=csv,noheader
# If an earlier attempt of this seed timed out, training resumes from its
# last checkpoint (saved every 2,000 steps).
$PY $SCRIPTS/train_hover.py --config-name="$CONFIG" seed="$SEED" \
  max_steps="$STEPS" hydra.run.dir="$RUN_DIR" $EXTRA
$PY $SCRIPTS/eval_hover.py "$RUN_DIR"
