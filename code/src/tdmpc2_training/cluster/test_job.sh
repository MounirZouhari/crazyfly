#!/bin/bash
# Short GPU test before the real batch: checks the setup, then trains the
# small and the standard settings for 3,000 steps each and prints their speed.
#
# Submit from $SCRATCH so the log lands there:
#   cd $SCRATCH && sbatch ~/multi-agentCrazyflieLift/code/src/tdmpc2_training/cluster/test_job.sh
# Result: $SCRATCH/tdmpc2-test-<jobid>.out
#
#SBATCH --account=def-cbelling-ab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --job-name=tdmpc2-test
#SBATCH --output=%x-%j.out
set -e

cd ~/multi-agentCrazyflieLift
SCRIPTS=code/src/tdmpc2_training
PY="bash $SCRIPTS/cluster/python.sh"

nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -c "import jax; print('JAX devices:', jax.devices())"
$PY $SCRIPTS/check_hover.py

# One folder per job, so two test jobs never write into each other.
TEST_DIR=$SCRATCH/tdmpc2/speed_test/$SLURM_JOB_ID
mkdir -p $TEST_DIR
for CONFIG in config_hover config_hover_standard; do
  RUN_DIR=$TEST_DIR/$CONFIG
  $PY $SCRIPTS/train_hover.py --config-name=$CONFIG seed=0 max_steps=3000 \
    eval_interval_steps=1000 hydra.run.dir=$RUN_DIR > $RUN_DIR.log 2>&1
  $PY - "$RUN_DIR/learning_curve.csv" "$CONFIG" <<'EOF'
import csv, sys
seconds = {int(r["env_steps"]): float(r["train_seconds"]) for r in csv.DictReader(open(sys.argv[1]))}
speed = 2000 / (seconds[3000] - seconds[1000])
print(f"{sys.argv[2]}: {speed:.1f} steps/s -> about {50_000 / speed / 60:.0f} min for 50,000 steps")
EOF
done
