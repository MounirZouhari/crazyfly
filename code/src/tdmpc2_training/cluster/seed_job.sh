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
# DEPLOYMENT_TESTS=1 also tests every trained agent with its policy network
# alone (no planning) and with a small 64x2 planner.
#
# Batch 2 (see ../EXPERIMENTS.md), with a longer time limit:
#   sbatch --array=0-10 --time=10:00:00 --export=ALL,CONFIG=config_batch2_random_start,BATCH=batch2/random_start_1M,STEPS=1000000,DEPLOYMENT_TESTS=1 .../seed_job.sh
# or in four 3-hour pieces, each task waiting for the same task of the previous piece:
#   P1=$(sbatch --parsable --array=0-10 --time=3:00:00 --export=ALL,...,STEPS=250000,SKIP_TESTS=1 .../seed_job.sh)
#   P2=$(sbatch --parsable --dependency=aftercorr:$P1 ... STEPS=500000,SKIP_TESTS=1 ...)  and so on;
#   the last piece has STEPS=1000000 and no SKIP_TESTS.
#
# Results: $SCRATCH/tdmpc2/<BATCH>/seed_XX/  (same layout as outputs/ at home,
# so analyze_seeds.py works on it after copying it back). gpu_usage.csv in
# each run folder samples the GPU every 30 s.
#
# Runs on a 2/7 slice of an A100 (a100_2g.10gb): far shorter queue than a full
# GPU, and the standard settings train at ~49 steps/s on it (~30 min per 50k
# steps). Jobs of 3 hours or less usually start sooner.
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
DEPLOYMENT_TESTS=${DEPLOYMENT_TESTS:-}
SEED=$SLURM_ARRAY_TASK_ID
BATCH_DIR=$SCRATCH/tdmpc2/$BATCH
RUN_DIR=$BATCH_DIR/$(printf "seed_%02d" "$SEED")

cd ~/multi-agentCrazyflieLift
SCRIPTS=code/src/tdmpc2_training
PY="bash $SCRIPTS/cluster/python.sh"

# Record what this batch ran (first task to get here writes it).
mkdir -p "$RUN_DIR"
if [ ! -f "$BATCH_DIR/batch_info.json" ]; then
  COMMIT=$(git rev-parse HEAD)
  DIRTY=$([ -n "$(git status --porcelain)" ] && echo true || echo false)
  OVERRIDES="\"--config-name=$CONFIG\""
  for setting in $EXTRA; do OVERRIDES="$OVERRIDES, \"$setting\""; done
  cat > "$BATCH_DIR/batch_info.json" <<EOF
{"steps": $STEPS, "overrides": [$OVERRIDES], "deployment_tests": "$DEPLOYMENT_TESTS",
 "launches": [{"commit": "$COMMIT", "uncommitted_changes": $DIRTY, "cluster": "$CC_CLUSTER", "job": "$SLURM_ARRAY_JOB_ID"}]}
EOF
fi

echo "seed $SEED on $(hostname), $(date), commit $(git rev-parse --short HEAD)"
nvidia-smi -L
# GPU use while this job runs. MIG slices may report [N/A] for utilisation.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
  --format=csv -l 30 >> "$RUN_DIR/gpu_usage.csv" 2>&1 &
GPU_LOGGER=$!
trap 'kill $GPU_LOGGER 2>/dev/null || true' EXIT

# Train unless the final checkpoint exists. If an earlier attempt of this seed
# timed out, training resumes from its last checkpoint.
if [ ! -d "$RUN_DIR/checkpoint/$STEPS" ]; then
  $PY $SCRIPTS/train_hover.py --config-name="$CONFIG" seed="$SEED" \
    max_steps="$STEPS" hydra.run.dir="$RUN_DIR" $EXTRA
fi

# A long run can be split into pieces of at most 3 hours (they start much
# sooner): each piece resumes from the previous checkpoint and trains to its
# own STEPS; SKIP_TESTS=1 on all but the last piece.
if [ -n "${SKIP_TESTS:-}" ]; then
  echo "seed $SEED trained to $STEPS steps; testing is left to the last piece, $(date)"
  exit 0
fi

# Each test runs only if its results are not there yet.
test_once() {  # $1: output folder in RUN_DIR, then extra eval_hover.py arguments
  local out=$1; shift
  [ -f "$RUN_DIR/$out/summary.csv" ] || $PY $SCRIPTS/eval_hover.py "$RUN_DIR" "$@"
}
test_once evaluation
if [ -n "$DEPLOYMENT_TESTS" ]; then
  test_once evaluation_policy_only --no-mpc
  test_once evaluation_plan64x2 --plan 64x2
fi
echo "seed $SEED done, $(date)"
