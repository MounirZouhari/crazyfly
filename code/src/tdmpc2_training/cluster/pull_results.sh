#!/bin/bash
# Copy the Batch 2 results from Narval to the home PC, without checkpoints,
# snapshots or replay buffers (those stay on Narval). Run from the repository
# root in the Ubuntu terminal after logging in with `ssh narval`:
#
#   bash code/src/tdmpc2_training/cluster/pull_results.sh
#
# Batch 2 batches go to outputs/batch2/, the extra test results of the Batch 1
# runs (evaluation_batch2_tests*, disturbances*, model_error, planner_variants)
# into the existing outputs/batch1_* folders.
set -e
FILES=(--include='*/' --include='learning_curve.csv' --include='batch_info.json' --include='.hydra/***'
       --include='evaluation*/***' --include='disturbances*/***' --include='snapshot_evaluation*/***'
       --include='model_error*/***' --include='planner_variants/***' --include='gpu_usage.csv'
       --include='tensorboard/***' --include='*.log' --exclude='*')
mkdir -p outputs/batch2
for batch in random_start_1M fixed_start_1M random_start_small_1M; do
  rsync -am "${FILES[@]}" narval:/scratch/mounirz/tdmpc2/batch2/$batch/ outputs/batch2/$batch/
done
rsync -am narval:/scratch/mounirz/tdmpc2/batch2/profiling/ outputs/batch2/profiling_narval/
rsync -am narval:/scratch/mounirz/tdmpc2/batch2/parallel_projection/ outputs/batch2/parallel_projection_narval/ || true
rsync -am narval:/scratch/mounirz/tdmpc2/batch2/job_ids.txt outputs/batch2/
for batch in batch1_standard_1M batch1_standard batch1_small_narval; do
  rsync -am --include='*/' --include='evaluation_batch2_tests*/***' --include='disturbances*/***' \
    --include='model_error*/***' --include='planner_variants/***' --exclude='*' \
    narval:/scratch/mounirz/tdmpc2/$batch/ outputs/$batch/
done
echo "Pulled. Seeds with final tests:"
for batch in random_start_1M fixed_start_1M random_start_small_1M; do
  echo "  $batch: $(ls -d outputs/batch2/$batch/seed_*/evaluation 2>/dev/null | wc -l)"
done
