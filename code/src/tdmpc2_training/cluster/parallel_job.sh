#!/bin/bash
# Batch 2, E: could TD-MPC2 train faster with many simulated drones at once?
# Times simulating and planning for N drones in one batch, and one learning
# update, then projects training speed (bench_parallel.py). Nothing is trained.
#
#   cd $SCRATCH && sbatch ~/multi-agentCrazyflieLift/code/src/tdmpc2_training/cluster/parallel_job.sh
#
# Results: $SCRATCH/tdmpc2/batch2/parallel_projection/<config>/
#SBATCH --account=def-cbelling-ab
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=tdmpc2-parallel
#SBATCH --output=%x-%j.out
set -e

cd ~/multi-agentCrazyflieLift
SCRIPTS=code/src/tdmpc2_training
PY="bash $SCRIPTS/cluster/python.sh"
echo "parallel projection on $(hostname), $(date), commit $(git rev-parse --short HEAD)"
nvidia-smi -L
for config in config_hover_standard config_hover; do
  $PY $SCRIPTS/bench_parallel.py --config $config --out $SCRATCH/tdmpc2/batch2/parallel_projection/$config
done
echo "done, $(date)"
