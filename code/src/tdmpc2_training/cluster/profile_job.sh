#!/bin/bash
# Batch 2, P: speed and deployment measurements on a Narval GPU slice.
# Nothing is trained; no checkpoint is needed.
#   1. where one training step spends its time, small and published settings
#      (profile_training.py), with the GPU sampled every 0.5 s meanwhile
#   2. time per control decision for 3 network sizes x 5 planning budgets
#      (bench_latency.py), on the GPU slice and on the job's 2 CPU cores
#
#   cd $SCRATCH && sbatch ~/multi-agentCrazyflieLift/code/src/tdmpc2_training/cluster/profile_job.sh
#
# Results: $SCRATCH/tdmpc2/batch2/profiling/
#SBATCH --account=def-cbelling-ab
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=tdmpc2-profile
#SBATCH --output=%x-%j.out
set -e

OUT=$SCRATCH/tdmpc2/batch2/profiling
mkdir -p "$OUT"
cd ~/multi-agentCrazyflieLift
SCRIPTS=code/src/tdmpc2_training
PY="bash $SCRIPTS/cluster/python.sh"
echo "profiling on $(hostname), $(date), commit $(git rev-parse --short HEAD)"
nvidia-smi -L

for config in config_hover config_hover_standard; do
  $PY $SCRIPTS/profile_training.py $config --steps 1000 --modes attribute --csv "$OUT/training_step.csv"
  # The GPU is sampled only during the realistic pass (after ~60 s of setup and compiling).
  $PY $SCRIPTS/profile_training.py $config --steps 1500 --modes realistic --csv "$OUT/training_step.csv" &
  PROFILE=$!
  sleep 60
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -lms 500 \
    > "$OUT/gpu_usage_$config.csv" 2>&1 &
  SAMPLER=$!
  wait $PROFILE
  kill $SAMPLER 2>/dev/null || true
done

$PY $SCRIPTS/bench_latency.py --calls 1000 --max-seconds 40 --out "$OUT"
# --cleanenv keeps host variables out of the container; APPTAINERENV_ passes this one in.
APPTAINERENV_JAX_PLATFORMS=cpu $PY $SCRIPTS/bench_latency.py --calls 300 --max-seconds 20 --out "$OUT"
echo "done, $(date)"
