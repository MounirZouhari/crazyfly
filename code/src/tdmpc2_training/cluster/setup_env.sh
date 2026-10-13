#!/bin/bash
# One-time Python setup on an Alliance cluster. Run on a LOGIN node (compute
# nodes have no internet), from the repository root:
#
#   bash code/src/tdmpc2_training/cluster/setup_env.sh
#
# The cluster's own Python packages do not include MuJoCo 3.4.0 (the physics
# version used at home), so Python runs inside a plain Python 3.11 container
# where the exact same versions install from PyPI. The packages live in
# ~/envs/tdmpc2-ct; the container only provides Python itself.
set -e

module load apptainer
export APPTAINER_CACHEDIR=$SCRATCH/.apptainer
mkdir -p ~/containers ~/envs

IMAGE=~/containers/python311.sif
[ -f $IMAGE ] || apptainer pull $IMAGE docker://python:3.11-slim
# --cleanenv keeps the cluster's pip settings out of the container.
RUN="apptainer exec --cleanenv $IMAGE"

[ -d ~/envs/tdmpc2-ct ] || $RUN python -m venv ~/envs/tdmpc2-ct
$RUN ~/envs/tdmpc2-ct/bin/pip install --upgrade pip
# Exact versions from the home PC (made with `uv pip freeze`), so results from
# both machines can be compared directly.
$RUN ~/envs/tdmpc2-ct/bin/pip install -r code/src/tdmpc2_training/cluster/requirements-lock.txt

$RUN ~/envs/tdmpc2-ct/bin/python -c "import jax, mujoco; print('jax', jax.__version__, '| mujoco', mujoco.__version__)"
echo "Environment ready: ~/envs/tdmpc2-ct (run it with cluster/python.sh)"
