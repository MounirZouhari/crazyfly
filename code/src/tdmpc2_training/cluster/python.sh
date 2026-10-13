#!/bin/bash
# Run the project's Python on an Alliance GPU node, e.g.
#   bash code/src/tdmpc2_training/cluster/python.sh code/src/tdmpc2_training/check_hover.py
#
# --nv gives the container the node's GPU driver. Home, /scratch and /project
# are visible inside the container by default on Alliance clusters.
module load apptainer
exec apptainer exec --nv --cleanenv \
  ~/containers/python311.sif ~/envs/tdmpc2-ct/bin/python "$@"
