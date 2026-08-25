#!/bin/bash
# Launch Elastic Horizon training.
#
# Prerequisites:
#   1. conda activate elastic_horizon   (see install.sh)
#   2. Set up the environment service for your benchmark:
#        bash env_service/environments/appworld/setup.sh   # AppWorld
#        bash env_service/environments/bfcl/setup.sh       # BFCL
#   3. (Optional) export SWANLAB_API_KEY=<your_key> to log metrics to SwanLab.

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export SWANLAB_MODE=${SWANLAB_MODE:-"local"}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-"swanlog"}

# AppWorld
python launcher.py --conf examples/max_steps_acl.yaml --with-appworld

# BFCL
# python launcher.py --conf examples/bfcl_acl.yaml --with-bfcl
