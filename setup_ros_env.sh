#!/usr/bin/env bash

# Locate repository root based on this script's location
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ROS 2 Humble underlay
source /opt/ros/humble/setup.bash

# Project Python virtual environment
source "$REPO_ROOT/.venv/bin/activate"

# Original Python modules: src/rgbd_mapping
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ROS 2 workspace overlay
source "$REPO_ROOT/ros2_ws/install/setup.bash"

echo "ROS 2 development environment ready."
echo "Repository: $REPO_ROOT"
echo "Python: $(which python)"