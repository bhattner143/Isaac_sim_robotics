#!/bin/bash
# =============================================================================
# run_talker.sh — Launch Drake Talker → ROS 2 Bridge
# =============================================================================
# Drake (conda pydrake) computes pendulum state and pipes to
# the ROS 2 talker node (system Python 3.12) which publishes
# on the /drake_hello topic.
#
# Usage:
#   bash ros2_test_ubuntu/run_talker.sh [--rate 1.0] [--duration 30] [--angle 30]
#
# Run from repo root:
#   cd /home/user/Documents/isaac_sim_robotics
#   bash ros2_test_ubuntu/run_talker.sh

set -e

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS2_TEST_DIR="$(dirname "$LAUNCH_DIR")"

ROS2_SETUP="$HOME/ros2_jazzy/install/local_setup.bash"
SYSTEM_PYTHON="/usr/bin/python3"
CONDA_PYTHON="/home/user/anaconda3/envs/env_isaacsim/bin/python3"

SCENARIO_DIR="$LAUNCH_DIR"  # run_talker.sh lives inside talker_listener/
DRAKE_TALKER="$SCENARIO_DIR/drake_talker.py"
ROS2_TALKER_NODE="$SCENARIO_DIR/ros2_talker_node.py"

# Validate
if [[ ! -f "$ROS2_SETUP" ]]; then
    echo "ERROR: ROS 2 not found at $ROS2_SETUP"
    echo "       See ros2_test_ubuntu/INSTALLATION_ROS2_JAZZY_UBUNTU.md"
    exit 1
fi

if [[ ! -f "$CONDA_PYTHON" ]]; then
    echo "ERROR: env_isaacsim conda env not found at $CONDA_PYTHON"
    echo "       Run: conda activate env_isaacsim"
    exit 1
fi

echo "======================================================="
echo "  Drake Hello World Talker → ROS 2 Bridge"
echo "======================================================="
echo "  Drake script    : $DRAKE_TALKER"
echo "  ROS 2 node      : $ROS2_TALKER_NODE"
echo "  Drake Python    : $CONDA_PYTHON"
echo "  ROS 2 Python    : $SYSTEM_PYTHON"
echo "  Topic           : /drake_hello"
echo "  Args            : $@"
echo "======================================================="
echo ""

# Source ROS 2, then pipe Drake stdout → ROS 2 talker node
source "$ROS2_SETUP"

"$CONDA_PYTHON" "$DRAKE_TALKER" "$@" \
    | "$SYSTEM_PYTHON" "$ROS2_TALKER_NODE"
