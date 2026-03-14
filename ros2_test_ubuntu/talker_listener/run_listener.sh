#!/bin/bash
# =============================================================================
# run_listener.sh — Launch ROS 2 Drake Listener
# =============================================================================
# Starts the ROS 2 listener node (system Python 3.12) that subscribes
# to /drake_hello and prints all messages received from Drake.
#
# Usage:
#   bash ros2_test_ubuntu/run_listener.sh
#
# Run from repo root:
#   cd /home/user/Documents/isaac_sim_robotics
#   bash ros2_test_ubuntu/run_listener.sh

set -e

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS2_TEST_DIR="$(dirname "$LAUNCH_DIR")"

ROS2_SETUP="$HOME/ros2_jazzy/install/local_setup.bash"
SYSTEM_PYTHON="/usr/bin/python3"

SCENARIO_DIR="$LAUNCH_DIR"  # run_listener.sh lives inside talker_listener/
ROS2_LISTENER_NODE="$SCENARIO_DIR/ros2_listener_node.py"

# Validate
if [[ ! -f "$ROS2_SETUP" ]]; then
    echo "ERROR: ROS 2 not found at $ROS2_SETUP"
    echo "       See ros2_test_ubuntu/INSTALLATION_ROS2_JAZZY_UBUNTU.md"
    exit 1
fi

if [[ ! -f "$ROS2_LISTENER_NODE" ]]; then
    echo "ERROR: ROS 2 listener node not found at $ROS2_LISTENER_NODE"
    exit 1
fi

echo "======================================================="
echo "  Drake Hello World → ROS 2 Listener"
echo "======================================================="
echo "  ROS 2 node  : $ROS2_LISTENER_NODE"
echo "  ROS 2 Python: $SYSTEM_PYTHON"
echo "  Topic       : /drake_hello"
echo "======================================================="
echo ""
echo "Start the talker in another terminal:"
echo "  bash ros2_test_ubuntu/launch/run_talker.sh"
echo ""

source "$ROS2_SETUP"
"$SYSTEM_PYTHON" "$ROS2_LISTENER_NODE"
