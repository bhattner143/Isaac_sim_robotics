#!/bin/bash
# =============================================================================
# run_cube_commander.sh — Drake cube commander → ROS 2 bridge
# =============================================================================
# PyDrake computes cube target positions (1cm steps along X)
# and publishes them to the /cube_target_pos ROS 2 topic.
#
# Usage (from repo root):
#   bash ros2_test_ubuntu/launch/run_cube_commander.sh [--steps 10] [--period 1.0]
#
# Start BEFORE run_cube_isaac.sh.

set -e

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS2_TEST_DIR="$(dirname "$LAUNCH_DIR")"

ROS2_SETUP="$HOME/ros2_jazzy/install/local_setup.bash"
SYSTEM_PYTHON="/usr/bin/python3"
CONDA_PYTHON="/home/user/anaconda3/envs/env_isaacsim/bin/python3"

SCENARIO_DIR="$LAUNCH_DIR"  # run_commander.sh lives inside cube_commander/
DRAKE_COMMANDER="$SCENARIO_DIR/drake_logic.py"
ROS2_COMMANDER_NODE="$SCENARIO_DIR/ros2_publisher.py"

# ── Validate ──────────────────────────────────────────────────────────────────
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

# ── Info ──────────────────────────────────────────────────────────────────────
echo "======================================================="
echo "  Drake Cube Commander → ROS 2 Bridge"
echo "======================================================="
echo "  Drake script     : $DRAKE_COMMANDER"
echo "  ROS 2 node       : $ROS2_COMMANDER_NODE"
echo "  Drake Python     : $CONDA_PYTHON"
echo "  ROS 2 Python     : $SYSTEM_PYTHON"
echo "  Topic            : /cube_target_pos (geometry_msgs/Point)"
echo "  Args             : $@"
echo "======================================================="
echo ""
echo "Start Isaac Sim in another terminal:"
echo "  bash ros2_test_ubuntu/cube_commander/run_isaac.sh"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
source "$ROS2_SETUP"

"$CONDA_PYTHON" "$DRAKE_COMMANDER" "$@" \
    | "$SYSTEM_PYTHON" "$ROS2_COMMANDER_NODE"
