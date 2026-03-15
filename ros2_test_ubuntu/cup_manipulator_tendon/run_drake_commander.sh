#!/bin/bash
# =============================================================================
# run_drake_commander.sh — Drake Manipulator → ROS 2 bridge
# =============================================================================
# PyDrake generates joint/EE trajectories and pipes them through the
# ROS 2 publisher node to /manip/joint_command or /manip/ee_command.
#
# Usage (from repo root):
#   bash ros2_test_ubuntu/cup_manipulator_tendon/run_drake_commander.sh
#   bash ros2_test_ubuntu/cup_manipulator_tendon/run_drake_commander.sh --mode ee_command
#   bash ros2_test_ubuntu/cup_manipulator_tendon/run_drake_commander.sh --mode joint_command --duration 60
#
# Start BEFORE run_isaac.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROS2_SETUP="$HOME/ros2_jazzy/install/local_setup.bash"
SYSTEM_PYTHON="/usr/bin/python3"
CONDA_PYTHON="/home/user/anaconda3/envs/env_isaacsim/bin/python3"

DRAKE_COMMANDER="$SCRIPT_DIR/drake_logic.py"
ROS2_PUBLISHER="$SCRIPT_DIR/ros2_publisher.py"

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ ! -f "$ROS2_SETUP" ]]; then
    echo "ERROR: ROS 2 not found at $ROS2_SETUP"
    echo "       See ros2_test_ubuntu/INSTALLATION_ROS2_JAZZY_UBUNTU.md"
    exit 1
fi

if [[ ! -f "$CONDA_PYTHON" ]]; then
    echo "ERROR: env_isaacsim conda env not found at $CONDA_PYTHON"
    exit 1
fi

# ── Info ──────────────────────────────────────────────────────────────────────
echo "======================================================="
echo "  Drake Manipulator Commander → ROS 2 Bridge"
echo "======================================================="
echo "  Drake script     : $DRAKE_COMMANDER"
echo "  ROS 2 node       : $ROS2_PUBLISHER"
echo "  Drake Python     : $CONDA_PYTHON"
echo "  ROS 2 Python     : $SYSTEM_PYTHON"
echo "  Topics           : /manip/joint_command, /manip/ee_command"
echo "  Args             : $@"
echo "======================================================="
echo ""
echo "Start Isaac Sim in another terminal:"
echo "  bash ros2_test_ubuntu/cup_manipulator_tendon/run_isaac.sh [--mode ...]"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
source "$ROS2_SETUP"

"$CONDA_PYTHON" "$DRAKE_COMMANDER" "$@" \
    | "$SYSTEM_PYTHON" "$ROS2_PUBLISHER"
