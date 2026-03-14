#!/bin/bash
# =============================================================================
# run_cube_isaac.sh — ROS 2 subscriber → Isaac Sim bridge
# =============================================================================
# Receives /cube_target_pos topic and moves a blue cube inside Isaac Sim.
#
# Usage (from repo root):
#   bash ros2_test_ubuntu/launch/run_cube_isaac.sh
#
# Start AFTER run_cube_commander.sh in a separate terminal.

set -e

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS2_TEST_DIR="$(dirname "$LAUNCH_DIR")"

ROS2_SETUP="$HOME/ros2_jazzy/install/local_setup.bash"
SYSTEM_PYTHON="/usr/bin/python3"
CONDA_PYTHON="/home/user/anaconda3/envs/env_isaacsim/bin/python3"

SCENARIO_DIR="$LAUNCH_DIR"  # run_isaac.sh lives inside cube_commander/
ROS2_LISTENER_NODE="$SCENARIO_DIR/ros2_subscriber.py"
ISAAC_CUBE_TEST="$SCENARIO_DIR/isaac_sim.py"

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
echo "  ROS 2 Bridge → Isaac Sim Cube Mover"
echo "======================================================="
echo "  ROS 2 node       : $ROS2_LISTENER_NODE"
echo "  Isaac Sim script : $ISAAC_CUBE_TEST"
echo "  ROS 2 Python     : $SYSTEM_PYTHON"
echo "  Isaac Sim Python : $CONDA_PYTHON"
echo "  Topic            : /cube_target_pos (geometry_msgs/Point)"
echo "======================================================="
echo ""
echo "Waiting for Drake commander in another terminal:"
echo "  bash ros2_test_ubuntu/cube_commander/run_commander.sh"
echo ""
echo "Isaac Sim window will open momentarily..."
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
source "$ROS2_SETUP"

"$SYSTEM_PYTHON" "$ROS2_LISTENER_NODE" \
    | "$CONDA_PYTHON" "$ISAAC_CUBE_TEST"
