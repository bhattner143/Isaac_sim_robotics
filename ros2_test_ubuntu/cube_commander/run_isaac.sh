#!/bin/bash
# =============================================================================
# run_isaac.sh — Native ROS 2 bridge → Isaac Sim
# =============================================================================
# Launches Isaac Sim which subscribes to /cube_target_pos natively via the
# isaacsim.ros2.bridge extension (no intermediate pipe process needed).
#
# Architecture:
#   Drake → [OS pipe] → ros2_publisher.py → [DDS] → isaac_sim.py
#
# Usage (from repo root):
#   bash ros2_test_ubuntu/cube_commander/run_isaac.sh
#
# Start AFTER run_drake_commander.sh in a separate terminal.

set -e

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_PYTHON="/home/user/anaconda3/envs/env_isaacsim/bin/python3"

SCENARIO_DIR="$LAUNCH_DIR"
ISAAC_CUBE_TEST="$SCENARIO_DIR/isaac_sim.py"

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ ! -f "$CONDA_PYTHON" ]]; then
    echo "ERROR: env_isaacsim conda env not found at $CONDA_PYTHON"
    echo "       Run: conda activate env_isaacsim"
    exit 1
fi

# ── Info ──────────────────────────────────────────────────────────────────────
echo "======================================================="
echo "  Isaac Sim Cube Commander (Native ROS 2 Bridge)"
echo "======================================================="
echo "  Isaac Sim script : $ISAAC_CUBE_TEST"
echo "  Isaac Sim Python : $CONDA_PYTHON"
echo "  Topic            : /cube_target_pos (geometry_msgs/Point)"
echo "  Bridge           : isaacsim.ros2.bridge (no pipe needed)"
echo "======================================================="
echo ""
echo "Waiting for Drake commander in another terminal:"
echo "  bash ros2_test_ubuntu/cube_commander/run_drake_commander.sh"
echo ""
echo "Isaac Sim window will open momentarily..."
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
"$CONDA_PYTHON" "$ISAAC_CUBE_TEST"
