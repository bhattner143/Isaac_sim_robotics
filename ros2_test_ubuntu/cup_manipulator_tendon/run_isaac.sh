#!/bin/bash
# =============================================================================
# run_isaac.sh — Cup Manipulator Tendon Isaac Sim ROS 2 Bridge
# =============================================================================
# Launches Isaac Sim which subscribes to joint/EE commands from Drake
# and publishes feedback back.
#
# Architecture:
#   Drake → [pipe] → ros2_publisher.py → [DDS] → isaac_sim.py (this script)
#
# Usage (from repo root):
#   bash ros2_test_ubuntu/cup_manipulator_tendon/run_isaac.sh
#   bash ros2_test_ubuntu/cup_manipulator_tendon/run_isaac.sh --mode ee_command
#   bash ros2_test_ubuntu/cup_manipulator_tendon/run_isaac.sh --render headless
#
# Start AFTER run_drake_commander.sh in a separate terminal.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_PYTHON="/home/user/anaconda3/envs/env_isaacsim/bin/python3"
ISAAC_SCRIPT="$SCRIPT_DIR/cup_manipulator_tendon_isaac_sim.py"

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ ! -f "$CONDA_PYTHON" ]]; then
    echo "ERROR: env_isaacsim conda env not found at $CONDA_PYTHON"
    echo "       Run: conda activate env_isaacsim"
    exit 1
fi

# ── Info ──────────────────────────────────────────────────────────────────────
echo "======================================================="
echo "  Cup Manipulator Tendon — Isaac Sim ROS 2 Bridge"
echo "======================================================="
echo "  Script           : $ISAAC_SCRIPT"
echo "  Python           : $CONDA_PYTHON"
echo "  Bridge           : isaacsim.ros2.bridge (native)"
echo "  Args             : $@"
echo "======================================================="
echo ""
echo "Start Drake commander in another terminal:"
echo "  bash ros2_test_ubuntu/cup_manipulator_tendon/run_drake_commander.sh"
echo ""

# ── Clean ROS 2 system Python paths ──────────────────────────────────────────
# Isaac Sim uses Python 3.11, but the system ROS 2 Jazzy was built for 3.12.
# If the system rclpy paths leak into PYTHONPATH, the bridge extension's
# internal rclpy (built for 3.11) cannot load.  Strip all ros2_jazzy entries
# and let the bridge use its own bundled libraries.

BRIDGE_EXT="$("$CONDA_PYTHON" -c "import isaacsim, pathlib; print(pathlib.Path(isaacsim.__file__).parent / 'exts' / 'isaacsim.ros2.bridge')")"

# Remove every ros2_jazzy path from PYTHONPATH
CLEAN_PYTHONPATH=""
IFS=':' read -ra PP <<< "$PYTHONPATH"
for p in "${PP[@]}"; do
    [[ "$p" != *ros2_jazzy* ]] && CLEAN_PYTHONPATH="${CLEAN_PYTHONPATH:+$CLEAN_PYTHONPATH:}$p"
done
export PYTHONPATH="$CLEAN_PYTHONPATH"

# Required for the bridge to load its internal rclpy + FastDDS transport
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="${BRIDGE_EXT}/jazzy/lib:${LD_LIBRARY_PATH}"

echo "  Bridge ext       : $BRIDGE_EXT"
echo "  RMW              : $RMW_IMPLEMENTATION"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
"$CONDA_PYTHON" "$ISAAC_SCRIPT" "$@"
