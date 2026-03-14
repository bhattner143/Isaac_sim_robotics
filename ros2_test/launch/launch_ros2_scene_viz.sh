#!/usr/bin/env bash
# launch_scene_viz_ros2.sh
# ────────────────────────
# Launches the scene-viz plant node (Drake + Meshcat) and the joint position
# commander node side by side via ROS 2 topics.
#
# Usage:
#   bash ros2_test/launch_scene_viz_ros2.sh [--traj sine|hold|step]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Parse optional trajectory flag (default: sine) ──────────────────────────
TRAJ="${1:-sine}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Scene-Viz with ROS 2  —  Cable Manipulator (Drake)        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Plant node  : ros2_drake_plant_node.py --mode scene-viz   ║"
echo "║  Commander   : ros2_joint_position_commander_node.py       ║"
echo "║  Trajectory  : $TRAJ                                       "
echo "╚══════════════════════════════════════════════════════════════╝"

# ── 1. Start the plant node in the background ───────────────────────────────
echo ""
echo "▸ Starting scene-viz plant node …"
python "$SCRIPT_DIR/../nodes/ros2_drake_plant_node.py" \
    --mode scene-viz \
    --rate 30 \
    &
PLANT_PID=$!
echo "  Plant PID = $PLANT_PID"

# Give the plant a moment to load the URDF and start Meshcat
sleep 3

# ── 2. Start the commander node in the foreground ───────────────────────────
echo ""
echo "▸ Starting joint position commander (traj=$TRAJ) …"
python "$SCRIPT_DIR/../nodes/ros2_scene_viz_joint_position_commander_node.py" \
    --traj "$TRAJ" \
    --rate 30 \
    &
CMD_PID=$!
echo "  Commander PID = $CMD_PID"

# ── 3. Wait for Ctrl-C, then clean up both ──────────────────────────────────
cleanup() {
    echo ""
    echo "▸ Stopping nodes …"
    kill "$CMD_PID" 2>/dev/null || true
    kill "$PLANT_PID" 2>/dev/null || true
    wait "$CMD_PID" 2>/dev/null || true
    wait "$PLANT_PID" 2>/dev/null || true
    echo "✓ Done."
}
trap cleanup EXIT INT TERM

echo ""
echo "Both nodes running. Press Ctrl-C to stop."
wait
