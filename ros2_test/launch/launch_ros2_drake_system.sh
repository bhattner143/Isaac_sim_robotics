#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Launch the ROS 2 Drake cable-manipulator system (plant + controller)
#
# Uses the same cable (tendon) manipulator plant as
# script_cup_manipulator_pendulam_with_spring_damper.py.
#
# Usage:
#   conda activate pydrake_ros2
#   bash ros2_test/launch_ros2_drake_system.sh [--q-goal 60,-120] [--duration 3]
#
# This starts TWO processes:
#   1. Plant node      — Drake simulation, publishes /joint_states
#   2. Controller node — Computed-torque, publishes /torque_command
#
# Stop with Ctrl-C (kills both).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Defaults (can be overridden on the command line)
Q_GOAL="${1:---q-goal}"
Q_GOAL_VAL="${2:-60,-120}"
DUR_FLAG="${3:---duration}"
DUR_VAL="${4:-3.0}"

echo "══════════════════════════════════════════════════════════════"
echo "  ROS 2 Drake Cable-Manipulator System"
echo "══════════════════════════════════════════════════════════════"
echo "  Plant node:      ros2_drake_plant_node.py  (manipulator_cable)"
echo "  Controller node: ros2_computed_torque_controller_node.py"
echo "  Goal angles:     ${Q_GOAL_VAL}°"
echo "  Duration:        ${DUR_VAL}s"
echo "══════════════════════════════════════════════════════════════"
echo ""

cleanup() {
    echo ""
    echo "Shutting down …"
    kill 0 2>/dev/null
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

# 1) Start plant node in the background
echo "[1/2] Starting Drake cable-manipulator plant node …"
python "${SCRIPT_DIR}/../nodes/ros2_drake_plant_node.py" --timestep 0.002 --rate 500 &
PLANT_PID=$!
sleep 3  # give plant time to build diagram and start publishing

# 2) Start controller node in the background
echo "[2/2] Starting computed-torque controller node …"
python "${SCRIPT_DIR}/../nodes/ros2_computed_torque_controller_node.py" \
    --kp 10000 --kd 400 --tau-max 10 --rate 500 \
    --mode min-jerk --q-start 0,0 --q-goal "${Q_GOAL_VAL}" --duration "${DUR_VAL}" &
CTRL_PID=$!

echo ""
echo "Both nodes running (plant PID=$PLANT_PID, controller PID=$CTRL_PID)."
echo "Press Ctrl-C to stop."
echo ""
echo "Useful commands in another terminal:"
echo "  ros2 topic echo /joint_states"
echo "  ros2 topic echo /torque_command"
echo "  ros2 topic echo /ee_position"
echo "  ros2 topic hz /joint_states"
echo ""

wait
