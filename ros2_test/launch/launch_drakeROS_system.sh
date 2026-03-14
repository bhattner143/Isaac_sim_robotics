#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_drakeROS_system.sh
# ─────────────────────────────────────────────────────────────────────────────
# Launches the drake-ROS cable-manipulator system (plant + controller).
#
# All ROS I/O is wired as LeafSystems inside Drake Diagrams — no raw rclpy
# timers. This is the drake-ros equivalent of launch_ros2_drake_system.sh.
#
# ── Prerequisites ─────────────────────────────────────────────────────────────
#   conda activate pydrake_ros2
#
# ── Arguments (all positional, all optional) ──────────────────────────────────
#   $1  Q_GOAL    — Target joint angles in DEGREES, comma-separated [q1,q2].
#                   q1 = link1_base (shoulder), q2 = link2_link1 (elbow).
#                   Positive = counter-clockwise from home (0°).
#                   Default: "60,-120"
#
#   $2  DURATION  — Trajectory duration in seconds (min-jerk only).
#                   Shorter = faster motion; longer = smoother.
#                   Default: "3.0"
#
#   $3  MODE      — Controller trajectory mode:
#                     min-jerk  — 5th-order polynomial from q-start → q-goal.
#                     hold      — Stay at q-start (q-goal ignored).
#                   Default: "min-jerk"
#
#   $4  TIMESTEP  — Plant simulation timestep in seconds.
#                   Smaller = more accurate but slower (min ~0.001).
#                   Default: "0.002"
#
# ── Usage Examples ────────────────────────────────────────────────────────────
#   # Default: move to (60°, -120°) over 3 seconds
#   bash ros2_test/launch_drakeROS_system.sh
#
#   # Move to (30°, -60°) over 2 seconds
#   bash ros2_test/launch_drakeROS_system.sh 30,-60 2.0
#
#   # Hold joint 1 at 45°, joint 2 at 0° indefinitely
#   bash ros2_test/launch_drakeROS_system.sh 45,0 3.0 hold
#
#   # Fast motion (1s) with fine timestep (1ms)
#   bash ros2_test/launch_drakeROS_system.sh 90,-90 1.0 min-jerk 0.001
#
# ── ROS Topics ────────────────────────────────────────────────────────────────
#   IN  (controller → plant): /torque_command  [std_msgs/Float64MultiArray]
#   OUT (plant → controller): /joint_states    [sensor_msgs/JointState]
#   OUT (plant):              /ee_position     [geometry_msgs/Point]
#   OUT (both):               /clock           [rosgraph_msgs/Clock]
#
# Stop with Ctrl-C (kills both nodes).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Parse positional args with defaults ──────────────────────────────────────
Q_GOAL="${1:-60,-120}"
DURATION="${2:-3.0}"
MODE="${3:-min-jerk}"
TIMESTEP="${4:-0.002}"

# ── Auto-detect which backend is available ───────────────────────────────────
# drake_ros_compat.py handles this automatically; we just report it here.
BACKEND=$(python -c "
import sys; sys.path.insert(0,'${SCRIPT_DIR}')
try:
    import drake_ros.core; print('drake_ros')
except ImportError:
    print('drake_ros_bridge')
" 2>/dev/null)

echo "══════════════════════════════════════════════════════════════"
echo "  drake-ROS Cable-Manipulator System"
echo "══════════════════════════════════════════════════════════════"
echo "  Backend     : ${BACKEND}"
echo "  Plant node  : ros2_drakeROS_plant_node.py  --mode dynamics"
echo "  Controller  : ros2_drakeROS_controller_node.py"
echo "  Traj mode   : ${MODE}"
echo "  Goal angles : ${Q_GOAL}°"
echo "  Duration    : ${DURATION}s"
echo "  Timestep    : ${TIMESTEP}s"
echo "══════════════════════════════════════════════════════════════"
if [[ "${BACKEND}" == "drake_ros" ]]; then
    echo "  ✓ Using real drake_ros C++ (Docker/Linux — native DDS transport)"
else
    echo "  ℹ Using drake_ros_bridge (pure-Python — macOS compatible)"
    echo "    To use real drake_ros: run inside the drake-ros Docker container"
    echo "    and ensure 'source /opt/ros/humble/setup.bash' + colcon build"
fi
echo "══════════════════════════════════════════════════════════════"
echo ""

cleanup() {
    echo ""
    echo "▸ Shutting down …"
    kill 0 2>/dev/null
    wait 2>/dev/null
    echo "✓ Done."
}
trap cleanup EXIT INT TERM

# ── 1) Plant node (background) ──────────────────────────────────────────────
echo "[1/2] Starting drake-ROS plant node …"
python "${SCRIPT_DIR}/../nodes/ros2_drakeROS_plant_node.py" \
    --mode dynamics \
    --timestep "${TIMESTEP}" \
    &
PLANT_PID=$!
echo "  Plant PID = ${PLANT_PID}"

# Give the plant time to load URDF, build diagram, start Meshcat
sleep 5

# ── 2) Controller node (background) ─────────────────────────────────────────
echo "[2/2] Starting drake-ROS computed-torque controller …"
python "${SCRIPT_DIR}/../nodes/ros2_drakeROS_controller_node.py" \
    --kp 10000 --kd 400 --tau-max 10 \
    --mode "${MODE}" \
    --q-start 0,0 \
    --q-goal "${Q_GOAL}" \
    --duration "${DURATION}" \
    &
CTRL_PID=$!
echo "  Controller PID = ${CTRL_PID}"

echo ""
echo "Both nodes running (plant=${PLANT_PID}, ctrl=${CTRL_PID})."
echo "Press Ctrl-C to stop."
echo ""
echo "Useful commands in another terminal:"
echo "  ros2 topic echo /joint_states"
echo "  ros2 topic echo /torque_command"
echo "  ros2 topic echo /ee_position"
echo "  ros2 topic hz /joint_states"
echo ""

wait
