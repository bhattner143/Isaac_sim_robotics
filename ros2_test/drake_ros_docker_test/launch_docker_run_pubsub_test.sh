#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_docker_run_pubsub_test.sh
#
# Step 2 of 3 — Run the Drake-ROS pub/sub test inside the running container.
#
# HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
# Drake has NO pip wheel for ARM64 Linux (the Docker container architecture on
# Apple Silicon).  The drake_ros.core C++ extension is also only importable via
# Bazel's runfiles mechanism — direct `python3` execution cannot work.
#
# Instead, we use `bazel run //examples/pubsub_test:pubsub_test_py` from
# inside drake_ros_examples/, exactly like the official rs_flip_flop and
# multirobot examples.  Bazel's ros_py_binary rule automatically:
#   • Fetches @drake//bindings/pydrake (no pip needed)
#   • Links @drake_ros//:drake_ros_py via runfiles
#   • Configures DDS middleware (rmw_cyclonedds_cpp)
#   • Sets up PYTHONPATH and ROS environment
#
# Requires launch_docker_start.sh to have been run first.
# Can be run multiple times — Bazel caches the build after first run.
#
# Usage
# ─────────────────────────────────────────────────────────────────────────────
#   bash ros2_test/drake_ros_docker_test/launch_docker_run_pubsub_test.sh [DURATION] [PREFIX] [TIMESTEP] [JOBS] [ROS_DISTRO]
#
#   $1  DURATION     Sim duration in seconds        (default: 15.0)
#   $2  PREFIX       ROS topic name prefix          (default: /drake_test)
#   $3  TIMESTEP     Sim timestep in seconds        (default: 0.1)
#   $4  JOBS         Bazel parallel jobs            (default: 4)
#   $5  ROS_DISTRO   ROS 2 distro inside container  (default: jazzy)
#
# Examples
# ─────────────────────────────────────────────────────────────────────────────
#   bash ros2_test/drake_ros_docker_test/launch_docker_run_pubsub_test.sh
#   bash ros2_test/drake_ros_docker_test/launch_docker_run_pubsub_test.sh 30.0
#   bash ros2_test/drake_ros_docker_test/launch_docker_run_pubsub_test.sh 30.0 /my_robot 0.05
#   bash ros2_test/drake_ros_docker_test/launch_docker_run_pubsub_test.sh 10.0 /drake_test 0.1 8
#
# Monitor topics (separate terminal while test is running)
# ─────────────────────────────────────────────────────────────────────────────
#   docker exec -it drake_ros_container bash
#   source /opt/ros/jazzy/setup.bash
#   ros2 topic echo /drake_test/echo
#   ros2 topic pub  /drake_test/echo std_msgs/msg/String "data: 'hello'"
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DURATION="${1:-15.0}"
PREFIX="${2:-/drake_test}"
TIMESTEP="${3:-0.1}"
JOBS="${4:-4}"
ROS_DISTRO="${5:-jazzy}"

CONTAINER_NAME="drake_ros_container"
BAZEL_TARGET="//examples/pubsub_test:pubsub_test_py"
EXAMPLES_DIR="/drake-ros/drake_ros_examples"

# ── Guard: container must be running ─────────────────────────────────────────
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "[ERROR] Container '$CONTAINER_NAME' is not running."
    echo "        Start it first with:  bash ros2_test/drake_ros_docker_test/launch_docker_start.sh"
    exit 1
fi

# Normalise prefix (ensure leading slash, no trailing slash)
PREFIX="${PREFIX%/}"
[[ "$PREFIX" != /* ]] && PREFIX="/$PREFIX"

echo "════════════════════════════════════════════════════════════"
echo "  Drake-ROS Pub/Sub Test  (via bazel run)"
echo "  Container : $CONTAINER_NAME"
echo "  Bazel dir : $EXAMPLES_DIR"
echo "  Target    : $BAZEL_TARGET"
echo "  Topics    : ${PREFIX}/echo   ${PREFIX}/status"
echo "  Duration  : $DURATION s"
echo "  Timestep  : $TIMESTEP s"
echo "  Bazel jobs: $JOBS"
echo "  ROS distro: $ROS_DISTRO"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "NOTE: First run fetches Drake and builds everything — this can"
echo "      take 15-30 minutes.  Subsequent runs use Bazel's cache."
echo ""

docker exec --interactive --tty "$CONTAINER_NAME" bash -c "
    set -eo pipefail

    # ── 1. Source ROS 2 (disable -u: ROS setup.bash uses unbound vars) ────────
    ROS_SETUP='/opt/ros/${ROS_DISTRO}/setup.bash'
    if [ ! -f \"\$ROS_SETUP\" ]; then
        FOUND=\$(ls /opt/ros/ 2>/dev/null | head -1)
        echo \"[WARN] '${ROS_DISTRO}' not found, using '\${FOUND}'\"
        ROS_SETUP=\"/opt/ros/\${FOUND}/setup.bash\"
    fi
    set +u; source \"\$ROS_SETUP\"; set -u
    echo '[OK] ROS 2 sourced: '\$ROS_SETUP

    # ── 2. Enter drake_ros_examples workspace (has MODULE.bazel) ─────────────
    cd '${EXAMPLES_DIR}'
    echo \"[OK] Working directory: \$(pwd)\"

    # ── 3. Run via bazel (builds + runs in one step, uses cache on re-runs) ──
    echo '[INFO] Running: bazel run --jobs=${JOBS} ${BAZEL_TARGET}'
    echo '[INFO] Pass -- before script args so Bazel does not consume them.'
    bazel run --jobs='${JOBS}' '${BAZEL_TARGET}' -- \
        --duration '${DURATION}' \
        --prefix   '${PREFIX}'   \
        --timestep '${TIMESTEP}'
"
