#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# docker_start.sh
#
# Step 1 of 2 — Build the drake-ros Docker image and start the container.
#
# Run this ONCE. The container stays alive in the background.
# After this, use docker_run_pubsub_test.sh to run scripts inside it.
#
# Usage
# ─────────────────────────────────────────────────────────────────────────────
#   cd /Volumes/Data/Isaac_sim_robotics
#   bash ros2_test/docker_start.sh [ROS_DISTRO]
#
#   $1  ROS_DISTRO   ROS 2 distro inside container (default: jazzy)
#
# Examples
# ─────────────────────────────────────────────────────────────────────────────
#   bash ros2_test/docker_start.sh
#   bash ros2_test/docker_start.sh humble
#
# Stop the container when done
# ─────────────────────────────────────────────────────────────────────────────
#   docker stop drake_ros_container
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS2_TEST_DIR="$(dirname "$SCRIPT_DIR")"          # ros2_test/
WORKSPACE_DIR="$(dirname "$ROS2_TEST_DIR")"       # Isaac_sim_robotics/
DRAKE_ROS_DIR="$WORKSPACE_DIR/drake-ros"
ROS_DISTRO="${1:-jazzy}"

IMAGE_TAG="drake-ros-dev:local"
CONTAINER_NAME="drake_ros_container"
BAZEL_CACHE_DIR="$HOME/.cache/drake_ros_bazel_cache"
mkdir -p "$BAZEL_CACHE_DIR"

# ── Guards ────────────────────────────────────────────────────────────────────
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker not found. Install Docker Desktop and try again."
    exit 1
fi

if [ ! -d "$DRAKE_ROS_DIR" ]; then
    echo "[ERROR] drake-ros not found at: $DRAKE_ROS_DIR"
    echo "        Clone with: git clone https://github.com/RobotLocomotion/drake-ros $DRAKE_ROS_DIR"
    exit 1
fi

# ── Build image ───────────────────────────────────────────────────────────────
echo "────────────────────────────────────────────────────────────"
echo "  Building Docker image: $IMAGE_TAG"
echo "────────────────────────────────────────────────────────────"
docker build \
    --tag "$IMAGE_TAG" \
    --file "$DRAKE_ROS_DIR/.devcontainer/Dockerfile" \
    "$DRAKE_ROS_DIR"
echo "  Image ready: $IMAGE_TAG"

# ── Remove stale container if exists ──────────────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "  Removing stale container: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" > /dev/null
fi

# ── Start container in background ─────────────────────────────────────────────
# tail -f /dev/null keeps it alive without doing anything
echo ""
echo "  Starting container: $CONTAINER_NAME (detached)"
docker run \
    --detach \
    --name "$CONTAINER_NAME" \
    --privileged \
    --volume "$DRAKE_ROS_DIR":/drake-ros:cached \
    --volume "$ROS2_TEST_DIR":/ros2_test:ro \
    --volume "$BAZEL_CACHE_DIR":/root/.cache/bazel:cached \
    --workdir /drake-ros \
    "$IMAGE_TAG" \
    tail -f /dev/null

# ── Pre-build pubsub_test inside drake_ros_examples (cached for future runs) ──
# NOTE: Drake has no pip wheel for ARM64 Linux.  Everything must come from
# Bazel.  We build from drake_ros_examples/ which has its own MODULE.bazel
# and uses ros_py_binary to fetch @drake//bindings/pydrake automatically.
echo ""
echo "  Pre-building pubsub_test with Bazel (first run fetches Drake ~15-30 min)..."
docker exec "$CONTAINER_NAME" bash -c "
    set +u
    source /opt/ros/${ROS_DISTRO}/setup.bash 2>/dev/null || \
        source /opt/ros/\$(ls /opt/ros/ | head -1)/setup.bash
    set -u
    cd /drake-ros/drake_ros_examples
    bazel build --jobs=4 //examples/pubsub_test:pubsub_test_py
    echo '[OK] Bazel build complete — subsequent runs will use cache'
"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Container '$CONTAINER_NAME' is running."
echo ""
echo "  Next step — run the pub/sub test:"
echo "    bash ros2_test/drake_ros_docker_test/launch_docker_run_pubsub_test.sh"
echo ""
echo "  Open a shell inside the container:"
echo "    docker exec -it $CONTAINER_NAME bash"
echo ""
echo "  Stop when done:"
echo "    bash ros2_test/drake_ros_docker_test/launch_docker_stop.sh"
echo "════════════════════════════════════════════════════════════"
