#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# docker_stop.sh
#
# Step 3 of 3 — Stop and remove the drake-ros Docker container.
#
# Usage
# ─────────────────────────────────────────────────────────────────────────────
#   bash ros2_test/docker_stop.sh
# ─────────────────────────────────────────────────────────────────────────────

CONTAINER_NAME="drake_ros_container"

if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "  Stopping container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME"
    echo "  Done."
else
    echo "  Container '$CONTAINER_NAME' is not running."
fi
