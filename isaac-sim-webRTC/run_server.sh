#!/bin/bash
# =============================================================================
# run_server.sh — Start Isaac Sim with WebRTC streaming
# =============================================================================
# Launches the standalone Isaac Sim in headless streaming mode.
# The streaming server listens on:
#   - Port 49100  : WebRTC websocket signaling (for Omniverse Streaming Client)
#   - Port 8011   : REST/HTTP API (FastAPI, for the web streaming client)
#
# Usage:
#   ./run_server.sh              # default (no ROS env)
#   ./run_server.sh --no-ros-env # skip ROS environment setup
# =============================================================================

ISAAC_SIM_DIR="/home/user/Documents/isaac-sim"
SCRIPT="$ISAAC_SIM_DIR/isaac-sim.streaming.sh"

if [ ! -f "$SCRIPT" ]; then
    echo "[ERROR] isaac-sim.streaming.sh not found at: $ISAAC_SIM_DIR"
    echo "        Update ISAAC_SIM_DIR in this script."
    exit 1
fi

echo "============================================================"
echo "  Isaac Sim WebRTC Streaming Server"
echo "============================================================"
echo "  WebRTC port  : 49100  (Omniverse Streaming Client)"
echo "  HTTP API port: 8011   (web client: http://localhost:8011/streaming/client/)"
echo ""
echo "  Starting server... (Ctrl+C to stop)"
echo "============================================================"
echo ""

exec "$SCRIPT" "$@"
