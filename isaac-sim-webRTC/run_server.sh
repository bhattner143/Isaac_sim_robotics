#!/bin/bash
# =============================================================================
# run_server.sh — Stream the full Isaac Sim GUI editor over WebRTC
# =============================================================================
#
# Use this ONLY when you want to stream the Isaac Sim editor UI itself
# (load scenes manually, use the GUI tools, etc.).
#
# For Python-driven simulations, run the script directly instead:
#   python example_webrtc_scene.py        # auto-switches to env_isaacsim
#   python test_cup_manipulator_*.py      # same — no conda activate needed
#
# Usage:
#   ./run_server.sh                       # starts Isaac Sim GUI streaming
#   ./run_server.sh [extra kit args...]   # pass-through to isaac-sim.streaming.sh
#
# Ports:
#   49100  WebRTC websocket signaling  ← connect launch_client.sh here
#   8011   HTTP REST API
# =============================================================================

ISAAC_SIM_DIR="/home/user/Documents/isaac-sim"
STANDALONE_SCRIPT="$ISAAC_SIM_DIR/isaac-sim.streaming.sh"

if [ ! -f "$STANDALONE_SCRIPT" ]; then
    echo "[ERROR] Isaac Sim not found at: $ISAAC_SIM_DIR"
    echo "        Update ISAAC_SIM_DIR in this script to match your installation."
    echo ""
    echo "  For Python scripts, run them directly (no run_server.sh needed):"
    echo "    python example_webrtc_scene.py"
    exit 1
fi

echo "============================================================"
echo "  Isaac Sim GUI — WebRTC Streaming (Tailscale)"
echo "  WebRTC port  : 49100"
echo "  HTTP API port: 8011"
echo "  Starting... (Ctrl+C to stop)"
echo "============================================================"
echo ""

# Use Tailscale IP (works across internet without port forwarding)
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null)
if [ -n "$TAILSCALE_IP" ]; then
    echo "  Tailscale IP : $TAILSCALE_IP"
    echo "  Mac client   : enter $TAILSCALE_IP in the WebRTC Streaming Client app"
    echo ""
    exec "$STANDALONE_SCRIPT" \
        --/app/livestream/publicEndpointAddress="$TAILSCALE_IP" \
        --/app/livestream/port=49100 \
        "$@"
else
    echo "  [WARNING] Tailscale not running — falling back to LAN-only mode"
    echo "  Run: sudo tailscale up"
    exec "$STANDALONE_SCRIPT" "$@"
fi
