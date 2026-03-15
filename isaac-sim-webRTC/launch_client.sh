#!/bin/bash
# =============================================================================
# launch_client.sh — Connect to Isaac Sim WebRTC streaming
# =============================================================================
# Tries to connect to a running ./run_server.sh instance using the best
# available client (Omniverse Streaming Client → browser → manual).
#
# Usage:
#   ./launch_client.sh           # wait for server, then launch client
#   ./launch_client.sh --nowait  # skip port readiness check
# =============================================================================

WEBRTC_PORT=49100
HTTP_PORT=8011
WEBRTC_HOST="localhost"

# -----------------------------------------------------------------------------
# 1. Optionally wait for the server to be ready
# -----------------------------------------------------------------------------
NOWAIT=false
for arg in "$@"; do
    [ "$arg" = "--nowait" ] && NOWAIT=true
done

if ! $NOWAIT; then
    echo "Waiting for streaming server on port $WEBRTC_PORT..."
    READY=false
    for i in $(seq 1 60); do
        if nc -z "$WEBRTC_HOST" "$WEBRTC_PORT" 2>/dev/null; then
            echo "  Server is up! (${i}s elapsed)"
            READY=true
            break
        fi
        if [ $((i % 5)) -eq 0 ]; then
            echo "  Still waiting... (${i}s)"
        fi
        sleep 1
    done
    if ! $READY; then
        echo ""
        echo "[WARNING] Server did not respond on port $WEBRTC_PORT after 60s."
        echo "          Is ./run_server.sh running? Continuing anyway..."
    fi
fi

echo ""
echo "============================================================"
echo "  Isaac Sim WebRTC Client Launcher"
echo "  Target: ${WEBRTC_HOST}:${WEBRTC_PORT}"
echo "============================================================"

# -----------------------------------------------------------------------------
# 2. Look for the AppImage bundled in this folder (highest priority)
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPIMAGE="$(ls "$SCRIPT_DIR"/isaacsim-webrtc-streaming-client-*.AppImage 2>/dev/null | head -1)"

if [ -n "$APPIMAGE" ]; then
    echo "Found bundled AppImage: $(basename "$APPIMAGE")"
    echo ""
    echo "  When the client opens, enter this server address in the connection form:"
    echo "    Server: ${WEBRTC_HOST}      Port: ${WEBRTC_PORT}"
    echo ""
    echo "Launching..."
    exec "$APPIMAGE" 2>/dev/null
fi

# -----------------------------------------------------------------------------
# 3. Look for the Omniverse Streaming Client binary in standard locations
# -----------------------------------------------------------------------------
STREAMING_CLIENT=""
SEARCH_PATHS=(
    "$HOME/.local/share/ov/pkg/streaming-client"*/streaming_client
    "$HOME/.local/share/ov/pkg/streaming-client"*/omni-streaming-client
    "/opt/ov/pkg/streaming-client"*/streaming_client
    "/opt/ov/pkg/streaming-client"*/omni-streaming-client
    "$HOME/Downloads/streaming-client"*/streaming_client
)

for path in "${SEARCH_PATHS[@]}"; do
    # Expand glob manually
    for match in $path; do
        if [ -f "$match" ] && [ -x "$match" ]; then
            STREAMING_CLIENT="$match"
            break 2
        fi
    done
done

if [ -n "$STREAMING_CLIENT" ]; then
    echo "Found Omniverse Streaming Client: $STREAMING_CLIENT"
    echo "  Connect to: ${WEBRTC_HOST}:${WEBRTC_PORT}"
    echo "Launching..."
    exec "$STREAMING_CLIENT" 2>/dev/null
fi

# -----------------------------------------------------------------------------
# 3. Try the web-based streaming client (served by Isaac Sim HTTP API)
# -----------------------------------------------------------------------------
WEB_CLIENT_URL="http://${WEBRTC_HOST}:${HTTP_PORT}/streaming/client/"

echo "Omniverse Streaming Client not found."
echo "Trying browser at: $WEB_CLIENT_URL"

if command -v xdg-open &>/dev/null; then
    xdg-open "$WEB_CLIENT_URL" 2>/dev/null && exit 0
elif command -v google-chrome &>/dev/null; then
    google-chrome "$WEB_CLIENT_URL" &>/dev/null & exit 0
elif command -v firefox &>/dev/null; then
    firefox "$WEB_CLIENT_URL" &>/dev/null & exit 0
fi

# -----------------------------------------------------------------------------
# 4. Manual instructions fallback
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Cannot auto-launch a client. Connect manually:"
echo ""
echo "  Option A: Omniverse Streaming Client (recommended)"
echo "    Download: https://docs.omniverse.nvidia.com/streaming-client/"
echo "    Connect to: ${WEBRTC_HOST}:${WEBRTC_PORT}"
echo ""
echo "  Option B: Browser (if Isaac Sim HTTP API is serving the client)"
echo "    Open: $WEB_CLIENT_URL"
echo ""
echo "  Option C: WebRTC raw (for dev/testing)"
echo "    WebSocket signaling: ws://${WEBRTC_HOST}:${WEBRTC_PORT}"
echo "============================================================"
