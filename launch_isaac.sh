#!/bin/bash
# ISAAC Sim Launcher - Uses ~/isaacsim

ISAAC_SIM_PATH="$HOME/isaacsim"
RENDER_OPTS="--/app/window/asyncRendering=true --/app/window/drawOnMainThread=false --/renderer/active=RayTracedLighting --disable-ui-cache"

case "$1" in
    "gui")
        cd "$ISAAC_SIM_PATH" && ./isaac-sim.sh $RENDER_OPTS
        ;;
    "python")
        SCRIPT_PATH="$2"
        # Convert to absolute path if relative
        [[ "$SCRIPT_PATH" != /* ]] && SCRIPT_PATH="$(pwd)/$SCRIPT_PATH"
        cd "$ISAAC_SIM_PATH" && ./python.sh "$SCRIPT_PATH" "${@:3}"
        ;;
    *)
        echo "Usage: $0 {gui|python <script>}"
        echo "Location: $ISAAC_SIM_PATH"
        ;;
esac
