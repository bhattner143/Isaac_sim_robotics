#!/usr/bin/env bash
# Idempotent per-boot startup: bring up a persistent virtual X display on :99
# so the repo's GUI-forcing PyDrake scripts (matplotlib TkAgg, Meshcat) run
# headlessly. Prevents duplicates and returns promptly.
set -euo pipefail

if pgrep -x Xvfb >/dev/null 2>&1; then
    echo "Xvfb already running."
    exit 0
fi

nohup Xvfb :99 -screen 0 1280x1024x24 >/tmp/xvfb.log 2>&1 &

# Wait briefly for the display socket to appear.
for _ in $(seq 1 20); do
    if [ -e /tmp/.X11-unix/X99 ]; then
        echo "Xvfb started on :99."
        exit 0
    fi
    sleep 0.25
done

echo "Warning: Xvfb did not report ready within timeout; see /tmp/xvfb.log" >&2
exit 0
