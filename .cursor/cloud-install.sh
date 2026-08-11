#!/usr/bin/env bash
# Idempotent Cloud Agent install for the CPU-only PyDrake development stack.
# Safe to run repeatedly (apt install + venv + pip are all idempotent).
set -euo pipefail

cd "$(dirname "$0")/.."

# --- System packages ---------------------------------------------------------
# python3-venv : virtual environments
# python3-tk   : Tk backend that the repo's matplotlib scripts force on Linux
# xvfb         : headless virtual X display for GUI code paths (Tk / Meshcat)
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends python3-venv python3-tk xvfb

# --- Python virtual environment ---------------------------------------------
python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-pydrake.txt

# Route the repo's GUI-forcing scripts at the virtual display started by
# cloud-start.sh so plotting/Meshcat work headlessly without per-command wrappers.
if ! grep -q 'export DISPLAY=:99' .venv/bin/activate; then
    echo 'export DISPLAY=:99' >> .venv/bin/activate
fi

echo "PyDrake environment ready. Activate with: source .venv/bin/activate"
