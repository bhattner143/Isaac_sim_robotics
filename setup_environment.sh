#!/bin/bash
# =============================================================================
# setup_environment.sh — Recreate the env_isaacsim conda environment
# =============================================================================
# Tested on: Ubuntu 24.04 LTS, NVIDIA RTX 5090, Driver 590.48
#
# This project uses ONE conda environment (env_isaacsim) for everything:
#   - Isaac Sim 5.1.0 (GPU-accelerated PhysX simulation)
#   - PyDrake (analytical dynamics, Meshcat visualization)
#   - PyTorch 2.7 (optional, for RL/learning)
#   - trimesh (STL→OBJ mesh conversion)
#
# Usage:
#   chmod +x setup_environment.sh
#   ./setup_environment.sh
#
# After installation:
#   conda activate env_isaacsim
# =============================================================================

set -e

ENV_NAME="env_isaacsim"
PYTHON_VERSION="3.11"

echo "============================================================"
echo "  Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
echo "============================================================"

# ── 1. Create conda environment ──────────────────────────────────────────────
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ── 2. Isaac Sim (must be installed first — pins numpy, etc.) ────────────────
echo ""
echo "Installing Isaac Sim 5.1.0..."
pip install isaacsim==5.1.0.0

# ── 3. Core scientific packages ──────────────────────────────────────────────
echo ""
echo "Installing core packages..."
pip install \
    numpy==1.26.0 \
    scipy==1.15.3 \
    matplotlib==3.10.3 \
    trimesh==4.5.1 \
    termcolor==3.3.0 \
    pillow==11.3.0 \
    opencv-python-headless==4.11.0.86 \
    numpy-stl==3.2.0

# ── 4. PyDrake (Drake robotics toolkit) ─────────────────────────────────────
echo ""
echo "Installing PyDrake..."
pip install pydrake

# ── 5. PyTorch (CUDA — for RL and learning, optional) ───────────────────────
echo ""
echo "Installing PyTorch 2.7 (CUDA)..."
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# ── 6. USD support ──────────────────────────────────────────────────────────
echo ""
echo "Installing USD..."
pip install usd-core==25.11

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Environment '$ENV_NAME' created successfully!"
echo ""
echo "  Activate with:"
echo "    conda activate $ENV_NAME"
echo ""
echo "  Verify Isaac Sim:"
echo "    python -c 'import isaacsim; print(isaacsim.__version__)'"
echo ""
echo "  Verify PyDrake:"
echo "    python -c 'import pydrake; print(pydrake.__version__)'"
echo "============================================================"
