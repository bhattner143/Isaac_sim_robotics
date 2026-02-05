#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define project paths
PLANT_DIR="$SCRIPT_DIR/model_using_onshape_to_robot/ball"
ONSHAPE_DIR="$SCRIPT_DIR/model_using_onshape_to_robot"

# Change to the ball directory
cd "$PLANT_DIR"

# Load Onshape API credentials from .env file
source "$ONSHAPE_DIR/.env"

# Export environment variables for onshape-to-robot
export ONSHAPE_API
export ONSHAPE_ACCESS_KEY
export ONSHAPE_SECRET_KEY

# Initialize conda for bash shell if not already done
if ! command -v conda &> /dev/null; then
    # Try to initialize conda from common locations
    if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        source "/opt/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    fi
fi

# Run onshape-to-robot with config.json
echo "Converting ball from Onshape..."
echo "Working directory: $PLANT_DIR"
conda run -n pydrake onshape-to-robot config.json

echo ""
echo "✓ Conversion complete!"
echo "Generated files in: $PLANT_DIR"
echo "  - ball.urdf"
echo "  - assets/ (mesh files)"