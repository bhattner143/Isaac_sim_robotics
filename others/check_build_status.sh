#!/bin/bash
# Script to check ISAAC Sim build status

echo "================================"
echo "ISAAC Sim Build Status Checker"
echo "================================"
echo

# Check if build directory exists
if [ -d "/home/dipankar/Documents/isac_sim_pydrake/isaacsim/_build" ]; then
    echo "✓ Build directory exists"
    echo "  Contents:"
    ls -lh /home/dipankar/Documents/isac_sim_pydrake/isaacsim/_build/ 2>/dev/null | head -10
else
    echo "⏳ Build directory not yet created (build in progress)"
fi

echo
echo "Cache directory size:"
du -sh ~/.cache/packman 2>/dev/null || echo "  Not created yet"

echo
echo "Checking for build process:"
ps aux | grep -E "(build\.sh|premake|packman)" | grep -v grep

echo
echo "================================"
echo "To run ISAAC Sim after build completes:"
echo "  cd _build/linux-x86_64/release"
echo "  ./isaac-sim.sh"
echo "================================"
