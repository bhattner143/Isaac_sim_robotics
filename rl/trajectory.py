"""
rl/trajectory.py
-----------------
DEPRECATED — moved to ``controller/trajectory.py``.

This file is a backward-compatible import redirect.  Update your
imports to::

    from controller.trajectory import RectTrajectory, build_move_to_start
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Re-export everything from the new location
from controller.trajectory import (  # noqa: F401
    LoopingCubicTrajectory,
    RectTrajectory,
    CircleTrajectory,
    LineTrajectory,
    MoveToStartSpline,
    PreambleTrajectorySource,
    build_move_to_start,
)
