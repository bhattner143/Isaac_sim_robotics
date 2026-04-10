"""
rl/cable_viz.py
───────────────
DEPRECATED — moved to ``project_utils/viz_cables_isaacsim.py``.

This file is a backward-compatible import redirect.  Update your
imports to::

    from project_utils.viz_cables_isaacsim import CableVisualizerIsaac
    from project_utils.viz_cables_isaacsim import draw_cables_usd, update_cables_usd
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Re-export everything from the new location
from project_utils.viz_cables_isaacsim import (  # noqa: F401
    CableVisualizerIsaac,
    draw_cables_usd,
    update_cables_usd,
    _usd_cylinder,
    _route_color,
)
