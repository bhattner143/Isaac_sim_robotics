"""
rl/computed_torque_controller.py
---------------------------------
DEPRECATED — moved to ``controller/computed_torque_isaacsim.py``.

This file is a backward-compatible import redirect.  Update your
imports to::

    from controller.computed_torque_isaacsim import ComputedTorqueController
    from controller.computed_torque_isaacsim import ik_to_joint_space_references
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Re-export everything from the new location
from controller.computed_torque_isaacsim import (  # noqa: F401
    CTControllerOutput,
    ComputedTorqueController,
    ik_to_joint_space_references,
)
