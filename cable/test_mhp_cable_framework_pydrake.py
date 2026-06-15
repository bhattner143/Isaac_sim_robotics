"""
test_mhp_cable_framework_pydrake.py
───────────────────────────────────
Thin entry point for MHP Plant A cable-framework simulation.

Delegates to ``script_mhp_manipulator_cable_framework_pydrake.py`` at repo root.

Stack::

  Plant A (MultibodyPlant)
  CT → MHPCableFramework (W(q), tension, dummy MIT) → actuation
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "script_mhp_manipulator_cable_framework_pydrake.py"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

runpy.run_path(str(_SCRIPT), run_name="__main__")
