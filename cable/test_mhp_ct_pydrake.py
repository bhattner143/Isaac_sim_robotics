"""
test_mhp_ct_pydrake.py
──────────────────────
Thin entry point for MHP computed-torque simulation.

Delegates to ``script_mhp_manipulator_ct_pydrake.py`` at the repo root so the
Drake diagram, controller, and cable modules stay in their respective folders:

  robots/mhp_manipulator.py
  controller/mhp_ct_controller.py
  cable/simulation_mhp_ct.py
  cable/*_mhp.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "script_mhp_manipulator_ct_pydrake.py"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

runpy.run_path(str(_SCRIPT), run_name="__main__")
