"""
robots/cup_manipulator_tendon_with_exo.py
──────────────────────────────────────────
Thin wrapper around CupManipulatorTendon that adds exosuit cable-rig
support (Method B — centred elbow pulley).

The exo cable rig is purely for **visualisation** of the two exo cables,
matching the drive-cable ``init_cable_rig()`` / ``compute_tangents()`` API.
The actual exo force model lives in ``actuators/sea_exo.py``.

Usage::

    from robots.cup_manipulator_tendon_with_exo import CupManipulatorTendonWithExo

    manip = CupManipulatorTendonWithExo(config)
    # ... load URDF, finalize plant ...
    manip.init_cable_rig(urdf_path, springs_enabled=True)
    manip.init_exo_cable_rig(urdf_path, springs_enabled=True)
    manip.compute_tangents(plant, plant_ctx)        # drive cables
    manip.compute_exo_tangents(plant, plant_ctx)    # exo cables
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robots.cup_manipulator_tendon import CupManipulatorTendon
from cable.pulley import PulleyBase, _parse_urdf_part_origins
from cable.cable_with_exo_springs_elbow_follow import (
    ExoElbowPulleyBig,
    ExoCableRig,
)
from configs.robot.robot_types import ManipulatorConfig
from termcolor import colored


class CupManipulatorTendonWithExo(CupManipulatorTendon):
    """Cable-driven 2-DOF manipulator with exosuit cable support.

    Extends CupManipulatorTendon with an ``ExoCableRig`` for Method B
    (centred elbow pulley) exo cable visualisation.  The exo pulley
    radius is the pitch radius of the big shared elbow pulley.
    """

    EXO_PULLEY_RADIUS: float = 0.04775   # from ExoElbowPulleyBig mesh

    def __init__(self, config: ManipulatorConfig, enable_visualization: bool = True):
        super().__init__(config, enable_visualization=enable_visualization)
        self.exo_rig: ExoCableRig | None = None

    def init_exo_cable_rig(
        self,
        urdf_path: str | None = None,
        assets_dir: str | None = None,
        springs_enabled: bool = True,
    ) -> None:
        """Initialize exo cable rig.  Call after the plant is built."""
        if urdf_path is None:
            urdf_path = self.config.urdf_path
        if assets_dir is None:
            assets_dir = str(Path(urdf_path).parent / "assets")
        PulleyBase._urdf_origins = _parse_urdf_part_origins(urdf_path)
        PulleyBase.assets_dir = assets_dir
        self.exo_rig = ExoCableRig(springs_enabled=springs_enabled)
        print(colored(
            f"✓ Exo cable rig initialised  (springs={'on' if springs_enabled else 'off'})",
            "green",
        ))

    def compute_exo_tangents(self, plant, plant_context) -> None:
        """Recompute exo cable tangent contacts at the current configuration."""
        if self.exo_rig is None:
            raise RuntimeError("init_exo_cable_rig() must be called first")
        self.exo_rig.compute_tangents(plant, plant_context, self)
