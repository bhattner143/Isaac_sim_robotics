"""
robots/cup_manipulator_tendon_with_exo_isaac.py
───────────────────────────────────────────────
Isaac Sim counterpart of :class:`robots.cup_manipulator_tendon_with_exo.
CupManipulatorTendonWithExo` (PyDrake).

Extends :class:`CupManipulatorTendonIsaac` with exosuit-specific constants
(centred elbow pulley radius — Method B).  The URDF contains BOTH the
drive cable pulleys (HTD 5M 60T) and the exo pulleys / cables / springs
on the forearm side of the elbow joint.

Method names mirror the PyDrake class so scene-building code is
engine-agnostic.  Cable visualisation for both drive AND exo cables is
driven by a headless PyDrake plant created by
``project_utils.viz_cables_isaacsim.ExoCableVisualizerIsaac``.
"""

from __future__ import annotations

import numpy as np

from robots.cup_manipulator_tendon_isaac import (
    CupManipulatorTendonIsaac,
    create_cable_manipulator_config,
    solve_2r_ik,
    forward_kinematics_2r,
    analytical_jacobian_2r,
)

# Re-export so callers can write a single import.
__all__ = [
    "CupManipulatorTendonWithExoIsaac",
    "create_cable_manipulator_config",
    "solve_2r_ik",
    "forward_kinematics_2r",
    "analytical_jacobian_2r",
]


class CupManipulatorTendonWithExoIsaac(CupManipulatorTendonIsaac):
    """Cable-driven 2-DOF manipulator with exosuit cables (Isaac Sim).

    Loads ``manipulator_cable_exo_springs_elbow_follow_obj.urdf`` which
    contains the drive cable pulleys *plus* two antagonistic exo cables
    routed through a shared centred elbow pulley (Method B).

    The joint structure is identical to the drive-only variant:

        JT1_NAME  = "link1_base"     (q₁ — shoulder)
        JT2_NAME  = "link2_link1"    (q₂ — elbow, actuated via cable SEA)

    Only the exo pulley radius differs from the drive pulley:

        PULLEY_RADIUS      = 60·0.005/(2π) ≈ 0.04775 m   (HTD 5M 60T)
        EXO_PULLEY_RADIUS  =                0.04775 m     (same — big
                                                           shared elbow pulley)

    The two values are numerically identical by design (both pulleys are
    the HTD 5M 60T pitch radius), but we keep them as separate constants
    to match the PyDrake class API and to allow independent tuning.
    """

    # Exo elbow pulley pitch radius — same as ExoElbowPulleyBig in
    # cable/cable_with_exo_springs_elbow_follow.py.
    EXO_PULLEY_RADIUS: float = 0.04775

    @property
    def r_exo(self) -> float:
        """Exo elbow pulley radius [m]."""
        return self.EXO_PULLEY_RADIUS
