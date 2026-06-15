"""Configuration helpers for the MHP (manipulator hybrid planar) robot."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from configs.robot.robot_types import JointConfig, ManipulatorConfig

_REPO = Path(__file__).resolve().parents[2]
MHP_DIR = _REPO / "model_using_onshape_to_robot" / "manipulator_hybrid_planar_fusion"
MHP_URDF = MHP_DIR / "manipulator_hybrid_planar_fusion_obj.urdf"


def create_mhp_config(
    *,
    urdf_path: str | Path | None = None,
    joint_angles: dict[str, float] | None = None,
    damping: tuple[float, float] = (0.05, 0.05),
    stiffness: tuple[float, float] = (0.0, 0.0),
    tilt_roll_deg: float = 0.0,
    tilt_pitch_deg: float = 0.0,
    motor_name: str | None = None,
) -> ManipulatorConfig:
    """Build a :class:`ManipulatorConfig` for the real MHP URDF."""
    urdf = Path(urdf_path) if urdf_path is not None else MHP_URDF
    angles = joint_angles or {
        "jt_upper_base": 0.0,
        "jt_lower_upper": 0.0,
    }
    return ManipulatorConfig(
        name="mhp_manipulator",
        urdf_path=str(urdf),
        joint_configs={
            "jt_upper_base": JointConfig(
                position=float(angles.get("jt_upper_base", 0.0)),
                damping=float(damping[0]),
                stiffness=float(stiffness[0]),
            ),
            "jt_lower_upper": JointConfig(
                position=float(angles.get("jt_lower_upper", 0.0)),
                damping=float(damping[1]),
                stiffness=float(stiffness[1]),
            ),
        },
        tilt_roll_deg=tilt_roll_deg,
        tilt_pitch_deg=tilt_pitch_deg,
        motor_name=motor_name,
        package_map={"assets": str(MHP_DIR / "assets")},
    )
