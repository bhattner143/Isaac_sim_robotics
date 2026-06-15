"""CubeMars motor configuration.

Datasheet values from AK Series Manual V3.2.0 (V3.0 driver firmware).
Keep these consistent with `actuators/motor.py` (the simulation side).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class MotorMode(IntEnum):
    """CubeMars V3 driver function IDs (used as the high byte of the
    extended CAN ID for servo modes; MIT mode uses the special prelude)."""
    DUTY            = 0
    CURRENT         = 1
    CURRENT_BRAKE   = 2
    VELOCITY        = 3
    POSITION        = 4
    SET_ORIGIN      = 5
    POS_VEL         = 6
    MIT             = 8     # impedance / force-control (single 8-byte frame)
    DISABLE         = 15
    FRAME_CONFIG    = 16


@dataclass(frozen=True)
class MotorConfig:
    """Static datasheet values plus project-specific defaults."""
    # ---- identity ----------------------------------------------------------
    name:           str
    can_id:         int           # CAN driver_id (0x01..0xFF)
    gear_ratio:     float

    # ---- MIT-mode signed quantization ranges (from manual S2 table) -------
    p_min:          float         # rad
    p_max:          float
    v_min:          float         # rad/s (joint side, after gearbox)
    v_max:          float
    tau_min:        float         # Nm (joint side, after gearbox)
    tau_max:        float
    kp_max:         float = 500.0
    kd_max:         float = 5.0

    # ---- electromechanical -------------------------------------------------
    kt:             float = 1.0   # Nm/A, joint-side torque constant

    # ---- recommended on-board impedance (host CT does the heavy work) -----
    kp_default:     float = 30.0
    kd_default:     float = 1.5

    # ---- safety clamps the host applies before sending --------------------
    tau_clamp:      float | None = None   # Nm; None = use tau_max

    def clamped_tau(self) -> float:
        return self.tau_max if self.tau_clamp is None else self.tau_clamp


# ============================================================================
# Catalogue
# ============================================================================
# Shoulder: CM-06-03-AK80-8-KV60-With-Driver
AK80_8 = MotorConfig(
    name="AK80-8-KV60",
    can_id=0x01,
    gear_ratio=8.0,
    p_min=-12.56, p_max=12.56,
    v_min=-38.0,  v_max=38.0,
    tau_min=-32.0, tau_max=32.0,
    kt=1.0569,
    kp_default=30.0, kd_default=1.5,
    tau_clamp=20.0,                # mechanical safety, < 25 Nm peak rating
)

# Elbow: CM-05-02-AK60-6-KV80-V3.0-D
AK60_6 = MotorConfig(
    name="AK60-6-KV80",
    can_id=0x02,
    gear_ratio=6.0,
    p_min=-12.56, p_max=12.56,
    v_min=-60.0,  v_max=60.0,
    tau_min=-12.0, tau_max=12.0,
    kt=0.5994,
    kp_default=15.0, kd_default=0.5,
    tau_clamp=8.0,                 # mechanical safety, < 9 Nm peak rating
)
