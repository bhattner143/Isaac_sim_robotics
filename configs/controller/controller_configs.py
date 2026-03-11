"""
Controller configuration dataclasses.

Contains parameters that describe the CONTROL ALGORITHMS:
  - MuscleDynamicsConfig      : first-order actuator / muscle model
  - ImpedanceForceConfig      : impedance stiffness & damping
  - ZFTReferenceMassConfig    : ZFT reference-mass spring-damper
  - FiniteHorizonLQRConfig    : finite-horizon LQR weights & horizon
  - ZFTJointReferenceIKConfig : ZFT → joint-space IK tuning (runtime)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# MUSCLE / ACTUATOR DYNAMICS
# ============================================================================

@dataclass
class MuscleDynamicsConfig:
    """First-order muscle/actuator dynamics parameters (2D forces).

    Dynamics:  Ḟ = (-F + u) / muscle_tau
    """
    muscle_tau: float = 0.03          # Time constant [s] (~human muscle)
    initial_force: np.ndarray = None  # [F_x, F_y] initial state
    command_limit: float | None = None  # Optional symmetric saturation on u

    def __post_init__(self):
        if self.initial_force is None:
            self.initial_force = np.zeros(2)


# ============================================================================
# IMPEDANCE FORCE
# ============================================================================

@dataclass
class ImpedanceForceConfig:
    """Impedance control parameters (2D).

    F_imp = K_imp * Δp + D_imp * Δṗ
    """
    K_imp: float = 50.0   # Stiffness [N/m]
    D_imp: float = 10.0   # Damping [N·s/m]


# ============================================================================
# ZFT REFERENCE MASS
# ============================================================================

@dataclass
class ZFTReferenceMassConfig:
    """ZFT (Zero-Force Trajectory) reference mass parameters (2D).

    Dynamics:  M_ref * p̈_ref = K_imp*(p_ee - p_ref) + D_imp*(ṗ_ee - ṗ_ref) + F
    """
    M_ref: float = 1.0    # Reference mass [kg]
    K_imp: float = 50.0   # Spring coupling [N/m]
    D_imp: float = 10.0   # Damper coupling [N·s/m]
    initial_ref: np.ndarray = None  # [x_ref, y_ref, ẋ_ref, ẏ_ref]

    def __post_init__(self):
        if self.initial_ref is None:
            self.initial_ref = np.zeros(4)


# ============================================================================
# FINITE-HORIZON LQR
# ============================================================================

@dataclass
class FiniteHorizonLQRConfig:
    """Finite-horizon LQR parameters for the 14D OFC system.

    State (14D):
        [x, y, α, β, ẋ, ẏ, α̇, β̇, F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref]
    Control (2D):
        u = [u_x, u_y]  (neural command to muscle)
    """
    Q: np.ndarray = field(default_factory=lambda: np.diag([
        100.0, 100.0,    # Cart position (x, y)
        500.0, 500.0,    # Pendulum angles (α, β)
        10.0,  10.0,     # Cart velocities (ẋ, ẏ)
        100.0, 100.0,    # Pendulum angular velocities (α̇, β̇)
        0.1,   0.1,      # Muscle forces (F_x, F_y)
        1.0,   1.0,      # Reference position (x_ref, y_ref)
        0.1,   0.1,      # Reference velocity (ẋ_ref, ẏ_ref)
    ]))  # Running state cost (14×14)
    QN: Optional[np.ndarray] = field(default_factory=lambda: np.diag([
        200.0, 200.0,    # Cart position (2×)
        1000.0, 1000.0,  # Pendulum angles (2×)
        20.0,  20.0,     # Cart velocities (2×)
        200.0, 200.0,    # Pendulum angular velocities (2×)
        0.2,   0.2,      # Muscle forces (2×)
        2.0,   2.0,      # Reference position (2×)
        0.2,   0.2,      # Reference velocity (2×)
    ]))  # Terminal cost (≈ 2× Q for better convergence)
    R: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0]))  # Control cost (2×2)
    x_goal: np.ndarray = field(default_factory=lambda: np.zeros(14))    # Goal state (14D)
    horizon: float = 10.0    # Planning horizon [s]
    timestep: float = 0.01   # Discretization timestep [s]
    u_limits: Optional[tuple] = None  # (u_min, u_max) for saturation


# ============================================================================
# ZFT JOINT REFERENCE IK (runtime config)
# ============================================================================

@dataclass
class ZFTJointReferenceIKConfig:
    """
    Configuration for ZFTJointReferenceIK.

    Holds runtime references and tuning parameters so that
    LQRWithOFCForCompleteSystem.__init__() can create the config
    once (from plant queries) and pass it cleanly to the LeafSystem.

    NOTE: plant and manipulator are runtime objects — not serializable to JSON.
    """
    plant: object                              # Finalized MultibodyPlant
    manipulator: object                        # CupManipulator instance
    ik_method: str = "differential"            # "ik" | "differential"
    pos_tol: float = 0.01                      # IK position tolerance [m]
    dt: float = 0.001                          # Integration timestep [s]
    Kp: float = 10.0                           # Position feedback gain ("differential")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_muscle_config(**kwargs) -> MuscleDynamicsConfig:
    return MuscleDynamicsConfig(**kwargs)

def create_impedance_config(**kwargs) -> ImpedanceForceConfig:
    return ImpedanceForceConfig(**kwargs)

def create_zft_config(**kwargs) -> ZFTReferenceMassConfig:
    return ZFTReferenceMassConfig(**kwargs)

def create_lqr_config(**kwargs) -> FiniteHorizonLQRConfig:
    return FiniteHorizonLQRConfig(**kwargs)
