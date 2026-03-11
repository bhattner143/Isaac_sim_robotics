"""
Robot physical configuration dataclasses.

Contains parameters that describe the PHYSICAL SYSTEM:
  - CartPendulumPhysicsConfig  : mass, length, damping of cart-pendulum
  - EndEffectorKinematics2DConfig : runtime plant/manipulator references for EE kinematics
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# CART-PENDULUM PHYSICS
# ============================================================================

@dataclass
class CartPendulumPhysicsConfig:
    """Physical parameters for 2D cart-pendulum system."""
    mass_cart: float = 3.0
    mass_pendulum: float = 0.3
    length_pendulum: float = 0.25
    damping_cart: float = 0.5        # Cart slider damping [N·s/m]
    damping_pendulum: float = 0.1    # Pendulum gimbal damping [N·m·s/rad]
    gravity: float = 9.81

    # Initial state [x, y, α, β] — if None, computed from manipulator EE
    cart_initial_position: Optional[np.ndarray] = None   # [x, y] in meters
    pendulum_initial_angles: Optional[np.ndarray] = None  # [α, β] in radians


# ============================================================================
# END-EFFECTOR KINEMATICS (runtime reference config)
# ============================================================================

@dataclass
class EndEffectorKinematics2DConfig:
    """
    Configuration for EndEffectorKinematics2D.

    Holds runtime references to the finalized plant and manipulator so that
    the LeafSystem can perform forward kinematics and Jacobian queries without
    needing to re-create its own plant.

    NOTE: plant and manipulator are runtime objects — not serializable to JSON.
    """
    plant: object       # Finalized MultibodyPlant
    manipulator: object  # CupManipulator instance
    nq_total: int        # Total positions in the combined plant
    nv_total: int        # Total velocities in the combined plant


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_physics_config(**kwargs) -> CartPendulumPhysicsConfig:
    """Create a CartPendulumPhysicsConfig, overriding defaults with any kwargs."""
    return CartPendulumPhysicsConfig(**kwargs)
