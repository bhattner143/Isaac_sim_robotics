#!/usr/bin/env python3
"""
2D Cart-Pendulum System with Muscle Dynamics and Optimal Feedback Control

STATE VECTOR (14D):
===================
[0-7]:   Cart-pendulum [x, y, α, β, ẋ, ẏ, α̇, β̇]      (8D)
[8-9]:   Muscle forces [F_x, F_y]                       (2D)
[10-13]: ZFT reference [x_ref, y_ref, ẋ_ref, ẏ_ref]   (4D)

CONTROL:
========
Input: [u_x, u_y] (neural commands to muscles)
Control law: Finite-horizon LQR with time-varying gains

ARCHITECTURE:
=============
LQR → Muscle Dynamics (2D) → ZFT Reference (2D) → Impedance (2D) → Cart-Pendulum

USAGE:
======
    # Finite-horizon LQR control
    python script_cart_pendulum_2d_muscle_dynamics_ofc.py --mode finite-horizon-lqr-for-min-effort
    
    # Scene visualization only
    python script_cart_pendulum_2d_muscle_dynamics_ofc.py --mode scene-viz
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
from termcolor import colored
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Simulator,
    VectorLogSink,
    LeafSystem,
    BasicVector,
    MeshcatVisualizer,
    StartMeshcat,
    Multiplexer,
    Demultiplexer,
    Saturation,
    SceneGraph,
    SpatialInertia,
    UnitInertia,
    RotationalInertia,
    RigidTransform,
    RevoluteJoint,
    PrismaticJoint,
    Sphere,
    Cylinder,
    Rgba,
    Parser,
)

# Import from existing script
sys.path.append(str(Path(__file__).parent))
from robot_types import create_cart_pendulum_config, CartPendulumConfig, create_cup_manipulator_config

from scipy.linalg import solve_discrete_are

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================


@dataclass
class CartPendulumPhysicsConfig:
    """Parameters for 2D cart-pendulum system dynamics."""
    # Cart parameters
    mass_cart: float = 3.0  # kg
    damping_cart: float = 1.0  # N·s/m (increased damping)
    
    # Pendulum parameters
    mass_pendulum: float = 0.3  # kg
    length_pendulum: float = 0.5  # m (center of mass)
    damping_pendulum: float = 0.5  # N·s/m (significantly increased to prevent divergence)
    
    # System parameters
    gravity: float = 9.81  # m/s²


@dataclass
class MuscleDynamicsConfig:
    """Parameters for 2D muscle/actuator dynamics."""
    muscle_tau: float = 0.03  # s (time constant)
    initial_force: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [F_x, F_y]
    command_limit: Optional[float] = None  # Optional saturation on command


@dataclass
class ImpedanceForceConfig:
    """Parameters for 2D impedance force computation."""
    K_imp: float = 50.0  # N/m (stiffness)
    D_imp: float = 10.0  # N·s/m (damping)


@dataclass
class ZFTReferenceMassConfig:
    """Parameters for 2D ZFT/reference-mass dynamics."""
    M_ref: float = 1.0  # kg (reference mass)
    K_imp: float = 50.0  # N/m (stiffness)
    D_imp: float = 10.0  # N·s/m (damping)
    initial_ref: np.ndarray = field(default_factory=lambda: np.zeros(4))  # [x_ref, y_ref, ẋ_ref, ẏ_ref]


@dataclass
class FiniteHorizonLQRConfig:
    """Parameters for finite-horizon LQR controller."""
    Q: np.ndarray = field(default_factory=lambda: np.diag([
        100.0, 100.0,    # Cart position
        2000.0, 2000.0,  # Pendulum angles (massively increased to prevent divergence)
        10.0, 10.0,      # Cart velocities
        500.0, 500.0,    # Pendulum angular velocities (increased)
        0.1, 0.1,        # Muscle forces
        1.0, 1.0,        # Reference position
        1.0, 1.0,        # Reference velocity
    ]))  
    QN: Optional[np.ndarray] = field(default_factory=lambda: np.diag([
        200.0, 200.0,    # Cart position (2x)
        4000.0, 4000.0,  # Pendulum angles (2x)
        20.0, 20.0,      # Cart velocities (2x)
        1000.0, 1000.0,  # Pendulum velocities (2x)
        0.2, 0.2,        # Muscle forces (2x)
        2.0, 2.0,        # Reference position (2x)
        2.0, 2.0,        # Reference velocity (2x)
    ]))  # Terminal cost (2x Q for better convergence)
    R: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0]))
    x_goal: np.ndarray = field(default_factory=lambda: np.zeros(14))
    horizon: float = 10.0  # s
    timestep: float = 0.01  # s
    u_limits: Optional[tuple] = None  # (min, max) for control saturation


# ============================================================================
# CONFIG CREATION FUNCTIONS
# ============================================================================

def create_cart_pendulum_physics_config(
    cart_mass: float = 3.0,
    cart_damping: float = 1.0,
    pendulum_mass: float = 0.3,
    pendulum_length: float = 0.5,
    pendulum_damping: float = 0.5,
    gravity: float = 9.81,
) -> CartPendulumPhysicsConfig:
    """Create CartPendulumPhysicsConfig with custom parameters."""
    return CartPendulumPhysicsConfig(
        mass_cart=cart_mass,
        damping_cart=cart_damping,
        mass_pendulum=pendulum_mass,
        length_pendulum=pendulum_length,
        damping_pendulum=pendulum_damping,
        gravity=gravity,
    )


def create_muscle_dynamics_config(
    muscle_tau: float = 0.03,
    initial_force: Optional[np.ndarray] = None,
    command_limit: Optional[float] = None,
) -> MuscleDynamicsConfig:
    """Create MuscleDynamicsConfig with custom parameters."""
    return MuscleDynamicsConfig(
        muscle_tau=muscle_tau,
        initial_force=initial_force if initial_force is not None else np.zeros(2),
        command_limit=command_limit,
    )


def create_impedance_force_config(
    K_imp: float = 50.0,
    D_imp: float = 10.0,
) -> ImpedanceForceConfig:
    """Create ImpedanceForceConfig with custom parameters."""
    return ImpedanceForceConfig(
        K_imp=K_imp,
        D_imp=D_imp,
    )


def create_zft_reference_mass_config(
    M_ref: float = 1.0,
    K_imp: float = 50.0,
    D_imp: float = 10.0,
    initial_ref: Optional[np.ndarray] = None,
) -> ZFTReferenceMassConfig:
    """Create ZFTReferenceMassConfig with custom parameters."""
    return ZFTReferenceMassConfig(
        M_ref=M_ref,
        K_imp=K_imp,
        D_imp=D_imp,
        initial_ref=initial_ref if initial_ref is not None else np.zeros(4),
    )


def create_finite_horizon_lqr_config(
    Q: Optional[np.ndarray] = None,
    QN: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    x_goal: Optional[np.ndarray] = None,
    horizon: float = 10.0,
    timestep: float = 0.01,
    u_limits: Optional[tuple] = None,
) -> FiniteHorizonLQRConfig:
    """Create FiniteHorizonLQRConfig with custom parameters."""
    config = FiniteHorizonLQRConfig()
    
    if Q is not None:
        config.Q = Q
    if QN is not None:
        config.QN = QN
    else:
        config.QN = config.Q.copy()
    if R is not None:
        config.R = R
    if x_goal is not None:
        config.x_goal = x_goal
    
    config.horizon = horizon
    config.timestep = timestep
    config.u_limits = u_limits
    
    return config


# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='2D Cart-Pendulum with Muscle Dynamics and OFC')
parser.add_argument('--mode', type=str, 
                    choices=['finite-horizon-lqr-for-min-effort', 'scene-viz'],
                    default='finite-horizon-lqr-for-min-effort', 
                    help='Controller type')
parser.add_argument('--target-x', type=float, default=0.0,
                    help='Target X position [m]')
parser.add_argument('--target-y', type=float, default=0.5,
                    help='Target Y position [m]')
parser.add_argument('--duration', type=float, default=5.0,
                    help='Simulation duration [s]')
parser.add_argument('--horizon', type=float, default=10.0,
                    help='LQR planning horizon [s]')
args = parser.parse_args()


# ============================================================================
# GLOBAL CONFIG INSTANCES
# ============================================================================


PHYSICS_CONFIG = create_cart_pendulum_physics_config()
MUSCLE_CONFIG = create_muscle_dynamics_config()
IMPEDANCE_CONFIG = create_impedance_force_config()
ZFT_CONFIG = create_zft_reference_mass_config()

# Create LQR configuration
LQR_CONFIG = create_finite_horizon_lqr_config(
    Q=np.diag([
        100.0, 100.0,   # Cart position (x, y)
        1000.0, 1000.0, # Pendulum angles (α, β)
        10.0, 10.0,     # Cart velocities
        100.0, 100.0,   # Pendulum velocities
        0.1, 0.1,       # Muscle forces
        1.0, 1.0,       # Reference position
        1.0, 1.0,       # Reference velocity
    ]),
    R=np.diag([1.0, 1.0]),
    x_goal=np.array([
        args.target_x, args.target_y,  # Cart target
        0.0, 0.0,                       # Pendulums upright
        0.0, 0.0, 0.0, 0.0,            # Zero velocities
        0.0, 0.0,                       # Zero forces
        args.target_x, args.target_y,  # Reference target
        0.0, 0.0,                       # Zero ref velocities
    ]),
    horizon=args.horizon,
    timestep=0.01,
)


# ============================================================================
# CART-PENDULUM 3D CLASS (INDEPENDENT - NO PENDULUM3D DEPENDENCY)
# ============================================================================

class CartPendulum3D:
    """
    Independent 3D Cart-Pendulum System with 2D cart motion and gimbal-mounted pendulum.
    
    This class is fully self-contained and does NOT depend on Pendulum3D.
    It creates both the cart and pendulum bodies internally.
    
    SYSTEM ARCHITECTURE:
    --------------------
    Cart: 2 DOF (x, y position in horizontal plane)
        - Actuated by forces [F_x, F_y]
        - Mass: configurable
        - Visualization: Optional (can be disabled)
    
    Pendulum: 2 DOF (pitch, roll gimbal angles)
        - Attached to cart at specified offset
        - Passive (no direct actuation)
        - Visualization: Sphere + rod (always visible)
    
    Total System: 4 DOF → 8D state [x, y, α, β, ẋ, ẏ, α̇, β̇]
    
    COUPLING:
    ---------
    - Cart acceleration affects pendulum motion (inertial coupling)
    - Pendulum motion creates reaction forces on cart
    - Full nonlinear coupled dynamics via MultibodyPlant
    
    COORDINATE SYSTEM:
    ------------------
    - x, y: Cart position in horizontal plane (m)
    - α (pitch): Pendulum rotation about Y-axis (rad)
    - β (roll): Pendulum rotation about X-axis (rad)
    - Zero angles: pendulum hanging down
    
    USE CASES:
    ----------
    - Testing linearization methods
    - OFC controller development
    - Coupled dynamics analysis
    """
    
    def __init__(
        self,
        config: CartPendulumConfig,
        visualize_cart: bool = False,
        add_cart_actuators: bool = True,
        z_offset: float = 0.0,
    ):
        """
        Initialize cart-pendulum system.
        
        Args:
            config: CartPendulumConfig with all system parameters
            visualize_cart: If True, add visual geometry to cart; if False, cart is invisible
            add_cart_actuators: If True, add actuators to cart joints (active); if False, cart is passive
            z_offset: Vertical offset (height) at which to position the cart base
        """
        self.config = config
        self.visualize_cart = visualize_cart
        self.add_cart_actuators = add_cart_actuators
        self.z_offset = z_offset
        
        # Will be populated during attach_to_plant()
        self.cart_body = None
        self.x_slider_body = None
        self.x_joint = None
        self.y_joint = None
        self.pitch_joint = None
        self.roll_joint = None
        self.pendulum_body = None
        self.pitch_body = None  # Intermediate body for gimbal
        
    def attach_to_plant(self, plant: MultibodyPlant, model_instance, register_visuals: bool = True):
        """
        Attach cart-pendulum system to plant.
        
        Creates:
        1. Cart body with mass/inertia (optionally invisible)
        2. Prismatic joints for x and y motion
        3. Pendulum with gimbal joints (always visible if visuals enabled)
        4. Actuators for cart forces
        
        Args:
            plant: MultibodyPlant (before Finalize)
            model_instance: Model instance for all bodies
            register_visuals: If True, register visual geometry (requires scene graph)
        """
        from pydrake.all import Sphere, Cylinder, Rgba, RigidTransform, RevoluteJoint
        
        # ====================================================================
        # CREATE CART BODY
        # ====================================================================
        I_cart = (1.0/6.0) * self.config.cart_mass * (self.config.cart_size**2)
        cart_inertia = SpatialInertia(
            mass=self.config.cart_mass,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(I_cart, I_cart, I_cart)
        )
        
        self.cart_body = plant.AddRigidBody(
            f"{self.config.name}_cart",
            model_instance,
            cart_inertia
        )
        
        # Add visual geometry only if requested and scene graph is available
        if self.visualize_cart and register_visuals:
            from pydrake.all import Box
            cart_shape = Box(self.config.cart_size, self.config.cart_size, self.config.cart_size)
            plant.RegisterVisualGeometry(
                self.cart_body,
                RigidTransform(),
                cart_shape,
                f"{self.config.name}_cart_visual",
                np.array([0.3, 0.3, 0.8, 0.5])  # Semi-transparent blue
            )
        
        # ====================================================================
        # CREATE INTERMEDIATE SLIDER FOR 2-DOF CART MOTION
        # ====================================================================
        # Chain: world --[x]--> x_slider --[y]--> cart
        x_slider_inertia = SpatialInertia(
            mass=0.001,  # Negligible mass
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(1e-6, 1e-6, 1e-6)
        )
        self.x_slider_body = plant.AddRigidBody(
            f"{self.config.name}_x_slider",
            model_instance,
            x_slider_inertia
        )
        
        # ====================================================================
        # ADD PRISMATIC JOINTS
        # ====================================================================
        # Create a fixed offset frame at the desired height (if z_offset is non-zero)
        if abs(self.z_offset) > 1e-6:
            # Create a massless body at the offset height
            offset_inertia = SpatialInertia(
                mass=0.001,
                p_PScm_E=np.zeros(3),
                G_SP_E=UnitInertia(1e-6, 1e-6, 1e-6)
            )
            offset_body = plant.AddRigidBody(
                f"{self.config.name}_base_offset",
                model_instance,
                offset_inertia
            )
            # Weld offset body to world at the desired height
            plant.WeldFrames(
                plant.world_frame(),
                offset_body.body_frame(),
                RigidTransform([0.0, 0.0, self.z_offset])
            )
            parent_frame = offset_body.body_frame()
        else:
            parent_frame = plant.world_frame()
        
        # X-axis joint (connects to offset frame or world)
        self.x_joint = plant.AddJoint(
            PrismaticJoint(
                name=f"{self.config.name}_x",
                frame_on_parent=parent_frame,
                frame_on_child=self.x_slider_body.body_frame(),
                axis=[1.0, 0.0, 0.0],
                damping=self.config.cart_damping
            )
        )
        
        # Y-axis joint
        self.y_joint = plant.AddJoint(
            PrismaticJoint(
                name=f"{self.config.name}_y",
                frame_on_parent=self.x_slider_body.body_frame(),
                frame_on_child=self.cart_body.body_frame(),
                axis=[0.0, 1.0, 0.0],
                damping=self.config.cart_damping
            )
        )
        
        # Add cart actuators (if enabled)
        if self.add_cart_actuators:
            plant.AddJointActuator(f"{self.config.name}_force_x", self.x_joint)
            plant.AddJointActuator(f"{self.config.name}_force_y", self.y_joint)
        # Note: Pendulum pitch/roll joints are always passive (no actuators)
        
        # ====================================================================
        # CREATE PENDULUM BODIES AND JOINTS (GIMBAL MOUNT)
        # ====================================================================
        # Pendulum mass properties (cylinder)
        m_p = self.config.pendulum_mass
        L = self.config.pendulum_length
        r = self.config.pendulum_radius
        
        # Cylinder inertia about its own center (z-axis aligned)
        I_xx_com = I_yy_com = (1.0/12.0) * m_p * (3*r**2 + L**2)
        I_zz_com = 0.5 * m_p * r**2
        
        # Intermediate body for pitch rotation
        pitch_inertia = SpatialInertia(
            mass=0.001,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(1e-6, 1e-6, 1e-6)
        )
        self.pitch_body = plant.AddRigidBody(
            f"{self.config.name}_pitch_body",
            model_instance,
            pitch_inertia
        )
        
        # Pendulum body - use SpatialInertia.MakeFromCentralInertia()
        # CRITICAL: COM must be offset from pivot to create pendulum dynamics!
        # For a hanging pendulum, COM is at -L/2 in the Z-direction (body frame)
        com_offset = np.array([0.0, 0.0, -L/2])  # COM at -L/2 below pivot
        
        # Inertia about COM (for a cylinder)
        I_com = RotationalInertia(I_xx_com, I_yy_com, I_zz_com)
        
        # Use MakeFromCentralInertia to automatically apply parallel axis theorem
        pendulum_inertia = SpatialInertia.MakeFromCentralInertia(
            mass=m_p,
            p_PScm_E=com_offset,  # COM offset from body origin P
            I_SScm_E=I_com  # Inertia about COM
        )
        self.pendulum_body = plant.AddRigidBody(
            f"{self.config.name}_pendulum",
            model_instance,
            pendulum_inertia
        )
        
        # Add visual geometry for pendulum (always visible if register_visuals=True)
        if register_visuals:
            # Position sphere at -L (bottom of pendulum in body frame)
            pendulum_sphere = Sphere(r)
            plant.RegisterVisualGeometry(
                self.pendulum_body,
                RigidTransform(p=[0.0, 0.0, -L]),  # At bottom of pendulum
                pendulum_sphere,
                f"{self.config.name}_pendulum_visual",
                np.array([0.9, 0.1, 0.1, 1.0])  # Red
            )
            
            # Add cylinder rod (visual from origin to -L)
            pendulum_rod = Cylinder(radius=r/3, length=L)
            plant.RegisterVisualGeometry(
                self.pendulum_body,
                RigidTransform(p=[0.0, 0.0, -L/2]),  # Cylinder center at -L/2
                pendulum_rod,
                f"{self.config.name}_rod_visual",
                np.array([0.6, 0.6, 0.6, 1.0])  # Gray
            )
        
        # ====================================================================
        # ADD GIMBAL JOINTS
        # ====================================================================
        # Pitch joint (rotation about Y-axis) - cart to pitch_body
        self.pitch_joint = plant.AddJoint(
            RevoluteJoint(
                name=f"{self.config.name}_pitch",
                frame_on_parent=self.cart_body.body_frame(),
                frame_on_child=self.pitch_body.body_frame(),
                axis=[0.0, 1.0, 0.0],  # Y-axis
                damping=self.config.pendulum_damping
            )
        )
        
        # Roll joint (rotation about X-axis) - pitch_body to pendulum
        self.roll_joint = plant.AddJoint(
            RevoluteJoint(
                name=f"{self.config.name}_roll",
                frame_on_parent=self.pitch_body.body_frame(),
                frame_on_child=self.pendulum_body.body_frame(),
                axis=[1.0, 0.0, 0.0],  # X-axis
                damping=self.config.pendulum_damping
            )
        )
        
        # ====================================================================
        # PRINT SUMMARY
        # ====================================================================
        print(colored(f"\n✓ Cart-Pendulum 3D System Created:", 'green', attrs=['bold']))
        print(colored(f"  Cart mass: {self.config.cart_mass} kg", 'cyan'))
        print(colored(f"  Cart DOF: 2 (x, y)", 'cyan'))
        print(colored(f"  Cart visualization: {'ON' if self.visualize_cart else 'OFF'}", 'cyan'))
        print(colored(f"  Pendulum mass: {self.config.pendulum_mass} kg", 'cyan'))
        print(colored(f"  Pendulum length: {self.config.pendulum_length} m", 'cyan'))
        print(colored(f"  Pendulum DOF: 2 (pitch, roll)", 'cyan'))
        print(colored(f"  Pendulum visualization: ON (always)", 'cyan'))
        print(colored(f"  Total DOF: 4", 'cyan'))
        print(colored(f"  Total state: 8D [x, y, α, β, ẋ, ẏ, α̇, β̇]", 'cyan'))
        print(colored(f"  Inputs: 2D [F_x, F_y]", 'cyan'))
    
    def set_cart_state(self, context, x=0.0, y=0.0, x_dot=0.0, y_dot=0.0):
        """Set cart position and velocity."""
        self.x_joint.set_translation(context, x)
        self.y_joint.set_translation(context, y)
        self.x_joint.set_translation_rate(context, x_dot)
        self.y_joint.set_translation_rate(context, y_dot)
    
    def set_pendulum_state(self, context, pitch=0.0, roll=0.0, pitch_dot=0.0, roll_dot=0.0):
        """Set pendulum angles and velocities."""
        self.pitch_joint.set_angle(context, pitch)
        self.roll_joint.set_angle(context, roll)
        self.pitch_joint.set_angular_rate(context, pitch_dot)
        self.roll_joint.set_angular_rate(context, roll_dot)
    
    def get_cart_state(self, context):
        """Get cart position and velocity."""
        x = self.x_joint.get_translation(context)
        y = self.y_joint.get_translation(context)
        x_dot = self.x_joint.get_translation_rate(context)
        y_dot = self.y_joint.get_translation_rate(context)
        return np.array([x, y, x_dot, y_dot])
    
    def get_pendulum_state(self, context):
        """Get pendulum angles and velocities."""
        pitch = self.pitch_joint.get_angle(context)
        roll = self.roll_joint.get_angle(context)
        pitch_dot = self.pitch_joint.get_angular_rate(context)
        roll_dot = self.roll_joint.get_angular_rate(context)
        return np.array([pitch, roll, pitch_dot, roll_dot])
    
    def get_full_state(self, context):
        """Get complete 8D state vector."""
        cart_state = self.get_cart_state(context)
        pend_state = self.get_pendulum_state(context)
        return np.concatenate([
            [cart_state[0], cart_state[1]],  # x, y
            [pend_state[0], pend_state[1]],  # pitch, roll
            [cart_state[2], cart_state[3]],  # x_dot, y_dot
            [pend_state[2], pend_state[3]],  # pitch_dot, roll_dot
        ])
    
    def finite_difference_linearization(self, plant, context, epsilon=1e-6):
        """
        Compute linearization A, B matrices using numerical finite differences.
        
        Uses central differences: f(x+ε) - f(x-ε) / (2ε) for each state and input dimension.
        
        Args:
            plant: MultibodyPlant
            context: Context at equilibrium
            epsilon: Perturbation size (default 1e-6)
        
        Returns:
            A: State matrix (8×8) - ∂ẋ/∂x at equilibrium
            B: Input matrix (8×2) - ∂ẋ/∂u at equilibrium
        """
        n = plant.num_multibody_states()  # Should be 8
        m = plant.num_actuators()  # Should be 2
        
        # Get equilibrium state and input
        x0 = plant.GetPositionsAndVelocities(context)
        u0 = plant.get_actuation_input_port().Eval(context)
        
        # Initialize matrices
        A = np.zeros((n, n))
        B = np.zeros((n, m))
        
        # Compute A matrix (∂ẋ/∂x) using central differences
        for i in range(n):
            # Perturb state +epsilon
            context_plus = context.Clone()
            x_plus = x0.copy()
            x_plus[i] += epsilon
            plant.SetPositionsAndVelocities(context_plus, x_plus)
            plant.get_actuation_input_port().FixValue(context_plus, u0)
            xdot_plus = plant.EvalTimeDerivatives(context_plus).CopyToVector()
            
            # Perturb state -epsilon
            context_minus = context.Clone()
            x_minus = x0.copy()
            x_minus[i] -= epsilon
            plant.SetPositionsAndVelocities(context_minus, x_minus)
            plant.get_actuation_input_port().FixValue(context_minus, u0)
            xdot_minus = plant.EvalTimeDerivatives(context_minus).CopyToVector()
            
            # Central difference
            A[:, i] = (xdot_plus - xdot_minus) / (2 * epsilon)
        
        # Compute B matrix (∂ẋ/∂u) using central differences
        for j in range(m):
            # Perturb input +epsilon
            context_plus = context.Clone()
            plant.SetPositionsAndVelocities(context_plus, x0)
            u_plus = u0.copy()
            u_plus[j] += epsilon
            plant.get_actuation_input_port().FixValue(context_plus, u_plus)
            xdot_plus = plant.EvalTimeDerivatives(context_plus).CopyToVector()
            
            # Perturb input -epsilon
            context_minus = context.Clone()
            plant.SetPositionsAndVelocities(context_minus, x0)
            u_minus = u0.copy()
            u_minus[j] -= epsilon
            plant.get_actuation_input_port().FixValue(context_minus, u_minus)
            xdot_minus = plant.EvalTimeDerivatives(context_minus).CopyToVector()
            
            # Central difference
            B[:, j] = (xdot_plus - xdot_minus) / (2 * epsilon)
        
        return A, B


# ============================================================================
# MUSCLE DYNAMICS (2D)
# ============================================================================

class MuscleDynamics2D(LeafSystem):
    """
    First-order muscle dynamics for 2D force:
        Ḟ = (-F + u) / τ
    
    Input:  u (2)  = neural command [u_x, u_y] (N)
    Output: F (2)  = muscle force [F_x, F_y] (N)
    State:  F (2)
    """
    def __init__(self, config: MuscleDynamicsConfig):
        super().__init__()
        self.muscle_tau = config.muscle_tau
        self.initial_force = config.initial_force
        
        self.DeclareVectorInputPort("u", BasicVector(2))
        self.DeclareContinuousState(2)  # [F_x, F_y]
        self.DeclareVectorOutputPort(
            "F", BasicVector(2), self._calc_output,
            prerequisites_of_calc={self.all_state_ticket()}
        )
    
    def SetDefaultState(self, context, state):
        state.get_mutable_continuous_state_vector().SetFromVector(self.initial_force)
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        u = self.get_input_port(0).Eval(context)
        F = context.get_continuous_state_vector().CopyToVector()
        Fdot = (-F + u) / self.muscle_tau
        derivatives.get_mutable_vector().SetFromVector(Fdot)
    
    def _calc_output(self, context, output):
        F = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(F)


# ============================================================================
# ZFT REFERENCE MASS (2D)
# ============================================================================

class ZFTReferenceMass2D(LeafSystem):
    """
    2D ZFT/reference-mass dynamics:
    
        ẋ_ref = ẋ_ref
        ẍ_ref = (K*(x-x_ref) + D*(ẋ-ẋ_ref) + F) / M_ref
    
    Inputs:
        0: cart_state (4) = [x, y, ẋ, ẏ]
        1: muscle_force (2) = [F_x, F_y]
    
    Output:
        0: ref_state (4) = [x_ref, y_ref, ẋ_ref, ẏ_ref]
    
    State: [x_ref, y_ref, ẋ_ref, ẏ_ref] (4D)
    """
    def __init__(self, config: ZFTReferenceMassConfig):
        super().__init__()
        self.M_ref = config.M_ref
        self.K_imp = config.K_imp
        self.D_imp = config.D_imp
        self.initial_ref = config.initial_ref
        
        self.DeclareVectorInputPort("cart_state", BasicVector(4))
        self.DeclareVectorInputPort("muscle_force", BasicVector(2))
        self.DeclareContinuousState(4)  # [x_ref, y_ref, ẋ_ref, ẏ_ref]
        self.DeclareVectorOutputPort(
            "ref_state", BasicVector(4), self._calc_output,
            prerequisites_of_calc={self.all_state_ticket()}
        )
    
    def SetDefaultState(self, context, state):
        state.get_mutable_continuous_state_vector().SetFromVector(self.initial_ref)
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        cart_state = self.get_input_port(0).Eval(context)
        muscle_force = self.get_input_port(1).Eval(context)
        ref_state = context.get_continuous_state_vector().CopyToVector()
        
        x, y, vx, vy = cart_state
        x_ref, y_ref, vx_ref, vy_ref = ref_state
        F_x, F_y = muscle_force
        
        # Reference dynamics
        x_ref_dot = vx_ref
        y_ref_dot = vy_ref
        
        vx_ref_dot = (self.K_imp * (x - x_ref) + self.D_imp * (vx - vx_ref) + F_x) / self.M_ref
        vy_ref_dot = (self.K_imp * (y - y_ref) + self.D_imp * (vy - vy_ref) + F_y) / self.M_ref
        
        derivatives.get_mutable_vector().SetFromVector([x_ref_dot, y_ref_dot, vx_ref_dot, vy_ref_dot])
    
    def _calc_output(self, context, output):
        output.SetFromVector(context.get_continuous_state_vector().CopyToVector())


# ============================================================================
# IMPEDANCE FORCE (2D)
# ============================================================================

class ImpedanceForce2D(LeafSystem):
    """
    2D impedance force:
        F_imp = K*(x_ref - x) + D*(ẋ_ref - ẋ)
    
    Inputs:
        0: cart_state (4) = [x, y, ẋ, ẏ]
        1: ref_state (4) = [x_ref, y_ref, ẋ_ref, ẏ_ref]
    
    Output:
        0: F_imp (2) = [F_x, F_y]
    """
    def __init__(self, config: ImpedanceForceConfig):
        super().__init__()
        self.K_imp = config.K_imp
        self.D_imp = config.D_imp
        
        self.DeclareVectorInputPort("cart_state", BasicVector(4))
        self.DeclareVectorInputPort("ref_state", BasicVector(4))
        self.DeclareVectorOutputPort("F_imp", BasicVector(2), self._calc_output)
    
    def _calc_output(self, context, output):
        cart_state = self.get_input_port(0).Eval(context)
        ref_state = self.get_input_port(1).Eval(context)
        
        x, y, vx, vy = cart_state
        x_ref, y_ref, vx_ref, vy_ref = ref_state
        
        F_x = self.K_imp * (x_ref - x) + self.D_imp * (vx_ref - vx)
        F_y = self.K_imp * (y_ref - y) + self.D_imp * (vy_ref - vy)
        
        output.SetFromVector([F_x, F_y])


# ============================================================================
# FINITE-HORIZON LQR CONTROLLER (2D)
# ============================================================================

class FiniteHorizonLQRController2D(LeafSystem):
    """
    Finite-horizon LQR for 14D state space.
    
    State: [x, y, α, β, ẋ, ẏ, α̇, β̇, F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref]
    Input: [u_x, u_y]
    
    Control law: u(t) = -K(t) (x(t) - x_goal)
    """
    
    def __init__(self, A, B, config: FiniteHorizonLQRConfig):
        super().__init__()
        
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.Q = config.Q
        self.QN = config.QN if config.QN is not None else config.Q.copy()
        self.R = config.R
        self.x_goal = config.x_goal
        self.T = float(config.horizon)
        self.dt = float(config.timestep)
        self.u_limits = config.u_limits
        
        # Compute number of timesteps
        self.N = int(np.round(self.T / self.dt))
        self.T = self.N * self.dt
        
        # Discretize system
        self.Ad, self.Bd = self._discretize_zoh(self.A, self.B, self.dt)
        
        # Discretized cost matrices
        Qd = self.Q * self.dt
        Rd = self.R * self.dt
        QNd = self.QN
        
        # Compute time-varying gains
        self.K_list, self.P_list = self._finite_horizon_dlqr(
            self.Ad, self.Bd, Qd, Rd, QNd, self.N
        )
        
        # Drake ports
        n = self.A.shape[0]
        m = self.B.shape[1]
        self.DeclareVectorInputPort("x", BasicVector(n))
        self.DeclareVectorOutputPort("u", BasicVector(m), self.CalcU)
        
        print(colored(f"✓ LQR Controller: {n}D state → {m}D control", "green"))
        print(colored(f"  Horizon: {self.T:.1f} s, Timestep: {self.dt:.3f} s", "cyan"))
    
    def CalcU(self, context, output):
        x = self.get_input_port(0).Eval(context)
        t = context.get_time()
        
        k = int(np.clip(np.floor(t / self.dt), 0, self.N - 1))
        K = self.K_list[k]
        x_err = (x - self.x_goal).reshape((-1, 1))
        u = -(K @ x_err).flatten()
        
        # Apply control limits if specified
        if self.u_limits is not None:
            u = np.clip(u, self.u_limits[0], self.u_limits[1])
        
        output.SetFromVector(u)
    
    @staticmethod
    def _finite_horizon_dlqr(Ad, Bd, Q, R, QN, N):
        """Solve finite-horizon discrete-time LQR via backward Riccati recursion."""
        n = Ad.shape[0]
        m = Bd.shape[1]
        
        P_list = [None] * (N + 1)
        K_list = [None] * N
        
        P_list[N] = QN
        
        for k in range(N - 1, -1, -1):
            P_next = P_list[k + 1]
            
            K = np.linalg.solve(
                R + Bd.T @ P_next @ Bd,
                Bd.T @ P_next @ Ad
            )
            
            P = Q + Ad.T @ P_next @ Ad - Ad.T @ P_next @ Bd @ K
            
            P_list[k] = P
            K_list[k] = K
        
        return K_list, P_list
    
    @staticmethod
    def _discretize_zoh(A, B, dt):
        """Zero-order hold discretization."""
        from scipy.linalg import expm
        n = A.shape[0]
        m = B.shape[1]
        
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A * dt
        M[:n, n:] = B * dt
        
        EM = expm(M)
        
        Ad = EM[:n, :n]
        Bd = EM[:n, n:]
        
        return Ad, Bd


# ============================================================================
# LINEARIZED SYSTEM BUILDER
# ============================================================================

def build_linearized_system_2d(
    physics_config: CartPendulumPhysicsConfig,
    impedance_config: ImpedanceForceConfig,
    zft_config: ZFTReferenceMassConfig,
    muscle_config: MuscleDynamicsConfig,
):
    """
    Build 14D linearized system using Drake's Linearize().
    
    Returns:
        A (14x14), B (14x2): Linearized system matrices
    """
    
    K_imp = impedance_config.K_imp
    D_imp = impedance_config.D_imp
    M_ref = zft_config.M_ref
    muscle_tau = muscle_config.muscle_tau
    M_cart = physics_config.mass_cart
    
    # Create temporary plant for linearization
    temp_builder = DiagramBuilder()
    temp_plant = MultibodyPlant(time_step=0.0)
    
    cart_config = create_cart_pendulum_config(
        cart_mass=physics_config.mass_cart,
        cart_damping=physics_config.damping_cart,
        pendulum_mass=physics_config.mass_pendulum,
        pendulum_length=physics_config.length_pendulum,
    )
    temp_cart = CartPendulum3D(cart_config, visualize_cart=False, add_cart_actuators=True)
    temp_model = temp_plant.AddModelInstance("cart_temp")
    temp_cart.attach_to_plant(temp_plant, temp_model, register_visuals=False)
    
    temp_plant.Finalize()
    
    # Linearize cart-pendulum
    temp_context = temp_plant.CreateDefaultContext()
    temp_plant.SetPositions(temp_context, np.zeros(temp_plant.num_positions()))
    temp_plant.SetVelocities(temp_context, np.zeros(temp_plant.num_velocities()))
    temp_plant.get_actuation_input_port().FixValue(temp_context, np.zeros(2))
    
    from pydrake.systems.primitives import Linearize
    linear_sys = Linearize(
        temp_plant,
        temp_context,
        input_port_index=temp_plant.get_actuation_input_port().get_index(),
        output_port_index=temp_plant.get_state_output_port().get_index()
    )
    
    A_cp = linear_sys.A()  # 8x8
    B_cp = linear_sys.B()  # 8x2
    
    # Muscle dynamics (2x2)
    A_muscle = np.array([
        [-1.0/muscle_tau, 0.0],
        [0.0, -1.0/muscle_tau]
    ])
    
    B_muscle = np.array([
        [1.0/muscle_tau, 0.0],
        [0.0, 1.0/muscle_tau]
    ])
    
    # ZFT dynamics (4x4)
    A_zft = np.zeros((4, 4))
    A_zft[0, 2] = 1.0  # ẋ_ref
    A_zft[1, 3] = 1.0  # ẏ_ref
    A_zft[2, 0] = -K_imp / M_ref
    A_zft[2, 2] = -D_imp / M_ref
    A_zft[3, 1] = -K_imp / M_ref
    A_zft[3, 3] = -D_imp / M_ref
    
    # Assemble 14x14 A matrix
    A = np.zeros((14, 14))
    
    # Cart-pendulum block
    A[0:8, 0:8] = A_cp
    
    # Muscle block
    A[8:10, 8:10] = A_muscle
    
    # ZFT block
    A[10:14, 10:14] = A_zft
    
    # Coupling: Cart acceleration affected by impedance
    A[4, 10] = K_imp / M_cart   # ẍ depends on x_ref
    A[4, 12] = D_imp / M_cart   # ẍ depends on ẋ_ref
    A[5, 11] = K_imp / M_cart   # ÿ depends on y_ref
    A[5, 13] = D_imp / M_cart   # ÿ depends on ẏ_ref
    
    A[4, 0] += -K_imp / M_cart  # ẍ depends on -x
    A[4, 4] += -D_imp / M_cart  # ẍ depends on -ẋ
    A[5, 1] += -K_imp / M_cart  # ÿ depends on -y
    A[5, 5] += -D_imp / M_cart  # ÿ depends on -ẏ
    
    # ZFT acceleration depends on cart and muscle
    A[12, 0] = K_imp / M_ref
    A[12, 4] = D_imp / M_ref
    A[12, 8] = 1.0 / M_ref
    A[13, 1] = K_imp / M_ref
    A[13, 5] = D_imp / M_ref
    A[13, 9] = 1.0 / M_ref
    
    # Assemble 14x2 B matrix
    B = np.zeros((14, 2))
    B[8:10, 0:2] = B_muscle
    
    return A, B


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def main():
    print("\n" + "="*80)
    print(colored("2D CART-PENDULUM - MUSCLE DYNAMICS & OPTIMAL FEEDBACK CONTROL", "cyan", attrs=["bold"]))
    print("="*80)
    print(colored(f"Mode: {args.mode}", "yellow"))
    print(colored(f"Target: ({args.target_x:.2f}, {args.target_y:.2f}) m", "yellow"))
    print(colored(f"Duration: {args.duration:.1f} s", "yellow"))
    print(colored(f"Horizon: {args.horizon:.1f} s", "yellow"))
    print("="*80 + "\n")
    
    # Get global configurations
    physics_config = PHYSICS_CONFIG
    muscle_config = MUSCLE_CONFIG
    impedance_config = IMPEDANCE_CONFIG
    zft_config = ZFT_CONFIG
    
    # Start Meshcat
    meshcat = StartMeshcat()
    print(colored(f"🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))
    
    # Build system
    builder = DiagramBuilder()
    plant = MultibodyPlant(time_step=0.001)
    scene_graph = builder.AddSystem(SceneGraph())
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    
    # Calculate initial cart position (at manipulator EE location)
    # Using typical manipulator configuration from test file
    from script_cup_manipulator_controller_ofc import CupManipulator
    manipulator_config = create_cup_manipulator_config(
        urdf_path="model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf",
        joint_angles=(np.deg2rad(-10.0), np.deg2rad(20.0)),
        damping=(0.1, 0.1),
    )
    temp_manipulator = CupManipulator(manipulator_config, enable_visualization=False)
    
    # Create temporary plant to calculate EE position
    temp_plant = MultibodyPlant(time_step=0.0)
    temp_parser = Parser(temp_plant)
    temp_manipulator.load_urdf_to_plant(temp_plant, temp_parser)
    temp_manipulator.weld_base_to_world(temp_plant)
    temp_plant.Finalize()
    
    temp_context = temp_plant.CreateDefaultContext()
    temp_plant.SetPositions(temp_context, temp_manipulator.model_instance, 
                           np.array([np.deg2rad(-10.0), np.deg2rad(20.0)]))
    ee_pos_init = temp_manipulator.CalcPosition(temp_plant, temp_context)  # Get full [x, y, z]
    
    # Add cart-pendulum at EE height
    cart_config = create_cart_pendulum_config(
        cart_mass=physics_config.mass_cart,
        cart_damping=physics_config.damping_cart,
        pendulum_mass=physics_config.mass_pendulum,
        pendulum_length=physics_config.length_pendulum,
    )
    cart_pendulum = CartPendulum3D(cart_config, visualize_cart=True, add_cart_actuators=True, z_offset=ee_pos_init[2])
    cart_model = plant.AddModelInstance("cart_pendulum")
    cart_pendulum.attach_to_plant(plant, cart_model, register_visuals=True)
    
    print(colored("✓ Cart-Pendulum created at manipulator EE height", "green"))
    print(colored(f"  Cart base position: x={ee_pos_init[0]:.3f} m, y={ee_pos_init[1]:.3f} m, z={ee_pos_init[2]:.3f} m", "cyan"))
    
    plant.Finalize()
    builder.AddSystem(plant)
    
    # Connect plant to scene graph
    builder.Connect(
        plant.get_geometry_pose_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id())
    )
    builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port()
    )
    
    if args.mode == 'scene-viz':
        # Just visualize
        builder.AddSystem(plant)
        visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        diagram = builder.Build()
        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()
        
        plant_context = plant.GetMyMutableContextFromRoot(context)
        plant.SetPositions(plant_context, np.zeros(plant.num_positions()))
        plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
        
        visualizer.StartRecording()
        simulator.AdvanceTo(0.1)
        visualizer.PublishRecording()
        
        print(colored("\n✓ Scene visualization ready", "green"))
        input(colored("Press Enter to exit...", "yellow"))
        return
    
    # Build linearized system
    A, B = build_linearized_system_2d(physics_config, impedance_config, zft_config, muscle_config)
    print(colored(f"✓ Linearized system: A({A.shape[0]}x{A.shape[1]}), B({B.shape[0]}x{B.shape[1]})", "green"))
    
   
    # Create LQR controller
    lqr_controller = builder.AddSystem(
        FiniteHorizonLQRController2D(A, B, LQR_CONFIG)
    )
    
    # Create subsystems
    muscle_dynamics = builder.AddSystem(MuscleDynamics2D(muscle_config))
    zft_ref_mass = builder.AddSystem(ZFTReferenceMass2D(zft_config))
    impedance_force = builder.AddSystem(ImpedanceForce2D(impedance_config))
    
    # State demux/mux for 14D state assembly
    plant_state_demux = builder.AddSystem(Demultiplexer([2, 2, 2, 2]))  # [pos, angles, vel, ang_vel]
    cart_state_mux = builder.AddSystem(Multiplexer([2, 2]))  # [pos, vel] -> 4D
    state_assembler = builder.AddSystem(Multiplexer([2, 2, 2, 2, 2, 2, 2]))  # 14D
    zft_state_demux = builder.AddSystem(Demultiplexer([2, 2]))  # [pos_ref, vel_ref]
    
    # Wire plant state extraction
    builder.Connect(plant.get_state_output_port(), plant_state_demux.get_input_port())
    
    # Assemble cart state [x, y, ẋ, ẏ]
    builder.Connect(plant_state_demux.get_output_port(0), cart_state_mux.get_input_port(0))  # pos
    builder.Connect(plant_state_demux.get_output_port(2), cart_state_mux.get_input_port(1))  # vel
    
    # Wire ZFT
    builder.Connect(cart_state_mux.get_output_port(), zft_ref_mass.get_input_port(0))
    builder.Connect(muscle_dynamics.get_output_port(), zft_ref_mass.get_input_port(1))
    
    # Wire impedance
    builder.Connect(cart_state_mux.get_output_port(), impedance_force.get_input_port(0))
    builder.Connect(zft_ref_mass.get_output_port(), impedance_force.get_input_port(1))
    
    # Wire impedance to plant
    builder.Connect(impedance_force.get_output_port(), plant.get_actuation_input_port())
    
    # Assemble 14D state for LQR
    builder.Connect(zft_ref_mass.get_output_port(), zft_state_demux.get_input_port())
    builder.Connect(plant_state_demux.get_output_port(0), state_assembler.get_input_port(0))  # cart pos
    builder.Connect(plant_state_demux.get_output_port(1), state_assembler.get_input_port(1))  # pend angles
    builder.Connect(plant_state_demux.get_output_port(2), state_assembler.get_input_port(2))  # cart vel
    builder.Connect(plant_state_demux.get_output_port(3), state_assembler.get_input_port(3))  # pend vel
    builder.Connect(muscle_dynamics.get_output_port(), state_assembler.get_input_port(4))     # muscle F
    builder.Connect(zft_state_demux.get_output_port(0), state_assembler.get_input_port(5))    # ref pos
    builder.Connect(zft_state_demux.get_output_port(1), state_assembler.get_input_port(6))    # ref vel
    
    # Wire LQR control loop
    builder.Connect(state_assembler.get_output_port(), lqr_controller.get_input_port(0))
    builder.Connect(lqr_controller.get_output_port(), muscle_dynamics.get_input_port(0))
    
    # Visualization
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    meshcat.SetProperty("/Background", "visible", False)
    
    # Loggers
    state_logger = builder.AddSystem(VectorLogSink(plant.num_multibody_states()))
    builder.Connect(plant.get_state_output_port(), state_logger.get_input_port())
    
    ref_logger = builder.AddSystem(VectorLogSink(4))
    builder.Connect(zft_ref_mass.get_output_port(), ref_logger.get_input_port())
    
    force_logger = builder.AddSystem(VectorLogSink(2))
    builder.Connect(muscle_dynamics.get_output_port(), force_logger.get_input_port())
    
    # Build and simulate
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Set initial state
    plant_context = plant.GetMyMutableContextFromRoot(context)
    # Set cart at EE position, pendulum hanging down
    plant.SetPositions(plant_context, np.array([
        ee_pos_init[0], ee_pos_init[1],  # Cart at manipulator EE position
        0.0, 0.0,                         # Pendulum hanging down (α=0, β=0)
    ]))
    plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
    
    visualizer.StartRecording()
    
    print(colored(f"\nSimulating for {args.duration} s...", "yellow"))
    simulator.AdvanceTo(args.duration)
    print(colored("✓ Simulation complete\n", "green"))
    
    visualizer.PublishRecording()
    print(colored(f"🎬 Animation: {meshcat.web_url()}\n", "green", attrs=["bold"]))
    
    # Extract data
    state_log = state_logger.FindLog(context)
    ref_log = ref_logger.FindLog(context)
    force_log = force_logger.FindLog(context)
    
    t = state_log.sample_times()
    state_data = state_log.data()
    ref_data = ref_log.data()
    force_data = force_log.data()
    
    # Plot results
    print(colored("📈 Generating plots...", "yellow"))
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    # Cart trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, state_data[0, :], 'b-', label='x (actual)')
    ax1.plot(t, ref_data[0, :], 'r--', label='x_ref')
    ax1.axhline(args.target_x, color='g', linestyle=':', label='target')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('X Position [m]')
    ax1.legend()
    ax1.grid(True)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, state_data[1, :], 'b-', label='y (actual)')
    ax2.plot(t, ref_data[1, :], 'r--', label='y_ref')
    ax2.axhline(args.target_y, color='g', linestyle=':', label='target')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Y Position [m]')
    ax2.legend()
    ax2.grid(True)
    
    # XY trajectory
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(state_data[0, :], state_data[1, :], 'b-', label='actual')
    ax3.plot(ref_data[0, :], ref_data[1, :], 'r--', label='reference')
    ax3.plot(args.target_x, args.target_y, 'g*', markersize=15, label='target')
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # Pendulum angles
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, np.rad2deg(state_data[2, :]), 'b-', label='α')
    ax4.plot(t, np.rad2deg(state_data[3, :]), 'r-', label='β')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Pendulum Angles [deg]')
    ax4.legend()
    ax4.grid(True)
    
    # Cart velocities
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, state_data[4, :], 'b-', label='ẋ')
    ax5.plot(t, state_data[5, :], 'r-', label='ẏ')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Cart Velocity [m/s]')
    ax5.legend()
    ax5.grid(True)
    
    # Muscle forces
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t, force_data[0, :], 'b-', label='F_x')
    ax6.plot(t, force_data[1, :], 'r-', label='F_y')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Muscle Force [N]')
    ax6.legend()
    ax6.grid(True)
    
    # Position errors
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(t, state_data[0, :] - args.target_x, 'b-', label='x error')
    ax7.plot(t, state_data[1, :] - args.target_y, 'r-', label='y error')
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Position Error [m]')
    ax7.legend()
    ax7.grid(True)
    
    # Reference tracking
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(t, state_data[0, :] - ref_data[0, :], 'b-', label='x - x_ref')
    ax8.plot(t, state_data[1, :] - ref_data[1, :], 'r-', label='y - y_ref')
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('Tracking Error [m]')
    ax8.legend()
    ax8.grid(True)
    
    # Summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    final_x = state_data[0, -1]
    final_y = state_data[1, -1]
    error = np.sqrt((final_x - args.target_x)**2 + (final_y - args.target_y)**2)
    max_force = np.max(np.abs(force_data))
    
    summary_text = f"""
2D Cart-Pendulum LQR Results
{'='*30}

Target: ({args.target_x:.3f}, {args.target_y:.3f}) m
Final:  ({final_x:.3f}, {final_y:.3f}) m
Error:  {error:.4f} m

Max Force: {max_force:.2f} N
Duration:  {args.duration:.1f} s
Horizon:   {args.horizon:.1f} s

State Dimension: 14D
Control Dimension: 2D
"""
    ax9.text(0.1, 0.5, summary_text, fontfamily='monospace', fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.show()
    
    print(colored("✓ Plots generated\n", "green"))
    print("="*80)
    print(colored("Execution Complete!", "green", attrs=["bold"]))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
