"""
Cart-Pendulum System - Drake Controller Architecture

═══════════════════════════════════════════════════════════════════════════════
TWO-SYSTEM ARCHITECTURE EXPLANATION
═══════════════════════════════════════════════════════════════════════════════

Classic underactuated system: Cart on track with inverted pendulum

┌─────────────────────────────────────────────────────────────────────────────┐
│                           DRAKE DIAGRAM ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐  state[x,θ,ẋ,θ̇]  ┌──────────────────┐              │
│  │                  │────────────────────>│                  │              │
│  │  MultibodyPlant  │                     │   Controller     │              │
│  │   (Physics)      │<────────────────────│   (Control Law)  │              │
│  │                  │  force[F]           │                  │              │
│  └──────────────────┘                     └──────────────────┘              │
│         │                                                                   │
│         │ geometry                                                          │
│         v                                                                   │
│  ┌──────────────────┐                                                      │
│  │   SceneGraph     │───────> MeshcatVisualizer                           │
│  └──────────────────┘                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

SYSTEM 1: MultibodyPlant (Physics)
──────────────────────────────────
• Cart on 1D track (prismatic joint)
• Inverted pendulum (revolute joint)
• 2 DOF: x (cart position), θ (pendulum angle)
• 1 actuator: horizontal force on cart
• Classic underactuated control problem

SYSTEM 2: Controller
─────────────────────
• PD: Simple position control
• Energy Shaping: Swing-up controller
• LQR: Balancing around upright
• Computed Torque: Trajectory tracking

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import dataclasses
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from datetime import datetime
from termcolor import colored

# Drake imports
from pydrake.all import (
    # Core simulation
    Simulator,
    DiagramBuilder,
    LeafSystem,
    BasicVector,
    
    # Multibody dynamics
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
    Parser,
    RevoluteJoint,
    PrismaticJoint,
    SpatialInertia,
    UnitInertia,
    RotationalInertia,
    
    # Visualization
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    StartMeshcat,
    Role,
    
    # Geometry
    Box,
    Cylinder,
    Sphere,
    Rgba,
    CoulombFriction,
    
    # Controllers
    LinearQuadraticRegulator,
    LinearSystem,
    Saturation,
    Multiplexer,
    Demultiplexer,
    ZeroOrderHold,
    
    # Mathematical utilities
    RotationMatrix,
    RigidTransform,
)

# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Cart-Pendulum with Drake Controllers')
parser.add_argument('--mode', type=str, 
                    choices=['computed-torque', 'finite-horizon-lqr-for-min-effort', 'scene-viz'],
                    default='finite-horizon-lqr-for-min-effort', 
                    help='Controller type')
parser.add_argument('--visualize', action='store_true', default=True, help='Enable visualization')
parser.add_argument('--initial_theta', type=float, default=None, 
                    help='Initial pendulum angle (degrees, 180=up, 0=down)')
parser.add_argument('--use-model-plant', type=bool, default=True,
                    help='Use separate model plant for computed torque (True) or use real plant (False)')
parser.add_argument('--plant-type', type=str, default='equations',
                    choices=['multibody', 'equations', 'linearized'],
                    help='Plant type: multibody (MultibodyPlant), equations (nonlinear), or linearized (equations 2.1 & 2.2)')
parser.add_argument('--plot-diagram', action='store_true', 
                    help='Generate and display BuildSystem block diagram')
args, _ = parser.parse_known_args()

# Skip interactive input when plotting diagram
if hasattr(args, 'plot_diagram') and args.plot_diagram:
    if args.initial_theta is None:
        args.initial_theta = 0.0  # Default for diagram mode
# Interactive input for initial angle if not provided (only if running as main script and not plotting diagram)
elif args.initial_theta is None and __name__ == "__main__":
    print("\n" + "="*70)
    print(colored("CART-PENDULUM INTERACTIVE SETUP", 'cyan', attrs=['bold']))
    print("="*70)
    print("\nEnter initial pendulum angle:")
    print(colored("  0°   = Hanging down", 'red'))
    print(colored("  45°  = Tilted from down", 'yellow'))
    print(colored("  90°  = Horizontal", 'yellow'))
    print(colored("  180° = Upright (balanced)", 'green'))
    print(colored("  225° = Tilted from up", 'yellow'))
    print("-"*70)
    
    while True:
        try:
            angle_input = input(colored("\nInitial angle (degrees, press Enter for 0°): ", 'cyan')).strip()
            if angle_input == '':
                args.initial_theta = 0.0
                print(colored("✓ Using default: 0° (hanging down)", 'green'))
                break
            args.initial_theta = float(angle_input)
            print(colored(f"✓ Using θ = {args.initial_theta}°", 'green'))
            break
        except ValueError:
            print(colored("❌ Invalid input. Please enter a number (e.g., 45, 90, 180)", 'red'))
    print("="*70 + "\n")
elif args.initial_theta is None:
    # Default when imported as module (not main)
    args.initial_theta = 0.0
else:
    print(colored(f"\n✓ Using command-line angle: θ = {args.initial_theta}°\n", 'green'))

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class CartPendulumPhysicsConfig:
    """Parameters for cart-pendulum system dynamics (models)."""
    # Cart parameters
    mass_cart: float = 3.0  # kg (M)
    width_cart: float = 0.4  # m (visualization)
    height_cart: float = 0.3  # m (visualization)
    depth_cart: float = 0.3  # m (visualization)
    damping_cart: float = 0.0  # N·s/m
    
    # Pendulum parameters
    mass_pendulum: float = 0.3  # kg (m)
    length_pendulum: float = 0.5  # m (l - center of mass)
    radius_pendulum: float = 0.04  # m (visualization)
    damping_pendulum: float = 0.0  # N·s/m
    
    # System parameters
    coupling_gain: float = 1.0  # G parameter
    gravity: float = 9.81  # m/s²
    
    # Impedance/arm parameters (for linearized plant)
    mass_arm: float = 2.0  # kg (M_arm)
    
    # Motor parameters
    motor_time_constant: float = 0.030  # s (τ)
    motor_noise_mean: float = 1e-4  # N
    motor_noise_std: float = 5e-3  # N
    
    # Track parameters
    track_length: float = 4.0  # m
    track_limit: float = 2.0  # m (position limits ±2m)
    
    # Muscle dynamics
    enable_muscle_dynamics: bool = True  # Whether to add muscle dynamics actuation


@dataclass
class MuscleDynamicsConfig:
    """Parameters for muscle/actuator dynamics (first-order system)."""
    # Muscle actuation dynamics: F_dot = (-F + u) / tau
    muscle_tau: float = 0.03  # s (time constant)
    muscle_initial_force: float = 0.0  # N (initial force state)
    command_limit: float | None = None  # N (optional saturation on command input)

@dataclass
class ImpedanceForceConfig:
    """
    Parameters for impedance force law:
        F_imp = kp*(y_ref - y) + kd*(v_ref - v)

    Typical usage:
    - y is cart position, v is cart velocity
    - y_ref, v_ref come from your ZFT/reference-mass block (or a desired trajectory)
    """
    kp: float = 50.0   # N/m  (or equivalent for your coordinate scaling)
    kd: float = 10.0   # N/(m/s)
    force_limit: float | None = None  # optional saturation on F_imp

@dataclass
class ZFTReferenceMassConfig:
    """
    ZFT / reference-mass dynamics:

      yref_dot = vref
      vref_dot = ( kp*(y - yref) + kd*(v - vref) + F ) / Mh

    Inputs:
      - y_v : [y, v]           (2) from plant (cart position/velocity)
      - F   : muscle force     (1)

    Output:
      - yref_vref : [yref, vref] (2)

    State:
      - [yref, vref]
    """
    Mh: float = 1.0
    kp: float = 50.0
    kd: float = 10.0
    yref0: float = 0.0
    vref0: float = 0.0

@dataclass
class FiniteHorizonLQRForMinEffortConfig:
    """
    Parameters for Finite-Horizon LQR with linearized plant.
    
    State vector (7D): [x, θ, ẋ, θ̇, F, y_ref, v_ref]
    - x: cart position
    - θ: pendulum angle
    - ẋ: cart velocity
    - θ̇: pendulum angular velocity
    - F: muscle force
    - y_ref: ZFT reference position
    - v_ref: ZFT reference velocity
    """
    # State cost matrix (7x7) - penalizes deviation from goal state
    Q: np.ndarray = field(default_factory=lambda: np.diag([
        100.,   # x: cart position
        1000.,  # θ: pendulum angle (high cost for deviation from upright)
        10.,    # ẋ: cart velocity
        100.,   # θ̇: pendulum angular velocity
        0.1,    # F: muscle force (low priority)
        1.0,    # y_ref: reference position
        1.0,    # v_ref: reference velocity
    ]))
    
    # Terminal cost matrix (7x7) - penalizes final state deviation
    QN: np.ndarray = field(default_factory=lambda: np.diag([
        100.,   # x: cart position
        1000.,  # θ: pendulum angle
        10.,    # ẋ: cart velocity
        100.,   # θ̇: pendulum angular velocity
        0.1,    # F: muscle force
        1.0,    # y_ref: reference position
        1.0,    # v_ref: reference velocity
    ]))
    
    # Input cost matrix (1x1) - penalizes control effort
    R: np.ndarray = field(default_factory=lambda: np.array([[1.0]]))
    
    # Goal state (7D)
    x_goal: np.ndarray = field(default_factory=lambda: np.array([
        3.0,  # x: target cart position [m]
        0.0,  # θ: upright (0 rad)
        0.0,  # ẋ: zero velocity
        0.0,  # θ̇: zero angular velocity
        0.0,  # F: zero muscle force
        3.0,  # y_ref: reference matches target position
        0.0,  # v_ref: zero reference velocity
    ]))
    
    horizon: float = 10.0  # seconds
    timestep: float = 0.01  # seconds (for discretization)
    u_limits: tuple = (-100.0, 100.0)  # Control saturation limits (N)
    discretization: str = "zoh"  # Discretization method: "zoh" or "euler"


@dataclass
class SimulationConfig:
    """Global simulation parameters."""
    # Simulation timing
    timestep: float = 0.001  # s (1 kHz)
    simulation_time: float = 10.0  # s
    realtime_rate: float = 1.0  # 1.0 = real-time
    
    # Logging
    print_interval: float = 0.5  # s
    logging_interval: float = 0.02  # s (50 Hz)
    
    # Trajectory configuration
    trajectory_mode: str = 'cart-motion'  # balance, track, or cart-motion
    cart_start_position: float = -0.5  # m
    cart_end_position: float = 0.5  # m
    cart_motion_duration: float = 3.0  # s
    pendulum_start_angle: float = 0.0  # degrees (0° = down, 180° = up)
    cart_settle_time: float = 0.5  # s
    
    # Noise parameters
    sensory_delay: float = 0.05  # s
    control_dependent_noise_std: float = 1.0
    state_dependent_sensory_noise_std: float = 0.0
    additive_process_noise_std: float = 1e-4
    additive_sensory_noise_cov: np.ndarray = field(default_factory=lambda: np.diag([1e-5, 1e-5, 1e-5, 1e-5]))
    internal_estimator_noise_cov: np.ndarray = field(default_factory=lambda: np.diag([1e-8, 1e-8, 1e-8, 1e-8]))
    target_hold_steps: int = 50


# ============================================================================
# CONFIG CREATION FUNCTIONS
# ============================================================================

def create_cart_pendulum_physics_config(
    mass_cart: float = 1.0,
    mass_pendulum: float = 0.5,
    length_pendulum: float = 0.5,
    radius_pendulum: float = 0.025,
    damping_cart: float = 0.1,
    damping_pendulum: float = 0.1,
    coupling_gain: float = 0.0,
    gravity: float = 9.81,
    mass_arm: float = 2.0,
    motor_time_constant: float = 0.030,
    track_length: float = 4.0,
    track_limit: float = 2.0,
    enable_muscle_dynamics: bool = True,
) -> CartPendulumPhysicsConfig:
    """
    Create a CartPendulumPhysicsConfig with custom parameters.
    Similar to create_cup_manipulator_config() pattern.
    
    Returns:
        CartPendulumPhysicsConfig instance
    """
    return CartPendulumPhysicsConfig(
        mass_cart=mass_cart,
        width_cart=0.2,
        height_cart=0.2,
        depth_cart=0.2,
        mass_pendulum=mass_pendulum,
        length_pendulum=length_pendulum,
        radius_pendulum=radius_pendulum,
        damping_cart=damping_cart,
        damping_pendulum=damping_pendulum,
        coupling_gain=coupling_gain,
        gravity=gravity,
        mass_arm=mass_arm,
        motor_time_constant=motor_time_constant,
        track_length=track_length,
        track_limit=track_limit,
        enable_muscle_dynamics=enable_muscle_dynamics,
    )

def create_impedance_force_config(
    kp: float = 50.0,
    kd: float = 10.0,
    force_limit: float | None = None,
) -> ImpedanceForceConfig:
    """
    Create an ImpedanceForceConfig with custom parameters.

    Args:
        kp: stiffness gain
        kd: damping gain
        force_limit: if set, clamp output force to ±force_limit

    Returns:
        ImpedanceForceConfig instance
    """
    return ImpedanceForceConfig(kp=kp, kd=kd, force_limit=force_limit)

def create_muscle_dynamics_config(
    muscle_tau: float = 0.03,
    muscle_initial_force: float = 0.0,
    command_limit: float | None = None,
) -> MuscleDynamicsConfig:
    """
    Create a MuscleDynamicsConfig with custom parameters.
    
    Args:
        muscle_tau: Muscle time constant [s]
        muscle_initial_force: Initial force state [N]
        command_limit: Optional saturation limit on command input [N]
    
    Returns:
        MuscleDynamicsConfig instance
    """
    return MuscleDynamicsConfig(
        muscle_tau=muscle_tau,
        muscle_initial_force=muscle_initial_force,
        command_limit=command_limit,
    )

def create_zft_reference_mass_config(
    Mh: float = 1.0,
    kp: float = 50.0,
    kd: float = 10.0,
    yref0: float = 0.0,
    vref0: float = 0.0,
) -> ZFTReferenceMassConfig:
    """Factory helper matching your template style."""
    return ZFTReferenceMassConfig(Mh=Mh, kp=kp, kd=kd, yref0=yref0, vref0=vref0)

def create_finite_horizon_lqr_for_min_effort_config(
    Q: np.ndarray = None,
    QN: np.ndarray = None,
    R: np.ndarray = None,
    x_goal: np.ndarray = None,
    horizon: float = 10.0,
    timestep: float = 0.01,
) -> FiniteHorizonLQRForMinEffortConfig:
    """
    Create a FiniteHorizonLQRForMinEffortConfig with custom parameters.
    
    Args:
        Q: State cost matrix (7x7) for [x, θ, ẋ, θ̇, F, y_ref, v_ref]
        QN: Terminal state cost matrix (7x7)
        R: Input cost matrix (1x1)
        x_goal: Goal state (7D)
        horizon: Planning horizon [s]
        timestep: Discretization timestep [s]
    
    Returns:
        FiniteHorizonLQRForMinEffortConfig instance
    """
    if Q is None:
        Q = np.diag([100., 1000., 10., 100., 0.1, 1.0, 1.0])
    if QN is None:
        QN = np.diag([100., 1000., 10., 100., 0.1, 1.0, 1.0])
    if R is None:
        R = np.array([[1.0]])
    if x_goal is None:
        x_goal = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0])
    
    config = FiniteHorizonLQRForMinEffortConfig()
    config.Q = Q
    config.QN = QN
    config.R = R
    config.x_goal = x_goal
    config.horizon = horizon
    config.timestep = timestep
    return config

def create_simulation_config(
    timestep: float = 0.001,
    simulation_time: float = 10.0,
    realtime_rate: float = 1.0,
    print_interval: float = 0.5,
    logging_interval: float = 0.02,
    trajectory_mode: str = 'cart-motion',
    cart_start_position: float = -0.5,
    cart_end_position: float = 0.5,
    cart_motion_duration: float = 3.0,
    pendulum_start_angle: float = 0.0,
    cart_settle_time: float = 0.5,
    sensory_delay: float = 0.05,
    control_dependent_noise_std: float = 1.0,
    state_dependent_sensory_noise_std: float = 0.0,
    additive_process_noise_std: float = 1e-4,
) -> SimulationConfig:
    """
    Create a SimulationConfig with custom parameters.
    
    Args:
        timestep: Simulation timestep [s]
        simulation_time: Total simulation duration [s]
        realtime_rate: Playback speed (1.0 = real-time)
        print_interval: Terminal output frequency [s]
        logging_interval: Data logging frequency [s]
        trajectory_mode: Trajectory mode ('cart-motion', 'balance', etc.)
        cart_start_position: Initial cart position [m]
        cart_end_position: Final cart position [m]
        cart_motion_duration: Cart motion duration [s]
        pendulum_start_angle: Initial pendulum angle [deg]
        cart_settle_time: Settlement time [s]
        sensory_delay: Sensor delay [s]
        control_dependent_noise_std: Control-dependent noise std
        state_dependent_sensory_noise_std: State-dependent sensory noise std
        additive_process_noise_std: Additive process noise std
    
    Returns:
        SimulationConfig instance
    """
    return SimulationConfig(
        timestep=timestep,
        simulation_time=simulation_time,
        realtime_rate=realtime_rate,
        print_interval=print_interval,
        logging_interval=logging_interval,
        trajectory_mode=trajectory_mode,
        cart_start_position=cart_start_position,
        cart_end_position=cart_end_position,
        cart_motion_duration=cart_motion_duration,
        pendulum_start_angle=pendulum_start_angle,
        cart_settle_time=cart_settle_time,
        sensory_delay=sensory_delay,
        control_dependent_noise_std=control_dependent_noise_std,
        state_dependent_sensory_noise_std=state_dependent_sensory_noise_std,
        additive_process_noise_std=additive_process_noise_std,
        additive_sensory_noise_cov=np.diag([1e-5, 1e-5, 1e-5, 1e-5]),
        internal_estimator_noise_cov=np.diag([1e-8, 1e-8, 1e-8, 1e-8]),
        target_hold_steps=50,
    )

# ============================================================================
# CREATE GLOBAL CONFIG INSTANCES
# ============================================================================

# Physics/Dynamics configuration (used by all models)
PHYSICS_CONFIG = create_cart_pendulum_physics_config()

# Muscle dynamics configuration
MUSCLE_DYNAMICS_CONFIG = create_muscle_dynamics_config()

# Finite-Horizon LQR config
FINITE_HORIZON_LQR_CONFIG = create_finite_horizon_lqr_for_min_effort_config()

# Global simulation config
SIM_CONFIG = create_simulation_config()

# ============================================================================
# MODE AND CONFIGURATION SETUP
# ============================================================================
CONTROLLER_MODE = args.mode
USE_MODEL_PLANT = args.use_model_plant
PLANT_TYPE = args.plant_type

# ============================================================================
# BACKWARD COMPATIBILITY: EXPOSE COMMONLY USED PARAMS
# ============================================================================

# Physics parameters
CART_MASS = PHYSICS_CONFIG.mass_cart
PENDULUM_MASS = PHYSICS_CONFIG.mass_pendulum
PENDULUM_LENGTH = PHYSICS_CONFIG.length_pendulum
GRAVITY = PHYSICS_CONFIG.gravity

# Simulation parameters
TIMESTEP = SIM_CONFIG.timestep
SIMULATION_TIME = SIM_CONFIG.simulation_time
LOGGING_INTERVAL = SIM_CONFIG.logging_interval

# ============================================================================
# CART-PENDULUM CLASS
# ============================================================================

class CartPendulumSystemDynamics:
    """
    Cart-Pendulum Plant Builder.
    
    RESPONSIBILITY: Build the physics plant ONLY
    - Creates MultibodyPlant with cart and pendulum
    
    Does NOT handle:
    - Muscle dynamics (handled by DrakeSceneManager)
    - Diagram building/wiring
    - Visualization
    - Simulation execution
    
    Those are handled by DrakeSceneManager.
    """

    def __init__(
        self,
        config: CartPendulumPhysicsConfig,
        builder: DiagramBuilder,
    ):
        """Initialize plant builder.
        
        Args:
            config: CartPendulumPhysicsConfig with physics parameters
            builder: Drake DiagramBuilder (passed in from SceneManager)
        """
        self.config = config
        self.builder = builder
        self.plant = None
        self.scene_graph = None

    def build_plant(self):
        """Build the cart-pendulum MultibodyPlant (physics only, no muscle dynamics wiring)."""
        print(colored("\n" + "=" * 70, "yellow"))
        print(colored("Building Cart-Pendulum System", "yellow", attrs=["bold"]))
        print(colored("=" * 70, "yellow"))

        # Create plant and scene graph
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=self.config.motor_time_constant
        )

        # --- Cart ---
        cart_inertia = SpatialInertia(
            mass=self.config.mass_cart,
            p_PScm_E=np.array([0.0, 0.0, 0.0]),
            G_SP_E=UnitInertia.SolidBox(self.config.width_cart, self.config.depth_cart, self.config.height_cart),
        )
        cart_body = self.plant.AddRigidBody("cart", cart_inertia)

        cart_shape = Box(self.config.width_cart, self.config.depth_cart, self.config.height_cart)
        self.plant.RegisterVisualGeometry(
            cart_body, RigidTransform(), cart_shape, "cart_visual",
            np.array([0.3, 0.3, 0.8, 1.0])
        )
        self.plant.RegisterCollisionGeometry(
            cart_body, RigidTransform(), cart_shape, "cart_collision",
            CoulombFriction(0.9, 0.8)
        )

        self.plant.AddJoint(
            PrismaticJoint(
                "cart_slider",
                self.plant.world_frame(),
                cart_body.body_frame(),
                [1, 0, 0],
                -self.config.track_limit,
                self.config.track_limit,
                damping=self.config.damping_cart,
            )
        )

        # --- Pendulum ---
        I_about_pivot = self.config.length_pendulum**2  # per unit mass for point mass at distance L

        pendulum_inertia = SpatialInertia(
            mass=self.config.mass_pendulum,
            p_PScm_E=np.array([0.0, 0.0, -self.config.length_pendulum]),
            G_SP_E=UnitInertia(Ixx=I_about_pivot, Iyy=I_about_pivot, Izz=0.0),
        )
        pendulum_body = self.plant.AddRigidBody("pendulum", pendulum_inertia)

        ball_shape = Sphere(0.08)
        self.plant.RegisterVisualGeometry(
            pendulum_body,
            RigidTransform(p=[0, 0, -self.config.length_pendulum]),
            ball_shape,
            "pendulum_ball",
            np.array([0.9, 0.2, 0.2, 1.0]),
        )

        # Thin connecting rod from cart to pendulum mass
        rod_radius = 0.015  # Thin rod
        rod_shape = Cylinder(rod_radius, self.config.length_pendulum)
        self.plant.RegisterVisualGeometry(
            pendulum_body,
            RigidTransform(p=[0, 0, -self.config.length_pendulum / 2]),
            rod_shape,
            "pendulum_rod",
            np.array([0.6, 0.3, 0.1, 0.9]),
        )

        self.plant.AddJoint(
            RevoluteJoint(
                "pendulum_pin",
                cart_body.body_frame(),
                pendulum_body.body_frame(),
                [0, 1, 0],
                damping=self.config.damping_pendulum,
            )
        )

        # Track visualization
        track_shape = Box(self.config.track_length, 0.1, 0.1)
        self.plant.RegisterVisualGeometry(
            self.plant.world_body(),
            RigidTransform(p=[0, 0, -self.config.height_cart / 2 - 0.1]),
            track_shape,
            "track_visual",
            np.array([0.5, 0.5, 0.5, 0.8]),
        )

        # Actuator
        self.plant.AddJointActuator(
            "cart_force", self.plant.GetJointByName("cart_slider")
        )

        # Finalize plant BEFORE wiring actuation graph
        self.plant.Finalize()

        print(colored("✓ Cart-Pendulum plant created:", "green"))
        print(colored(f"  Cart mass: {self.config.mass_cart} kg", "cyan"))
        print(colored(f"  Pendulum mass: {self.config.mass_pendulum} kg", "cyan"))
        print(colored(f"  Pendulum length: {self.config.length_pendulum} m", "cyan"))
        print(colored(f"  DOF: {self.plant.num_positions()}", "cyan"))
        print(colored(f"  Actuators: {self.plant.num_actuators()}", "cyan"))

    def create_model_for_controller(self):
        """Create a separate dynamics model for controller (model-plant separation)."""
        print(colored("\nCreating Controller Model (separate from plant)", "yellow", attrs=["bold"]))

        model_plant = MultibodyPlant(time_step=0.0)

        cart_inertia = SpatialInertia(
            mass=self.config.mass_cart,
            p_PScm_E=np.array([0.0, 0.0, 0.0]),
            G_SP_E=UnitInertia.SolidBox(self.config.width_cart, self.config.depth_cart, self.config.height_cart),
        )
        cart_body = model_plant.AddRigidBody("cart", cart_inertia)

        model_plant.AddJoint(
            PrismaticJoint(
                "cart_slider",
                model_plant.world_frame(),
                cart_body.body_frame(),
                [1, 0, 0],
                -self.config.track_limit,
                self.config.track_limit,
                damping=self.config.damping_cart,
            )
        )

        I_about_pivot = self.config.length_pendulum**2
        pendulum_inertia = SpatialInertia(
            mass=self.config.mass_pendulum,
            p_PScm_E=np.array([0.0, 0.0, -self.config.length_pendulum]),
            G_SP_E=UnitInertia(Ixx=I_about_pivot, Iyy=I_about_pivot, Izz=0.0),
        )
        pendulum_body = model_plant.AddRigidBody("pendulum", pendulum_inertia)

        model_plant.AddJoint(
            RevoluteJoint(
                "pendulum_pin",
                cart_body.body_frame(),
                pendulum_body.body_frame(),
                [0, 1, 0],
                damping=self.config.damping_pendulum,
            )
        )

        model_plant.AddJointActuator(
            "cart_force", model_plant.GetJointByName("cart_slider")
        )

        model_plant.Finalize()

        print(colored("✓ Controller model created (identical dynamics)", "green"))
        print(colored(f"  Model DOF: {model_plant.num_positions()}", "cyan"))
        print(colored("  Model purpose: Inverse dynamics calculations", "cyan"))

        return model_plant

    def linearize_cart_pendulum(self):
        """
        Linearize the cart-pendulum dynamics around the upright equilibrium.
        
        State: [x, θ, ẋ, θ̇]
        Input: F (force on cart)
        
        Returns:
            A: State transition matrix (4x4)
            B: Input matrix (4x1)
        
        Linearization at: θ=0, θ̇=0 (upright equilibrium)
        """
        # Extract parameters from config
        m_c = self.config.mass_cart
        m_p = self.config.mass_pendulum
        L = self.config.length_pendulum
        g = self.config.gravity
        b_c = self.config.damping_cart
        b_p = self.config.damping_pendulum
        
        # For small angles around upright (θ≈0):
        # sin(θ) ≈ θ, cos(θ) ≈ 1
        
        # Linearized equations of motion:
        # ẍ = (F - b_c*ẋ + m_p*L*θ̈ - m_p*g*θ) / (m_c + m_p)
        # θ̈ = (F + m_p*L*θ̈ - b_p*θ̇ - m_p*g*L*θ) / (m_p*L^2)
        
        # Rearrange to standard form:
        # State: [x, θ, ẋ, θ̇]
        # ẋ = ẋ
        # θ̇ = θ̇
        # ẍ = a11*x + a12*θ + a13*ẋ + a14*θ̇ + b1*F
        # θ̈ = a31*x + a32*θ + a33*ẋ + a34*θ̇ + b2*F
        
        # Denominator for acceleration equations
        denom = m_c * m_p * L + m_p**2 * L  # (m_c + m_p)*m_p*L
        
        # Linearized A matrix
        A = np.zeros((4, 4))
        
        # First row: ẋ derivative
        A[0, 2] = 1.0  # ∂ẋ/∂ẋ
        
        # Second row: θ̇ derivative
        A[1, 3] = 1.0  # ∂θ̇/∂θ̇
        
        # Third row: ẍ derivative (acceleration of cart)
        # From: (m_c + m_p)*ẍ = F - b_c*ẋ + m_p*L*θ̈ - m_p*g*θ
        # ẍ = (F - b_c*ẋ - m_p*g*θ + m_p*L*θ̈) / (m_c + m_p)
        A[2, 1] = -m_p * g / (m_c + m_p)  # ∂ẍ/∂θ
        A[2, 2] = -b_c / (m_c + m_p)  # ∂ẍ/∂ẋ
        
        # Fourth row: θ̈ derivative (angular acceleration)
        # From: m_p*L^2*θ̈ = -b_p*θ̇ + m_p*g*L*θ + F*L
        # θ̈ = (m_p*g*L*θ - b_p*θ̇ + F*L) / (m_p*L^2)
        A[3, 1] = m_p * g / (m_p * L)  # ∂θ̈/∂θ
        A[3, 3] = -b_p / (m_p * L**2)  # ∂θ̈/∂θ̇
        
        # Simplify
        A[3, 1] = g / L  # ∂θ̈/∂θ (gravity effect)
        
        # B matrix (input: Force F)
        B = np.zeros((4, 1))
        B[2, 0] = 1.0 / (m_c + m_p)  # ∂ẍ/∂F
        B[3, 0] = L / (m_p * L**2)  # ∂θ̈/∂F = 1/(m_p*L)
        B[3, 0] = 1.0 / (m_p * L)  # ∂θ̈/∂F
        
        return A, B

    # Plant builder: no wiring or simulation methods here
    # All orchestration handled by DrakeSceneManager




# ============================================================================
# LINEARIZED CART-PENDULUM WITH MUSCLE DYNAMICS CLASS
# ============================================================================

class CartPendulumLinearizedSystem:
    """
    Linearized Cart-Pendulum (Using Drake's Linearize).
    
    ARCHITECTURE:
    - Builds full nonlinear MultibodyPlant for cart-pendulum
    - Uses Drake's Linearize() to compute Jacobian-based linearization
    - Linearizes around upright equilibrium (θ=0, θ̇=0)
    
    ADVANTAGES:
    - Uses Drake's built-in Jacobian computation (numerical differentiation)
    - Works for ANY nonlinear system (no manual formula derivation needed)
    - Scales to complex systems easily
    - Automatically handles all state/input interactions
    
    STATE: [x, θ, ẋ, θ̇] (4D)
    - x, θ, ẋ, θ̇: cart-pendulum state (linearized via Jacobian)
    
    INPUT: F (force)
    OUTPUT: [x, θ, ẋ, θ̇] (cart-pendulum state)
    """

    def __init__(
        self,
        config: CartPendulumPhysicsConfig,
        builder: DiagramBuilder,
        linearization_method: str = 'numerical',
    ):
        """Initialize linearized system.
        
        Args:
            config: CartPendulumPhysicsConfig with physics parameters
            builder: Drake DiagramBuilder (passed in from SceneManager)
            linearization_method: Method for linearization ('numerical' or 'drake')
                - 'numerical': Custom finite difference implementation
                - 'drake': Drake's built-in Linearize() function
        """
        self.config = config
        self.builder = builder
        self.linearization_method = linearization_method
        
        # Linearized plant system objects
        self.nonlinear_plant = None
        self.nonlinear_builder = None
        self.linearized_system = None
        
        # Linearization point (equilibrium)
        self.equilibrium_state = None
        self.equilibrium_input = None
        self.linearized_matrices = dict(A_plant=None, B_plant=None, C_plant=None, D_plant=None)

    def build_plant(self):
        """
        Build linearized cart-pendulum using Drake's Linearize().
        
        Process:
        1. Create nonlinear MultibodyPlant
        2. Set equilibrium point (θ=0, all velocities=0)
        3. Use Drake's Linearize() to compute Jacobians numerically
        """
        print(colored("\n" + "=" * 70, "yellow"))
        print(colored("Building Linearized Cart-Pendulum (Drake Jacobian-based)", "yellow", attrs=["bold"]))
        print(colored("=" * 70, "yellow"))

        # Step 1: Build standalone nonlinear plant (for linearization only)
        print(colored("  [1/3] Creating nonlinear MultibodyPlant...", "cyan"))
        nonlinear_plant = MultibodyPlant(time_step=0.0)

        # Add cart
        cart_inertia = SpatialInertia(
            mass=self.config.mass_cart,
            p_PScm_E=np.array([0.0, 0.0, 0.0]),
            G_SP_E=UnitInertia.SolidBox(self.config.width_cart, self.config.depth_cart, self.config.height_cart),
        )
        cart_body = nonlinear_plant.AddRigidBody("cart", cart_inertia)

        # Add prismatic joint (cart slides on track)
        nonlinear_plant.AddJoint(
            PrismaticJoint(
                "cart_slider",
                nonlinear_plant.world_frame(),
                cart_body.body_frame(),
                [1, 0, 0],
                -self.config.track_limit,
                self.config.track_limit,
                damping=self.config.damping_cart,
            )
        )

        # Add pendulum
        I_about_pivot = self.config.length_pendulum**2
        pendulum_inertia = SpatialInertia(
            mass=self.config.mass_pendulum,
            p_PScm_E=np.array([0.0, 0.0, -self.config.length_pendulum]),
            G_SP_E=UnitInertia(Ixx=I_about_pivot, Iyy=I_about_pivot, Izz=0.0),
        )
        pendulum_body = nonlinear_plant.AddRigidBody("pendulum", pendulum_inertia)

        # Add revolute joint (pendulum rotates about cart)
        nonlinear_plant.AddJoint(
            RevoluteJoint(
                "pendulum_pin",
                cart_body.body_frame(),
                pendulum_body.body_frame(),
                [0, 1, 0],
                damping=self.config.damping_pendulum,
            )
        )

        # Add actuator (force on cart)
        nonlinear_plant.AddJointActuator(
            "cart_force", nonlinear_plant.GetJointByName("cart_slider")
        )

        nonlinear_plant.Finalize()
        print(colored("    ✓ Nonlinear plant created", "green"))

        # Step 2: Define equilibrium point (upright with zero velocity)
        print(colored("  [2/3] Computing linearization at equilibrium...", "cyan"))
        
        # State: [x, θ, ẋ, θ̇]
        # Equilibrium: cart at origin, pendulum upright, no motion
        eq_state = np.array([
            0.0,  # x = 0 (cart position)
            0.0,  # θ = 0 (pendulum down)
            0.0,  # ẋ = 0 (cart velocity)
            0.0,  # θ̇ = 0 (angular velocity)
        ])
        eq_input = np.array([0.0])  # F = 0 (no force at equilibrium)
        
        self.equilibrium_state = eq_state
        self.equilibrium_input = eq_input

        # Create a simple context for linearization
        self.context = nonlinear_plant.CreateDefaultContext()
        nonlinear_plant.SetPositionsAndVelocities(self.context, eq_state)
        nonlinear_plant.get_actuation_input_port().FixValue(self.context, eq_input)
        
        # Step 3: Linearize using selected method
        self.nonlinear_plant = nonlinear_plant
        
        if self.linearization_method == 'drake':
            print(colored("    Computing linearization using Drake's Linearize()...", "cyan"))
            self._linearize_by_drake()
        elif self.linearization_method == 'numerical':
            print(colored("    Computing numerical linearization via finite differences...", "cyan"))
            self._linearize_by_numerical()
        else:
            raise ValueError(f"Unknown linearization method: {self.linearization_method}")

        print(colored("    ✓ Linearization computed", "green"))
        print(colored(f"    A: {self.linearized_matrices['A_plant'].shape}, B: {self.linearized_matrices['B_plant'].shape}", "cyan"))
        
        # Step 4: Add linearized system to builder
        print(colored("  [3/3] Adding linearized system to diagram...", "cyan"))
        self.linearized_system = self.builder.AddSystem(self.linearized_io_sys)
        self.linearized_system.set_name("linearized_cart_pendulum")
        
        print(colored("✓ Linearized cart-pendulum created", "green"))
        print(colored("="*70 + "\n", "yellow"))


    def _linearize_by_drake(self):
        """
        Linearize using Drake's built-in Linearize() function.
        Uses automatic differentiation for exact Jacobians.
        """
        from pydrake.all import Linearize
        
        linearized_io_sys = Linearize(
            self.nonlinear_plant,
            self.context,
            input_port_index=self.nonlinear_plant.get_actuation_input_port().get_index(),
            output_port_index=self.nonlinear_plant.get_state_output_port().get_index(),
        )
        
        # Store linearized matrices
        self.linearized_matrices['A_plant'] = linearized_io_sys.A()
        self.linearized_matrices['B_plant'] = linearized_io_sys.B()
        self.linearized_matrices['C_plant'] = linearized_io_sys.C()
        self.linearized_matrices['D_plant'] = linearized_io_sys.D()
        
        # Store the LinearSystem
        self.linearized_io_sys = linearized_io_sys
    
    def _linearize_by_numerical(self):
        """
        Linearize using custom numerical finite difference implementation.
        Computes Jacobians via central finite differences.
        """
        from pydrake.systems.primitives import LinearSystem
        
        A_plant, B_plant = self.finite_difference_linearization(self.nonlinear_plant, self.context, epsilon=1e-6)
        
        # Store linearized matrices
        self.linearized_matrices['A_plant'] = A_plant
        self.linearized_matrices['B_plant'] = B_plant
        self.linearized_matrices['C_plant'] = np.eye(4)  # Output = state
        self.linearized_matrices['D_plant'] = np.zeros((4, 1))  # No direct feedthrough
        
        # Create LinearSystem from computed matrices
        self.linearized_io_sys = LinearSystem(
            A=A_plant,
            B=B_plant,
            C=self.linearized_matrices['C_plant'],
            D=self.linearized_matrices['D_plant']
        )

    def finite_difference_linearization(self, plant, context, epsilon=1e-6):
        """
        Compute linearization A, B matrices using numerical finite differences.
        
        Method:
        - A matrix: ∂f/∂x ≈ [f(x+ε) - f(x-ε)] / (2ε)
        - B matrix: ∂f/∂u ≈ [f(u+ε) - f(u-ε)] / (2ε)
        
        Args:
            plant: Drake MultibodyPlant
            context: Context at equilibrium point
            epsilon: Perturbation size for finite differences
        
        Returns:
            A, B: Linearized system matrices
        """
        # Get state and input dimensions
        x0 = plant.GetPositionsAndVelocities(context)
        u0 = plant.get_actuation_input_port().Eval(context)
        
        n_x = len(x0)  # State dimension
        n_u = len(u0)  # Input dimension
        
        # Create derivative function
        def get_state_derivative(x, u):
            """Compute state derivative ẋ = f(x, u)"""
            temp_context = plant.CreateDefaultContext()
            plant.SetPositionsAndVelocities(temp_context, x)
            plant.get_actuation_input_port().FixValue(temp_context, u)
            
            # Get continuous state derivatives
            derivatives = plant.EvalTimeDerivatives(temp_context)
            x_dot = derivatives.CopyToVector()
            return x_dot
        
        # Compute A matrix: ∂f/∂x (central differences)
        A = np.zeros((n_x, n_x))
        for i in range(n_x):
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            
            f_plus = get_state_derivative(x_plus, u0)
            f_minus = get_state_derivative(x_minus, u0)
            
            A[:, i] = (f_plus - f_minus) / (2 * epsilon)
        
        # Compute B matrix: ∂f/∂u (central differences)
        B = np.zeros((n_x, n_u))
        for i in range(n_u):
            u_plus = u0.copy()
            u_minus = u0.copy()
            u_plus[i] += epsilon
            u_minus[i] -= epsilon
            
            f_plus = get_state_derivative(x0, u_plus)
            f_minus = get_state_derivative(x0, u_minus)
            
            B[:, i] = (f_plus - f_minus) / (2 * epsilon)
        
        return A, B
    
    def verify_linearization(self, epsilon=1e-5):
        """
        Verify numerical linearization by comparing with different epsilon.
        Can be used to check accuracy of finite difference approximation.
        """
        A_verify, B_verify = self.finite_difference_linearization(self.nonlinear_plant, self.context, epsilon)
        
        # Compute difference from stored linearization
        A_diff = np.linalg.norm(A_verify - self.linearized_matrices['A_plant'])
        B_diff = np.linalg.norm(B_verify - self.linearized_matrices['B_plant'])
        
        print(colored(f"\n    Linearization verification (ε={epsilon}):", "cyan"))
        print(colored(f"    ||A_diff|| = {A_diff:.2e}", "cyan"))
        print(colored(f"    ||B_diff|| = {B_diff:.2e}", "cyan"))
        
        return A_verify, B_verify

    def get_output_port(self):
        """Get linearized plant output (4D state: [x, θ, ẋ, θ̇])."""
        return self.linearized_system.get_output_port(0)

# ============================================================================
# MUSCLE DYNAMICS CLASS
# ============================================================================
class MuscleDynamics(LeafSystem):
    
    """
    First-order muscle/actuator dynamics:
        F_dot = (-F + u) / tau

    Input:  u (1)  = neural command / desired force (N)
    Output: F (1)  = muscle force applied to plant (N)
    State:  F (1)
    """
    def __init__(self, config: MuscleDynamicsConfig):
        super().__init__()
        if config.muscle_tau <= 0:
            raise ValueError("tau must be > 0")

        self.tau = float(config.muscle_tau)
        self.initial_force = float(config.muscle_initial_force)

        self.DeclareVectorInputPort("u", BasicVector(1))
        self.DeclareContinuousState(1)  # [F]
        # Output depends on state only, not direct-feedthrough from input
        self.DeclareVectorOutputPort(
            "F", BasicVector(1), self._calc_output,
            prerequisites_of_calc={self.all_state_ticket()}
        )

    def SetDefaultState(self, context, state):
        state.get_mutable_continuous_state_vector().SetFromVector([self.initial_force])

    def DoCalcTimeDerivatives(self, context, derivatives):
        u = float(self.get_input_port(0).Eval(context)[0])
        F = float(context.get_continuous_state_vector().GetAtIndex(0))
        Fdot = (-F + u) / self.tau
        derivatives.get_mutable_vector().SetAtIndex(0, Fdot) # Set Fdot in the derivatives vector, integrated by the simulator

    def _calc_output(self, context, output):
        F = float(context.get_continuous_state_vector().GetAtIndex(0)) # Muscle force state
        output.SetFromVector([F])

# ============================================================================
# Impedance FORCE
# ============================================================================
class ImpedanceForce(LeafSystem):
    """
    Computes impedance force:
        Fimp = kp*(yref - y) + kd*(vref - v)

    Inputs:
      0: y_v       (2) = [y, v]
      1: yref_vref (2) = [yref, vref]
    Output:
      0: Fimp      (1)
    """
    def __init__(self, config: ImpedanceForceConfig):
        super().__init__()
        self.kp = float(config.kp)
        self.kd = float(config.kd)
        self.force_limit = None if config.force_limit is None else float(config.force_limit)

        self.DeclareVectorInputPort("y_v", BasicVector(2))
        self.DeclareVectorInputPort("yref_vref", BasicVector(2))
        self.DeclareVectorOutputPort("Fimp", BasicVector(1), self._calc_output)

    def _calc_output(self, context, output):
        y, v = self.get_input_port(0).Eval(context)
        yref, vref = self.get_input_port(1).Eval(context)

        Fimp = self.kp * (float(yref) - float(y)) + self.kd * (float(vref) - float(v))

        if self.force_limit is not None:
            Fimp = float(np.clip(Fimp, -self.force_limit, self.force_limit))

        output.SetFromVector([Fimp])

# ============================================================================
# ZFT / REFERENCE MASS (CONFIG + LEAFSYSTEM)
# ============================================================================
class ZFTReferenceMass(LeafSystem):
    """
    Drake LeafSystem implementing the ZFT/reference-mass ODE:

      yref_dot = vref
      vref_dot = ( kp*(y - yref) + kd*(v - vref) + F ) / Mh

    Ports:
      In(0) "y_v"  size 2
      In(1) "F"    size 1
      Out(0) "yref_vref" size 2
    """

    def __init__(self, config: ZFTReferenceMassConfig):
        super().__init__()

        if config.Mh <= 0:
            raise ValueError("ZFTReferenceMassConfig.Mh must be > 0")

        self.Mh = float(config.Mh)
        self.kp = float(config.kp)
        self.kd = float(config.kd)
        self._yref0 = float(config.yref0)
        self._vref0 = float(config.vref0)

        # Inputs
        self.DeclareVectorInputPort("y_v", BasicVector(2))
        self.DeclareVectorInputPort("F", BasicVector(1))

        # Continuous state: [yref, vref]
        self.DeclareContinuousState(2)

        # Output: [yref, vref] - depends on state only, not direct-feedthrough
        self.DeclareVectorOutputPort(
            "yref_vref", BasicVector(2), self._calc_output,
            prerequisites_of_calc={self.all_state_ticket()}
        )

    def SetDefaultState(self, context, state):
        # Set initial condition for [yref, vref]
        state.get_mutable_continuous_state_vector().SetFromVector(
            [self._yref0, self._vref0]
        )

    def DoCalcTimeDerivatives(self, context, derivatives):
        y, v = self.get_input_port(0).Eval(context)
        F = float(self.get_input_port(1).Eval(context)[0])

        x = context.get_continuous_state_vector()
        yref = float(x[0])
        vref = float(x[1])

        yref_dot = vref
        vref_dot = (self.kp * (float(y) - yref) + self.kd * (float(v) - vref) + F) / self.Mh

        derivatives.get_mutable_vector().SetFromVector([yref_dot, vref_dot])

    def _calc_output(self, context, output):
        output.SetFromVector(context.get_continuous_state_vector().CopyToVector())


# ============================================================================
# FINITE-HORIZON LQR CONTROLLER CLASS TO MINIMIZE EFFORT
# ============================================================================
class FiniteHorizonLQRController(LeafSystem):
    """
    Finite-horizon, continuous-time LQR implemented as time-varying state feedback.
    
    Control law: u(t) = -K(t) (x(t) - x_goal)
    
    Cost function:
        J = ∫_0^T [ x'Qx + u'Ru ] dt + x(T)'QN·x(T)
    
    Implementation:
    - Discretize continuous (A,B) with timestep dt
    - Solve finite-horizon discrete Riccati recursion backward
    - At runtime, select time-varying gain K(t) based on current time
    
    State vector for cart-pendulum: [x, φ, ẋ, φ̇, F, y_ref, v_ref]^T (7D)
    Input: command u (1D)
    """
    
    def __init__(self, A, B, config: FiniteHorizonLQRForMinEffortConfig):
        """
        Initialize Finite Horizon LQR Controller.
        
        Args:
            A: Continuous-time state matrix (n×n)
            B: Continuous-time input matrix (n×m)
            config: FiniteHorizonLQRForMinEffortConfig with all parameters (cost matrices,
                    horizon, timestep, goal state, control limits, discretization method)
        """
        super().__init__()
        
        # Store system matrices
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float).reshape((-1, 1))
        
        # Extract all parameters from config
        self.Q = np.array(config.Q, dtype=float)
        self.R = np.array(config.R, dtype=float).reshape((1, 1))
        self.QN = np.array(config.QN, dtype=float)
        self.x_goal = np.array(config.x_goal, dtype=float)
        self.T = float(config.horizon)
        self.dt = float(config.timestep)
        self.u_limits = config.u_limits
        discretization = config.discretization
        
        # Validate dimensions
        n = self.A.shape[0]
        assert self.A.shape == (n, n), f"A shape {self.A.shape} != ({n}, {n})"
        assert self.B.shape == (n, 1), f"B shape {self.B.shape} != ({n}, 1)"
        assert self.Q.shape == (n, n), f"Q shape {self.Q.shape} != ({n}, {n})"
        assert self.QN.shape == (n, n), f"QN shape {self.QN.shape} != ({n}, {n})"
        assert self.R.shape == (1, 1), f"R shape {self.R.shape} != (1, 1)"
        assert self.x_goal.shape == (n,), f"x_goal shape {self.x_goal.shape} != ({n},)"
        
        assert self.dt > 0, f"dt must be positive, got {self.dt}"
        assert self.T > 0, f"T must be positive, got {self.T}"
        
        # Compute number of timesteps
        self.N = int(np.round(self.T / self.dt))
        
        if self.N < 1:
            raise ValueError(f"Horizon too short: N={self.N} must be >= 1")
        
        # Adjust horizon to be exactly N*dt for consistency
        self.T = self.N * self.dt
        
        # Discretize continuous system
        self.Ad, self.Bd = self._discretize(self.A, self.B, self.dt, method=discretization)
        
        # Discretized cost matrices
        Qd = self.Q * self.dt
        Rd = self.R * self.dt
        QNd = self.QN  # Terminal cost not multiplied by dt
        
        # Compute time-varying gains via backward Riccati recursion
        self.K_list, self.P_list = self._finite_horizon_dlqr(
            self.Ad, self.Bd, Qd, Rd, QNd, self.N
        )
        
        # Drake ports
        self.DeclareVectorInputPort("x", BasicVector(n))
        # Output depends on context time (not input), breaking the algebraic loop
        self.DeclareVectorOutputPort("u", BasicVector(1), self.CalcU)
        
        print(colored(f"\n✓ FiniteHorizonLQRController created:", "green", attrs=["bold"]))
        print(colored(f"  Horizon: {self.T:.2f} s (N={self.N} steps)", "cyan"))
        print(colored(f"  Discretization: {discretization} with dt={self.dt:.4f} s", "cyan"))
        print(colored(f"  State dimension: {n}", "cyan"))
        print(colored(f"  Input dimension: 1", "cyan"))
        print(colored(f"  Terminal cost QN diagonal: {np.diag(self.QN)}", "cyan"))
    
    def CalcU(self, context, output):
        """Compute finite-horizon LQR control input."""
        x = self.get_input_port(0).Eval(context)
        t = context.get_time()
        
        # Select which gain to use based on current time
        k = int(np.floor(t / self.dt))
        k = int(np.clip(k, 0, self.N - 1))
        
        K = self.K_list[k]
        x_err = (x - self.x_goal).reshape((-1, 1))
        u = float(-(K @ x_err)[0, 0])
        
        # Apply control limits if specified
        if self.u_limits is not None:
            u_min, u_max = self.u_limits
            u = float(np.clip(u, u_min, u_max))
        
        output.SetFromVector([u])
    
    @staticmethod
    def _finite_horizon_dlqr(Ad, Bd, Q, R, QN, N):
        """
        Backward Riccati recursion for finite-horizon discrete-time LQR.
        
        System: x_{k+1} = Ad·x_k + Bd·u_k
        
        Cost: J = Σ_{k=0}^{N-1} (x_k'Qx_k + u_k'Ru_k) + x_N'QN·x_N
        
        Returns:
            K_list: List of N gain matrices, u_k = -K_list[k]·x_k
            P_list: List of N+1 Riccati matrices P_k
        """
        n = Ad.shape[0]
        P_list = [None] * (N + 1)
        K_list = [None] * N
        
        # Initialize at terminal time with terminal cost
        P = QN.copy()
        P_list[N] = P
        
        # Backward recursion: k = N-1, ..., 0
        for k in reversed(range(N)):
            # Compute gain: K_k = (R + Bd'P_{k+1}Bd)^{-1} Bd'P_{k+1}Ad
            S = R + Bd.T @ P @ Bd  # (1,1) since m=1
            K = np.linalg.solve(S, Bd.T @ P @ Ad)
            K_list[k] = K
            
            # Update P: P_k = Q + Ad'P_{k+1}(Ad - Bd·K_k)
            P = Q + Ad.T @ P @ (Ad - Bd @ K)
            P_list[k] = P
        
        # Verify shapes
        assert all(K.shape == (1, n) for K in K_list), "K shapes incorrect"
        assert all(P.shape == (n, n) for P in P_list), "P shapes incorrect"
        
        return K_list, P_list
    
    @staticmethod
    def _discretize(A, B, dt, method="zoh"):
        """
        Discretize continuous-time linear system.
        
        Continuous: ẋ = Ax + Bu
        Discrete:   x_{k+1} = Ad·x_k + Bd·u_k
        
        Args:
            A: Continuous state matrix
            B: Continuous input matrix
            dt: Timestep
            method: "zoh" (exact zero-order hold) or "euler" (forward Euler approx)
        
        Returns:
            Ad, Bd: Discrete system matrices
        """
        method = method.lower()
        
        if method == "euler":
            # Forward Euler: Ad = I + A·dt, Bd = B·dt
            Ad = np.eye(A.shape[0]) + A * dt
            Bd = B * dt
            return Ad, Bd
        
        if method == "zoh":
            # Exact zero-order hold via matrix exponential
            # Using augmented matrix exponential:
            #   exp([[A, B], [0, 0]] * dt) = [[Ad, Bd], [0, I]]
            from scipy.linalg import expm
            
            n = A.shape[0]
            m = B.shape[1]
            M = np.zeros((n + m, n + m))
            M[:n, :n] = A
            M[:n, n:] = B
            Md = expm(M * dt)
            
            Ad = Md[:n, :n]
            Bd = Md[:n, n:]
            return Ad, Bd
        
        raise ValueError(f"Unknown discretization method: {method}")


# ============================================================================
# BUILD THE COMPLETE SYSTEM
# ============================================================================

class BuildSystem:
    """
    Complete OFC system builder that creates and connects:
      u -> MuscleDynamics -> F -> ZFTReferenceMass -> (y_ref, v_ref)
         Plant state (4D) -> (y, v) -----------------------------┐
                                                                v
                                              ImpedanceForce -> F_imp -> LinearizedPlant

    All systems are created internally for consistency.
    
    STATE VECTOR (7D):
    ==================
    1. x       - Cart position [m]
    2. θ       - Pendulum angle [rad] (0 = down, π = up)
    3. ẋ       - Cart velocity [m/s]
    4. θ̇       - Pendulum angular velocity [rad/s]
    5. F       - Muscle force state [N] (from first-order actuator dynamics)
    6. y_ref   - ZFT reference position (cart position reference) [m]
    7. v_ref   - ZFT reference velocity (cart velocity reference) [m/s]
    
    Components:
    -----------
    - States 1-4: Linearized cart-pendulum plant (Jacobian-based)
    - State 5:    Muscle dynamics (F_dot = (-F + u) / τ)
    - States 6-7: ZFT reference mass (internal model, impedance control target)
    
    Exposes:
      - command input port (u): Neural command to muscle [N]
      - assembled output state: [x, θ, ẋ, θ̇, F, y_ref, v_ref] → 7D
        (If you don't need it, set assemble_output_state=False)
    """

    def __init__(
        self,
        physics_config: CartPendulumPhysicsConfig,
        builder: DiagramBuilder,
        muscle_config: MuscleDynamicsConfig,
        impedance_config: ImpedanceForceConfig,         
        zft_config: ZFTReferenceMassConfig,
        assemble_output_state: bool = True,
    ):
        self.physics_config = physics_config
        self.builder = builder

        if muscle_config is None:
            raise ValueError("muscle_config is required.")
        if impedance_config is None:
            raise ValueError("impedance_config is required.")
        if zft_config is None:
            raise ValueError("zft_config is required.")

        self.muscle_config = muscle_config
        self.impedance_config = impedance_config
        self.zft_config = zft_config
        self.assemble_output_state = bool(assemble_output_state)

        # Drake subsystems (all created in build())
        self.linearized_system = None  # CartPendulumLinearizedSystem instance
        self.linearized_plant = None   # The actual Drake LinearSystem
        self.muscle = None
        self.u_saturation = None
        self.zft = None
        self.impedance = None
        self.state_mux = None

        # Exposed ports
        self.command_input_port = None
        self.output_port = None

    # -------------------------
    # Helpers: extract y, v from 4D plant state
    # -------------------------
    @staticmethod
    def _select_cart_y_v(x4: np.ndarray) -> np.ndarray:
        """
        Given x4 = [x, theta, xdot, thetadot], return [y, v] = [x, xdot].
        This must match your linearized state ordering.
        """
        return np.array([x4[0], x4[2]])

    # -------------------------
    # Build
    # -------------------------
    def build(self):
        # 1) Create linearized cart-pendulum system
        self.linearized_system = CartPendulumLinearizedSystem(
            config=self.physics_config,
            builder=self.builder
        )
        self.linearized_system.build_plant()
        self.linearized_plant = self.linearized_system.linearized_system
        
        # 2) Muscle dynamics
        self.muscle = self.builder.AddSystem(MuscleDynamics(config=self.muscle_config))
        self.muscle.set_name("muscle_dynamics")

        # Optional saturation on u
        if self.muscle_config.command_limit is not None:
            lim = float(self.muscle_config.command_limit)
            self.u_saturation = self.builder.AddSystem(
                Saturation(min_value=[-lim], max_value=[lim])
            )
            self.u_saturation.set_name("u_saturation")
            self.command_input_port = self.u_saturation.get_input_port(0)
            self.builder.Connect(self.u_saturation.get_output_port(0),
                                self.muscle.get_input_port(0))
        else:
            self.command_input_port = self.muscle.get_input_port(0)

        # 3) ZFT/reference mass
        self.zft = self.builder.AddSystem(ZFTReferenceMass(config=self.zft_config))
        self.zft.set_name("zft_reference_mass")

        # 4) Impedance force block
        self.impedance = self.builder.AddSystem(ImpedanceForce(config=self.impedance_config))
        self.impedance.set_name("impedance_force")

        # 5) Provide plant state (4D) to ZFT and impedance as [y, v] = [x, xdot]
        #
        # Use a small LeafSystem to select indices (cleanest), but to keep this
        # snippet minimal, we’ll implement a 2x4 MatrixGain selector.
        from pydrake.all import MatrixGain
        S = np.array([[1.0, 0.0, 0.0, 0.0],   # y = x cart position
                      [0.0, 0.0, 1.0, 0.0]])  # v = xdot cart velocity
        selector = self.builder.AddSystem(MatrixGain(S))
        selector.set_name("select_y_v_from_x4")
        self.builder.Connect(self.linearized_plant.get_output_port(0),
                            selector.get_input_port(0))

        # plant [y,v] -> zft input 0
        self.builder.Connect(selector.get_output_port(0),
                            self.zft.get_input_port(0))
        # muscle F -> zft input 1
        self.builder.Connect(self.muscle.get_output_port(0),
                            self.zft.get_input_port(1))

        # plant [y,v] -> impedance input 0
        self.builder.Connect(selector.get_output_port(0),
                            self.impedance.get_input_port(0))
        # zft [yref,vref] -> impedance input 1
        self.builder.Connect(self.zft.get_output_port(0),
                            self.impedance.get_input_port(1))

        # 6) IMPORTANT: plant input is impedance force, NOT raw muscle force
        self.builder.Connect(self.impedance.get_output_port(0),
                            self.linearized_plant.get_input_port(0))

        # 7) Output wiring
        if self.assemble_output_state:
            # Output: [plant(4D), F(1D), yref_vref(2D)] => 7D
            self.state_mux = self.builder.AddSystem(Multiplexer([4, 1, 2]))
            self.state_mux.set_name("state_mux_7d")

            self.builder.Connect(self.linearized_plant.get_output_port(0),
                                self.state_mux.get_input_port(0))
            self.builder.Connect(self.muscle.get_output_port(0),
                                self.state_mux.get_input_port(1))
            self.builder.Connect(self.zft.get_output_port(0),
                                self.state_mux.get_input_port(2))

            self.output_port = self.state_mux.get_output_port(0)
        else:
            # Just expose plant output (4D)
            self.output_port = self.linearized_plant.get_output_port(0)

        return self

    # -------------------------
    # Ports
    # -------------------------
    def get_command_input_port(self):
        return self.command_input_port

    def get_state_output_port(self):
        return self.output_port
    
    # -------------------------
    # Get Full System Linearization (7D)
    # -------------------------
    def get_full_system_matrices(self):
        """
        Construct full 7D system linearization: A (7x7), B (7x1)
        
        State: [x, θ, ẋ, θ̇, F, y_ref, v_ref]
        Input: u (command to muscle)
        
        Returns:
            A (7x7): Full system state matrix
            B (7x1): Full system input matrix
        """
        
        # Method 2: Automatic Differentiation (AutoDiff)
        # -----------------------------------------------
        # Drake supports AutoDiff types that compute exact derivatives:
        #
        #   context_autodiff = diagram.CreateDefaultContext()
        #   # Convert scalar context to AutoDiff
        #   context_autodiff = diagram.ToAutoDiffXd()
        #   # Evaluate with AutoDiff to get exact Jacobians
        #
        # Advantages:
        #   - Exact derivatives (no numerical errors)
        #   - Efficient (single forward pass)
        #
        # Disadvantages:
        #   - Requires AutoDiff-compatible systems
        #   - More complex to set up
        #
        # WHY WE USE ANALYTICAL HERE:
        # ===========================
        # 1. Plant (4D): Already using Drake's Linearize() (numerical Jacobian)
        # 2. Muscle/ZFT/Impedance: Simple linear systems with known equations
        # 3. Combined approach: Fast + accurate for this specific problem
        #
        # For general black-box systems → use Method 1 (Drake's Linearize())
        # ============================================================================
        
        # Get 4D plant linearization (already computed via Drake's Linearize)
        A_plant = self.linearized_system.linearized_matrices['A_plant']
        B_plant = self.linearized_system.linearized_matrices['B_plant']
        
        # Extract impedance and muscle parameters
        kp_imp = self.impedance_config.kp
        kd_imp = self.impedance_config.kd
        tau_muscle = self.muscle_config.muscle_tau
        
        # Extract ZFT parameters
        Mh = self.zft_config.Mh
        kp_zft = self.zft_config.kp
        kd_zft = self.zft_config.kd
        
        # Build full 7x7 A matrix
        # State order: [x, θ, ẋ, θ̇, F, y_ref, v_ref]
        A_full = np.zeros((7, 7))
        
        # Plant dynamics (4x4 block): affected by impedance force F_imp
        # ẋ_plant = A_plant * x_plant + B_plant * F_imp
        # F_imp = kp_imp*(y_ref - x) + kd_imp*(v_ref - ẋ)
        # Linearize: ∂F_imp/∂x = -kp_imp, ∂F_imp/∂ẋ = -kd_imp, ∂F_imp/∂y_ref = kp_imp, ∂F_imp/∂v_ref = kd_imp
        
        # Top-left 4x4: Plant + impedance coupling
        A_full[:4, :4] = A_plant
        A_full[:4, 0] += B_plant.flatten() * (-kp_imp)  # coupling with x via impedance
        A_full[:4, 2] += B_plant.flatten() * (-kd_imp)  # coupling with ẋ via impedance
        A_full[:4, 5] += B_plant.flatten() * kp_imp     # coupling with y_ref
        A_full[:4, 6] += B_plant.flatten() * kd_imp     # coupling with v_ref
        
        # Muscle dynamics: Ḟ = (-F + u) / tau
        A_full[4, 4] = -1.0 / tau_muscle
        
        # ZFT dynamics: ẏ_ref = v_ref
        A_full[5, 6] = 1.0
        
        # ZFT dynamics: v̇_ref = (kp_zft*(x - y_ref) + kd_zft*(ẋ - v_ref) + F) / Mh
        A_full[6, 0] = kp_zft / Mh      # ∂v̇_ref/∂x
        A_full[6, 2] = kd_zft / Mh      # ∂v̇_ref/∂ẋ
        A_full[6, 4] = 1.0 / Mh         # ∂v̇_ref/∂F
        A_full[6, 5] = -kp_zft / Mh     # ∂v̇_ref/∂y_ref
        A_full[6, 6] = -kd_zft / Mh     # ∂v̇_ref/∂v_ref
        
        # Build 7x1 B matrix (input affects only muscle dynamics)
        B_full = np.zeros((7, 1))
        B_full[4, 0] = 1.0 / tau_muscle  # u -> Ḟ
        
        return A_full, B_full
    
    # -------------------------
    # Visualization
    # -------------------------
    def plot_diagram(self, diagram=None, filename=None, show=True):
        """
        Generate and display/save a graphviz visualization of the block diagram.
        
        Args:
            diagram: Drake Diagram to visualize. If None, uses self.builder.Build()
            filename: Path to save the diagram image (e.g., 'diagram.png', 'diagram.pdf')
            show: Whether to display the diagram interactively (requires matplotlib)
        
        Returns:
            The generated diagram object (if diagram was None)
        
        Example:
            system = BuildSystem(...)
            system.build()
            system.plot_diagram(filename='plots/system_diagram.png')
        """
        from pydrake.all import plot_system_graphviz
        import matplotlib.pyplot as plt
        import os
        
        # Build diagram if not provided
        built_diagram = None
        if diagram is None:
            built_diagram = self.builder.Build()
            diagram = built_diagram
        
        print(colored("\n" + "="*70, "cyan"))
        print(colored("Block Diagram Visualization", "cyan", attrs=["bold"]))
        print(colored("="*70, "cyan"))
        
        # Generate graphviz plot
        try:
            plot_system_graphviz(diagram, max_depth=3)
            
            if filename:
                # Create directory if needed
                os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(colored(f"✓ Diagram saved: {filename}", "green"))
            
            if show:
                plt.show()
            else:
                plt.close()
            
            print(colored("✓ Block diagram generated successfully", "green"))
            print(colored("="*70 + "\n", "cyan"))
            
        except Exception as e:
            print(colored(f"⚠ Could not generate diagram: {e}", "yellow"))
            print(colored("  Note: graphviz must be installed (brew install graphviz)", "yellow"))
        
        return built_diagram


# ============================================================================
# DRAKE SCENE MANAGER CLASS
# ============================================================================

class DrakeSceneManager:
    """2
    Scene Manager for Drake simulation with Diagram-based controller.
    
    Manages:
    - Diagram construction (plant + controller wiring)
    - Controller creation and configuration
    - Simulator setup and execution
    - Visualization and data logging
    
    Pattern: Following cup manipulator architecture
    - CartPendulumSystem: Minimal class for plant setup only
    - DrakeSceneManager: Orchestrates all simulation aspects
    """
    
    def __init__(self, 
                 cart_pendulum_config: CartPendulumPhysicsConfig | None = None, 
                 simulation_config: SimulationConfig | None = None,
                 controller_mode: str = 'pd', 
                 plant_type: str = 'multibody', 
                 visualize: bool = True, 
                 constant_force: float = 0.0, 
                 muscle_tau: float | None = None, 
                 simulation_time: float = 5.0,
                 initial_angle: float = np.deg2rad(180),
                 impedance_config: ImpedanceForceConfig | None = None,
                 zft_config: ZFTReferenceMassConfig | None = None):
        """Initialize scene manager.
        
        Args:
            cart_pendulum_config: CartPendulumPhysicsConfig with physics parameters
                                 (if None, uses global PHYSICS_CONFIG)
            simulation_config: SimulationConfig with simulation parameters
                              (if None, uses global SIM_CONFIG)
            controller_mode: Control mode ('pd', 'computed-torque', 'scene-viz', etc.)
            plant_type: Plant type ('multibody' or 'equations')
            visualize: Enable visualization
            constant_force: Constant force input (only used with muscle_tau)
            muscle_tau: Muscle dynamics time constant. If set, enables muscle dynamics
            simulation_time: Simulation duration (only used with muscle_tau)
            initial_angle: Initial pendulum angle in radians (only used with muscle_tau)
            impedance_config: ImpedanceForceConfig for OFC mode
            zft_config: ZFTReferenceMassConfig for OFC mode
        """
        # Use global configs if not provided
        if cart_pendulum_config is None:
            cart_pendulum_config = PHYSICS_CONFIG
        if simulation_config is None:
            simulation_config = SIM_CONFIG
        
        self.cart_pendulum_config = cart_pendulum_config
        self.simulation_config = simulation_config
        self.controller_mode = controller_mode
        self.plant_type = plant_type
        self.visualize = visualize
        self.constant_force = constant_force
        
        # Initialize muscle config (either from global or create custom)
        if muscle_tau is not None:
            self.muscle_config = create_muscle_dynamics_config(muscle_tau=muscle_tau)
        else:
            self.muscle_config = MUSCLE_DYNAMICS_CONFIG
        
        # Initialize impedance and ZFT configs
        if impedance_config is None:
            self.impedance_config = create_impedance_force_config()
        else:
            self.impedance_config = impedance_config
            
        if zft_config is None:
            self.zft_config = create_zft_reference_mass_config()
        else:
            self.zft_config = zft_config
        
        self.simulation_time = simulation_time
        self.initial_angle = initial_angle
        self.use_muscle_dynamics = muscle_tau is not None
        
        # Drake objects
        self.builder = DiagramBuilder()
        self.plant = None
        self.scene_graph = None
        self.controller = None
        self.meshcat = None
        self.diagram = None
        self.simulator = None
        
        # Robot system (will be initialized in setup_drake_system)
        self.system = None
        
        # Data logging
        self.time_log = []
        self.state_log = []
        self.force_log = []
        self.desired_state_log = []
        self.error_log = []
        
        print(colored("\n" + "="*70, 'cyan'))
        print(colored("DrakeSceneManager Initialization", 'cyan', attrs=['bold']))
        print(colored("="*70, 'cyan'))
        print(colored(f"  Controller Mode: {controller_mode}", 'yellow'))
        print(colored(f"  Plant Type: {plant_type}", 'yellow'))
        print(colored(f"  Visualization: {'Enabled' if visualize else 'Disabled'}", 'yellow'))
        print(colored("="*70 + "\n", 'cyan'))
    
    def _build_cart_pendulum_geometry(self, plant: MultibodyPlant) -> tuple:
        """Helper to build cart-pendulum geometry in a MultibodyPlant.
        
        Args:
            plant: MultibodyPlant to add geometry to (must not be finalized)
            
        Returns:
            tuple: (cart_body, pendulum_body)
        """
        from pydrake.all import Box, Cylinder, Sphere
        
        # Add cart
        cart_inertia = SpatialInertia(
            mass=self.cart_pendulum_config.mass_cart,
            p_PScm_E=np.array([0.0, 0.0, 0.0]),
            G_SP_E=UnitInertia.SolidBox(
                self.cart_pendulum_config.width_cart,
                self.cart_pendulum_config.depth_cart,
                self.cart_pendulum_config.height_cart
            ),
        )
        cart_body = plant.AddRigidBody("cart", cart_inertia)
        
        cart_shape = Box(
            self.cart_pendulum_config.width_cart,
            self.cart_pendulum_config.depth_cart,
            self.cart_pendulum_config.height_cart
        )
        plant.RegisterVisualGeometry(
            cart_body, RigidTransform(), cart_shape, "cart_visual",
            np.array([0.3, 0.5, 0.8, 1.0])
        )
        
        plant.AddJoint(
            PrismaticJoint(
                "cart_slider",
                plant.world_frame(),
                cart_body.body_frame(),
                [1, 0, 0],
                -self.cart_pendulum_config.track_limit,
                self.cart_pendulum_config.track_limit,
                damping=0.0,
            )
        )
        
        # Add pendulum
        I_about_pivot = self.cart_pendulum_config.length_pendulum**2
        pendulum_inertia = SpatialInertia(
            mass=self.cart_pendulum_config.mass_pendulum,
            p_PScm_E=np.array([0.0, 0.0, -self.cart_pendulum_config.length_pendulum]),
            G_SP_E=UnitInertia(Ixx=I_about_pivot, Iyy=I_about_pivot, Izz=0.0),
        )
        pendulum_body = plant.AddRigidBody("pendulum", pendulum_inertia)
        
        # Pendulum rod (cylinder pointing downward in -Z direction)
        # Use very thin radius to look like a thread
        thread_radius = self.cart_pendulum_config.radius_pendulum * 0.1  # 10x thinner
        pend_rod = Cylinder(
            thread_radius,
            self.cart_pendulum_config.length_pendulum
        )
        # Cylinder axis is Z by default, so no rotation needed
        # Place it so it extends from origin (pivot) down to -length_pendulum
        X_pend_rod = RigidTransform(
            RotationMatrix(),  # Identity - cylinder already points in Z
            [0, 0, -self.cart_pendulum_config.length_pendulum/2]
        )
        plant.RegisterVisualGeometry(
            pendulum_body, X_pend_rod, pend_rod, "pendulum_rod",
            np.array([0.8, 0.3, 0.3, 1.0])  # Red
        )
        
        # Pendulum mass (sphere at the end)
        sphere_radius = self.cart_pendulum_config.radius_pendulum * 2.5  # Make it visible
        pend_mass = Sphere(sphere_radius)
        X_pend_mass = RigidTransform(
            RotationMatrix(),
            [0, 0, -self.cart_pendulum_config.length_pendulum]  # At the end
        )
        plant.RegisterVisualGeometry(
            pendulum_body, X_pend_mass, pend_mass, "pendulum_mass",
            np.array([0.9, 0.2, 0.2, 1.0])  # Bright red
        )
        
        plant.AddJoint(
            RevoluteJoint(
                "pendulum_pin",
                cart_body.body_frame(),
                pendulum_body.body_frame(),
                [0, 1, 0],
                damping=0.0,
            )
        )
        
        return cart_body, pendulum_body
    
    def setup_drake_system(self):
        """Build the Drake system - handles both standard and muscle dynamics modes."""
        
        # Check if we need BuildSystem (for OFC/finite-horizon LQR mode)
        if self.controller_mode == 'finite-horizon-lqr-for-min-effort':
            print(colored("\n" + "="*70, 'yellow'))
            print(colored("Building OFC System (BuildSystem)", 'yellow', attrs=['bold']))
            print(colored("="*70 + "\n", 'yellow'))
            
            # Use stored configs for BuildSystem
            print(colored("  Using system configurations:", 'cyan'))
            print(colored(f"    - Muscle config: τ={self.muscle_config.muscle_tau}s", 'cyan'))
            print(colored(f"    - Impedance config: kp={self.impedance_config.kp}, kd={self.impedance_config.kd}", 'cyan'))
            print(colored(f"    - ZFT config: Mh={self.zft_config.Mh}, kp={self.zft_config.kp}, kd={self.zft_config.kd}", 'cyan'))
            
            # Create BuildSystem which includes linearized plant, muscle, ZFT, impedance
            self.system = BuildSystem(
                physics_config=self.cart_pendulum_config,
                builder=self.builder,
                muscle_config=self.muscle_config,
                impedance_config=self.impedance_config,
                zft_config=self.zft_config,
                assemble_output_state=True
            )
            self.system.build()
            
            # For simulation, extract the nonlinear plant reference for state initialization
            self.plant = self.system.linearized_system.nonlinear_plant
            
            print(colored("✓ BuildSystem created with 7D state output", "green"))
            print(colored("  State: [x, θ, ẋ, θ̇, F, y_ref, v_ref]", "cyan"))

    
    def add_controller(self):
        """Add controller to the diagram."""
        print(colored(f"\nAdding Controller: {self.controller_mode}", 'yellow', attrs=['bold']))
        
        if self.controller_mode == 'scene-viz':
            from pydrake.systems.primitives import ConstantVectorSource
            self.controller = self.builder.AddSystem(
                ConstantVectorSource(np.zeros(1))
            )
            print(colored(f"✓ Zero-force controller (visualization only)", 'green'))
            self.builder.Connect(
                self.controller.get_output_port(0),
                self.plant.get_actuation_input_port()
            )
            return
        
        elif self.controller_mode == 'finite-horizon-lqr-for-min-effort':
            # Get full 7D system linearization from BuildSystem
            print(colored("  Extracting 7D system linearization...", 'cyan'))
            A_full, B_full = self.system.get_full_system_matrices()
            
            print(colored(f"  A matrix shape: {A_full.shape}", 'cyan'))
            print(colored(f"  B matrix shape: {B_full.shape}", 'cyan'))
            
            # Get LQR configuration
            config = FINITE_HORIZON_LQR_CONFIG
            
            print(colored(f"  Horizon: {config.horizon} s", 'cyan'))
            print(colored(f"  Timestep: {config.timestep} s", 'cyan'))
            print(colored(f"  Goal state: {config.x_goal}", 'cyan'))
            
            # Create Finite-Horizon LQR Controller (config provides all parameters)
            self.controller = self.builder.AddSystem(
                FiniteHorizonLQRController(
                    A=A_full,
                    B=B_full,
                    config=config
                )
            )
            self.controller.set_name("finite_horizon_lqr_controller")
            
            print(colored(f"✓ Finite-Horizon LQR Controller created", 'green'))
            print(colored(f"  State dimension: 7D [x, θ, ẋ, θ̇, F, y_ref, v_ref]", 'cyan'))
            print(colored(f"  Control dimension: 1D [u]", 'cyan'))
        
        else:
            raise ValueError(f"Unknown controller mode: {self.controller_mode}")
        
        # Wire controller to plant based on mode
        if self.controller_mode == 'finite-horizon-lqr-for-min-effort':
            # Wire BuildSystem's 7D state output to controller input
            self.builder.Connect(
                self.system.get_state_output_port(),
                self.controller.get_input_port(0)
            )
            # Wire controller output (u) to BuildSystem's command input
            self.builder.Connect(
                self.controller.get_output_port(0),
                self.system.get_command_input_port()
            )
            print(colored(f"✓ Controller wired: 7D state -> LQR -> command (u)", 'green'))
        
        else:
            pass
        
        print(colored(f"✓ Controller wired to plant", 'green'))
    
    def setup_visualization(self):
        """Setup Meshcat visualization."""
        if not self.visualize:
            return
        
        # For BuildSystem mode, create separate visualization plant
        if self.controller_mode == 'finite-horizon-lqr-for-min-effort':
            print(colored("\n" + "="*70, 'yellow'))
            print(colored("Setting up Visualization", 'yellow', attrs=['bold']))
            print(colored("="*70 + "\n", 'yellow'))
            
            print(colored("  Creating visualization plant...", "cyan"))
            self.viz_plant, self.scene_graph = AddMultibodyPlantSceneGraph(
                self.builder, time_step=0.0
            )
            
            # Build cart-pendulum geometry using helper method
            self._build_cart_pendulum_geometry(self.viz_plant)
            
            self.viz_plant.Finalize()
            print(colored("  ✓ Visualization plant created", "green"))
        
        if self.scene_graph is None:
            print(colored(f"⚠ Visualization not available (no geometry)", 'yellow'))
            return
        
        self.meshcat = StartMeshcat()
        visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder, self.scene_graph, self.meshcat
        )
        
        print(colored(f"✓ Meshcat visualization enabled", 'green'))
        print(colored(f"  URL: {self.meshcat.web_url()}", 'cyan'))
    
    def build_diagram(self):
        """Build the diagram."""
        self.diagram = self.builder.Build()
        print(colored(f"✓ Diagram built successfully", 'green'))
    
    def create_simulator(self):
        """Create simulator."""
        self.simulator = Simulator(self.diagram)
        self.simulator.set_target_realtime_rate(self.simulation_config.realtime_rate)
        
        context = self.simulator.get_mutable_context()
        
        # For BuildSystem mode, we need to initialize the internal states directly
        if self.controller_mode == 'finite-horizon-lqr-for-min-effort':
            # Use user's initial angle input
            initial_theta = np.deg2rad(args.initial_theta)
            
            # Initialize linearized plant state [x, θ, ẋ, θ̇]
            linearized_context = self.system.linearized_plant.GetMyMutableContextFromRoot(context)
            initial_plant_state = np.array([0.0, initial_theta, 0.0, 0.0])
            linearized_context.get_mutable_continuous_state_vector().SetFromVector(initial_plant_state)
            
            # Initialize muscle force to 0
            muscle_context = self.system.muscle.GetMyMutableContextFromRoot(context)
            muscle_context.get_mutable_continuous_state_vector().SetFromVector([0.0])
            
            # Initialize ZFT reference [y_ref, v_ref] to [0, 0]
            zft_context = self.system.zft.GetMyMutableContextFromRoot(context)
            zft_context.get_mutable_continuous_state_vector().SetFromVector([0.0, 0.0])
            
            print(colored(f"  BuildSystem mode: x=0.0m, θ={args.initial_theta:.1f}° ({np.rad2deg(initial_theta):.1f}°)", 'cyan'))
        else:
            # Standard mode: set initial conditions via plant
            plant_context = self.plant.GetMyMutableContextFromRoot(context)
            
            if self.use_muscle_dynamics:
                initial_x = 0.0
                initial_theta = self.initial_angle
                print(colored(f"  Muscle dynamics mode: x={initial_x:.3f}m, θ={np.rad2deg(initial_theta):.1f}°", 'cyan'))
            else:
                initial_x = 0.0
                initial_theta = np.deg2rad(args.initial_theta)
                
                if self.controller_mode == 'computed-torque' and self.simulation_config.trajectory_mode == 'cart-motion':
                    initial_x = self.simulation_config.cart_start_position
                    initial_theta = np.deg2rad(self.simulation_config.pendulum_start_angle)
                    print(colored(f"  Cart-motion mode: starting at x={initial_x}m, θ={self.simulation_config.pendulum_start_angle}°", 'yellow'))
            
            self.plant.SetPositions(plant_context, [initial_x, initial_theta])
            self.plant.SetVelocities(plant_context, [0.0, 0.0])
        
        self.diagram.ForcedPublish(context)
        
        print(colored(f"✓ Simulator created", 'green'))
    
    def run_simulation(self):
        """Run the simulation."""
        print(colored("\n" + "="*70, 'yellow'))
        print(colored("Running Simulation", 'yellow', attrs=['bold']))
        print(colored("="*70, 'yellow'))
        
        # Print Meshcat URL if visualization is enabled
        if self.meshcat is not None:
            print(colored(f"\n🌐 Meshcat Visualization: {self.meshcat.web_url()}", 'cyan', attrs=['bold']))
            print(colored("   Open this URL in your browser to view the simulation\n", 'cyan'))
        
        context = self.simulator.get_context()
        
        # For BuildSystem, we can't access plant context the normal way
        if self.controller_mode != 'finite-horizon-lqr-for-min-effort':
            plant_context = self.plant.GetMyContextFromRoot(context)
        
        t_next_print = 0.0
        t_next_log = 0.0
        t_next_viz = 0.0
        viz_interval = 0.05
        
        while context.get_time() < self.simulation_config.simulation_time:
            self.simulator.AdvanceTo(context.get_time() + self.simulation_config.timestep)
            
            t = context.get_time()
            
            if t >= t_next_viz and self.meshcat is not None and hasattr(self, 'diagram'):
                # For BuildSystem mode, manually update viz plant state from linearized plant state
                if self.controller_mode == 'finite-horizon-lqr-for-min-effort' and hasattr(self, 'viz_plant'):
                    # Get linearized plant state [x, θ, ẋ, θ̇]
                    linearized_context = self.system.linearized_plant.GetMyContextFromRoot(context)
                    plant_state = linearized_context.get_continuous_state_vector().CopyToVector()
                    
                    # Update viz plant
                    viz_context = self.viz_plant.GetMyMutableContextFromRoot(context)
                    self.viz_plant.SetPositionsAndVelocities(viz_context, plant_state)
                
                self.diagram.ForcedPublish(context)
                t_next_viz += viz_interval
            
            if t >= t_next_print:
                if self.controller_mode == 'finite-horizon-lqr-for-min-effort':
                    # For BuildSystem, read the 7D state output
                    full_state = self.system.get_state_output_port().Eval(
                        self.system.state_mux.GetMyContextFromRoot(context)
                    )
                    x, theta, x_dot, theta_dot = full_state[0], full_state[1], full_state[2], full_state[3]
                    F, y_ref, v_ref = full_state[4], full_state[5], full_state[6]
                    theta_deg = np.rad2deg(theta)
                    print(f"[{t:5.2f}s/{self.simulation_config.simulation_time}s {int(100*t/self.simulation_config.simulation_time):3d}%] "
                          f"x={x:6.3f}m θ={theta_deg:7.2f}° | "
                          f"ẋ={x_dot:6.3f}m/s θ̇={np.rad2deg(theta_dot):7.2f}°/s | F={F:6.2f}N")
                else:
                    state = self.plant.GetPositionsAndVelocities(plant_context)
                    x, theta, x_dot, theta_dot = state
                    theta_deg = np.rad2deg(theta)
                    print(f"[{t:5.2f}s/{self.simulation_config.simulation_time}s {int(100*t/self.simulation_config.simulation_time):3d}%] "
                          f"x={x:6.3f}m θ={theta_deg:7.2f}° | "
                          f"ẋ={x_dot:6.3f}m/s θ̇={np.rad2deg(theta_dot):7.2f}°/s")
                
                t_next_print += self.simulation_config.print_interval
            
            if t >= t_next_log:
                self.time_log.append(t)
                
                if self.controller_mode == 'finite-horizon-lqr-for-min-effort':
                    # Log 7D state for BuildSystem
                    full_state = self.system.get_state_output_port().Eval(
                        self.system.state_mux.GetMyContextFromRoot(context)
                    )
                    self.state_log.append(full_state.copy())
                    
                    # Log controller output
                    controller_context = self.controller.GetMyContextFromRoot(context)
                    u_command = self.controller.get_output_port(0).Eval(controller_context)
                    self.force_log.append(u_command[0])
                else:
                    current_state = self.plant.GetPositionsAndVelocities(plant_context).copy()
                    self.state_log.append(current_state)
                    
                    controller_context = self.controller.GetMyContextFromRoot(context)
                    force = self.controller.get_output_port(0).Eval(controller_context)
                    self.force_log.append(force[0])
                    
                    if self.controller_mode == 'computed-torque':
                        traj_gen = self.controller.trajectory_generator
                        q_d, v_d, a_d = traj_gen.compute_trajectory(t)
                        desired_state = np.concatenate([q_d, v_d])
                        self.desired_state_log.append(desired_state)
                        
                        q = current_state[:2]
                        v = current_state[2:]
                        theta = q[1]
                        theta_d = q_d[1]
                        theta_normalized = np.arctan2(np.sin(theta), np.cos(theta))
                        theta_d_normalized = np.arctan2(np.sin(theta_d), np.cos(theta_d))
                        
                        errors = np.array([
                            q_d[0] - q[0],
                            theta_d_normalized - theta_normalized,
                            v_d[0] - v[0],
                            v_d[1] - v[1]
                        ])
                        self.error_log.append(errors)
                
                t_next_log += self.simulation_config.logging_interval
        
        print(colored(f"\n✓ Simulation completed successfully!", 'green', attrs=['bold']))
    
    def run_scene_viz(self):
        """Run interactive scene visualization."""
        if self.meshcat is None or self.scene_graph is None:
            print(colored("❌ Scene visualization requires MultibodyPlant geometry", 'red'))
            return
        
        print(colored("\n" + "="*70, 'cyan', attrs=['bold']))
        print(colored("Interactive Scene Visualization", 'cyan', attrs=['bold']))
        print(colored("="*70, 'cyan', attrs=['bold']))
        
        print(colored("\nVisualization Mode: Interactive Static Scene", 'yellow'))
        print(colored("  - No physics simulation", 'yellow'))
        print(colored("  - Manual state control via terminal", 'yellow'))
        print(colored("  - Type 'q' to exit\n", 'yellow'))
        
        print(colored(f"\n✓ Meshcat URL: {self.meshcat.web_url()}", 'green', attrs=['bold']))
        print(colored("  👉 Open this URL in your browser to view the scene\n", 'yellow', attrs=['bold']))
        
        context = self.simulator.get_context()
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        
        self.diagram.ForcedPublish(context)
        
        state = self.plant.GetPositionsAndVelocities(plant_context)
        x, theta, x_dot, theta_dot = state
        print(colored(f"\nInitial State:", 'magenta', attrs=['bold']))
        print(colored(f"  Cart position: {x:+.3f} m", 'cyan'))
        print(colored(f"  Pendulum angle: {np.rad2deg(theta):+.2f}° ({theta:+.4f} rad)", 'cyan'))
        print(colored(f"  Cart velocity: {x_dot:+.3f} m/s", 'cyan'))
        print(colored(f"  Pendulum velocity: {np.rad2deg(theta_dot):+.2f}°/s ({theta_dot:+.4f} rad/s)", 'cyan'))
        
        print("\n" + "=" * 70)
        print("Interactive State Control")
        print("=" * 70)
        print(f"\nEnter cart position and pendulum angle (space-separated):")
        print(f"  Format: <cart_x_meters> <pendulum_angle_degrees>")
        print(f"  Examples:")
        print(colored(f"    0 0      ", 'yellow') + "- Cart at center, pendulum hanging down")
        print(colored(f"    0.5 45   ", 'yellow') + "- Cart at 0.5m, pendulum at 45°")
        print(colored(f"    -1.0 180 ", 'yellow') + "- Cart at -1m, pendulum upright")
        print(f"\n  Type 'q' or 'quit' to exit")
        print("=" * 70 + "\n")
        
        try:
            while True:
                user_input = input(f"\nState [x(m), θ(°)]: ").strip()
                
                if user_input.lower() in ['q', 'quit', 'exit']:
                    break
                
                try:
                    values = [float(v.strip()) for v in user_input.split()]
                    if len(values) != 2:
                        print(colored(f"❌ Error: Expected 2 values, got {len(values)}", 'red'))
                        continue
                    
                    x_desired, theta_deg = values
                    theta_desired = np.deg2rad(theta_deg)
                    
                    self.plant.SetPositions(plant_context, [x_desired, theta_desired])
                    self.plant.SetVelocities(plant_context, [0.0, 0.0])
                    self.diagram.ForcedPublish(context)
                    
                    state = self.plant.GetPositionsAndVelocities(plant_context)
                    x_actual, theta_actual, _, _ = state
                    theta_actual_deg = np.rad2deg(theta_actual)
                    
                    print(colored(f"\n← Updated state:", 'cyan'))
                    print(colored(f"    Cart position: {x_actual:+.3f} m", 'cyan'))
                    print(colored(f"    Pendulum angle: {theta_actual_deg:+.2f}°", 'cyan'))
                    
                except ValueError:
                    print(colored(f"❌ Error: Invalid input", 'red'))
        
        except KeyboardInterrupt:
            print(colored("\n\n✓ Scene visualization closed by user", 'green'))
        
        print(colored("\n" + "="*70, 'green'))
        print(colored("Scene visualization complete!", 'green', attrs=['bold']))
        print(colored("="*70 + "\n", 'green'))
    
    def extract_data(self):
        """Extract and print data summary."""
        if len(self.time_log) == 0:
            print(colored("No data to extract", 'yellow'))
            return
        
        print(colored(f"Data logged: {len(self.time_log)} samples", 'cyan'))
    
    def plot_results(self):
        """Plot simulation results."""
        if len(self.time_log) == 0:
            return
        
        times = np.array(self.time_log)
        states = np.array(self.state_log)
        forces = np.array(self.force_log)
        
        if CONTROLLER_MODE == 'computed-torque' and len(self.desired_state_log) > 0:
            desired_states = np.array(self.desired_state_log)
            errors = np.array(self.error_log)
            
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            
            axes[0,0].plot(times, states[:, 0], 'b-', linewidth=2, label='Actual')
            axes[0,0].plot(times, desired_states[:, 0], 'r--', linewidth=2, label='Desired')
            axes[0,0].set_ylabel('Cart Position (m)', fontsize=12)
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].legend()
            axes[0,0].set_title('Cart Position Tracking', fontweight='bold')
            
            axes[0,1].plot(times, errors[:, 0]*1000, 'r-', linewidth=2)
            axes[0,1].set_ylabel('Position Error (mm)', fontsize=12)
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[0,1].set_title(f'Cart Error (RMS: {np.sqrt(np.mean(errors[:,0]**2))*1000:.2f} mm)', fontweight='bold')
            
            axes[1,0].plot(times, np.rad2deg(states[:, 1]), 'b-', linewidth=2, label='Actual')
            axes[1,0].plot(times, np.rad2deg(desired_states[:, 1]), 'r--', linewidth=2, label='Desired')
            axes[1,0].set_ylabel('Pendulum Angle (°)', fontsize=12)
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].legend()
            axes[1,0].set_title('Pendulum Angle Tracking', fontweight='bold')
            
            axes[1,1].plot(times, np.rad2deg(errors[:, 1]), 'r-', linewidth=2)
            axes[1,1].set_ylabel('Angle Error (°)', fontsize=12)
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[1,1].set_title(f'Pendulum Error (RMS: {np.sqrt(np.mean(errors[:,1]**2))*180/np.pi:.2f}°)', fontweight='bold')
            
            axes[2,0].plot(times, states[:, 2], 'b-', linewidth=2, label='Cart velocity')
            axes[2,0].plot(times, np.rad2deg(states[:, 3]), 'r-', linewidth=2, label='Pendulum velocity')
            axes[2,0].set_ylabel('Velocities (m/s, °/s)', fontsize=12)
            axes[2,0].set_xlabel('Time (s)', fontsize=12)
            axes[2,0].grid(True, alpha=0.3)
            axes[2,0].legend(fontsize=9)
            axes[2,0].set_title('Velocities', fontweight='bold')
            
            axes[2,1].plot(times, forces, 'g-', linewidth=2)
            axes[2,1].set_ylabel('Control Force (N)', fontsize=12)
            axes[2,1].set_xlabel('Time (s)', fontsize=12)
            axes[2,1].grid(True, alpha=0.3)
            axes[2,1].set_title('Control Effort', fontweight='bold')
            
            print(colored("\n" + "="*70, 'cyan'))
            print(colored("TRACKING PERFORMANCE METRICS", 'cyan', attrs=['bold']))
            print(colored("="*70, 'cyan'))
            print(colored(f"Cart Position Error:", 'yellow'))
            print(colored(f"  RMS: {np.sqrt(np.mean(errors[:,0]**2))*1000:.2f} mm", 'cyan'))
            print(colored(f"Pendulum Angle Error:", 'yellow'))
            print(colored(f"  RMS: {np.sqrt(np.mean(errors[:,1]**2))*180/np.pi:.2f}°", 'cyan'))
            print(colored("="*70 + "\n", 'cyan'))
        
        else:
            # Check if this is OFC mode with 7D state
            if self.controller_mode == 'finite-horizon-lqr-for-min-effort' and states.shape[1] == 7:
                # OFC Mode: 7D state [x, θ, ẋ, θ̇, F, y_ref, v_ref]
                fig = plt.figure(figsize=(18, 12))
                gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
                
                # Row 1: Cart position and pendulum angle
                ax_cart_pos = fig.add_subplot(gs[0, 0])
                ax_pend_angle = fig.add_subplot(gs[0, 1])
                ax_y_ref = fig.add_subplot(gs[0, 2])
                
                # Row 2: Cart velocity and pendulum velocity
                ax_cart_vel = fig.add_subplot(gs[1, 0])
                ax_pend_vel = fig.add_subplot(gs[1, 1])
                ax_v_ref = fig.add_subplot(gs[1, 2])
                
                # Row 3: Muscle force and impedance force
                ax_muscle_force = fig.add_subplot(gs[2, 0])
                ax_imp_force = fig.add_subplot(gs[2, 1])
                ax_y_tracking = fig.add_subplot(gs[2, 2])
                
                # Row 4: Control effort (full width)
                ax_control = fig.add_subplot(gs[3, :])
                
                # Extract 7D state components
                x = states[:, 0]
                theta = states[:, 1]
                x_dot = states[:, 2]
                theta_dot = states[:, 3]
                F_muscle = states[:, 4]
                y_ref = states[:, 5]
                v_ref = states[:, 6]
                
                # Compute impedance force F_imp = kp*(y_ref - y) + kd*(v_ref - v)
                # For cart-pendulum: y = x (cart position), v = x_dot
                kp = self.impedance_config.kp
                kd = self.impedance_config.kd
                F_imp = kp * (y_ref - x) + kd * (v_ref - x_dot)
                
                # Cart position
                ax_cart_pos.plot(times, x, 'b-', linewidth=2.5, label='x')
                ax_cart_pos.set_ylabel('Position (m)', fontsize=12, fontweight='bold')
                ax_cart_pos.grid(True, alpha=0.3)
                ax_cart_pos.legend(loc='upper right', fontsize=11)
                ax_cart_pos.set_title('CART POSITION', fontsize=13, fontweight='bold', color='darkblue')
                
                # Pendulum angle
                pend_angle_deg = np.rad2deg(theta)
                ax_pend_angle.plot(times, pend_angle_deg, 'r-', linewidth=2.5, label='θ')
                ax_pend_angle.set_ylabel('Angle (°)', fontsize=12, fontweight='bold')
                ax_pend_angle.grid(True, alpha=0.3)
                ax_pend_angle.legend(loc='upper right', fontsize=11)
                ax_pend_angle.set_title('PENDULUM ANGLE', fontsize=13, fontweight='bold', color='darkred')
                
                # Reference position tracking
                ax_y_ref.plot(times, x, 'b-', linewidth=2.5, label='y (actual)')
                ax_y_ref.plot(times, y_ref, 'r--', linewidth=2.5, label='y_ref')
                ax_y_ref.set_ylabel('Position (m)', fontsize=12, fontweight='bold')
                ax_y_ref.grid(True, alpha=0.3)
                ax_y_ref.legend(loc='upper right', fontsize=11)
                ax_y_ref.set_title('REFERENCE TRACKING', fontsize=13, fontweight='bold', color='purple')
                
                # Cart velocity
                ax_cart_vel.plot(times, x_dot, 'b-', linewidth=2.5, label='ẋ')
                ax_cart_vel.set_ylabel('Velocity (m/s)', fontsize=12, fontweight='bold')
                ax_cart_vel.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
                ax_cart_vel.grid(True, alpha=0.3)
                ax_cart_vel.legend(loc='upper right', fontsize=11)
                ax_cart_vel.set_title('CART VELOCITY', fontsize=13, fontweight='bold', color='darkblue')
                
                # Pendulum velocity
                ax_pend_vel.plot(times, np.rad2deg(theta_dot), 'r-', linewidth=2.5, label='θ̇')
                ax_pend_vel.set_ylabel('Angular Vel (°/s)', fontsize=12, fontweight='bold')
                ax_pend_vel.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
                ax_pend_vel.grid(True, alpha=0.3)
                ax_pend_vel.legend(loc='upper right', fontsize=11)
                ax_pend_vel.set_title('PENDULUM VELOCITY', fontsize=13, fontweight='bold', color='darkred')
                
                # Reference velocity
                ax_v_ref.plot(times, x_dot, 'b-', linewidth=2.5, label='v (actual)')
                ax_v_ref.plot(times, v_ref, 'r--', linewidth=2.5, label='v_ref')
                ax_v_ref.set_ylabel('Velocity (m/s)', fontsize=12, fontweight='bold')
                ax_v_ref.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
                ax_v_ref.grid(True, alpha=0.3)
                ax_v_ref.legend(loc='upper right', fontsize=11)
                ax_v_ref.set_title('REFERENCE VELOCITY', fontsize=13, fontweight='bold', color='purple')
                
                # Muscle force
                ax_muscle_force.plot(times, F_muscle, 'm-', linewidth=2.5, label='F (muscle)')
                ax_muscle_force.set_ylabel('Force (N)', fontsize=12, fontweight='bold')
                ax_muscle_force.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
                ax_muscle_force.grid(True, alpha=0.3)
                ax_muscle_force.legend(loc='upper right', fontsize=11)
                ax_muscle_force.set_title('MUSCLE FORCE STATE', fontsize=13, fontweight='bold', color='darkmagenta')
                
                # Impedance force
                ax_imp_force.plot(times, F_imp, 'orange', linewidth=2.5, label='F_imp')
                ax_imp_force.set_ylabel('Force (N)', fontsize=12, fontweight='bold')
                ax_imp_force.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
                ax_imp_force.grid(True, alpha=0.3)
                ax_imp_force.legend(loc='upper right', fontsize=11)
                ax_imp_force.set_title('IMPEDANCE FORCE', fontsize=13, fontweight='bold', color='darkorange')
                
                # Tracking error
                tracking_error = y_ref - x
                ax_y_tracking.plot(times, tracking_error * 1000, 'purple', linewidth=2.5, label='Error (y_ref - y)')
                ax_y_tracking.set_ylabel('Error (mm)', fontsize=12, fontweight='bold')
                ax_y_tracking.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
                ax_y_tracking.grid(True, alpha=0.3)
                ax_y_tracking.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                ax_y_tracking.legend(loc='upper right', fontsize=11)
                rms_error = np.sqrt(np.mean(tracking_error**2)) * 1000
                ax_y_tracking.set_title(f'TRACKING ERROR (RMS: {rms_error:.2f} mm)', 
                                       fontsize=13, fontweight='bold', color='purple')
                
                # Control effort (human command u)
                ax_control.plot(times, forces, 'g-', linewidth=2.5, label='u (human command)')
                ax_control.set_ylabel('Command (N)', fontsize=12, fontweight='bold')
                ax_control.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
                ax_control.grid(True, alpha=0.3)
                ax_control.legend(loc='upper right', fontsize=11)
                ax_control.set_title('CONTROL EFFORT (HUMAN COMMAND)', fontsize=13, fontweight='bold', color='darkgreen')
                
            else:
                # Standard mode (4D state)
                fig = plt.figure(figsize=(16, 10))
                gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
                
                ax_cart_pos = fig.add_subplot(gs[0, 0])
                ax_cart_vel = fig.add_subplot(gs[1, 0])
                ax_pend_angle = fig.add_subplot(gs[0, 1])
                ax_pend_vel = fig.add_subplot(gs[1, 1])
                ax_control = fig.add_subplot(gs[2, :])
                
                ax_cart_pos.plot(times, states[:, 0], 'b-', linewidth=2.5, label='Position')
                ax_cart_pos.set_ylabel('Position (m)', fontsize=12, fontweight='bold')
                ax_cart_pos.grid(True, alpha=0.3)
                ax_cart_pos.legend(loc='upper right', fontsize=11)
                ax_cart_pos.set_title('CART POSITION', fontsize=13, fontweight='bold', color='darkblue')
                
                ax_cart_vel.plot(times, states[:, 2], 'b-', linewidth=2.5, label='Velocity')
                ax_cart_vel.set_ylabel('Velocity (m/s)', fontsize=12, fontweight='bold')
                ax_cart_vel.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
                ax_cart_vel.grid(True, alpha=0.3)
                ax_cart_vel.legend(loc='upper right', fontsize=11)
                ax_cart_vel.set_title('CART VELOCITY', fontsize=13, fontweight='bold', color='darkblue')
                
                pend_angle_deg = np.rad2deg(states[:, 1])
                ax_pend_angle.plot(times, pend_angle_deg, 'r-', linewidth=2.5, label='Angle')
                ax_pend_angle.set_ylabel('Angle (°)', fontsize=12, fontweight='bold')
                ax_pend_angle.grid(True, alpha=0.3)
                ax_pend_angle.legend(loc='upper right', fontsize=11)
                ax_pend_angle.set_title('PENDULUM ANGLE', fontsize=13, fontweight='bold', color='darkred')
                
                ax_pend_vel.plot(times, np.rad2deg(states[:, 3]), 'r-', linewidth=2.5, label='Angular Velocity')
                ax_pend_vel.set_ylabel('Angular Velocity (°/s)', fontsize=12, fontweight='bold')
                ax_pend_vel.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
                ax_pend_vel.grid(True, alpha=0.3)
                ax_pend_vel.legend(loc='upper right', fontsize=11)
                ax_pend_vel.set_title('PENDULUM VELOCITY', fontsize=13, fontweight='bold', color='darkred')
                
                ax_control.plot(times, forces, 'g-', linewidth=2.5, label='Control Force')
                ax_control.set_ylabel('Force (N)', fontsize=12, fontweight='bold')
                ax_control.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
                ax_control.grid(True, alpha=0.3)
                ax_control.legend(loc='upper right', fontsize=11)
                ax_control.set_title('CONTROL EFFORT', fontsize=13, fontweight='bold', color='darkgreen')
        
        plt.suptitle(f'Cart-Pendulum Simulation - {CONTROLLER_MODE.upper()} Controller', 
                     fontsize=14, fontweight='bold')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plots/cart_pendulum_{CONTROLLER_MODE}_{timestamp}.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(colored(f"✓ Plot saved: {filename}", 'green'))
        
        plt.show()
    
    def print_summary(self):
        """Print simulation summary."""
        print(colored("\n" + "="*70, 'cyan'))
        print(colored("SIMULATION SUMMARY", 'cyan', attrs=['bold']))
        print(colored("="*70, 'cyan'))
        print(colored(f"Controller Mode: {self.controller_mode}", 'yellow'))
        print(colored(f"Simulation Duration: {self.simulation_config.simulation_time}s", 'yellow'))
        print(colored(f"Data Samples: {len(self.time_log)}", 'yellow'))
        if self.meshcat is not None:
            print(colored(f"Meshcat URL: {self.meshcat.web_url()}", 'yellow'))
        print(colored("="*70 + "\n", 'cyan'))
    
    def run_full_simulation(self):
        """Execute complete simulation pipeline."""
        try:
            self.setup_drake_system()
            self.add_controller()
            self.setup_visualization()
            self.build_diagram()
            self.create_simulator()
            
            if self.use_muscle_dynamics or self.controller_mode == 'scene-viz':
                if self.controller_mode == 'scene-viz':
                    self.run_scene_viz()
                else:
                    self.run_simulation()
            else:
                self.run_simulation()
                self.extract_data()
                self.plot_results()
                self.print_summary()
        
        except KeyboardInterrupt:
            print(colored("\n\nSimulation interrupted by user", 'yellow'))
        except Exception as e:
            print(colored(f"\n\nError: {e}", 'red'))
            import traceback
            traceback.print_exc()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function using DrakeSceneManager."""
    print("\n" + "=" * 70)
    print(colored("CART-PENDULUM SYSTEM - Drake Controller Architecture", 'cyan', attrs=['bold']))
    print(colored("Classic Underactuated Control Problem", 'cyan'))
    print("=" * 70)
    print(colored(f"Controller Mode: {CONTROLLER_MODE}", 'yellow', attrs=['bold']))
    print(colored(f"Plant Type: {PLANT_TYPE}", 'yellow'))
    print(colored(f"Initial Angle: {args.initial_theta}° (0=up, 180=down)", 'yellow'))
    print(colored(f"Time Step: {TIMESTEP} s", 'yellow'))
    print(colored(f"Duration: {SIMULATION_TIME} s", 'yellow'))
    print(colored(f"Visualization: {'Enabled' if args.visualize else 'Disabled'}", 'yellow'))
    print("=" * 70 + "\n")
    
    # Use DrakeSceneManager for all simulation modes
    manager = DrakeSceneManager(
        controller_mode=CONTROLLER_MODE,
        plant_type=PLANT_TYPE,
        visualize=args.visualize
    )
    
    manager.run_full_simulation()
    
    print(colored("\n" + "="*70, 'green'))
    print(colored("Execution Complete!", 'green', attrs=['bold']))
    print(colored("="*70 + "\n", 'green'))


def plot_buildsystem_diagram():
    """
    Generate and display the BuildSystem block diagram.
    
    This creates a minimal example showing the full OFC architecture with:
    - Linearized cart-pendulum plant (created internally)
    - Muscle dynamics
    - ZFT reference mass
    - Impedance force
    """
    print("\n" + "=" * 70)
    print(colored("BUILDSYSTEM DIAGRAM GENERATOR", 'cyan', attrs=['bold']))
    print("=" * 70 + "\n")
    
    # Create BuildSystem - it will create the linearized system internally
    builder = DiagramBuilder()
    
    # Create configs for all systems
    impedance_cfg = ImpedanceForceConfig(kp=50.0, kd=10.0)
    zft_cfg = ZFTReferenceMassConfig(
        Mh=10.0,
        kp=impedance_cfg.kp,
        kd=impedance_cfg.kd,
        yref0=0.0,
        vref0=0.0
    )
    
    build_system = BuildSystem(
        physics_config=PHYSICS_CONFIG,
        builder=builder,
        muscle_config=MUSCLE_DYNAMICS_CONFIG,
        impedance_config=impedance_cfg,
        zft_config=zft_cfg,
        assemble_output_state=True
    )
    
    print(colored("✓ BuildSystem wrapper created", 'green'))
    
    # Build the system
    build_system.build()
    print(colored("✓ System built with all components", 'green'))
    
    # Plot the diagram
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plots/buildsystem_diagram_{timestamp}.png"
    
    build_system.plot_diagram(filename=filename, show=True)
    
    print(colored("\n✓ BuildSystem diagram generation complete!", 'green', attrs=['bold']))
    print(colored(f"  Diagram saved to: {filename}\n", 'cyan'))


if __name__ == "__main__":
    # Check if user wants to plot the diagram
    if args.plot_diagram:
        plot_buildsystem_diagram()
    else:
        main()
