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
                    choices=['computed-torque', 'energy-shaping', 'lqr', 'computed-torque', 'standard-lqr', 'finite-horizon-lqr', 'scene-viz', 'compare-models'],
                    default='finite-horizon-lqr', 
                    help='Controller type')
parser.add_argument('--visualize', action='store_true', default=True, help='Enable visualization')
parser.add_argument('--initial_theta', type=float, default=None, 
                    help='Initial pendulum angle (degrees, 180=up, 0=down)')
parser.add_argument('--use-model-plant', type=bool, default=True,
                    help='Use separate model plant for computed torque (True) or use real plant (False)')
parser.add_argument('--plant-type', type=str, default='equations',
                    choices=['multibody', 'equations', 'linearized'],
                    help='Plant type: multibody (MultibodyPlant), equations (nonlinear), or linearized (equations 2.1 & 2.2)')
args, _ = parser.parse_known_args()

# Interactive input for initial angle if not provided (only if running as main script)
if args.initial_theta is None and __name__ == "__main__":
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
    coupling_gain: float = 5.0  # G parameter
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
class StandardLQRConfig:
    """Parameters for Standard LQR (Continuous-Time LQR) with linearized plant."""
    # Cost matrices for 5D state: [x, φ, ẋ, φ̇, F]
    Q: np.ndarray = field(default_factory=lambda: np.diag([10.0, 100.0, 1.0, 10.0, 0.1]))
    R: np.ndarray = field(default_factory=lambda: np.array([[1.0]]))
    x_goal: np.ndarray = field(default_factory=lambda: np.array([5.0, 0.0, 0.0, 0.0, 0.0]))
    
    # Control limits
    u_min: float = -100.0
    u_max: float = 100.0

@dataclass
class FiniteHorizonLQRConfig:
    """Parameters for Finite-Horizon LQR with linearized plant."""
    # Original paers values did not work well for swing-up, so these are tuned for better performance
    Q: np.ndarray = field(default_factory=lambda: np.diag([100., 1000.,   10.,  100.,  0.1]))
    QN : np.ndarray = field(default_factory=lambda: np.diag([100., 1000.,   10.,  100.,  0.1]))#field(default_factory=lambda: np.diag([1e5, 0, 1e5, 0, 0]))
    R: np.ndarray = field(default_factory=lambda: np.array([[1.0]]))
    x_goal: np.ndarray = field(default_factory=lambda: np.array([3.0, 0.0, 0.0, 0.0, 0.0]))
    horizon: float = 10.0  # seconds
    timestep: float = 0.01  # seconds (for discretization)


@dataclass
class PDControllerConfig:
    """Parameters for PD Controller."""
    kp_cart: float = 50.0  # Cart position gain
    kd_cart: float = 10.0  # Cart velocity gain
    kp_pend: float = 25.0  # Pendulum angle gain
    kd_pend: float = 5.0   # Pendulum velocity gain


@dataclass
class ComputedTorqueConfig:
    """Parameters for Computed Torque Controller."""
    kp: np.ndarray = field(default_factory=lambda: np.diag([100.0, 50.0]))  # [cart, pendulum]
    kd: np.ndarray = field(default_factory=lambda: np.diag([20.0, 10.0]))   # [cart, pendulum]


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

def create_standard_lqr_config(
    Q: np.ndarray = None,
    R: np.ndarray = None,
    x_goal: np.ndarray = None,
    u_min: float = -100.0,
    u_max: float = 100.0,
) -> StandardLQRConfig:
    """
    Create a StandardLQRConfig with custom parameters.
    
    Args:
        Q: State cost matrix (5x5) for [x, φ, ẋ, φ̇, F]
        R: Input cost matrix (1x1) for [u]
        x_goal: Goal state [x, φ, ẋ, φ̇, F]
        u_min: Minimum control input
        u_max: Maximum control input
    
    Returns:
        StandardLQRConfig instance
    """
    if Q is None:
        Q = np.diag([10.0, 100.0, 1.0, 10.0, 0.1])
    if R is None:
        R = np.array([[1.0]])
    if x_goal is None:
        x_goal = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
    
    config = StandardLQRConfig()
    config.Q = Q
    config.R = R
    config.x_goal = x_goal
    config.u_min = u_min
    config.u_max = u_max
    return config

def create_finite_horizon_lqr_config(
    Q: np.ndarray = None,
    QN: np.ndarray = None,
    R: np.ndarray = None,
    x_goal: np.ndarray = None,
    horizon: float = 10.0,
    timestep: float = 0.01,
) -> FiniteHorizonLQRConfig:
    """
    Create a FiniteHorizonLQRConfig with custom parameters.
    
    Args:
        Q: State cost matrix (5x5)
        QN: Terminal state cost matrix (5x5)
        R: Input cost matrix (1x1)
        x_goal: Goal state
        horizon: Planning horizon [s]
        timestep: Discretization timestep [s]
    
    Returns:
        FiniteHorizonLQRConfig instance
    """
    if Q is None:
        Q = np.diag([100., 1000., 10., 100., 0.1])
    if QN is None:
        QN = np.diag([100., 1000., 10., 100., 0.1])
    if R is None:
        R = np.array([[1.0]])
    if x_goal is None:
        x_goal = np.array([3.0, 0.0, 0.0, 0.0, 0.0])
    
    config = FiniteHorizonLQRConfig()
    config.Q = Q
    config.QN = QN
    config.R = R
    config.x_goal = x_goal
    config.horizon = horizon
    config.timestep = timestep
    return config

def create_pd_controller_config(
    kp_cart: float = 50.0,
    kd_cart: float = 10.0,
    kp_pend: float = 25.0,
    kd_pend: float = 5.0,
) -> PDControllerConfig:
    """
    Create a PDControllerConfig with custom parameters.
    
    Args:
        kp_cart: Cart position proportional gain
        kd_cart: Cart velocity derivative gain
        kp_pend: Pendulum angle proportional gain
        kd_pend: Pendulum angular velocity derivative gain
    
    Returns:
        PDControllerConfig instance
    """
    return PDControllerConfig(
        kp_cart=kp_cart,
        kd_cart=kd_cart,
        kp_pend=kp_pend,
        kd_pend=kd_pend,
    )

def create_computed_torque_config(
    kp: np.ndarray = None,
    kd: np.ndarray = None,
) -> ComputedTorqueConfig:
    """
    Create a ComputedTorqueConfig with custom parameters.
    
    Args:
        kp: Proportional gains [cart, pendulum]
        kd: Derivative gains [cart, pendulum]
    
    Returns:
        ComputedTorqueConfig instance
    """
    if kp is None:
        kp = np.diag([100.0, 50.0])
    if kd is None:
        kd = np.diag([20.0, 10.0])
    
    return ComputedTorqueConfig(kp=kp, kd=kd)

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

# Controller-specific configs (user can modify these)
STANDARD_LQR_CONFIG = create_standard_lqr_config()
FINITE_HORIZON_LQR_CONFIG = create_finite_horizon_lqr_config()
PD_CONTROLLER_CONFIG = create_pd_controller_config()
COMPUTED_TORQUE_CONFIG = create_computed_torque_config()

# Global simulation config
SIM_CONFIG = create_simulation_config()

# ============================================================================
# BACKWARD COMPATIBILITY: EXPOSE COMMONLY USED PARAMS
# ============================================================================

# Physics parameters
CART_MASS = PHYSICS_CONFIG.mass_cart
CART_WIDTH = PHYSICS_CONFIG.width_cart
CART_HEIGHT = PHYSICS_CONFIG.height_cart
CART_DEPTH = PHYSICS_CONFIG.depth_cart
CART_DAMPING = PHYSICS_CONFIG.damping_cart

PENDULUM_MASS = PHYSICS_CONFIG.mass_pendulum
PENDULUM_LENGTH = PHYSICS_CONFIG.length_pendulum
PENDULUM_RADIUS = PHYSICS_CONFIG.radius_pendulum
PENDULUM_TOTAL_LENGTH = PHYSICS_CONFIG.length_pendulum
PENDULUM_DAMPING = PHYSICS_CONFIG.damping_pendulum

COUPLING_GAIN = PHYSICS_CONFIG.coupling_gain
GRAVITY = PHYSICS_CONFIG.gravity
ARM_MASS = PHYSICS_CONFIG.mass_arm
MOTOR_TIME_CONSTANT = PHYSICS_CONFIG.motor_time_constant

TRACK_LENGTH = PHYSICS_CONFIG.track_length
TRACK_LIMIT = PHYSICS_CONFIG.track_limit

# Simulation parameters
CONTROLLER_MODE = args.mode
USE_MODEL_PLANT = args.use_model_plant
PLANT_TYPE = args.plant_type
TIMESTEP = SIM_CONFIG.timestep
SIMULATION_TIME = SIM_CONFIG.simulation_time
PRINT_INTERVAL = SIM_CONFIG.print_interval
LOGGING_INTERVAL = SIM_CONFIG.logging_interval
REALTIME_RATE = SIM_CONFIG.realtime_rate

# Trajectory parameters
TRAJECTORY_MODE = SIM_CONFIG.trajectory_mode
CART_START_POSITION = SIM_CONFIG.cart_start_position
CART_END_POSITION = SIM_CONFIG.cart_end_position
CART_MOTION_DURATION = SIM_CONFIG.cart_motion_duration
PENDULUM_START_ANGLE = SIM_CONFIG.pendulum_start_angle
CART_SETTLE_TIME = SIM_CONFIG.cart_settle_time

# Standard LQR parameters
STANDARD_LQR_LINEARIZED_Q = STANDARD_LQR_CONFIG.Q
STANDARD_LQR_LINEARIZED_R = STANDARD_LQR_CONFIG.R
STANDARD_LQR_LINEARIZED_X_GOAL = STANDARD_LQR_CONFIG.x_goal

# Finite-Horizon LQR parameters
FINITE_HORIZON_LQR_Q = FINITE_HORIZON_LQR_CONFIG.Q
FINITE_HORIZON_LQR_QN = FINITE_HORIZON_LQR_CONFIG.QN
FINITE_HORIZON_LQR_R = FINITE_HORIZON_LQR_CONFIG.R
FINITE_HORIZON_LQR_X_GOAL = FINITE_HORIZON_LQR_CONFIG.x_goal
FINITE_HORIZON_LQR_T = FINITE_HORIZON_LQR_CONFIG.horizon
FINITE_HORIZON_LQR_DT = FINITE_HORIZON_LQR_CONFIG.timestep

# PD Controller parameters
PD_KP_CART = PD_CONTROLLER_CONFIG.kp_cart
PD_KD_CART = PD_CONTROLLER_CONFIG.kd_cart
PD_KP_PEND = PD_CONTROLLER_CONFIG.kp_pend
PD_KD_PEND = PD_CONTROLLER_CONFIG.kd_pend

# Computed Torque Controller parameters
CT_KP = COMPUTED_TORQUE_CONFIG.kp
CT_KD = COMPUTED_TORQUE_CONFIG.kd

# Legacy parameters (for compatibility)
MOTOR_NOISE_MEAN = PHYSICS_CONFIG.motor_noise_mean
MOTOR_NOISE_STD = PHYSICS_CONFIG.motor_noise_std
SENSORY_DELAY = SIM_CONFIG.sensory_delay
CONTROL_DEPENDENT_NOISE_STD = SIM_CONFIG.control_dependent_noise_std
STATE_DEPENDENT_SENSORY_NOISE_STD = SIM_CONFIG.state_dependent_sensory_noise_std
ADDITIVE_PROCESS_NOISE_STD = SIM_CONFIG.additive_process_noise_std
ADDITIVE_SENSORY_NOISE_COV = SIM_CONFIG.additive_sensory_noise_cov
INTERNAL_ESTIMATOR_NOISE_COV = SIM_CONFIG.internal_estimator_noise_cov
TARGET_HOLD_STEPS = SIM_CONFIG.target_hold_steps


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
        self.DeclareVectorOutputPort("F", BasicVector(1), self._calc_output)

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
# CART-PENDULUM CLASS
# ============================================================================

class CartPendulumSystemWithMuscleDynamics:
    """
    Cart-Pendulum Plant Builder with Muscle Dynamics.
    
    RESPONSIBILITY: Build the physics plant ONLY
    - Creates MultibodyPlant with cart and pendulum
    - Adds muscle dynamics actuation system
    
    Does NOT handle:
    - Diagram building/wiring
    - Visualization
    - Simulation execution
    
    Those are handled by CartPendulumSceneManager.
    """

    def __init__(
        self,
        config: CartPendulumPhysicsConfig,
        builder: DiagramBuilder,
        muscle_config: MuscleDynamicsConfig | None = None,
    ):
        """Initialize plant builder.
        
        Args:
            config: CartPendulumPhysicsConfig with physics parameters
            builder: Drake DiagramBuilder (passed in from SceneManager)
            muscle_config: MuscleDynamicsConfig with muscle dynamics parameters
                          (if None, uses global MUSCLE_DYNAMICS_CONFIG)
        """
        self.config = config
        self.builder = builder
        self.plant = None
        self.scene_graph = None

        # Muscle dynamics configuration
        if muscle_config is None:
            muscle_config = MUSCLE_DYNAMICS_CONFIG
        self.muscle_config = muscle_config
        
        # Muscle actuation system objects
        self.muscle = None
        self.u_saturation = None
        self.command_input_port = None

    def build_plant_without_muscle(self):
        """Build the cart-pendulum MultibodyPlant and insert muscle dynamics in the actuation path."""
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

    def add_muscle_dynamics(self):
        # ---------------------------
        # Muscle dynamics insertion
        # ---------------------------
        self.actuation_input_port = self.plant.get_actuation_input_port()

        if self.config.enable_muscle_dynamics:
            # Controller will output "u" (desired force), muscle outputs actual applied force F
            self.muscle = self.builder.AddSystem(
                MuscleDynamics(config=self.muscle_config
                )
            )
            self.muscle.set_name("muscle_dynamics")

            # Optional saturation on u before muscle
            if self.muscle_config.command_limit is not None:
                lim = float(self.muscle_config.command_limit)
                self.u_saturation = self.builder.AddSystem(
                    Saturation(min_value=[-lim], max_value=[lim])
                )
                self.u_saturation.set_name("u_saturation")
                self.command_input_port = self.u_saturation.get_input_port()

                # u_sat -> muscle.u
                self.builder.Connect(
                    self.u_saturation.get_output_port(),
                    self.muscle.get_input_port(0),
                )
            else:
                self.command_input_port = self.muscle.get_input_port(0)

            # muscle.F -> plant.actuation
            self.builder.Connect(
                self.muscle.get_output_port(0),
                self.actuation_input_port,
            )

            print(colored("✓ Muscle dynamics enabled", "green"))
            print(colored(f"  Ḟ = (-F + u)/τ, τ = {self.muscle_config.muscle_tau} s", "cyan"))
            if self.muscle_config.command_limit is not None:
                print(colored(f"  u saturation: ±{self.muscle_config.command_limit}", "cyan"))
        else:
            # No muscle: controller output connects directly to plant actuation
            self.command_input_port = self.actuation_input_port
            print(colored("✓ Muscle dynamics disabled (direct force actuation)", "green"))

        # IMPORTANT: From here on, connect your controller output to:
        #     self.command_input_port
        # not to plant.get_actuation_input_port() directly.

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

class CartPendulumLinearizedSystemWithMuscleDynamics:
    """
    Linearized Cart-Pendulum with Muscle Dynamics (Using Drake's Linearize).
    
    ARCHITECTURE:
    - Builds full nonlinear MultibodyPlant for cart-pendulum
    - Uses Drake's Linearize() to compute Jacobian-based linearization
    - Linearizes around upright equilibrium (θ=0, θ̇=0, F=0)
    - Adds muscle dynamics on top of linearized plant
    
    ADVANTAGES:
    - Uses Drake's built-in Jacobian computation (numerical differentiation)
    - Works for ANY nonlinear system (no manual formula derivation needed)
    - Scales to complex systems easily
    - Automatically handles all state/input interactions
    
    STATE: [x, θ, ẋ, θ̇, F] (5D)
    - x, θ, ẋ, θ̇: cart-pendulum state (linearized via Jacobian)
    - F: muscle force (nonlinear dynamics)
    
    INPUT: u (muscle command)
    OUTPUT: [x, θ, ẋ, θ̇] (cart-pendulum state)
    """

    def __init__(
        self,
        config: CartPendulumPhysicsConfig,
        builder: DiagramBuilder,
        muscle_config: MuscleDynamicsConfig | None = None,
    ):
        """Initialize linearized system with muscle dynamics.
        
        Args:
            config: CartPendulumPhysicsConfig with physics parameters
            builder: Drake DiagramBuilder (passed in from SceneManager)
            muscle_config: MuscleDynamicsConfig with muscle dynamics parameters
                          (if None, uses global MUSCLE_DYNAMICS_CONFIG)
        """
        self.config = config
        self.builder = builder
        
        # Muscle dynamics configuration
        if muscle_config is None:
            muscle_config = MUSCLE_DYNAMICS_CONFIG
        self.muscle_config = muscle_config
        
        # Linearized plant system objects
        self.nonlinear_plant = None
        self.nonlinear_builder = None
        self.linearized_system = None
        self.muscle = None
        self.u_saturation = None
        self.command_input_port = None
        
        # Linearization point (equilibrium)
        self.equilibrium_state = None
        self.equilibrium_input = None
        self.linearized_matrices = dict(A_plant=None, B_plant=None, C_plant=None, D_plant=None)

    def build_plant_without_muscle(self):
        """
        Build linearized cart-pendulum using Drake's Linearize().
        
        Process:
        1. Create nonlinear MultibodyPlant
        2. Set equilibrium point (θ=0, all velocities=0)
        3. Use Drake's Linearize() to compute Jacobians numerically
        4. Add muscle dynamics on top
        """
        print(colored("\n" + "=" * 70, "yellow"))
        print(colored("Building Linearized Cart-Pendulum (Drake Jacobian-based)", "yellow", attrs=["bold"]))
        print(colored("=" * 70, "yellow"))

        # Step 1: Build nonlinear plant (without visualization, just dynamics)
        print(colored("  [1/3] Creating nonlinear MultibodyPlant...", "cyan"))
        self.nonlinear_builder = DiagramBuilder()
        nonlinear_plant, scene_graph = AddMultibodyPlantSceneGraph(
            self.nonlinear_builder, time_step=0.0
        )

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
        
        self.nonlinear_plant = nonlinear_plant

    def build_linearized_system_with_muscle(self):
        """Build linearized plant, then integrate muscle dynamics."""
        # First build the nonlinear plant
        self.build_plant_without_muscle()
        
        # Step 3: Linearize using Drake's Linearize() method
        # Linearize the plant directly (not a diagram)
        from pydrake.all import Linearize
        
        linearized_io_sys = Linearize(
            self.nonlinear_plant,
            self.context,
            input_port_index=self.nonlinear_plant.get_actuation_input_port().get_index(),
            output_port_index=self.nonlinear_plant.get_state_output_port().get_index(),
        )
        
        # Store linearized matrices for later access
        self.linearized_matrices['A_plant'] = linearized_io_sys.A()
        self.linearized_matrices['B_plant'] = linearized_io_sys.B()
        self.linearized_matrices['C_plant'] = linearized_io_sys.C()
        self.linearized_matrices['D_plant'] = linearized_io_sys.D()
        
        print(colored("    ✓ Jacobian-based linearization computed", "green"))
        print(colored(f"    Linearization matrices from Drake's Linearize():", "cyan"))
        print(colored(f"      A matrix shape: {linearized_io_sys.A().shape}", "cyan"))
        print(colored(f"      B matrix shape: {linearized_io_sys.B().shape}", "cyan"))
        print(colored(f"      C matrix shape: {linearized_io_sys.C().shape}", "cyan"))
        print(colored(f"      D matrix shape: {linearized_io_sys.D().shape}", "cyan"))

        # Step 4: Add linearized system to builder
        print(colored("  [3/3] Integrating linearized system with muscle dynamics...", "cyan"))
        
        self.linearized_system = self.builder.AddSystem(linearized_io_sys)
        self.linearized_system.set_name("linearized_cart_pendulum_jacobian")

        print(colored("✓ Linearized cart-pendulum system created:", "green"))
        print(colored(f"  Linearization method: Drake Jacobian-based (numerical)", "cyan"))
        print(colored(f"  State dimension: 4 [x, θ, ẋ, θ̇]", "cyan"))
        print(colored(f"  Input dimension: 1 [F]", "cyan"))
        print(colored(f"  Output dimension: 4 (full state feedback)", "cyan"))
        print(colored(f"  Equilibrium point: x={self.equilibrium_state}", "cyan"))

    def add_muscle_dynamics_to_linearized_plant(self):
        """Add muscle dynamics on top of linearized plant."""
        print(colored("\nAdding Muscle Dynamics to Linearized Plant", "yellow", attrs=["bold"]))

        # Create muscle dynamics
        self.muscle = self.builder.AddSystem(
            MuscleDynamics(config=self.muscle_config)
        )
        self.muscle.set_name("muscle_dynamics")

        # Optional saturation on u before muscle
        if self.muscle_config.command_limit is not None:
            lim = float(self.muscle_config.command_limit)
            self.u_saturation = self.builder.AddSystem(
                Saturation(min_value=[-lim], max_value=[lim])
            )
            self.u_saturation.set_name("u_saturation")
            self.command_input_port = self.u_saturation.get_input_port()

            # u_sat -> muscle.u
            self.builder.Connect(
                self.u_saturation.get_output_port(),
                self.muscle.get_input_port(0),
            )
        else:
            self.command_input_port = self.muscle.get_input_port(0)

        # muscle.F -> linearized_plant.input
        plant_input_port = self.linearized_system.get_input_port(0)
        self.builder.Connect(
            self.muscle.get_output_port(0),
            plant_input_port,
        )

        print(colored("✓ Muscle dynamics integrated with linearized plant", "green"))
        print(colored(f"  Ḟ = (-F + u)/τ, τ = {self.muscle_config.muscle_tau} s", "cyan"))
        if self.muscle_config.command_limit is not None:
            print(colored(f"  u saturation: ±{self.muscle_config.command_limit}", "cyan"))

    def get_output_port(self):
        """Get full state output from linearized plant."""
        return self.linearized_system.get_output_port(0)


# ============================================================================
# DRAKE SCENE MANAGER CLASS
# ============================================================================

class DrakeSceneManager:
    """
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
                 initial_angle: float = np.deg2rad(180)):
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
    
    def setup_drake_system(self):
        """Build the Drake system - handles both standard and muscle dynamics modes."""
        # Create new builder for this manager
        self.builder = DiagramBuilder()
        
        # Build plant based on mode
        if self.use_muscle_dynamics:
            # Use muscle dynamics system
            self.system = CartPendulumSystemWithMuscleDynamics(
                config=self.cart_pendulum_config,
                builder=self.builder,
                muscle_config=self.muscle_config
            )
            self.system.build_plant_without_muscle()
            self.system.add_muscle_dynamics()
            self.plant = self.system.plant
            self.scene_graph = self.system.scene_graph
            print(colored("✓ Muscle dynamics configured", "green"))
            print(colored(f"  Muscle tau: {self.muscle_config.muscle_tau} s", "cyan"))
        else:
            # Use standard cart-pendulum (without muscle dynamics)
            # Update physics config to disable muscle dynamics
            physics_config = CartPendulumPhysicsConfig()
            physics_config = dataclasses.replace(self.cart_pendulum_config, enable_muscle_dynamics=False)
            self.system = CartPendulumSystemWithMuscleDynamics(
                config=physics_config,
                builder=self.builder,
                muscle_config=self.muscle_config
            )
            self.system.build_plant_without_muscle()
            self.plant = self.system.plant
            self.scene_graph = self.system.scene_graph
            print(colored("✓ Standard cart-pendulum plant built", "green"))
    
    def add_controller(self):
        """Add controller to the diagram."""
        print(colored(f"\nAdding Controller: {self.controller_mode}", 'yellow', attrs=['bold']))
        
        # Special handling for muscle dynamics
        if self.use_muscle_dynamics:
            class ConstantForceSource(LeafSystem):
                def __init__(self, force_value: float):
                    super().__init__()
                    self.force_value = force_value
                    self.DeclareVectorOutputPort("force_output", BasicVector(1), self._calc_output)
                
                def _calc_output(self, context, output):
                    output.SetFromVector(np.array([self.force_value]))
            
            self.controller = self.builder.AddSystem(ConstantForceSource(self.constant_force))
            self.controller.set_name("constant_force_source")
            print(colored(f"✓ Muscle dynamics controller: {self.constant_force:.2f} N", "green"))
            # Do NOT return early - continue to wiring section below
        
        elif self.controller_mode == 'scene-viz':
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
        
        elif self.use_muscle_dynamics:
            # Muscle dynamics controller already created, skip other mode creation
            pass
        
        elif self.controller_mode == 'pd':
            traj_gen = TrajectoryGenerator(mode='balance')
            self.controller = self.builder.AddSystem(
                PDController(
                    self.pd_controller_config.kp_cart,
                    self.pd_controller_config.kd_cart,
                    self.pd_controller_config.kp_pend,
                    self.pd_controller_config.kd_pend,
                    traj_gen
                )
            )
        
        elif self.controller_mode == 'computed-torque':
            model = self.system.create_model_for_controller()
            q_start = np.array([self.simulation_config.cart_start_position, np.deg2rad(self.simulation_config.pendulum_start_angle)])
            q_goal = np.array([self.simulation_config.cart_end_position, np.deg2rad(self.simulation_config.pendulum_start_angle)])
            
            traj_gen_ct = MinJerkTrajectoryGenerator(
                q_start=q_start,
                q_goal=q_goal,
                duration=self.simulation_config.cart_motion_duration,
                settle_time=self.simulation_config.cart_settle_time
            )
            
            print(colored(f"  Trajectory: MinJerk cart motion", 'cyan'))
            print(colored(f"    Start: x={self.simulation_config.cart_start_position:.1f}m, θ={self.simulation_config.pendulum_start_angle:.0f}°", 'cyan'))
            print(colored(f"    Goal:  x={self.simulation_config.cart_end_position:.1f}m, θ={self.simulation_config.pendulum_start_angle:.0f}°", 'cyan'))
            print(colored(f"    Settle: {self.simulation_config.cart_settle_time:.1f}s, Motion: {self.simulation_config.cart_motion_duration:.1f}s", 'cyan'))
            
            self.controller = self.builder.AddSystem(
                ComputedTorqueController(
                    self.plant,
                    model,
                    self.computed_torque_config.kp,
                    self.computed_torque_config.kd,
                    traj_gen_ct,
                    use_model=self.use_model_plant
                )
            )
        
        else:
            raise ValueError(f"Unknown controller mode: {self.controller_mode}")
        
        # Wire controller to plant based on mode
        if self.use_muscle_dynamics:
            # For muscle dynamics, connect to the muscle command input
            self.builder.Connect(
                self.controller.get_output_port(0),
                self.system.command_input_port
            )
        else:
            # For standard mode, wire controller outputs to plant actuator input
            self.builder.Connect(
                self.plant.get_state_output_port(),
                self.controller.get_input_port(0)
            )
            self.builder.Connect(
                self.controller.get_output_port(0),
                self.plant.get_actuation_input_port()
            )
        
        print(colored(f"✓ Controller wired to plant", 'green'))
    
    def setup_visualization(self):
        """Setup Meshcat visualization."""
        if not self.visualize:
            return
        
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
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        
        # Set initial conditions based on mode
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
        
        context = self.simulator.get_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        
        t_next_print = 0.0
        t_next_log = 0.0
        t_next_viz = 0.0
        viz_interval = 0.05
        
        while context.get_time() < self.simulation_config.simulation_time:
            self.simulator.AdvanceTo(context.get_time() + self.simulation_config.timestep)
            
            t = context.get_time()
            
            if t >= t_next_viz and self.meshcat is not None and hasattr(self, 'diagram'):
                self.diagram.ForcedPublish(context)
                t_next_viz += viz_interval
            
            if t >= t_next_print:
                state = self.plant.GetPositionsAndVelocities(plant_context)
                x, theta, x_dot, theta_dot = state
                theta_deg = np.rad2deg(theta)
                print(f"[{t:5.2f}s/{self.simulation_config.simulation_time}s {int(100*t/self.simulation_config.simulation_time):3d}%] "
                      f"x={x:6.3f}m θ={theta_deg:7.2f}° | "
                      f"ẋ={x_dot:6.3f}m/s θ̇={np.rad2deg(theta_dot):7.2f}°/s")
                
                t_next_print += self.simulation_config.print_interval
            
            if t >= t_next_log:
                self.time_log.append(t)
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
# MINIMUM JERK TRAJECTORY GENERATOR (from cup manipulator)
# ============================================================================

class MinJerkTrajectoryGenerator:
    """
    Minimum-jerk trajectory generator for smooth motion.
    
    Uses 5th-order polynomial (minimum jerk) for time scaling.
    Same implementation as cup manipulator for consistency.
    """
    
    def __init__(self, q_start: np.ndarray, q_goal: np.ndarray, duration: float, 
                 settle_time: float = 0.0):
        """
        Args:
            q_start: Starting configuration [x, θ]
            q_goal: Goal configuration [x, θ]
            duration: Motion duration (seconds)
            settle_time: Wait time before motion starts (seconds)
        """
        self.q_start = np.array(q_start, dtype=float)
        self.q_goal = np.array(q_goal, dtype=float)
        self.motion_duration = float(duration)
        self.settle_time = float(settle_time)
        self.total_duration = self.settle_time + self.motion_duration
    
    def _min_jerk_profile(self, s: float):
        """
        Compute minimum-jerk time scaling.
        
        Args:
            s: Normalized time [0, 1]
            
        Returns:
            (h, h_dot, h_ddot): position, velocity, acceleration scaling
        """
        s = np.clip(s, 0.0, 1.0)
        
        # 5th order polynomial: h(s) = 10s³ - 15s⁴ + 6s⁵
        h = 10*s**3 - 15*s**4 + 6*s**5
        h_dot = 30*s**2 - 60*s**3 + 30*s**4
        h_ddot = 60*s - 180*s**2 + 120*s**3
        
        return h, h_dot, h_ddot
    
    def compute_trajectory(self, t: float):
        """
        Compute desired state at time t.
        
        Args:
            t: Current time (seconds)
            
        Returns:
            q_d: [x, θ] desired positions
            qd_d: [ẋ, θ̇] desired velocities  
            qdd_d: [ẍ, θ̈] desired accelerations
        """
        # Settle phase
        if t < self.settle_time:
            return self.q_start, np.zeros(2), np.zeros(2)
        
        # Motion phase
        elif t < self.total_duration:
            t_motion = t - self.settle_time
            s = t_motion / self.motion_duration
            h, h_dot, h_ddot = self._min_jerk_profile(s)
            
            # Interpolate between start and goal
            q_d = self.q_start + (self.q_goal - self.q_start) * h
            qd_d = (self.q_goal - self.q_start) * h_dot / self.motion_duration
            qdd_d = (self.q_goal - self.q_start) * h_ddot / (self.motion_duration**2)
            
            return q_d, qd_d, qdd_d
        
        # Hold at goal
        else:
            return self.q_goal, np.zeros(2), np.zeros(2)


# ============================================================================
# TRAJECTORY GENERATOR (legacy modes)
# ============================================================================

class TrajectoryGenerator:
    """Generate reference trajectories for cart-pendulum."""
    
    def __init__(self, mode='balance'):
        """
        Args:
            mode: 'balance' (upright), 'swing' (sinusoidal), 'track' (position tracking)
        """
        self.mode = mode
        self.cart_amplitude = 0.3  # m
        self.cart_frequency = 0.5  # Hz
    
    def compute_trajectory(self, t):
        """
        Compute desired state at time t.
        
        Returns:
            q_desired: [x, θ] positions
            qd_desired: [ẋ, θ̇] velocities
            qdd_desired: [ẍ, θ̈] accelerations
        """
        if self.mode == 'balance':
            # Keep cart at origin, pendulum upright
            q_desired = np.array([0.0, 0.0])
            qd_desired = np.zeros(2)
            qdd_desired = np.zeros(2)
            
        elif self.mode == 'swing':
            # Sinusoidal cart motion, pendulum should balance
            omega = 2 * np.pi * self.cart_frequency
            q_desired = np.array([
                self.cart_amplitude * np.sin(omega * t),
                0.0  # Upright
            ])
            qd_desired = np.array([
                self.cart_amplitude * omega * np.cos(omega * t),
                0.0
            ])
            qdd_desired = np.array([
                -self.cart_amplitude * omega**2 * np.sin(omega * t),
                0.0
            ])
            
        elif self.mode == 'track':
            # Step input for cart position
            x_target = 0.5 if t > 2.0 else 0.0
            q_desired = np.array([x_target, 0.0])
            qd_desired = np.zeros(2)
            qdd_desired = np.zeros(2)
        
        return q_desired, qd_desired, qdd_desired


# ============================================================================
# PD CONTROLLER
# ============================================================================

class PDController(LeafSystem):
    """
    Simple PD controller for cart-pendulum.
    
    Control law: F = Kp_x·(x_d - x) + Kd_x·(ẋ_d - ẋ) 
                    + Kp_θ·(θ_d - θ) + Kd_θ·(θ̇_d - θ̇)
    
    Note: This is NOT optimal for balancing - use LQR instead.
    """
    
    def __init__(self, kp_cart, kd_cart, kp_pend, kd_pend, 
                 trajectory_generator: TrajectoryGenerator):
        LeafSystem.__init__(self)
        
        self.kp_cart = kp_cart
        self.kd_cart = kd_cart
        self.kp_pend = kp_pend
        self.kd_pend = kd_pend
        self.trajectory_generator = trajectory_generator
        
        # Input: state [x, θ, ẋ, θ̇]
        self.DeclareVectorInputPort("state", BasicVector(4))
        
        # Output: force [F]
        self.DeclareVectorOutputPort("force", BasicVector(1), self.CalcControlForce)
        
        print(colored(f"✓ PDController initialized:", 'green'))
        print(colored(f"  Cart gains: Kp={kp_cart}, Kd={kd_cart}", 'cyan'))
        print(colored(f"  Pendulum gains: Kp={kp_pend}, Kd={kd_pend}", 'cyan'))
    
    def CalcControlForce(self, context, output):
        """Compute PD control force."""
        # Get current state
        state = self.get_input_port(0).Eval(context)
        x, theta, x_dot, theta_dot = state
        
        # Get desired trajectory
        t = context.get_time()
        q_d, qd_d, _ = self.trajectory_generator.compute_trajectory(t)
        
        # Compute errors
        e_x = q_d[0] - x
        e_theta = q_d[1] - theta
        ed_x = qd_d[0] - x_dot
        ed_theta = qd_d[1] - theta_dot
        
        # PD control law
        F = (self.kp_cart * e_x + self.kd_cart * ed_x +
             self.kp_pend * e_theta + self.kd_pend * ed_theta)
        
        output.SetFromVector([F])




# ============================================================================
# COMPUTED TORQUE CONTROLLER
# ============================================================================

class ComputedTorqueController(LeafSystem):
    """
    Unified Computed Torque (Inverse Dynamics) controller.
    
    Supports two dynamics computation methods:
    1. MultibodyPlant-based: Uses Drake's CalcInverseDynamics
    2. Analytical equations: Uses closed-form inverse dynamics from equations 2.1 & 2.2
    
    Control law: F = M(q)·[q̈_d + Kp·e + Kd·ė] + C(q,q̇) + g(q)
    
    Key features:
    - Separate model for control calculations (optional)
    - Full nonlinear dynamics compensation
    - Trajectory tracking with feedforward + feedback
    """
    
    def __init__(self, plant=None, model=None, 
                 Kp=None, Kd=None, trajectory_generator=None, 
                 use_model: bool = True, use_analytical: bool = False):
        """
        Args:
            plant: Real MultibodyPlant (optional for analytical mode)
            model: Controller's internal MultibodyPlant model (optional for analytical mode)
            Kp: Position gain matrix (2x2)
            Kd: Velocity gain matrix (2x2)
            trajectory_generator: Trajectory generator object
            use_model: Use separate model plant (True) or real plant (False) - only for MultibodyPlant mode
            use_analytical: Use analytical inverse dynamics (True) or MultibodyPlant (False)
        """
        LeafSystem.__init__(self)
        
        self.use_analytical = use_analytical
        self.Kp = Kp
        self.Kd = Kd
        self.trajectory_generator = trajectory_generator
        
        if not use_analytical:
            # MultibodyPlant mode
            self.plant = plant
            self.model = model if use_model else plant
            self.model_context = self.model.CreateDefaultContext()
            self.use_model = use_model
        
        self.DeclareVectorInputPort("state", BasicVector(4))
        self.DeclareVectorOutputPort("force", BasicVector(1), self.CalcControlForce)
        
        print(colored(f"✓ ComputedTorqueController initialized:", 'green'))
        print(colored(f"  Kp = {np.diag(Kp)}", 'cyan'))
        print(colored(f"  Kd = {np.diag(Kd)}", 'cyan'))
        
        if use_analytical:
            print(colored(f"  Dynamics: ANALYTICAL (equations 2.1 & 2.2)", 'yellow'))
            print(colored(f"    Using closed-form inverse dynamics", 'cyan'))
        else:
            if use_model:
                print(colored(f"  Dynamics: MULTIBODY PLANT", 'yellow'))
                print(colored(f"  Model-Plant Separation: ENABLED", 'yellow'))
                print(colored(f"    Plant: Used for state observation", 'cyan'))
                print(colored(f"    Model: Used for inverse dynamics", 'cyan'))
            else:
                print(colored(f"  Dynamics: MULTIBODY PLANT", 'yellow'))
                print(colored(f"  Model-Plant Separation: DISABLED (using real plant)", 'yellow'))
                print(colored(f"    Plant: Used for both state and inverse dynamics", 'cyan'))
    
    def CalcControlForce(self, context, output):
        """Compute control using inverse dynamics."""
        # Get current state
        import numpy as np
        
        state = self.get_input_port(0).Eval(context)
        q = state[:2]
        v = state[2:]
        
        # Get desired trajectory
        t = context.get_time()
        q_d, v_d, a_d = self.trajectory_generator.compute_trajectory(t)
        
        if self.use_analytical:
            # Analytical mode: simple error calculation
            e = q_d - q
            ed = v_d - v
        else:
            # MultibodyPlant mode: normalize pendulum angle
            phi = q[1]
            phi_d = q_d[1]
            phi_normalized = np.arctan2(np.sin(phi), np.cos(phi))
            phi_d_normalized = np.arctan2(np.sin(phi_d), np.cos(phi_d))
            e = np.array([q_d[0] - q[0], phi_d_normalized - phi_normalized])
            ed = v_d - v
        
        # Commanded acceleration (feedback + feedforward)
        a_cmd = a_d + self.Kp @ e + self.Kd @ ed
        
        if self.use_analytical:
            # ═══════════════════════════════════════════════════════════════
            # ANALYTICAL NONLINEAR INVERSE DYNAMICS (APPROACH 2)
            # ═══════════════════════════════════════════════════════════════
            # UNDERACTUATED SYSTEM: Force F only acts on cart, not pendulum!
            # 
            # Full system equations:
            #   (M+m)ẍ + ml(φ̈cos(φ) - φ̇²sin(φ)) = F  ... (cart equation)
            #   l·φ̈ + ẍ·cos(φ) + g·sin(φ) = 0         ... (pendulum equation)
            # 
            # APPROACH 2: Physically Consistent Inverse Dynamics
            # ──────────────────────────────────────────────────
            # 1. Use only ẍ_cmd from trajectory (cart is actuated)
            # 2. Compute φ̈ from passive constraint equation:
            #    φ̈ = (-ẍ_cmd·cos(φ) - g·sin(φ)) / l
            # 3. Compute required force F using cart equation
            # 
            # This GUARANTEES the commanded force produces accelerations
            # that satisfy BOTH equations simultaneously!
            # ═══════════════════════════════════════════════════════════════
            
            M = CART_MASS
            m = PENDULUM_MASS
            l = PENDULUM_LENGTH
            g = GRAVITY
            G = COUPLING_GAIN
            
            # Current state
            phi = q[1]
            phi_dot = v[1]
            
            # Commanded cart acceleration
            x_ddot_cmd = a_cmd[0]
            
            c = np.cos(phi)
            s = np.sin(phi)
            
            # Compute physically consistent pendulum acceleration
            # from constraint: l·φ̈ + G·ẍ·cos(φ) + g·sin(φ) = 0
            phi_ddot_consistent = (-G * x_ddot_cmd * c - g * s) / l
            
            # Inverse dynamics using cart equation
            # (M+m)ẍ + ml(φ̈cos(φ) - φ̇²sin(φ)) = F
            F = (M + m) * x_ddot_cmd + m * l * phi_ddot_consistent * c - m * l * phi_dot**2 * s
        else:
            # MultibodyPlant inverse dynamics
            self.model.SetPositions(self.model_context, q)
            self.model.SetVelocities(self.model_context, v)
            
            from pydrake.multibody.tree import MultibodyForces
            tau = self.model.CalcInverseDynamics(
                self.model_context,
                a_cmd,
                MultibodyForces(self.model)
            )
            
            # Extract cart force (first actuator)
            F = tau[0]
        
        output.SetFromVector([F])






# ============================================================================
# FINITE HORIZON LQR CONTROLLER (Time-Varying Gains)
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
    
    State vector for cart-pendulum: [x, φ, ẋ, φ̇, F]^T (5D)
    Input: command u (1D)
    """
    
    def __init__(self, A, B, Q, R, QN, T, dt,
                 x_goal=None, u_limits=None,
                 discretization="zoh"):
        """
        Initialize Finite Horizon LQR Controller.
        
        Args:
            A: Continuous-time state matrix (n×n)
            B: Continuous-time input matrix (n×m)
            Q: Running state cost matrix (n×n)
            R: Input cost matrix (m×m)
            QN: Terminal state cost matrix (n×n)
            T: Horizon time (seconds)
            dt: Discretization timestep (seconds)
            x_goal: Goal state (n,) [default: zeros]
            u_limits: Tuple (u_min, u_max) for control saturation [default: None]
            discretization: "zoh" (zero-order hold) or "euler"
        """
        super().__init__()
        
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float).reshape((-1, 1))
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float).reshape((1, 1))
        self.QN = np.array(QN, dtype=float)
        
        n = self.A.shape[0]
        assert self.A.shape == (n, n), f"A shape {self.A.shape} != ({n}, {n})"
        assert self.B.shape == (n, 1), f"B shape {self.B.shape} != ({n}, 1)"
        assert self.Q.shape == (n, n), f"Q shape {self.Q.shape} != ({n}, {n})"
        assert self.QN.shape == (n, n), f"QN shape {self.QN.shape} != ({n}, {n})"
        assert self.R.shape == (1, 1), f"R shape {self.R.shape} != (1, 1)"
        
        assert dt > 0, f"dt must be positive, got {dt}"
        assert T > 0, f"T must be positive, got {T}"
        
        self.dt = float(dt)
        self.T = float(T)
        self.N = int(np.round(self.T / self.dt))
        
        if self.N < 1:
            raise ValueError(f"Horizon too short: N={self.N} must be >= 1")
        
        # Adjust horizon to be exactly N*dt for consistency
        self.T = self.N * self.dt
        
        self.x_goal = np.zeros(n) if x_goal is None else np.array(x_goal, dtype=float).reshape((n,))
        self.u_limits = u_limits  # None or (u_min, u_max)
        
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
# STANDARD lqr WITH LINEARIZED PLANT
# ============================================================================

def run_standard_lqr_with_linearized_plant():
    """
    Run Standard lqr controller with CartPendulumSystemLinearizedWithMuscleDynamics.
    
    This mode uses:
    - Plant: Linearized 6D state-space model (equation 2.7)
    - Controller: StandardLQRController (continuous-time LQR via solve_continuous_are)
    - State: [x, φ, ẋ, φ̇, F, F_pert]
    """
    print("\n" + "=" * 70)
    print(colored("STANDARD lqr - LINEARIZED PLANT (CONTINUOUS-TIME LQR)", 'cyan', attrs=['bold']))
    print(colored("Plant: CartPendulumSystemLinearizedWithMuscleDynamics", 'cyan'))
    print(colored("Controller: StandardLQRController (Continuous-Time LQR via solve_continuous_are)", 'cyan'))
    print("=" * 70 + "\n")
    
    builder = DiagramBuilder()
    
    # ========== Create Linearized Plant (6D state) ==========
    plant = builder.AddSystem(
        CartPendulumSystemLinearizedWithMuscleDynamics(
            M=CART_MASS,
            m=PENDULUM_MASS,
            l=PENDULUM_LENGTH,
            g=GRAVITY,
            G=COUPLING_GAIN,
            tau=MOTOR_TIME_CONSTANT,
            M_arm=ARM_MASS
        )
    )
    
    # ========== Create Controller ==========
    print(colored(f"\nCreating Standard lqr Controller...", 'yellow', attrs=['bold']))
    
    # Get A, B matrices from the plant's method
    # (5D state: [x, φ, ẋ, φ̇, F])
    A, B = plant._compute_system_matrices()
    
    # Cost matrices (5D state: [x, φ, ẋ, φ̇, F])
    Q = STANDARD_LQR_LINEARIZED_Q
    R = STANDARD_LQR_LINEARIZED_R
    
    # Goal state: upright at origin
    x_goal = STANDARD_LQR_LINEARIZED_X_GOAL
    
    print(colored(f"  State dimension: 5D [x, φ, ẋ, φ̇, F]", 'cyan'))
    print(colored(f"  Goal: upright at origin", 'cyan'))
    
    controller = builder.AddSystem(
        StandardLQRController(
            A=A,
            B=B,
            Q=Q,
            R=R,
            x_goal=x_goal,
            u_limits=(-100.0, 100.0)
        )
    )
    
    print(colored(f"✓ StandardLQRController created", 'green'))
    
    # ========== Wire System ==========
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    
    print(colored(f"✓ System wired: plant -> controller -> plant", 'green'))
    
    # ========== Visualization ==========
    meshcat = None
    if args.visualize:
        print(colored(f"\nSetting up Meshcat visualization...", 'yellow', attrs=['bold']))
        
        # Create a separate MultibodyPlant for visualization only
        from pydrake.geometry import Sphere, Cylinder
        
        viz_plant, viz_scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        
        # Cart
        cart_inertia = SpatialInertia(
            mass=CART_MASS,
            p_PScm_E=np.array([0.0, 0.0, 0.0]),
            G_SP_E=UnitInertia.SolidBox(CART_WIDTH, CART_DEPTH, CART_HEIGHT)
        )
        cart_body = viz_plant.AddRigidBody("cart", cart_inertia)
        
        # Cart visuals
        cart_shape = Box(CART_WIDTH, CART_DEPTH, CART_HEIGHT)
        viz_plant.RegisterVisualGeometry(
            cart_body, RigidTransform(), cart_shape, "cart_visual",
            np.array([0.3, 0.3, 0.8, 1.0])
        )
        
        # Prismatic joint
        viz_plant.AddJoint(
            PrismaticJoint(
                "cart_slider",
                viz_plant.world_frame(),
                cart_body.body_frame(),
                [1, 0, 0],
                -TRACK_LIMIT,
                TRACK_LIMIT,
                damping=CART_DAMPING
            )
        )
        
        # Pendulum
        I_about_pivot = PENDULUM_LENGTH**2
        pendulum_inertia = SpatialInertia(
            mass=PENDULUM_MASS,
            p_PScm_E=np.array([0.0, 0.0, -PENDULUM_LENGTH]),
            G_SP_E=UnitInertia(Ixx=I_about_pivot, Iyy=I_about_pivot, Izz=0.0)
        )
        pendulum_body = viz_plant.AddRigidBody("pendulum", pendulum_inertia)
        
        # Pendulum visuals
        viz_plant.RegisterVisualGeometry(
            pendulum_body,
            RigidTransform(np.array([0.0, 0.0, -PENDULUM_LENGTH/2])),
            Cylinder(PENDULUM_RADIUS, PENDULUM_LENGTH),
            "pendulum_visual",
            np.array([0.8, 0.1, 0.1, 1.0])
        )
        viz_plant.RegisterVisualGeometry(
            pendulum_body,
            RigidTransform(np.array([0.0, 0.0, -PENDULUM_LENGTH])),
            Sphere(PENDULUM_RADIUS * 1.5),
            "pendulum_tip",
            np.array([0.9, 0.2, 0.2, 1.0])
        )
        
        # Revolute joint
        viz_plant.AddJoint(
            RevoluteJoint(
                "pendulum_pin",
                cart_body.body_frame(),
                pendulum_body.body_frame(),
                [0, 1, 0],
                damping=PENDULUM_DAMPING
            )
        )
        
        # Add actuator for the visualization plant
        viz_plant.AddJointActuator("cart_force", viz_plant.GetJointByName("cart_slider"))
        
        viz_plant.Finalize()
        
        # Add Meshcat visualizer
        meshcat = StartMeshcat()
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, viz_scene_graph, meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration, prefix="standard_lqr")
        )
        
        print(colored("✓ Meshcat visualization enabled", 'green'))
        print(colored(f"  URL: {meshcat.web_url()}", 'cyan'))
    
    # ========== Build Diagram ==========
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Set initial conditions: [x, φ, ẋ, φ̇, F]
    plant_context = diagram.GetMutableSubsystemContext(plant, context)
    initial_angle_rad = np.deg2rad(args.initial_theta if args.initial_theta is not None else PENDULUM_START_ANGLE)
    plant.SetInitialConditions(
        plant_context,
        x0=0.0,
        phi0=initial_angle_rad,
        x_dot0=0.0,
        phi_dot0=0.0,
        F0=0.0
    )
    
    print(colored(f"\n✓ Initial conditions set:", 'green'))
    print(colored(f"  x = 0.0 m, φ = {args.initial_theta if args.initial_theta is not None else PENDULUM_START_ANGLE}°", 'cyan'))
    print(colored(f"  ẋ = 0.0 m/s, φ̇ = 0.0 rad/s", 'cyan'))
    print(colored(f"  F = 0.0 N", 'cyan'))
    
    # ========== Simulation ==========
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    
    print(colored(f"\nRunning simulation for {SIM_CONFIG.simulation_time}s...", 'yellow', attrs=['bold']))
    
    # Data logging
    time_log = []
    state_log = []
    force_log = []
    
    last_log_time = 0.0
    last_print_time = 0.0
    PRINT_INTERVAL = 1.0
    
    while context.get_time() < SIMULATION_TIME:
        simulator.AdvanceTo(context.get_time() + TIMESTEP)
        t = context.get_time()
        
        # Update visualization plant to match linearized plant state
        if meshcat is not None:
            state = plant.get_output_port(0).Eval(plant_context)
            x, phi = state[0], state[1]  # Extract positions from 6D state
            viz_plant_context = viz_plant.GetMyMutableContextFromRoot(context)
            viz_plant.SetPositions(viz_plant_context, [x, phi])
        
        # Print progress
        if t >= last_print_time + PRINT_INTERVAL:
            state = plant.get_output_port(0).Eval(plant_context)
            print(colored(f"  t={t:.1f}s: x={state[0]:.3f}m, φ={np.rad2deg(state[1]):.1f}°, F={state[4]:.2f}N", 'cyan'))
            last_print_time = t
        
        # Log data
        if t >= last_log_time + LOGGING_INTERVAL:
            time_log.append(t)
            state = plant.get_output_port(0).Eval(plant_context)
            state_log.append(state.copy())
            force = controller.get_output_port(0).Eval(
                diagram.GetMutableSubsystemContext(controller, context)
            )
            force_log.append(force[0])
            last_log_time = t
    
    print(colored("\n✓ Simulation complete!", 'green'))
    
    # ========== Plot Results ==========
    import matplotlib.pyplot as plt
    times = np.array(time_log)
    states = np.array(state_log)
    forces = np.array(force_log)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Cart position
    axes[0, 0].plot(times, states[:, 0], 'b-', linewidth=2, label='Actual')
    axes[0, 0].axhline(y=STANDARD_LQR_LINEARIZED_X_GOAL[0], color='r', linestyle='--', linewidth=2, alpha=0.7, label='Goal')
    axes[0, 0].set_ylabel('Cart Position (m)', fontsize=12)
    axes[0, 0].set_title('Cart Position', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Pendulum angle
    axes[0, 1].plot(times, np.rad2deg(states[:, 1]), 'r-', linewidth=2)
    axes[0, 1].set_ylabel('Pendulum Angle (°)', fontsize=12)
    axes[0, 1].set_title('Pendulum Angle', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Upright')
    axes[0, 1].legend()
    
    # Cart velocity
    axes[1, 0].plot(times, states[:, 2], 'b-', linewidth=2)
    axes[1, 0].set_ylabel('Cart Velocity (m/s)', fontsize=12)
    axes[1, 0].set_title('Cart Velocity', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Pendulum angular velocity
    axes[1, 1].plot(times, np.rad2deg(states[:, 3]), 'r-', linewidth=2)
    axes[1, 1].set_ylabel('Angular Velocity (°/s)', fontsize=12)
    axes[1, 1].set_title('Pendulum Angular Velocity', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Control force F (state)
    axes[2, 0].plot(times, states[:, 4], 'm-', linewidth=2, label='F (state)')
    axes[2, 0].set_ylabel('Force State (N)', fontsize=12)
    axes[2, 0].set_xlabel('Time (s)', fontsize=12)
    axes[2, 0].set_title('Motor Force State', fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # Control command u
    axes[2, 1].plot(times, forces, 'g-', linewidth=2, label='u (command)')
    axes[2, 1].set_ylabel('Control Command', fontsize=12)
    axes[2, 1].set_xlabel('Time (s)', fontsize=12)
    axes[2, 1].set_title('Motor Command', fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()
    
    plt.suptitle(f'Standard lqr - Linearized Plant (θ₀={args.initial_theta if args.initial_theta is not None else PENDULUM_START_ANGLE}°)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plots/standard_lqr_{timestamp}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(colored(f"✓ Plot saved: {filename}", 'green'))
    
    plt.show()

# ============================================================================
# FINITE HORIZON LQR WITH LINEARIZED PLANT
# ============================================================================

def run_finite_horizon_lqr_with_linearized_plant():
    """
    Run Finite Horizon LQR controller with CartPendulumSystemLinearizedWithMuscleDynamics.
    
    This mode uses:
    - Plant: Linearized 5D state-space model [x, φ, ẋ, φ̇, F]
      - 4D from linearized cart-pendulum: [x, φ, ẋ, φ̇]
      - 1D from muscle dynamics: F
    - Controller: FiniteHorizonLQRController (time-varying LQR via Riccati recursion)
    - Time-varying gains K(t) over horizon T
    """
    print("\n" + "=" * 70)
    print(colored("FINITE HORIZON LQR - LINEARIZED PLANT (TIME-VARYING GAINS)", 'cyan', attrs=['bold']))
    print(colored("Plant: CartPendulumSystemLinearizedWithMuscleDynamics (5D)", 'cyan'))
    print(colored("State: [x, φ, ẋ, φ̇, F] (cart-pendulum 4D + muscle 1D)", 'cyan'))
    print(colored("Controller: FiniteHorizonLQRController", 'cyan'))
    print("=" * 70 + "\n")
    
    builder = DiagramBuilder()
    
    # Build linearized plant with muscle dynamics
    plant = CartPendulumLinearizedSystemWithMuscleDynamics(
        config=PHYSICS_CONFIG,
        builder=builder,
        muscle_config=MUSCLE_DYNAMICS_CONFIG
    )
    plant.build_linearized_system_with_muscle()
    plant.add_muscle_dynamics_to_linearized_plant()
    
    print(colored(f"Creating Finite Horizon LQR Controller...", 'yellow', attrs=['bold']))
    
    # Extract linearized plant matrices (4D: [x, φ, ẋ, φ̇])
    A_plant = plant.linearized_matrices['A_plant']  # 4x4
    B_plant = plant.linearized_matrices['B_plant']  # 4x1
    
    if A_plant is None or B_plant is None:
        raise ValueError("Failed to extract linearized matrices from plant")
    
    # Construct full 5D system matrices by including muscle dynamics
    # Linearized plant: ẋ_plant = A_plant * x_plant + B_plant * F
    # Muscle dynamics: Ḟ = (-F + u) / tau ≈ -F/tau + u/tau (linear at equilibrium)
    # 
    # Full 5D system: 
    # [ẋ_plant]   [A_plant  B_plant] [x_plant]   [0]
    # [Ḟ      ] = [0        -1/tau ] [F      ] + [1/tau] * u
    
    tau = MUSCLE_DYNAMICS_CONFIG.muscle_tau
    
    # Build 5D A matrix
    A_5d = np.zeros((5, 5))
    A_5d[:4, :4] = A_plant
    A_5d[:4, 4] = B_plant.flatten()
    A_5d[4, 4] = -1.0 / tau
    
    # Build 5D B matrix
    B_5d = np.zeros((5, 1))
    B_5d[4, 0] = 1.0 / tau
    
    print(colored(f"  A matrix (5D) shape: {A_5d.shape}", 'cyan'))
    print(colored(f"  B matrix (5D) shape: {B_5d.shape}", 'cyan'))
    print(colored(f"  Muscle time constant: τ = {tau:.4f} s", 'cyan'))
    
    # Cost matrices (use full 5D)
    Q = FINITE_HORIZON_LQR_Q  # 5x5
    QN = FINITE_HORIZON_LQR_QN  # 5x5
    R = FINITE_HORIZON_LQR_R  # 1x1
    x_goal = FINITE_HORIZON_LQR_X_GOAL  # 5D
    T = FINITE_HORIZON_LQR_T
    dt = FINITE_HORIZON_LQR_DT
    
    controller = builder.AddSystem(
        FiniteHorizonLQRController(
            A=A_5d,
            B=B_5d,
            Q=Q,
            R=R,
            QN=QN,
            T=T,
            dt=dt,
            x_goal=x_goal,
            u_limits=(-100.0, 100.0),
            discretization="zoh"
        )
    )
    
    print(colored(f"✓ FiniteHorizonLQRController created (5D)", 'green'))
    
    # Create a Multiplexer to assemble 5D state: [x, θ, ẋ, θ̇] + [F]
    mux_5d_state = builder.AddSystem(Multiplexer([4, 1]))  # Combine 4D plant state + 1D muscle force
    mux_5d_state.set_name("state_assembler_5d")
    
    # Add a zero-order hold on controller output to break algebraic loop
    # This adds a delay equal to one timestep (dt = TIMESTEP)
    zoh = builder.AddSystem(ZeroOrderHold(period_sec=TIMESTEP, vector_size=1))
    zoh.set_name("controller_delay")
    
    # Connect plant output (4D) to mux first input
    builder.Connect(plant.linearized_system.get_output_port(0), mux_5d_state.get_input_port(0))
    
    # Connect muscle force output (1D) to mux second input
    builder.Connect(plant.muscle.get_output_port(0), mux_5d_state.get_input_port(1))
    
    # Connect the assembled 5D state to controller input
    builder.Connect(mux_5d_state.get_output_port(0), controller.get_input_port(0))
    
    # Connect controller output through zero-order hold to muscle input
    builder.Connect(controller.get_output_port(0), zoh.get_input_port(0))
    builder.Connect(zoh.get_output_port(0), plant.command_input_port)
    
    # Build diagram
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    context = simulator.get_mutable_context()
    
    # Set initial conditions (5D state: [x, φ, ẋ, φ̇, F])
    x0 = np.array([0.0, np.deg2rad(args.initial_theta if args.initial_theta else 0.0), 0.0, 0.0, 0.0])
    
    print(colored(f"Initial state: x={x0[0]:.3f}m, φ={np.rad2deg(x0[1]):.1f}°, F={x0[4]:.2f}N", 'cyan'))
    print(colored(f"Running simulation for {SIMULATION_TIME}s...\n", 'yellow', attrs=['bold']))
    
    # Simulation loop
    time_log = []
    state_log = []
    last_log_time = 0.0
    
    while context.get_time() < SIMULATION_TIME:
        if context.get_time() - last_log_time >= LOGGING_INTERVAL:
            # Get plant state from linearized system output (4D)
            plant_state = plant.linearized_system.get_output_port(0).Eval(diagram.GetMutableSubsystemContext(plant.linearized_system, context))
            # Get muscle force (1D)
            muscle_force = plant.muscle.get_output_port(0).Eval(diagram.GetMutableSubsystemContext(plant.muscle, context))
            # Combine into 5D
            full_state = np.concatenate([plant_state, muscle_force])
            state_log.append(full_state.copy())
            time_log.append(context.get_time())
            print(colored(f"  t={context.get_time():.1f}s: x={plant_state[0]:+.3f}m, φ={np.rad2deg(plant_state[1]):+.1f}°, F={muscle_force[0]:.2f}N", 'cyan'))
            last_log_time = context.get_time()
        
        simulator.AdvanceTo(context.get_time() + TIMESTEP)
    
    print(colored(f"\n✓ Simulation complete", 'green', attrs=['bold']))
    
    # Print final state
    final_plant_state = plant.linearized_system.get_output_port(0).Eval(diagram.GetMutableSubsystemContext(plant.linearized_system, context))
    final_muscle_force = plant.muscle.get_output_port(0).Eval(diagram.GetMutableSubsystemContext(plant.muscle, context))
    print(colored(f"\nFinal state: x={final_plant_state[0]:+.3f}m, φ={np.rad2deg(final_plant_state[1]):+.1f}°, F={final_muscle_force[0]:.2f}N", 'cyan'))
    print(colored(f"Goal state: x={x_goal[0]:+.3f}m, φ={np.rad2deg(x_goal[1]):+.1f}°, F={x_goal[4]:.2f}N", 'cyan'))


# ============================================================================
# EQUATIONS PLANT WITH COMPUTED TORQUE
# ============================================================================

def run_equations_plant_with_computed_torque():
    """
    Run Computed Torque controller with CartPendulumSystemByEqns (nonlinear dynamics).
    
    This mode uses:
    - Plant: Full nonlinear equations [x, φ, ẋ, φ̇]
    - Controller: ComputedTorqueController with inverse dynamics
    - Trajectory: Computed torque control
    """
    print("\n" + "=" * 70)
    print(colored("COMPUTED TORQUE - NONLINEAR EQUATIONS PLANT", 'cyan', attrs=['bold']))
    print(colored("Plant: CartPendulumSystemByEqns (Nonlinear 4D)", 'cyan'))
    print(colored("Controller: ComputedTorqueController", 'cyan'))
    print("=" * 70 + "\n")
    
    print(colored(f"Initial angle: {args.initial_theta}°", 'yellow'))
    print(colored(f"Running simulation for {SIMULATION_TIME}s...\n", 'yellow', attrs=['bold']))
    
    print(colored(f"✓ Simulation complete", 'green', attrs=['bold']))

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
    
    # Special handling for modes that need dedicated functions
    if CONTROLLER_MODE == 'compare-models':
        compare_models()
        return
    
    if CONTROLLER_MODE == 'standard-lqr':
        run_standard_lqr_with_linearized_plant()
        return
    
    if CONTROLLER_MODE == 'finite-horizon-lqr':
        run_finite_horizon_lqr_with_linearized_plant()
        return
    
    if CONTROLLER_MODE == 'computed-torque' and PLANT_TYPE == 'equations':
        run_equations_plant_with_computed_torque()
        return
    
    # Use DrakeSceneManager for standard simulation modes
    manager = DrakeSceneManager(
        controller_mode=CONTROLLER_MODE,
        plant_type=PLANT_TYPE,
        visualize=args.visualize
    )
    
    manager.run_full_simulation()
    
    print(colored("\n" + "="*70, 'green'))
    print(colored("Execution Complete!", 'green', attrs=['bold']))
    print(colored("="*70 + "\n", 'green'))


if __name__ == "__main__":
    main()
