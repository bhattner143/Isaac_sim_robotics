"""
Cup Manipulator + 3D Pendulum System - OFC Controller Architecture

═══════════════════════════════════════════════════════════════════════════════
SYSTEM OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

2-DOF Cup Manipulator with 3D Pendulum (gimbal-mounted ball)

COMPLETE SYSTEM:
• Manipulator: 2 DOF (q₁, q₂) - planar arm
• Pendulum: 2 DOF (pitch α, roll β) - ball on gimbal mount
• Total Plant: 4 DOF → 8D state [q₁, q₂, q̇₁, q̇₂, α, β, α̇, β̇]
• Control: Muscle dynamics + ZFT + Impedance (6D internal states)
• Full System: 14D state

CRITICAL IMPLEMENTATION NOTE:
═══════════════════════════════════════════════════════════════════════════════
⚠️ CURRENT STATUS: Pendulum states NOT YET in linearized system!

Current Implementation (INCOMPLETE):
- Plant: 4D (manipulator only) ❌ Missing pendulum!
- Muscle: 2D ✓
- ZFT: 4D ✓
- Total: 10D (should be 14D)

Required Fix:
- Plant: 8D (manipulator + pendulum) ← Need to linearize FULL coupled system
- Muscle: 2D
- ZFT: 4D
- Total: 14D

WHY THIS MATTERS:
- Pendulum dynamics affect manipulator via inertial coupling
- Manipulator motion affects pendulum via pivot connection
- F_imp applied to pendulum creates reaction forces on manipulator
- Cannot treat manipulator and pendulum as separate systems!

ANALOGY:
- Cart-Pendulum: [x, θ, ẋ, θ̇] (4D) - cart + pendulum as ONE coupled plant ✓
- Cup-Manipulator: [q₁, q₂, q̇₁, q̇₂, α, β, α̇, β̇] (8D) - manipulator + pendulum as ONE coupled plant ← TODO

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import dataclasses
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
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
    InverseDynamicsController,
    ConstantVectorSource,
    ConstantVectorSource,
    
    # Mathematical utilities
    Quaternion,
    RotationMatrix,
    RollPitchYaw,
    RigidTransform,

    # Trajectory Optimization
    DirectCollocation,
    PiecewisePolynomial,

    # Optimization
    Solve,
    BoundingBoxConstraint,
    LinearEqualityConstraint,


    # Frames
    FixedOffsetFrame,

    
)

# Custom robot types
from configs.robot.robot_types import (
    ManipulatorConfig,
    SimulationConfig,
    VisualizationConfig,
    CartPendulumConfig,
    create_cup_manipulator_config,
    create_cart_pendulum_config,
)

# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Cart-Pendulum with Drake Controllers')
parser.add_argument('--mode', type=str, 
                    choices=['computed-torque', 'finite-horizon-lqr-for-min-effort', 'scene-viz', 'manipulator-pushes-cart'],
                    default='manipulator-pushes-cart', 
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
# Manipulator-pushes-cart mode arguments
parser.add_argument('--distance', type=float, default=0.5,
                    help='Cart travel distance in X direction [m] (for manipulator-pushes-cart mode)')
parser.add_argument('--duration', type=float, default=5.0,
                    help='Simulation duration [s] (for manipulator-pushes-cart mode)')
parser.add_argument('--k_imp', type=float, default=100.0,
                    help='Impedance stiffness [N/m] (for manipulator-pushes-cart mode)')
parser.add_argument('--d_imp', type=float, default=20.0,
                    help='Impedance damping [N·s/m] (for manipulator-pushes-cart mode)')
args, _ = parser.parse_known_args()

# Skip interactive input when plotting diagram
if hasattr(args, 'plot_diagram') and args.plot_diagram:
    if args.initial_theta is None:
        args.initial_theta = 0.0  # Default for diagram mode
# Set default value when running as main script (non-interactive mode for cup manipulator demo)
elif args.initial_theta is None:
    args.initial_theta = 0.0  # Default for cup manipulator demo
else:
    print(colored(f"\n✓ Using command-line angle: θ = {args.initial_theta}°\n", 'green'))

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class MuscleDynamicsConfig:
    """Parameters for muscle/actuator dynamics (first-order system) - 2D version.
    
    For 2-DOF manipulator: F = [F₁, F₂]
    Dynamics: Ḟ = (u - F) / τ
    """
    # Muscle actuation dynamics: F_dot = (-F + u) / tau
    muscle_tau: np.ndarray = field(default_factory=lambda: np.array([0.03, 0.03]))  # s (time constant for each DOF)
    muscle_initial_force: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))  # N (initial force state [F₁, F₂])
    command_limit: np.ndarray | None = None  # N (optional saturation on command input [u₁_max, u₂_max])

@dataclass
class ImpedanceForceConfig:
    """
    Parameters for impedance force law (2D version):
        F_imp = K_p·(y_ref - y) + K_d·(v_ref - v)

    For 2-DOF manipulator:
    - y, v: end-effector position and velocity [2D]
    - y_ref, v_ref: reference position and velocity from ZFT [2D]
    - K_p, K_d: 2×2 diagonal stiffness and damping matrices
    """
    kp: np.ndarray = field(default_factory=lambda: np.diag([50.0, 50.0]))   # 2×2 diagonal stiffness matrix (N/m)
    kd: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0]))   # 2×2 diagonal damping matrix (N·s/m)
    force_limit: np.ndarray | None = None  # optional saturation on F_imp [F₁_max, F₂_max]

@dataclass
class ZFTReferenceMassConfig:
    """
    ZFT / reference-mass dynamics (2D version):

      ẏref = vref
      v̇ref = M_h⁻¹ · ( K_p·(y - yref) + K_d·(v - vref) + F )

    For 2-DOF manipulator:
    Inputs:
      - y_v : [y₁, y₂, v₁, v₂]     (4) from plant (end-effector pos/vel)
      - F   : [F₁, F₂]             (2) muscle forces

    Output:
      - yref_vref : [yref₁, yref₂, vref₁, vref₂] (4)

    State:
      - [yref₁, yref₂, vref₁, vref₂] (4D)
    """
    Mh: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0]))  # 2×2 diagonal reference mass matrix (kg)
    kp: np.ndarray = field(default_factory=lambda: np.diag([50.0, 50.0]))  # 2×2 diagonal stiffness matrix (N/m)
    kd: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0]))  # 2×2 diagonal damping matrix (N·s/m)
    yref0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))  # Initial reference position [yref₁, yref₂]
    vref0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))  # Initial reference velocity [vref₁, vref₂]

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
class ManipulatorPushesCartConfig:
    """Configuration for manipulator-pushes-cart simulation.
    
    Architecture: Manipulator → Impedance Force → Cart-Pendulum
    - Manipulator: 2-DOF (active with impedance control)
    - Cart-Pendulum: 4-DOF (passive, receives F_imp force)
    - Control: M_ref → x_ref → F_imp → Cart (NO virtual mass!)
    """
    # Cart-Pendulum parameters
    cart_mass: float = 5.0  # kg
    cart_damping: float = 0.1  # N·s/m
    pendulum_mass: float = 0.5  # kg
    pendulum_length: float = 0.2  # m
    pendulum_damping: float = 0.1  # N·m·s/rad
    
    # Impedance control parameters
    K_imp: float = 100.0  # N/m (stiffness, 1D for X-axis motion)
    D_imp: float = 20.0  # N·s/m (damping, 1D for X-axis motion)
    M_ref: float = 2.0  # kg (reference mass for ZFT)
    
    # Desired motion
    distance: float = 0.5  # m (desired cart travel in X direction)
    duration: float = 5.0  # s (simulation time)
    
    # Initial configuration
    q1_init: float = -10.0  # deg (manipulator joint 1)
    q2_init: float = 20.0  # deg (manipulator joint 2)
    initial_pitch: float = 0.0  # rad (pendulum pitch angle α)
    initial_roll: float = 0.0  # rad (pendulum roll angle β)


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

def create_impedance_force_config(
    kp: np.ndarray = None,
    kd: np.ndarray = None,
    force_limit: np.ndarray | None = None,
) -> ImpedanceForceConfig:
    """
    Create an ImpedanceForceConfig with custom parameters (2D version).

    Args:
        kp: 2×2 stiffness matrix (use np.diag([kp1, kp2]) for diagonal)
        kd: 2×2 damping matrix (use np.diag([kd1, kd2]) for diagonal)
        force_limit: if set, clamp output force to ±force_limit [F₁_max, F₂_max] (1D array)

    Returns:
        ImpedanceForceConfig instance
    """
    if kp is None:
        kp = np.diag([50.0, 50.0])
    if kd is None:
        kd = np.diag([10.0, 10.0])
    
    return ImpedanceForceConfig(kp=kp, kd=kd, force_limit=force_limit)

def create_muscle_dynamics_config(
    muscle_tau: np.ndarray = None,
    muscle_initial_force: np.ndarray = None,
    command_limit: np.ndarray | None = None,
) -> MuscleDynamicsConfig:
    """
    Create a MuscleDynamicsConfig with custom parameters (2D version).
    
    Args:
        muscle_tau: Muscle time constant [s] for each DOF (2D array)
        muscle_initial_force: Initial force state [N] for each DOF (2D array)
        command_limit: Optional saturation limit on command input [N] (2D array)
    
    Returns:
        MuscleDynamicsConfig instance
    """
    if muscle_tau is None:
        muscle_tau = np.array([0.03, 0.03])
    if muscle_initial_force is None:
        muscle_initial_force = np.array([0.0, 0.0])
    
    return MuscleDynamicsConfig(
        muscle_tau=muscle_tau,
        muscle_initial_force=muscle_initial_force,
        command_limit=command_limit,
    )

def create_zft_reference_mass_config(
    Mh: np.ndarray = None,
    kp: np.ndarray = None,
    kd: np.ndarray = None,
    yref0: np.ndarray = None,
    vref0: np.ndarray = None,
) -> ZFTReferenceMassConfig:
    """Factory helper for 2D ZFT reference mass configuration.
    
    Args:
        Mh: 2×2 reference mass matrix (use np.diag([M1, M2]) for diagonal)
        kp: 2×2 stiffness matrix (use np.diag([kp1, kp2]) for diagonal)
        kd: 2×2 damping matrix (use np.diag([kd1, kd2]) for diagonal)
        yref0: Initial reference position [yref₁, yref₂]
        vref0: Initial reference velocity [vref₁, vref₂]
    """
    if Mh is None:
        Mh = np.diag([1.0, 1.0])
    if kp is None:
        kp = np.diag([50.0, 50.0])
    if kd is None:
        kd = np.diag([10.0, 10.0])
    if yref0 is None:
        yref0 = np.array([0.0, 0.0])
    if vref0 is None:
        vref0 = np.array([0.0, 0.0])
    
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
# --- Cup Manipulator Configuration ---
MANIPULATOR_CONFIG = create_cup_manipulator_config(
    urdf_path=str(Path("model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf").absolute()),
    joint_angles=(0.0, 0.0),
    damping=(0.0, 0.0),
    stiffness=(0.0, 0.0),
    friction=(0.05, 0.05),
)

# Cart-Pendulum configuration (for testing/validation purposes)
# Cart position matches cup manipulator's cup center pivot position in URDF
# Cup center at home position: x=-2.2075m, y=0, z=1.10625m
CART_PENDULUM_CONFIG = create_cart_pendulum_config(
    cart_mass=0.3,
    cart_size=0.1,
    cart_damping=0.0,
    pendulum_mass=0.5,
    pendulum_length=0.2,
    pendulum_radius=0.05,
    pendulum_damping=0.0,
    attachment_offset=(0.0, 0.0, 0.0),
    initial_cart_x=-2.2075,  # Matches cup center X position (link1=-0.953 + link2=-1.2545)
    initial_cart_y=0.0,
    initial_pitch=0.0,
    initial_roll=0.0,
    name="cart_pendulum"
)

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

# Physics parameters (cup manipulator specific)
# Note: Cup manipulator uses ManipulatorConfig, not PhysicsConfig
# CART_MASS = PHYSICS_CONFIG.mass_cart  # Not applicable for cup manipulator
# PENDULUM_MASS = PHYSICS_CONFIG.mass_pendulum  # Not applicable
# PENDULUM_LENGTH = PHYSICS_CONFIG.length_pendulum  # Not applicable

# Simulation parameters
TIMESTEP = SIM_CONFIG.timestep
SIMULATION_TIME = SIM_CONFIG.simulation_time
LOGGING_INTERVAL = SIM_CONFIG.logging_interval
GRAVITY = -9.81  # Standard gravity (m/s²)

# ============================================================================
# ROBOT BASE CLASS (ABSTRACT)
# ============================================================================

class RobotBase(ABC):
    """
    Abstract Base Class for Robots using Drake
    
    DESIGN PATTERN: Template Method Pattern
    Provides common interface for all robots
    """
    
    def __init__(self, config: ManipulatorConfig, name: Optional[str] = None, enable_visualization: bool = True):
        """Initialize robot with configuration.
        
        Args:
            config: Manipulator configuration
            name: Robot name (optional, defaults to config.name)
            enable_visualization: If True, initialize Meshcat visualization
        """
        self.config = config
        self.name = name or config.name
        self.model_instance: Optional[int] = None
        self.dof_names: List[str] = []
        
        # Visualization setup (can be disabled by child classes)
        self.enable_visualization = enable_visualization
        self.meshcat = None
        self.visualizer_params = None
        
        if self.enable_visualization:
            self.meshcat = StartMeshcat()
            self.visualizer_params = MeshcatVisualizerParams()
            self.visualizer_params.show_hydroelastic = True
            self.visualizer_params.show_contact_forces = True
    
    def load_urdf_to_plant(self, plant: MultibodyPlant, parser: Parser) -> int:
        """
        Load URDF to plant using Drake's URDF parser.
        
        Args:
            plant: Drake MultibodyPlant
            parser: Drake URDF parser
            
        Returns:
            model_instance: Drake's model instance ID
        """
        urdf_path = str(self.config.get_urdf_path())
        print(f"\nLoading robot from URDF: {urdf_path}")
        
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        # Set package map for mesh loading
        for package_name, package_path in self.config.package_map.items():
            parser.package_map().Add(package_name, package_path)
        
        # AddModels returns a list of model instances
        model_instances = parser.AddModels(urdf_path)
        if not model_instances:
            raise RuntimeError(f"Failed to load URDF from {urdf_path}")
        
        print(colored(f"✓ Loaded {len(model_instances)} model instance(s) from URDF", 'green'))
        for idx, instance in enumerate(model_instances):
            print(colored(f"  [{idx}] Model instance: {instance}", 'cyan'))
        
        model_instance = model_instances[0]
        self.model_instance = model_instance
        print(colored(f"✓ Robot '{self.name}' using model instance: {model_instance}", 'green'))
        return model_instance
    
    def initialize_state(self, plant: MultibodyPlant):
        """Initialize robot state after plant is finalized."""
        if not self.model_instance:
            raise RuntimeError("Model not loaded - call load_urdf_to_plant first")
        
        # Get DOF names (only actuated joints)
        self.dof_names = []
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0 and joint.num_positions() > 0:
                self.dof_names.append(joint.name())
        
        num_dof = len(self.dof_names)
        print(colored(f"✓ Robot '{self.name}' initialized with {num_dof} DOFs", 'green', attrs=['bold']))
        print(colored(f"  DOF names: {self.dof_names}", 'cyan'))
    
    def set_joint_properties(self, plant: MultibodyPlant):
        """Set joint properties (damping, stiffness, friction) BEFORE plant is finalized.
        
        NOTE: These properties only affect passive dynamics when NO torque is commanded!
        - Damping: Opposes joint velocity (τ_damping = -b·q̇)
        - Stiffness: Opposes displacement from zero position (τ_spring = -k·q)
        - When using computed torque control, these are typically overridden by commanded torques.
        """
        print(colored(f"\nSetting joint properties for '{self.name}':", 'yellow'))
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_mutable_joint(joint_idx)
            joint_name = joint.name()
            
            if joint.num_velocities() > 0 and joint_name in self.config.joint_configs:
                config = self.config.joint_configs[joint_name]
                
                # Set damping
                if hasattr(joint, 'set_default_damping_vector') and config.damping > 0:
                    joint.set_default_damping_vector([config.damping])
                    print(colored(f"  ✓ {joint_name}: damping={config.damping}", 'cyan'))
                else:
                    print(colored(f"  ✓ {joint_name}: damping=0.0 (default)", 'cyan'))
                
                # Set stiffness (if supported)
                if hasattr(joint, 'set_default_stiffness_vector') and config.stiffness > 0:
                    joint.set_default_stiffness_vector([config.stiffness])
                    print(colored(f"  ✓ {joint_name}: stiffness={config.stiffness}", 'cyan'))
                elif config.stiffness > 0:
                    print(colored(f"  ⚠ {joint_name}: stiffness={config.stiffness} (not supported by joint type)", 'yellow'))
                    
        print(colored(f"✓ Joint properties configured", 'green'))
        print(colored(f"  NOTE: Damping/stiffness only affect passive dynamics (not active when commanding torques)", 'yellow'))
    
    def set_initial_positions(self, plant: MultibodyPlant, context):
        """Set initial joint positions from configuration."""
        print(colored(f"\nSetting initial positions for '{self.name}':", 'yellow'))
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_mutable_joint(joint_idx)
            joint_name = joint.name()
            
            if joint_name in self.config.joint_configs:
                position = self.config.joint_configs[joint_name].position
                
                if isinstance(joint, RevoluteJoint):
                    joint.set_angle(context, position)
                    print(colored(f"  ✓ {joint_name}: {np.rad2deg(position):.2f}° ({position:.4f} rad)", 'cyan'))
                elif isinstance(joint, PrismaticJoint):
                    joint.set_translation(context, position)
                    print(colored(f"  ✓ {joint_name}: {position:.4f} m", 'cyan'))
        print(colored(f"✓ Initial positions set", 'green'))
    
    def get_joint_positions(self, plant: MultibodyPlant, context) -> Dict[str, float]:
        """Get current joint positions as a dictionary."""
        positions = {}
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0:
                if isinstance(joint, RevoluteJoint):
                    positions[joint.name()] = joint.get_angle(context)
                elif isinstance(joint, PrismaticJoint):
                    positions[joint.name()] = joint.get_translation(context)
        return positions
    
    def set_joint_angles(self, plant: MultibodyPlant, context, joint_angles: dict):
        """
        Set joint angles.
        
        Args:
            plant: Drake MultibodyPlant
            context: Mutable context
            joint_angles: Dictionary of {joint_name: angle_in_radians}
        """
        for joint_name, angle in joint_angles.items():
            try:
                joint = plant.GetJointByName(joint_name, self.model_instance)
                
                if isinstance(joint, RevoluteJoint):
                    joint.set_angle(context, angle)
                elif isinstance(joint, PrismaticJoint):
                    joint.set_translation(context, angle)
            except Exception as e:
                print(colored(f"Warning: Could not set joint {joint_name}: {e}", "yellow"))
    
    def setup_visualization_in_builder(self, builder, scene_graph):
        """Add Meshcat visualization to the diagram builder.
        
        Args:
            builder: DiagramBuilder instance
            scene_graph: SceneGraph instance
            
        Returns:
            bool: True if visualization was set up, False otherwise
        """
        if self.enable_visualization and self.meshcat and self.visualizer_params:
            MeshcatVisualizer.AddToBuilder(
                builder, scene_graph, self.meshcat, self.visualizer_params
            )
            return True
        return False
    
    def setup_diagram_and_simulator(self, builder, plant, scene_graph, num_actuators, add_zero_torque=True):
        """Setup visualization, build diagram, and create simulator.
        
        This method encapsulates:
        - Meshcat visualization setup (Step 5)
        - Zero-torque controller addition (optional)
        - Diagram building (Step 6)
        - Simulator creation
        
        Args:
            builder: DiagramBuilder instance
            plant: MultibodyPlant instance
            scene_graph: SceneGraph instance for visualization
            num_actuators: Number of actuators in the plant
            add_zero_torque: If True, add zero-torque controller (default True)
            
        Returns:
            tuple: (diagram, simulator) - Built diagram and simulator instances
        """
        from pydrake.all import Simulator
        from pydrake.systems.primitives import ConstantVectorSource
        
        # Step 5: Setup Meshcat visualization (if enabled)
        if self.enable_visualization:
            print(colored("\n[5/6] Setting up Meshcat visualization...", "cyan"))
            
            # Use setup_visualization_in_builder to add visualization
            viz_added = self.setup_visualization_in_builder(builder, scene_graph)
            
            if viz_added:
                print(colored(f"  ✓ Meshcat visualization started", "green"))
                print(colored(f"    URL: {self.get_meshcat_url()}", "cyan", attrs=["bold"]))
                print(colored(f"    Hydroelastic: {self.visualizer_params.show_hydroelastic}", "cyan"))
                print(colored(f"    Contact forces: {self.visualizer_params.show_contact_forces}", "cyan"))
                print(colored(f"\n  👉 Open the URL above in your browser to view the robot", "yellow", attrs=["bold"]))
            else:
                print(colored(f"  ⚠ Visualization setup failed", "yellow"))
        else:
            print(colored("\n[5/6] Skipping visualization (disabled)", "yellow"))
        
        # Add zero-torque controller if requested
        if add_zero_torque and num_actuators > 0:
            zero_torque = builder.AddSystem(ConstantVectorSource(np.zeros(num_actuators)))
            zero_torque.set_name("ZeroTorqueController")
            builder.Connect(
                zero_torque.get_output_port(0),
                plant.get_actuation_input_port()
            )
            print(colored(f"  ✓ Zero-torque controller added ({num_actuators} actuators)", "green"))
        
        # Step 6: Build diagram and create simulator
        print(colored("\n[6/6] Building diagram and creating simulator...", "cyan"))
        diagram = builder.Build()
        simulator = Simulator(diagram)
        simulator.set_target_realtime_rate(1.0)
        
        print(colored("  ✓ Diagram built", "green"))
        print(colored("  ✓ Simulator created", "green"))
        
        return diagram, simulator
    
    def publish_visualization(self, diagram, context):
        """Publish current state to Meshcat visualization.
        
        Args:
            diagram: Drake diagram
            context: Diagram context
        """
        if self.meshcat:
            diagram.ForcedPublish(context)
    
    def get_meshcat_url(self) -> Optional[str]:
        """Get Meshcat visualization URL if available.
        
        Returns:
            Meshcat URL string or None if visualization not enabled
        """
        if self.meshcat:
            return self.meshcat.web_url()
        return None
    
    def run_simulation(self, duration: float = 5.0):
        """Run simulation for specified duration.
        
        Args:
            duration: Simulation time in seconds
        """
        print(colored("\n" + "=" * 80, "yellow"))
        print(colored("RUNNING SIMULATION", "yellow", attrs=["bold"]))
        print(colored("=" * 80, "yellow"))
        
        print(colored(f"\nSimulating for {duration} seconds...", "cyan"))
        
        try:
            self.simulator.AdvanceTo(duration)
            print(colored(f"✓ Simulation completed successfully!", "green", attrs=["bold"]))
        except Exception as e:
            print(colored(f"✗ Simulation error: {e}", "red"))
            raise
        
        print(colored("\n" + "=" * 80, "green"))
        print(colored("SIMULATION COMPLETE", "green", attrs=["bold"]))
        print(colored("=" * 80, "green"))
        
        if self.meshcat:
            print(colored(f"\n🌐 Meshcat still running at: {self.meshcat.web_url()}", "cyan", attrs=["bold"]))
            print(colored("   Keep this window open to view the visualization", "yellow"))
    
    def run_simulation(self, duration: float = 5.0):
        """Run simulation for specified duration.
        
        Args:
            duration: Simulation time in seconds
        """
        print(colored("\n" + "=" * 80, "yellow"))
        print(colored("RUNNING SIMULATION", "yellow", attrs=["bold"]))
        print(colored("=" * 80, "yellow"))
        
        print(colored(f"\nSimulating for {duration} seconds...", "cyan"))
        
        try:
            self.simulator.AdvanceTo(duration)
            print(colored(f"✓ Simulation completed successfully!", "green", attrs=["bold"]))
        except Exception as e:
            print(colored(f"✗ Simulation error: {e}", "red"))
            raise
        
        print(colored("\n" + "=" * 80, "green"))
        print(colored("SIMULATION COMPLETE", "green", attrs=["bold"]))
        print(colored("=" * 80, "green"))
        
        if self.meshcat:
            print(colored(f"\n🌐 Meshcat still running at: {self.meshcat.web_url()}", "cyan", attrs=["bold"]))
            print(colored("   Keep this window open to view the visualization", "yellow"))


# ============================================================================
# CUP MANIPULATOR CLASS
# ============================================================================

class CupManipulator(RobotBase):
    """
    Cup Manipulator for Drake with controller integration.
    
    Manages:
    - URDF loading and joint configuration
    - State queries (positions, velocities)
    - End-effector kinematics
    - Base welding and body access
    """
    
    # End-effector offset from link2 frame to simple_ball (from URDF)
    # This represents the center of the cup where the ball sits
    EE_OFFSET = np.array([-1.2545, 0.0, -0.188125])
    
    def __init__(self, config: ManipulatorConfig, enable_visualization: bool = False):
        """Initialize CupManipulator.
        
        Args:
            config: Manipulator configuration
            enable_visualization: If True, initialize Meshcat (default False for component usage)
        """
        super().__init__(config, enable_visualization=enable_visualization)
    
    def weld_base_to_world(self, plant: MultibodyPlant):
        """
        Weld the manipulator base to the world frame (ground).
        
        Args:
            plant: MultibodyPlant instance (must be called before Finalize)
        """
        base_frame = plant.GetFrameByName("base_mount_manipulator", self.model_instance)
        plant.WeldFrames(plant.world_frame(), base_frame)
    
    def get_body_by_name(self, plant: MultibodyPlant, body_name: str):
        """
        Get a body from the manipulator by name.
        
        Args:
            plant: MultibodyPlant instance
            body_name: Name of the body (e.g., 'link2')
        
        Returns:
            Body instance
        """
        return plant.GetBodyByName(body_name, self.model_instance)
    
    def setup_in_plant(self, plant: MultibodyPlant, parser: Parser, weld_base: bool = True):
        """
        Complete setup of cup manipulator in plant.
        
        This is a convenience method that:
        1. Loads URDF
        2. Optionally welds base to world
        
        Args:
            plant: MultibodyPlant instance (must be called before Finalize)
            parser: Parser instance for URDF loading
            weld_base: If True, weld base to world frame
        """
        # Load URDF
        self.load_urdf_to_plant(plant, parser)
        
        # Weld base if requested
        if weld_base:
            self.weld_base_to_world(plant)
    
    def CalcPosition(self, plant: MultibodyPlant, context, 
                     point_offset: np.ndarray = None, 
                     body_name: str = "link2",
                     expressed_in_frame=None) -> np.ndarray:
        """
        Custom position calculation method that wraps Drake's CalcPointsPositions.
        
        This method provides a convenient interface with the built-in EE_OFFSET,
        so you don't need to specify it every time.
        
        Args:
            plant: MultibodyPlant instance
            context: Context for the plant
            point_offset: Point offset from body frame (defaults to EE_OFFSET for simple_ball)
            body_name: Name of body to compute position from (defaults to "link2")
            expressed_in_frame: Frame to express position in (defaults to world frame)
        
        Returns:
            3D position vector as numpy array
        
        Example:
            # Get end effector (simple_ball) position
            ee_pos = manipulator.CalcPosition(plant, context)
            
            # Get link2 origin position
            link2_pos = manipulator.CalcPosition(plant, context, point_offset=np.zeros(3))
            
            # Get custom point on link2
            custom_pos = manipulator.CalcPosition(plant, context, point_offset=np.array([1.0, 0.0, 0.0]))
        """
        # Use EE_OFFSET by default (simple_ball location)
        if point_offset is None:
            point_offset = self.EE_OFFSET
        
        # Use world frame by default
        if expressed_in_frame is None:
            expressed_in_frame = plant.world_frame()
        
        try:
            body = plant.GetBodyByName(body_name, self.model_instance)
            
            # Use Drake's CalcPointsPositions
            position = plant.CalcPointsPositions(
                context,
                body.body_frame(),
                point_offset,
                expressed_in_frame
            ).flatten()
            
            return position
        except Exception as e:
            print(f"Warning: Could not calculate position: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def get_end_effector_position(self, plant: MultibodyPlant, context) -> np.ndarray:
        """Get end effector (cup center/simple_ball) position in world frame.
        
        Uses the EE_OFFSET to get the position of the simple_ball geometry
        which represents the center of the cup where the ball sits.
        
        Note: This is a convenience wrapper around CalcPosition with default args.
        """
        return self.CalcPosition(plant, context)
        
    def build_in_plant(self, plant: MultibodyPlant, parser: Parser, weld_base: bool = True):
        """
        Complete build of cup manipulator in plant including all setup.
        
        This method handles:
        1. URDF loading
        2. Base welding (optional)
        3. Adding actuators
        4. Setting joint properties
        
        Args:
            plant: MultibodyPlant instance (must be called before Finalize)
            parser: Parser instance for URDF loading
            weld_base: If True, weld base to world frame
        """
        # Step 1: Load URDF and weld base
        self.setup_in_plant(plant, parser, weld_base=weld_base)
        
        # Step 2: Add actuators (must be before Finalize)
        for joint_name in ["link1_base", "link2_link1"]:
            joint = plant.GetJointByName(joint_name, self.model_instance)
            plant.AddJointActuator(joint_name, joint)
        
        # Step 3: Set joint properties (damping, friction, stiffness)
        # Note: This can be called before or after Finalize
        self.set_joint_properties(plant)

    





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
    ):
        """
        Initialize cart-pendulum system.
        
        Args:
            config: CartPendulumConfig with all system parameters
            visualize_cart: If True, add visual geometry to cart; if False, cart is invisible
            add_cart_actuators: If True, add actuators to cart joints (active); if False, cart is passive
        """
        self.config = config
        self.visualize_cart = visualize_cart
        self.add_cart_actuators = add_cart_actuators
        
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
        # X-axis joint
        self.x_joint = plant.AddJoint(
            PrismaticJoint(
                name=f"{self.config.name}_x",
                frame_on_parent=plant.world_frame(),
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
# MUSCLE DYNAMICS CLASS (2D VERSION FOR CUP MANIPULATOR)
# ============================================================================
class MuscleDynamics(LeafSystem):
    """
    2D Muscle/Actuator dynamics: First-order low-pass filter for each actuator.
    
    Dynamics: τ_m * Ḟ = u - F  (element-wise for each DOF)
    
    State: F ∈ ℝ² (muscle force vector)
    Input: u ∈ ℝ² (neural/motor command vector)
    Output: F ∈ ℝ² (actual muscle force vector)
    
    This models the delay between motor command and actual force output.
    """
    
    def __init__(self, config: MuscleDynamicsConfig):
        LeafSystem.__init__(self)
        self.config = config
        
        # Input: command u (2D)
        self.DeclareVectorInputPort("u", BasicVector(2))
        
        # Continuous state: F (2D)
        self.DeclareContinuousState(2)
        
        # Output: F (2D)
        self.DeclareVectorOutputPort("F", BasicVector(2), self.CalcOutput)
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        """Compute Ḟ = (u - F) / τ_m"""
        u = self.get_input_port(0).Eval(context)  # (2,)
        F = context.get_continuous_state_vector().CopyToVector()  # (2,)
        
        # First-order dynamics: Ḟ = (u - F) / τ
        F_dot = (u - F) / self.config.muscle_tau
        
        derivatives.get_mutable_vector().SetFromVector(F_dot)
    
    def CalcOutput(self, context, output):
        """Output current muscle force F"""
        F = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(F)


# ============================================================================
# IMPEDANCE FORCE (2D VERSION FOR CUP MANIPULATOR)
# ============================================================================
class ImpedanceForce(LeafSystem):
    """
    2D Impedance force law (in task/Cartesian space):
    
    F_imp = K_p · (y_ref - y) + K_d · (ẏ_ref - ẏ)
    
    where:
    - y, ẏ: actual end-effector position/velocity (2D)
    - y_ref, ẏ_ref: reference position/velocity from ZFT (2D)
    - K_p, K_d: Stiffness and damping matrices (2x2) or scalars
    
    Inputs:
      - y_v: [y, ẏ] (4D: [y₁, y₂, ẏ₁, ẏ₂])
      - yref_vref: [y_ref, ẏ_ref] (4D: [y_ref₁, y_ref₂, ẏ_ref₁, ẏ_ref₂])
    
    Output:
      - F_imp: (2D: [F₁, F₂])
    """
    
    def __init__(self, config: ImpedanceForceConfig):
        LeafSystem.__init__(self)
        self.config = config
        
        # Input 0: actual [y, ẏ] (4D)
        self.DeclareVectorInputPort("y_v", BasicVector(4))
        
        # Input 1: reference [y_ref, ẏ_ref] (4D)
        self.DeclareVectorInputPort("yref_vref", BasicVector(4))
        
        # Output: F_imp (2D)
        self.DeclareVectorOutputPort("F_imp", BasicVector(2), self.CalcOutput)
    
    def CalcOutput(self, context, output):
        """Compute impedance force: F_imp = K_p*(y_ref - y) + K_d*(ẏ_ref - ẏ)"""
        y_v = self.get_input_port(0).Eval(context)  # [y₁, y₂, ẏ₁, ẏ₂]
        yref_vref = self.get_input_port(1).Eval(context)  # [y_ref₁, y_ref₂, ẏ_ref₁, ẏ_ref₂]
        
        # Extract positions and velocities
        y = y_v[:2]  # [y₁, y₂]
        v = y_v[2:]  # [ẏ₁, ẏ₂]
        yref = yref_vref[:2]  # [y_ref₁, y_ref₂]
        vref = yref_vref[2:]  # [ẏ_ref₁, ẏ_ref₂]
        
        # Compute impedance force
        # Use @ for matrix-vector multiplication (kp and kd are 2x2 matrices)
        F_imp = self.config.kp @ (yref - y) + self.config.kd @ (vref - v)
        
        # Optional saturation
        if self.config.force_limit is not None:
            F_imp = np.clip(F_imp, -self.config.force_limit, self.config.force_limit)
        
        output.SetFromVector(F_imp)


# ============================================================================
# ZFT / REFERENCE MASS (2D VERSION FOR CUP MANIPULATOR)
# ============================================================================
class ZFTReferenceMass(LeafSystem):
    """
    2D Zero-Force Tracking (ZFT) / Reference Mass dynamics.
    
    This is an internal model that generates reference trajectories
    based on actual motion and muscle force input.
    
    Dynamics:
        ẏ_ref = v_ref
        M_ref · v̇_ref = K_p·(y - y_ref) + K_d·(v - v_ref) + F
    
    where:
    - M_ref: Reference mass matrix (scalar×I or 2x2 matrix)
    - K_p, K_d: Coupling gains (diagonal or full matrices)
    - F: Muscle force (2D)
    - y, v: Actual end-effector position/velocity (2D each)
    
    Inputs:
      - y_v: [y, v] (4D: [y₁, y₂, v₁, v₂])
      - F: muscle force (2D: [F₁, F₂])
    
    Output:
      - yref_vref: [y_ref, v_ref] (4D: [y_ref₁, y_ref₂, v_ref₁, v_ref₂])
    
    State:
      - [y_ref, v_ref] (4D total: [y_ref₁, y_ref₂, v_ref₁, v_ref₂])
    """
    
    def __init__(self, config: ZFTReferenceMassConfig):
        LeafSystem.__init__(self)
        self.config = config
        
        # Input 0: actual [y, v] (4D)
        self.DeclareVectorInputPort("y_v", BasicVector(4))
        
        # Input 1: muscle force F (2D)
        self.DeclareVectorInputPort("F", BasicVector(2))
        
        # Continuous state: [y_ref, v_ref] (4D)
        self.DeclareContinuousState(4)
        
        # Output: [y_ref, v_ref] (4D)
        self.DeclareVectorOutputPort("yref_vref", BasicVector(4), self.CalcOutput)
    
    def SetDefaultState(self, context, state):
        """Initialize reference position and velocity."""
        # Initialize from config: [y_ref_init, v_ref_init]
        yref0 = self.config.yref0  # (2,)
        vref0 = self.config.vref0  # (2,)
        state_init = np.concatenate([yref0, vref0])  # (4,)
        state.get_mutable_continuous_state_vector().SetFromVector(state_init)
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        """
        Compute ZFT dynamics:
            ẏ_ref = v_ref
            v̇_ref = (K_p·(y - y_ref) + K_d·(v - v_ref) + F) / M_ref
        """
        y_v = self.get_input_port(0).Eval(context)  # [y₁, y₂, v₁, v₂]
        F = self.get_input_port(1).Eval(context)  # [F₁, F₂]
        state = context.get_continuous_state_vector().CopyToVector()  # [y_ref₁, y_ref₂, v_ref₁, v_ref₂]
        
        # Extract
        y = y_v[:2]  # [y₁, y₂]
        v = y_v[2:]  # [v₁, v₂]
        yref = state[:2]  # [y_ref₁, y_ref₂]
        vref = state[2:]  # [v_ref₁, v_ref₂]
        
        # Dynamics
        yref_dot = vref  # (2,)
        # Use @ for matrix-vector multiplication and np.linalg.solve for M^{-1}
        vref_dot = np.linalg.solve(
            self.config.Mh,
            self.config.kp @ (y - yref) + self.config.kd @ (v - vref) + F
        )  # (2,)
        
        # Assemble derivative: [ẏ_ref₁, ẏ_ref₂, v̇_ref₁, v̇_ref₂]
        state_dot = np.concatenate([yref_dot, vref_dot])
        derivatives.get_mutable_vector().SetFromVector(state_dot)
    
    def CalcOutput(self, context, output):
        """Output current reference: [y_ref, v_ref]"""
        state = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(state)


# ============================================================================
# END-EFFECTOR KINEMATICS (FOR MANIPULATOR-PUSHES-CART)
# ============================================================================
class EndEffectorKinematics(LeafSystem):
    """
    Computes end-effector position and velocity from manipulator state.
    
    For 2-DOF planar manipulator with link lengths L1, L2:
    - Forward Kinematics: (x_ee, y_ee) = FK(q1, q2)
    - Velocity: (vx_ee, vy_ee) = J(q) @ q_dot
    
    INPUT:
        - manipulator_state: [q1, q2, q̇1, q̇2]^T (4D)
    
    OUTPUTS:
        - ee_position: [x, y]^T (2D)
        - ee_velocity: [ẋ, ẏ]^T (2D)
    """
    
    def __init__(self, plant, manipulator):
        LeafSystem.__init__(self)
        
        self.plant = plant
        self.manipulator = manipulator
        self.model_instance = manipulator.model_instance
        self.ee_body = plant.GetBodyByName("link2", manipulator.model_instance)
        self.world_frame = plant.world_frame()
        
        # Plant context for FK/Jacobian computation
        self.plant_context = plant.CreateDefaultContext()
        
        # Input: manipulator state [q1, q2, q̇1, q̇2]
        self.DeclareVectorInputPort("manipulator_state", BasicVector(4))
        
        # Outputs
        self.DeclareVectorOutputPort(
            "ee_position",
            BasicVector(2),
            self._calc_position
        )
        self.DeclareVectorOutputPort(
            "ee_velocity",
            BasicVector(2),
            self._calc_velocity
        )
    
    def _calc_position(self, context, output):
        """Compute EE position via forward kinematics."""
        state = self.GetInputPort("manipulator_state").Eval(context)
        q = state[:2]
        
        # Set plant positions
        self.plant.SetPositions(self.plant_context, self.model_instance, q)
        
        # Get EE position in world frame
        ee_pose = self.plant.EvalBodyPoseInWorld(self.plant_context, self.ee_body)
        ee_pos_3d = ee_pose.translation() + ee_pose.rotation() @ self.manipulator.EE_OFFSET
        
        # Extract X, Y (planar)
        output.SetFromVector([ee_pos_3d[0], ee_pos_3d[1]])
    
    def _calc_velocity(self, context, output):
        """Compute EE velocity via Jacobian."""
        state = self.GetInputPort("manipulator_state").Eval(context)
        q = state[:2]
        q_dot = state[2:]
        
        # Set plant state
        self.plant.SetPositions(self.plant_context, self.model_instance, q)
        
        # Compute Jacobian
        from pydrake.multibody.tree import JacobianWrtVariable
        J_spatial = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            self.ee_body.body_frame(),
            self.manipulator.EE_OFFSET,
            self.world_frame,
            self.world_frame
        )
        
        # Extract translational part (rows 3:6) and manipulator columns (first 2)
        J_trans = J_spatial[3:6, :2]  # 3×2
        
        # Compute velocity
        v_ee_3d = J_trans @ q_dot  # 3D velocity
        
        # Extract X, Y
        output.SetFromVector([v_ee_3d[0], v_ee_3d[1]])


# ============================================================================
# MANIPULATOR JACOBIAN TRANSPOSE CONTROLLER
# ============================================================================
class ManipulatorJacobianTransposeController(LeafSystem):
    """
    Converts task-space impedance force to joint torques via Jacobian transpose.
    
    τ = -J^T(q) F_imp
    
    INPUTS:
        - F_imp: Scalar impedance force (1D for X-axis only)
        - manipulator_state: [q1, q2, q̇1, q̇2]^T
    
    OUTPUT:
        - joint_torques: [τ1, τ2]^T
    """
    
    def __init__(self, plant, manipulator):
        LeafSystem.__init__(self)
        
        self.plant = plant
        self.manipulator = manipulator
        self.model_instance = manipulator.model_instance
        self.ee_body = plant.GetBodyByName("link2", manipulator.model_instance)
        self.world_frame = plant.world_frame()
        
        # Plant context for Jacobian computation
        self.plant_context = plant.CreateDefaultContext()
        
        # Input ports
        self.F_imp_input = self.DeclareVectorInputPort("F_imp", BasicVector(1))
        self.state_input = self.DeclareVectorInputPort("manipulator_state", BasicVector(4))
        
        # Output port
        self.DeclareVectorOutputPort(
            "joint_torques",
            BasicVector(2),
            self._calc_torques
        )
    
    def _calc_torques(self, context, output):
        """Compute τ = -J^T F_imp"""
        # Get inputs
        state = self.state_input.Eval(context)
        q = state[:2]
        F_imp_scalar = float(self.F_imp_input.Eval(context)[0])
        
        # 3D force vector (X-axis only)
        F_imp_3d = np.array([F_imp_scalar, 0.0, 0.0])
        
        # Set plant configuration
        full_q = self.plant.GetPositions(self.plant_context)
        full_q[:2] = q
        self.plant.SetPositions(self.plant_context, full_q)
        
        # Compute Jacobian
        from pydrake.multibody.tree import JacobianWrtVariable
        J_spatial = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            self.ee_body.body_frame(),
            self.manipulator.EE_OFFSET,
            self.world_frame,
            self.world_frame
        )
        
        # Extract translational part, manipulator columns only
        J_translational = J_spatial[3:6, :2]  # 3×2
        
        # Joint torques: τ = -J^T F_imp
        tau = -J_translational.T @ F_imp_3d  # (2×3) @ (3,) = (2,)
        
        output.SetFromVector(tau)


# ============================================================================
# IMPEDANCE TO CART FORCE CONVERTER
# ============================================================================
class ImpedanceToCartForce(LeafSystem):
    """
    Converts 1D impedance force F_imp to 2D cart force [F_x, F_y].
    
    For X-axis motion only: [F_imp, 0]^T
    
    INPUT:
        - F_imp: scalar impedance force [N]
    
    OUTPUT:
        - cart_force: [F_x, F_y]^T = [F_imp, 0]^T
    """
    
    def __init__(self):
        LeafSystem.__init__(self)
        
        self.DeclareVectorInputPort("F_imp", BasicVector(1))
        self.DeclareVectorOutputPort(
            "cart_force",
            BasicVector(2),
            self._calc_output
        )
    
    def _calc_output(self, context, output):
        """Convert 1D force to 2D cart force."""
        F_imp = float(self.GetInputPort("F_imp").Eval(context)[0])
        output.SetFromVector([F_imp, 0.0])


# ============================================================================
# BUILD SYSTEM FOR CUP MANIPULATOR (2-DOF VERSION)
# ============================================================================

class BuildCupManipulatorSystem:
    """
    Complete OFC system builder for 2-DOF Cup Manipulator with Pendulum Interaction.
    
    ARCHITECTURE (following cart-pendulum pattern + manipulator specifics):
    ========================================================================
    
      u -> MuscleDynamics -> F -> ZFTReferenceMass -> (y_ref, v_ref)
                             ↓
         Plant state (4D) -> J(q) -> (y, ẏ) ---------------------┐
                                                                  v
                                              ImpedanceForce -> F_imp -> J(q)ᵀ -> τ -> Plant
                                                               ↓
                                                            -F_imp -> Pendulum Pivot (external force)
    
    KEY INSIGHT FROM CART-PENDULUM:
    ================================
    - In cart-pendulum: impedance force F_imp directly affects cart (1D actuator)
    - In cup manipulator: 
      * F_imp affects manipulator via τ = J(q)ᵀ·F_imp (torque mapping)
      * F_imp also applied to pendulum pivot as external force (Newton's 3rd law)
      * Manipulator "reacts" to impedance just like cart does
    
    FORCE INTERACTION (from images):
    =================================
    1. Virtual reference mass (ZFT):
       M_ref·ÿ_ref = K_p(y - y_ref) + K_d(ẏ - ẏ_ref) + F
       - Receives muscle force F
       - Tracks actual end-effector motion
    
    2. Impedance force (task space):
       F_imp = K_p(y_ref - y) + K_d(ẏ_ref - ẏ)
       - Applied at end-effector in task space
       - Can be diagonal (decoupled x/y) or full 2×2 matrix
    
    3. Torque command (joint space):
       τ = J(q)ᵀ·F_imp + τ_stab
       - Jacobian transpose maps task force to joint torques
       - τ_stab = -K_d·q̇ (joint damping/stabilization)
    
    4. Environment interaction (pendulum):
       - Pendulum pivot receives -F_imp (reaction force)
       - For full MultibodyPlant: apply as external force
       - For linearized: already coupled in dynamics
    
    IMPLEMENTATION MODES:
    =====================
    
    MODE 1: LINEARIZED SYSTEM (Current Implementation)
    --------------------------------------------------
    - Uses CupManipulatorLinearizedSystem
    - Impedance force affects linearized dynamics via τ = J₀ᵀ·F_imp
    - Pendulum dynamics linearized and coupled
    - Advantages: Fast, suitable for control design
    - Limitations: Valid only near equilibrium
    
    MODE 2: FULL MULTIBODY PLANT (Future/Advanced)
    -----------------------------------------------
    - Uses CupManipulatorWithPendulum (full nonlinear plant)
    - Apply F_imp to manipulator EE via external force port
    - Apply -F_imp to pendulum pivot (Newton's 3rd law)
    - Advantages: Accurate for large motions
    - Requires: External force ports on MultibodyPlant
    
    STATE VECTOR (14D TOTAL):
    =========================
    
    PLANT STATES (8D):
    ------------------
    1. q₁         - Manipulator Joint 1 position [rad]
    2. q₂         - Manipulator Joint 2 position [rad]
    3. q̇₁         - Manipulator Joint 1 velocity [rad/s]
    4. q̇₂         - Manipulator Joint 2 velocity [rad/s]
    5. pitch      - Pendulum pitch angle (α) [rad]
    6. roll       - Pendulum roll angle (β) [rad]
    7. pitch_dot  - Pendulum pitch velocity (α̇) [rad/s]
    8. roll_dot   - Pendulum roll velocity (β̇) [rad/s]
    
    MUSCLE STATES (2D):
    -------------------
    9. F₁         - Muscle force 1 state [N]
    10. F₂        - Muscle force 2 state [N]
    
    ZFT REFERENCE STATES (4D):
    --------------------------
    11. y_ref₁    - ZFT reference position 1 (EE x) [m]
    12. y_ref₂    - ZFT reference position 2 (EE y) [m]
    13. v_ref₁    - ZFT reference velocity 1 (EE ẋ) [m/s]
    14. v_ref₂    - ZFT reference velocity 2 (EE ẏ) [m/s]
    
    CRITICAL: Pendulum States in Linearized System
    ===============================================
    
    **CURRENT ISSUE**: CupManipulatorLinearizedSystem currently only linearizes
    the manipulator (4D), but DOES NOT include the pendulum states!
    
    **REQUIRED FIX**: The linearization must be done on the COMPLETE system:
    - Build CupManipulatorWithPendulum (full plant with pendulum attached)
    - Linearize the FULL 8D system around equilibrium
    - This gives A_plant (8×8), B_plant (8×2) including coupled dynamics
    
    **WHY THIS MATTERS**:
    - Pendulum motion affects manipulator via inertial coupling
    - Manipulator motion affects pendulum via pivot connection
    - F_imp applied to pendulum must be in the state equations
    - Cannot separate manipulator and pendulum dynamics!
    
    **ANALOGY TO CART-PENDULUM**:
    - Cart-pendulum: linearized as [x, θ, ẋ, θ̇] (4D) - ONE coupled system
    - Cup-manipulator+pendulum: should be [q₁, q₂, q̇₁, q̇₂, α, β, α̇, β̇] (8D) - ONE coupled system
    
    TODO: Update build_plant_for_linearization() to use full pendulum plant!
    5. F₁      - Muscle force 1 state [N]
    6. F₂      - Muscle force 2 state [N]
    7. y_ref₁  - ZFT reference position 1 (EE x) [m]
    8. y_ref₂  - ZFT reference position 2 (EE y) [m]
    9. v_ref₁  - ZFT reference velocity 1 (EE ẋ) [m/s]
    10. v_ref₂ - ZFT reference velocity 2 (EE ẏ) [m/s]
    
    Components:
    -----------
    - States 1-8:  Cup manipulator + pendulum plant (FULL 8D coupled system)
                   * Manipulator: q₁, q₂, q̇₁, q̇₂
                   * Pendulum: pitch, roll, pitch_dot, roll_dot
    - States 9-10:  Muscle dynamics (Ḟ = (u - F) / τ_m)
    - States 11-14: ZFT reference mass (M_ref·v̇_ref = K_p·(y-y_ref) + K_d·(v-v_ref) + F)
    
    Exposes:
      - command input port (u): Neural command to muscle [N] (2D)
      - assembled output state: [manipulator(4D), pendulum(4D), muscle(2D), zft(4D)] → 14D
      - (optional) F_imp output for external force application to pendulum
    
    IMPORTANT: Linearization Strategy
    ==================================
    The linearization MUST include the pendulum! Options:
    
    1. **Full Coupled Linearization** (Recommended):
       - Build CupManipulatorWithPendulum as MultibodyPlant
       - Add pendulum to link2 (as done in build() method)
       - Linearize ENTIRE plant → 8D state [q₁, q₂, q̇₁, q̇₂, α, β, α̇, β̇]
       - This captures manipulator-pendulum coupling naturally
    
    2. **Separate Linearization** (Alternative, less accurate):
       - Linearize manipulator alone → 4D
       - Linearize pendulum alone → 4D  
       - Manually couple via external force F_imp
       - May miss some dynamic coupling terms
    """

    def __init__(
        self,
        physics_config: ManipulatorConfig,
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
        self.linearized_system = None  # CupManipulatorLinearizedSystem instance
        self.linearized_plant = None   # The actual Drake LinearSystem
        self.muscle = None
        self.u_saturation = None
        self.zft = None
        self.impedance = None
        self.jacobian_selector = None
        self.jacobian_transpose = None
        self.state_mux = None

        # Exposed ports
        self.command_input_port = None
        self.output_port = None
        self.f_imp_output_port = None  # For applying to pendulum pivot (optional)

    # -------------------------
    # Build
    # -------------------------
    def build(self):
        """Build complete OFC system with muscle dynamics, ZFT, and impedance control."""
        print(colored("\n" + "=" * 80, "cyan"))
        print(colored("BUILDING CUP MANIPULATOR OFC SYSTEM", "cyan", attrs=["bold"]))
        print(colored("=" * 80, "cyan"))
        
        print(colored("\n⚠ WARNING: Current implementation uses 4D plant (manipulator only)", "yellow"))
        print(colored("  TODO: Update to use 8D plant (manipulator + pendulum)", "yellow"))
        print(colored("  Required: Linearize CupManipulatorWithPendulum, not just manipulator\n", "yellow"))

        # Step 1: Create linearized cup manipulator plant
        # TODO: This should linearize the FULL system with pendulum!
        # Currently only linearizes manipulator (4D) - missing pendulum (4D)
        print(colored("\n[1/7] Creating linearized cup manipulator plant...", "yellow"))
        self.linearized_system = CupManipulatorLinearizedSystem(
            config=self.physics_config,
            enable_visualization=False,
            linearization_method='drake',
        )
        self.linearized_system.build_plant_for_linearization(self.builder)
        self.linearized_plant = self.linearized_system.linearized_io_sys
        print(colored("  ✓ Linearized plant created (4D - manipulator only)", "green"))
        print(colored("  ⚠ Missing pendulum states (4D) - plant should be 8D!", "yellow"))

        # Step 2: Muscle dynamics
        print(colored("\n[2/7] Adding muscle dynamics (2D)...", "yellow"))
        self.muscle = self.builder.AddSystem(MuscleDynamics(config=self.muscle_config))
        self.muscle.set_name("muscle_dynamics_2d")
        print(colored("  ✓ Muscle dynamics added", "green"))

        # Optional saturation on u
        if self.muscle_config.command_limit is not None:
            lim = float(self.muscle_config.command_limit)
            from pydrake.all import Saturation
            self.u_saturation = self.builder.AddSystem(
                Saturation(min_value=[-lim, -lim], max_value=[lim, lim])
            )
            self.u_saturation.set_name("u_saturation")
            self.command_input_port = self.u_saturation.get_input_port(0)
            self.builder.Connect(self.u_saturation.get_output_port(0),
                                self.muscle.get_input_port(0))
            print(colored("  ✓ Command saturation added", "green"))
        else:
            self.command_input_port = self.muscle.get_input_port(0)

        # Step 3: ZFT/reference mass
        print(colored("\n[3/7] Adding ZFT reference mass (2D)...", "yellow"))
        self.zft = self.builder.AddSystem(ZFTReferenceMass(config=self.zft_config))
        self.zft.set_name("zft_reference_mass_2d")
        print(colored("  ✓ ZFT reference mass added", "green"))

        # Step 4: Impedance force block
        print(colored("\n[4/7] Adding impedance force (2D)...", "yellow"))
        self.impedance = self.builder.AddSystem(ImpedanceForce(config=self.impedance_config))
        self.impedance.set_name("impedance_force_2d")
        print(colored("  ✓ Impedance force added", "green"))

        # Step 5: Jacobian forward kinematics: q → (y, ẏ)
        # For 2-DOF planar manipulator: J(q) maps joint velocities to end-effector velocities
        # y = forward_kinematics(q), ẏ = J(q)·q̇
        #
        # LINEARIZED APPROXIMATION:
        # At equilibrium q₀, we use constant Jacobian J₀ = J(q₀)
        # This selector extracts [y, ẏ] ≈ [J₀·q, J₀·q̇] from linearized plant state
        #
        # For now: Identity mapping (assumes plant state is already in task space)
        # TODO: Add proper Jacobian computation from equilibrium configuration
        print(colored("\n[5/7] Setting up Jacobian transformations...", "yellow"))
        from pydrake.all import MatrixGain
        
        # Simplified selector: assumes joint space ≈ task space for small angles
        # S maps [q₁, q₂, q̇₁, q̇₂] → [y₁, y₂, ẏ₁, ẏ₂]
        # For proper implementation, compute actual Jacobian at equilibrium
        S = np.eye(4)  # Identity for now (joint space = task space approximation)
        self.jacobian_selector = self.builder.AddSystem(MatrixGain(S))
        self.jacobian_selector.set_name("jacobian_forward_kinematics")
        
        self.builder.Connect(self.linearized_plant.get_output_port(0),
                            self.jacobian_selector.get_input_port(0))
        print(colored("  ✓ Forward kinematics (Jacobian) added", "green"))

        # Step 6: Connect feedback loops
        print(colored("\n[6/7] Wiring feedback connections...", "yellow"))
        
        # CRITICAL: Force flow following cart-pendulum pattern
        # =====================================================
        # 
        # 1. Plant state → [y, ẏ] → ZFT input 0
        #    - ZFT tracks actual end-effector motion
        #
        # 2. Muscle force F → ZFT input 1
        #    - ZFT reference mass evolves based on muscle force
        #
        # 3. Plant state [y, ẏ] → Impedance input 0 (actual motion)
        #    ZFT output [y_ref, v_ref] → Impedance input 1 (reference)
        #    - Impedance computes F_imp = K_p(y_ref - y) + K_d(v_ref - v)
        #
        # 4. F_imp → J^T → τ → Plant input
        #    - Manipulator receives torque via Jacobian transpose
        #    - This makes manipulator "react" to impedance (like cart does)
        #
        # 5. F_imp → -F_imp → Pendulum pivot (future: external force)
        #    - Pendulum receives reaction force (Newton's 3rd law)
        #    - For full MultibodyPlant: apply via external force port
        #    - For linearized: already coupled in dynamics
        
        # plant [y,ẏ] -> zft input 0
        self.builder.Connect(self.jacobian_selector.get_output_port(0),
                            self.zft.get_input_port(0))
        print(colored("    ✓ Plant → ZFT: actual motion feedback", "green"))
        
        # muscle F -> zft input 1
        self.builder.Connect(self.muscle.get_output_port(0),
                            self.zft.get_input_port(1))
        print(colored("    ✓ Muscle → ZFT: force input", "green"))
        
        # plant [y,ẏ] -> impedance input 0
        self.builder.Connect(self.jacobian_selector.get_output_port(0),
                            self.impedance.get_input_port(0))
        print(colored("    ✓ Plant → Impedance: actual motion", "green"))
        
        # zft [yref,vref] -> impedance input 1
        self.builder.Connect(self.zft.get_output_port(0),
                            self.impedance.get_input_port(1))
        print(colored("    ✓ ZFT → Impedance: reference motion", "green"))
        
        # Jacobian transpose: F_imp → τ (τ = J(q)ᵀ·F_imp)
        # For linearized system at equilibrium, use constant J₀ᵀ
        # TODO: Compute actual Jacobian at equilibrium configuration
        Jᵀ = np.eye(2)  # Identity for now (proper Jacobian transpose needed)
        self.jacobian_transpose = self.builder.AddSystem(MatrixGain(Jᵀ))
        self.jacobian_transpose.set_name("jacobian_transpose_torque")
        
        # impedance F_imp -> Jᵀ -> plant τ
        # This is the KEY CONNECTION: manipulator receives impedance force as torque
        # (analogous to cart receiving F_imp directly in cart-pendulum)
        self.builder.Connect(self.impedance.get_output_port(0),
                            self.jacobian_transpose.get_input_port(0))
        self.builder.Connect(self.jacobian_transpose.get_output_port(0),
                            self.linearized_plant.get_input_port(0))
        print(colored("    ✓ Impedance → J^T → Plant: torque command", "green"))
        
        # Export F_imp for optional external use (e.g., apply to pendulum pivot)
        self.f_imp_output_port = self.impedance.get_output_port(0)
        print(colored("    ✓ F_imp available for pendulum interaction", "green"))
        
        print(colored("  ✓ All feedback loops connected", "green"))

        # Step 7: Output wiring
        print(colored("\n[7/7] Assembling output state...", "yellow"))
        if self.assemble_output_state:
            # Output: [plant(4D - SHOULD BE 8D), F(2D), yref_vref(4D)] => 10D (SHOULD BE 14D)
            # TODO: Update to [plant(8D), F(2D), yref_vref(4D)] => 14D when pendulum added
            from pydrake.all import Multiplexer
            self.state_mux = self.builder.AddSystem(Multiplexer([4, 2, 4]))
            self.state_mux.set_name("state_mux_10d")  # TODO: Should be 14d

            self.builder.Connect(self.linearized_plant.get_output_port(0),
                                self.state_mux.get_input_port(0))
            self.builder.Connect(self.muscle.get_output_port(0),
                                self.state_mux.get_input_port(1))
            self.builder.Connect(self.zft.get_output_port(0),
                                self.state_mux.get_input_port(2))

            self.output_port = self.state_mux.get_output_port(0)
            print(colored("  ✓ 10D output state assembled (4D plant + 2D muscle + 4D ZFT)", "green"))
            print(colored("  ⚠ Should be 14D when pendulum states added!", "yellow"))
        else:
            # Just expose plant output (4D, should be 8D)
            self.output_port = self.linearized_plant.get_output_port(0)
            print(colored("  ✓ 4D plant output exposed (manipulator only)", "green"))
            print(colored("  ⚠ Missing 4D pendulum states!", "yellow"))

        print(colored("\n" + "=" * 80, "green"))
        print(colored("✓ BUILD COMPLETE", "green", attrs=["bold"]))
        print(colored("=" * 80, "green"))
        
        return self

    # -------------------------
    # Ports
    # -------------------------
    def get_command_input_port(self):
        """Get command input port for muscle command u (2D)."""
        return self.command_input_port

    def get_state_output_port(self):
        """Get assembled state output port (4D or 10D depending on config)."""
        return self.output_port
    
    def get_impedance_force_output_port(self):
        """
        Get impedance force output port F_imp (2D).
        
        This can be used to apply external forces:
        - To pendulum pivot: -F_imp (Newton's 3rd law)
        - To environment model
        - For logging/visualization
        
        Returns:
            Output port for F_imp ∈ ℝ² [F_x, F_y]
        """
        return self.f_imp_output_port
    
    # -------------------------
    # Get Full System Linearization (14D)
    # -------------------------
    def get_full_system_matrices(self):
        """
        Construct full 14D system linearization: A (14x14), B (14x2)
        
        State: [q₁, q₂, q̇₁, q̇₂, α, β, α̇, β̇, F₁, F₂, y_ref₁, y_ref₂, v_ref₁, v_ref₂]
        Input: u (2D command to muscle)
        
        **CURRENT LIMITATION**: This method currently only implements 10D (manipulator-only)
        **TODO**: Update to 14D when pendulum states are added to linearized plant
        
        FULL STATE BREAKDOWN:
        =====================
        - Plant (8D): [q₁, q₂, q̇₁, q̇₂, α, β, α̇, β̇]
          * Manipulator: q₁, q₂, q̇₁, q̇₂ (joint angles/velocities)
          * Pendulum: α, β, α̇, β̇ (pitch/roll angles/velocities)
        - Muscle (2D): [F₁, F₂]
        - ZFT (4D): [y_ref₁, y_ref₂, v_ref₁, v_ref₂]
        
        FORCE INTERACTION DYNAMICS:
        ===========================
        
        1. Manipulator Plant (4D): q, q̇
           - Receives torque: τ = J(q)ᵀ·F_imp
           - F_imp from impedance controller
           - Dynamics: M(q)q̈ + C(q,q̇) = τ
           - Linearized: ẋ_plant = A_plant·x_plant + B_plant·τ
        
        2. Muscle Dynamics (2D): F₁, F₂
           - Ḟ = (u - F) / τ_m
           - Provides force to ZFT reference mass
        
        3. ZFT Reference Mass (4D): y_ref, v_ref
           - ẏ_ref = v_ref
           - M_ref·v̇_ref = K_p(y - y_ref) + K_d(ẏ - ẏ_ref) + F
           - Receives: actual EE motion (y, ẏ) and muscle force F
           - Outputs: reference trajectory (y_ref, v_ref)
        
        4. Impedance Force (computed, not a state):
           - F_imp = K_p(y_ref - y) + K_d(v_ref - v)
           - Affects plant via: τ = J^T·F_imp
           - Applied to pendulum as: -F_imp (Newton's 3rd law)
        
        COUPLING STRUCTURE:
        ===================
        
        Plant ←τ=J^T·F_imp← Impedance ←(y,v)← Plant (closed loop)
                                     ↑
                                  (y_ref, v_ref)
                                     ↑
                                    ZFT ←F← Muscle ←u← Command
                                     ↑
                                   (y,v)
        
        Returns:
            A (10x10): Full system state matrix
            B (10x2): Full system input matrix
        
        Note: Uses analytical coupling similar to cart-pendulum.
              For exact linearization, use Drake's Linearize() on full diagram.
        """
        
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
        
        # Get Jacobian transpose (currently identity, TODO: compute actual)
        J_T = np.eye(2)  # Jacobian transpose at equilibrium
        
        # Build full 10x10 A matrix
        # State order: [q₁, q₂, q̇₁, q̇₂, F₁, F₂, y_ref₁, y_ref₂, v_ref₁, v_ref₂]
        A_full = np.zeros((10, 10))
        
        # Plant dynamics (4x4 block): affected by impedance force F_imp via J^T
        # ẋ_plant = A_plant·x_plant + B_plant·τ
        # τ = J^T·F_imp = J^T·[kp_imp*(y_ref - y) + kd_imp*(v_ref - v)]
        # 
        # Linearize around equilibrium:
        # ∂τ/∂q ≈ -J^T·kp_imp  (assuming y ≈ q at equilibrium)
        # ∂τ/∂q̇ ≈ -J^T·kd_imp  (assuming ẏ ≈ q̇ at equilibrium)
        # ∂τ/∂y_ref = J^T·kp_imp
        # ∂τ/∂v_ref = J^T·kd_imp
        
        # Top-left 4x4: Plant dynamics
        A_full[:4, :4] = A_plant
        
        # Plant coupling with impedance via Jacobian transpose
        # For 2-DOF planar: assume y ≈ [q₁, q₂] for simplicity (TODO: proper FK)
        A_full[:4, :2] += B_plant @ J_T @ (-kp_imp * np.eye(2))  # ∂τ/∂q
        A_full[:4, 2:4] += B_plant @ J_T @ (-kd_imp * np.eye(2))  # ∂τ/∂q̇
        A_full[:4, 6:8] += B_plant @ J_T @ (kp_imp * np.eye(2))  # ∂τ/∂y_ref
        A_full[:4, 8:10] += B_plant @ J_T @ (kd_imp * np.eye(2))  # ∂τ/∂v_ref
        
        # Muscle dynamics (2x2 block): Ḟ = (-F + u) / τ_m
        A_full[4:6, 4:6] = -np.eye(2) / tau_muscle
        
        # ZFT dynamics: ẏ_ref = v_ref
        A_full[6:8, 8:10] = np.eye(2)
        
        # ZFT dynamics: v̇_ref = (K_p(y - y_ref) + K_d(ẏ - ẏ_ref) + F) / M_ref
        # Assume y ≈ q₁,q₂ and ẏ ≈ q̇₁,q̇₂ for linearization
        A_full[8:10, :2] = (kp_zft / Mh) * np.eye(2)     # ∂v̇_ref/∂q (≈ ∂v̇_ref/∂y)
        A_full[8:10, 2:4] = (kd_zft / Mh) * np.eye(2)    # ∂v̇_ref/∂q̇ (≈ ∂v̇_ref/∂ẏ)
        A_full[8:10, 4:6] = (1.0 / Mh) * np.eye(2)       # ∂v̇_ref/∂F
        A_full[8:10, 6:8] = (-kp_zft / Mh) * np.eye(2)   # ∂v̇_ref/∂y_ref
        A_full[8:10, 8:10] = (-kd_zft / Mh) * np.eye(2)  # ∂v̇_ref/∂v_ref
        
        # Build 10x2 B matrix (input affects only muscle dynamics)
        B_full = np.zeros((10, 2))
        B_full[4:6, :] = np.eye(2) / tau_muscle  # u → Ḟ
        
        print(colored("\n⚠ WARNING: Returning 10D system (manipulator-only)", "yellow"))
        print(colored("  Current: A (10×10), B (10×2)", "yellow"))
        print(colored("  Should be: A (14×14), B (14×2) with pendulum states", "yellow"))
        print(colored("  Missing: 4D pendulum block in A matrix\n", "yellow"))
        
        return A_full, B_full


# ============================================================================
# BUILD PLANT WITH PENDULUM AND VISUALIZATION
# ============================================================================

class CupManipulatorWithPendulum(RobotBase):
    """
    Wrapper class for cup manipulator with 3D pendulum system.
    
    This class encapsulates the complete Drake plant setup including:
    - Cup manipulator from URDF
    - Optional 3D pendulum attachment
    - Meshcat visualization
    - Simulation control
    - Interactive scene visualization
    """
    
    def __init__(
        self,
        cup_manipulator: CupManipulator,
        pendulum: Optional[Pendulum3D] = None,
        enable_visualization: bool = True,
        initial_pendulum_pitch: float = 0.0,
        initial_pendulum_roll: float = 180.0,
    ):
        """
        Initialize the cup manipulator with pendulum system.
        
        Args:
            cup_manipulator: CupManipulator instance (already configured)
            pendulum: Pendulum3D instance (optional, None if no pendulum)
            enable_visualization: If True, setup Meshcat visualization
            initial_pendulum_pitch: Initial pendulum pitch angle in degrees
            initial_pendulum_roll: Initial pendulum roll angle in degrees
        """
        # Initialize RobotBase with cup manipulator's config and visualization settings
        super().__init__(cup_manipulator.config, name="cup_manipulator_with_pendulum", enable_visualization=enable_visualization)
        
        # Store component instances
        self.cup_manipulator = cup_manipulator
        self.pendulum = pendulum
        self.initial_pendulum_pitch = initial_pendulum_pitch
        self.initial_pendulum_roll = initial_pendulum_roll
        
        # System components (initialized in build())
        self.diagram = None
        self.simulator = None
        self.plant = None
        self.scene_graph = None
        self.context = None
        self.plant_context = None
        self.builder = DiagramBuilder()
        # Note: self.meshcat and self.visualizer_params already initialized by RobotBase

    def build(self):
        """
        Build the complete system using stored configuration.
        
        This method builds the Drake diagram with the cup manipulator and pendulum
        that were passed to the constructor.
        
        Returns:
            self: For method chaining
        """
        from pydrake.all import DiagramBuilder, Simulator
        
        print(colored("\n" + "=" * 80, "yellow"))
        print(colored("BUILDING CUP MANIPULATOR WITH 3D PENDULUM", "yellow", attrs=["bold"]))
        print(colored("=" * 80, "yellow"))
        
        # Step 1: Create DiagramBuilder and add MultibodyPlant + SceneGraph
        print(colored("\n[1/6] Creating DiagramBuilder and MultibodyPlant...", "cyan"))
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=0.001)
        
        print(colored("  ✓ DiagramBuilder created", "green"))
        print(colored("  ✓ MultibodyPlant + SceneGraph added", "green"))
        
        # Step 2: Build cup manipulator in plant (complete setup)
        print(colored("\n[2/6] Building cup manipulator in plant...", "cyan"))
        parser = Parser(self.plant)
        self.cup_manipulator.build_in_plant(self.plant, parser, weld_base=True)
        
        print(colored(f"  ✓ Cup manipulator loaded from: {self.cup_manipulator.config.urdf_path}", "green"))
        print(colored("  ✓ Base welded to world frame", "green"))
        print(colored("  ✓ Actuators added: link1_base, link2_link1", "green"))
        print(colored("  ✓ Joint properties configured", "green"))
        
        # Step 3: Attach 3D pendulum (if provided)
        if self.pendulum:
            print(colored("\n[3/6] Attaching 3D pendulum to link2...", "cyan"))
            link2_body = self.cup_manipulator.get_body_by_name(self.plant, "link2")
            self.pendulum.attach_to_body(self.plant, link2_body, self.cup_manipulator.model_instance)
            print(colored("  ✓ 3D pendulum attached to link2", "green"))
        else:
            print(colored("\n[3/6] Skipping pendulum (not provided)", "yellow"))
        
        # Set gravity
        gravity_field = self.plant.mutable_gravity_field()
        gravity_field.set_gravity_vector([0.0, 0.0, -9.81])
        
        # Step 4: Finalize plant
        print(colored("\n[4/6] Finalizing plant...", "cyan"))
        self.plant.Finalize()
        
        num_positions = self.plant.num_positions()
        num_velocities = self.plant.num_velocities()
        num_actuators = self.plant.num_actuators()
        
        print(colored(f"  ✓ Plant finalized", "green"))
        print(colored(f"    Positions: {num_positions}", "cyan"))
        print(colored(f"    Velocities: {num_velocities}", "cyan"))
        print(colored(f"    Actuators: {num_actuators}", "cyan"))
        print(colored(f"    State dimension: {num_positions + num_velocities}", "cyan"))
        
        # Steps 5-6: Setup visualization and build diagram (use RobotBase method directly)
        self.diagram, self.simulator = self.setup_diagram_and_simulator(
            self.builder, self.plant, self.scene_graph, num_actuators, add_zero_torque=True
        )
        
        # Setup contexts
        self.context = self.simulator.get_mutable_context()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        
        # Set initial state
        self.set_initial_state()
        
        # Publish initial visualization
        self.publish_visualization(self.diagram, self.context)
        
        print(colored("\n" + "=" * 80, "green"))
        print(colored("✓ BUILD COMPLETE", "green", attrs=["bold"]))
        print(colored("=" * 80, "green"))
        
        if self.enable_visualization:
            print(colored(f"\n  🌐 Meshcat URL: {self.meshcat.web_url()}", "cyan", attrs=["bold"]))
        
        print(colored(f"\n  System ready for simulation!", "yellow"))
        print(colored(f"    Use: simulator.AdvanceTo(time) to simulate", "cyan"))
        print(colored(f"    Example: simulator.AdvanceTo(5.0)  # Simulate for 5 seconds", "cyan"))
        print()
        
        return self
    
    def set_initial_state(self):
        """
        Initialize the complete system state (manipulator + pendulum).
        
        This method encapsulates all initial condition setup:
        - Cup manipulator joint positions
        - Pendulum swing angles (if pendulum provided)
        """
        # Set initial joint positions for cup manipulator
        self.cup_manipulator.set_initial_positions(self.plant, self.plant_context)
        
        # Set initial pendulum angles (if pendulum provided)
        if self.pendulum:
            pitch_rad = np.deg2rad(self.initial_pendulum_pitch)
            roll_rad = np.deg2rad(self.initial_pendulum_roll)
            self.pendulum.set_initial_swing(self.plant_context, pitch_rad, roll_rad)
            print(colored(f"  ✓ Pendulum initial angles: pitch={self.initial_pendulum_pitch}°, roll={self.initial_pendulum_roll}°", "green"))
    
    def run_scene_viz(self):
        """Run interactive scene visualization mode."""
        run_scene_viz_interactive(
            self.diagram,
            self.simulator,
            self.plant,
            self.scene_graph,
            self.meshcat,
            self.cup_manipulator,
            self.pendulum,
            self.context,
            self.plant_context
        )
    
    def get_joint_positions(self):
        """
        Get current joint positions for both manipulator and pendulum.
        
        Returns:
            Dictionary of {joint_name: position_in_radians}
        """
        # Get manipulator joint positions using parent class method
        positions = self.cup_manipulator.get_joint_positions(self.plant, self.plant_context)
        
        # Add pendulum joint positions if pendulum exists
        if self.pendulum:
            try:
                pitch_joint = self.plant.GetJointByName("pendulum_pitch")
                roll_joint = self.plant.GetJointByName("pendulum_roll")
                positions['pendulum_pitch'] = pitch_joint.get_angle(self.plant_context)
                positions['pendulum_roll'] = roll_joint.get_angle(self.plant_context)
            except Exception as e:
                print(colored(f"Warning: Could not get pendulum joints: {e}", "yellow"))
        
        return positions
    
    def set_joint_angles(self, joint_angles: dict):
        """
        Set joint angles for both manipulator and pendulum.
        
        Args:
            joint_angles: Dictionary of {joint_name: angle_in_radians}
        """
        from pydrake.multibody.tree import RevoluteJoint
        
        # Separate manipulator and pendulum joints
        manipulator_joints = {}
        pendulum_joints = {}
        
        for joint_name, angle in joint_angles.items():
            if joint_name in ["link1_base", "link2_link1"]:
                manipulator_joints[joint_name] = angle
            elif joint_name in ["pendulum_pitch", "pendulum_roll"]:
                pendulum_joints[joint_name] = angle
        
        # Set manipulator joints using parent class method
        if manipulator_joints:
            super().set_joint_angles(self.plant, self.plant_context, manipulator_joints)
        
        # Set pendulum joints (not part of cup_manipulator model instance)
        for joint_name, angle in pendulum_joints.items():
            try:
                joint = self.plant.GetJointByName(joint_name)
                if isinstance(joint, RevoluteJoint):
                    joint.set_angle(self.plant_context, angle)
            except Exception as e:
                print(colored(f"Warning: Could not set joint {joint_name}: {e}", "yellow"))
        
        # Update visualization
        if self.meshcat:
            self.diagram.ForcedPublish(self.context)

# ============================================================================
# SCENE VISUALIZATION MODE (INTERACTIVE CONTROL)
# ============================================================================

def run_scene_viz_interactive(diagram, simulator, plant, scene_graph, meshcat, cup_manipulator, pendulum, context, plant_context):
    """
    Interactive scene visualization mode.
    Allows manual control of manipulator joints and pendulum via terminal input.
    
    Args:
        diagram: Drake diagram
        simulator: Drake simulator
        plant: MultibodyPlant
        scene_graph: SceneGraph
        meshcat: Meshcat visualizer
        cup_manipulator: CupManipulator instance
        pendulum: Pendulum3D instance (or None)
        context: Simulator context (already initialized)
        plant_context: Plant context (already initialized)
    """
    print(colored("\n" + "=" * 70, "cyan", attrs=["bold"]))
    print(colored("Interactive Scene Visualization", "cyan", attrs=["bold"]))
    print(colored("=" * 70, "cyan", attrs=["bold"]))
    
    print(colored("\nVisualization Mode: Interactive Static Scene", "yellow"))
    print(colored("  - No physics simulation", "yellow"))
    print(colored("  - Manual joint control via terminal", "yellow"))
    print(colored("  - Type 'q' to exit\n", "yellow"))
    
    if not meshcat:
        print(colored("\n✗ Visualization not enabled", "red"))
        return
    
    print(colored(f"\n✓ Meshcat URL: {meshcat.web_url()}", "green", attrs=["bold"]))
    print(colored("  👉 Open this URL in your browser to view the scene\n", "yellow", attrs=["bold"]))
    
    # Verify context is valid
    if context is None or plant_context is None:
        print(colored("\n✗ Error: Context not properly initialized", "red"))
        return
    
    # Force publish initial state
    try:
        diagram.ForcedPublish(context)
    except Exception as e:
        print(colored(f"\n⚠ Warning: Could not publish initial state: {e}", "yellow"))
    
    # Print initial state
    joint_positions = cup_manipulator.get_joint_positions(plant, plant_context)
    print(colored(f"\nInitial Joint Positions:", "magenta", attrs=["bold"]))
    for name, pos in joint_positions.items():
        print(colored(f"  {name}: {np.rad2deg(pos):+7.2f}° ({pos:+.4f} rad)", "cyan"))
    
    # Interactive joint control
    print("\n" + "=" * 70)
    print("Interactive Joint Control")
    print("=" * 70)
    
    if pendulum:
        print(f"\nEnter joint positions (4 values in degrees, space-separated):")
        print(f"  Format: <link1_base> <link2_link1> <pendulum_pitch> <pendulum_roll>")
        print(f"  Example: 0 45 0 180  (manipulator at 45°, pendulum hanging down)")
        print(f"  Example: 30 60 20 170 (all joints moved)")
        joint_names = ["link1_base", "link2_link1", "pendulum_pitch", "pendulum_roll"]
        expected_count = 4
    else:
        print(f"\nEnter joint positions (2 values in degrees, space-separated):")
        print(f"  Format: <link1_base> <link2_link1>")
        print(f"  Example: 0 45")
        joint_names = ["link1_base", "link2_link1"]
        expected_count = 2
    
    print(f"  Type 'q' or 'quit' to exit")
    print("=" * 70 + "\n")
    
    # Interactive loop
    try:
        while True:
            # Prompt for input
            user_input = input(f"\nJoint angles (deg) [{', '.join(joint_names)}]: ").strip()
            
            # Check for exit
            if user_input.lower() in ["q", "quit", "exit"]:
                print("\nExiting interactive mode...")
                break
            
            # Parse input
            try:
                values = [float(x.strip()) for x in user_input.split()]
                
                if len(values) != expected_count:
                    print(colored(f"❌ Error: Expected {expected_count} values, got {len(values)}. Try again.", "red"))
                    continue
                
                # Convert degrees to radians
                angles_rad = [np.deg2rad(v) for v in values]
                
                # Display what we're about to set
                print(colored(f"\n→ Setting joints:", "yellow"))
                for joint_name, angle_deg, angle_rad in zip(joint_names, values, angles_rad):
                    print(colored(f"    {joint_name}: {angle_deg:+7.2f}° ({angle_rad:+.4f} rad)", "yellow"))
                
                # Update joint positions
                from pydrake.multibody.tree import RevoluteJoint
                for joint_name, angle in zip(joint_names, angles_rad):
                    try:
                        # Try to get joint from cup_manipulator first, then try without model instance
                        if joint_name in ["link1_base", "link2_link1"]:
                            joint = plant.GetJointByName(joint_name, cup_manipulator.model_instance)
                        else:
                            # Pendulum joints are not part of cup_manipulator model instance
                            joint = plant.GetJointByName(joint_name)
                        
                        if isinstance(joint, RevoluteJoint):
                            joint.set_angle(plant_context, angle)
                            print(colored(f"  ✓ Set {joint_name}", "green"))
                    except Exception as e:
                        print(colored(f"  ⚠ Warning: Could not set joint {joint_name}: {e}", "red"))
                        import traceback
                        traceback.print_exc()
                
                # Force publish to update Meshcat visualization
                diagram.ForcedPublish(context)
                
                # Get updated state
                joint_positions = cup_manipulator.get_joint_positions(plant, plant_context)
                
                # Display updated state (actual values read back from plant)
                print(colored(f"\n← Actual joint values (read from plant):", "cyan"))
                for name, pos in joint_positions.items():
                    print(colored(f"    {name}: {np.rad2deg(pos):+7.2f}° ({pos:+.4f} rad)", "cyan"))
                
                # Check for discrepancies
                print(colored(f"\n🔍 Verification (set vs. read):", "magenta"))
                for joint_name, set_value in zip(joint_names, values):
                    if joint_name in joint_positions:
                        read_value = np.rad2deg(joint_positions[joint_name])
                        diff = read_value - set_value
                        if abs(diff) > 0.01:  # More than 0.01° difference
                            print(colored(f"  ⚠ {joint_name}: set={set_value:+7.2f}° → read={read_value:+7.2f}° (Δ={diff:+.2f}°)", "yellow"))
                        else:
                            print(colored(f"  ✓ {joint_name}: {set_value:+7.2f}° (match)", "green"))
                
            except ValueError as e:
                print(colored(f"❌ Error: Invalid input. Please enter {expected_count} numbers separated by spaces.", "red"))
                print(f"   Example: {'0 45 0 180' if pendulum else '0 45'}")
            except Exception as e:
                print(colored(f"❌ Error: {e}", "red"))
                import traceback
                traceback.print_exc()
    
    except KeyboardInterrupt:
        print(colored("\n\n✓ Scene visualization closed by user", "green"))
    
    print(colored("\n" + "=" * 70, "green"))
    print(colored("Scene visualization complete!", "green", attrs=["bold"]))
    print(colored("=" * 70 + "\n", "green"))


# ============================================================================
# MANIPULATOR PUSHES CART SIMULATION
# ============================================================================

def run_manipulator_pushes_cart(config: ManipulatorPushesCartConfig):
    """
    Run simulation where manipulator pushes cart via direct impedance control.
    
    Architecture (from notes_ss_cart_pendulam_manipulator.tex):
        M_ref → x_ref → F_imp → Cart (direct)
                         ↓
                      -J^T F_imp → Manipulator (reaction)
    
    NO Virtual Mass! Direct force coupling.
    """
    from pydrake.all import (
        ExternallyAppliedSpatialForce, SpatialForce, AbstractValue,
        VectorLogSink, ConstantVectorSource
    )
    from pathlib import Path
    
    print(colored("\n" + "=" * 80, "cyan"))
    print(colored("MANIPULATOR PUSHES CART VIA IMPEDANCE CONTROL", "cyan", attrs=["bold"]))
    print(colored("(Architecture from notes_ss_cart_pendulam_manipulator.tex)", "cyan"))
    print(colored("=" * 80 + "\n", "cyan"))
    
    # Build scene and get components
    diagram, simulator, context, plant, manipulator, state_logger, visualizer, meshcat = build_scene(config)
    
    # ========================================
    # Run Simulation
    # ========================================
    print(colored(f"Simulating for {config.duration} s...", "yellow"))
    visualizer.StartRecording()
    simulator.AdvanceTo(config.duration)
    visualizer.PublishRecording()
    print(colored("✓ Simulation complete\n", "green"))
    print(colored(f"🎬 Animation: {meshcat.web_url()}\n", "green", attrs=["bold"]))
    
    # Extract and plot results
    state_log = state_logger.FindLog(context)
    time_data = state_log.sample_times()
    state_data = state_log.data()
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t = time_data
    
    # Joint angles
    axes[0, 0].plot(t, np.rad2deg(state_data[0, :]), 'b-', label='q₁')
    axes[0, 0].plot(t, np.rad2deg(state_data[1, :]), 'r-', label='q₂')
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Joint Angle [deg]')
    axes[0, 0].set_title('Manipulator Joints')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Cart X position
    axes[0, 1].plot(t, state_data[2, :], 'g-', label='Cart X')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('X Position [m]')
    axes[0, 1].set_title('Cart Motion (X-axis)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Cart Y position
    axes[1, 0].plot(t, state_data[3, :], 'c-', label='Cart Y')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Y Position [m]')
    axes[1, 0].set_title('Cart Motion (Y-axis)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Pendulum angles
    axes[1, 1].plot(t, np.rad2deg(state_data[4, :]), 'm-', label='Pitch α')
    axes[1, 1].plot(t, np.rad2deg(state_data[5, :]), 'y-', label='Roll β')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Angle [deg]')
    axes[1, 1].set_title('Pendulum Angles')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    cart_displacement = state_data[2, -1] - state_data[2, 0]
    print(colored(f"✓ Cart displacement: {cart_displacement:.3f} m", "green"))
    print(colored(f"✓ Target displacement: {config.distance:.3f} m", "green"))


def build_scene(config: ManipulatorPushesCartConfig):
    """
    Build the complete scene for manipulator-pushes-cart mode.
    
    Returns:
        tuple: (diagram, simulator, context, plant, manipulator, state_logger, visualizer, meshcat)
    """
    from pydrake.all import (
        ExternallyAppliedSpatialForce, SpatialForce, AbstractValue,
        VectorLogSink, ConstantVectorSource
    )
    from pathlib import Path
    
    # ========================================
    # Setup
    # ========================================
    meshcat = StartMeshcat()
    print(colored(f"🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))
    
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    
    # Load manipulator
    print(colored("Loading manipulator...", "yellow"))
    urdf_path = Path("model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf").absolute()
    cup_config = create_cup_manipulator_config(
        urdf_path=str(urdf_path),
        joint_angles=(np.deg2rad(config.q1_init), np.deg2rad(config.q2_init)),
        damping=(0.1, 0.1),
        stiffness=(0.0, 0.0),
        friction=(0.05, 0.05),
    )
    manipulator = CupManipulator(cup_config)
    parser = Parser(plant)
    manipulator.build_in_plant(plant, parser, weld_base=True)
    print(colored("✓ Manipulator loaded\n", "green"))
    
    # Load cart-pendulum
    print(colored("Creating cart-pendulum system...", "yellow"))
    cart_pendulum_config = create_cart_pendulum_config(
        cart_mass=config.cart_mass,
        cart_damping=config.cart_damping,
        pendulum_mass=config.pendulum_mass,
        pendulum_length=config.pendulum_length,
        pendulum_damping=config.pendulum_damping,
        initial_cart_x=0.0,  # Will be set to EE position
        initial_cart_y=0.0,
        initial_pitch=config.initial_pitch,
        initial_roll=config.initial_roll,
    )
    cart_pendulum = CartPendulum3D(cart_pendulum_config, visualize_cart=True, add_cart_actuators=True)
    cart_model_instance = plant.AddModelInstance("cart_pendulum")
    cart_pendulum.attach_to_plant(plant, cart_model_instance, register_visuals=True)
    print(colored("✓ Cart-Pendulum created\n", "green"))
    
    # Finalize plant
    plant.Finalize()
    
    # Get initial EE position
    temp_context = plant.CreateDefaultContext()
    plant.SetPositions(temp_context, manipulator.model_instance, 
                      np.array([np.deg2rad(config.q1_init), np.deg2rad(config.q2_init)]))
    ee_body = plant.GetBodyByName("link2", manipulator.model_instance)
    ee_pose = plant.EvalBodyPoseInWorld(temp_context, ee_body)
    ee_pos_init = ee_pose.translation() + ee_pose.rotation() @ manipulator.EE_OFFSET
    
    print(colored(f"Initial EE position: x={ee_pos_init[0]:.3f} m, y={ee_pos_init[1]:.3f} m\n", "cyan"))
    
    # ========================================
    # Build Control System
    # ========================================
    print(colored("Building control system...", "yellow"))
    
    # Create systems
    ee_kinematics = builder.AddSystem(EndEffectorKinematics(plant, manipulator))
    
    # Use 2D classes but configure for X-axis only motion (Y components will be zero)
    # This allows generalization: for 1D motion use X only, for 2D use both components
    
    # ZFT Reference Mass (2D, but Y-component unused for this mode)
    zft_config = ZFTReferenceMassConfig(
        Mh=np.diag([config.M_ref, config.M_ref]),  # 2×2 diagonal, but only X matters
        kp=np.diag([config.K_imp, config.K_imp]),  # Coupling gains
        kd=np.diag([config.D_imp, config.D_imp]),  # Coupling damping
        yref0=np.array([ee_pos_init[0], ee_pos_init[1]]),  # Initialize with EE position
        vref0=np.array([0.0, 0.0])  # Zero initial velocity
    )
    zft_ref_mass = builder.AddSystem(ZFTReferenceMass(zft_config))
    
    # Impedance Force (2D, but Y-component unused for this mode)
    imp_config = ImpedanceForceConfig(
        kp=np.diag([config.K_imp, config.K_imp]),  # 2×2 diagonal stiffness
        kd=np.diag([config.D_imp, config.D_imp]),  # 2×2 diagonal damping
        force_limit=None  # No saturation
    )
    impedance_force = builder.AddSystem(ImpedanceForce(imp_config))
    
    jacobian_controller = builder.AddSystem(ManipulatorJacobianTransposeController(plant, manipulator))
    imp_to_cart = builder.AddSystem(ImpedanceToCartForce())
    
    # Zero muscle force (no LQR for now) - 2D to match ZFT input
    zero_muscle_force = builder.AddSystem(ConstantVectorSource(np.zeros(2)))
    
    # State demultiplexers
    state_demux = builder.AddSystem(Demultiplexer([4, 8]))  # [manip(4), cart(8)]
    cart_demux = builder.AddSystem(Demultiplexer([2, 2, 2, 2]))  # [pos, angles, vel, ang_vel]
    
    # Impedance force demux to extract X-component only (for 1D control)
    imp_force_demux = builder.AddSystem(Demultiplexer([1, 1]))  # [F_x, F_y]
    
    # Create 4D vectors [y, v] from 2D components (for 2D classes)
    # [x, y, vx, vy] format required by ZFTReferenceMass and ImpedanceForce
    ee_state_mux = builder.AddSystem(Multiplexer([2, 2]))  # [pos(2), vel(2)] -> [y, v](4)
    
    # ========================================
    # Connect Systems
    # ========================================
    # Connect plant state
    builder.Connect(plant.get_state_output_port(), state_demux.get_input_port())
    
    # Connect manipulator state
    builder.Connect(state_demux.get_output_port(0), ee_kinematics.GetInputPort("manipulator_state"))
    builder.Connect(state_demux.get_output_port(0), jacobian_controller.GetInputPort("manipulator_state"))
    
    # Connect cart state
    builder.Connect(state_demux.get_output_port(1), cart_demux.get_input_port())
    
    # Create 4D EE state [x, y, vx, vy] for 2D systems
    builder.Connect(ee_kinematics.GetOutputPort("ee_position"), ee_state_mux.get_input_port(0))  # [x, y]
    builder.Connect(ee_kinematics.GetOutputPort("ee_velocity"), ee_state_mux.get_input_port(1))  # [vx, vy]
    
    # Connect to ZFT (2D)
    builder.Connect(ee_state_mux.get_output_port(0), zft_ref_mass.GetInputPort("y_v"))  # [x, y, vx, vy]
    builder.Connect(zero_muscle_force.get_output_port(0), zft_ref_mass.GetInputPort("F"))  # [0, 0]
    
    # Connect to impedance (2D)
    builder.Connect(ee_state_mux.get_output_port(0), impedance_force.GetInputPort("y_v"))  # [x, y, vx, vy]
    builder.Connect(zft_ref_mass.GetOutputPort("yref_vref"), impedance_force.GetInputPort("yref_vref"))  # [x_ref, y_ref, vx_ref, vy_ref]
    
    # Extract X-component of impedance force for 1D control
    builder.Connect(impedance_force.GetOutputPort("F_imp"), imp_force_demux.get_input_port())  # [F_x, F_y]
    
    # Connect to controllers (using X-component only)
    builder.Connect(imp_force_demux.get_output_port(0), jacobian_controller.GetInputPort("F_imp"))  # F_x only
    builder.Connect(imp_force_demux.get_output_port(0), imp_to_cart.GetInputPort("F_imp"))  # F_x only
    
    # Apply F_imp to cart
    class CartForceApplicator(LeafSystem):
        """Apply force to cart."""
        def __init__(self, cart_body_index):
            LeafSystem.__init__(self)
            self.cart_body_index = cart_body_index
            self.DeclareVectorInputPort("cart_force", BasicVector(2))
            self.DeclareAbstractOutputPort(
                "spatial_forces",
                lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
                self._calc_output
            )
        
        def _calc_output(self, context, output):
            cart_force_2d = self.GetInputPort("cart_force").Eval(context)
            
            spatial_force = ExternallyAppliedSpatialForce()
            spatial_force.body_index = self.cart_body_index
            spatial_force.F_Bq_W = SpatialForce(
                tau=np.zeros(3),
                f=np.array([cart_force_2d[0], cart_force_2d[1], 0.0])
            )
            spatial_force.p_BoBq_B = np.zeros(3)
            
            output.set_value([spatial_force])
    
    cart_force_applicator = builder.AddSystem(CartForceApplicator(cart_pendulum.cart_body.index()))
    builder.Connect(imp_to_cart.GetOutputPort("cart_force"), cart_force_applicator.GetInputPort("cart_force"))
    builder.Connect(cart_force_applicator.GetOutputPort("spatial_forces"), plant.get_applied_spatial_force_input_port())
    
    # Connect joint torques to manipulator (use model-instance-specific port)
    builder.Connect(jacobian_controller.get_output_port(), 
                   plant.get_actuation_input_port(manipulator.model_instance))
    
    # Visualization
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    meshcat.SetProperty("/Background", "visible", False)
    
    # Loggers
    state_logger = builder.AddSystem(VectorLogSink(plant.num_multibody_states()))
    builder.Connect(plant.get_state_output_port(), state_logger.get_input_port())
    
    # Build diagram
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Set initial state
    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant.SetPositions(plant_context, np.array([
        np.deg2rad(config.q1_init), np.deg2rad(config.q2_init),  # Manipulator
        ee_pos_init[0], ee_pos_init[1],  # Cart at EE
        0.0, 0.0  # Pendulum
    ]))
    plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
    
    print(colored("✓ System built\n", "green"))
    
    return diagram, simulator, context, plant, manipulator, state_logger, visualizer, meshcat


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print(colored("CUP MANIPULATOR WITH 3D PENDULUM - DEMO", 'cyan', attrs=['bold']))
    print("=" * 80 + "\n")
    print(colored(f"Mode: {args.mode}", "yellow", attrs=["bold"]))
    
    # Check if manipulator-pushes-cart mode
    if args.mode == 'manipulator-pushes-cart':
        config = ManipulatorPushesCartConfig(
            cart_mass=5.0,
            cart_damping=0.1,
            pendulum_mass=0.5,
            pendulum_length=0.2,
            pendulum_damping=0.1,
            K_imp=args.k_imp,
            D_imp=args.d_imp,
            M_ref=2.0,
            distance=args.distance,
            duration=args.duration,
            q1_init=-10.0,
            q2_init=20.0,
        )
        run_manipulator_pushes_cart(config)
        return
    
    # Create cup manipulator config and instance (for other modes)
    urdf_path = str(Path("model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj.urdf").absolute())
    cup_config = create_cup_manipulator_config(
        urdf_path=urdf_path,
        joint_angles=(0.0, 0.0),
        damping=(0.1, 0.1),
        stiffness=(0.0, 0.0),
        friction=(0.05, 0.05),
    )
    cup_manipulator = CupManipulator(cup_config)
    
    # Create pendulum instance
    pendulum_config = create_pendulum_config(
        mass=0.5,
        length=0.2,
        radius=0.05,
        damping=0.1,
        attachment_point=(-1.2545, 0.0, -0.188125),
        initial_pitch=0.0,
        initial_roll=180.0,
        name="pendulum"
    )
    pendulum = Pendulum3D(pendulum_config)
    
    # Create and build the cup manipulator system with dependency injection
    system = CupManipulatorWithPendulum(
        cup_manipulator=cup_manipulator,
        pendulum=pendulum,
        enable_visualization=True,
        initial_pendulum_pitch=0.0,
        initial_pendulum_roll=180.0,
    ).build()
    
    # Check mode
    if args.mode == 'scene-viz':
        # Interactive visualization mode
        system.run_scene_viz()
    else:
        # Run simulation
        system.run_simulation(duration=5.0)
    
    print()


if __name__ == "__main__":
    main()


