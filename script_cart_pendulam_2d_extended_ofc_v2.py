#!/usr/bin/env python3
"""
2D Cart-Pendulum with Muscle Dynamics & Optimal Feedback Control
Extended from 1D CartPendulum architecture

This script extends the CartPendulum class from script_cart_pendulum_muscle_dynamics_ofc.py
to 2D motion (x, y) instead of just 1D (x).

STATE VECTOR (14D):
==================
1. x       - Cart X position [m]
2. y       - Cart Y position [m]
3. α       - Pendulum pitch angle [rad]
4. β       - Pendulum roll angle [rad]
5. ẋ       - Cart X velocity [m/s]
6. ẏ       - Cart Y velocity [m/s]
7. α̇       - Pendulum pitch velocity [rad/s]
8. β̇       - Pendulum roll velocity [rad/s]
9. F_x     - Muscle force state X [N]
10. F_y    - Muscle force state Y [N]
11. x_ref  - ZFT reference X position [m]
12. y_ref  - ZFT reference Y position [m]
13. ẋ_ref  - ZFT reference X velocity [m/s]
14. ẏ_ref  - ZFT reference Y velocity [m/s]

CONTROL: u = [u_x, u_y] (2D neural command)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import sys
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from abc import ABC, abstractmethod
from termcolor import colored
from scipy.linalg import solve_discrete_are

# Drake imports
from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Simulator,
    VectorLogSink,
    LogVectorOutput,
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
    Parser,
    ZeroOrderHold,
    JacobianWrtVariable,
    InverseKinematics,
    Solve,
)
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import FixedOffsetFrame, RevoluteJoint, PrismaticJoint
from pydrake.math import RigidTransform, RollPitchYaw

from pydrake.multibody.tree import MultibodyForces

# Import from existing script
import sys
sys.path.append(str(Path(__file__).parent))
from robot_types import (
    create_cart_pendulum_config, 
    create_cup_manipulator_config,
    ManipulatorConfig
)

from scipy.linalg import solve_discrete_are

# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description='2D Cart-Pendulum with Muscle Dynamics & OFC')
parser.add_argument('--mode', type=str, 
                    choices=['scene-viz',
                             'lqr-applied-to-cart-manip-following-cart',
                             'lqr-applied-to-both-cart-manip'], 
                    # default='lqr-applied-to-cart-manip-following-cart',
                    default='lqr-applied-to-both-cart-manip',
                    # default='scene-viz',
                    help='Simulation mode')
parser.add_argument('--duration', type=float, default=10.0, help='Simulation duration [s]')
parser.add_argument('--target-x', type=float, default=-1, help='Target X position [m]')
parser.add_argument('--target-y', type=float, default=0.5, help='Target Y position [m]')
parser.add_argument('--cart-x-init', type=float, default=2, help='Initial cart X position [m] (default: use manipulator EE position)')
parser.add_argument('--cart-y-init', type=float, default=0.0, help='Initial cart Y position [m] (default: use manipulator EE position)')
parser.add_argument('--horizon', type=float, default=10.0, help='LQR horizon [s]')
parser.add_argument('--speed-scale', type=float, default=0.5, help='Trajectory speed scaling (0-1, lower=slower)')
# Parse and save our args FIRST
_parsed_args, _ = parser.parse_known_args()

# Temporarily clear sys.argv to prevent CupManipulator module from parsing our args
# (it has its own argparse that would fail on our mode choices)
import sys
_saved_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]  # Keep only script name

# Now import CupManipulator safely
# from script_cup_manipulator_controller_ofc import CupManipulator

# Restore sys.argv and our parsed args
sys.argv = _saved_argv
args = _parsed_args

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class CartPendulumPhysicsConfig:
    """Physical parameters for 2D cart-pendulum system."""
    mass_cart: float = 3.0
    mass_pendulum: float = 0.3
    length_pendulum: float = 0.5
    damping_cart: float = 0.5  # Add cart damping
    damping_pendulum: float = 0.1  # Add pendulum damping to reduce oscillations
    gravity: float = 9.81
    
    # Initial state [x, y, α, β] - if None, will be computed from manipulator EE
    cart_initial_position: Optional[np.ndarray] = None  # [x, y] in meters
    pendulum_initial_angles: Optional[np.ndarray] = None  # [α, β] in radians

@dataclass
class MuscleDynamicsConfig:
    """Muscle/actuator dynamics parameters (2D forces)."""
    muscle_tau: float = 0.03  # Time constant [s]
    initial_force: np.ndarray = None  # [F_x, F_y] initial state
    command_limit: float | None = None
    
    def __post_init__(self):
        if self.initial_force is None:
            self.initial_force = np.zeros(2)

@dataclass
class ImpedanceForceConfig:
    """Impedance control parameters (2D)."""
    K_imp: float = 50.0  # Stiffness [N/m]
    D_imp: float = 10.0  # Damping [N·s/m]

@dataclass
class ZFTReferenceMassConfig:
    """ZFT reference mass parameters (2D)."""
    M_ref: float = 1.0  # Reference mass [kg]
    K_imp: float = 50.0
    D_imp: float = 10.0
    initial_ref: np.ndarray = None  # [x_ref, y_ref, ẋ_ref, ẏ_ref]
    
    def __post_init__(self):
        if self.initial_ref is None:
            self.initial_ref = np.zeros(4)

@dataclass
class FiniteHorizonLQRConfig:
    """Finite-horizon LQR parameters for 14D system."""
    Q: np.ndarray = field(default_factory=lambda: np.diag([
        100.0, 100.0,    # Cart position (x, y)
        500.0, 500.0,    # Pendulum angles (α, β)
        10.0, 10.0,      # Cart velocities (ẋ, ẏ)
        100.0, 100.0,    # Pendulum angular velocities (α̇, β̇)
        0.1, 0.1,        # Muscle forces (F_x, F_y)
        1.0, 1.0,        # Reference position (x_ref, y_ref)
        0.1, 0.1,        # Reference velocity (ẋ_ref, ẏ_ref)
    ]))  # State cost (14×14)
    QN: Optional[np.ndarray] = field(default_factory=lambda: np.diag([
        200.0, 200.0,    # Cart position (2x)
        1000.0, 1000.0,  # Pendulum angles (2x)
        20.0, 20.0,      # Cart velocities (2x)
        200.0, 200.0,    # Pendulum angular velocities (2x)
        0.2, 0.2,        # Muscle forces (2x)
        2.0, 2.0,        # Reference position (2x)
        0.2, 0.2,        # Reference velocity (2x)
    ]))  # Terminal cost (2x Q for better convergence)
    R: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0]))  # Control cost (2×2)
    x_goal: np.ndarray = field(default_factory=lambda: np.zeros(14))  # Goal state (14D)
    # 14D state: [x, y, α, β, ẋ, ẏ, α̇, β̇, F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref]
    horizon: float = 10.0  # LQR horizon [s]
    timestep: float = 0.01  # Discretization timestep [s]
    u_limits: Optional[tuple] = None  # (u_min, u_max) for saturation

@dataclass
class SimulationConfig:
    """
    Configuration for simulation execution.
    
    Consolidates all parameters needed to set up and run a simulation,
    including physics, control, and visualization settings.
    """
    # System configs
    physics_config: CartPendulumPhysicsConfig
    muscle_config: MuscleDynamicsConfig
    impedance_config: ImpedanceForceConfig
    zft_config: ZFTReferenceMassConfig
    
    # Manipulator setup
    manipulator_urdf_path: str
    manipulator_joint_angles: Dict[str, float]
    manipulator_damping: tuple = (0.1, 0.1)
    
    # Simulation parameters
    target_x: float = 0.0
    target_y: float = 0.5
    duration: float = 10.0
    horizon: float = 10.0
    
    # Visualization
    meshcat: Optional[object] = None
    
    @classmethod
    def from_args(cls, args, physics_config, muscle_config, impedance_config, zft_config, meshcat):
        """
        Create SimulationConfig from command-line arguments and existing configs.
        
        Args:
            args: argparse.Namespace with target_x, target_y, duration, horizon
            physics_config: CartPendulumPhysicsConfig instance
            muscle_config: MuscleDynamicsConfig instance
            impedance_config: ImpedanceForceConfig instance
            zft_config: ZFTReferenceMassConfig instance
            meshcat: Meshcat instance
            
        Returns:
            SimulationConfig instance
        """
        # Extract joint angles and damping from ManipulatorConfig
        joint_angles_dict = MANIPULATOR_CONFIG.get_joint_positions_dict()
        damping_tuple = tuple(
            MANIPULATOR_CONFIG.joint_configs[jt].damping 
            for jt in ['link1_base', 'link2_link1']
        )
        
        return cls(
            physics_config=physics_config,
            muscle_config=muscle_config,
            impedance_config=impedance_config,
            zft_config=zft_config,
            manipulator_urdf_path=MANIPULATOR_CONFIG.urdf_path,
            manipulator_joint_angles=joint_angles_dict,
            manipulator_damping=damping_tuple,
            target_x=args.target_x,
            target_y=args.target_y,
            duration=args.duration,
            horizon=args.horizon,
            meshcat=meshcat,
        )

# Factory functions
def create_physics_config(**kwargs):
    return CartPendulumPhysicsConfig(**kwargs)

def create_muscle_config(**kwargs):
    return MuscleDynamicsConfig(**kwargs)

def create_impedance_config(**kwargs):
    return ImpedanceForceConfig(**kwargs)

def create_zft_config(**kwargs):
    return ZFTReferenceMassConfig(**kwargs)

def create_lqr_config(**kwargs):
    return FiniteHorizonLQRConfig(**kwargs)

# Global configurations
PHYSICS_CONFIG = create_physics_config()
MUSCLE_CONFIG = create_muscle_config()
IMPEDANCE_CONFIG = create_impedance_config()
ZFT_CONFIG = create_zft_config()
LQR_CONFIG = create_lqr_config(
    x_goal=np.array([args.target_x, args.target_y, 0, 0, 0, 0, 0, 0, 0, 0, args.target_x, args.target_y, 0, 0]),
    horizon=args.horizon
)
MANIPULATOR_CONFIG = create_cup_manipulator_config(
    urdf_path="model_using_onshape_to_robot/cup_manipulator2/cup_manipulator_obj_right_frame.urdf",
    joint_angles={
        'link1_base': np.deg2rad(0.0),   # q1: Base to link1
        'link2_link1': np.deg2rad(20.0), # q2: Link1 to link2
    },
    damping=(0.1, 0.1),
)
SIMULATION_CONFIG = SimulationConfig.from_args(args, PHYSICS_CONFIG, MUSCLE_CONFIG, IMPEDANCE_CONFIG, ZFT_CONFIG, None)

# Set welding mode based on selected mode
WELD_CART_TO_MANIP_EE = True if args.mode == 'lqr-applied-to-both-cart-manip' else False      

# ============================================================================
# ROBOT BASE CLASS (ABSTRACT)
# ============================================================================

class RobotBase(ABC):
    """
    Abstract Base Class for Robots using Drake
    
    DESIGN PATTERN: Template Method Pattern
    Provides common interface for all robots
    """
    
    def __init__(self, config: ManipulatorConfig, name: Optional[str] = None):
        """Initialize robot with configuration."""
        self.config = config
        self.name = name or config.name
        self.model_instance: Optional[int] = None
        self.dof_names: List[str] = []
    
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
        
        # Auto-detect joint names from URDF (excluding weld joints)
        # CRITICAL: Identify joints by their connectivity, NOT parse order
        # Drake's GetJointIndices() returns in URDF parse order, which varies between files
        revolute_joints_info = []
        for joint_idx in plant.GetJointIndices(model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0 and "weld" not in joint.name().lower():
                revolute_joints_info.append({
                    'name': joint.name(),
                    'parent': joint.parent_body().name(),
                    'child': joint.child_body().name()
                })
        
        if len(revolute_joints_info) == 2:
            # Identify q1 (base→link1) and q2 (link1→link2) by checking parent/child bodies
            jt1_name, jt2_name = None, None
            for jt_info in revolute_joints_info:
                # q1 connects base to link1
                if 'base' in jt_info['parent'].lower() and 'link1' in jt_info['child'].lower():
                    jt1_name = jt_info['name']
                # q2 connects link1 to link2
                elif 'link1' in jt_info['parent'].lower() and 'link2' in jt_info['child'].lower():
                    jt2_name = jt_info['name']
            
            if jt1_name and jt2_name:
                self.JT1_NAME = jt1_name  # Joint from base to link1 (q1)
                self.JT2_NAME = jt2_name  # Joint from link1 to link2 (q2)
                self.ACT1_NAME = f"tau_{self.JT1_NAME}"
                self.ACT2_NAME = f"tau_{self.JT2_NAME}"
                self.joint_names = [self.JT1_NAME, self.JT2_NAME]
                print(colored(f"✓ Auto-detected joint names by connectivity:", 'green'))
                print(colored(f"  JT1 (q1, base→link1): {self.JT1_NAME}", 'cyan'))
                print(colored(f"  JT2 (q2, link1→link2): {self.JT2_NAME}", 'cyan'))
            else:
                print(colored(f"⚠️  Could not identify joints by connectivity. Using default names.", 'yellow'))
        else:
            print(colored(f"⚠️  Expected 2 revolute joints, found {len(revolute_joints_info)}", 'yellow'))
        
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
        """Set joint properties (damping, friction) BEFORE plant is finalized."""
        print(colored(f"\nSetting joint properties for '{self.name}':", 'yellow'))
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_mutable_joint(joint_idx)
            joint_name = joint.name()
            
            if joint.num_velocities() > 0 and joint_name in self.config.joint_configs:
                config = self.config.joint_configs[joint_name]
                
                if hasattr(joint, 'set_default_damping_vector') and config.damping > 0:
                    joint.set_default_damping_vector([config.damping])
                    print(colored(f"  ✓ {joint_name}: damping={config.damping}", 'cyan'))
                else:
                    print(colored(f"  ✓ {joint_name}: damping=0.0 (default)", 'cyan'))
        print(colored(f"✓ Joint properties configured", 'green'))
    
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


# ============================================================================
# CUP MANIPULATOR CLASS
# ============================================================================

class CupManipulator(RobotBase):
    """
    Cup Manipulator for Drake with controller integration.

    CRITICAL: Drake joint ordering is [link2_link1, link1_base] = [q2, q1]
    This class handles all conversions internally, so external code can use [q1, q2].

    NOTE: For cup_manipulator_obj_right_frame.urdf, the coordinate frame is already 
    aligned correctly (no -90° Y rotation needed), so EE offsets use positive X/Z.

    Manages:
    - URDF loading and joint configuration
    - State queries (positions, velocities)
    - End-effector kinematics
    - Automatic conversion between user [q1, q2] and Drake [q2, q1] ordering
    """

    # --- Cup-center EE pose relative to link2, from URDF simple_ball (cup middle) origin ---
    # For cup_manipulator_obj_natural_order.urdf: actual joint names from URDF
    # URDF joint names: link1_base, link2_link1
    # Note: Joint and link names are auto-detected from URDF in load_urdf_to_plant()
    JT1_NAME = "link1_base"  # q1: base to link1 (will be overridden by auto-detection)
    JT2_NAME = "link2_link1"  # q2: link1 to link2 (will be overridden by auto-detection)
    ACT1_NAME = f"tau_{JT1_NAME}"
    ACT2_NAME = f"tau_{JT2_NAME}"
    LINK2_NAME = "link2"  # End-effector attachment link (consistent across URDFs)

    EE_XYZ_LINK2 = np.array([1.2515, 0.0, 0.15])   # meters, when q1 = q2 = 0 
    EE_XYZ_LINK1 = np.array([2.2045,0,0.071875])             # meters, when q1 = 0
    EE_XYZ_BASE = np.array([2.2045,0,1.248125])            # meters (no offset)
    EE_RPY_LINK2 = np.array([0.0, 0.0, 0.0])       # radians (no rotation)when q1 = q2 = 0 
    EE_RPY_LINK1 = np.array([0.0, 0.0, 0.0])       # radians (no rotation)when q1 = 0 
    EE_RPY_BASE = np.array([0.0, 0.0, 0.0])       # radians (no rotation)when q1 = q2 = 0 
    EE_FRAME_NAME = "cup_center"  # the canonical EE frame name inside Drake 
    
    # Alias for end-effector offset (offset from link2 frame to cup center)
    # Used by IK and controllers - points to the same location as the cup_center frame
    EE_OFFSET = EE_XYZ_LINK2  # [1.2515, 0.0, 0.15] meters from link2 origin

    def __init__(self, config: ManipulatorConfig, enable_visualization: bool = True):
        super().__init__(config)
        self.joint_names = [self.JT1_NAME, self.JT2_NAME]
        self.actuator_names: List[str] = []
        self.enable_visualization = enable_visualization

    # ------------------------------------------------------------------
    # ADD END EFFECTOR FRAME (call this BEFORE plant.Finalize())
    # ------------------------------------------------------------------
    def add_end_effector_frame(self, plant: MultibodyPlant):
        """
        Creates a Drake frame at the simple_ball's location (cup center).
        
        The URDF's simple_ball visual element is at offset [1.2515, 0, 0.15] from link2,
        but Drake doesn't auto-create frames for visual elements. This method explicitly
        creates a named frame at that exact ball location for IK and kinematics queries.

        Must be called AFTER the model is added (self.model_instance is valid)
        and BEFORE plant.Finalize().

        Returns:
            The created Frame (FixedOffsetFrame) at the ball's location
        """
        if plant.is_finalized():
            raise RuntimeError("Cannot add EE frame after plant is finalized")

        link2_body = plant.GetBodyByName(self.LINK2_NAME, self.model_instance)

        X_L2_EE = RigidTransform(
            RollPitchYaw(self.EE_RPY_LINK2),
            self.EE_XYZ_LINK2
        )

        # Avoid double-adding if called multiple times
        try:
            existing = plant.GetFrameByName(self.EE_FRAME_NAME, self.model_instance)
            return existing
        except Exception:
            pass

        ee_frame = plant.AddFrame(
            FixedOffsetFrame(
                self.EE_FRAME_NAME,
                link2_body.body_frame(),
                X_L2_EE,
                self.model_instance
            )
        )
        return ee_frame

    # Convenience accessor
    def get_end_effector_frame(self, plant: MultibodyPlant):
        """Get the Drake frame object for the end effector (cup center)."""
        return plant.GetFrameByName(self.EE_FRAME_NAME, self.model_instance)
    
    # ------------------------------------------------------------------
    # ADD JOINT ACTUATORS (call this BEFORE plant.Finalize())
    # ------------------------------------------------------------------
    def add_joint_actuators(self, plant: MultibodyPlant):
        """
        Add actuators to manipulator joints using explicit joint names.
        
        Must be called AFTER the model is added (self.model_instance is valid)
        and BEFORE plant.Finalize().
        
        This allows torques to be applied to the manipulator joints.
        """
        if plant.is_finalized():
            raise RuntimeError("Cannot add actuators after plant is finalized")
        
        # Add actuators using explicit joint names
        jt1 = self.get_joint_by_name(plant, self.JT1_NAME)
        jt2 = self.get_joint_by_name(plant, self.JT2_NAME)
        plant.AddJointActuator(self.ACT1_NAME, jt1)
        plant.AddJointActuator(self.ACT2_NAME, jt2)
        self.actuator_names = [self.ACT1_NAME, self.ACT2_NAME]
        print(colored(f"✓ Added actuators: {self.ACT1_NAME}, {self.ACT2_NAME}", 'green'))
    
    # ------------------------------------------------------------------
    # EE QUERIES (fixed to use cup_center frame)
    # ------------------------------------------------------------------
    def get_end_effector_position(self, plant: MultibodyPlant, context) -> np.ndarray:
        """World position of the cup middle (end effector)."""
        ee_frame = self.get_end_effector_frame(plant)
        X_WE = plant.CalcRelativeTransform(context, plant.world_frame(), ee_frame)
        return X_WE.translation()

    def CalcPosition(self, plant: MultibodyPlant, context) -> np.ndarray:
        """Same as get_end_effector_position(), kept for compatibility."""
        return self.get_end_effector_position(plant, context)

    # ------------------------------------------------------------------
    # STATE HELPERS (unchanged)
    # ------------------------------------------------------------------
    def get_state_from_plant(self, plant: MultibodyPlant, context) -> np.ndarray:
        return plant.GetPositionsAndVelocities(context, self.model_instance)

    def set_state_in_plant(self, plant: MultibodyPlant, context, user_state: np.ndarray):
        """Set full state in user order [q1, q2, q1_dot, q2_dot]."""
        q1, q2, q1_dot, q2_dot = user_state
        # Use new unified methods with Drake ordering [JT1=q2, JT2=q1]
        self.set_jt([self.JT1_NAME, self.JT2_NAME], plant, context, [q2, q1])
        self.set_jt_velocity([self.JT1_NAME, self.JT2_NAME], plant, context, [q2_dot, q1_dot])

    def get_positions_user_order(self, plant: MultibodyPlant, context) -> np.ndarray:
        """Get positions in user order [q1, q2]."""
        # Get in Drake order [q2, q1], then reverse to user order
        drake_positions = self.get_jt([self.JT1_NAME, self.JT2_NAME], plant, context)
        return np.array([drake_positions[1], drake_positions[0]])  # [q1, q2]

    def set_positions_user_order(self, plant: MultibodyPlant, context, user_positions):
        """Set positions by joint name using a dictionary.
        
        Args:
            user_positions: Dict[str, float] mapping joint names to angles, e.g.
                           {'link1_base': 0.0, 'link2_link1': 0.349}
                           OR np.ndarray [q1, q2] for backward compatibility
        """
        if isinstance(user_positions, dict):
            # Use dict directly - explicit and unambiguous
            for joint_name, angle in user_positions.items():
                self.set_jt([joint_name], plant, context, [angle])
        else:
            # Backward compatibility: array [q1, q2]
            q1, q2 = user_positions
            self.set_jt([self.JT1_NAME, self.JT2_NAME], plant, context, [q1, q2])

    def get_velocities_user_order(self, plant: MultibodyPlant, context) -> np.ndarray:
        """Get velocities in user order [q1_dot, q2_dot]."""
        # JT1_NAME="link1_base", JT2_NAME="link2_link1" - returns [q1_dot, q2_dot]
        return self.get_jt_velocity([self.JT1_NAME, self.JT2_NAME], plant, context)

    def set_velocities_user_order(self, plant: MultibodyPlant, context, user_velocities):
        """Set velocities by joint name using a dictionary.
        
        Args:
            user_velocities: Dict[str, float] mapping joint names to velocities, e.g.
                            {'link1_base': 0.0, 'link2_link1': 0.1}
                            OR np.ndarray [q1_dot, q2_dot] for backward compatibility
        """
        if isinstance(user_velocities, dict):
            # Use dict directly - explicit and unambiguous
            for joint_name, velocity in user_velocities.items():
                self.set_jt_velocity([joint_name], plant, context, [velocity])
        else:
            # Backward compatibility: array [q1_dot, q2_dot]
            q1_dot, q2_dot = user_velocities
            self.set_jt_velocity([self.JT1_NAME, self.JT2_NAME], plant, context, [q1_dot, q2_dot])

    def get_joint_positions(self, plant: MultibodyPlant, context):
        positions = {}
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0:
                if isinstance(joint, RevoluteJoint):
                    positions[joint.name()] = joint.get_angle(context)
                elif isinstance(joint, PrismaticJoint):
                    positions[joint.name()] = joint.get_translation(context)
        return positions

    def get_joint_velocities(self, plant: MultibodyPlant, context):
        velocities = {}
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0:
                if isinstance(joint, RevoluteJoint):
                    velocities[joint.name()] = joint.get_angular_rate(context)
                elif isinstance(joint, PrismaticJoint):
                    velocities[joint.name()] = joint.get_translation_rate(context)
        return velocities

    # ------------------------------------------------------------------
    # JOINT-SPECIFIC HELPERS (using JT1_NAME and JT2_NAME)
    # ------------------------------------------------------------------
    def get_joint_by_name(self, plant: MultibodyPlant, joint_name: str):
        """Get joint object by name."""
        return plant.GetJointByName(joint_name, self.model_instance)
    
    def get_jt(self, joint_name: str | List[str], plant: MultibodyPlant, context):
        """
        Get joint angle(s) [rad] for one or more joints by name.
        
        Args:
            joint_name: Single joint name or list of joint names
                       (e.g., self.JT1_NAME or [self.JT1_NAME, self.JT2_NAME])
            plant: Drake MultibodyPlant
            context: Drake context
            
        Returns:
            float if single joint name provided
            np.ndarray if list of joint names provided
        """
        if isinstance(joint_name, list):
            return np.array([self.get_joint_by_name(plant, name).get_angle(context) 
                           for name in joint_name])
        else:
            joint = self.get_joint_by_name(plant, joint_name)
            return joint.get_angle(context)
    
    def set_jt(self, joint_name: str | List[str], plant: MultibodyPlant, context, 
               angle: float | np.ndarray | List[float]):
        """
        Set joint angle(s) [rad] for one or more joints by name.
        
        Args:
            joint_name: Single joint name or list of joint names
            plant: Drake MultibodyPlant
            context: Drake context
            angle: Single angle or array/list of angles (must match joint_name length)
        """
        if isinstance(joint_name, list):
            angles = np.atleast_1d(angle)
            if len(angles) != len(joint_name):
                raise ValueError(f"Number of angles ({len(angles)}) must match "
                               f"number of joints ({len(joint_name)})")
            for name, ang in zip(joint_name, angles):
                self.get_joint_by_name(plant, name).set_angle(context, float(ang))
        else:
            joint = self.get_joint_by_name(plant, joint_name)
            joint.set_angle(context, float(angle))
    
    def get_jt_velocity(self, joint_name: str | List[str], plant: MultibodyPlant, context):
        """
        Get joint angular velocity(ies) [rad/s] for one or more joints by name.
        
        Args:
            joint_name: Single joint name or list of joint names
            plant: Drake MultibodyPlant
            context: Drake context
            
        Returns:
            float if single joint name provided
            np.ndarray if list of joint names provided
        """
        if isinstance(joint_name, list):
            return np.array([self.get_joint_by_name(plant, name).get_angular_rate(context) 
                           for name in joint_name])
        else:
            joint = self.get_joint_by_name(plant, joint_name)
            return joint.get_angular_rate(context)
    
    def set_jt_velocity(self, joint_name: str | List[str], plant: MultibodyPlant, context, 
                       velocity: float | np.ndarray | List[float]):
        """
        Set joint angular velocity(ies) [rad/s] for one or more joints by name.
        
        Args:
            joint_name: Single joint name or list of joint names
            plant: Drake MultibodyPlant
            context: Drake context
            velocity: Single velocity or array/list of velocities (must match joint_name length)
        """
        if isinstance(joint_name, list):
            velocities = np.atleast_1d(velocity)
            if len(velocities) != len(joint_name):
                raise ValueError(f"Number of velocities ({len(velocities)}) must match "
                               f"number of joints ({len(joint_name)})")
            for name, vel in zip(joint_name, velocities):
                self.get_joint_by_name(plant, name).set_angular_rate(context, float(vel))
        else:
            joint = self.get_joint_by_name(plant, joint_name)
            joint.set_angular_rate(context, float(velocity))
    
    # ------------------------------------------------------------------
    # INVERSE KINEMATICS
    # ------------------------------------------------------------------
    def solve_initial_pose_via_ik(
        self,
        plant,
        target_xy,
        q_seed,
        pos_tol=1e-3,
        verbose=False,
        ee_frame_name=None,
        target_z=None,
    ):
        """
        Solve for joint angles [q1, q2] (USER order) that place the EE at target (x, y, z).

        Args:
            plant: MultibodyPlant containing this manipulator
            target_xy: Target [x, y] position for end-effector
            q_seed: Initial guess for joint angles [q1, q2] in user order
            pos_tol: Position tolerance [m] for IK constraint
            verbose: Print detailed solver information
            ee_frame_name: Name of EE frame (defaults to self.EE_FRAME_NAME)
            target_z: Target Z coordinate (if None, uses seed configuration's Z)

        Returns:
            q_sol_user: Solution joint angles [q1, q2] in user order
            success: Boolean indicating if IK succeeded
        """
        from pydrake.multibody.inverse_kinematics import InverseKinematics
        from pydrake.solvers import Solve
        
        target_xy = np.asarray(target_xy).reshape(2,)
        q_seed = np.asarray(q_seed).reshape(2,)

        # Create IK and use its internal context for FK evaluations
        ik = InverseKinematics(plant)
        ik_context = ik.context()

        # Put IK context at seed configuration (so we can read seed z)
        self.set_positions_user_order(plant, ik_context, q_seed)

        world = plant.world_frame()

        # Prefer a named EE frame you added (e.g., "cup_center")
        if ee_frame_name is None:
            ee_frame_name = self.EE_FRAME_NAME

        if ee_frame_name is not None:
            try:
                ee_frame = plant.GetFrameByName(ee_frame_name, self.model_instance)
                # constrain the ORIGIN of the EE frame
                p_BQ = np.zeros(3)
            except:
                # Fallback: constrain a point on link2 body frame using EE_XYZ_LINK2 offset
                link2_body = plant.GetBodyByName("link2", self.model_instance)
                ee_frame = link2_body.body_frame()
                p_BQ = np.asarray(self.EE_XYZ_LINK2).reshape(3,)
        else:
            # Fallback: constrain a point on link2 body frame using EE_XYZ_LINK2 offset
            link2_body = plant.GetBodyByName("link2", self.model_instance)
            ee_frame = link2_body.body_frame()
            p_BQ = np.asarray(self.EE_XYZ_LINK2).reshape(3,)

        # Compute seed EE position (to pick z if not specified)
        ee_pos_seed = plant.CalcPointsPositions(
            ik_context,
            ee_frame,
            p_BQ.reshape(3, 1),
            world,
        ).ravel()
        z_target = target_z if target_z is not None else ee_pos_seed[2]

        if verbose:
            print(f"  Seed EE position: ({ee_pos_seed[0]:.3f}, {ee_pos_seed[1]:.3f}, {ee_pos_seed[2]:.3f})")
            print(f"  Target: ({target_xy[0]:.3f}, {target_xy[1]:.3f}, {z_target:.3f})")
            print(f"  Tolerance: ±{pos_tol:.6f} m")

        # Position constraint in world
        lower = np.array([target_xy[0], target_xy[1], z_target]) - pos_tol
        upper = np.array([target_xy[0], target_xy[1], z_target]) + pos_tol
        ik.AddPositionConstraint(
            frameB=ee_frame,
            p_BQ=p_BQ,
            frameA=world,
            p_AQ_lower=lower,
            p_AQ_upper=upper,
        )

        prog = ik.prog()
        q_vars = ik.q()

        # Add cost: stay near seed configuration with STRONG weight
        q0_all = plant.GetPositions(ik_context)  # All positions in the plant
        
        # Use much higher weight (1000x) to strongly prefer staying near current config
        # This prevents jumping between different IK solutions (elbow-up vs elbow-down)
        weight_matrix = 1000.0 * np.eye(len(q0_all))
        prog.AddQuadraticErrorCost(weight_matrix, q0_all, q_vars)
        prog.SetInitialGuess(q_vars, q0_all)

        result = Solve(prog)

        if verbose:
            print(f"  IK solver status: {result.get_solver_id().name()}")
            print(f"  Success: {result.is_success()}")
            if not result.is_success():
                print(f"  Solver details: {result.get_solution_result()}")

        if not result.is_success():
            return q_seed, False

        # Extract the solution
        q_sol_all = result.GetSolution(q_vars)
        
        # Create a temporary context to extract the manipulator-specific positions
        temp_context = plant.CreateDefaultContext()
        plant.SetPositions(temp_context, q_sol_all)
        
        # Get manipulator positions in Drake order [q2, q1]
        q_sol_drake = plant.GetPositions(temp_context, self.model_instance)
        
        # Convert from Drake order [q2, q1] to user order [q1, q2]
        q_sol_user = np.array([q_sol_drake[1], q_sol_drake[0]])
        
        if verbose:
            print(f"  Solution (user): q1={np.rad2deg(q_sol_user[0]):.2f}°, q2={np.rad2deg(q_sol_user[1]):.2f}°")

        return q_sol_user, True

    # ------------------------------------------------------------------
    # WELD BASE (unchanged; but note you want the BODY frame, not a frame named as the link)
    # ------------------------------------------------------------------
    def weld_base_to_world(
        self,
        plant: MultibodyPlant,
        position: np.ndarray = np.array([0.0, 0.0, 0.0]),
        orientation: np.ndarray = np.array([0.0, 0.0, 0.0])
    ):
        if plant.is_finalized():
            raise RuntimeError("Cannot weld base after plant is finalized")

        world_frame = plant.world_frame()

        # Better: weld the base BODY frame (robust) rather than GetFrameByName on a link name
        base_body = plant.GetBodyByName("base_mount_manipulator", self.model_instance)
        base_frame = base_body.body_frame()

        X_WB = RigidTransform(RollPitchYaw(orientation), position)
        plant.WeldFrames(world_frame, base_frame, X_WB)



# ============================================================================
# EXTENDED 2D CART-PENDULUM CLASS
# ============================================================================
class CartPendulum2DExtended:
    """
    Extended CartPendulum class for 2D motion.
    
    Extends the architecture from script_cart_pendulum_muscle_dynamics_ofc.py
    to support 2D cart motion (x, y) with 3D pendulum (pitch, roll).
    
    STATE: [x, y, α, β, ẋ, ẏ, α̇, β̇] (8D)
    INPUT: [F_x, F_y] (2D force)
    
    Structure:
    - world → x_slider (prismatic X) → y_slider (prismatic Y) → cart → pendulum (gimbal)
    """
    
    def __init__(self, config: CartPendulumPhysicsConfig, z_offset: float = 0.0):
        """
        Initialize 2D cart-pendulum system.
        
        Args:
            config: Physical parameters
            z_offset: Vertical offset for cart base [m]
        """
        self.config = config
        self.z_offset = z_offset
        
        # Will be populated during build
        self.cart_body = None
        self.x_slider_body = None
        self.y_slider_body = None
        self.x_joint = None
        self.y_joint = None
        self.pitch_joint = None
        self.roll_joint = None
        self.pendulum_body = None
        self.pitch_body = None
    
    def build_plant(self, plant: MultibodyPlant, model_instance, register_visuals: bool = True) -> None:
        """
        Build 2D cart-pendulum in the given plant.
        
        Similar to CartPendulumSystemDynamics.build_plant() but extended to 2D.
        
        Args:
            plant: MultibodyPlant to add bodies to
            model_instance: Model instance index
            register_visuals: Whether to register visual geometry (requires SceneGraph)
        """
        # ====================================================================
        # CREATE CART BODY
        # ====================================================================
        cart_size = 0.1
        m_c = self.config.mass_cart
        
        # Cart inertia (box approximation)
        I_c = (1/12) * m_c * cart_size**2
        cart_inertia = SpatialInertia(
            mass=m_c,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(I_c, I_c, I_c)
        )
        
        self.cart_body = plant.AddRigidBody(
            "cart",
            model_instance,
            cart_inertia
        )
        
        # Cart visual geometry (only if SceneGraph is registered)
        if register_visuals:
            plant.RegisterVisualGeometry(
                self.cart_body,
                RigidTransform(),
                Sphere(cart_size / 2),
                "cart_visual",
                np.array([0.3, 0.3, 0.8, 1.0])
            )
        
        # ====================================================================
        # CREATE SLIDER BODIES (for 2D motion)
        # ====================================================================
        slider_inertia = SpatialInertia(
            mass=0.001,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(1e-6, 1e-6, 1e-6)
        )
        
        self.x_slider_body = plant.AddRigidBody(
            "x_slider",
            model_instance,
            slider_inertia
        )
        
        self.y_slider_body = plant.AddRigidBody(
            "y_slider",
            model_instance,
            slider_inertia
        )
        
        # ====================================================================
        # CREATE PRISMATIC JOINTS FOR 2D MOTION
        # ====================================================================
        # Create offset base if z_offset is specified
        if abs(self.z_offset) > 1e-6:
            offset_body = plant.AddRigidBody(
                "base_offset",
                model_instance,
                slider_inertia
            )
            plant.WeldFrames(
                plant.world_frame(),
                offset_body.body_frame(),
                RigidTransform([0.0, 0.0, self.z_offset])
            )
            parent_frame = offset_body.body_frame()
        else:
            parent_frame = plant.world_frame()
        
        # X-axis joint
        self.x_joint = plant.AddJoint(
            PrismaticJoint(
                name="cart_x",
                frame_on_parent=parent_frame,
                frame_on_child=self.x_slider_body.body_frame(),
                axis=[1.0, 0.0, 0.0],
                damping=self.config.damping_cart
            )
        )
        
        # Y-axis joint
        self.y_joint = plant.AddJoint(
            PrismaticJoint(
                name="cart_y",
                frame_on_parent=self.x_slider_body.body_frame(),
                frame_on_child=self.y_slider_body.body_frame(),
                axis=[0.0, 1.0, 0.0],
                damping=self.config.damping_cart
            )
        )
        
        # Connect y_slider to cart (fixed)
        plant.WeldFrames(
            self.y_slider_body.body_frame(),
            self.cart_body.body_frame(),
            RigidTransform()
        )
        
        # Add actuators
        plant.AddJointActuator("force_x", self.x_joint)
        plant.AddJointActuator("force_y", self.y_joint)
        
        # ====================================================================
        # CREATE PENDULUM (GIMBAL MOUNT)
        # ====================================================================
        m_p = self.config.mass_pendulum
        L = self.config.length_pendulum
        r = 0.05
        
        # Pitch body (intermediate gimbal)
        pitch_inertia = SpatialInertia(
            mass=0.001,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(1e-6, 1e-6, 1e-6)
        )
        self.pitch_body = plant.AddRigidBody(
            "pitch_gimbal",
            model_instance,
            pitch_inertia
        )
        
        # Pendulum body (ball at end)
        I_p = (2/5) * m_p * r**2 + m_p * L**2
        pendulum_inertia = SpatialInertia(
            mass=m_p,
            p_PScm_E=[0, 0, -L],
            G_SP_E=UnitInertia(I_p / m_p, I_p / m_p, (2/5) * r**2)
        )
        self.pendulum_body = plant.AddRigidBody(
            "pendulum",
            model_instance,
            pendulum_inertia
        )
        
        # Pitch joint (rotation about Y-axis)
        self.pitch_joint = plant.AddJoint(
            RevoluteJoint(
                name="pendulum_pitch",
                frame_on_parent=self.cart_body.body_frame(),
                frame_on_child=self.pitch_body.body_frame(),
                axis=[0, 1, 0],
                damping=self.config.damping_pendulum
            )
        )
        
        # Roll joint (rotation about X-axis)
        self.roll_joint = plant.AddJoint(
            RevoluteJoint(
                name="pendulum_roll",
                frame_on_parent=self.pitch_body.body_frame(),
                frame_on_child=self.pendulum_body.body_frame(),
                axis=[1, 0, 0],
                damping=self.config.damping_pendulum
            )
        )
        
        # Pendulum visual geometry (only if SceneGraph is registered)
        if register_visuals:
            # Rod
            plant.RegisterVisualGeometry(
                self.pendulum_body,
                RigidTransform([0, 0, -L/2]),
                Cylinder(radius=0.01, length=L),
                "pendulum_rod",
                np.array([0.6, 0.4, 0.2, 1.0])
            )
            
            # Ball
            plant.RegisterVisualGeometry(
                self.pendulum_body,
                RigidTransform([0, 0, -L]),
                Sphere(r),
                "pendulum_ball",
                np.array([0.8, 0.2, 0.2, 1.0])
            )
    
    def build_plant_welded(self, plant: MultibodyPlant, model_instance, register_visuals: bool = True):
        """
        Build cart-pendulum for welded mode (no world connection).
        
        Creates cart body as a free body (to be welded by caller) with 2-DOF gimbal pendulum attached.
        This avoids kinematic loops when cart is welded to manipulator EE.
        
        Args:
            plant: MultibodyPlant to add bodies to
            model_instance: Model instance index
            register_visuals: Whether to register visual geometry
            
        Returns:
            cart_body: The cart RigidBody (for welding by caller)
        """
        # ====================================================================
        # CREATE CART BODY (FREE BODY - NO WORLD CONNECTION)
        # ====================================================================
        cart_size = 0.1
        m_c = self.config.mass_cart
        
        # Cart inertia (box approximation)
        I_c = (1/12) * m_c * cart_size**2
        cart_inertia = SpatialInertia(
            mass=m_c,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(I_c, I_c, I_c)
        )
        
        self.cart_body = plant.AddRigidBody(
            "cart",
            model_instance,
            cart_inertia
        )
        
        # Cart visual geometry
        if register_visuals:
            plant.RegisterVisualGeometry(
                self.cart_body,
                RigidTransform(),
                Sphere(cart_size / 2),
                "cart_visual",
                np.array([0.3, 0.3, 0.8, 1.0])
            )
        
        # ====================================================================
        # CREATE PENDULUM (2-DOF GIMBAL MOUNT)
        # ====================================================================
        m_p = self.config.mass_pendulum
        L = self.config.length_pendulum
        r = 0.05
        
        # Pitch body (intermediate gimbal)
        pitch_inertia = SpatialInertia(
            mass=0.001,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia(1e-6, 1e-6, 1e-6)
        )
        self.pitch_body = plant.AddRigidBody(
            "pitch_gimbal",
            model_instance,
            pitch_inertia
        )
        
        # Pendulum body (ball at end)
        I_p = (2/5) * m_p * r**2 + m_p * L**2
        pendulum_inertia = SpatialInertia(
            mass=m_p,
            p_PScm_E=[0, 0, -L],
            G_SP_E=UnitInertia(I_p / m_p, I_p / m_p, (2/5) * r**2)
        )
        self.pendulum_body = plant.AddRigidBody(
            "pendulum",
            model_instance,
            pendulum_inertia
        )
        
        # Pitch joint (rotation about Y-axis)
        self.pitch_joint = plant.AddJoint(
            RevoluteJoint(
                name="pendulum_pitch",
                frame_on_parent=self.cart_body.body_frame(),
                frame_on_child=self.pitch_body.body_frame(),
                axis=[0, 1, 0],
                damping=self.config.damping_pendulum
            )
        )
        
        # Roll joint (rotation about X-axis)
        self.roll_joint = plant.AddJoint(
            RevoluteJoint(
                name="pendulum_roll",
                frame_on_parent=self.pitch_body.body_frame(),
                frame_on_child=self.pendulum_body.body_frame(),
                axis=[1, 0, 0],
                damping=self.config.damping_pendulum
            )
        )
        
        # Pendulum visual geometry
        if register_visuals:
            # Rod
            plant.RegisterVisualGeometry(
                self.pendulum_body,
                RigidTransform([0, 0, -L/2]),
                Cylinder(radius=0.01, length=L),
                "pendulum_rod",
                np.array([0.6, 0.4, 0.2, 1.0])
            )
            
            # Ball
            plant.RegisterVisualGeometry(
                self.pendulum_body,
                RigidTransform([0, 0, -L]),
                Sphere(r),
                "pendulum_ball",
                np.array([0.8, 0.2, 0.2, 1.0])
            )
        
        return self.cart_body


# ============================================================================
# FRAME UPDATER SYSTEM
# ============================================================================

class MeshcatFrameUpdater(LeafSystem):
    """
    Updates coordinate frame visualizations in Meshcat during simulation.
    This system reads the plant state and updates all frame transforms.
    """
    
    def __init__(self, meshcat, plant, frame_list, update_period=0.033):
        """
        Args:
            meshcat: Meshcat instance
            plant: MultibodyPlant
            frame_list: List of (frame_name, frame, length) tuples
            update_period: Update frequency in seconds (default 30 Hz)
        """
        LeafSystem.__init__(self)
        self.meshcat = meshcat
        self.plant = plant
        self.frame_list = frame_list
        
        # Input port for plant state
        self.DeclareVectorInputPort("plant_state", plant.num_multibody_states())
        
        # Periodic update
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=update_period,
            offset_sec=0.0,
            update=self._update_frames
        )
    
    def _update_frames(self, context, state):
        """Update frame positions in Meshcat."""
        # Get plant context
        plant_context = self.plant.CreateDefaultContext()
        
        # Get state from input port
        x = self.get_input_port(0).Eval(context)
        self.plant.SetPositionsAndVelocities(plant_context, x)
        
        # Update all frames
        for frame_name, frame, length in self.frame_list:
            X_WF = self.plant.CalcRelativeTransform(
                plant_context, 
                self.plant.world_frame(), 
                frame
            )
            self.meshcat.SetTransform(f"/Frames/{frame_name}", X_WF)


# ============================================================================
# MUSCLE DYNAMICS (2D)
# ============================================================================

class MuscleDynamics2D(LeafSystem):
    """
    2D muscle dynamics: Ḟ = (-F + u) / τ
    
    Input: u (2) = [u_x, u_y] neural command
    Output: F (2) = [F_x, F_y] muscle force
    State: F (2)
    """
    
    def __init__(self, config: MuscleDynamicsConfig):
        LeafSystem.__init__(self)
        self.muscle_tau = config.muscle_tau
        self.initial_force = config.initial_force
        
        # State: [F_x, F_y]
        self.DeclareContinuousState(2)
        
        # Input: command u
        self.DeclareVectorInputPort("u", 2)
        
        # Output: force F
        self.DeclareVectorOutputPort("F", 2, self.calc_output)
    
    def SetDefaultState(self, context, state):
        state.SetFromVector(self.initial_force)
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        F = context.get_continuous_state_vector().CopyToVector()
        u = self.get_input_port().Eval(context)
        F_dot = (-F + u) / self.muscle_tau
        derivatives.get_mutable_vector().SetFromVector(F_dot)
    
    def calc_output(self, context, output):
        F = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(F)


# ============================================================================
# ZFT REFERENCE MASS (2D)
# ============================================================================

class ZFTReferenceMass2D(LeafSystem):
    """
    2D ZFT reference mass dynamics:
    
    ẍ_ref = (K*(x - x_ref) + D*(ẋ - ẋ_ref) + F) / M
    ÿ_ref = (K*(y - y_ref) + D*(ẏ - ẏ_ref) + F) / M
    
    Inputs:
      0: cart_state (4) = [x, y, ẋ, ẏ]
      1: F (2) = [F_x, F_y]
    Output:
      0: ref_state (4) = [x_ref, y_ref, ẋ_ref, ẏ_ref]
    State: [x_ref, y_ref, ẋ_ref, ẏ_ref]
    """
    
    def __init__(self, config: ZFTReferenceMassConfig):
        LeafSystem.__init__(self)
        self.M_ref = config.M_ref
        self.K_imp = config.K_imp
        self.D_imp = config.D_imp
        self.initial_ref = config.initial_ref
        
        # State: [x_ref, y_ref, ẋ_ref, ẏ_ref]
        self.DeclareContinuousState(4)
        
        # Inputs
        self.DeclareVectorInputPort("cart_state", 4)
        self.DeclareVectorInputPort("F", 2)
        
        # Output
        self.DeclareVectorOutputPort("ref_state", 4, self.calc_output)
    
    def SetDefaultState(self, context, state):
        state.SetFromVector(self.initial_ref)
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        ref_state = context.get_continuous_state_vector().CopyToVector()
        cart_state = self.get_input_port(0).Eval(context)
        F = self.get_input_port(1).Eval(context)
        
        x, y, x_dot, y_dot = cart_state
        x_ref, y_ref, x_ref_dot, y_ref_dot = ref_state
        F_x, F_y = F
        
        # Reference dynamics
        x_ref_ddot = (self.K_imp * (x - x_ref) + self.D_imp * (x_dot - x_ref_dot) + F_x) / self.M_ref
        y_ref_ddot = (self.K_imp * (y - y_ref) + self.D_imp * (y_dot - y_ref_dot) + F_y) / self.M_ref
        
        derivatives.get_mutable_vector().SetFromVector(
            np.array([x_ref_dot, y_ref_dot, x_ref_ddot, y_ref_ddot])
        )
    
    def calc_output(self, context, output):
        ref_state = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(ref_state)


# ============================================================================
# IMPEDANCE FORCE (2D)
# ============================================================================

class ImpedanceForce2D(LeafSystem):
    """
    2D impedance force: F_imp = K*(r_ref - r) + D*(ṙ_ref - ṙ)
    
    Inputs:
      0: cart_state (4) = [x, y, ẋ, ẏ]
      1: ref_state (4) = [x_ref, y_ref, ẋ_ref, ẏ_ref]
    Output:
      0: F_imp (2) = [F_x, F_y]
    """
    
    def __init__(self, config: ImpedanceForceConfig):
        LeafSystem.__init__(self)
        self.K_imp = config.K_imp
        self.D_imp = config.D_imp
        
        # Inputs
        self.DeclareVectorInputPort("cart_state", 4)
        self.DeclareVectorInputPort("ref_state", 4)
        
        # Output
        self.DeclareVectorOutputPort("F_imp", 2, self.calc_output)
    
    def calc_output(self, context, output):
        cart = self.get_input_port(0).Eval(context)
        ref = self.get_input_port(1).Eval(context)
        
        x, y, x_dot, y_dot = cart
        x_ref, y_ref, x_ref_dot, y_ref_dot = ref
        
        F_x = self.K_imp * (x_ref - x) + self.D_imp * (x_ref_dot - x_dot)
        F_y = self.K_imp * (y_ref - y) + self.D_imp * (y_ref_dot - y_dot)
        
        output.SetFromVector(np.array([F_x, F_y]))


# ============================================================================
# FINITE-HORIZON LQR CONTROLLER (2D)
# ============================================================================

class FiniteHorizonLQRController2D(LeafSystem):
    """
    Finite-horizon LQR for 14D system with 2D control.
    
    State: [x, y, α, β, ẋ, ẏ, α̇, β̇, F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref]
    Control: [u_x, u_y]
    
    Solves backward Riccati recursion and applies time-varying gain.
    """
    
    def __init__(self, A, B, config: FiniteHorizonLQRConfig):
        LeafSystem.__init__(self)
        self.Q = config.Q
        self.QN = config.QN if config.QN is not None else config.Q.copy()
        self.R = config.R
        self.x_goal = config.x_goal
        self.T = float(config.horizon)
        self.dt = float(config.timestep)
        self.u_limits = config.u_limits
        
        # Discretize system
        n = A.shape[0]
        m = B.shape[1]
        I = np.eye(n)
        self.Ad = I + A * self.dt
        self.Bd = B * self.dt
        
        # Solve Riccati recursion backward
        N = int(self.T / self.dt)
        self.K_gains = []
        P = self.QN.copy()
        
        for i in range(N):
            K = np.linalg.solve(self.R + self.Bd.T @ P @ self.Bd, self.Bd.T @ P @ self.Ad)
            self.K_gains.insert(0, K)
            P = self.Q + self.Ad.T @ P @ (self.Ad - self.Bd @ K)
        
        # Input: state (14D)
        self.DeclareVectorInputPort("state", 14)
        
        # Output: control (2D)
        self.DeclareVectorOutputPort("u", 2, self.calc_control)
    
    def calc_control(self, context, output):
        x = self.get_input_port().Eval(context)
        t = context.get_time()
        
        # Get time-varying gain
        idx = int(t / self.dt)
        idx = min(idx, len(self.K_gains) - 1)
        K = self.K_gains[idx]
        
        # Compute control
        u = -K @ (x - self.x_goal)
        
        # Apply limits if specified
        if self.u_limits is not None:
            u = np.clip(u, self.u_limits[0], self.u_limits[1])
        
        output.SetFromVector(u)


# ============================================================================
# LINEARIZATION FUNCTION
# ============================================================================

def build_linearized_system_2d(
    physics_config: CartPendulumPhysicsConfig,
    impedance_config: ImpedanceForceConfig,
    zft_config: ZFTReferenceMassConfig,
    muscle_config: MuscleDynamicsConfig,
):
    """
    Build linearized 14D system matrices.
    
    Uses Drake's Linearize() for cart-pendulum (8D), then assembles
    with muscle dynamics (2D) and ZFT dynamics (4D).
    
    Returns:
        A (14×14), B (14×2): Linearized system matrices
    """
    # Extract parameters
    K_imp = impedance_config.K_imp
    D_imp = impedance_config.D_imp
    M_ref = zft_config.M_ref
    muscle_tau = muscle_config.muscle_tau
    M_cart = physics_config.mass_cart
    
    # Need to create a temporary plant since the one we are initializing in SystemBuilder is discrete time and we want a continuous-time linearization
    # Create temporary plant for linearization
    temp_cart_config = create_cart_pendulum_config(
        cart_mass=physics_config.mass_cart,
        cart_damping=physics_config.damping_cart,
        pendulum_mass=physics_config.mass_pendulum,
        pendulum_length=physics_config.length_pendulum,
    )
    
    temp_builder = DiagramBuilder()
    temp_plant = MultibodyPlant(time_step=0.0)
    temp_model = temp_plant.AddModelInstance("cart_temp")
    
    # Build 2D cart-pendulum
    temp_cart = CartPendulum2DExtended(physics_config)
    temp_cart.build_plant(temp_plant, temp_model, register_visuals=False)
    
    temp_plant.Finalize()
    temp_builder.AddSystem(temp_plant)
    temp_diagram = temp_builder.Build()
    
    # Linearize around equilibrium
    temp_context = temp_diagram.CreateDefaultContext()
    temp_plant_context = temp_plant.GetMyContextFromRoot(temp_context)
    
    # Set equilibrium: cart at origin, pendulum hanging down
    temp_plant.SetPositions(temp_plant_context, np.zeros(4))
    temp_plant.SetVelocities(temp_plant_context, np.zeros(4))
    
    # Get input port for forces and set to zero for linearization
    input_port = temp_plant.get_actuation_input_port()
    input_port.FixValue(temp_plant_context, np.zeros(2))
    output_port = temp_plant.get_state_output_port()
    
    # Linearize using Drake's Linearize function
    from pydrake.systems.primitives import Linearize
    linear_sys = Linearize(temp_plant, temp_plant_context, 
                          input_port_index=input_port.get_index(),
                          output_port_index=output_port.get_index())
    
    A_cp = linear_sys.A()
    B_cp = linear_sys.B()
    
    # Muscle dynamics (2D): Ḟ = (-F + u) / τ
    A_muscle = -np.eye(2) / muscle_tau
    B_muscle = np.eye(2) / muscle_tau
    
    # Assemble 14×14 A matrix
    A = np.zeros((14, 14))
    
    # Cart-pendulum block (8×8)
    A[0:8, 0:8] = A_cp
    
    # Coupling: cart-pendulum affected by impedance force
    # ẍ += (K*(x_ref - x) + D*(ẋ_ref - ẋ) + F_x) / M
    # ÿ += (K*(y_ref - y) + D*(ẏ_ref - ẏ) + F_y) / M
    A[4, 0] = -K_imp / M_cart  # ẍ ← -K*x/M
    A[4, 4] = -D_imp / M_cart  # ẍ ← -D*ẋ/M
    A[4, 8] = 1.0 / M_cart     # ẍ ← F_x/M
    A[4, 10] = K_imp / M_cart  # ẍ ← K*x_ref/M
    A[4, 12] = D_imp / M_cart  # ẍ ← D*ẋ_ref/M
    
    A[5, 1] = -K_imp / M_cart  # ÿ ← -K*y/M
    A[5, 5] = -D_imp / M_cart  # ÿ ← -D*ẏ/M
    A[5, 9] = 1.0 / M_cart     # ÿ ← F_y/M
    A[5, 11] = K_imp / M_cart  # ÿ ← K*y_ref/M
    A[5, 13] = D_imp / M_cart  # ÿ ← D*ẏ_ref/M
    
    # Muscle dynamics block (2×2)
    A[8:10, 8:10] = A_muscle
    
    # ZFT dynamics block (4×4)
    # ẋ_ref = ẋ_ref (position from velocity)
    A[10, 12] = 1.0
    A[11, 13] = 1.0
    
    # ẍ_ref = (K*(x - x_ref) + D*(ẋ - ẋ_ref) + F) / M_ref
    A[12, 0] = K_imp / M_ref
    A[12, 4] = D_imp / M_ref
    A[12, 8] = 1.0 / M_ref
    A[12, 10] = -K_imp / M_ref
    A[12, 12] = -D_imp / M_ref
    
    A[13, 1] = K_imp / M_ref
    A[13, 5] = D_imp / M_ref
    A[13, 9] = 1.0 / M_ref
    A[13, 11] = -K_imp / M_ref
    A[13, 13] = -D_imp / M_ref
    
    # Assemble 14×2 B matrix
    B = np.zeros((14, 2))
    B[8:10, 0:2] = B_muscle
    
    return A, B


def build_linearized_for_complete_system_2d(
    plant: MultibodyPlant,
    manipulator,
    cart_model,
    physics_config: CartPendulumPhysicsConfig,
):
    """
    Build linearized system matrices for welded cart-pendulum-manipulator.
    
    For the welded configuration where cart is attached to manipulator EE:
    - State: [q1, q2, α, β, q̇1, q̇2, α̇, β̇] (8D)
      - q1, q2: Manipulator joint angles
      - α, β: Pendulum gimbal angles (pitch, roll)
    - Control: [τ1, τ2] (2D) - manipulator joint torques
    
    Uses Drake's Linearize() around equilibrium (pendulum hanging).
    
    Args:
        plant: Finalized MultibodyPlant with welded system
        manipulator: CupManipulator instance
        cart_model: ModelInstanceIndex for cart-pendulum
        physics_config: Physical parameters
    
    Returns:
        A (8×8): State transition matrix
        B (8×2): Control input matrix
    """
    from pydrake.systems.primitives import Linearize
    
    # Need to create a temporary plant since the one we are initializing in SystemBuilder is discrete time and we want a continuous-time linearization
    # Create temporary plant for linearization (continuous-time, NO SceneGraph)
    # Following same pattern as build_linearized_system_2d
    temp_builder = DiagramBuilder()
    temp_plant = MultibodyPlant(time_step=0.0)  # Continuous time
    
    # Rebuild the welded system in the temporary plant
    # Add manipulator
    temp_parser = Parser(temp_plant)
    temp_manip = CupManipulator(manipulator.config, enable_visualization=False)
    temp_manip.load_urdf_to_plant(temp_plant, temp_parser)
    temp_manip.weld_base_to_world(temp_plant, orientation=np.array([0.0, 0.0, 0.0]))
    temp_manip.add_joint_actuators(temp_plant)
    temp_manip.add_end_effector_frame(temp_plant)
    
    # Add welded cart-pendulum
    temp_cart_model = temp_plant.AddModelInstance("cart_pendulum")
    temp_cart_pend = CartPendulum2DExtended(physics_config, z_offset=0.0)
    cart_body = temp_cart_pend.build_plant_welded(temp_plant, temp_cart_model, register_visuals=False)
    
    # Weld cart to EE
    ee_frame = temp_manip.get_end_effector_frame(temp_plant)
    temp_plant.WeldFrames(
        frame_on_parent_F=ee_frame,
        frame_on_child_M=cart_body.body_frame(),
        X_FM=RigidTransform()
    )
    
    temp_plant.Finalize()
    temp_builder.AddSystem(temp_plant)
    temp_diagram = temp_builder.Build()
    
    # Linearize around equilibrium
    temp_context = temp_diagram.CreateDefaultContext()
    temp_plant_context = temp_plant.GetMyContextFromRoot(temp_context)
    
    # Set equilibrium: manipulator at [0°, 20°], pendulum hanging (α=0, β=0)
    initial_q = np.array([np.deg2rad(0.0), np.deg2rad(20.0)])
    temp_manip.set_positions_user_order(temp_plant, temp_plant_context, {
        "link1_base": initial_q[0],
        "link2_link1": initial_q[1],
    })
    
    # Set pendulum positions (pitch = 0, roll = 0, hanging down)
    temp_plant.SetPositions(temp_plant_context, temp_cart_model, np.array([0.0, 0.0]))
    
    # Set all velocities to zero
    temp_plant.SetVelocities(temp_plant_context, np.zeros(temp_plant.num_velocities()))
    
    # Get input/output ports
    # For welded mode, we only control manipulator (2 torques)
    manip_input_port = temp_plant.get_actuation_input_port(temp_manip.model_instance)
    state_output_port = temp_plant.get_state_output_port()
    
    # Set actuator inputs to zero for linearization
    manip_input_port.FixValue(temp_plant_context, np.zeros(2))
    
    # Linearize using Drake
    linear_sys = Linearize(
        temp_plant, 
        temp_plant_context,
        input_port_index=manip_input_port.get_index(),
        output_port_index=state_output_port.get_index()
    )
    
    A = linear_sys.A()
    B = linear_sys.B()
    
    print(colored(f"  - Linearized around: q1={np.rad2deg(initial_q[0]):.1f}°, "
                  f"q2={np.rad2deg(initial_q[1]):.1f}°, α=0°, β=0°", "cyan"))
    print(colored(f"  - State dimension: {A.shape[0]} (q1, q2, α, β, q̇1, q̇2, α̇, β̇)", "cyan"))
    print(colored(f"  - Control dimension: {B.shape[1]} (τ1, τ2)", "cyan"))
    
    return A, B


# ============================================================================
# INVERSE KINEMATICS FEASIBILITY CHECK
# ============================================================================

def check_trajectory_feasibility(manipulator, plant, trajectory_points, q_init=None):
    """
    Check if the manipulator can reach all points in the trajectory using IK.
    
    Args:
        manipulator: CupManipulator instance
        plant: MultibodyPlant with manipulator
        trajectory_points: Nx2 array of [x, y] positions
        q_init: Initial joint configuration [q1, q2] (default: [-10°, 20°])
    
    Returns:
        feasible: Boolean array indicating which points are reachable
        joint_solutions: Nx2 array of joint angles (or None if infeasible)
        stats: Dictionary with feasibility statistics
    """
    from scipy.optimize import minimize
    
    if q_init is None:
        q_init = np.deg2rad([-10, 20])  # Default initial config
    
    N = len(trajectory_points)
    feasible = np.zeros(N, dtype=bool)
    joint_solutions = np.zeros((N, 2))
    
    # Get manipulator parameters for IK
    ee_frame = plant.GetFrameByName(manipulator.LINK2_NAME, manipulator.model_instance)
    world_frame = plant.world_frame()
    EE_OFFSET = manipulator.EE_OFFSET
    
    def forward_kinematics(q):
        """Compute EE position given joint angles [q1, q2]."""
        context = plant.CreateDefaultContext()
        # Use manipulator's set_positions_user_order with explicit joint names
        manipulator.set_positions_user_order(plant, context, {
            "link1_base": q[0],
            "link2_link1": q[1],
        })
        
        ee_pos = plant.CalcPointsPositions(
            context, ee_frame, EE_OFFSET.reshape(3, 1), world_frame
        ).flatten()
        return ee_pos[:2]  # Return x, y only
    
    def ik_cost(q, target_xy):
        """Cost function for IK: distance to target."""
        ee_xy = forward_kinematics(q)
        error = target_xy - ee_xy
        return np.sum(error**2)
    
    # Solve IK for each trajectory point
    q_prev = q_init.copy()
    
    for i, target_xy in enumerate(trajectory_points):
        # Try to find joint angles that reach target position
        # Use previous solution as initial guess for continuity
        result = minimize(
            ik_cost,
            q_prev,
            args=(target_xy,),
            method='SLSQP',
            bounds=[(-np.pi, np.pi), (-np.pi, np.pi)],  # Joint limits
            options={'ftol': 1e-6, 'maxiter': 100}
        )
        
        # Check if solution is feasible (error < 10mm)
        final_error = np.sqrt(result.fun)
        if final_error < 0.01:  # 10mm threshold
            feasible[i] = True
            joint_solutions[i] = result.x
            q_prev = result.x  # Use as next initial guess
        else:
            feasible[i] = False
            joint_solutions[i] = np.nan
    
    # Compute statistics
    stats = {
        'n_total': N,
        'n_feasible': np.sum(feasible),
        'n_infeasible': N - np.sum(feasible),
        'feasibility_rate': np.sum(feasible) / N * 100,
        'max_joint_range_deg': np.rad2deg([
            np.nanmax(joint_solutions[:, 0]) - np.nanmin(joint_solutions[:, 0]),
            np.nanmax(joint_solutions[:, 1]) - np.nanmin(joint_solutions[:, 1])
        ]) if np.sum(feasible) > 0 else [0, 0]
    }
    
    return feasible, joint_solutions, stats


def test_and_visualize_ik_feasibility(
    manipulator, plant, duration, dt=0.001, 
    trajectory_func=None, x_target=None, y_target=None
):
    """
    Test IK feasibility for the entire trajectory and visualize results.
    
    Args:
        manipulator: CupManipulator instance
        plant: MultibodyPlant
        duration: Simulation duration [s]
        dt: Time step [s]
        trajectory_func: Optional function(t) -> (x, y) for custom trajectory
        x_target, y_target: Target position (used if trajectory_func is None)
    """
    print(colored("\n🔍 Testing Inverse Kinematics Feasibility...", "cyan"))
    
    # Generate trajectory points
    t_points = np.arange(0, duration, dt)
    N = len(t_points)
    trajectory_points = np.zeros((N, 2))
    
    if trajectory_func is not None:
        # Use custom trajectory function
        for i, t in enumerate(t_points):
            trajectory_points[i] = trajectory_func(t)
    else:
        # Use simple point-to-point trajectory
        # Assuming linear motion from initial position to target
        # Initial position is at manipulator EE (approximated)
        x0, y0 = -2.174, 0.052  # Typical initial position
        for i, t in enumerate(t_points):
            alpha = min(t / duration, 1.0)
            trajectory_points[i, 0] = x0 + alpha * (x_target - x0)
            trajectory_points[i, 1] = y0 + alpha * (y_target - y0)
    
    # Sample trajectory (every 10ms for faster IK solve)
    sample_indices = np.arange(0, N, 10)
    sampled_points = trajectory_points[sample_indices]
    sampled_times = t_points[sample_indices]
    
    # Check feasibility
    feasible, joint_solutions, stats = check_trajectory_feasibility(
        manipulator, plant, sampled_points
    )
    
    # Print results
    print(colored(f"📊 IK Feasibility Analysis:", "cyan"))
    print(f"   Total points checked: {stats['n_total']}")
    print(f"   Feasible points:      {stats['n_feasible']} ({stats['feasibility_rate']:.1f}%)")
    print(f"   Infeasible points:    {stats['n_infeasible']}")
    
    if stats['n_feasible'] > 0:
        print(f"   Joint 1 range:        {stats['max_joint_range_deg'][0]:.1f}°")
        print(f"   Joint 2 range:        {stats['max_joint_range_deg'][1]:.1f}°")
        
        # Show min/max joint angles
        q1_min, q1_max = np.rad2deg(np.nanmin(joint_solutions[:, 0])), np.rad2deg(np.nanmax(joint_solutions[:, 0]))
        q2_min, q2_max = np.rad2deg(np.nanmin(joint_solutions[:, 1])), np.rad2deg(np.nanmax(joint_solutions[:, 1]))
        print(f"   Joint 1 limits:       [{q1_min:+6.1f}°, {q1_max:+6.1f}°]")
        print(f"   Joint 2 limits:       [{q2_min:+6.1f}°, {q2_max:+6.1f}°]")
    
    # Identify infeasible regions
    if stats['n_infeasible'] > 0:
        infeasible_indices = np.where(~feasible)[0]
        infeasible_times = sampled_times[infeasible_indices]
        print(colored(f"\n⚠️  Warning: {stats['n_infeasible']} points are unreachable!", "yellow"))
        if len(infeasible_times) <= 10:
            print(f"   Infeasible times: {infeasible_times}")
        else:
            print(f"   First infeasible time: {infeasible_times[0]:.3f}s")
            print(f"   Last infeasible time:  {infeasible_times[-1]:.3f}s")
    else:
        print(colored("✓ All trajectory points are reachable!", "green"))
    
    return feasible, joint_solutions, trajectory_points, stats


class ComputedTorqueEEController(LeafSystem):
    """
    Computed torque controller for end-effector trajectory tracking.
    
    Inputs:
      0: desired_trajectory (4) = [x_d, y_d, ẋ_d, ẏ_d]
      1: manipulator_state (4) = [q1, q2, q̇1, q̇2] (from plant with natural URDF)
    Output:
      0: joint_torques (2) = [τ1, τ2] (natural order matches actuator order)
    """
    
    def __init__(self, manipulator, plant, Kp=200.0, Kd=30.0, tau_max=100.0):
        LeafSystem.__init__(self)
        self.manipulator = manipulator
        self.plant = plant
        self.Kp = Kp
        self.Kd = Kd
        self.tau_max = tau_max
        self.call_count = 0
        
        # Inputs
        self.DeclareVectorInputPort("desired_trajectory", 4)
        self.DeclareVectorInputPort("manipulator_state", 4)
        
        # Output
        self.DeclareVectorOutputPort("joint_torques", 2, self.calc_torques)
    
    def calc_torques(self, context, output):
        # Get inputs
        traj = self.get_input_port(0).Eval(context)
        manip_state = self.get_input_port(1).Eval(context)
        
        x_d, y_d, x_dot_d, y_dot_d = traj
        
        x_ddot_d, y_ddot_d = 0.0, 0.0  # Zero acceleration
        # manip_state is in natural [q1, q2, q̇1, q̇2] order (from manipulator.get_state_from_plant)
        q1, q2, q1_dot, q2_dot = manip_state
        q_manip = np.array([q1, q2])
        q_dot_manip = np.array([q1_dot, q2_dot])
        
        # Create plant context
        plant_context = self.plant.CreateDefaultContext()
        
        # Set manipulator state using helper method that handles Drake ordering
        temp_state = np.array([q1, q2, q1_dot, q2_dot])
        self.manipulator.set_state_in_plant(self.plant, plant_context, temp_state)
        
        # Get EE frame and compute current position
        ee_frame = self.plant.GetFrameByName(self.manipulator.LINK2_NAME, self.manipulator.model_instance)
        world_frame = self.plant.world_frame()
        EE_OFFSET = self.manipulator.EE_OFFSET
        
        ee_pos = self.plant.CalcPointsPositions(
            plant_context, ee_frame, EE_OFFSET.reshape(3, 1), world_frame
        ).flatten()
        x_current, y_current = ee_pos[0], ee_pos[1]
        
        # Compute Jacobian
        J_full = self.plant.CalcJacobianTranslationalVelocity(
            plant_context,
            JacobianWrtVariable.kQDot,
            ee_frame,
            EE_OFFSET,
            world_frame,
            world_frame
        )
        
        # Extract manipulator velocity indices using joint names (Drake order: [JT1=q2, JT2=q1])
        jt1 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT1_NAME)
        jt2 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT2_NAME)
        manip_velocity_indices = [jt1.velocity_start(), jt2.velocity_start()]
        
        J_xy = J_full[0:2, manip_velocity_indices]  # 2x2
        
        # Current EE velocity
        ee_vel = J_xy @ q_dot_manip
        
        # Task-space errors
        e_pos = np.array([x_d - x_current, y_d - y_current])
        e_vel = np.array([x_dot_d - ee_vel[0], y_dot_d - ee_vel[1]])
        
        # Desired task-space acceleration with PD feedback
        x_ddot_des_vec = np.array([x_ddot_d, y_ddot_d])
        x_ddot_control = x_ddot_des_vec + self.Kp * e_pos + self.Kd * e_vel
        
        # Map to joint space
        J_pinv = np.linalg.pinv(J_xy)
        q_ddot_desired = J_pinv @ x_ddot_control
        
        # Compute inverse dynamics
        vd = np.zeros(self.plant.num_velocities())
        for i, vel_idx in enumerate(manip_velocity_indices):
            vd[vel_idx] = q_ddot_desired[i]
        
        external_forces = MultibodyForces(self.plant)
        tau_all = self.plant.CalcInverseDynamics(
            plant_context,
            vd,
            external_forces
        ).flatten()
        
        tau = np.array([tau_all[idx] for idx in manip_velocity_indices])
        tau = np.clip(tau, -self.tau_max, self.tau_max)
        
        # Debug output
        self.call_count += 1
        if self.call_count % 100 == 1:
            error_mm = np.linalg.norm(e_pos) * 1000
            sat = " SAT" if np.any(np.abs(tau) >= self.tau_max - 0.1) else ""
            print(f"[t={context.get_time():.2f}s] EE=[{x_current:+5.2f},{y_current:+5.2f}] "
                  f"Desired=[{x_d:+5.2f},{y_d:+5.2f}] Err={error_mm:4.0f}mm "
                  f"τ=[{tau[0]:+5.1f},{tau[1]:+5.1f}]{sat}")
        
        output.SetFromVector(tau)


class ComputedTorqueJointSpaceController(LeafSystem):
    """
    Joint-space computed torque controller.
    
    Inputs:
      0: desired_joint_state (4) = [q1_d, q2_d, q̇1_d, q̇2_d]
      1: manipulator_state (4) = [q1, q2, q̇1, q̇2] (from plant with natural URDF)
    Output:
      0: joint_torques (2) = [τ1, τ2] (natural order matches actuator order)
    """
    
    def __init__(self, manipulator, plant, Kp=200.0, Kd=60.0, tau_max=100.0):
        LeafSystem.__init__(self)
        self.manipulator = manipulator
        self.plant = plant
        self.Kp = Kp
        self.Kd = Kd
        self.tau_max = tau_max
        self.call_count = 0
        
        # Inputs
        self.DeclareVectorInputPort("desired_joint_state", 4)
        self.DeclareVectorInputPort("manipulator_state", 4)
        
        # Output
        self.DeclareVectorOutputPort("joint_torques", 2, self.calc_torques)
    
    def calc_torques(self, context, output):
        # Get inputs
        desired = self.get_input_port(0).Eval(context)  # [q1_d, q2_d, q̇1_d, q̇2_d]
        manip_state = self.get_input_port(1).Eval(context)  # [q1, q2, q̇1, q̇2] natural order
        
        q1_d, q2_d, q1_dot_d, q2_dot_d = desired
        q1, q2, q1_dot, q2_dot = manip_state
        
        # Joint space errors (in [q1, q2] order)
        e_q = np.array([q1_d - q1, q2_d - q2])
        e_q_dot = np.array([q1_dot_d - q1_dot, q2_dot_d - q2_dot])
        
        # Desired joint accelerations with PD feedback
        q_ddot_desired = self.Kp * e_q + self.Kd * e_q_dot
        
        # Create plant context with current state
        plant_context = self.plant.CreateDefaultContext()
        
        # Set manipulator state using helper method that handles Drake ordering
        temp_state = np.array([q1, q2, q1_dot, q2_dot])
        self.manipulator.set_state_in_plant(self.plant, plant_context, temp_state)
        
        # Compute inverse dynamics with natural [q̈1, q̈2] order
        # Get velocity indices using joint names (Drake order: [JT1=q2, JT2=q1])
        jt1 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT1_NAME)
        jt2 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT2_NAME)
        
        # Compute inverse dynamics
        vd = np.zeros(self.plant.num_velocities())
        # Map user-order accelerations [q̈1, q̈2] to Drake order [q̈2, q̈1]
        vd[jt1.velocity_start()] = q_ddot_desired[1]  # JT1 = q2
        vd[jt2.velocity_start()] = q_ddot_desired[0]  # JT2 = q1
        
        external_forces = MultibodyForces(self.plant)
        tau_all = self.plant.CalcInverseDynamics(
            plant_context,
            vd,
            external_forces
        ).flatten()
        
        # Extract torques in Drake order [τ2, τ1], then convert to user order [τ1, τ2]
        tau_drake = np.array([tau_all[jt1.velocity_start()], tau_all[jt2.velocity_start()]])
        tau = np.array([tau_drake[1], tau_drake[0]])  # Convert to user order [τ1, τ2]
        tau = np.clip(tau, -self.tau_max, self.tau_max)
        
        # Debug output
        self.call_count += 1
        if self.call_count % 100 == 1:
            error_deg = np.linalg.norm(e_q) * 180 / np.pi
            sat = " SAT" if np.any(np.abs(tau) >= self.tau_max - 0.1) else ""
            print(f"[JS-CT t={context.get_time():.2f}s] q_err={error_deg:4.1f}° "
                  f"τ=[{tau[0]:+5.1f},{tau[1]:+5.1f}]{sat}")
        
        output.SetFromVector(tau)


class ManipulatorIKDesiredAngles(LeafSystem):
    """
    Velocity-based manipulator controller with position feedback.
    
    Inputs:
      0: cart_state (4) = [x, y, ẋ, ẏ]  - desired cart trajectory
      1: plant_state (n) = full plant state vector
    Output: desired_joint_state (4) = [q1_d, q2_d, q̇1_d, q̇2_d]
    """
    
    def __init__(self, manipulator, plant, dt=0.001, Kp=10.0):
        LeafSystem.__init__(self)
        self.manipulator = manipulator
        self.plant = plant
        self.dt = dt
        self.Kp = Kp  # Position feedback gain
        
        # Extract link lengths from URDF
        self.L1, self.L2 = self._extract_link_lengths()
        
        self.DeclareVectorInputPort("cart_state", 4)
        self.DeclareVectorInputPort("plant_state", plant.num_multibody_states())
        self.DeclareVectorOutputPort("desired_joint_state", 4, self.calc_desired_angles)
    
    def _extract_link_lengths(self):
        """
        Extract link lengths L1 and L2 from URDF geometry.
        
        L1: Distance from base (joint1) to joint2
        L2: Distance from joint2 to end-effector (EE_OFFSET magnitude)
        
        Returns:
            L1, L2: Link lengths in meters
        """
        # Create a temporary context to query geometry
        temp_context = self.plant.CreateDefaultContext()
        
        # Get joint frames
        j1 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT1_NAME)
        j2 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT2_NAME)
        
        # Get the frames for both joints
        j1_frame = j1.frame_on_child()  # Link1 frame
        j2_frame = j2.frame_on_parent()  # Link1 frame (parent of joint2)
        j2_child_frame = j2.frame_on_child()  # Link2 frame
        
        # Set joints to zero configuration (positions + velocities)
        self.manipulator.set_state_in_plant(self.plant, temp_context, np.array([0.0, 0.0, 0.0, 0.0]))
        
        # Get transform from joint1 frame to joint2 frame at zero configuration
        X_j1_j2 = self.plant.CalcRelativeTransform(temp_context, j1_frame, j2_child_frame)
        
        # L1 is the distance between the two joints (XY plane distance)
        L1 = np.linalg.norm(X_j1_j2.translation()[:2])
        
        # L2 is the EE offset from link2 (already computed in manipulator)
        L2 = np.linalg.norm(self.manipulator.EE_OFFSET[:2])
        
        return L1, L2
    
    def compute_jacobian_manual(self, q1, q2):
        """
        Manually compute 2×2 Jacobian for 2-link planar manipulator.
        
        Forward kinematics (2D):
            x = L1*cos(q1) + L2*cos(q1+q2)
            y = L1*sin(q1) + L2*sin(q1+q2)
        
        Jacobian J = [∂x/∂q1  ∂x/∂q2]
                     [∂y/∂q1  ∂y/∂q2]
        
        Args:
            q1: Joint 1 angle (radians)
            q2: Joint 2 angle (radians)
            
        Returns:
            J_xy: 2×2 Jacobian matrix mapping [q̇1, q̇2] → [ẋ, ẏ]
        """
        # Use link lengths extracted from URDF
        L1 = self.L1
        L2 = self.L2
        
        # Compute Jacobian elements
        s1 = np.sin(q1)
        c1 = np.cos(q1)
        s12 = np.sin(q1 + q2)
        c12 = np.cos(q1 + q2)
        
        # J = [[-L1*sin(q1) - L2*sin(q1+q2),  -L2*sin(q1+q2)],
        #      [ L1*cos(q1) + L2*cos(q1+q2),   L2*cos(q1+q2)]]
        
        J_xy = np.array([
            [-L1*s1 - L2*s12,  -L2*s12],
            [ L1*c1 + L2*c12,   L2*c12]
        ])
        
        return J_xy
    
    def calc_desired_angles(self, context, output):
        from pydrake.all import JacobianWrtVariable
        
        # Get inputs
        cart_state = self.get_input_port(0).Eval(context)
        cart_pos_xy = cart_state[0:2]  # Desired EE position
        cart_vel_xy = cart_state[2:4]  # Desired EE velocity
        plant_state = self.get_input_port(1).Eval(context)
        
        # Setup plant context with current state
        plant_context = self.plant.CreateDefaultContext()
        self.plant.SetPositionsAndVelocities(plant_context, plant_state)
        q_current = self.plant.GetPositions(plant_context)
        
        # Get manipulator joint indices
        j1 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT1_NAME)
        j2 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT2_NAME)
        vel_idx_j1 = j1.velocity_start()
        vel_idx_j2 = j2.velocity_start()
        pos_idxs = [j1.position_start(), j2.position_start()]
        q_current_manip = np.array([q_current[pos_idxs[0]], q_current[pos_idxs[1]]])
        
        # Get EE frame (cup_center already includes EE_OFFSET)
        ee_frame = self.manipulator.get_end_effector_frame(self.plant)
        p_BQ = np.zeros(3)  # Zero offset since cup_center frame already includes EE_OFFSET
        
        # Compute current EE position
        X_WB = self.plant.CalcRelativeTransform(plant_context, self.plant.world_frame(), ee_frame)
        ee_current_3d = X_WB.translation()
        ee_current_xy = ee_current_3d[0:2]
        
        # Position error: how far is EE from desired cart position?
        pos_error_xy = cart_pos_xy - ee_current_xy
        
        # Compute Jacobian: maps joint velocities to EE velocity
        Jv_full = self.plant.CalcJacobianTranslationalVelocity(
            plant_context, JacobianWrtVariable.kV, ee_frame, p_BQ,
            self.plant.world_frame(), self.plant.world_frame()
        )
        J_xy_drake = Jv_full[0:2, [vel_idx_j1, vel_idx_j2]]  # Extract 2×2 manipulator Jacobian
        
        # Compute Jacobian manually for comparison
        J_xy_manual = self.compute_jacobian_manual(q_current_manip[0], q_current_manip[1])
        
        # Use manual Jacobian (Drake is for verification only)
        J_xy = J_xy_manual
        
        # Desired EE velocity: feedforward + position feedback
        ee_vel_desired = cart_vel_xy + self.Kp * pos_error_xy
        
        # Map to joint velocities
        qdot_des = np.linalg.pinv(J_xy) @ ee_vel_desired
        
        # Integrate to get desired positions: q_des = q_current + qdot_des * dt
        q_des = q_current_manip + qdot_des * self.dt
        
        output.SetFromVector(np.concatenate([q_des, qdot_des]))

# Add system to compute end-effector position and velocity
class ManipulatorEEStateComputer(LeafSystem):
    """Computes manipulator end-effector position and velocity from joint state."""
    def __init__(self, plant, manipulator):
        LeafSystem.__init__(self)
        self.plant = plant
        self.manipulator = manipulator
        
        # Input: manipulator state in Drake order from plant.get_state_output_port(model_instance)
        # For cup_manipulator with joints [link2_link1, link1_base], Drake order is [q2, q1, q̇2, q̇1]
        self.DeclareVectorInputPort("manip_state", 4)
        
        # Output: EE state [x, y, ẋ, ẏ]
        self.DeclareVectorOutputPort(
            "ee_state",
            4,
            self.CalcEEState
        )
    
    def CalcEEState(self, context, output):
        """Calculate EE position and velocity from joint state."""
        # Get manipulator state in DRAKE order [q2, q1, q̇2, q̇1] from plant output
        manip_state_drake = self.get_input_port(0).Eval(context)
        
        # Convert from Drake order to user order [q1, q2, q̇1, q̇2] for set_state_in_plant
        # Drake order for cup_manipulator: [q2, q1, q̇2, q̇1] (link2_link1, link1_base)
        # User order: [q1, q2, q̇1, q̇2] (link1_base, link2_link1)
        manip_state_user = np.array([manip_state_drake[1], manip_state_drake[0], 
                                      manip_state_drake[3], manip_state_drake[2]])
        
        # Create fresh context for this computation
        temp_context = self.plant.CreateDefaultContext()
        
        # Set state in temp context (expects user order)
        self.manipulator.set_state_in_plant(self.plant, temp_context, manip_state_user)
        
        # Calculate EE position using custom EE frame with offset
        ee_pos = self.manipulator.get_end_effector_position(self.plant, temp_context)
        
        # Calculate EE velocity using Jacobian
        ee_frame = self.plant.GetFrameByName(self.manipulator.LINK2_NAME, self.manipulator.model_instance)
        J_full = self.plant.CalcJacobianTranslationalVelocity(
            temp_context,
            JacobianWrtVariable.kQDot,
            ee_frame,
            self.manipulator.EE_OFFSET,
            self.plant.world_frame(),
            self.plant.world_frame()
        )
        
        # Extract manipulator velocity indices
        jt1 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT1_NAME)
        jt2 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT2_NAME)
        manip_velocity_indices = [jt1.velocity_start(), jt2.velocity_start()]
        
        # Compute EE velocity: v_ee = J * q̇
        J_xy = J_full[0:2, manip_velocity_indices]
        ee_vel = J_xy @ manip_state_user[2:4]
        
        # Output [x, y, ẋ, ẏ]
        output.SetFromVector(np.array([ee_pos[0], ee_pos[1], ee_vel[0], ee_vel[1]]))


# ============================================================================
# PREVIEW VISUALIZATION
# ============================================================================

def visualize_plant_meshcat(
    plant: MultibodyPlant,
    scene_graph: SceneGraph,
    meshcat,
    positions_dict: dict = None,
    message: str = "Visualizing plant state in Meshcat..."
):
    """
    Visualize the plant configuration in Meshcat.
    
    Args:
        plant: Finalized MultibodyPlant
        scene_graph: SceneGraph for visualization
        meshcat: Meshcat instance for visualization
        positions_dict: Optional dict mapping ModelInstance to position arrays
                       e.g., {manipulator.model_instance: initial_q, cart_model: cart_pos}
        message: Custom message to display (default: "Visualizing plant state in Meshcat...")
    
    Returns:
        tuple: (preview_diagram, preview_context) for reuse
    
    Example:
        # Preview with default positions (zeros)
        diagram, ctx = visualize_plant_meshcat(plant, scene_graph, meshcat)
        
        # Preview with specific positions
        diagram, ctx = visualize_plant_meshcat(plant, scene_graph, meshcat, 
                     positions_dict={
                         manipulator.model_instance: initial_q,
                         cart_model: cart_init_pos
                     },
                     message="Configured initial state")
    """
    print(colored(f"\n📸 {message}", "cyan"))
    
    # Create a simple diagram just for visualization
    preview_builder = DiagramBuilder()
    preview_builder.AddSystem(plant)
    preview_builder.AddSystem(scene_graph)
    
    # Connect geometry ports
    preview_builder.Connect(
        plant.get_geometry_pose_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id())
    )
    preview_builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port()
    )
    
    # Add visualizer and build
    preview_visualizer = MeshcatVisualizer.AddToBuilder(preview_builder, scene_graph, meshcat)
    preview_diagram = preview_builder.Build()
    
    # Create default context
    preview_context = preview_diagram.CreateDefaultContext()
    preview_plant_context = plant.GetMyContextFromRoot(preview_context)
    
    # Set positions if provided
    if positions_dict:
        for model_instance, positions in positions_dict.items():
            plant.SetPositions(preview_plant_context, model_instance, positions)
    
    # Force visualization update
    preview_diagram.ForcedPublish(preview_context)
    
    print(colored(f"✓ State visualized at: {meshcat.web_url()}", "green"))
    
    return preview_diagram, preview_context
    
    


def add_frames_to_meshcat(meshcat, plant, context, manipulator=None, cart_model=None):
    """
    Add coordinate frame visualizations to Meshcat with visible XYZ triads.
    Creates RGB cylinders showing X (red), Y (green), Z (blue) axes.
    
    Args:
        meshcat: Meshcat instance
        plant: MultibodyPlant
        context: Plant context with current state
        manipulator: CupManipulator instance (optional)
        cart_model: Cart model instance (optional)
    
    Returns:
        frame_list: List of (frame_name, frame, length) tuples for updating
    """
    from pydrake.all import RigidTransform, RotationMatrix, Cylinder, Rgba
    from pydrake.multibody.tree import FrameIndex
    
    # Helper function to create a coordinate frame triad
    def add_frame_triad(meshcat, path, length=0.1, opacity=1.0):
        """Add XYZ coordinate frame to Meshcat with RGB colors.
        
        Args:
            meshcat: Meshcat instance
            path: Path for the frame (e.g., "/Frames/World")
            length: Length of axes
            opacity: Transparency (0=transparent, 1=opaque)
        """
        # Standard RGB colors
        x_color = Rgba(1.0, 0.0, 0.0, opacity)  # Red
        y_color = Rgba(0.0, 1.0, 0.0, opacity)  # Green
        z_color = Rgba(0.0, 0.0, 1.0, opacity)  # Blue
        
        radius = length * 0.015  # Cylinder radius proportional to length
        
        # X-axis (red) - rotate 90° around Y to align with +X
        meshcat.SetObject(f"{path}/X", Cylinder(radius=radius, length=length),
                        rgba=x_color)
        meshcat.SetTransform(f"{path}/X", 
                           RigidTransform(RotationMatrix.MakeYRotation(np.pi/2), 
                                        [length/2, 0, 0]))
        
        # Y-axis (green) - rotate -90° around X to align with +Y
        meshcat.SetObject(f"{path}/Y", Cylinder(radius=radius, length=length),
                        rgba=y_color)
        meshcat.SetTransform(f"{path}/Y", 
                           RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2), 
                                        [0, length/2, 0]))
        
        # Z-axis (blue) - already aligned with +Z
        meshcat.SetObject(f"{path}/Z", Cylinder(radius=radius, length=length),
                        rgba=z_color)
        meshcat.SetTransform(f"{path}/Z", 
                           RigidTransform([0, 0, length/2]))
    
    # Add world frame at origin
    add_frame_triad(meshcat, "/Frames/World", length=0.20)
    meshcat.SetTransform("/Frames/World", RigidTransform())
    
    # Frame list to return for updates
    frame_list = []
    
    # Add all frames from the plant
    for i in range(plant.num_frames()):
        frame = plant.get_frame(FrameIndex(i))
        frame_name = frame.name()
        
        # Skip world frame (already added)
        if frame_name == "world":
            continue
        
        # Determine frame length based on frame type
        if "link" in frame_name.lower() or "cup_center" in frame_name.lower():
            length = 0.15  # Manipulator links and EE
        elif "cart" in frame_name.lower():
            length = 0.12  # Cart frame
        elif "pendulum" in frame_name.lower() or "gimbal" in frame_name.lower():
            length = 0.10  # Pendulum frames
        else:
            length = 0.08  # Other frames
        
        # Add frame triad
        path = f"/Frames/{frame_name}"
        add_frame_triad(meshcat, path, length=length)
        
        # Update frame position
        X_WF = plant.CalcRelativeTransform(context, plant.world_frame(), frame)
        meshcat.SetTransform(path, X_WF)
        
        # Store for updates
        frame_list.append((frame_name, frame, length))
    
    print(colored("✓ Coordinate frame triads added to Meshcat", "green"))
    print(colored("  Legend: X=Red, Y=Green, Z=Blue", "yellow"))
    
    return frame_list


def plot_frames_top_view(plant, context, manipulator, cart_model, title="Frame Orientation (Top View)"):
    """
    Plot coordinate frames of manipulator and cart from top view (looking down Z-axis).
    
    Args:
        plant: MultibodyPlant
        context: Plant context with current state
        manipulator: CupManipulator instance
        cart_model: Cart model instance
        title: Plot title
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Helper function to draw frame axes
    def draw_frame_2d(ax, origin, rotation_matrix, scale=0.3, colors=['r', 'g'], labels=['X', 'Z'], 
                     axis_indices=[0, 2], alpha=1.0):
        """
        Draw coordinate frame in 2D projection.
        
        Args:
            ax: Matplotlib axis
            origin: 2D origin position [x, y] where frame is located in plot
            rotation_matrix: 3D rotation matrix (3x3)
            scale: Arrow length
            colors: Colors for each axis to draw
            labels: Labels for each axis
            axis_indices: Which columns of rotation matrix to draw (e.g., [0,2] for X-Z plane)
            alpha: Transparency
        """
        # Extract which components to use for 2D plotting
        # e.g., for X-Z plane: use components [0,2] of each 3D vector
        for idx, (axis_idx, color, label) in enumerate(zip(axis_indices, colors, labels)):
            # Get the 3D axis vector from rotation matrix
            axis_3d = rotation_matrix[:, axis_idx] * scale
            # Project onto 2D using axis_indices (e.g., [X, Z] components)
            axis_2d = np.array([axis_3d[axis_indices[0]], axis_3d[axis_indices[1]]])
            
            ax.arrow(origin[0], origin[1], 
                    axis_2d[0], axis_2d[1],
                    head_width=scale*0.15, head_length=scale*0.1, 
                    fc=color, ec=color, alpha=alpha, linewidth=2)
            # Label at the end of arrow
            ax.text(origin[0] + axis_2d[0]*1.2, 
                   origin[1] + axis_2d[1]*1.2,
                   label, color=color, fontsize=12, fontweight='bold')
    
    # ============================================================================
    # PLOT 1: Manipulator Frames
    # ============================================================================
    ax1.set_title('Cup Manipulator Frames (Top View: X-Y plane)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X [m]', fontsize=12)
    ax1.set_ylabel('Y [m]', fontsize=12)  # Both systems now in X-Y plane
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # World frame (origin)
    world_origin = np.array([0.0, 0.0, 0.0])
    draw_frame_2d(ax1, world_origin[[0, 1]], np.eye(3), scale=0.2, 
                 colors=['red', 'blue'], labels=['X_w', 'Y_w'], axis_indices=[0, 1], alpha=0.5)
    ax1.plot(0, 0, 'ko', markersize=8, label='World Origin')
    
    # Manipulator base frame
    base_frame = plant.GetFrameByName("base_mount_manipulator", manipulator.model_instance)
    X_WB = plant.CalcRelativeTransform(context, plant.world_frame(), base_frame)
    base_pos = X_WB.translation()
    base_rot = X_WB.rotation().matrix()
    draw_frame_2d(ax1, base_pos[[0, 1]], base_rot, scale=0.25,
                 colors=['darkred', 'darkblue'], labels=['X_b', 'Y_b'], axis_indices=[0, 1], alpha=0.7)
    ax1.plot(base_pos[0], base_pos[1], 'rs', markersize=10, label='Base Frame')
    
    # Link1 frame
    link1_frame = plant.GetFrameByName("link1", manipulator.model_instance)
    X_WL1 = plant.CalcRelativeTransform(context, plant.world_frame(), link1_frame)
    link1_pos = X_WL1.translation()
    link1_rot = X_WL1.rotation().matrix()
    draw_frame_2d(ax1, link1_pos[[0, 1]], link1_rot, scale=0.3,
                 colors=['crimson', 'cyan'], labels=['X_1', 'Y_1'], axis_indices=[0, 1], alpha=0.9)
    ax1.plot(link1_pos[0], link1_pos[1], 'go', markersize=10, label='Link1 Frame')
    
    # Link2 (EE) frame
    link2_frame = plant.GetFrameByName(manipulator.LINK2_NAME, manipulator.model_instance)
    X_WL2 = plant.CalcRelativeTransform(context, plant.world_frame(), link2_frame)
    link2_pos = X_WL2.translation()
    link2_rot = X_WL2.rotation().matrix()
    draw_frame_2d(ax1, link2_pos[[0, 1]], link2_rot, scale=0.35,
                 colors=['orangered', 'deepskyblue'], labels=['X_2', 'Y_2'], axis_indices=[0, 1])
    ax1.plot(link2_pos[0], link2_pos[1], 'mo', markersize=10, label='Link2 Frame')
    
    # EE position using cup_center frame
    ee_pos = manipulator.get_end_effector_position(plant, context)
    # Get cup_center frame rotation
    cup_center_frame = manipulator.get_end_effector_frame(plant)
    X_WEE = plant.CalcRelativeTransform(context, plant.world_frame(), cup_center_frame)
    ee_rot = X_WEE.rotation().matrix()
    # Draw EE frame
    draw_frame_2d(ax1, ee_pos[[0, 1]], ee_rot, scale=0.25,
                 colors=['gold', 'lime'], labels=['X_ee', 'Y_ee'], axis_indices=[0, 1], alpha=0.8)
    ax1.plot(ee_pos[0], ee_pos[1], 'r*', markersize=20, label='EE Position (cup center)', zorder=10)
    
    ax1.legend(loc='upper right', fontsize=9)
    
    # ============================================================================
    # PLOT 2: Cart Frames  
    # ============================================================================
    ax2.set_title('Cart-Pendulum Frames (Top View: X-Y plane)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X [m]', fontsize=12)
    ax2.set_ylabel('Y [m]', fontsize=12)  # Cart works in X-Y plane
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # World frame
    draw_frame_2d(ax2, world_origin[[0, 1]], np.eye(3), scale=0.2,
                 colors=['red', 'green'], labels=['X_w', 'Y_w'], axis_indices=[0, 1], alpha=0.5)
    ax2.plot(0, 0, 'ko', markersize=8, label='World Origin')
    
    # Cart frame
    try:
        cart_body = plant.GetBodyByName("cart", cart_model)
        cart_frame = cart_body.body_frame()
        X_WCart = plant.CalcRelativeTransform(context, plant.world_frame(), cart_frame)
        cart_pos = X_WCart.translation()
        cart_rot = X_WCart.rotation().matrix()
        draw_frame_2d(ax2, cart_pos[[0, 1]], cart_rot, scale=0.3,
                     colors=['purple', 'orange'], labels=['X_c', 'Y_c'], axis_indices=[0, 1])
        ax2.plot(cart_pos[0], cart_pos[1], 'bs', markersize=12, label='Cart Frame')
        
        # Mark cart position
        ax2.plot(cart_pos[0], cart_pos[1], 'b*', markersize=20, label='Cart Position', zorder=10)
    except Exception as e:
        print(colored(f"Warning: Could not get cart frame: {e}", "yellow"))
    
    # Try to get pendulum frame
    try:
        pend_body = plant.GetBodyByName("pendulum", cart_model)
        pend_frame = pend_body.body_frame()
        X_WPend = plant.CalcRelativeTransform(context, plant.world_frame(), pend_frame)
        pend_pos = X_WPend.translation()
        pend_rot = X_WPend.rotation().matrix()
        draw_frame_2d(ax2, pend_pos[[0, 1]], pend_rot, scale=0.25,
                     colors=['darkviolet', 'gold'], labels=['X_p', 'Y_p'], axis_indices=[0, 1], alpha=0.7)
        ax2.plot(pend_pos[0], pend_pos[1], 'mo', markersize=10, label='Pendulum Frame')
    except Exception as e:
        print(colored(f"Info: No pendulum frame found: {e}", "cyan"))
    
    ax2.legend(loc='upper right', fontsize=9)
    
    # ============================================================================
    # PLOT 3: Combined View - All Frames with Offset to Avoid Overlap
    # ============================================================================
    ax3.set_title('All Frames Combined (Side View: X-Z plane)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X [m]', fontsize=12)
    ax3.set_ylabel('Z [m]', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # World frame at origin
    draw_frame_2d(ax3, world_origin[[0, 2]], np.eye(3), scale=0.15, 
                 colors=['red', 'blue'], labels=['W_x', 'W_z'], axis_indices=[0, 2], alpha=0.4)
    ax3.plot(0, 0, 'ko', markersize=6, label='World Origin', alpha=0.5)
    
    # Manipulator frames (in X-Z plane, actual positions)
    draw_frame_2d(ax3, base_pos[[0, 2]], base_rot, scale=0.18,
                 colors=['darkred', 'darkblue'], labels=['B_x', 'B_z'], axis_indices=[0, 2], alpha=0.6)
    ax3.plot(base_pos[0], base_pos[2], 'r^', markersize=7, label='Base', alpha=0.7)
    
    draw_frame_2d(ax3, link1_pos[[0, 2]], link1_rot, scale=0.20,
                 colors=['crimson', 'cyan'], labels=['L1_x', 'L1_z'], axis_indices=[0, 2], alpha=0.7)
    ax3.plot(link1_pos[0], link1_pos[2], 'gs', markersize=7, label='Link1', alpha=0.7)
    
    draw_frame_2d(ax3, link2_pos[[0, 2]], link2_rot, scale=0.22,
                 colors=['orangered', 'deepskyblue'], labels=['L2_x', 'L2_z'], axis_indices=[0, 2], alpha=0.8)
    ax3.plot(link2_pos[0], link2_pos[2], 'mo', markersize=7, label='Link2', alpha=0.7)
    
    # EE position with offset (cup center)
    # Draw EE frame at offset position
    draw_frame_2d(ax3, ee_pos[[0, 2]], link2_rot, scale=0.18,
                 colors=['gold', 'lime'], labels=['EE_x', 'EE_z'], axis_indices=[0, 2], alpha=0.9)
    ax3.plot(ee_pos[0], ee_pos[2], 'r*', markersize=15, label='EE (cup center)', zorder=10)
    
    # Cart frame - plot in X-Z plane at its Y position (shifted vertically for visibility)
    try:
        # Cart is at height z_offset, position (cart_x, cart_y) in X-Y plane
        # Map cart's X-Y position to X-Z for visualization: (cart_X, cart_Y) → (cart_X, cart_Y_as_Z)
        cart_viz_pos = np.array([cart_pos[0], cart_pos[1]])  # Use Y as Z for visualization
        draw_frame_2d(ax3, cart_viz_pos, cart_rot, scale=0.20,
                     colors=['purple', 'orange'], labels=['C_x', 'C_y'], axis_indices=[0, 1], alpha=0.8)
        ax3.plot(cart_viz_pos[0], cart_viz_pos[1], 'bd', markersize=9, label='Cart (X-Y plane)', zorder=9)
        
        # Draw line connecting EE and Cart to show mapping
        ax3.plot([ee_pos[0], cart_viz_pos[0]], [ee_pos[2], cart_viz_pos[1]], 
                'k--', linewidth=1.5, alpha=0.4, label='EE[X,Z]→Cart[X,Y]')
        
        # Annotate the mapping
        mid_x = (ee_pos[0] + cart_viz_pos[0]) / 2
        mid_z = (ee_pos[2] + cart_viz_pos[1]) / 2
        ax3.annotate('X-Z to X-Y mapping', xy=(mid_x, mid_z), fontsize=9, 
                    ha='center', color='black', alpha=0.6,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    except Exception as e:
        print(colored(f"Warning: Could not plot cart in combined view: {e}", "yellow"))
    
    # Add annotations for clarity
    ax3.text(0.02, 0.98, 'Manipulator: X-Z plane\nCart: X-Y plane (Y shown as Z)', 
            transform=ax3.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax3.legend(loc='upper right', fontsize=8, ncol=2)
    
    # Overall figure title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    print(colored(f"\n📊 Frame visualization plotted with 3 views", "green"))
    print(colored(f"   Left: Manipulator frames in X-Z plane (after 180° URDF flip)", "cyan"))
    print(colored(f"   Middle: Cart-Pendulum frames in X-Y plane", "cyan"))
    print(colored(f"   Right: Combined view showing all frames and X-Z→X-Y mapping", "cyan"))
    
    return fig


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_lqr_manip_ee_traj_track_results(t, state_data, ref_data, cart_traj_data, 
                                         ee_positions, ee_velocities, force_data, 
                                         impedance_data, manip_state_data, 
                                         manip_desired_state_data, manip_js_torque_data, config):
    """
    Generate comprehensive plots for LQR manipulator end-effector trajectory tracking.
    
    Args:
        t: Time vector
        state_data: Cart-pendulum state [x, y, α, β, ẋ, ẏ, α̇, β̇]
        ref_data: ZFT reference state [x_ref, y_ref, ẋ_ref, ẏ_ref]
        cart_traj_data: Cart trajectory sent to manipulator [x, y, ẋ, ẏ]
        ee_positions: End-effector positions [x_EE, y_EE]
        ee_velocities: End-effector velocities [ẋ_EE, ẏ_EE]
        force_data: Muscle forces [F_x, F_y]
        impedance_data: Impedance forces [F_x_imp, F_y_imp]
        manip_state_data: Manipulator state [q1, q2, q̇1, q̇2]
        manip_desired_state_data: Desired manipulator state from IK [q1_d, q2_d, q̇1_d, q̇2_d]
        manip_js_torque_data: Joint-space controller torques [τ1, τ2]
        config: SimulationConfig with target_x, target_y attributes
        args: Command-line arguments (target_x, target_y)
    """
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(5, 4, figure=fig)
    
    # Row 1: Cart position
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, state_data[0, :], 'b-', label='x (cart)', linewidth=2)
    ax1.plot(t, ref_data[0, :], 'r--', label='x_ref', linewidth=1.5)
    ax1.plot(t, cart_traj_data[0, :], 'c-.', label='x (to manip)', linewidth=1.5, alpha=0.7)
    ax1.plot(t, ee_positions[0, :], 'g:', label='x_EE', linewidth=2)
    ax1.axhline(config.target_x, color='m', linestyle=':', label='target')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('X Position [m]')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Cart X Position')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, state_data[1, :], 'b-', label='y (cart)', linewidth=2)
    ax2.plot(t, ref_data[1, :], 'r--', label='y_ref', linewidth=1.5)
    ax2.plot(t, cart_traj_data[1, :], 'c-.', label='y (to manip)', linewidth=1.5, alpha=0.7)
    ax2.plot(t, ee_positions[1, :], 'g:', label='y_EE', linewidth=2)
    ax2.axhline(config.target_y, color='m', linestyle=':', label='target')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Y Position [m]')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Cart Y Position')
    
    # 2D trajectory
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(state_data[0, :], state_data[1, :], 'b-', label='cart', linewidth=2)
    ax3.plot(ref_data[0, :], ref_data[1, :], 'r--', label='reference', linewidth=1.5)
    ax3.plot(cart_traj_data[0, :], cart_traj_data[1, :], 'c-.', label='to manip', linewidth=1.5, alpha=0.7)
    ax3.plot(ee_positions[0, :], ee_positions[1, :], 'g:', label='EE', linewidth=2)
    ax3.plot(config.target_x, config.target_y, 'm*', markersize=15, label='target')
    ax3.plot(state_data[0, 0], state_data[1, 0], 'ko', markersize=8, label='start')
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    ax3.set_title('2D Trajectory')
    
    # Tracking error
    ax4 = fig.add_subplot(gs[0, 3])
    error_x = state_data[0, :] - ref_data[0, :]
    error_y = state_data[1, :] - ref_data[1, :]
    error_mag = np.sqrt(error_x**2 + error_y**2)
    ax4.plot(t, error_mag, 'r-', linewidth=2)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Tracking Error [m]')
    ax4.grid(True)
    ax4.set_title('Position Tracking Error')
    
    # Row 2: Cart velocity
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.plot(t, state_data[4, :], 'b-', label='ẋ (cart)', linewidth=2)
    ax5.plot(t, ref_data[2, :], 'r--', label='ẋ_ref', linewidth=1.5)
    ax5.plot(t, cart_traj_data[2, :], 'c-.', label='ẋ (to manip)', linewidth=1.5, alpha=0.7)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('X Velocity [m/s]')
    ax5.legend()
    ax5.grid(True)
    ax5.set_title('Cart X Velocity')
    
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.plot(t, state_data[5, :], 'b-', label='ẏ (cart)', linewidth=2)
    ax6.plot(t, ref_data[3, :], 'r--', label='ẏ_ref', linewidth=1.5)
    ax6.plot(t, cart_traj_data[3, :], 'c-.', label='ẏ (to manip)', linewidth=1.5, alpha=0.7)
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Y Velocity [m/s]')
    ax6.legend()
    ax6.grid(True)
    ax6.set_title('Cart Y Velocity')
    
    # Combined velocity magnitude
    ax7 = fig.add_subplot(gs[1, 2])
    vel_cart = np.sqrt(state_data[4, :]**2 + state_data[5, :]**2)
    vel_ref = np.sqrt(ref_data[2, :]**2 + ref_data[3, :]**2)
    vel_to_manip = np.sqrt(cart_traj_data[2, :]**2 + cart_traj_data[3, :]**2)
    ax7.plot(t, vel_cart, 'b-', label='|v| cart', linewidth=2)
    ax7.plot(t, vel_ref, 'r--', label='|v| ref', linewidth=1.5)
    ax7.plot(t, vel_to_manip, 'c-.', label='|v| to manip', linewidth=1.5, alpha=0.7)
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Velocity Magnitude [m/s]')
    ax7.legend()
    ax7.grid(True)
    ax7.set_title('Velocity Magnitude')
    
    # Velocity tracking error
    ax8 = fig.add_subplot(gs[1, 3])
    vel_error = vel_cart - vel_ref
    ax8.plot(t, vel_error, 'r-', linewidth=2)
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('Velocity Error [m/s]')
    ax8.grid(True)
    ax8.set_title('Velocity Tracking Error')
    
    # Row 3: Pendulum angles and angular velocities
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.plot(t, np.rad2deg(state_data[2, :]), 'b-', label='pitch (α)', linewidth=2)
    ax9.set_xlabel('Time [s]')
    ax9.set_ylabel('Pitch Angle [deg]')
    ax9.legend()
    ax9.grid(True)
    ax9.set_title('Pendulum Pitch Angle')
    
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.plot(t, np.rad2deg(state_data[3, :]), 'r-', label='roll (β)', linewidth=2)
    ax10.set_xlabel('Time [s]')
    ax10.set_ylabel('Roll Angle [deg]')
    ax10.legend()
    ax10.grid(True)
    ax10.set_title('Pendulum Roll Angle')
    
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.plot(t, np.rad2deg(state_data[6, :]), 'b-', label='α̇', linewidth=2)
    ax11.set_xlabel('Time [s]')
    ax11.set_ylabel('Pitch Angular Velocity [deg/s]')
    ax11.legend()
    ax11.grid(True)
    ax11.set_title('Pendulum Pitch Angular Velocity')
    
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.plot(t, np.rad2deg(state_data[7, :]), 'r-', label='β̇', linewidth=2)
    ax12.set_xlabel('Time [s]')
    ax12.set_ylabel('Roll Angular Velocity [deg/s]')
    ax12.legend()
    ax12.grid(True)
    ax12.set_title('Pendulum Roll Angular Velocity')
    
    # Row 4: Forces
    ax13 = fig.add_subplot(gs[3, 0])
    ax13.plot(t, force_data[0, :], 'b-', label='F_x (muscle)', linewidth=2)
    ax13.plot(t, impedance_data[0, :], 'c--', label='F_x (impedance)', linewidth=1.5, alpha=0.7)
    ax13.set_xlabel('Time [s]')
    ax13.set_ylabel('Force X [N]')
    ax13.legend()
    ax13.grid(True)
    ax13.set_title('X-Direction Forces')
    
    ax14 = fig.add_subplot(gs[3, 1])
    ax14.plot(t, force_data[1, :], 'r-', label='F_y (muscle)', linewidth=2)
    ax14.plot(t, impedance_data[1, :], 'm--', label='F_y (impedance)', linewidth=1.5, alpha=0.7)
    ax14.set_xlabel('Time [s]')
    ax14.set_ylabel('Force Y [N]')
    ax14.legend()
    ax14.grid(True)
    ax14.set_title('Y-Direction Forces')
    
    # Force magnitude
    ax15 = fig.add_subplot(gs[3, 2])
    force_muscle_mag = np.sqrt(force_data[0, :]**2 + force_data[1, :]**2)
    force_impedance_mag = np.sqrt(impedance_data[0, :]**2 + impedance_data[1, :]**2)
    ax15.plot(t, force_muscle_mag, 'b-', label='|F| muscle', linewidth=2)
    ax15.plot(t, force_impedance_mag, 'c--', label='|F| impedance', linewidth=1.5, alpha=0.7)
    ax15.set_xlabel('Time [s]')
    ax15.set_ylabel('Force Magnitude [N]')
    ax15.legend()
    ax15.grid(True)
    ax15.set_title('Force Magnitude')
    
    # Energy-like metric
    ax16 = fig.add_subplot(gs[3, 3])
    cart_kinetic = 0.5 * (state_data[4, :]**2 + state_data[5, :]**2)
    ax16.plot(t, cart_kinetic, 'b-', linewidth=2)
    ax16.set_xlabel('Time [s]')
    ax16.set_ylabel('Kinetic Energy (cart) [normalized]')
    ax16.grid(True)
    ax16.set_title('Cart Kinetic Energy')
    
    # Row 5: Manipulator state (joint angles and velocities, EE position/velocity)
    ax17 = fig.add_subplot(gs[4, 0])
    q1_deg = np.rad2deg(manip_state_data[0, :])
    q2_deg = np.rad2deg(manip_state_data[1, :])
    q1_des_deg = np.rad2deg(manip_desired_state_data[0, :])
    q2_des_deg = np.rad2deg(manip_desired_state_data[1, :])
    ax17.plot(t, q1_deg, 'b-', linewidth=2.5, label='q1 actual', alpha=0.8)
    ax17.plot(t, q2_deg, 'r-', linewidth=2.5, label='q2 actual', alpha=0.8)
    ax17.plot(t, q1_des_deg, 'b--', linewidth=1.5, label='q1 desired (IK)', alpha=0.7)
    ax17.plot(t, q2_des_deg, 'r--', linewidth=1.5, label='q2 desired (IK)', alpha=0.7)
    ax17.set_xlabel('Time [s]')
    ax17.set_ylabel('Joint Angles [deg]')
    ax17.legend(fontsize=8)
    ax17.grid(True)
    ax17.set_title('Manipulator Joint Angles: Actual vs Desired (IK from cart)')
    
    ax18 = fig.add_subplot(gs[4, 1])
    q1_dot_deg = np.rad2deg(manip_state_data[2, :])
    q2_dot_deg = np.rad2deg(manip_state_data[3, :])
    q1_dot_des_deg = np.rad2deg(manip_desired_state_data[2, :])
    q2_dot_des_deg = np.rad2deg(manip_desired_state_data[3, :])
    ax18.plot(t, q1_dot_deg, 'b-', linewidth=2.5, label='q̇1 actual', alpha=0.8)
    ax18.plot(t, q2_dot_deg, 'r-', linewidth=2.5, label='q̇2 actual', alpha=0.8)
    ax18.plot(t, q1_dot_des_deg, 'b--', linewidth=1.5, label='q̇1 desired (IK)', alpha=0.7)
    ax18.plot(t, q2_dot_des_deg, 'r--', linewidth=1.5, label='q̇2 desired (IK)', alpha=0.7)
    ax18.set_xlabel('Time [s]')
    ax18.set_ylabel('Joint Velocities [deg/s]')
    ax18.legend(fontsize=8)
    ax18.grid(True)
    ax18.set_title('Manipulator Joint Velocities: Actual vs Desired (IK from cart)')
    
    ax19 = fig.add_subplot(gs[4, 2])
    ax19.plot(t, ee_positions[0, :], 'b-', linewidth=2, label='EE x')
    ax19.plot(t, ee_positions[1, :], 'r-', linewidth=2, label='EE y')
    ax19.plot(t, state_data[0, :], 'b:', linewidth=1.5, alpha=0.7, label='cart x')
    ax19.plot(t, state_data[1, :], 'r:', linewidth=1.5, alpha=0.7, label='cart y')
    ax19.set_xlabel('Time [s]')
    ax19.set_ylabel('EE Position [m]')
    ax19.legend()
    ax19.grid(True)
    ax19.set_title('Manipulator End-Effector Position vs Cart')
    
    ax20 = fig.add_subplot(gs[4, 3])
    ax20.plot(t, ee_velocities[0, :], 'b-', linewidth=2, label='EE ẋ')
    ax20.plot(t, ee_velocities[1, :], 'r-', linewidth=2, label='EE ẏ')
    ax20.plot(t, state_data[4, :], 'b:', linewidth=1.5, alpha=0.7, label='cart ẋ')
    ax20.plot(t, state_data[5, :], 'r:', linewidth=1.5, alpha=0.7, label='cart ẏ')
    ax20.set_xlabel('Time [s]')
    ax20.set_ylabel('EE Velocity [m/s]')
    ax20.legend()
    ax20.grid(True)
    ax20.set_title('Manipulator End-Effector Velocity vs Cart')
    
    plt.tight_layout()
    
    # Save plots
    plot_path = 'plots/lqr_manip_ee_traj_track_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(colored(f"✓ Main plots saved to {plot_path}", "green"))
    plt.show()

    # =======================================================================
    # TORQUE COMPARISON FIGURE (COMMENTED OUT - EE Controller disabled)
    # =======================================================================
    # fig_torque = plt.figure(figsize=(16, 10))
    # gs_torque = GridSpec(3, 2, figure=fig_torque)
    # 
    # # Torque comparison τ1
    # ax_tau1 = fig_torque.add_subplot(gs_torque[0, 0])
    # ax_tau1.plot(t, manip_torque_data[1, :], 'b-', label='τ1 (EE-Controller)', linewidth=2)
    # ax_tau1.plot(t, manip_js_torque_data[1, :], 'r--', label='τ1 (JS-Controller)', linewidth=2)
    # ax_tau1.axhline(100, color='k', linestyle=':', alpha=0.5, label='saturation')
    # ax_tau1.axhline(-100, color='k', linestyle=':', alpha=0.5)
    # ax_tau1.set_xlabel('Time [s]')
    # ax_tau1.set_ylabel('Torque τ1 [Nm]')
    # ax_tau1.legend()
    # ax_tau1.grid(True)
    # ax_tau1.set_title('Joint 1 Torque Comparison')
    # 
    # # Torque comparison τ2
    # ax_tau2 = fig_torque.add_subplot(gs_torque[0, 1])
    # ax_tau2.plot(t, manip_torque_data[0, :], 'b-', label='τ2 (EE-Controller)', linewidth=2)
    # ax_tau2.plot(t, manip_js_torque_data[0, :], 'r--', label='τ2 (JS-Controller)', linewidth=2)
    # ax_tau2.axhline(100, color='k', linestyle=':', alpha=0.5, label='saturation')
    # ax_tau2.axhline(-100, color='k', linestyle=':', alpha=0.5)
    # ax_tau2.set_xlabel('Time [s]')
    # ax_tau2.set_ylabel('Torque τ2 [Nm]')
    # ax_tau2.legend()
    # ax_tau2.grid(True)
    # ax_tau2.set_title('Joint 2 Torque Comparison')
    # 
    # # Torque magnitude comparison
    # ax_tau_mag = fig_torque.add_subplot(gs_torque[1, :])
    # tau_ee_mag = np.sqrt(manip_torque_data[0, :]**2 + manip_torque_data[1, :]**2)
    # tau_js_mag = np.sqrt(manip_js_torque_data[0, :]**2 + manip_js_torque_data[1, :]**2)
    # ax_tau_mag.plot(t, tau_ee_mag, 'b-', label='|τ| (EE-Controller)', linewidth=2)
    # ax_tau_mag.plot(t, tau_js_mag, 'r--', label='|τ| (JS-Controller)', linewidth=2)
    # ax_tau_mag.set_xlabel('Time [s]')
    # ax_tau_mag.set_ylabel('Torque Magnitude [Nm]')
    # ax_tau_mag.legend()
    # ax_tau_mag.grid(True)
    # ax_tau_mag.set_title('Torque Magnitude Comparison')
    # 
    # # Torque difference τ1
    # ax_tau_diff1 = fig_torque.add_subplot(gs_torque[2, 0])
    # tau_diff1 = manip_js_torque_data[1, :] - manip_torque_data[1, :]
    # ax_tau_diff1.plot(t, tau_diff1, 'g-', linewidth=2)
    # ax_tau_diff1.axhline(0, color='k', linestyle='--', alpha=0.3)
    # ax_tau_diff1.set_xlabel('Time [s]')
    # ax_tau_diff1.set_ylabel('Δτ1 [Nm]')
    # ax_tau_diff1.grid(True)
    # ax_tau_diff1.set_title('Torque Difference τ1 (JS - EE)')
    # 
    # # Torque difference τ2
    # ax_tau_diff2 = fig_torque.add_subplot(gs_torque[2, 1])
    # tau_diff2 = manip_js_torque_data[0, :] - manip_torque_data[0, :]
    # ax_tau_diff2.plot(t, tau_diff2, 'g-', linewidth=2)
    # ax_tau_diff2.axhline(0, color='k', linestyle='--', alpha=0.3)
    # ax_tau_diff2.set_xlabel('Time [s]')
    # ax_tau_diff2.set_ylabel('Δτ2 [Nm]')
    # ax_tau_diff2.grid(True)
    # ax_tau_diff2.set_title('Torque Difference τ2 (JS - EE)')
    # 
    # fig_torque.tight_layout()
    # torque_path = 'plots/lqr_manip_torque_comparison.png'
    # fig_torque.savefig(torque_path, dpi=150, bbox_inches='tight')
    # print(colored(f"✓ Torque comparison saved to {torque_path}", "green"))


# ============================================================================
# SYSTEM BUILDER CLASS
# ============================================================================

class SystemBuilder:
    """
    Builds a Drake MultibodyPlant with two model instances:
    1. Manipulator (2-DOF cup manipulator)
    2. Cart-Pendulum (4-DOF system)
    
    This class encapsulates the plant setup logic to avoid code duplication
    across different simulation modes.
    
    Attributes:
        builder: DiagramBuilder instance (created during build())
        plant: MultibodyPlant instance (created during build())
        scene_graph: SceneGraph instance (created during build())
    """
    
    def __init__(self, physics_config, manipulator_urdf_path, 
                 manipulator_joint_angles=None, manipulator_damping=(0.1, 0.1)):
        """
        Initialize the system builder with configurations.
        
        Args:
            physics_config: PhysicsConfig for cart-pendulum
            manipulator_urdf_path: Path to manipulator URDF file
            manipulator_joint_angles: Dict of joint names to angles (radians)
            manipulator_damping: Tuple of (q1_damping, q2_damping)
        """
        self.physics_config = physics_config
        self.manipulator_urdf_path = manipulator_urdf_path
        self.manipulator_joint_angles = manipulator_joint_angles or {
            'link1_base': np.deg2rad(0.0),
            'link2_link1': np.deg2rad(20.0),
        }
        self.manipulator_damping = manipulator_damping
        
        # These will be set during build()
        self.builder = None
        self.plant = None
        self.scene_graph = None
        self.weld_cart_to_ee = WELD_CART_TO_MANIP_EE  # Set to True to weld cart to manipulator EE, False for independent actuation
        

    def add_manipulator(self, plant):
        """
        Add manipulator to the plant as a model instance.
        
        Args:
            plant: MultibodyPlant to add manipulator to
            
        Returns:
            CupManipulator instance
        """
        manipulator_config = create_cup_manipulator_config(
            urdf_path=self.manipulator_urdf_path,
            joint_angles=self.manipulator_joint_angles,
            damping=self.manipulator_damping,
        )
        
        # Initialize manipulator and load URDF into plant
        manipulator = CupManipulator(manipulator_config, enable_visualization=True)
        parser = Parser(plant)
        manipulator.load_urdf_to_plant(plant, parser)
        
        # Rotate base -90° around Y to align manipulator with X-Y plane (same as cart)
        # This makes manipulator X-axis → world X-axis, manipulator Z-axis → world Y-axis
        manipulator.weld_base_to_world(plant, orientation=np.array([0.0, 0.0, 0.0]))
        
        # Add actuators and end-effector frame BEFORE finalization
        manipulator.add_joint_actuators(plant)
        manipulator.add_end_effector_frame(plant)
        
        print(colored(f"✓ End-effector frame '{manipulator.EE_FRAME_NAME}' added to manipulator", "green"))
        print(colored(f"  - EE_OFFSET (relative to link2): {manipulator.EE_OFFSET}", "cyan"))
        print(colored(f"✓ Manipulator loaded (ModelInstance: {manipulator.model_instance})", "green"))
        print(colored(f"  - State dimension: 4 (2 positions + 2 velocities)", "cyan"))
        print(colored(f"  - Joints: link1_base, link2_link1", "cyan"))
        
        return manipulator
    
    def add_cart_pendulum_components(self, plant, manipulator=None):
        """
        Add cart-pendulum to the plant as a model instance.
        
        Args:
            plant: MultibodyPlant to add cart-pendulum to
            manipulator: CupManipulator instance (optional, required if weld_cart_to_ee=True)
        Returns:
            Tuple of (cart_pendulum, cart_model)
        """
        # Determine z-offset for cart-pendulum based on URDF joint origin
        z_offset_from_urdf = CupManipulator.EE_XYZ_BASE[2]
        print(colored(f"📍 Using z-offset from URDF: {z_offset_from_urdf:.5f} m", "cyan"))
        
        cart_model = plant.AddModelInstance("cart_pendulum")
        
        # ====================================================================
        # DECISION: WELDED TO EE vs INDEPENDENT ACTUATION
        # ====================================================================
        if self.weld_cart_to_ee:
            if manipulator is None:
                raise ValueError("manipulator must be provided when weld_cart_to_ee=True")
            
            print(colored(f"\n🔗 Mode: Cart WELDED to manipulator EE (dependent control)", "yellow"))
            print(colored(f"  - Kinematic chain: World → Manipulator → EE → Cart → Pendulum", "cyan"))
            print(colored(f"  - Cart follows EE motion (NO independent cart actuators)", "cyan"))
            
            # Create cart-pendulum using CartPendulum2DExtended.build_plant_welded()
            cart_pendulum = CartPendulum2DExtended(self.physics_config, z_offset=z_offset_from_urdf)
            cart_body = cart_pendulum.build_plant_welded(plant, cart_model, register_visuals=True)
            
            # Get the manipulator's EE frame
            ee_frame = manipulator.get_end_effector_frame(plant)
            
            # Weld cart body to EE frame (zero offset = cart center aligns with EE)
            plant.WeldFrames(
                frame_on_parent_F=ee_frame,
                frame_on_child_M=cart_body.body_frame(),
                X_FM=RigidTransform()
            )
            
            print(colored(f"✓ Cart welded to EE frame '{manipulator.EE_FRAME_NAME}'", "green"))
            print(colored(f"  - Cart DOF: 0 (welded, follows EE)", "cyan"))
            print(colored(f"  - Pendulum DOF: 2 (pitch + roll gimbal)", "cyan"))
            print(colored(f"  - NO cart actuators (controlled via manipulator)", "yellow"))
            
        else:
            print(colored(f"\n⚙️ Mode: Cart INDEPENDENT actuation (uncoupled control)", "yellow"))
            print(colored(f"  - Kinematic chain: World → Cart (via sliders) → Pendulum", "cyan"))
            print(colored(f"  - Manipulator: World → Manip → EE (separate tree)", "cyan"))
            print(colored(f"  - Cart has independent actuators (LQR controls both systems)", "cyan"))
            
            # Build cart-pendulum with normal world connection via prismatic joints
            cart_pendulum = CartPendulum2DExtended(self.physics_config, z_offset=z_offset_from_urdf)
            cart_pendulum.build_plant(plant, cart_model)
            
            print(colored(f"✓ Cart-pendulum has independent actuation", "green"))
        
        print(colored(f"✓ Cart-Pendulum created (ModelInstance: {cart_model})", "green"))
        print(colored(f"  - Z-plane height: {z_offset_from_urdf:.5f} m", "cyan"))
        
        return cart_pendulum, cart_model
    
    def finalize_and_print_info(self, plant, manipulator):
        """
        Finalize the plant and print configuration information.
        
        Args:
            plant: MultibodyPlant to finalize
            manipulator: CupManipulator instance for info printing
        """
        plant.Finalize()
        
        print(colored(f"\n✓ Plant finalized with {plant.num_positions()} total positions, "
                     f"{plant.num_velocities()} total velocities", "green"))
        
        # Extract and display manipulator configuration
        config_q1 = self.manipulator_joint_angles['link1_base']
        config_q2 = self.manipulator_joint_angles['link2_link1']
        
        # Calculate EE position at config angles for display
        temp_context = plant.CreateDefaultContext()
        manipulator.set_positions_user_order(plant, temp_context, {
            "link1_base": config_q1,
            "link2_link1": config_q2,
        })
        ee_world_pos = manipulator.get_end_effector_position(plant, temp_context)
        
        print(colored(f"  - EE position in world frame (at config q1={np.rad2deg(config_q1):.1f}°, "
                     f"q2={np.rad2deg(config_q2):.1f}°): {ee_world_pos}", "cyan"))
    
    def build(self, time_step=0.001, meshcat=None):
        """
        Build the complete system with manipulator and cart-pendulum.
        Creates the DiagramBuilder and stores it as a class attribute.
        Adds plant and scene_graph to the builder and connects them.
        
        Args:
            time_step: Simulation time step in seconds
            meshcat: Optional Meshcat instance for printing URL
            
        Returns:
            Tuple of (builder, plant, scene_graph, manipulator, cart_pendulum, cart_model)
        """
        # Create builder, plant, and scene graph (stored as attributes)
        self.builder = DiagramBuilder()
        self.plant = MultibodyPlant(time_step=time_step)
        self.scene_graph = self.builder.AddSystem(SceneGraph())
        self.plant.RegisterAsSourceForSceneGraph(self.scene_graph)
    
        
        # Add manipulator
        manipulator = self.add_manipulator(self.plant)
        
        # Add cart-pendulum
        if self.weld_cart_to_ee:
            print(colored(f"\n⚠️ WARNING: Cart-pendulum will be WELDED to manipulator EE. "
                         f"Cart actuation inputs will be IGNORED.", "red"))
            cart_pendulum, cart_model = self.add_cart_pendulum_components(self.plant, manipulator=manipulator)
        else:            
            print(colored(f"\n⚠️ WARNING: Cart-pendulum will have INDEPENDENT ACTUATION. "
                         f"Ensure control system accounts for this.", "red"))
            cart_pendulum, cart_model = self.add_cart_pendulum_components(self.plant)
        
        # Finalize and print info
        self.finalize_and_print_info(self.plant, manipulator)
        
        # Add plant to builder and connect to scene_graph
        self.builder.AddSystem(self.plant)
        self.builder.Connect(
            self.plant.get_geometry_pose_output_port(),
            self.scene_graph.get_source_pose_port(self.plant.get_source_id())
        )
        self.builder.Connect(
            self.scene_graph.get_query_output_port(),
            self.plant.get_geometry_query_input_port()
        )
        
        # Print meshcat URL if provided
        if meshcat is not None:
            print(colored(f"  - Meshcat will be available at: {meshcat.web_url()}", "cyan"))
        
        return self.builder, self.plant, self.scene_graph, manipulator, cart_pendulum, cart_model


# ============================================================================
# CONTROL SYSTEM BUILDER (STRATEGY PATTERN)
# ============================================================================

class ControlSystemBuilder(ABC):
    """
    Abstract base class for building different control strategies.
    
    Strategy Pattern: Encapsulates control system construction algorithms.
    Each concrete builder creates a specific control architecture:
    - LQR + Muscle Dynamics + ZFT (OFC)
    - PD Control
    - Model Predictive Control
    - etc.
    
    Concrete classes have full freedom to structure control systems.
    Only logging methods must be implemented.
    """
    
    def __init__(self, builder: DiagramBuilder, plant: MultibodyPlant, 
                 cart_model, manipulator):
        """
        Initialize builder with Drake components.
        
        Args:
            builder: DiagramBuilder to add systems to
            plant: MultibodyPlant with cart-pendulum and manipulator
            cart_model: Cart-pendulum ModelInstance
            manipulator: CupManipulator instance
        """
        self.builder = builder
        self.plant = plant
        self.cart_model = cart_model
        self.manipulator = manipulator
        
        # Will store created systems for connection and logging
        self.systems = {}
        self.loggers = {}
    
    @abstractmethod
    def add_loggers(self) -> Dict[str, VectorLogSink]:
        """
        Build data loggers for all control signals.
        
        Must be implemented by concrete classes.
        
        Returns:
            Dictionary of logger names to VectorLogSink instances
        """
        pass
    
    @abstractmethod
    def connect_loggers(self):
        """
        Connect loggers to their signal sources.
        
        Must be implemented by concrete classes.
        """
        pass
    
    def build_and_connect(self):
        """
        Build all control systems and connect them.
        
        Concrete classes should override this to define their own
        control system construction and connection logic.
        
        Returns:
            Tuple of (systems dict, loggers dict)
        """
        print(colored(f"\n🔧 Building control system: {self.__class__.__name__}", "yellow"))
        
        # Build and connect loggers (required by abstract methods)
        self.loggers = self.add_loggers()
        self.connect_loggers()
        
        print(colored(f"✓ Control system built with {len(self.systems)} blocks", "green"))
        
        return self.systems, self.loggers


# ============================================================================
# LQR + OFC CONTROL BUILDER (CONCRETE STRATEGY)
# ============================================================================

class LQRWithOFCOnlyCartPendulumBuilder(ControlSystemBuilder):
    """
    Builds LQR control with Optimal Feedback Control (OFC) architecture:
    
    Cart-Pendulum Control:
    - Muscle Dynamics (low-pass filter on neural commands)
    - ZFT Reference Mass (virtual mass for smooth trajectories)
    - Impedance Force (spring-damper connection to reference)
    - Finite-Horizon LQR (optimal time-varying feedback)
    
    Manipulator Control:
    - IK Solver (converts cart trajectory to joint angles)
    - Computed Torque Controller (joint-space tracking)
    """
    
    def __init__(self, builder: DiagramBuilder, plant: MultibodyPlant, 
                 cart_model, manipulator, config):
        super().__init__(builder, plant, cart_model, manipulator)
        self.config = config
        
        # Pre-compute linearization
        self.A, self.B = build_linearized_system_2d(
            config.physics_config,
            config.impedance_config,
            config.zft_config,
            config.muscle_config
        )
        print(colored(f"✓ Linearized system: A ({self.A.shape}), B ({self.B.shape})", "green"))
    
    def build_and_connect(self):
        """
        Build all control systems and connect them.
        
        Returns:
            Tuple of (systems dict, loggers dict)
        """
        print(colored(f"\n🔧 Building control system: {self.__class__.__name__}", "yellow"))
        
        # Build control systems
        cp_systems = self.add_cart_system_blocks()
        manip_systems = self.add_manipulator_system_blocks()
        
        self.systems.update(cp_systems)
        self.systems.update(manip_systems)
        
        # Connect control loops
        self.connect_cart_pendulum_control()
        self.connect_manipulator_control()
        
        # Build and connect loggers
        self.loggers = self.add_loggers()
        self.connect_loggers()
        
        print(colored(f"✓ Control system built with {len(self.systems)} blocks", "green"))
        
        return self.systems, self.loggers
    
    def add_cart_system_blocks(self) -> Dict[str, LeafSystem]:
        """Build LQR + OFC blocks for cart-pendulum."""
        systems = {}
        
        # Muscle dynamics (2D low-pass filter)
        systems['muscle'] = self.builder.AddSystem(
            MuscleDynamics2D(self.config.muscle_config)
        )
        systems['muscle'].set_name("muscle_dynamics")
        
        # ZFT reference mass (virtual mass dynamics)
        systems['zft'] = self.builder.AddSystem(
            ZFTReferenceMass2D(self.config.zft_config)
        )
        systems['zft'].set_name("zft_reference")
        
        # Impedance force (spring-damper connection)
        systems['impedance'] = self.builder.AddSystem(
            ImpedanceForce2D(self.config.impedance_config)
        )
        systems['impedance'].set_name("impedance_force")
        
        # LQR controller (finite-horizon optimal control)
        systems['lqr'] = self.builder.AddSystem(
            FiniteHorizonLQRController2D(self.A, self.B, LQR_CONFIG)
        )
        systems['lqr'].set_name("lqr_controller")
        
        # ZeroOrderHold (breaks algebraic loop)
        systems['state_hold'] = self.builder.AddSystem(ZeroOrderHold(0.01, 14))
        systems['state_hold'].set_name("state_hold")
        
        # State extraction mux/demux
        systems['cart_state_demux'] = self.builder.AddSystem(Demultiplexer([2, 2, 2, 2]))
        systems['full_state_mux'] = self.builder.AddSystem(Multiplexer([2, 2, 2, 2, 2, 4]))
        systems['cart_state_mux'] = self.builder.AddSystem(Multiplexer([2, 2]))
        
        return systems
    
    def add_manipulator_system_blocks (self) -> Dict[str, LeafSystem]:
        """Build IK + computed torque control for manipulator."""
        systems = {}
        
        # IK solver (cart trajectory → desired joint angles)
        systems['manip_ik'] = self.builder.AddSystem(
            ManipulatorIKDesiredAngles(self.manipulator, self.plant)
        )
        systems['manip_ik'].set_name("manipulator_ik_solver")
        
        # Joint-space computed torque controller
        systems['manip_controller'] = self.builder.AddSystem(
            ComputedTorqueJointSpaceController(
                self.manipulator, self.plant, Kp=200.0, Kd=60.0, tau_max=100.0
            )
        )
        systems['manip_controller'].set_name("manipulator_js_controller")
        
        return systems

    def add_loggers(self) -> Dict[str, VectorLogSink]:
        """Build loggers for LQR + OFC signals."""
        loggers = {}
        
        # Base loggers
        loggers['state'] = self.builder.AddSystem(VectorLogSink(8))
        loggers['state'].set_name("state_logger")
        
        loggers['manip_state'] = self.builder.AddSystem(VectorLogSink(4))
        loggers['manip_state'].set_name("manip_state_logger")
        
        ee_computer = self.builder.AddSystem(
            ManipulatorEEStateComputer(self.plant, self.manipulator)
        )
        ee_computer.set_name("ee_state_computer")
        loggers['ee_state'] = self.builder.AddSystem(VectorLogSink(4))
        loggers['ee_state'].set_name("ee_state_logger")
        loggers['ee_computer'] = ee_computer
        
        # OFC-specific loggers
        loggers['ref'] = self.builder.AddSystem(VectorLogSink(4))
        loggers['ref'].set_name("ref_logger")
        
        loggers['force'] = self.builder.AddSystem(VectorLogSink(2))
        loggers['force'].set_name("force_logger")
        
        loggers['impedance'] = self.builder.AddSystem(VectorLogSink(2))
        loggers['impedance'].set_name("impedance_logger")
        
        loggers['cart_traj'] = self.builder.AddSystem(VectorLogSink(4))
        loggers['cart_traj'].set_name("cart_traj_logger")
        
        loggers['manip_desired'] = self.builder.AddSystem(VectorLogSink(4))
        loggers['manip_desired'].set_name("manip_desired_state_logger")
        
        loggers['manip_torque'] = self.builder.AddSystem(VectorLogSink(2))
        loggers['manip_torque'].set_name("manip_js_torque_logger")
        
        return loggers
    
    def connect_cart_pendulum_control(self):
        """Connect LQR + OFC control loop for cart-pendulum."""
        # Extract systems for convenience
        muscle = self.systems['muscle']
        zft = self.systems['zft']
        impedance = self.systems['impedance']
        lqr = self.systems['lqr']
        state_hold = self.systems['state_hold']
        cart_state_demux = self.systems['cart_state_demux']
        full_state_mux = self.systems['full_state_mux']
        cart_state_mux = self.systems['cart_state_mux']
        
        # ====================================================================
        # STATE EXTRACTION: Plant → Demux → Mux
        # ====================================================================
        # Plant state: q = [x, y, α, β]ᵀ ∈ ℝ⁴, q̇ = [ẋ, ẏ, α̇, β̇]ᵀ ∈ ℝ⁴
        # Full state: s_plant = [q; q̇] ∈ ℝ⁸
        self.builder.Connect(
            self.plant.get_state_output_port(self.cart_model),  # s_plant ∈ ℝ⁸
            cart_state_demux.get_input_port()  # Split into 4 blocks of size 2
        )
        # Demux outputs: [0]→[x,y], [1]→[α,β], [2]→[ẋ,ẏ], [3]→[α̇,β̇]
        
        # Build cart state: s_cart = [x, y, ẋ, ẏ]ᵀ ∈ ℝ⁴
        self.builder.Connect(
            cart_state_demux.get_output_port(0),  # [x, y]ᵀ ∈ ℝ²
            cart_state_mux.get_input_port(0)
        )
        self.builder.Connect(
            cart_state_demux.get_output_port(2),  # [ẋ, ẏ]ᵀ ∈ ℝ²
            cart_state_mux.get_input_port(1)
        )
        # Output: s_cart = [x, y, ẋ, ẏ]ᵀ ∈ ℝ⁴
        
        # ====================================================================
        # ZFT REFERENCE MASS: Dynamics for smooth reference trajectory
        # ====================================================================
        # ZFT dynamics: ṡ_ref = f_zft(s_cart, F_muscle)
        # State: s_ref = [x_ref, y_ref, ẋ_ref, ẏ_ref]ᵀ ∈ ℝ⁴
        self.builder.Connect(
            cart_state_mux.get_output_port(),  # s_cart ∈ ℝ⁴
            zft.get_input_port(0)
        )
        self.builder.Connect(
            muscle.get_output_port(),  # F_muscle = [F_x, F_y]ᵀ ∈ ℝ²
            zft.get_input_port(1)
        )
        # Output: s_ref = [x_ref, y_ref, ẋ_ref, ẏ_ref]ᵀ ∈ ℝ⁴
        
        # ====================================================================
        # IMPEDANCE FORCE: Spring-damper connection to reference
        # ====================================================================
        # Impedance law: F_imp = K(x_ref - x) + B(ẋ_ref - ẋ)
        # where K ∈ ℝ²ˣ² (stiffness), B ∈ ℝ²ˣ² (damping)
        self.builder.Connect(
            cart_state_mux.get_output_port(),  # s_cart = [x, y, ẋ, ẏ]ᵀ ∈ ℝ⁴
            impedance.get_input_port(0)
        )
        self.builder.Connect(
            zft.get_output_port(),  # s_ref = [x_ref, y_ref, ẋ_ref, ẏ_ref]ᵀ ∈ ℝ⁴
            impedance.get_input_port(1)
        )
        # Output: F_imp = [F_imp,x, F_imp,y]ᵀ ∈ ℝ²
        
        # ====================================================================
        # ACTUATION: Impedance force → cart-pendulum
        # ====================================================================
        # Cart-pendulum equations: M(q)q̈ + C(q,q̇)q̇ + G(q) = τ_ext
        # where τ_ext = [F_imp,x, F_imp,y, 0, 0]ᵀ (force on cart, no direct torque on gimbal)
        self.builder.Connect(
            impedance.get_output_port(),  # F_imp ∈ ℝ²
            self.plant.get_actuation_input_port(self.cart_model)
        )
        
        # ====================================================================
        # FULL STATE ASSEMBLY: Build 14D state for LQR
        # ====================================================================
        # [x, y, α, β, ẋ, ẏ, α̇, β̇, F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref]
        for i, port_idx in enumerate([0, 1, 2, 3]):
            self.builder.Connect(
                cart_state_demux.get_output_port(port_idx),
                full_state_mux.get_input_port(i)
            )
        self.builder.Connect(muscle.get_output_port(), full_state_mux.get_input_port(4))
        self.builder.Connect(zft.get_output_port(), full_state_mux.get_input_port(5))
        
        # ====================================================================
        # LQR FEEDBACK LOOP: State → Hold → LQR → Muscle
        # ====================================================================
        self.builder.Connect(full_state_mux.get_output_port(), state_hold.get_input_port())
        self.builder.Connect(state_hold.get_output_port(), lqr.get_input_port())
        self.builder.Connect(lqr.get_output_port(), muscle.get_input_port())
    
    def connect_manipulator_control(self):
        """Connect IK + computed torque control for manipulator."""
        manip_ik = self.systems['manip_ik']
        manip_controller = self.systems['manip_controller']
        cart_state_mux = self.systems['cart_state_mux']
        
        # ====================================================================
        # IK SOLVER: Cart trajectory → desired joint angles
        # ====================================================================
        self.builder.Connect(
            cart_state_mux.get_output_port(),
            manip_ik.get_input_port(0)  # desired cart trajectory
        )
        self.builder.Connect(
            self.plant.get_state_output_port(),
            manip_ik.get_input_port(1)  # full plant state
        )
        
        # ====================================================================
        # COMPUTED TORQUE CONTROLLER: IK → Controller → Actuators
        # ====================================================================
        self.builder.Connect(
            manip_ik.get_output_port(),
            manip_controller.get_input_port(0)  # desired joint state
        )
        self.builder.Connect(
            self.plant.get_state_output_port(self.manipulator.model_instance),
            manip_controller.get_input_port(1)  # current joint state
        )
        self.builder.Connect(
            manip_controller.get_output_port(),
            self.plant.get_actuation_input_port(self.manipulator.model_instance)
        )
    
    
    
    def connect_loggers(self):
        """Connect loggers to OFC-specific signals."""
        # Connect base loggers
        self.builder.Connect(
            self.plant.get_state_output_port(self.cart_model),
            self.loggers['state'].get_input_port()
        )
        self.builder.Connect(
            self.plant.get_state_output_port(self.manipulator.model_instance),
            self.loggers['manip_state'].get_input_port()
        )
        self.builder.Connect(
            self.plant.get_state_output_port(self.manipulator.model_instance),
            self.loggers['ee_computer'].get_input_port(0)
        )
        self.builder.Connect(
            self.loggers['ee_computer'].get_output_port(),
            self.loggers['ee_state'].get_input_port()
        )
        
        # Connect OFC-specific loggers
        self.builder.Connect(
            self.systems['zft'].get_output_port(),
            self.loggers['ref'].get_input_port()
        )
        self.builder.Connect(
            self.systems['muscle'].get_output_port(),
            self.loggers['force'].get_input_port()
        )
        self.builder.Connect(
            self.systems['impedance'].get_output_port(),
            self.loggers['impedance'].get_input_port()
        )
        self.builder.Connect(
            self.systems['cart_state_mux'].get_output_port(),
            self.loggers['cart_traj'].get_input_port()
        )
        self.builder.Connect(
            self.systems['manip_ik'].get_output_port(),
            self.loggers['manip_desired'].get_input_port()
        )
        self.builder.Connect(
            self.systems['manip_controller'].get_output_port(),
            self.loggers['manip_torque'].get_input_port()
        )

class LQRWithOFCForCompleteSystem(ControlSystemBuilder):
    """
    Builds LQR control with Optimal Feedback Control (OFC) architecture:
    
    Cart-Pendulum Control:
    - Muscle Dynamics (low-pass filter on neural commands)
    - ZFT Reference Mass (virtual mass for smooth trajectories)
    - Impedance Force (spring-damper connection to reference)
    - Finite-Horizon LQR (optimal time-varying feedback)
    
    Manipulator Control:
    - IK Solver (converts cart trajectory to joint angles)
    - Computed Torque Controller (joint-space tracking)
    """
    
    def __init__(self, builder: DiagramBuilder, plant: MultibodyPlant, 
                 cart_model, manipulator, config):
        super().__init__(builder, plant, cart_model, manipulator)
        self.config = config
        
        # Pre-compute linearization for welded system
        print(colored("\n🔧 Linearizing welded cart-pendulum-manipulator system...", "yellow"))
        self.A, self.B = build_linearized_for_complete_system_2d(
            plant=plant,
            manipulator=manipulator,
            cart_model=cart_model,
            physics_config=config.physics_config,
        )
        print(colored(f"✓ Linearized system: A ({self.A.shape}), B ({self.B.shape})", "green"))
    
    def build_and_connect(self):
        """
        Build all control systems and connect them.
        
        Returns:
            Tuple of (systems dict, loggers dict)
        """
        print(colored(f"\n🔧 Building control system: {self.__class__.__name__}", "yellow"))
        
        # Build control blocks
        self.add_cart_pen_manip_lqr_computed_torque_blocks()
        
        # Connect control loops
        self.connect_system_control()
        
        # Build and connect loggers
        self.loggers = self.add_loggers()
        self.connect_loggers()
        
        print(colored(f"✓ Control system built with {len(self.systems)} blocks", "green"))
        
        return self.systems, self.loggers

    def add_cart_pen_manip_lqr_computed_torque_blocks(self) -> Dict[str, object]:
        """Create and name all blocks used by connect_system_control()."""
        systems: Dict[str, object] = {}

        # ----------------------------
        # State handling: x -> (q, v)
        # ----------------------------
        systems["plant_state_demux"] = self.builder.AddSystem(
            Demultiplexer([self.config.nq_total, self.config.nv_total])
        )
        systems["plant_state_demux"].set_name("plant_state_demux_q_v")

        # A held copy for controller (avoid double-feeding the same demux)
        systems["state_hold"] = self.builder.AddSystem(
            ZeroOrderHold(self.config.controller_dt, self.config.nq_total + self.config.nv_total)
        )
        systems["state_hold"].set_name("state_hold_x")

        systems["plant_state_demux_hold"] = self.builder.AddSystem(
            Demultiplexer([self.config.nq_total, self.config.nv_total])
        )
        systems["plant_state_demux_hold"].set_name("plant_state_demux_hold_q_v")

        # Optional split into arm vs pendulum (kept if you need it later)
        systems["q_demux"] = self.builder.AddSystem(
            Demultiplexer([self.config.nq_arm, self.config.nq_pend])
        )
        systems["q_demux"].set_name("q_demux_arm_pend")

        systems["v_demux"] = self.builder.AddSystem(
            Demultiplexer([self.config.nv_arm, self.config.nv_pend])
        )
        systems["v_demux"].set_name("v_demux_arm_pend")

        # ----------------------------
        # Kinematics: (q,v) -> (p, pdot)
        # ----------------------------
        systems["ee_kin"] = self.builder.AddSystem(
            EndEffectorKinematics2D(self.config.ee_kin_config)
        )
        systems["ee_kin"].set_name("end_effector_kinematics_2d")

        # ----------------------------
        # Muscle: u -> F
        # ----------------------------
        systems["muscle"] = self.builder.AddSystem(
            MuscleDynamics2D(self.config.muscle_config)
        )
        systems["muscle"].set_name("muscle_dynamics")

        # ----------------------------
        # ZFT: (p, pdot, F) -> (pzft, pzft_dot, pzft_ddot)
        # ----------------------------
        systems["zft"] = self.builder.AddSystem(
            ZFTReferenceMass2D(self.config.zft_config)
        )
        systems["zft"].set_name("zft_reference_mass")

        # Keep impedance block only if you really use it in Option A.
        # If unused, don't create it (or you’ll forget to connect it).
        if getattr(self.config, "use_impedance", False):
            systems["impedance"] = self.builder.AddSystem(
                ImpedanceForce2D(self.config.impedance_config)
            )
            systems["impedance"].set_name("impedance_force")

        # ----------------------------
        # IK: task -> joint refs
        # ----------------------------
        systems["manip_ik"] = self.builder.AddSystem(
            ManipulatorIKDesiredAngles(self.config.ik_config)
        )
        systems["manip_ik"].set_name("ik_task_to_joint_reference")

        # ----------------------------
        # Computed torque
        # ----------------------------
        systems["computed_torque"] = self.builder.AddSystem(
            ComputedTorqueInverseDynamicsController(self.config.computed_torque_config)
        )
        systems["computed_torque"].set_name("computed_torque_inverse_dynamics")

        # Optional torque limits
        if getattr(self.config, "use_torque_limits", True):
            systems["torque_limit"] = self.builder.AddSystem(
                ActuatorLimit2D(self.config.actuator_limit_config)
            )
            systems["torque_limit"].set_name("actuator_torque_limits")

        # ----------------------------
        # Convenience muxes
        # ----------------------------
        systems["ref_mux"] = self.builder.AddSystem(Multiplexer([2, 2, 2]))
        systems["ref_mux"].set_name("joint_reference_mux")

        systems["meas_mux"] = self.builder.AddSystem(
            Multiplexer([self.config.nq_total, self.config.nv_total])
        )
        systems["meas_mux"].set_name("measured_state_mux")

        systems["log_mux"] = self.builder.AddSystem(
            Multiplexer([
                2, 2,      # p, pdot
                2, 2, 2,   # pzft, pzft_dot, pzft_ddot
                2,         # F
                self.config.nq_total,
                self.config.nv_total,
            ])
        )
        systems["log_mux"].set_name("logging_signal_mux")

        # persist
        self.systems.update(systems)
        return systems

    # ------------------------------------------------------------------
    # Loggers
    # ------------------------------------------------------------------
    def add_loggers(self) -> Dict[str, object]:
        """Build loggers for LQR + OFC signals."""
        loggers = {}

        # Keep your original ones if you want; just don’t hardcode wrong dims.
        # If you truly want 8D, ensure your plant state really is 8D.
        if getattr(self.config, "log_plant_state_dim", None) is not None:
            n = self.config.log_plant_state_dim
        else:
            n = self.config.nq_total + self.config.nv_total

        loggers["state"] = self.builder.AddSystem(VectorLogSink(n))
        loggers["state"].set_name("state_logger")

        loggers["torques"] = self.builder.AddSystem(VectorLogSink(self.config.nu))
        loggers["torques"].set_name("torques_logger")

        # Full “complete-system” muxed log
        loggers["complete_system_log"] = LogVectorOutput(
            self.systems["log_mux"].get_output_port(), self.builder
        )

        self.loggers.update(loggers)
        return loggers

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------
    def connect_system_control(self):
        """Wire the blocks together (NO plant creation here)."""
        # Required systems
        plant_state_demux = self.systems["plant_state_demux"]
        state_hold = self.systems["state_hold"]
        plant_state_demux_hold = self.systems["plant_state_demux_hold"]

        q_demux = self.systems["q_demux"]
        v_demux = self.systems["v_demux"]

        ee_kin = self.systems["ee_kin"]
        muscle = self.systems["muscle"]
        zft = self.systems["zft"]
        ik = self.systems["ik"]
        computed_torque = self.systems["computed_torque"]

        torque_limit: Optional[object] = self.systems.get("torque_limit", None)

        ref_mux = self.systems["ref_mux"]
        meas_mux = self.systems["meas_mux"]

        # --------------------------------------------------------------
        # 1) Plant state -> demux (raw) and -> state_hold -> demux (held)
        # --------------------------------------------------------------
        self.builder.Connect(self.plant.get_state_output_port(), plant_state_demux.get_input_port())
        self.builder.Connect(self.plant.get_state_output_port(), state_hold.get_input_port())
        self.builder.Connect(state_hold.get_output_port(), plant_state_demux_hold.get_input_port())

        # Raw q,v (useful for kinematics/logging)
        q_port = plant_state_demux.get_output_port(0)
        v_port = plant_state_demux.get_output_port(1)

        # Held q,v (useful for controller stability / sampled-data control)
        qh_port = plant_state_demux_hold.get_output_port(0)
        vh_port = plant_state_demux_hold.get_output_port(1)

        # Optional arm/pend split (currently not used downstream, but correct to wire)
        self.builder.Connect(q_port, q_demux.get_input_port())
        self.builder.Connect(v_port, v_demux.get_input_port())

        # --------------------------------------------------------------
        # 2) EE kinematics (use raw q,v)
        # --------------------------------------------------------------
        self.builder.Connect(q_port, ee_kin.get_input_port(0))
        self.builder.Connect(v_port, ee_kin.get_input_port(1))

        # --------------------------------------------------------------
        # 3) ZFT: (p, pdot, F) -> (pzft, pzft_dot, pzft_ddot)
        # --------------------------------------------------------------
        self.builder.Connect(ee_kin.get_output_port(0), zft.get_input_port(0))   # p
        self.builder.Connect(ee_kin.get_output_port(1), zft.get_input_port(1))   # pdot
        self.builder.Connect(muscle.get_output_port(),  zft.get_input_port(2))   # F

        # --------------------------------------------------------------
        # 4) IK: (pzft, pzft_dot, pzft_ddot) -> joint refs
        # --------------------------------------------------------------
        self.builder.Connect(zft.get_output_port(0), ik.get_input_port(0))
        self.builder.Connect(zft.get_output_port(1), ik.get_input_port(1))
        self.builder.Connect(zft.get_output_port(2), ik.get_input_port(2))

        self.builder.Connect(ik.get_output_port(0), ref_mux.get_input_port(0))
        self.builder.Connect(ik.get_output_port(1), ref_mux.get_input_port(1))
        self.builder.Connect(ik.get_output_port(2), ref_mux.get_input_port(2))

        # --------------------------------------------------------------
        # 5) Computed torque: (q,v, refs) -> tau
        # Use HELD q,v here (common for sampled controllers)
        # --------------------------------------------------------------
        self.builder.Connect(qh_port, computed_torque.get_input_port(0))  # q
        self.builder.Connect(vh_port, computed_torque.get_input_port(1))  # v
        self.builder.Connect(ik.get_output_port(0), computed_torque.get_input_port(2))  # qa_ref
        self.builder.Connect(ik.get_output_port(1), computed_torque.get_input_port(3))  # qd_ref
        self.builder.Connect(ik.get_output_port(2), computed_torque.get_input_port(4))  # qdd_ref

        # --------------------------------------------------------------
        # 6) Torque -> plant actuation (NO local 'plant' variable)
        # --------------------------------------------------------------
        if torque_limit is not None:
            self.builder.Connect(computed_torque.get_output_port(), torque_limit.get_input_port())
            self.builder.Connect(torque_limit.get_output_port(), self.plant.get_actuation_input_port())
        else:
            self.builder.Connect(computed_torque.get_output_port(), self.plant.get_actuation_input_port())

        # --------------------------------------------------------------
        # 7) Expose measured state mux (if other controllers use it)
        # --------------------------------------------------------------
        self.builder.Connect(q_port, meas_mux.get_input_port(0))
        self.builder.Connect(v_port, meas_mux.get_input_port(1))

    def connect_loggers(self):
        """Connect all loggers to their signal sources."""
        # Connect base loggers (state, manip_state, ee_state)
        self.builder.Connect(
            self.plant.get_state_output_port(),
            self.loggers["state"].get_input_port()
        )
        
        self.builder.Connect(
            self.plant.get_state_output_port(self.manipulator.model_instance),
            self.loggers["manip_state"].get_input_port()
        )
        
        self.builder.Connect(
            self.plant.get_state_output_port(self.manipulator.model_instance),
            self.loggers["ee_computer"].get_input_port(0)
        )
        self.builder.Connect(
            self.loggers["ee_computer"].get_output_port(),
            self.loggers["ee_state"].get_input_port()
        )
        
        # Connect torques logger
        if "computed_torque" in self.systems:
            self.builder.Connect(
                self.systems["computed_torque"].get_output_port(),
                self.loggers["torques"].get_input_port()
            )
        
        # Optional: Create complete system log from log_mux if it exists
        if "log_mux" in self.systems:
            plant_state_demux = self.systems["plant_state_demux"]
            ee_kin = self.systems["ee_kin"]
            muscle = self.systems["muscle"]
            zft = self.systems["zft"]
            log_mux = self.systems["log_mux"]

            # log_mux order: [p(2), pdot(2), pzft(2), pzft_dot(2), pzft_ddot(2), F(2), q(nq), v(nv)]
            self.builder.Connect(ee_kin.get_output_port(0), log_mux.get_input_port(0))
            self.builder.Connect(ee_kin.get_output_port(1), log_mux.get_input_port(1))
            self.builder.Connect(zft.get_output_port(0), log_mux.get_input_port(2))
            self.builder.Connect(zft.get_output_port(1), log_mux.get_input_port(3))
            self.builder.Connect(zft.get_output_port(2), log_mux.get_input_port(4))
            self.builder.Connect(muscle.get_output_port(), log_mux.get_input_port(5))
            self.builder.Connect(plant_state_demux.get_output_port(0), log_mux.get_input_port(6))
            self.builder.Connect(plant_state_demux.get_output_port(1), log_mux.get_input_port(7))
            
            # Create complete system log
            self.loggers["complete_system_log"] = LogVectorOutput(
                log_mux.get_output_port(), self.builder
            )


# ============================================================================
# SIMULATION CLASS
# ============================================================================

class Simulation:
    """
    Manages simulation execution for cart-pendulum-manipulator system.
    
    Uses composition with:
    - SystemBuilder: Builds the multibody plant
    - ControlSystemBuilder: Builds and connects control systems (strategy pattern)
    
    Responsibilities:
    - Configure initial states (EE position, cart position)
    - Store simulation components (plant, manipulator, etc.)
    - Run simulation loop with pluggable control strategies
    
    Attributes:
        config: SimulationConfig with all simulation parameters
        system_builder: SystemBuilder instance for creating the multibody system
        control_builder: ControlSystemBuilder instance (optional, set later)
        builder: DiagramBuilder (set during setup)
        plant: MultibodyPlant (set during setup)
        scene_graph: SceneGraph (set during setup)
        manipulator: CupManipulator (set during setup)
        cart_model: ModelInstanceIndex for cart-pendulum (set during setup)
        ee_world_pos: End-effector position in world frame
        cart_init_pos: Initial cart position [x, y, α, β]
        control_systems: Dictionary of control system blocks
        loggers: Dictionary of data loggers
    """
    
    def __init__(self, config: SimulationConfig, system_builder: SystemBuilder,
                 control_builder: ControlSystemBuilder = None):
        """
        Initialize simulation with configuration and builders.
        
        Args:
            config: SimulationConfig with all simulation parameters
            system_builder: SystemBuilder for creating the multibody system
            control_builder: ControlSystemBuilder for control architecture (optional)
        """
        self.config = config
        self.system_builder = system_builder
        self.control_builder = control_builder
        
        # Will be set during setup_system()
        self.builder = None
        self.plant = None
        self.scene_graph = None
        self.manipulator = None
        self.cart_model = None
        self.cart_pendulum = None
        
        # Will be set during configure_initial_state()
        self.ee_world_pos = None
        self.cart_init_pos = None
        self.manipulator_initial_q = None
        
        # Will be set after control builder runs
        self.control_systems = {}
        self.loggers = {}
        
    def setup_system(self):
        """
        Build the multibody system using the SystemBuilder.
        
        Creates the DiagramBuilder, MultibodyPlant, SceneGraph, and adds
        manipulator and cart-pendulum to the plant.
        """
        (self.builder, self.plant, self.scene_graph, 
         self.manipulator, self.cart_pendulum, self.cart_model) = self.system_builder.build(meshcat=self.config.meshcat)
        
    def configure_initial_state(self, context=None, cart_x_override=None, cart_y_override=None):
        """
        Calculate and optionally apply initial state to a Drake context.
        
        When context=None: Calculates values using temporary context, stores them.
        When context provided: Applies stored values to the given context.
        
        This unified method handles both initial calculation and later application,
        eliminating duplication.
        
        Args:
            context: Optional Drake context to apply state to. If None, only calculates.
            cart_x_override: Optional override for cart X position
            cart_y_override: Optional override for cart Y position
        """
        # Calculate manipulator initial angles (only once)
        if self.manipulator_initial_q is None:
            self.manipulator_initial_q = np.array([
                self.system_builder.manipulator_joint_angles['link1_base'],     # q1
                self.system_builder.manipulator_joint_angles['link2_link1'],    # q2
            ])
        
        # Determine which context to use
        needs_temp_context = context is None
        work_context = self.plant.CreateDefaultContext() if needs_temp_context else context
        plant_context = self.plant.GetMyMutableContextFromRoot(work_context)
        
        # Set manipulator positions (needed for both calculation and application)
        self.manipulator.set_positions_user_order(self.plant, plant_context, {
            "link1_base": self.manipulator_initial_q[0],
            "link2_link1": self.manipulator_initial_q[1],
        })
        
        # Calculate EE position and cart position (only once)
        if self.ee_world_pos is None:
            self.ee_world_pos = self.manipulator.get_end_effector_position(self.plant, plant_context)
        
        # if cart_init_pos is None, calculate it based on config or default to EE position
        if self.cart_init_pos is None:
            # Get cart position from config or default to EE position
            if self.config.physics_config.cart_initial_position is not None:
                cart_x = self.config.physics_config.cart_initial_position[0]
                cart_y = self.config.physics_config.cart_initial_position[1]
            else:
                cart_x = self.ee_world_pos[0]
                cart_y = self.ee_world_pos[1]
            
            # Get pendulum angles from config or default to hanging
            if self.config.physics_config.pendulum_initial_angles is not None:
                alpha = self.config.physics_config.pendulum_initial_angles[0]
                beta = self.config.physics_config.pendulum_initial_angles[1]
            else:
                alpha = 0.0  # α (pitch) = 0 (hanging)
                beta = 0.0   # β (roll) = 0 (hanging)
            
            self.cart_init_pos = np.array([cart_x, cart_y, alpha, beta])
        
        # If context was provided, apply full state (positions + velocities)
        if context is not None:
            # Set manipulator velocities to zero
            self.manipulator.set_velocities_user_order(self.plant, plant_context, {
                "link1_base": 0.0,
                "link2_link1": 0.0,
            })
            
            # Set cart positions (with optional override)
            cart_x = cart_x_override if cart_x_override is not None else self.cart_init_pos[0]
            cart_y = cart_y_override if cart_y_override is not None else self.cart_init_pos[1]
            
            cart_pendulum_positions = np.array([cart_x, cart_y, self.cart_init_pos[2], self.cart_init_pos[3]])
            self.plant.SetPositions(plant_context, self.cart_model, cart_pendulum_positions)
            self.plant.SetVelocities(plant_context, self.cart_model, np.zeros(4))
        
        # Print summary only on first call (with temp context)
        if needs_temp_context:
            self._print_initial_config_summary(work_context)
    
    def _print_initial_config_summary(self, context=None):
        """
        Print summary of initial configuration.
        
        Args:
            context: Optional context for verifying cart world position
        """
        print(colored(f"\n📄 Initial Configuration:", "cyan"))
        print(colored(f"  - Manipulator: q1={np.rad2deg(self.manipulator_initial_q[0]):.1f}°, "
                     f"q2={np.rad2deg(self.manipulator_initial_q[1]):.1f}°", "cyan"))
        print(colored(f"  - Cart positioned at EE: ({self.cart_init_pos[0]:.3f}, "
                     f"{self.cart_init_pos[1]:.3f}) m", "cyan"))
        print(colored(f"  - Pendulum: α={np.rad2deg(self.cart_init_pos[2]):.1f}°, "
                     f"β={np.rad2deg(self.cart_init_pos[3]):.1f}°", "cyan"))
        print(colored(f"\n🌍 World Frame Positions:", "yellow", attrs=["bold"]))
        print(colored(f"  - EE in world frame: ({self.ee_world_pos[0]:.3f}, "
                     f"{self.ee_world_pos[1]:.3f}, {self.ee_world_pos[2]:.3f}) m", "yellow"))
        
        if context is not None:
            # Set cart position for verification
            plant_context = self.plant.GetMyMutableContextFromRoot(context)
            self.plant.SetPositions(plant_context, self.cart_model, self.cart_init_pos)
            
            cart_body = self.plant.GetBodyByName("cart", self.cart_model)
            cart_world_pos = self.plant.CalcPointsPositions(
                plant_context, cart_body.body_frame(), [0, 0, 0], self.plant.world_frame()
            ).flatten()
            print(colored(f"  - Cart in world frame: ({cart_world_pos[0]:.3f}, "
                         f"{cart_world_pos[1]:.3f}, {cart_world_pos[2]:.3f}) m", "yellow"))
    
    def setup_control_builder(self, control_builder: ControlSystemBuilder):
        """
        Set or change the control builder (strategy).
        
        Args:
            control_builder: ControlSystemBuilder instance
        """
        self.control_builder = control_builder
        self.control_builder.builder = self.builder
        self.control_builder.plant = self.plant
        self.control_builder.cart_model = self.cart_model
        self.control_builder.manipulator = self.manipulator
    
    def build_control_system(self):
        """
        Build control system using the provided ControlSystemBuilder.
        
        This is where the strategy pattern is applied:
        Different control builders create different control architectures.
        """
        if self.control_builder is None:
            raise ValueError("No control builder provided - use set_control_builder() first")
        
        # Set the control builder's drake components (if not already set during __init__)
        if not hasattr(self.control_builder, 'builder') or self.control_builder.builder is None:
            self.control_builder.builder = self.builder
            self.control_builder.plant = self.plant
            self.control_builder.cart_model = self.cart_model
            self.control_builder.manipulator = self.manipulator
        
        # Build and connect control systems
        self.control_systems, self.loggers = self.control_builder.build_and_connect()
        
        print(colored(f"✓ Control system configured: {type(self.control_builder).__name__}", "green"))

    def _setup_simulator(self, diagram):
        """
        Setup simulator with initial conditions.
        
        Args:
            cart_x_init: Optional override for cart X position
            cart_y_init: Optional override for cart Y position
        """
        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()
        
        # Apply pre-calculated initial state to simulation context
        self.configure_initial_state(context)
        
        # Publish initial state and add visualization
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        diagram.ForcedPublish(context)
        add_frames_to_meshcat(self.config.meshcat, self.plant, plant_context,
                            self.manipulator, self.cart_model)
        
        print(colored(f"✓ Initial state configured at: {self.config.meshcat.web_url()}", "green"))
        
        return simulator
    
    def _run_simulation_loop(self, simulator, visualizer):
        """Run the simulation loop."""
        visualizer.StartRecording()
        simulator.set_target_realtime_rate(1.0)
        
        dt_sim = 0.01
        current_time = 0.0
        debug_interval = 1.0
        next_debug_time = debug_interval
        
        print(colored("\n🚀 Starting simulation...", "cyan"))
        
        while current_time < self.config.duration:
            if current_time >= next_debug_time:
                self._print_debug_info(simulator, current_time)
                next_debug_time += debug_interval
            
            simulator.AdvanceTo(current_time + dt_sim)
            current_time += dt_sim
        
        print(colored(f"\n✓ Simulation complete at t={current_time:.2f}s", "green"))
        visualizer.PublishRecording()
    
    def run(self, cart_x_init=None, cart_y_init=None):
        """
        Run simulation with the configured control system.
        
        This method replaces the mode-specific run methods (run_lqr_with_manipulator_tracking, etc.)
        with a single generic run() that works with any control builder.
        
        Args:
            cart_x_init: Initial cart X position (defaults to self.cart_init_pos[0])
            cart_y_init: Initial cart Y position (defaults to self.cart_init_pos[1])
        """
        # Build control system if not already built
        if not self.control_systems:
            self.build_control_system()
        
        # Add visualization
        visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder, self.scene_graph, self.config.meshcat
        )
        
        # Add frame updater
        frame_list = self._build_frame_list()
        frame_updater = self.builder.AddSystem(
            MeshcatFrameUpdater(self.config.meshcat, self.plant, frame_list, update_period=0.033)
        )
        frame_updater.set_name("frame_updater")
        self.builder.Connect(self.plant.get_state_output_port(), frame_updater.get_input_port(0))
        
        # Build diagram and create simulator
        diagram = self.builder.Build()
        simulator = self._setup_simulator(diagram)
        
        # Run simulation loop
        self._run_simulation_loop(simulator, visualizer)
        
        # Extract and plot results
        self._extract_and_plot_results(simulator.get_context())
    
    def _build_frame_list(self):
        """Build frame list for visualization."""
        from pydrake.multibody.tree import FrameIndex
        
        temp_context = self.plant.CreateDefaultContext()
        frame_list = []
        
        for i in range(self.plant.num_frames()):
            frame = self.plant.get_frame(FrameIndex(i))
            frame_name = frame.name()
            if frame_name == "world":
                continue
            
            if "link" in frame_name.lower() or "cup_center" in frame_name.lower():
                length = 0.15
            elif "cart" in frame_name.lower():
                length = 0.12
            elif "pendulum" in frame_name.lower() or "gimbal" in frame_name.lower():
                length = 0.10
            else:
                length = 0.08
            
            frame_list.append((frame_name, frame, length))
        
        return frame_list
    
    def _print_debug_info(self, simulator, current_time):
        """Print debug information during simulation."""
        context = simulator.get_context()
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        
        cart_state = self.plant.GetPositionsAndVelocities(plant_context, self.cart_model)
        manip_state = self.plant.GetPositionsAndVelocities(plant_context, self.manipulator.model_instance)
        ee_pos = self.manipulator.get_end_effector_position(self.plant, plant_context)
        
        print(colored(f"\n[t={current_time:.2f}s]", "cyan"))
        print(f"  Cart: ({cart_state[0]:.3f}, {cart_state[1]:.3f})m")
        print(f"  Manip: q1={np.rad2deg(manip_state[0]):.1f}°, q2={np.rad2deg(manip_state[1]):.1f}°")
        print(f"  EE: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f})m")
        print(f"  Error: {np.sqrt((ee_pos[0]-cart_state[0])**2 + (ee_pos[1]-cart_state[1])**2)*1000:.1f} mm")
    
    def _extract_and_plot_results(self, context):
        """Extract logged data and generate plots."""
        # Extract all logs
        t = self.loggers['state'].FindLog(context).sample_times()
        state_data = self.loggers['state'].FindLog(context).data()
        manip_state_data = self.loggers['manip_state'].FindLog(context).data()
        ee_state_data = self.loggers['ee_state'].FindLog(context).data()
        
        # Extract control-specific logs if available
        if 'ref' in self.loggers:
            ref_data = self.loggers['ref'].FindLog(context).data()
            force_data = self.loggers['force'].FindLog(context).data()
            impedance_data = self.loggers['impedance'].FindLog(context).data()
            cart_traj_data = self.loggers['cart_traj'].FindLog(context).data()
            manip_desired_data = self.loggers['manip_desired'].FindLog(context).data()
            manip_torque_data = self.loggers['manip_torque'].FindLog(context).data()
            
            # Call LQR-specific plotting function
            plot_lqr_manip_ee_traj_track_results(
                t, state_data, ref_data, cart_traj_data,
                ee_state_data[0:2, :], ee_state_data[2:4, :],
                force_data, impedance_data, manip_state_data,
                manip_desired_data, manip_torque_data, self.config
            )
        else:
            # Generic plotting for non-LQR controllers
            print(colored("ℹ️  No LQR-specific loggers found, skipping detailed plots", "yellow"))
        
        plt.show(block=True)
        print(colored("\n✓ Simulation Complete!", "green", attrs=["bold"]))
    
    



# ============================================================================
# MAIN SIMULATION
# ============================================================================

def main():
    
    # ========================================================================
    # BUILD CONTROL DIAGRAM
    # ========================================================================
    # Build the diagram for the selected control mode
    # Plant and scene_graph are added fresh to each mode's builder
    # For control modes, we pass them to the mode functions which handle the setup
    
    if args.mode == 'scene-viz':
        print("\n" + "="*80)
        print(colored("2D CART-PENDULUM (EXTENDED) - MUSCLE DYNAMICS & OFC", "cyan", attrs=["bold"]))
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
        
        # ========================================================================
        # BUILD MULTIBODY PLANT WITH TWO SEPARATE MODEL INSTANCES
        # ========================================================================
        # The plant will contain TWO robots:
        # 1. Manipulator (2-DOF cup manipulator) - for visualization
        # 2. Cart-Pendulum (4-DOF system) - actively controlled
        #
        # Each robot gets its own ModelInstance, which allows us to:
        # - Query states separately: plant.get_state_output_port(model_instance)
        # - Apply forces separately: plant.get_actuation_input_port(model_instance)
        # - Set initial conditions separately
        #
        # This is how we separate "cart-pendulum only" states from "full system" states
        builder = DiagramBuilder()
        plant = MultibodyPlant(time_step=0.001)  # 1ms time step for simulation
        scene_graph = builder.AddSystem(SceneGraph())
        plant.RegisterAsSourceForSceneGraph(scene_graph)
        
        # ========================================================================
        # STEP 1: ADD MANIPULATOR TO MAIN PLANT (MODEL INSTANCE 1)
        # ========================================================================
        # Create manipulator configuration and add it to the main plant.
        # This creates the first ModelInstance.
        # The manipulator will remain FIXED at its initial configuration in most modes.
        
        # Use global MANIPULATOR_CONFIG
        manipulator_config = MANIPULATOR_CONFIG
        
        #Initialize manipulator and load URDF into plant
        manipulator = CupManipulator(manipulator_config, enable_visualization=True)
        parser = Parser(plant)
        manipulator.load_urdf_to_plant(plant, parser)  # Loads URDF, creates model instance
        
        # Calculate where to position the base so EE is at desired location
        # First, get EE position with base at origin
        initial_q = np.array([np.deg2rad(-0.0), np.deg2rad(4.0)])  # [q1, q2] - initial joint angles
        temp_plant = MultibodyPlant(0.0)
        temp_parser = Parser(temp_plant)
        temp_manip = CupManipulator(manipulator_config, enable_visualization=False)
        temp_manip.load_urdf_to_plant(temp_plant, temp_parser)
        temp_manip.weld_base_to_world(temp_plant, position=np.array([0.0, 0.0, 0.0]), orientation=np.array([0.0, 0, 0.0]))
        temp_manip.add_end_effector_frame(temp_plant)
        temp_plant.Finalize()
        temp_context = temp_plant.CreateDefaultContext()
        temp_manip.set_positions_user_order(temp_plant, temp_context, {
            "link1_base": initial_q[0],
            "link2_link1": initial_q[1],
        })
        ee_at_origin = temp_manip.get_end_effector_position(temp_plant, temp_context)
        
        # Calculate base offset: to center EE at [0, 0], base needs to be at [-ee_x, -ee_y, 0]
        base_offset = -ee_at_origin  # Negate to bring EE to origin
        base_offset[2] = 0.0  # Keep Z at zero (or desired height)
        
        print(colored(f"\n📍 Manipulator Base Positioning:", "yellow"))
        print(colored(f"  - EE position with base at origin: ({ee_at_origin[0]:.3f}, {ee_at_origin[1]:.3f}, {ee_at_origin[2]:.3f}) m", "yellow"))
        print(colored(f"  - Offsetting base to: ({base_offset[0]:.3f}, {base_offset[1]:.3f}, {base_offset[2]:.3f}) m", "yellow"))
        print(colored(f"  - This will center EE at approximately [0, 0]", "green"))
        
        # Rotate base -90° around Y to align manipulator with X-Y plane (same as cart)
        # This makes manipulator X-axis → world X-axis, manipulator Z-axis → world Y-axis
        # AND position it so the EE is at the origin
        manipulator.weld_base_to_world(plant, position=base_offset, orientation=np.array([0.0, 0, 0.0]))
        # Add actuators and end-effector frame BEFORE finalization
        manipulator.add_joint_actuators(plant)
        manipulator.add_end_effector_frame(plant)
        print(colored(f"✓ End-effector frame '{manipulator.EE_FRAME_NAME}' added to manipulator", "green"))
        
        print(colored(f"✓ Manipulator loaded (ModelInstance: {manipulator.model_instance})", "green"))
        print(colored(f"  - State dimension: 4 (2 positions + 2 velocities)", "cyan"))
        print(colored(f"  - Joints: link1_base, link2_link1", "cyan"))
        
        
        # Determine z-offset for cart-pendulum based on URDF joint origin
        z_offset_from_urdf = 1.17625  # meters, from link1_base joint origin
        print(colored(f"📍 Using z-offset from URDF: {z_offset_from_urdf:.5f} m", "cyan"))
        

        # Initialize cart-pendulum with this z-offset to ensure it is visually aligned with the manipulator's end-effector
        cart_pendulum = CartPendulum2DExtended(physics_config, z_offset=z_offset_from_urdf)
        cart_model = plant.AddModelInstance("cart_pendulum")  # Creates new model instance
        cart_pendulum.build_plant(plant, cart_model)  # Builds cart-pendulum in this instance
        
        print(colored(f"✓ Cart-Pendulum (2D Extended) created (ModelInstance: {cart_model})", "green"))
        print(colored(f"  - State dimension: 8 (4 positions + 4 velocities)", "cyan"))
        print(colored(f"  - DOFs: x, y (cart), α, β (pendulum gimbal angles)", "cyan"))
        print(colored(f"  - Z-plane height: {z_offset_from_urdf:.5f} m", "cyan"))
        
        
        # Set high damping on manipulator joints to lock them in scene-viz mode
        # Must be done BEFORE finalization
        jt1 = manipulator.get_joint_by_name(plant, manipulator.JT1_NAME)
        jt2 = manipulator.get_joint_by_name(plant, manipulator.JT2_NAME)
        jt1.set_default_damping_vector([1000.0])  # High damping to lock joint
        jt2.set_default_damping_vector([1000.0])  # High damping to lock joint

        
        plant.Finalize()  # Must be called before adding to diagram



        print(colored(f"\n✓ Plant finalized with {plant.num_positions()} total positions, "
                    f"{plant.num_velocities()} total velocities", "green"))
        

        # ========================================================================
        # CONFIGURE INITIAL STATE
        # ========================================================================
        # Initial configuration in natural [q1, q2] order
        initial_q = np.array([np.deg2rad(-0.0), np.deg2rad(4.0)])  # [q1, q2]
        
        # Calculate EE position at configured joint angles
        temp_context = plant.CreateDefaultContext()
        manipulator.set_positions_user_order(plant, temp_context, {
            "link1_base": initial_q[0],
            "link2_link1": initial_q[1],
        })
        
        # Get EE world frame position using the cup_center frame
        ee_world_pos = manipulator.get_end_effector_position(plant, temp_context)
        
        # Use world frame position for cart initialization
        # Both manipulator and cart work in X-Y plane after rotation: direct mapping [X, Y]
        ee_pos_3d = ee_world_pos  # Use actual world frame coordinates
        
        # Define cart initial position at manipulator EE world position
        cart_init_pos = np.array([ee_world_pos[0], ee_world_pos[1], 0.0, 0.0])  # [x from EE_X, y from EE_Y, α, β]
        
        # Set cart position in temp context for frame visualization
        plant.SetPositions(temp_context, cart_model, cart_init_pos)
        
        cart_body = plant.GetBodyByName("cart", cart_model)
        cart_world_pos = plant.CalcPointsPositions(
            temp_context, cart_body.body_frame(), [0, 0, 0], plant.world_frame()
        ).flatten()
        
        # Print configuration summary
        print(colored(f"\n📄 Initial Configuration:", "cyan"))
        print(colored(f"  - Manipulator: q1={np.rad2deg(initial_q[0]):.1f}°, q2={np.rad2deg(initial_q[1]):.1f}°", "cyan"))
        print(colored(f"  - Manipulator EE: ({ee_pos_3d[0]:.3f}, {ee_pos_3d[1]:.3f}, {ee_pos_3d[2]:.3f}) m", "cyan"))
        print(colored(f"  - Cart positioned at EE: ({cart_init_pos[0]:.3f}, {cart_init_pos[1]:.3f}) m", "cyan"))
        print(colored(f"  - Pendulum hanging: α=0°, β=0°", "cyan"))
        print(colored(f"\n🌍 World Frame Positions:", "yellow", attrs=["bold"]))
        print(colored(f"  - EE in world frame: ({ee_world_pos[0]:.3f}, {ee_world_pos[1]:.3f}, {ee_world_pos[2]:.3f}) m", "yellow"))
        print(colored(f"  - Cart in world frame: ({cart_world_pos[0]:.3f}, {cart_world_pos[1]:.3f}, {cart_world_pos[2]:.3f}) m", "yellow"))
        
        # Calculate and display alignment error
        offset_x = abs(ee_world_pos[0] - cart_world_pos[0])
        offset_y = abs(ee_world_pos[1] - cart_world_pos[1])
        print(colored(f"\n✓ EE-Cart Alignment Check:", "green", attrs=["bold"]))
        print(colored(f"  - X offset: {offset_x*1000:.2f} mm", "green" if offset_x < 0.01 else "red"))
        print(colored(f"  - Y offset: {offset_y*1000:.2f} mm", "green" if offset_y < 0.01 else "red"))
        if offset_x < 0.01 and offset_y < 0.01:
            print(colored(f"  ✓ EE and Cart are aligned (< 1cm error)", "green"))
        else:
            print(colored(f"  ⚠ EE and Cart have significant offset", "yellow"))
        
        # Plot coordinate frames to verify orientation
        plot_frames_top_view(plant, temp_context, manipulator, cart_model, 
                           title="Initial Frame Orientations - Scene Viz Mode")
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to display plot
        
        # Add plant to builder (scene_graph was already added when created)
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
        
        # Add Meshcat visualizer
        visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        diagram = builder.Build()
        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()
        
        plant_context = plant.GetMyMutableContextFromRoot(context)
        
        # Set manipulator to desired configuration (not zeros!)
        manipulator.set_positions_user_order(plant, plant_context, {
            "link1_base": initial_q[0],
            "link2_link1": initial_q[1],
        })
        
        # Set cart to initial position
        plant.SetPositions(plant_context, cart_model, cart_init_pos)
        
        # Set all velocities to zero
        plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
        
        # Initial publish to show the scene
        diagram.ForcedPublish(context)
        
        # Add coordinate frames to meshcat
        add_frames_to_meshcat(meshcat, plant, plant_context, manipulator, cart_model)
        
        print(colored("\n🎬 Scene visualization mode", "cyan"))
        print(colored(f"   View at: {meshcat.web_url()}", "cyan"))
        print(colored("\n   Interactive Mode Commands:", "yellow"))
        print(colored("   - 'c x y'  : Move cart to position (x, y)", "yellow"))
        print(colored("   - 'e x y'  : Move manipulator EE to position (x, y) via IK", "yellow"))
        print(colored("   - Ctrl+C   : Exit", "yellow"))
        print(colored(f"\n   Current cart: ({cart_init_pos[0]:.3f}, {cart_init_pos[1]:.3f})", "yellow"))
        print(colored(f"   Current EE:   ({ee_world_pos[0]:.3f}, {ee_world_pos[1]:.3f})\n", "yellow"))
        
        # Get current cart positions and manipulator joint angles
        current_cart_pos = cart_init_pos.copy()
        current_manip_q = initial_q.copy()
        
        # Get cart body for position queries
        cart_body = plant.GetBodyByName("cart", cart_model)
        
        try:
            while True:
                # Wait for user input (blocking - no continuous simulation)
                user_input = input(colored("Enter command (c x y | e x y) or Ctrl+C to exit: ", "cyan")).strip()
                    
                if user_input:
                    try:
                        parts = user_input.split()
                        if len(parts) == 3 and parts[0] in ['c', 'e']:
                            command = parts[0]
                            new_x = float(parts[1])
                            new_y = float(parts[2])
                            
                            if command == 'c':
                                # Update cart position
                                current_cart_pos[0] = new_x
                                current_cart_pos[1] = new_y
                                
                                # Set new positions in plant context
                                plant.SetPositions(plant_context, cart_model, current_cart_pos)
                                
                                # Force visualization update
                                diagram.ForcedPublish(context)
                                
                                # Update coordinate frames
                                add_frames_to_meshcat(meshcat, plant, plant_context, manipulator, cart_model)
                                
                                # Calculate world frame position
                                cart_world_pos = plant.CalcPointsPositions(
                                    plant_context, cart_body.body_frame(), [0, 0, 0], plant.world_frame()
                                ).flatten()
                                
                                print(colored(f"\n✓ Cart updated to: ({new_x:.3f}, {new_y:.3f})", "green"))
                                print(colored(f"  World frame: ({cart_world_pos[0]:.3f}, {cart_world_pos[1]:.3f}, {cart_world_pos[2]:.3f}) m\n", "yellow"))
                            
                            elif command == 'e':
                                # Solve IK for manipulator EE position
                                target_xy = np.array([new_x, new_y])
                                print(colored(f"  Solving IK for target ({new_x:.3f}, {new_y:.3f})...", "cyan"))
                                q_solution, success = manipulator.solve_initial_pose_via_ik(
                                    plant, target_xy, current_manip_q, pos_tol=0.001, verbose=True
                                )
                                
                                if success:
                                    current_manip_q = q_solution
                                    
                                    # Update manipulator joint positions
                                    manipulator.set_positions_user_order(plant, plant_context, {
                                        "link1_base": current_manip_q[0],
                                        "link2_link1": current_manip_q[1],
                                    })
                                    
                                    # Force visualization update
                                    diagram.ForcedPublish(context)
                                    
                                    # Update coordinate frames
                                    add_frames_to_meshcat(meshcat, plant, plant_context, manipulator, cart_model)
                                    
                                    # Get actual EE position using cup_center frame
                                    ee_actual = manipulator.get_end_effector_position(plant, plant_context)
                                    
                                    print(colored(f"\n✓ Manipulator EE updated to: ({ee_actual[0]:.3f}, {ee_actual[1]:.3f})", "green"))
                                    print(colored(f"  Joint angles: q1={np.rad2deg(current_manip_q[0]):.1f}°, q2={np.rad2deg(current_manip_q[1]):.1f}°\n", "yellow"))
                                else:
                                    print(colored(f"\n✗ IK failed for target ({new_x:.3f}, {new_y:.3f}) - may be out of reach", "red"))
                                    print(colored(f"  Manipulator workspace is limited by link lengths\n", "yellow"))
                        else:
                            print(colored("Invalid input. Format: 'c x y' or 'e x y'\n", "red"))
                    except ValueError:
                        print(colored("Invalid numbers. Format: 'c x y' or 'e x y'\n", "red"))
                
        except KeyboardInterrupt:
            print(colored("\n✓ Visualization stopped", "green"))
        
        return
    
    
    elif args.mode == 'lqr-applied-to-cart-manip-following-cart':
        print("\n" + "="*80)
        print(colored("2D CART-PENDULUM (EXTENDED) - MUSCLE DYNAMICS & OFC", "cyan", attrs=["bold"]))
        print("="*80)
        print(colored(f"Mode: {args.mode}", "yellow"))
        print(colored(f"Target: ({args.target_x:.2f}, {args.target_y:.2f}) m", "yellow"))
        print(colored(f"Duration: {args.duration:.1f} s", "yellow"))
        print(colored(f"Horizon: {args.horizon:.1f} s", "yellow"))
        print("="*80 + "\n")
        
        # Start Meshcat
        meshcat = StartMeshcat()
        print(colored(f"🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))
        
        # Create simulation configuration from args and global configs
        sim_config = SimulationConfig.from_args(
            args=args,
            physics_config=PHYSICS_CONFIG,
            muscle_config=MUSCLE_CONFIG,
            impedance_config=IMPEDANCE_CONFIG,
            zft_config=ZFT_CONFIG,
            meshcat=meshcat,
        )
        
        # ====================================================================
        # STEP 1: Build system (multibody plant, manipulator, cart)
        # ====================================================================
        system_builder = SystemBuilder(
            physics_config=sim_config.physics_config,
            manipulator_urdf_path=sim_config.manipulator_urdf_path,
            manipulator_joint_angles=sim_config.manipulator_joint_angles,
            manipulator_damping=sim_config.manipulator_damping,
        )
        
        # Build the system and get Drake components
        (builder, plant, scene_graph, 
         manipulator, cart_pendulum, cart_model) = system_builder.build(meshcat=meshcat)
        
        print(colored("\n🚀 Running LQR with manipulator EE trajectory tracking (computed torque)...", "cyan"))
        
        # ====================================================================
        # STEP 2: Create control builder using system components
        # ====================================================================
        control_builder = LQRWithOFCOnlyCartPendulumBuilder(
            builder=builder,
            plant=plant,
            cart_model=cart_model,
            manipulator=manipulator,
            config=sim_config
        )
        
        # ====================================================================
        # STEP 3: Create simulation with system builder and control builder
        # ====================================================================
        simulation = Simulation(
            config=sim_config,
            system_builder=system_builder,
            control_builder=control_builder
        )
        
        # Set the Drake components (already built by system_builder)
        simulation.builder = builder
        simulation.plant = plant
        simulation.scene_graph = scene_graph
        simulation.manipulator = manipulator
        simulation.cart_pendulum = cart_pendulum
        simulation.cart_model = cart_model
        
        # Configure initial state (EE position, cart position), only for vizualization. The actual initial states
        # are set before the simulation
        simulation.configure_initial_state()
        
        # ====================================================================
        # STEP 4: Run simulation with the configured control strategy
        # ====================================================================
        simulation.run(
            cart_x_init=simulation.cart_init_pos[0],
            cart_y_init=simulation.cart_init_pos[1]
        )

    elif args.mode == 'lqr-applied-to-both-cart-manip':
        print("\n" + "="*80)
        print(colored("2D CART-PENDULUM (EXTENDED) - LQR on BOTH CART & MANIPULATOR", "cyan", attrs=["bold"]))
        print("="*80)
        print(colored(f"Mode: {args.mode}", "yellow"))
        print(colored(f"Target: ({args.target_x:.2f}, {args.target_y:.2f}) m", "yellow"))
        print(colored(f"Duration: {args.duration:.1f} s", "yellow"))
        print(colored(f"Horizon: {args.horizon:.1f} s", "yellow"))
        print("="*80 + "\n")
        
        # Start Meshcat
        meshcat = StartMeshcat()
        print(colored(f"🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))
        
        # Create simulation configuration from args and global configs
        sim_config = SimulationConfig.from_args(
            args=args,
            physics_config=PHYSICS_CONFIG,
            muscle_config=MUSCLE_CONFIG,
            impedance_config=IMPEDANCE_CONFIG,
            zft_config=ZFT_CONFIG,
            meshcat=meshcat,
        )
        
        # ====================================================================
        # STEP 1: Build system (multibody plant, manipulator, cart)
        # ====================================================================
        system_builder = SystemBuilder(
            physics_config=sim_config.physics_config,
            manipulator_urdf_path=sim_config.manipulator_urdf_path,
            manipulator_joint_angles=sim_config.manipulator_joint_angles,
            manipulator_damping=sim_config.manipulator_damping,
        )
        
        # Build the system and get Drake components
        (builder, plant, scene_graph, 
         manipulator, cart_pendulum, cart_model) = system_builder.build(meshcat=meshcat)
        
        print(colored("\n🚀 Running LQR with manipulator and cart...", "cyan"))
        
        initial_viz = False  # Set to True to visualize initial configuration before control is applied
        
        if initial_viz:
            # Add Meshcat visualizer to view the system
            visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
            
            print(colored("\n🚀 Viewing welded cart-manipulator system...", "cyan"))
            print(colored(f"   View at: {meshcat.web_url()}", "cyan"))
            
            # Build diagram and create simulator
            diagram = builder.Build()
            simulator = Simulator(diagram)
            context = simulator.get_mutable_context()
            plant_context = plant.GetMyMutableContextFromRoot(context)
            
            # Set initial manipulator configuration
            initial_q = np.array([np.deg2rad(0.0), np.deg2rad(20.0)])  # [q1, q2]
            manipulator.set_positions_user_order(plant, plant_context, {
                "link1_base": initial_q[0],
                "link2_link1": initial_q[1],
            })
            
            # Set pendulum angle (only pitch in welded mode)
            plant.SetPositions(plant_context, cart_model, np.array([0.0]))  # pitch = 0
            
            # Set all velocities to zero
            plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
            
            # Publish initial state to Meshcat
            diagram.ForcedPublish(context)
            
            # Get EE position for info
            ee_world_pos = manipulator.get_end_effector_position(plant, plant_context)
            
            print(colored(f"\n📄 System Configuration:", "cyan"))
            print(colored(f"  - DOF: {plant.num_positions()} positions, {plant.num_velocities()} velocities", "cyan"))
            print(colored(f"  - Manipulator: q1={np.rad2deg(initial_q[0]):.1f}°, q2={np.rad2deg(initial_q[1]):.1f}°", "cyan"))
            print(colored(f"  - EE position: ({ee_world_pos[0]:.3f}, {ee_world_pos[1]:.3f}, {ee_world_pos[2]:.3f}) m", "cyan"))
            print(colored(f"  - Cart welded to EE (follows manipulator motion)", "green"))
            print(colored(f"  - Pendulum hanging from cart (pitch = 0°)", "cyan"))
            
            print(colored("\n🎬 System ready for viewing. Press Ctrl+C to exit.\n", "yellow"))
            
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print(colored("\n✓ Visualization stopped.", "green"))
        
        # ====================================================================
        # STEP 2: Create control builder using system components
        # ====================================================================
        control_builder = LQRWithOFCForCompleteSystem(
            builder=builder,
            plant=plant,
            cart_model=cart_model,
            manipulator=manipulator,
            config=sim_config
        )

if __name__ == "__main__":
    main()

