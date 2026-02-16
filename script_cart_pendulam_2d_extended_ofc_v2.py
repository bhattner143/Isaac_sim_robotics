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
    CartPendulumConfig, 
    create_cup_manipulator_config,
    ManipulatorConfig
)

from scipy.linalg import solve_discrete_are

# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description='2D Cart-Pendulum with Muscle Dynamics & OFC')
parser.add_argument('--mode', type=str, 
                    choices=['finite-horizon-lqr-for-min-effort_cart_pend_only', 
                             'lqr-manip-ee-traj-track', 
                             'lqr-manip-ik-track',
                             'manip-ik-follows-cart',
                             'scene-viz'],
                    default='lqr-manip-ee-traj-track',
                    # default='scene-viz',
                    help='Simulation mode')
parser.add_argument('--duration', type=float, default=10.0, help='Simulation duration [s]')
parser.add_argument('--target-x', type=float, default=1, help='Target X position [m]')
parser.add_argument('--target-y', type=float, default=-1.0, help='Target Y position [m]')
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
from script_cup_manipulator_controller_ofc import CupManipulator

# Restore sys.argv and our parsed args
sys.argv = _saved_argv
args = _parsed_args

# ============================================================================
# CONSTANTS
# ============================================================================
# DEPRECATED: End-effector offset from link2 frame to cup center (from old URDF)
# Use manipulator.EE_XYZ_BASE and cup_center frame instead
MANIPULATOR_EE_OFFSET = np.array([-1.2545, 0.0, -0.188125])

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
        Defines a named end-effector frame attached to link2 at the cup middle.

        Must be called AFTER the model is added (self.model_instance is valid)
        and BEFORE plant.Finalize().

        Returns:
            The created Frame (FixedOffsetFrame)
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

    def set_positions_user_order(self, plant: MultibodyPlant, context, user_positions: np.ndarray):
        """Set positions in user order [q1, q2]."""
        q1, q2 = user_positions
        # Set in Drake order [q2, q1]
        self.set_jt([self.JT1_NAME, self.JT2_NAME], plant, context, [q2, q1])

    def get_velocities_user_order(self, plant: MultibodyPlant, context) -> np.ndarray:
        """Get velocities in user order [q1_dot, q2_dot]."""
        # Get in Drake order [q2_dot, q1_dot], then reverse to user order
        drake_velocities = self.get_jt_velocity([self.JT1_NAME, self.JT2_NAME], plant, context)
        return np.array([drake_velocities[1], drake_velocities[0]])  # [q1_dot, q2_dot]

    def set_velocities_user_order(self, plant: MultibodyPlant, context, user_velocities: np.ndarray):
        """Set velocities in user order [q1_dot, q2_dot]."""
        q1_dot, q2_dot = user_velocities
        # Set in Drake order [q2_dot, q1_dot]
        self.set_jt_velocity([self.JT1_NAME, self.JT2_NAME], plant, context, [q2_dot, q1_dot])

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
        # Use manipulator's set_positions_user_order which handles Drake ordering internally
        manipulator.set_positions_user_order(plant, context, q)
        
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
    Real-time IK solver that computes desired joint angles from cart position.
    
    Inputs:
      0: cart_state (4) = [x, y, ẋ, ẏ]
    Output:
      0: desired_angles (2) = [q1_desired, q2_desired]
    """
    
    def __init__(self, manipulator, plant):
        LeafSystem.__init__(self)
        self.manipulator = manipulator
        self.plant = plant
        self.q_prev = np.array([np.deg2rad(-10.0), np.deg2rad(20.0)])  # [q1, q2] seed
        
        # Input: cart state
        self.DeclareVectorInputPort("cart_state", 4)
        
        # Output: desired joint state [q1, q2, q̇1, q̇2]
        self.DeclareVectorOutputPort("desired_joint_state", 4, self.calc_desired_angles)
    
    def calc_desired_angles(self, context, output):
        cart_state = self.get_input_port(0).Eval(context)
        cart_x, cart_y = cart_state[0], cart_state[1]
        x_dot, y_dot = cart_state[2], cart_state[3]
        
        t = context.get_time()
        
        # CRITICAL COORDINATE MAPPING for right_frame URDF (orientation=[0,0,0]):
        # - Manipulator operates in X-Y plane (planar manipulator)
        #   At q1=q2=0, all joint frames and EE frame are coplanar in X-Y plane
        
        # Solve IK using previous solution as warm start  
        q_desired, success = solve_initial_pose_via_ik(
            self.plant, self.manipulator, 
            np.array([cart_x, cart_y]),  # Direct mapping: cart [x,y] → manipulator [X,Y]
            self.q_prev, 
            pos_tol=0.05,  # Looser tolerance for real-time
            target_z=None,  # Let IK use seed Z (manipulator stays in its X-Y plane)
            verbose=(int(t * 2) % 10 == 0)  # Print every 5s
        )
        
        if success:
            # q_desired is already in natural [q1, q2] order from solve_initial_pose_via_ik
            self.q_prev = q_desired  # Update warm start
        else:
            q_desired = self.q_prev
        
        # Compute desired joint velocities using Jacobian
        # Set up plant context with desired configuration
        temp_context = self.plant.CreateDefaultContext()
        # Set desired configuration using natural [q1, q2] ordering
        self.manipulator.set_positions_user_order(self.plant, temp_context, q_desired)
        
        # Get Jacobian using cup_center frame (EE frame)
        ee_frame = self.manipulator.get_end_effector_frame(self.plant)
        J_full = self.plant.CalcJacobianTranslationalVelocity(
            temp_context,
            JacobianWrtVariable.kQDot,
            ee_frame,
            np.zeros(3),  # Point at cup_center frame origin
            self.plant.world_frame(),
            self.plant.world_frame()
        )
        
        # Extract manipulator velocity indices using joint names (Drake order: [JT1=q2, JT2=q1])
        jt1 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT1_NAME)
        jt2 = self.manipulator.get_joint_by_name(self.plant, self.manipulator.JT2_NAME)
        manip_velocity_indices = [jt1.velocity_start(), jt2.velocity_start()]
        
        # Extract X-Y rows (rows 0 and 1) - both systems now in X-Y plane
        J_xy = J_full[[0, 1], :][:, manip_velocity_indices]  # 2x2: rows [X, Y], manip columns in Drake order [q̇2, q̇1]
        
        # Compute desired joint velocities: q̇_d = J^+ * ẋ_cart
        # Direct mapping: cart velocity [ẋ, ẏ] = manipulator [ẋ, ẏ]
        J_pinv = np.linalg.pinv(J_xy)
        q_dot_drake = J_pinv @ np.array([x_dot, y_dot])  # Drake order [q̇2, q̇1]
        # Convert to user order [q̇1, q̇2]
        q_dot_desired = np.array([q_dot_drake[1], q_dot_drake[0]])
        
        # Output [q1, q2, q̇1, q̇2] in natural order
        output.SetFromVector(np.concatenate([q_desired, q_dot_desired]))


# ============================================================================
# IK WRAPPER (for backward compatibility)
# ============================================================================

def solve_initial_pose_via_ik(
    plant,
    manipulator,
    target_xy,
    q_seed,
    pos_tol=1e-3,
    verbose=False,
    ee_frame_name=None,
    target_z=None,
):
    """
    Wrapper function for backward compatibility.
    Calls manipulator.solve_initial_pose_via_ik() method.
    
    Args:
        plant: MultibodyPlant containing the manipulator
        manipulator: CupManipulator instance
        target_xy: Target [x, y] position
        q_seed: Initial guess [q1, q2]
        pos_tol: Position tolerance
        verbose: Print solver info
        ee_frame_name: EE frame name (optional)
        target_z: Target Z coordinate (optional, if None uses seed Z)
    
    Returns:
        q_solution: Joint angles [q1, q2] in user order
        success: Boolean indicating success
    """
    return manipulator.solve_initial_pose_via_ik(
        plant, target_xy, q_seed, pos_tol, verbose, ee_frame_name, target_z
    )


# ============================================================================
# FINITE-HORIZON LQR CART-PENDULUM CONTROL
# ============================================================================

def run_finite_horizon_lqr_cart_pend_only(
    builder, plant, scene_graph, meshcat, cart_model, manipulator,
    ee_pos_init, physics_config, impedance_config, zft_config, muscle_config, args,
    cart_x_init=None, cart_y_init=None, initial_q=None
):
    """
    Build and run finite-horizon LQR control for cart-pendulum system.
    
    This method creates the complete control architecture:
    - Linearizes the 14D system
    - Creates muscle dynamics, ZFT reference, impedance force, and LQR controller
    - Connects all systems in feedback loop
    - Runs simulation and generates plots
    
    Args:
        builder: DiagramBuilder instance
        plant: MultibodyPlant with cart-pendulum (and optionally manipulator)
        scene_graph: SceneGraph for visualization
        meshcat: Meshcat visualizer instance
        cart_model: Model instance for cart-pendulum
        manipulator: CupManipulator instance (optional, can be None)
        ee_pos_init: Initial end-effector position [x, y, z]
        physics_config: CartPendulumPhysicsConfig
        impedance_config: ImpedanceForceConfig
        zft_config: ZFTReferenceMassConfig
        muscle_config: MuscleDynamicsConfig
        args: Command-line arguments
        initial_q: Initial manipulator joint angles [q1, q2] (natural order)
    """
    # Linearize system
    print(colored("\n🔧 Building linearized 14D system...", "yellow"))
    A, B = build_linearized_system_2d(physics_config, impedance_config, zft_config, muscle_config)
    print(colored(f"✓ System matrices: A ({A.shape}), B ({B.shape})", "green"))
    
    # Create muscle dynamics
    muscle = builder.AddSystem(MuscleDynamics2D(muscle_config))
    muscle.set_name("muscle_dynamics")
    
    # Create ZFT reference mass
    zft = builder.AddSystem(ZFTReferenceMass2D(zft_config))
    zft.set_name("zft_reference")
    
    # Create impedance force
    impedance = builder.AddSystem(ImpedanceForce2D(impedance_config))
    impedance.set_name("impedance_force")
    
    # Create LQR controller
    lqr = builder.AddSystem(FiniteHorizonLQRController2D(A, B, LQR_CONFIG))
    lqr.set_name("lqr_controller")
    
    # Add ZeroOrderHold to break algebraic loop
    state_hold = builder.AddSystem(ZeroOrderHold(0.01, 14))
    state_hold.set_name("state_hold")
    
    # Demux/Mux for state extraction
    cart_state_demux = builder.AddSystem(Demultiplexer([2, 2, 2, 2]))  # [pos, ang, vel, ang_vel]
    full_state_mux = builder.AddSystem(Multiplexer([2, 2, 2, 2, 2, 4]))  # [x,y] [α,β] [ẋ,ẏ] [α̇,β̇] [F_x,F_y] [x_ref,y_ref,ẋ_ref,ẏ_ref]
    cart_state_mux = builder.AddSystem(Multiplexer([2, 2]))  # [x, y] + [ẋ, ẏ]
    
    # Connect cart-pendulum state → demux (only cart-pendulum, not manipulator)
    builder.Connect(
        plant.get_state_output_port(cart_model),
        cart_state_demux.get_input_port()
    )
    
    # Extract cart position and velocity
    builder.Connect(
        cart_state_demux.get_output_port(0),  # [x, y]
        cart_state_mux.get_input_port(0)
    )
    builder.Connect(
        cart_state_demux.get_output_port(2),  # [ẋ, ẏ]
        cart_state_mux.get_input_port(1)
    )
    
    # Connect cart state → ZFT
    builder.Connect(
        cart_state_mux.get_output_port(),
        zft.get_input_port(0)
    )
    
    # Connect muscle → ZFT
    builder.Connect(
        muscle.get_output_port(),
        zft.get_input_port(1)
    )
    
    # Connect cart state → impedance
    builder.Connect(
        cart_state_mux.get_output_port(),
        impedance.get_input_port(0)
    )
    
    # Connect ZFT → impedance
    builder.Connect(
        zft.get_output_port(),
        impedance.get_input_port(1)
    )
    
    # Connect impedance → cart-pendulum only (manipulator remains fixed)
    builder.Connect(
        impedance.get_output_port(),
        plant.get_actuation_input_port(cart_model)
    )
    
    # Assemble full state for LQR: [x, y, α, β, ẋ, ẏ, α̇, β̇, F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref]
    builder.Connect(
        cart_state_demux.get_output_port(0),  # [x, y]
        full_state_mux.get_input_port(0)
    )
    builder.Connect(
        cart_state_demux.get_output_port(1),  # [α, β]
        full_state_mux.get_input_port(1)
    )
    builder.Connect(
        cart_state_demux.get_output_port(2),  # [ẋ, ẏ]
        full_state_mux.get_input_port(2)
    )
    builder.Connect(
        cart_state_demux.get_output_port(3),  # [α̇, β̇]
        full_state_mux.get_input_port(3)
    )
    builder.Connect(
        muscle.get_output_port(),  # [F_x, F_y]
        full_state_mux.get_input_port(4)
    )
    builder.Connect(
        zft.get_output_port(),  # [x_ref, y_ref, ẋ_ref, ẏ_ref]
        full_state_mux.get_input_port(5)
    )
    
    # Connect state → LQR → muscle (with hold to break algebraic loop)
    builder.Connect(
        full_state_mux.get_output_port(),
        state_hold.get_input_port()
    )
    builder.Connect(
        state_hold.get_output_port(),
        lqr.get_input_port()
    )
    builder.Connect(
        lqr.get_output_port(),
        muscle.get_input_port()
    )
    
    # Add loggers
    state_logger = builder.AddSystem(VectorLogSink(8))
    state_logger.set_name("state_logger")
    builder.Connect(plant.get_state_output_port(cart_model), state_logger.get_input_port())
    
    ref_logger = builder.AddSystem(VectorLogSink(4))
    ref_logger.set_name("ref_logger")
    builder.Connect(zft.get_output_port(), ref_logger.get_input_port())
    
    force_logger = builder.AddSystem(VectorLogSink(2))
    force_logger.set_name("force_logger")
    builder.Connect(muscle.get_output_port(), force_logger.get_input_port())
    
    impedance_logger = builder.AddSystem(VectorLogSink(2))
    impedance_logger.set_name("impedance_logger")
    builder.Connect(impedance.get_output_port(), impedance_logger.get_input_port())
    
    # Add visualization
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    
    # Build diagram
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Get plant context
    plant_context = plant.GetMyMutableContextFromRoot(context)
    
    # ========================================================================
    # VISUALIZATION STAGE 1: Default positions (all zeros)
    # ========================================================================
    print(colored("\n📸 Visualizing default configuration (zeros)...", "cyan"))
    diagram.ForcedPublish(context)
    print(colored(f"✓ View at: {meshcat.web_url()}", "green"))
    
    # ========================================================================
    # SET INITIAL STATE
    # ========================================================================
    # Set manipulator joint angles (fixed position) if manipulator is provided
    if manipulator is not None:
        # initial_q is now in natural [q1, q2] order
        if initial_q is None:
            initial_q = np.array([np.deg2rad(-10.0), np.deg2rad(20.0)])  # [q1, q2]
        
        manipulator.set_positions_user_order(plant, plant_context, initial_q)
        plant.SetVelocities(plant_context, manipulator.model_instance, np.zeros(2))
    
    # Set cart-pendulum state
    # If manipulator exists and no cart position override: use EE position
    # Otherwise: use provided cart_x_init, cart_y_init (from command-line args)
    if manipulator is not None and cart_x_init is None and cart_y_init is None:
        # Use EE position when manipulator exists and no override
        cart_x = ee_pos_init[0]
        cart_y = ee_pos_init[1]
    else:
        # Use command-line arguments or defaults
        cart_x = cart_x_init if cart_x_init is not None else 0.0
        cart_y = cart_y_init if cart_y_init is not None else 0.0
    cart_pendulum_positions = np.array([
        cart_x, cart_y,  # Cart at specified or EE x,y
        0.0, 0.0,        # Pendulum hanging down
    ])
    plant.SetPositions(plant_context, cart_model, cart_pendulum_positions)
    plant.SetVelocities(plant_context, cart_model, np.zeros(4))
    
    # ========================================================================
    # VISUALIZATION STAGE 2: Configured initial positions
    # ========================================================================
    print(colored("\n📸 Visualizing configured initial state...", "cyan"))
    if manipulator is not None:
        print(colored(f"  - Manipulator: q1={np.rad2deg(initial_q[1]):.1f}°, q2={np.rad2deg(initial_q[0]):.1f}°", "cyan"))
    print(colored(f"  - Cart: ({cart_x:.3f}, {cart_y:.3f}) m", "cyan"))
    print(colored(f"  - Pendulum: α=0°, β=0° (hanging)", "cyan"))
    diagram.ForcedPublish(context)
    print(colored(f"✓ Configured state visible in Meshcat", "green"))
    
    # Start recording for simulation
    visualizer.StartRecording()
    
    print(colored("\n🚀 Starting simulation...", "cyan"))
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(args.duration)
    
    visualizer.PublishRecording()
    print(colored(f"🎬 Animation: {meshcat.web_url()}\n", "green", attrs=["bold"]))
    
    # Extract data
    state_log = state_logger.FindLog(context)
    ref_log = ref_logger.FindLog(context)
    force_log = force_logger.FindLog(context)
    impedance_log = impedance_logger.FindLog(context)
    
    t = state_log.sample_times()
    state_data = state_log.data()
    ref_data = ref_log.data()
    force_data = force_log.data()
    impedance_data = impedance_log.data()
    
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
    
    # 2D trajectory
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(state_data[0, :], state_data[1, :], 'b-', label='actual')
    ax3.plot(ref_data[0, :], ref_data[1, :], 'r--', label='reference')
    ax3.plot(args.target_x, args.target_y, 'g*', markersize=15, label='target')
    ax3.plot(state_data[0, 0], state_data[1, 0], 'ko', markersize=8, label='start')
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # Pendulum angles
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, np.rad2deg(state_data[2, :]), 'b-', label='pitch (α)')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Pitch Angle [deg]')
    ax4.legend()
    ax4.grid(True)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, np.rad2deg(state_data[3, :]), 'r-', label='roll (β)')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Roll Angle [deg]')
    ax5.legend()
    ax5.grid(True)
    
    # Velocities
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t, state_data[4, :], 'b-', label='ẋ')
    ax6.plot(t, state_data[5, :], 'r-', label='ẏ')
    ax6.plot(t, ref_data[2, :], 'b--', alpha=0.5, label='ẋ_ref')
    ax6.plot(t, ref_data[3, :], 'r--', alpha=0.5, label='ẏ_ref')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Velocity [m/s]')
    ax6.legend()
    ax6.grid(True)
    
    # Forces (Muscle and Impedance)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(t, force_data[0, :], 'b-', label='F_x (muscle)', linewidth=2)
    ax7.plot(t, impedance_data[0, :], 'c--', label='F_x (impedance)', linewidth=1.5, alpha=0.7)
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Force X [N]')
    ax7.legend()
    ax7.grid(True)
    
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(t, force_data[1, :], 'r-', label='F_y (muscle)', linewidth=2)
    ax8.plot(t, impedance_data[1, :], 'm--', label='F_y (impedance)', linewidth=1.5, alpha=0.7)
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('Force Y [N]')
    ax8.legend()
    ax8.grid(True)
    
    # Angular velocities
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(t, np.rad2deg(state_data[6, :]), 'b-', label='α̇')
    ax9.plot(t, np.rad2deg(state_data[7, :]), 'r-', label='β̇')
    ax9.set_xlabel('Time [s]')
    ax9.set_ylabel('Angular Velocity [deg/s]')
    ax9.legend()
    ax9.grid(True)
    
    plt.tight_layout()
    
    # Save plots
    plot_path = 'plots/cart_pendulum_2d_extended_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(colored(f"✓ Plots saved to {plot_path}", "green"))
    
    # Display plots (blocking mode)
    print(colored("📊 Displaying plots... (close window to continue)", "yellow"))
    plt.show(block=True)
    
    # Save cart position and velocity to CSV
    print(colored("\n💾 Saving cart data to CSV...", "yellow"))
    import csv
    from pathlib import Path
    
    csv_dir = Path('data')
    csv_dir.mkdir(exist_ok=True)
    
    csv_filename = csv_dir / 'cart_position_velocity.csv'
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['time_s', 'x_m', 'y_m', 'vx_m_s', 'vy_m_s'])
        
        # Write data rows
        for i in range(len(t)):
            writer.writerow([
                f'{t[i]:.6f}',           # time [s]
                f'{state_data[0, i]:.6f}',  # x position [m]
                f'{state_data[1, i]:.6f}',  # y position [m]
                f'{state_data[4, i]:.6f}',  # x velocity [m/s]
                f'{state_data[5, i]:.6f}',  # y velocity [m/s]
            ])
    
    print(colored(f"✓ Cart data saved to {csv_filename}", "green"))
    print(colored(f"  {len(t)} samples written", "cyan"))
    
    print(colored("\n" + "="*80, "green"))
    print(colored("✓ Simulation Complete!", "green", attrs=["bold"]))
    print(colored("="*80 + "\n", "green"))





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
# STUB FUNCTIONS FOR FUTURE IMPLEMENTATION
# ============================================================================

def run_lqr_manip_ee_traj_track(
        builder, plant, scene_graph, meshcat, cart_model, manipulator,
    ee_pos_init, physics_config, impedance_config, zft_config, muscle_config, args,
    cart_x_init=None, cart_y_init=None
    ):
    """
    LQR control for cart-pendulum with manipulator end-effector trajectory tracking.
    
    System Architecture:
    - Cart-pendulum: Controlled by LQR with muscle dynamics and ZFT reference
    - Manipulator: Tracks cart position using inverse kinematics
    
    Control Flow:
    1. Cart-pendulum state → LQR → muscle dynamics → impedance force → cart
    2. Cart position → IK → manipulator joint angles
    """
    
    # Linearize system
    print(colored("\n🔧 Building linearized 14D system...", "yellow"))
    A, B = build_linearized_system_2d(physics_config, impedance_config, zft_config, muscle_config)
    print(colored(f"✓ System matrices: A ({A.shape}), B ({B.shape})", "green"))
    
    # Create muscle dynamics
    muscle = builder.AddSystem(MuscleDynamics2D(muscle_config))
    muscle.set_name("muscle_dynamics")
    
    # Create ZFT reference mass
    zft = builder.AddSystem(ZFTReferenceMass2D(zft_config))
    zft.set_name("zft_reference")
    
    # Create impedance force
    impedance = builder.AddSystem(ImpedanceForce2D(impedance_config))
    impedance.set_name("impedance_force")
    
    # Create LQR controller
    lqr = builder.AddSystem(FiniteHorizonLQRController2D(A, B, LQR_CONFIG))
    lqr.set_name("lqr_controller")
    
    # Add ZeroOrderHold to break algebraic loop
    state_hold = builder.AddSystem(ZeroOrderHold(0.01, 14))
    state_hold.set_name("state_hold")
    
    # Demux/Mux for state extraction
    cart_state_demux = builder.AddSystem(Demultiplexer([2, 2, 2, 2]))  # [pos, ang, vel, ang_vel]
    full_state_mux = builder.AddSystem(Multiplexer([2, 2, 2, 2, 2, 4]))  # [x,y] [α,β] [ẋ,ẏ] [α̇,β̇] [F_x,F_y] [x_ref,y_ref,ẋ_ref,ẏ_ref]
    cart_state_mux = builder.AddSystem(Multiplexer([2, 2]))  # [x, y] + [ẋ, ẏ]
    
    # ========================================================================
    # STATE EXTRACTION FROM PLANT
    # ========================================================================
    # Cart-pendulum state → demultiplexer to extract individual components
    # Plant output: [x, y, α, β, ẋ, ẏ, α̇, β̇] (8D)
    # Demux splits into: [x,y], [α,β], [ẋ,ẏ], [α̇,β̇]
    builder.Connect(
        plant.get_state_output_port(cart_model),  # Cart-pendulum state (8D)
        cart_state_demux.get_input_port()         # → demux into 4 ports
    )
    
    # ========================================================================
    # CART STATE MUX (POSITION + VELOCITY)
    # ========================================================================
    # Combine cart position and velocity for ZFT and impedance blocks
    # This creates [x, y, ẋ, ẏ] (4D) from demuxed components
    
    # Extract cart position [x, y]
    builder.Connect(
        cart_state_demux.get_output_port(0),  # [x, y] from demux port 0
        cart_state_mux.get_input_port(0)      # → cart_state mux port 0
    )
    
    # Extract cart velocity [ẋ, ẏ]
    builder.Connect(
        cart_state_demux.get_output_port(2),  # [ẋ, ẏ] from demux port 2
        cart_state_mux.get_input_port(1)      # → cart_state mux port 1
    )
    
    # ========================================================================
    # ZFT (ZERO FORCE TRAJECTORY) REFERENCE MASS
    # ========================================================================
    # The ZFT block simulates a virtual mass connected to the cart via impedance
    # Dynamics: M * ẍ_ref = K*(x - x_ref) + D*(ẋ - ẋ_ref) + F_muscle
    # This provides a smooth, filtered reference trajectory
    
    # Input 0: Cart state [x, y, ẋ, ẏ]
    # The cart position and velocity drive the reference mass dynamics
    builder.Connect(
        cart_state_mux.get_output_port(),  # [x, y, ẋ, ẏ] cart state
        zft.get_input_port(0)              # → ZFT cart state input
    )
    
    # Input 1: Muscle forces [F_x, F_y]
    # Muscle forces also affect the reference mass (feedforward coupling)
    builder.Connect(
        muscle.get_output_port(),  # [F_x, F_y] from muscle dynamics
        zft.get_input_port(1)      # → ZFT force input
    )
    
    # ========================================================================
    # IMPEDANCE FORCE COMPUTATION
    # ========================================================================
    # Computes: F_imp = K_imp*(r_ref - r) + D_imp*(ṙ_ref - ṙ)
    # This is the actual force applied to the cart-pendulum system
    
    # Input 0: Cart state [x, y, ẋ, ẏ]
    builder.Connect(
        cart_state_mux.get_output_port(),  # [x, y, ẋ, ẏ] cart state
        impedance.get_input_port(0)        # → impedance cart state input
    )
    
    # Input 1: ZFT reference state [x_ref, y_ref, ẋ_ref, ẏ_ref]
    builder.Connect(
        zft.get_output_port(),       # [x_ref, y_ref, ẋ_ref, ẏ_ref] from ZFT
        impedance.get_input_port(1)  # → impedance reference input
    )
    
    # ========================================================================
    # ACTUATION CONNECTION
    # ========================================================================
    # Connect impedance force → cart-pendulum actuators
    # The impedance force F_imp = K*(r_ref - r) + D*(ṙ_ref - ṙ) is computed
    # from the ZFT reference mass and cart positions/velocities.
    # This force is applied ONLY to the cart-pendulum system, NOT the manipulator.
    # The manipulator remains at its fixed initial configuration.
    builder.Connect(
        impedance.get_output_port(),  # [F_x, F_y] impedance force
        plant.get_actuation_input_port(cart_model)  # Cart-pendulum actuation port
    )
    
    # ========================================================================
    # FULL STATE ASSEMBLY FOR LQR CONTROLLER
    # ========================================================================
    # The LQR controller needs the complete 14D state vector:
    # [x, y, α, β, ẋ, ẏ, α̇, β̇, F_x, F_y, x_ref, y_ref, ẋ_ref, ẏ_ref]
    #
    # This state combines:
    # - Cart-pendulum mechanical state (8D): positions, angles, velocities
    # - Muscle force state (2D): current muscle forces
    # - ZFT reference state (4D): reference mass position and velocity
    #
    # The state is assembled using a Multiplexer with 6 input ports
    
    # Port 0: Cart position [x, y]
    # Extract from demultiplexed cart-pendulum state (positions)
    builder.Connect(
        cart_state_demux.get_output_port(0),  # [x, y] cart position
        full_state_mux.get_input_port(0)      # → LQR state[0:2]
    )
    
    # Port 1: Pendulum angles [α, β]
    # Extract from demultiplexed cart-pendulum state (angles)
    # α = pitch angle, β = roll angle (gimbal representation)
    builder.Connect(
        cart_state_demux.get_output_port(1),  # [α, β] pendulum angles
        full_state_mux.get_input_port(1)      # → LQR state[2:4]
    )
    
    # Port 2: Cart velocity [ẋ, ẏ]
    # Extract from demultiplexed cart-pendulum state (velocities)
    builder.Connect(
        cart_state_demux.get_output_port(2),  # [ẋ, ẏ] cart velocity
        full_state_mux.get_input_port(2)      # → LQR state[4:6]
    )
    
    # Port 3: Pendulum angular velocity [α̇, β̇]
    # Extract from demultiplexed cart-pendulum state (angular velocities)
    builder.Connect(
        cart_state_demux.get_output_port(3),  # [α̇, β̇] pendulum angular velocity
        full_state_mux.get_input_port(3)      # → LQR state[6:8]
    )
    
    # Port 4: Muscle forces [F_x, F_y]
    # These are the state variables of the muscle dynamics system
    # Muscle dynamics: Ḟ = (-F + u) / τ where u is neural command from LQR
    builder.Connect(
        muscle.get_output_port(),       # [F_x, F_y] muscle force state
        full_state_mux.get_input_port(4)  # → LQR state[8:10]
    )
    
    # Port 5: ZFT reference state [x_ref, y_ref, ẋ_ref, ẏ_ref]
    # The ZFT (Zero Force Trajectory) reference mass tracks the cart with
    # impedance coupling. This provides a filtered reference trajectory.
    builder.Connect(
        zft.get_output_port(),          # [x_ref, y_ref, ẋ_ref, ẏ_ref] ZFT state
        full_state_mux.get_input_port(5)  # → LQR state[10:14]
    )
    
    # ========================================================================
    # CONTROL LOOP: LQR → MUSCLE DYNAMICS
    # ========================================================================
    # This forms the main control feedback loop with algebraic loop prevention
    #
    # Flow: state → hold → LQR → muscle → (back to state via ZFT/impedance)
    #
    # ZeroOrderHold breaks the algebraic loop by introducing a one-sample delay
    # This prevents circular dependencies in the diagram evaluation
    
    # Step 1: Full state → ZeroOrderHold (breaks algebraic loop)
    # The hold samples state at 100Hz (dt=0.01s) and holds until next sample
    builder.Connect(
        full_state_mux.get_output_port(),  # [14D] complete system state
        state_hold.get_input_port()        # → hold (breaks algebraic loop)
    )
    
    # Step 2: Held state → LQR controller
    # LQR computes optimal neural command: u = -K(t) * (x - x_goal)
    # Uses time-varying gain K(t) from backward Riccati recursion
    builder.Connect(
        state_hold.get_output_port(),  # [14D] delayed state
        lqr.get_input_port()           # → LQR controller
    )
    
    # Step 3: LQR command → Muscle dynamics
    # LQR outputs neural command u ∈ ℝ² which drives muscle dynamics
    # Muscle acts as a low-pass filter: Ḟ = (-F + u) / τ
    # This models the delay between neural activation and force production
    builder.Connect(
        lqr.get_output_port(),   # [u_x, u_y] neural command from LQR
        muscle.get_input_port()  # → muscle dynamics input
    )
    
    # ========================================================================
    # MANIPULATOR COMPUTED TORQUE CONTROLLER
    # ========================================================================
    # Create a computed torque controller for the manipulator
    # The manipulator end-effector will track the cart position
    
    # Controller with high gains for tight tracking
    # manip_controller = builder.AddSystem(
    #     ComputedTorqueEEController(manipulator, plant, Kp=200.0, Kd=60.0, tau_max=100.0)
    # )
    # manip_controller.set_name("manipulator_controller")

    # Add joint-space computed torque controller
    manip_js_controller = builder.AddSystem(
        ComputedTorqueJointSpaceController(manipulator, plant, Kp=200.0, Kd=60.0, tau_max=100.0)
    )
    manip_js_controller.set_name("manipulator_js_controller")
    
    # IK solver for desired joint angles (for comparison/plotting)
    manip_ik_solver = builder.AddSystem(
        ManipulatorIKDesiredAngles(manipulator, plant)
    )
    manip_ik_solver.set_name("manipulator_ik_solver")
    
    # Connect cart state to IK solver
    builder.Connect(
        cart_state_mux.get_output_port(),  # [x, y, ẋ, ẏ] cart state
        manip_ik_solver.get_input_port(0)  # → IK solver input
    )
    
    # Connect cart state directly to controller desired trajectory
    # Cart state [x, y, ẋ, ẏ] is exactly what the controller needs
    # builder.Connect(
    #     cart_state_mux.get_output_port(),  # [x, y, ẋ, ẏ] cart state
    #     manip_controller.get_input_port(0)  # → controller desired trajectory
    # )
    # 
    # # Connect manipulator state to controller input 1
    # builder.Connect(
    #     plant.get_state_output_port(manipulator.model_instance),  # [q2, q1, q̇2, q̇1] manipulator state (Drake GetJointIndices order!)
    #     manip_controller.get_input_port(1)                        # → controller manipulator state
    # )
    
    # Connect controller output to manipulator actuation
    builder.Connect(
        manip_js_controller.get_output_port(),                       # [τ1, τ2] joint torques (natural order)
        plant.get_actuation_input_port(manipulator.model_instance)  # → manipulator actuators
    )


    
    # Add loggers
    state_logger = builder.AddSystem(VectorLogSink(8))
    state_logger.set_name("state_logger")
    builder.Connect(plant.get_state_output_port(cart_model), state_logger.get_input_port())
     
    ref_logger = builder.AddSystem(VectorLogSink(4))
    ref_logger.set_name("ref_logger")
    builder.Connect(zft.get_output_port(), ref_logger.get_input_port())
    
    force_logger = builder.AddSystem(VectorLogSink(2))
    force_logger.set_name("force_logger")
    builder.Connect(muscle.get_output_port(), force_logger.get_input_port())
    
    impedance_logger = builder.AddSystem(VectorLogSink(2))
    impedance_logger.set_name("impedance_logger")
    builder.Connect(impedance.get_output_port(), impedance_logger.get_input_port())
    
    # Note: manipulator state logger is now connected after state converter (see below)
    
    # Add logger for cart trajectory being sent to manipulator controller (directly from cart_state_mux)
    cart_traj_logger = builder.AddSystem(VectorLogSink(4))
    cart_traj_logger.set_name("cart_traj_logger")
    builder.Connect(cart_state_mux.get_output_port(), cart_traj_logger.get_input_port())
    
    # Add logger for manipulator torques
    # manip_torque_logger = builder.AddSystem(VectorLogSink(2))
    # manip_torque_logger.set_name("manip_torque_logger")
    # builder.Connect(manip_controller.get_output_port(), manip_torque_logger.get_input_port())
    
    # Add logger for desired joint state from IK [q1, q2, q̇1, q̇2]
    manip_desired_state_logger = builder.AddSystem(VectorLogSink(4))
    manip_desired_state_logger.set_name("manip_desired_state_logger")
    builder.Connect(manip_ik_solver.get_output_port(), manip_desired_state_logger.get_input_port())
    
   
    
    # Connect desired joint state from IK to joint-space controller
    builder.Connect(
        manip_ik_solver.get_output_port(),  # [q1_d, q2_d, q̇1_d, q̇2_d]
        manip_js_controller.get_input_port(0)  # → JS controller desired state
    )
    
    # Connect plant state directly to joint-space controller (natural URDF ordering)
    builder.Connect(
        plant.get_state_output_port(manipulator.model_instance),  # [q1, q2, q̇1, q̇2] natural order
        manip_js_controller.get_input_port(1)  # → JS controller current state
    )
    
    # Add logger for joint-space controller torques
    manip_js_torque_logger = builder.AddSystem(VectorLogSink(2))
    manip_js_torque_logger.set_name("manip_js_torque_logger")
    builder.Connect(manip_js_controller.get_output_port(), manip_js_torque_logger.get_input_port())
    
    # Log manipulator state directly from plant (natural URDF ordering)
    manip_state_logger_natural = builder.AddSystem(VectorLogSink(4))
    manip_state_logger_natural.set_name("manip_state_logger")
    builder.Connect(plant.get_state_output_port(manipulator.model_instance), manip_state_logger_natural.get_input_port())
    
    # Add system to compute end-effector position and velocity
    class ManipulatorEEStateComputer(LeafSystem):
        """Computes manipulator end-effector position and velocity from joint state."""
        def __init__(self, plant, manipulator):
            LeafSystem.__init__(self)
            self.plant = plant
            self.manipulator = manipulator
            
            # Input: manipulator state [q1, q2, q̇1, q̇2]
            self.DeclareVectorInputPort("manip_state", 4)
            
            # Output: EE state [x, y, ẋ, ẏ]
            self.DeclareVectorOutputPort(
                "ee_state",
                4,
                self.CalcEEState
            )
        
        def CalcEEState(self, context, output):
            """Calculate EE position and velocity from joint state."""
            # Get manipulator state
            manip_state = self.get_input_port(0).Eval(context)
            
            # Create fresh context for this computation
            temp_context = self.plant.CreateDefaultContext()
            
            # Set state in temp context
            self.manipulator.set_state_in_plant(self.plant, temp_context, manip_state)
            
            # Calculate EE position
            ee_pos = self.manipulator.CalcPosition(self.plant, temp_context)
            
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
            ee_vel = J_xy @ manip_state[2:4]
            
            # Output [x, y, ẋ, ẏ]
            output.SetFromVector(np.array([ee_pos[0], ee_pos[1], ee_vel[0], ee_vel[1]]))
    
    # Add EE state computer and logger
    ee_state_computer = builder.AddSystem(ManipulatorEEStateComputer(plant, manipulator))
    ee_state_computer.set_name("ee_state_computer")
    
    builder.Connect(
        plant.get_state_output_port(manipulator.model_instance),
        ee_state_computer.get_input_port(0)
    )
    
    ee_state_logger = builder.AddSystem(VectorLogSink(4))
    ee_state_logger.set_name("ee_state_logger")
    builder.Connect(ee_state_computer.get_output_port(), ee_state_logger.get_input_port())
    
    # Add visualization
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    
    # Create frame list for updater (will be populated after plant is finalized)
    # We'll create it with a temporary context, then add updater system
    temp_context = plant.CreateDefaultContext()
    frame_list = []
    from pydrake.multibody.tree import FrameIndex
    for i in range(plant.num_frames()):
        frame = plant.get_frame(FrameIndex(i))
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
    
    # Add frame updater system
    frame_updater = builder.AddSystem(
        MeshcatFrameUpdater(meshcat, plant, frame_list, update_period=0.033)  # 30 Hz
    )
    frame_updater.set_name("frame_updater")
    
    # Connect plant state to frame updater
    # We need FULL plant state (all model instances)
    builder.Connect(
        plant.get_state_output_port(),  # Full plant state
        frame_updater.get_input_port(0)
    )
    
    # Build and simulate
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    # Set initial state
    plant_context = plant.GetMyMutableContextFromRoot(context)
    
    # Set cart-pendulum state FIRST
    # Use provided cart initial positions if given, otherwise use EE position
    cart_x = cart_x_init if cart_x_init is not None else ee_pos_init[0]
    cart_y = cart_y_init if cart_y_init is not None else ee_pos_init[1]
    
    # Check if cart position is within manipulator workspace
    # Manipulator workspace: roughly x ∈ [-2.5, 0.5], y ∈ [-0.5, 2.5] based on link lengths
    link1_length = 1.51  # meters (from URDF)
    link2_length = 1.509  # meters
    max_reach = link1_length + link2_length  # ~3.0m
    min_reach = abs(link1_length - link2_length)  # ~0m
    
    cart_distance = np.sqrt(cart_x**2 + cart_y**2)
    if cart_distance > max_reach * 0.9:  # 90% of max reach for safety
        print(colored(f"\n⚠ WARNING: Cart position ({cart_x:.3f}, {cart_y:.3f}) is near/outside manipulator workspace!", "yellow"))
        print(colored(f"  Distance from base: {cart_distance:.3f}m, Max safe reach: {max_reach*0.9:.3f}m", "yellow"))
        print(colored(f"  Manipulator tracking may be poor or fail!", "yellow"))
    
    cart_pendulum_positions = np.array([
        cart_x, cart_y,  # Cart at specified or EE x,y
        0.0, 0.0,        # Pendulum hanging down
    ])
    plant.SetPositions(plant_context, cart_model, cart_pendulum_positions)
    plant.SetVelocities(plant_context, cart_model, np.zeros(4))
    
    # Set manipulator joint angles using IK to match cart position
    # Solve IK to place manipulator EE at cart position
    print(colored(f"\n🔧 Solving IK for manipulator to match cart initial position ({cart_x:.3f}, {cart_y:.3f})...", "yellow"))
    q_seed = np.array([np.deg2rad(-10.0), np.deg2rad(20.0)])  # [q1, q2]
    q_init, ik_success = solve_initial_pose_via_ik(
        plant, manipulator, np.array([cart_x, cart_y]), q_seed, pos_tol=0.01
    )
    
    if ik_success:
        print(colored(f"✓ IK succeeded: q1={np.rad2deg(q_init[0]):.1f}°, q2={np.rad2deg(q_init[1]):.1f}°", "green"))
    else:
        print(colored(f"⚠ IK failed, using seed: q1={np.rad2deg(q_seed[0]):.1f}°, q2={np.rad2deg(q_seed[1]):.1f}°", "yellow"))
        q_init = q_seed
    
    # q_init is already in natural [q1, q2] order
    manipulator.set_positions_user_order(plant, plant_context, q_init)
    plant.SetVelocities(plant_context, manipulator.model_instance, np.zeros(2))
    
    # Calculate actual EE position after IK (may differ slightly from target due to IK tolerance)
    ee_pos_actual = manipulator.CalcPosition(plant, plant_context)
    
    # Both manipulator and cart work in X-Y plane after rotation
    # Direct 1:1 mapping: manipulator [X,Y] → cart [X,Y]
    cart_x, cart_y = ee_pos_actual[0], ee_pos_actual[1]  # Direct X-Y mapping
    cart_pendulum_positions = np.array([
        cart_x, cart_y,  # Cart at actual EE (x from manip X, y from manip Z)
        0.0, 0.0,        # Pendulum hanging down
    ])
    plant.SetPositions(plant_context, cart_model, cart_pendulum_positions)
    plant.SetVelocities(plant_context, cart_model, np.zeros(4))
    
    # ========================================================================
    # VISUALIZATION: Show configured initial state
    # ========================================================================
    print(colored("\n📸 Visualizing configured initial state...", "cyan"))
    print(colored(f"  - Manipulator: q1={np.rad2deg(q_init[0]):.1f}°, q2={np.rad2deg(q_init[1]):.1f}° (will track cart)", "cyan"))
    print(colored(f"  - Cart: ({cart_x:.3f}, {cart_y:.3f}) m", "cyan"))
    print(colored(f"  - Pendulum: α=0°, β=0° (hanging)", "cyan"))
    diagram.ForcedPublish(context)
    
    # Add coordinate frames to meshcat (initial setup)
    frame_list_init = add_frames_to_meshcat(meshcat, plant, plant_context, manipulator, cart_model)
    
    print(colored(f"✓ Initial state visible at: {meshcat.web_url()}", "green"))
    print(colored(f"✓ Frame updater active (30 Hz updates)", "cyan"))
    
    visualizer.StartRecording()
    
    print(colored("\n🚀 Starting simulation...", "cyan"))
    simulator.set_target_realtime_rate(1.0)
    
    # Use while loop for debugging instead of AdvanceTo
    dt_sim = 0.01  # 10ms simulation timestep
    current_time = 0.0
    debug_interval = 9.0  # Print debug info every second
    next_debug_time = debug_interval
    
    print(colored("📊 Simulation loop started (use VSCode debugger to step through)", "yellow"))
    print(colored("   Set breakpoints inside the loop for debugging", "cyan"))
    print(colored(f"   Target duration: {args.duration:.1f}s, timestep: {dt_sim*1000:.0f}ms\n", "cyan"))
    ee_pos_list = []
    while current_time < args.duration:
        # Advance simulation by small timestep
        simulator.AdvanceTo(current_time + dt_sim)
        current_time += dt_sim
        
        # Optional: Print debug info periodically
        if current_time >= next_debug_time:
            # Get current state for debugging
            plant_context = plant.GetMyMutableContextFromRoot(context)
            cart_state = plant.GetPositionsAndVelocities(plant_context, cart_model)
            manip_state = plant.GetPositionsAndVelocities(plant_context, manipulator.model_instance)
            
            ee_pos_list.append(manipulator.CalcPosition(plant, plant_context))
            ee_pos = manipulator.CalcPosition(plant, plant_context)
            
            print(colored(f"\n[DEBUG t={current_time:.2f}s]", "cyan"))
            print(f"  Cart: x={cart_state[0]:.3f}, y={cart_state[1]:.3f}")
            print(f"  Pendulum: α={np.rad2deg(cart_state[2]):.1f}°, β={np.rad2deg(cart_state[3]):.1f}°")
            print(f"  Cart velocity: ẋ={cart_state[4]:.3f}, ẏ={cart_state[5]:.3f}")
            print(f"  Manipulator: q1={np.rad2deg(manip_state[0]):.1f}°, q2={np.rad2deg(manip_state[1]):.1f}°")
            print(f"  End-Effector: x={ee_pos[0]:.3f}, y={ee_pos[1]:.3f}, z={ee_pos[2]:.3f}")
            print(f"  EE→Cart error: {np.sqrt((ee_pos[0]-cart_state[0])**2 + (ee_pos[1]-cart_state[1])**2)*1000:.1f} mm")
            
            next_debug_time += debug_interval
        else:
            # Calculate end-effector position
            ee_pos_list.append(manipulator.CalcPosition(plant, plant_context))
        
        # You can add conditional breakpoint here for specific time:
        # Example: Stop at t≈3.0s for inspection
        # if abs(current_time - 3.0) < dt_sim/2:
        #     # Set VSCode breakpoint on this line
        #     pass  # <-- Debugger will pause here when current_time ≈ 3.0s
    
    print(colored(f"\n✓ Simulation complete at t={current_time:.2f}s", "green"))
    
    visualizer.PublishRecording()
    print(colored(f"🎬 Animation: {meshcat.web_url()}\n", "green", attrs=["bold"]))
    
    # Extract data
    state_log = state_logger.FindLog(context)
    ref_log = ref_logger.FindLog(context)
    force_log = force_logger.FindLog(context)
    impedance_log = impedance_logger.FindLog(context)
    manip_state_log = manip_state_logger_natural.FindLog(context)  # Using converted natural state
    cart_traj_log = cart_traj_logger.FindLog(context)
    # manip_torque_log = manip_torque_logger.FindLog(context)
    manip_desired_state_log = manip_desired_state_logger.FindLog(context)
    manip_js_torque_log = manip_js_torque_logger.FindLog(context)
    ee_state_log = ee_state_logger.FindLog(context)
    
    t = state_log.sample_times()
    state_data = state_log.data()
    ref_data = ref_log.data()
    force_data = force_log.data()
    impedance_data = impedance_log.data()
    manip_state_data = manip_state_log.data()  # [q1, q2, q̇1, q̇2] natural order (from URDF)
    cart_traj_data = cart_traj_log.data()  # [x, y, ẋ, ẏ] sent to controller
    # manip_torque_data = manip_torque_log.data()  # [τ1, τ2] from EE controller
    manip_desired_state_data = manip_desired_state_log.data()  # [q1_d, q2_d, q̇1_d, q̇2_d] from IK
    manip_js_torque_data = manip_js_torque_log.data()  # [τ1, τ2] from joint-space controller
    ee_state_data = ee_state_log.data()  # [x, y, ẋ, ẏ] end-effector state
    
    # Debug: Print cart trajectory being sent to manipulator
    print(colored("\n🔍 DEBUG: Cart trajectory sent to manipulator controller:", "yellow"))
    print(f"  t=0.0s: x={cart_traj_data[0, 0]:.3f}, y={cart_traj_data[1, 0]:.3f}, ẋ={cart_traj_data[2, 0]:.3f}, ẏ={cart_traj_data[3, 0]:.3f}")
    print(f"  t=1.0s: x={cart_traj_data[0, int(len(t)/args.duration)]:.3f}, y={cart_traj_data[1, int(len(t)/args.duration)]:.3f}")
    print(f"  t=end:  x={cart_traj_data[0, -1]:.3f}, y={cart_traj_data[1, -1]:.3f}")
    print(f"  Cart actual at t=0: x={state_data[0, 0]:.3f}, y={state_data[1, 0]:.3f}")
    print(f"  Cart actual at end: x={state_data[0, -1]:.3f}, y={state_data[1, -1]:.3f}")
    # print(f"  Manipulator torques: τ1=[{manip_torque_data[0, 0]:.2f}, {manip_torque_data[0, -1]:.2f}], τ2=[{manip_torque_data[1, 0]:.2f}, {manip_torque_data[1, -1]:.2f}]")
    
    # Plot results
    print(colored("📈 Generating plots...", "yellow"))
    
    # Extract EE position and velocity from logged data (no post-processing needed!)
    ee_positions = ee_state_data[0:2, :]  # [x, y]
    ee_velocities = ee_state_data[2:4, :]  # [ẋ, ẏ]
    
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(5, 4, figure=fig)
    
    # Row 1: Cart position
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, state_data[0, :], 'b-', label='x (cart)', linewidth=2)
    ax1.plot(t, ref_data[0, :], 'r--', label='x_ref', linewidth=1.5)
    ax1.plot(t, cart_traj_data[0, :], 'c-.', label='x (to manip)', linewidth=1.5, alpha=0.7)
    ax1.plot(t, ee_positions[0, :], 'g:', label='x_EE', linewidth=2)
    ax1.axhline(args.target_x, color='m', linestyle=':', label='target')
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
    ax2.axhline(args.target_y, color='m', linestyle=':', label='target')
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
    ax3.plot(args.target_x, args.target_y, 'm*', markersize=15, label='target')
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
    # Manipulator joint angles from logged data (natural [q1, q2] order)
    q1_deg = np.rad2deg(manip_state_data[0, :])  # manip_state_data[0] is q1
    q2_deg = np.rad2deg(manip_state_data[1, :])  # manip_state_data[1] is q2
    # Desired angles from IK
    q1_des_deg = np.rad2deg(manip_desired_state_data[0, :])  # [q1_d, q2_d, q̇1_d, q̇2_d]
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
    # Joint velocities from logged data (natural [q1, q2, q̇1, q̇2] order)
    q1_dot_deg = np.rad2deg(manip_state_data[2, :])  # manip_state_data[2] is q̇1
    q2_dot_deg = np.rad2deg(manip_state_data[3, :])  # manip_state_data[3] is q̇2
    # Desired velocities from IK
    q1_dot_des_deg = np.rad2deg(manip_desired_state_data[2, :])  # q̇1_d from IK
    q2_dot_des_deg = np.rad2deg(manip_desired_state_data[3, :])  # q̇2_d from IK
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
    # EE position (computed from joint angles)
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
    # EE velocity (computed from Jacobian)
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
    
    # Display plots (blocking mode)
    print(colored("📊 Displaying plots... (close window to continue)", "yellow"))
    plt.show(block=True)
    
    print(colored("\n" + "="*80, "green"))
    print(colored("✓ Simulation Complete!", "green", attrs=["bold"]))
    print(colored("="*80 + "\n", "green"))

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
        
        manipulator_config = create_cup_manipulator_config(
            urdf_path="model_using_onshape_to_robot/cup_manipulator2/cup_manipulator_obj_right_frame.urdf", 
            joint_angles=(np.deg2rad(-10.0), np.deg2rad(20.0)),  # Initial config: q1=-10°, q2=20°
            damping=(0.1, 0.1),
        )
        #Initialize manipulator and load URDF into plant
        manipulator = CupManipulator(manipulator_config, enable_visualization=True)
        parser = Parser(plant)
        manipulator.load_urdf_to_plant(plant, parser)  # Loads URDF, creates model instance
        # Rotate base -90° around Y to align manipulator with X-Y plane (same as cart)
        # This makes manipulator X-axis → world X-axis, manipulator Z-axis → world Y-axis
        manipulator.weld_base_to_world(plant, orientation=np.array([0.0, 0, 0.0]))
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
        manipulator.set_positions_user_order(plant, temp_context, initial_q)
        
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
        manipulator.set_positions_user_order(plant, plant_context, initial_q)
        
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
                                q_solution, success = solve_initial_pose_via_ik(
                                    plant, manipulator, target_xy, current_manip_q, pos_tol=0.001, verbose=True
                                )
                                
                                if success:
                                    current_manip_q = q_solution
                                    
                                    # Update manipulator joint positions
                                    manipulator.set_positions_user_order(plant, plant_context, current_manip_q)
                                    
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
    
    elif args.mode == 'finite-horizon-lqr-for-min-effort_cart_pend_only':
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
        
        manipulator_config = create_cup_manipulator_config(
            urdf_path="model_using_onshape_to_robot/cup_manipulator/cup_manipulator_obj_natural_order.urdf", 
            joint_angles=(np.deg2rad(-10.0), np.deg2rad(20.0)),  # Initial config: q1=-10°, q2=20°
            damping=(0.1, 0.1),
        )
        #Initialize manipulator and load URDF into plant
        manipulator = CupManipulator(manipulator_config, enable_visualization=True)
        parser = Parser(plant)
        manipulator.load_urdf_to_plant(plant, parser)  # Loads URDF, creates model instance
        # Rotate base -90° around Y to align manipulator with X-Y plane (same as cart)
        # This makes manipulator X-axis → world X-axis, manipulator Z-axis → world Y-axis
        manipulator.weld_base_to_world(plant, orientation=np.array([0.0, -np.pi/2, 0.0]))
        
        # Add actuators to manipulator joints
        manipulator.add_joint_actuators(plant)
        
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
        
        
        plant.Finalize()  # Must be called before adding to diagram
        print(colored(f"\n✓ Plant finalized with {plant.num_positions()} total positions, "
                    f"{plant.num_velocities()} total velocities", "green"))
        

        # ========================================================================
        # CONFIGURE INITIAL STATE
        # ========================================================================
        # Initial configuration in natural [q1, q2] order
        initial_q = np.array([np.deg2rad(-40.0), np.deg2rad(80.0)])  # [q1, q2]
        
        # Calculate EE position at configured joint angles
        temp_context = plant.CreateDefaultContext()
        manipulator.set_positions_user_order(plant, temp_context, initial_q)
        ee_pos_3d = manipulator.CalcPosition(plant, temp_context)
        
        # Use actual 3D position from manipulator EE
        # Cart state: [x, y, α, β] where x,y are from EE's X,Z coordinates in world frame
        cart_init_pos = np.array([ee_pos_3d[0], ee_pos_3d[2], 0.0, 0.0])  # [x, y, α, β]
        
        # Get world frame positions for debugging
        ee_world_pos = manipulator.get_end_effector_position(plant, temp_context)
        
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
        print(colored("\n🚀 Running finite-horizon LQR control for cart-pendulum only...", "cyan"))
        # Create fresh builder for control diagram (cart-pendulum only, no manipulator)
        control_builder = DiagramBuilder()
        
        # Create a new plant with ONLY cart-pendulum (no manipulator)
        cart_only_plant = MultibodyPlant(time_step=0.001)
        cart_only_scene_graph = control_builder.AddSystem(SceneGraph())
        cart_only_plant.RegisterAsSourceForSceneGraph(cart_only_scene_graph)
        
        # Add cart-pendulum to the new plant
        cart_only_pendulum = CartPendulum2DExtended(physics_config, z_offset=z_offset_from_urdf)
        cart_only_model = cart_only_plant.AddModelInstance("cart_pendulum")
        cart_only_pendulum.build_plant(cart_only_plant, cart_only_model)
        cart_only_plant.Finalize()
        
        # Add the cart-only plant to builder
        control_builder.AddSystem(cart_only_plant)
        
        # Connect plant to scene graph
        control_builder.Connect(
            cart_only_plant.get_geometry_pose_output_port(),
            cart_only_scene_graph.get_source_pose_port(cart_only_plant.get_source_id())
        )
        control_builder.Connect(
            cart_only_scene_graph.get_query_output_port(),
            cart_only_plant.get_geometry_query_input_port()
        )
        
        run_finite_horizon_lqr_cart_pend_only(
            control_builder, cart_only_plant, cart_only_scene_graph, meshcat, cart_only_model, None,
            None, physics_config, impedance_config, zft_config, muscle_config, args,
            cart_x_init=args.cart_x_init, cart_y_init=args.cart_y_init, initial_q=None
        )
    
    elif args.mode == 'lqr-manip-ee-traj-track':
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
        
        manipulator_config = create_cup_manipulator_config(
            urdf_path="model_using_onshape_to_robot/cup_manipulator2/cup_manipulator_obj_right_frame.urdf", 
            joint_angles=(np.deg2rad(-10.0), np.deg2rad(20.0)),  # Initial config: q1=-10°, q2=20°
            damping=(0.1, 0.1),
        )
        #Initialize manipulator and load URDF into plant
        manipulator = CupManipulator(manipulator_config, enable_visualization=True)
        parser = Parser(plant)
        manipulator.load_urdf_to_plant(plant, parser)  # Loads URDF, creates model instance
        # Rotate base -90° around Y to align manipulator with X-Y plane (same as cart)
        # This makes manipulator X-axis → world X-axis, manipulator Z-axis → world Y-axis
        manipulator.weld_base_to_world(plant, orientation=np.array([0.0, 0.0, 0.0]))
        
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
        
        
        plant.Finalize()  # Must be called before adding to diagram
        print(colored(f"\n✓ Plant finalized with {plant.num_positions()} total positions, "
                    f"{plant.num_velocities()} total velocities", "green"))
        
        # Add plant to builder BEFORE passing to run function
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
        
        
        # ========================================================================
        # CONFIGURE INITIAL STATE
        # ========================================================================
        # Initial configuration in natural [q1, q2] order
        initial_q = np.array([np.deg2rad(-10.0), np.deg2rad(20.0)])  # [q1, q2]
        
        # Calculate EE position at configured joint angles
        temp_context = plant.CreateDefaultContext()
        manipulator.set_positions_user_order(plant, temp_context, initial_q)
        ee_pos_init = manipulator.CalcPosition(plant, temp_context)
        # For cup_manipulator_obj_right_frame.urdf (orientation=[0,0,0]):
        # Manipulator operates in X-Y plane (planar manipulator on X-Y)
        # Cart also operates in X-Y plane (prismatic joints along X and Y axes)
        # At q1=q2=0, all frames (joints, EE, cart) are coplanar in X-Y plane
        ee_pos_3d = ee_pos_init  # Keep full 3D coordinates [X, Y, Z] in world frame
        
        # Define cart initial position at manipulator EE
        # Cart state: [x, y, α, β] where:
        #   - x = world X (horizontal)
        #   - y = world Y (horizontal, perpendicular to X)
        # CRITICAL: Both manipulator and cart operate in X-Y plane, so direct mapping:
        cart_init_pos = np.array([ee_pos_3d[0], ee_pos_3d[1], 0.0, 0.0])  # [x from EE_X, y from EE_Y, α, β]
        
        # Set cart position in temp context for frame visualization
        plant.SetPositions(temp_context, cart_model, cart_init_pos)
        
        # Get world frame positions for debugging
        ee_world_pos = manipulator.get_end_effector_position(plant, temp_context)
        
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
        
        # # Plot coordinate frames to verify orientation
        # plot_frames_top_view(plant, temp_context, manipulator, cart_model, 
        #                    title="Initial Frame Orientations - LQR Tracking Mode")
        # plt.show(block=False)
        # plt.pause(0.1)  # Brief pause to display plot
        
        print(colored("\n🚀 Running LQR with manipulator EE trajectory tracking (computed torque)...", "cyan"))
        
        # Run the LQR controller with manipulator EE tracking
        # The function will build its own diagram with all control systems
        run_lqr_manip_ee_traj_track(
            builder, plant, scene_graph, meshcat, cart_model, manipulator,
            ee_pos_3d, physics_config, impedance_config, zft_config, muscle_config, args,
            cart_x_init=cart_init_pos[0], cart_y_init=cart_init_pos[1]
        )


if __name__ == "__main__":
    main()

