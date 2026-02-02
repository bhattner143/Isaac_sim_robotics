"""
Cart-Pendulum with 2-DOF Planar Manipulator - PyDrake Implementation

This is a PyDrake version of the Isaac Sim controller, using Drake's multibody dynamics
instead of Isaac Sim's PhysX simulation. It maintains the same class structure and
functionality as the Isaac Sim version.

Demonstrates:
1. Custom URDF robot import via Drake's URDF parser
2. Separate 2-DOF planar manipulator
3. Multi-body dynamics simulation with Drake
4. Scene visualization with Meshcat
5. Similar simulation modes as Isaac Sim version

System:
- Cart: Moves along X-axis on rails at height 1.325m
- Pendulum: Hangs downward, rotates in XZ plane
- 2-DOF Manipulator: Planar manipulator in XY plane
"""

# ============================================================================
# IMPORTS: Standard Python Libraries and Drake
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
import argparse
import os
import json
import math
from datetime import datetime
from termcolor import colored

# Drake imports
from pydrake.all import (
    # Core simulation
    Simulator,
    DiagramBuilder,
    
    # Multibody dynamics
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
    Parser,
    CoulombFriction,
    RevoluteJoint,
    PrismaticJoint,
    
    # Visualization
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    StartMeshcat,
    Meshcat,
    
    # Scene graph
    SceneGraph,
    
    # Controllers
    TrajectorySource,
    PassThrough,
    LogVectorOutput,
    
    # Mathematical utilities
    Quaternion,
    RotationMatrix,
    RollPitchYaw,
    RigidTransform,
    
    # Time
    AbstractValue,
    BasicVector,
    
    # Frames
    Frame,
)

# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    choices=["scene-viz", "simulation", "coupled-motion", "cart-toward-manipulator"],
    default="coupled-motion",
    help="Simulation mode: 'scene-viz' (static), 'simulation' (physics), 'coupled-motion' (manipulator moves cart), 'cart-toward-manipulator' (cart moves toward manipulator EE)",
)
parser.add_argument(
    "--visualize",
    type=bool,
    default=True,
    help="Enable Meshcat visualization",
)
parser.add_argument(
    "--interactive",
    type=bool,
    default=True,
    help="Enable interactive play/pause/repeat controls in Meshcat",
)
args, _ = parser.parse_known_args()

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# --- Cart-Pendulum Configuration ---
CART_PENDULUM_URDF_PATH = str(Path("model/manipulators/cart_pendulum_2dof.urdf").absolute())
CART_PENDULUM_INITIAL_JOINT_POSITIONS = [0.0, 0.0]  # [cart_slider, pendulum_joint]
CART_PENDULUM_POSITION = (0.0, 0.0, 0.0)
CART_PENDULUM_JOINT_DAMPING = [0.5, 0.05]
CART_PENDULUM_JOINT_STIFFNESS = [0, 0.1]
CART_PENDULUM_JOINT_FRICTION = [0.05, 0.0]

# --- Manipulator Configuration ---
MANIPULATOR_URDF_PATH = str(Path("model/manipulators/2dof_planar_manipulator_pydrake.urdf").absolute())
MANIPULATOR_POSITION = (0.0, 0.0, 0.0)  # Position is now in URDF at (-3, 0, 0)
MANIPULATOR_INITIAL_JOINT_POSITIONS = [math.radians(50), math.radians(-100.0)]
MANIPULATOR_JOINT_DAMPING = [0.1, 0.1]
MANIPULATOR_JOINT_FRICTION = [0.0, 0.0]

# --- EE-Cart Coupling Joint Configuration ---
COUPLING_JOINT_TYPE = "fixed"  # Options: "fixed", "revolute", or "prismatic"
EE_CART_COUPLING_JOINT_STIFFNESS = 500.0  # N·m/rad
EE_CART_COUPLING_JOINT_DAMPING = 100.0     # N·m·s/rad

# --- Coupled Motion Initial Positions (set before plant finalization) ---
CART_PENDULUM_COUPLED_MOTION = [-1.7144, 0.0]  # [cart_position (m), pendulum_position (rad)]
EE_MANIPULATOR_INITIAL_POSITION = [-2.5, 0.0, 0.5]  # [x (m), y (m), z (m)] - EE position to align with cart

# --- Simulation Configuration ---
SIMULATION_MODE = args.mode
VISUALIZE = args.visualize
INTERACTIVE = args.interactive
SIMULATOR_TIME_STEP = 0.001  # 1 kHz simulation
SIMULATION_DURATION = 20.0  # seconds
PENDULUM_SETTLING_TIME = 2.0  # seconds

# --- Visualization Configuration ---
REALTIME_RATE = 0.5  # 1.0 = real-time, 0.5 = half speed (slower for better observation)
VISUALIZATION_UPDATE_EVERY_STEP = True  # Update Meshcat every simulation step
VISUALIZATION_FRAME_RATE = 60.0  # Target FPS for visualization (if not updating every step)
PRINT_INTERVAL = 0.25  # Print status every N seconds

# --- Cart-Toward-Manipulator Mode Parameters ---
CART_SPEED = 0.2  # meters per second
CART_MODE_TIME_STEP = 0.05  # Time step for cart-toward-manipulator mode (50ms)
CONVERGENCE_THRESHOLD = 0.0001  # Stop when within this distance (meters), only for mode "cart-toward-manipulator"
MAX_ITERATIONS = 10000  # Safety limit for iterations

# ============================================================================
# PARAMETER CLASSES
# ============================================================================

@dataclass
class RobotParams:
    """Parameters for robot configuration."""
    urdf_path: str
    initial_joint_positions: List[float]
    joint_damping: List[float]
    joint_stiffness: List[float]
    joint_friction: List[float]
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # World position (x, y, z)
    link_lengths: List[float] = None


@dataclass
class RobotState:
    """Runtime state of initialized robot."""
    model_instance: int  # Drake's model instance ID
    num_dof: int
    dof_names: List[str]
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # World position


# ============================================================================
# ROBOT BASE CLASS (ABSTRACT)
# ============================================================================

class RobotBase(ABC):
    """
    Abstract Base Class for Robots using Drake
    
    DESIGN PATTERN: Template Method Pattern
    Provides common interface for all robots (CartPendulum, PlanarManipulator)
    """
    
    def __init__(self, params: RobotParams, name: str = "robot"):
        """Initialize robot with configuration parameters."""
        self.params = params
        self.name = name
        self.state: Optional[RobotState] = None
        self.model_instance: Optional[int] = None
    
    def load_urdf_to_plant(self, plant: MultibodyPlant, parser: Parser) -> int:
        """
        Load URDF to plant using Drake's URDF parser.
        
        Args:
            plant: Drake MultibodyPlant
            parser: Drake URDF parser
            
        Returns:
            model_instance: Drake's model instance ID
        """
        print(f"\nLoading robot from URDF: {self.params.urdf_path}")
        
        if not os.path.exists(self.params.urdf_path):
            raise FileNotFoundError(f"URDF not found: {self.params.urdf_path}")
        
        # AddModels returns a list of model instances
        model_instances = parser.AddModels(self.params.urdf_path)
        if not model_instances:
            raise RuntimeError(f"Failed to load URDF: {self.params.urdf_path}")
        
        model_instance = model_instances[0]
        print(f"✓ Robot loaded with model instance: {model_instance}")
        return model_instance
    
    def initialize_state(self, plant: MultibodyPlant, model_instance: int):
        """Initialize robot state after plant is finalized."""
        self.model_instance = model_instance
        
        # Get number of DOFs for this model
        num_dof = plant.num_actuated_dofs(model_instance)
        
        # Get DOF names
        dof_names = []
        for joint_idx in plant.GetJointIndices(model_instance):
            joint = plant.get_joint(joint_idx)
            dof_names.append(joint.name())
        
        self.state = RobotState(
            model_instance=model_instance,
            num_dof=num_dof,
            dof_names=dof_names,
            position=self.params.position
        )
        
        print(f"✓ Robot '{self.name}' initialized with {num_dof} DOFs at position {self.params.position}")
        print(f"  DOF names: {dof_names}")
    
    def set_pose(self, plant: MultibodyPlant, context):
        """Set the position/pose of the robot in the world by finding and adjusting base joints."""
        if not self.model_instance or not self.params.position:
            return
        
        try:
            position = np.array(self.params.position)
            joint_indices = plant.GetJointIndices(self.model_instance)
            
            if len(joint_indices) == 0:
                print(f"Warning: No joints found for {self.name}")
                return
            
            # Get current positions
            all_positions = plant.GetPositions(context)
            positioned = False
            
            # Try to find prismatic joints for XYZ positioning
            for joint_idx in joint_indices:
                joint = plant.get_joint(joint_idx)
                
                # Check if this is a prismatic joint
                try:
                    if hasattr(joint, 'translation_axis'):
                        axis = joint.translation_axis()
                        pos_idx = joint.position_start()
                        
                        # Match axis to position component
                        if np.allclose(axis, [1, 0, 0]):  # X-axis
                            all_positions[pos_idx] = position[0]
                            positioned = True
                        elif np.allclose(axis, [0, 1, 0]):  # Y-axis
                            all_positions[pos_idx] = position[1]
                            positioned = True
                        elif np.allclose(axis, [0, 0, 1]):  # Z-axis
                            all_positions[pos_idx] = position[2]
                            positioned = True
                except:
                    pass
            
            if positioned:
                plant.SetPositions(context, all_positions)
                print(f"✓ Robot '{self.name}' positioned at {tuple(position)}")
            else:
                print(f"Info: No prismatic joints found for {self.name} - keeping default position")
            
        except Exception as e:
            print(f"Info: Position setting for {self.name}: {e}")

    
    def set_joint_properties(self, plant: MultibodyPlant):
        """Set joint properties (damping, friction) after plant is finalized."""
        for joint_idx, damping, stiffness, friction in zip(
            plant.GetJointIndices(self.model_instance),
            self.params.joint_damping,
            self.params.joint_stiffness,
            self.params.joint_friction
        ):
            joint = plant.get_mutable_joint(joint_idx)
            
            # Set damping (viscous friction)
            if hasattr(joint, "set_damping_vector"):
                joint.set_damping_vector(np.array([damping]))
            
            # Set friction (Coulomb friction) if supported
            if hasattr(joint, "set_coulomb_friction"):
                try:
                    joint.set_coulomb_friction(CoulombFriction(static_friction=friction, dynamic_friction=friction))
                except:
                    pass  # Not all joints support friction setting
    
    @abstractmethod
    def get_end_effector_position(self, plant: MultibodyPlant, context) -> np.ndarray:
        """Get end-effector position. Override in subclasses."""
        pass


# ============================================================================
# CART-PENDULUM CLASS
# ============================================================================

class CartPendulum(RobotBase):
    """
    Cart-Pendulum system for Drake.
    
    Manages:
    - Cart on rails with prismatic joint
    - Pendulum hanging from cart
    """
    
    def __init__(self, params: RobotParams):
        super().__init__(params, name="cart_pendulum")
    
    def get_cart_position(self, plant: MultibodyPlant, context) -> float:
        """Get cart's X position."""
        # Get cart joint (usually first joint)
        joint_indices = plant.GetJointIndices(self.model_instance)
        if len(joint_indices) > 0:
            cart_joint = plant.get_joint(joint_indices[0])
            # Get position from context
            try:
                # This is simplified - actual implementation depends on URDF structure
                positions = plant.GetPositionsByName(context, self.model_instance)
                return positions[0] if len(positions) > 0 else 0.0
            except:
                return 0.0
        return 0.0
    
    def get_end_effector_position(self, plant: MultibodyPlant, context) -> np.ndarray:
        """Get cart position as 'end effector' for consistency."""
        cart_x = self.get_cart_position(plant, context)
        return np.array([cart_x, 0.0, 1.325])  # Cart height = 1.325m


# ============================================================================
# PLANAR MANIPULATOR CLASS
# ============================================================================

class PlanarManipulator(RobotBase):
    """
    2-DOF Planar Manipulator for Drake.
    
    Manages:
    - Two revolute joints for planar motion
    - End-effector frame computation
    - Inverse kinematics
    """
    
    def __init__(self, params: RobotParams):
        super().__init__(params, name="planar_manipulator")
        self.link_lengths = params.link_lengths or [1.0, 1.0]  # Default link lengths
    
    def forward_kinematics(self, theta1: float, theta2: float) -> Tuple[float, float]:
        """
        Compute end-effector position from joint angles.
        
        Args:
            theta1: First joint angle (radians)
            theta2: Second joint angle (radians)
            
        Returns:
            (x, y): End-effector position
        """
        L1, L2 = self.link_lengths[0], self.link_lengths[1]
        
        # Forward kinematics for 2-DOF planar arm
        x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
        y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
        
        return x, y
    
    def transform_point_world_to_base(self, world_x: float, world_y: float, world_z: float) -> Optional[Tuple[float, float, float]]:
        """
        Transform a point from world coordinates to base-relative coordinates.
        
        For PyDrake, we use the known base position (x=-3.0, y=0.0, z=0.5).
        
        Args:
            world_x: X coordinate in world frame
            world_y: Y coordinate in world frame
            world_z: Z coordinate in world frame
            
        Returns:
            tuple: (x_base, y_base, z_base) in base frame, or None if transformation fails
        """
        # Base position in world frame (from URDF placement)
        base_x = -3.0
        base_y = 0.0
        base_z = 0.5
        
        # Compute relative coordinates
        x_rel = world_x - base_x
        y_rel = world_y - base_y
        z_rel = world_z - base_z
        
        return (x_rel, y_rel, z_rel)
    
    def inverse_kinematics(self, target_x: float, target_y: float, target_z: float = 0.0) -> Optional[Tuple[float, float]]:
        """
        Compute inverse kinematics for 2-DOF planar arm using analytical solution.
        
        ALGORITHM (Analytical Closed-Form Solution):
        1. Transform world coordinates to base-relative coordinates
        2. Compute planar distance: r = sqrt(x² + y²) for XY plane OR sqrt(x² + z²) for XZ plane
        3. Check reachability: |L1 - L2| ≤ r ≤ L1 + L2
        4. Use Law of Cosines to find θ2:
           cos(θ2) = (r² - L1² - L2²) / (2*L1*L2)
        5. Compute θ1 using geometry:
           θ1 = atan2(y, x) - atan2(L2*sin(θ2), L1 + L2*cos(θ2))
        
        CONFIGURATION:
        This implementation uses "elbow-down" configuration (θ2 < 0)
        Alternative: "elbow-up" would use θ2 = +arccos(...)
        
        Args:
            target_x: Target X position in world frame (meters)
            target_y: Target Y position in world frame (meters)
            target_z: Target Z position in world frame (meters)
            
        Returns:
            tuple: (theta1, theta2) in radians, or None if unreachable
        
        EDUCATIONAL NOTES:
        - Closed-form solution: Fast and exact (no iteration needed)
        - Multiple solutions possible (elbow-up vs elbow-down)
        - Singularities occur when arm is fully extended or folded
        """
        # Get link lengths
        L1, L2 = self.link_lengths[0], self.link_lengths[1]
        
        # Convert world coordinates to base-relative coordinates
        base_coords = self.transform_point_world_to_base(target_x, target_y, target_z)
        if base_coords is None:
            return None
        
        x_rel, y_rel, z_rel = base_coords
        
        # For planar manipulator in XZ plane (Drake's configuration)
        # Compute planar distance from base
        r = np.sqrt(x_rel**2 + z_rel**2)
        
        # Check if target is reachable
        if r > (L1 + L2) or r < abs(L1 - L2):
            return None
        
        # Angle to target in XZ plane
        phi = np.arctan2(z_rel, x_rel)
        
        # Law of cosines to find theta2
        cos_theta2 = (r**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)  # Numerical safety
        
        # Elbow-down configuration (negative theta2)
        theta2 = -np.arccos(cos_theta2)
        
        # Find theta1 using geometry
        alpha = np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
        theta1 = phi - alpha
        
        return (theta1, theta2)
    
    def get_ee_world_position(self, plant: MultibodyPlant, context) -> Optional[Tuple[float, float, float]]:
        """
        Get end-effector world position from current joint states.
        
        Matches Isaac Sim's API: self.manipulator.get_ee_world_position()
        
        Args:
            plant: MultibodyPlant instance
            context: Current simulation context
            
        Returns:
            tuple: (x, y, z) in world frame, or None if computation fails
        """
        try:
            # Get current joint positions from plant
            positions = plant.GetPositions(context)
            
            # Extract manipulator joint angles
            joint_indices = plant.GetJointIndices(self.model_instance)
            joint_positions = []
            for joint_idx in joint_indices:
                joint = plant.get_joint(joint_idx)
                if joint.num_velocities() == 1:
                    joint_positions.append(positions[joint.position_start()])
            
            if len(joint_positions) < 2:
                return None
            
            theta1 = joint_positions[0]
            theta2 = joint_positions[1]
            
            # Compute EE position in base frame using forward kinematics
            L1, L2 = self.link_lengths[0], self.link_lengths[1]
            ee_x_local = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
            ee_z_local = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
            
            # Transform to world frame (base at x=-3.0, y=0.0, z=0.5)
            ee_x_world = -3.0 + ee_x_local
            ee_y_world = 0.0
            ee_z_world = 0.5 + ee_z_local
            
            return (ee_x_world, ee_y_world, ee_z_world)
        
        except Exception as e:
            return None
    
    def set_joint_positions(self, plant: MultibodyPlant, context, positions: list) -> bool:
        """
        Set manipulator joint positions.
        
        Helper method to match Isaac Sim's API: self.manipulator.set_joint_positions([theta1, theta2])
        In PyDrake, we need to manually get all positions, modify the manipulator joints, and set back.
        
        Args:
            plant: MultibodyPlant instance
            context: Current simulation context
            positions: List of joint positions [theta1, theta2] in radians
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get all positions
            all_positions = plant.GetPositions(context)
            
            # Get manipulator joint indices
            joint_indices = plant.GetJointIndices(self.model_instance)
            
            # Set manipulator joint positions
            joint_count = 0
            for joint_idx in joint_indices:
                joint = plant.get_joint(joint_idx)
                if joint.num_velocities() == 1:
                    if joint_count < len(positions):
                        all_positions[joint.position_start()] = positions[joint_count]
                        joint_count += 1
            
            # Update all positions
            plant.SetPositions(context, all_positions)
            return True
        
        except Exception as e:
            return False
    
    def get_joint_positions(self, plant: MultibodyPlant, context) -> Optional[list]:
        """
        Get current manipulator joint positions.
        
        Helper method to match Isaac Sim's API: positions = self.manipulator.get_joint_positions()
        
        Args:
            plant: MultibodyPlant instance
            context: Current simulation context
            
        Returns:
            list: Joint positions [theta1, theta2], or None if error
        """
        try:
            all_positions = plant.GetPositions(context)
            joint_indices = plant.GetJointIndices(self.model_instance)
            
            positions = []
            for joint_idx in joint_indices:
                joint = plant.get_joint(joint_idx)
                if joint.num_velocities() == 1:
                    positions.append(all_positions[joint.position_start()])
            
            return positions if positions else None
        
        except Exception as e:
            return None
    
    def inverse_kinematics_old(self, target_x: float, target_y: float) -> Optional[Tuple[float, float]]:
        """
        OLD METHOD: Compute joint angles for target position using analytical IK.
        (Kept for reference - uses 2D coordinates in local frame)
        
        Args:
            target_x: Target X position
            target_y: Target Y position
            
        Returns:
            (theta1, theta2): Joint angles, or None if unreachable
        """
        L1, L2 = self.link_lengths[0], self.link_lengths[1]
        
        # Compute distance to target
        d = np.sqrt(target_x**2 + target_y**2)
        
        # Check reachability
        if d > L1 + L2 or d < abs(L1 - L2):
            return None  # Unreachable
        
        # Use law of cosines to find joint angles
        cos_theta2 = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)  # Clamp to valid range
        
        theta2 = np.arccos(cos_theta2)
        
        # theta1 from geometry
        k1 = L1 + L2 * np.cos(theta2)
        k2 = L2 * np.sin(theta2)
        theta1 = np.arctan2(target_y, target_x) - np.arctan2(k2, k1)
        
        return theta1, theta2
    
    def get_end_effector_position(self, plant: MultibodyPlant, context) -> np.ndarray:
        """Get end-effector position from current joint angles."""
        try:
            positions = plant.GetPositionsByName(context, self.model_instance)
            if len(positions) >= 2:
                x, y = self.forward_kinematics(positions[0], positions[1])
                return np.array([x, y, 1.325])  # Add height
        except:
            pass
        
        return np.array([0.0, 0.0, 1.325])


# ============================================================================
# SCENE MANAGER CLASS
# ============================================================================

class DrakeSceneManager:
    """
    Scene Manager for Drake simulation.
    
    RESPONSIBILITIES:
    1. Setup: Create MultibodyPlant, add robots
    2. Initialization: Finalize plant, create simulator
    3. Execution: Run different simulation modes
    4. Visualization: Set up Meshcat visualization
    5. Data logging: Record and plot simulation results
    """
    
    def __init__(
        self,
        cart_pendulum_params: RobotParams,
        manipulator_params: RobotParams,
    ):
        """Initialize scene manager."""
        self.cart_pendulum = CartPendulum(cart_pendulum_params)
        self.manipulator = PlanarManipulator(manipulator_params)
        
        # Data logging
        self.data_log = {
            'time': [],
            'manip_joint1_pos': [],
            'manip_joint2_pos': [],
            'manip_joint1_vel': [],
            'manip_joint2_vel': [],
            'cart_pos': [],
            'cart_vel': [],
            'pendulum_pos': [],
            'pendulum_vel': [],
        }
        
        # Drake objects
        self.builder = None
        self.plant = None
        self.scene_graph = None
        self.meshcat = None
        self.simulator = None
        self.visualizer = None
        self.diagram = None
        
        # Simulation control state
        self.is_paused = False
        self.should_reset = False
        self.should_exit = False
        self._interactive_mode = False
    
    def setup_drake_system(self):
        """Create Drake diagram with multibody plant."""
        print(f"\n{'='*70}")
        print("SETTING UP DRAKE SIMULATION")
        print(f"{'='*70}\n")
        
        # Create diagram builder
        self.builder = DiagramBuilder()
        
        # Create MultibodyPlant with scene graph
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder,
            time_step=SIMULATOR_TIME_STEP
        )
        
        # Create URDF parser
        parser = Parser(self.plant)
        
        # Load robots
        print("Loading robots...")
        self.cart_pendulum_instance = self.cart_pendulum.load_urdf_to_plant(self.plant, parser)
        self.manipulator_instance = self.manipulator.load_urdf_to_plant(self.plant, parser)
        
        # No coupling joint needed - manipulator will physically push cart through contact
        print("\nNo coupling joint (manipulator will push cart through contact)")
        
        # Set cart and pendulum default positions before finalization
        print("\nSetting cart-pendulum default positions before finalization...")
        cart_joint_indices = self.plant.GetJointIndices(self.cart_pendulum_instance)
        for joint_idx in cart_joint_indices:
            joint = self.plant.get_joint(joint_idx)
            if joint.num_positions() == 1:
                # Set cart slider joint
                if "cart" in joint.name().lower() and "slider" in joint.name().lower():
                    joint.set_default_positions([CART_PENDULUM_COUPLED_MOTION[0]])
                    print(f"✓ Set default position for {joint.name()}: {CART_PENDULUM_COUPLED_MOTION[0]}m")
                # Set pendulum joint
                elif "pendulum" in joint.name().lower():
                    joint.set_default_positions([CART_PENDULUM_COUPLED_MOTION[1]])
                    print(f"✓ Set default position for {joint.name()}: {CART_PENDULUM_COUPLED_MOTION[1]} rad")
        
        # Set manipulator joint positions to align end-effector with cart
        # Target: EE at position specified in EE_MANIPULATOR_INITIAL_POSITION
        print("\nSetting manipulator default positions to align with cart...")
        target_ee_world = tuple(EE_MANIPULATOR_INITIAL_POSITION)
        joint_angles = self.manipulator.inverse_kinematics(*target_ee_world)
        if joint_angles is not None:
            theta1, theta2 = joint_angles
            manip_joint_indices = self.plant.GetJointIndices(self.manipulator_instance)
            joint_count = 0
            for joint_idx in manip_joint_indices:
                joint = self.plant.get_joint(joint_idx)
                if joint.num_positions() == 1:
                    if joint_count == 0:
                        joint.set_default_positions([theta1])
                        print(f"✓ Set default position for {joint.name()}: {np.degrees(theta1):.2f}°")
                    elif joint_count == 1:
                        joint.set_default_positions([theta2])
                        print(f"✓ Set default position for {joint.name()}: {np.degrees(theta2):.2f}°")
                    joint_count += 1
            print(f"✓ Manipulator EE positioned at world: {target_ee_world}")
        else:
            print("⚠ Warning: Could not compute IK for target position")
        
        # Finalize the plant (no more robots can be added after this)
        self.plant.Finalize()
        print("✓ Plant finalized")
        
        # Debug: Print plant info
        print(f"\nPlant Summary:")
        print(f"  Total bodies: {self.plant.num_bodies()}")
        print(f"  Total joints: {self.plant.num_joints()}")
        print(f"  Total positions: {self.plant.num_positions()}")
        print(f"  Total velocities: {self.plant.num_velocities()}")
        
        # Initialize robot states
        self.cart_pendulum.initialize_state(self.plant, self.cart_pendulum_instance)
        self.manipulator.initialize_state(self.plant, self.manipulator_instance)
        
        # Set joint properties
        self.cart_pendulum.set_joint_properties(self.plant)
        self.manipulator.set_joint_properties(self.plant)
        
        # Add visualization if enabled
        if VISUALIZE:
            print("\nSetting up Meshcat visualization...")
            self.meshcat = StartMeshcat()
            visualizer = MeshcatVisualizer.AddToBuilder(
                self.builder,
                self.scene_graph,
                self.meshcat
            )
            self.visualizer = visualizer
            print(f"✓ Meshcat running at http://127.0.0.1:7000")
        
        # Build the diagram
        self.diagram = self.builder.Build()
        print("✓ Diagram built successfully\n")
        
        # Print geometry information for debugging
        try:
            print("Geometry in SceneGraph:")
            inspector = self.scene_graph.model_inspector()
            frame_ids = inspector.GetAllFrameIds()
            for frame_id in frame_ids:
                frame_name = inspector.GetOwnerModelInstanceName(frame_id)
                geometry_ids = inspector.GetGeometries(frame_id)
                if geometry_ids:
                    print(f"  Frame {frame_name}: {len(geometry_ids)} geometries")
            print()
        except:
            pass  # Silently skip if inspector not available
    
    def create_ee_cart_coupling_joint(self, cart_pendulum_instance, manipulator_instance):
        """
        Create a weld joint between manipulator EE and cart.
        
        MUST be called BEFORE plant.Finalize()
        
        Args:
            cart_pendulum_instance: Drake model instance ID for cart-pendulum
            manipulator_instance: Drake model instance ID for manipulator
        """
        # try:
        # Find the manipulator's end-effector frame
        # Typically the last link in the manipulator
        manip_body_indices = self.plant.GetBodyIndices(manipulator_instance)
        if len(manip_body_indices) > 0:
            # Get the last body (end-effector)
            ee_body_id = manip_body_indices[-1]
            ee_body = self.plant.get_body(ee_body_id)
            ee_frame = ee_body.body_frame()
            print(f"✓ Found EE frame: {ee_body.name()}")
        else:
            print("WARNING: Could not find manipulator end-effector frame")
            return
        
        # Find the cart body frame
        cart_body_indices = self.plant.GetBodyIndices(cart_pendulum_instance)
        if len(cart_body_indices) > 0:
            # Get first body (cart, not pendulum)
            cart_body_id = cart_body_indices[0]
            cart_body = self.plant.get_body(cart_body_id)
            cart_frame = cart_body.body_frame()
            print(f"✓ Found Cart frame: {cart_body.name()}")
        else:
            print("WARNING: Could not find cart body frame")
            return
        
        # Create coupling joint between EE and cart based on configuration
        if COUPLING_JOINT_TYPE == "fixed":
            # Weld joint - rigid constraint
            self.plant.WeldFrames(
                frame_on_parent_F=ee_frame,
                frame_on_child_M=cart_frame,
                X_FM=RigidTransform()  # Identity transform (they start aligned)
            )
            print(f"✓ Coupling weld (fixed) created")
            
        elif COUPLING_JOINT_TYPE == "revolute":
            # Revolute joint - allows rotation around Y-axis (out of XZ plane)
            revolute_joint = self.plant.AddJoint(
                RevoluteJoint(
                    name="ee_cart_coupling",
                    frame_on_parent=ee_frame,
                    frame_on_child=cart_frame,
                    axis=[0, 1, 0],  # Y-axis rotation
                    damping=EE_CART_COUPLING_JOINT_DAMPING
                )
            )
            print(f"✓ Coupling revolute joint created: {revolute_joint.name()}")
            print(f"  Axis: Y (allows rotation in XZ plane)")
            
        elif COUPLING_JOINT_TYPE == "prismatic":
            # Prismatic joint - allows sliding along Z-axis
            prismatic_joint = self.plant.AddJoint(
                PrismaticJoint(
                    name="ee_cart_coupling",
                    frame_on_parent=ee_frame,
                    frame_on_child=cart_frame,
                    axis=[0, 0, 1],  # Z-axis translation
                    damping=EE_CART_COUPLING_JOINT_DAMPING
                )
            )
            print(f"✓ Coupling prismatic joint created: {prismatic_joint.name()}")
            print(f"  Axis: Z (allows sliding vertically)")
        
        print(f"  Constraint: EE ({ee_body.name()}) <-> Cart ({cart_body.name()})")
            
        # except Exception as e:
        #     print(colored(f"ERROR creating coupling joint: {e}", "red"))
        #     import traceback
        #     traceback.print_exc()
    
    def create_simulator(self):
        """Create Drake simulator."""
        self.simulator = Simulator(self.diagram)
        self.simulator.set_target_realtime_rate(REALTIME_RATE)
        rate_desc = "real-time" if REALTIME_RATE == 1.0 else f"{REALTIME_RATE}x speed" if REALTIME_RATE > 0 else "maximum"
        print(f"✓ Simulator created (running at {rate_desc} for visualization)")
        
        # Setup interactive controls if in interactive mode
        if self._interactive_mode:
            self.setup_control_buttons()
    
    def setup_control_buttons(self):
        """Setup interactive control buttons in Meshcat interface."""
        if not self.meshcat:
            return
        
        try:
            print("\n" + "="*70)
            print("INTERACTIVE CONTROLS SETUP")
            print("="*70)
            
            # Add control panel folder
            folder = self.meshcat["__controls"]
            
            # Add button-like sliders (0 = off, 1 = on/pressed)
            from meshcat.geometry import LineBasicMaterial, LineSegments, BufferGeometry
            import meshcat.geometry as g
            import meshcat.transformations as tf
            
            # Create simple button controls using Meshcat's slider interface
            folder["Play_Button"] = {"value": 0}
            folder["Pause_Button"] = {"value": 0}
            folder["Repeat_Button"] = {"value": 0}
            folder["Reset_Button"] = {"value": 0}
            folder["Status"] = {"value": "Ready - Press Play to Start"}
            
            print("✓ Play Button added (toggle to start/resume)")
            print("✓ Pause Button added (toggle to pause)")
            print("✓ Repeat Button added (toggle to restart)")
            print("✓ Reset Button added (toggle to reset simulation)")
            print("✓ Status Display added")
            print("\nOpen Meshcat interface at http://127.0.0.1:7000")
            print("Go to __controls folder to access buttons")
            print("="*70 + "\n")
            
        except Exception as e:
            print(colored(f"Warning: Could not setup control buttons - {e}", "yellow"))
    
    def check_control_buttons(self) -> dict:
        """Check control button states and return control flags."""
        controls = {
            'play': False,
            'pause': False,
            'repeat': False,
            'reset': False,
        }
        
        if not self.meshcat:
            return controls
        
        try:
            folder = self.meshcat["__controls"]
            controls['play'] = bool(folder["Play_Button"].value) if "Play_Button" in folder else False
            controls['pause'] = bool(folder["Pause_Button"].value) if "Pause_Button" in folder else False
            controls['repeat'] = bool(folder["Repeat_Button"].value) if "Repeat_Button" in folder else False
            controls['reset'] = bool(folder["Reset_Button"].value) if "Reset_Button" in folder else False
        except:
            pass
        
        return controls
    
    def update_status_display(self, status_text: str):
        """Update the status display in Meshcat."""
        if not self.meshcat:
            return
        
        try:
            folder = self.meshcat["__controls"]
            folder["Status"].value = status_text
        except:
            pass
    
    def set_initial_conditions(self):
        """Set initial joint positions and robot poses."""
        context = self.simulator.get_mutable_context()
        plant_context = self.diagram.GetMutableSubsystemContext(self.plant, context)
        
        # Set robot poses (world positions)
        self.cart_pendulum.set_pose(self.plant, plant_context)
        self.manipulator.set_pose(self.plant, plant_context)
        
        # Get total number of positions
        num_positions = self.plant.num_positions()
        all_positions = np.zeros(num_positions)
        
        # Get default positions first
        all_positions = self.plant.GetPositions(plant_context)
        
        # Set manipulator initial positions using IK to align with cart
        target_ee_world = tuple(EE_MANIPULATOR_INITIAL_POSITION)
        joint_angles = self.manipulator.inverse_kinematics(*target_ee_world)
        if joint_angles is not None:
            manip_indices = self.plant.GetJointIndices(self.manipulator.model_instance)
            manip_count = 0
            for joint_idx in manip_indices:
                joint = self.plant.get_joint(joint_idx)
                if joint.num_velocities() == 1 and manip_count < 2:
                    all_positions[joint.position_start()] = joint_angles[manip_count]
                    manip_count += 1
        
        # Set cart-pendulum initial positions (use coupled motion values)
        cart_indices = self.plant.GetJointIndices(self.cart_pendulum.model_instance)
        cart_count = 0
        for joint_idx in cart_indices:
            joint = self.plant.get_joint(joint_idx)
            if joint.num_velocities() == 1 and cart_count < len(CART_PENDULUM_COUPLED_MOTION):
                all_positions[joint.position_start()] = CART_PENDULUM_COUPLED_MOTION[cart_count]
                cart_count += 1
        
        # Set all positions
        self.plant.SetPositions(plant_context, all_positions)
        
        print("✓ Initial conditions set")
    
    def log_data(self, time_s: float):
        """Log simulation data."""
        context = self.simulator.get_context()
        plant_context = self.diagram.GetSubsystemContext(self.plant, context)
        
        self.data_log['time'].append(time_s)
        
        try:
            # Get all positions and velocities
            all_positions = self.plant.GetPositions(plant_context)
            all_velocities = self.plant.GetVelocities(plant_context)
            
            # Extract manipulator data
            manip_indices = self.plant.GetJointIndices(self.manipulator.model_instance)
            manip_pos_data = []
            manip_vel_data = []
            for joint_idx in manip_indices:
                joint = self.plant.get_joint(joint_idx)
                if joint.num_velocities() == 1:
                    manip_pos_data.append(all_positions[joint.position_start()])
                    manip_vel_data.append(all_velocities[joint.velocity_start()])
            
            if len(manip_pos_data) >= 2:
                self.data_log['manip_joint1_pos'].append(float(manip_pos_data[0]))
                self.data_log['manip_joint2_pos'].append(float(manip_pos_data[1]))
                self.data_log['manip_joint1_vel'].append(float(manip_vel_data[0]))
                self.data_log['manip_joint2_vel'].append(float(manip_vel_data[1]))
            else:
                self.data_log['manip_joint1_pos'].append(0.0)
                self.data_log['manip_joint2_pos'].append(0.0)
                self.data_log['manip_joint1_vel'].append(0.0)
                self.data_log['manip_joint2_vel'].append(0.0)
            
            # Extract cart-pendulum data
            cart_indices = self.plant.GetJointIndices(self.cart_pendulum.model_instance)
            cart_pos_data = []
            cart_vel_data = []
            for joint_idx in cart_indices:
                joint = self.plant.get_joint(joint_idx)
                if joint.num_velocities() == 1:
                    cart_pos_data.append(all_positions[joint.position_start()])
                    cart_vel_data.append(all_velocities[joint.velocity_start()])
            
            if len(cart_pos_data) >= 1:
                self.data_log['cart_pos'].append(float(cart_pos_data[0]))
                self.data_log['cart_vel'].append(float(cart_vel_data[0]))
            else:
                self.data_log['cart_pos'].append(0.0)
                self.data_log['cart_vel'].append(0.0)
            
            if len(cart_pos_data) >= 2:
                self.data_log['pendulum_pos'].append(float(cart_pos_data[1]))
                self.data_log['pendulum_vel'].append(float(cart_vel_data[1]))
            else:
                self.data_log['pendulum_pos'].append(0.0)
                self.data_log['pendulum_vel'].append(0.0)
        
        except Exception as e:
            print(colored(f"Warning: Could not log data - {e}", "yellow"))
    
    def plot_results(self):
        """Plot simulation results."""
        if len(self.data_log['time']) == 0:
            print("No data to plot")
            return
        
        print(f"\n{'='*70}")
        print("PLOTTING RESULTS")
        print(f"{'='*70}\n")
        
        time_array = np.array(self.data_log['time'])
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Cart-Pendulum with 2-DOF Manipulator - Drake Simulation Results', fontsize=16)
        
        # Manipulator Joint Positions
        axes[0, 0].plot(time_array, np.degrees(self.data_log['manip_joint1_pos']), 'b-', linewidth=2, label='Joint 1')
        axes[0, 0].plot(time_array, np.degrees(self.data_log['manip_joint2_pos']), 'r-', linewidth=2, label='Joint 2')
        axes[0, 0].set_ylabel('Joint Angle (degrees)')
        axes[0, 0].set_title('Manipulator Joint Positions')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Manipulator Joint Velocities
        axes[0, 1].plot(time_array, self.data_log['manip_joint1_vel'], 'b-', linewidth=2, label='Joint 1')
        axes[0, 1].plot(time_array, self.data_log['manip_joint2_vel'], 'r-', linewidth=2, label='Joint 2')
        axes[0, 1].set_ylabel('Joint Velocity (rad/s)')
        axes[0, 1].set_title('Manipulator Joint Velocities')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Cart Position and Velocity
        axes[1, 0].plot(time_array, self.data_log['cart_pos'], 'g-', linewidth=2)
        axes[1, 0].set_ylabel('Position (m)')
        axes[1, 0].set_title('Cart Position')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(time_array, self.data_log['cart_vel'], 'g-', linewidth=2)
        axes[1, 1].set_ylabel('Velocity (m/s)')
        axes[1, 1].set_title('Cart Velocity')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Pendulum Angle and Angular Velocity
        axes[2, 0].plot(time_array, np.degrees(self.data_log['pendulum_pos']), 'c-', linewidth=2)
        axes[2, 0].set_ylabel('Angle (degrees)')
        axes[2, 0].set_title('Pendulum Angle')
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(time_array, self.data_log['pendulum_vel'], 'c-', linewidth=2)
        axes[2, 1].set_ylabel('Angular Velocity (rad/s)')
        axes[2, 1].set_title('Pendulum Angular Velocity')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Add X labels
        for ax in axes[2, :]:
            ax.set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        # Save and show
        output_path = "plots/drake_simulation_results.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {output_path}")
        
        plt.show()
    
    def save_configuration_to_json(self, output_path: str = None):
        """
        Automatically save all simulation configuration to JSON file.
        
        Uses dataclasses.asdict() to automatically convert configuration dataclasses
        to dictionaries, making the process fully automatic and maintainable.
        
        Args:
            output_path: Optional custom path. If None, auto-generates filename with timestamp.
        
        Returns:
            str: Path to saved JSON file
        """
        if output_path is None:
            os.makedirs("configs", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"configs/simulation_config_pydrake_{timestamp}.json"
        
        # Automatically convert dataclasses to dicts using asdict()
        config = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "simulation_mode": SIMULATION_MODE,
                "visualize": VISUALIZE,
                "framework": "PyDrake",
            },
            "simulation_parameters": {
                "time_step": SIMULATOR_TIME_STEP,
                "duration": SIMULATION_DURATION,
                "settling_time": PENDULUM_SETTLING_TIME,
            },
            "cart_pendulum_config": asdict(self.cart_pendulum.params),
            "manipulator_config": asdict(self.manipulator.params),
            "coupling_joint_config": {
                "type": COUPLING_JOINT_TYPE,
                "stiffness": EE_CART_COUPLING_JOINT_STIFFNESS,
                "damping": EE_CART_COUPLING_JOINT_DAMPING,
            }
        }
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write to JSON file with nice formatting
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✓ Configuration saved to: {output_path}")
        return output_path
    
    def run_scene_viz(self):
        """Run static scene visualization."""
        print(f"\n{'='*70}")
        print("SCENE VISUALIZATION MODE (Static)")
        print(f"{'='*70}\n")
        
        # Save configuration for this run
        self.save_configuration_to_json()
        
        self.setup_drake_system()
        self.create_simulator()
        self.set_initial_conditions()
        
        # Publish the initial scene to Meshcat
        context = self.simulator.get_context()
        self.diagram.ForcedPublish(context)
        
        print("Scene loaded. Use Meshcat interface to view (http://127.0.0.1:7000)")
        print("Interacting with scene...")
        print("  - Left-click + drag: Rotate")
        print("  - Right-click + drag: Pan")
        print("  - Scroll: Zoom")
        print("Press Ctrl+C to exit.\n")
        
        # Run for a few steps to ensure geometry is rendered and animated
        print("Running simulation to animate scene...")
        for i in range(10):
            self.simulator.AdvanceTo((i + 1) * 0.1)
            self.diagram.ForcedPublish(self.simulator.get_context())
        
        print("✓ Scene published to Meshcat")
        print("\nKeeping visualization running...")
        try:
            import time
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nExiting visualization...")
    
    def run_simulation(self):
        """Run physics simulation without manipulator control."""
        print(f"\n{'='*70}")
        print("PHYSICS SIMULATION MODE")
        print(f"{'='*70}\n")
        
        # Save configuration for this run
        self.save_configuration_to_json()
        
        self.setup_drake_system()
        self.create_simulator()
        self.set_initial_conditions()
        
        # Publish initial scene to Meshcat
        context = self.simulator.get_context()
        self.diagram.ForcedPublish(context)
        
        # Print table header
        print(f"{'Time (s)':>10} | {'Joint1 (°)':>12} | {'Joint2 (°)':>12} | {'Cart X (m)':>12} | {'Pend (°)':>12}")
        print(f"{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
        print(f"Running simulation (visualizing in Meshcat at http://127.0.0.1:7000)...")
        print(f"Visualization: {'Every timestep' if VISUALIZATION_UPDATE_EVERY_STEP else f'{VISUALIZATION_FRAME_RATE} FPS'}\n")
        
        # Run simulation
        dt = SIMULATOR_TIME_STEP  # Use simulator timestep for fine control
        time_s = 0.0
        frame_count = 0
        print_interval_frames = max(1, int(PRINT_INTERVAL / SIMULATOR_TIME_STEP))
        
        while time_s < SIMULATION_DURATION:
            # Log data
            self.log_data(time_s)
            
            # Print progress at specified interval
            if frame_count % print_interval_frames == 0:
                context = self.simulator.get_context()
                plant_context = self.diagram.GetSubsystemContext(self.plant, context)
                
                all_positions = self.plant.GetPositions(plant_context)
                
                # Get manipulator joint indices
                manip_indices = self.plant.GetJointIndices(self.manipulator.model_instance)
                manip_pos = []
                for joint_idx in manip_indices:
                    joint = self.plant.get_joint(joint_idx)
                    if joint.num_velocities() == 1:
                        manip_pos.append(all_positions[joint.position_start()])
                
                # Get cart-pendulum joint indices
                cart_indices = self.plant.GetJointIndices(self.cart_pendulum.model_instance)
                cart_pos = []
                for joint_idx in cart_indices:
                    joint = self.plant.get_joint(joint_idx)
                    if joint.num_velocities() == 1:
                        cart_pos.append(all_positions[joint.position_start()])
                
                j1 = np.degrees(manip_pos[0]) if len(manip_pos) > 0 else 0.0
                j2 = np.degrees(manip_pos[1]) if len(manip_pos) > 1 else 0.0
                cx = cart_pos[0] if len(cart_pos) > 0 else 0.0
                pend = np.degrees(cart_pos[1]) if len(cart_pos) > 1 else 0.0
                
                print(f"{time_s:10.2f} | {j1:12.2f} | {j2:12.2f} | {cx:12.4f} | {pend:12.2f}")
            
            # Publish visualization update
            if VISUALIZATION_UPDATE_EVERY_STEP:
                # Update every simulation timestep for smooth animation
                context = self.simulator.get_context()
                self.diagram.ForcedPublish(context)
            
            # Advance simulator
            self.simulator.AdvanceTo(time_s + dt)
            time_s += dt
            frame_count += 1
        
        print(f"\n✓ Simulation complete")
        self.plot_results()
        self.save_configuration_to_json()
    
    def run_coupled_motion(self):
        """
        Run simulation with manipulator controlling cart via joint coupling.
        
        CONCEPT:
        This mode demonstrates mechanical coupling between two independent robots:
        1. Planar manipulator (2-DOF arm) - actuated (moves via joint commands)
        2. Cart-pendulum system - passive (follows manipulator motion)
        3. Weld constraint connects manipulator end-effector to cart
        
        PHYSICS:
        - Manipulator applies forces/torques through coupling constraint
        - Cart translates along rail (prismatic joint)
        - Pendulum swings due to cart acceleration (inertial forces)
        - System demonstrates multi-body dynamics with constraints
        
        IMPLEMENTATION:
        Since Drake requires all bodies/constraints to be defined before plant finalization,
        we use a kinematic coupling approach:
        1. Position cart at manipulator EE location
        2. Move manipulator with sinusoidal trajectory
        3. Compute EE position and set cart to follow
        4. Allow pendulum to swing naturally from cart motion
        """
        print(f"\n{'='*70}")
        print("COUPLED MOTION MODE: MANIPULATOR MOVES CART-PENDULUM")
        print(f"{'='*70}\n")
        
        # Save configuration for this run
        self.save_configuration_to_json()
        
        self.setup_drake_system()
        self.create_simulator()
        self.set_initial_conditions()  # Set joints to configured initial positions

        # CRITICAL: Publish initial scene to Meshcat so visualization shows up
        print("\n✓ Publishing initial scene to Meshcat...")
        context = self.simulator.get_context()
        self.diagram.ForcedPublish(context)
        
        # Get initial manipulator joint positions (now set to configured values)
        context = self.simulator.get_mutable_context()
        plant_context = self.diagram.GetMutableSubsystemContext(self.plant, context)
        all_positions = self.plant.GetPositions(plant_context)
        manip_indices = self.plant.GetJointIndices(self.manipulator.model_instance)
        initial_manip_pos = []
        for joint_idx in manip_indices:
            joint = self.plant.get_joint(joint_idx)
            if joint.num_velocities() == 1:
                initial_manip_pos.append(all_positions[joint.position_start()])
        
        theta1_init = initial_manip_pos[0] if len(initial_manip_pos) > 0 else 0.0
        theta2_init = initial_manip_pos[1] if len(initial_manip_pos) > 1 else 0.0
        
        # Compute initial EE position using forward kinematics
        L1, L2 = 1.0, 1.0  # Link lengths
        ee_x_in_base = L1 * np.cos(theta1_init) + L2 * np.cos(theta1_init + theta2_init)
        ee_x_world = -3.0 + ee_x_in_base  # Add manipulator base offset
        
        # Position cart so edge aligns with EE (cart center is 0.15m from edge)
        # User specified: cart position is -1.5644m when touching EE
        cart_length_x = 0.3  # meters
        cart_half_length = cart_length_x / 2.0
        cart_center_x = -1.5644  # Position when cart edge touches EE
        
        # Set initial cart position
        cart_indices = self.plant.GetJointIndices(self.cart_pendulum.model_instance)
        new_positions = all_positions.copy()
        cart_count = 0
        for joint_idx in cart_indices:
            joint = self.plant.get_joint(joint_idx)
            if joint.num_velocities() == 1:
                if cart_count == 0:  # Cart slider
                    new_positions[joint.position_start()] = cart_center_x
                elif cart_count == 1:  # Pendulum
                    new_positions[joint.position_start()] = 0.0  # Hanging down
                cart_count += 1
        
        self.plant.SetPositions(plant_context, new_positions)
        
        # CRITICAL: Publish initial scene to Meshcat so visualization shows up
        print("\n✓ Publishing initial scene to Meshcat...")
        context = self.simulator.get_context()
        self.diagram.ForcedPublish(context)
        print("✓ Initial scene visible at http://127.0.0.1:7000")
        
        print(f"\n{'='*70}")
        print("INITIAL SETUP (Physics-Based EE-Cart Coupling)")
        print(f"{'='*70}")
        print(f"Manipulator angles: theta1={np.degrees(theta1_init):.2f}°, theta2={np.degrees(theta2_init):.2f}°")
        print(f"EE position: X={ee_x_world:.4f}m")
        print(f"Cart positioned at: X={cart_center_x:.4f}m")
        print(f"Cart edge aligned with EE")
        print(f"{'='*70}\n")
        
        # Generate manipulator trajectory (straight line in workspace)
        print(f"\n{'='*70}")
        print("GENERATING MANIPULATOR TRAJECTORY")
        print(f"{'='*70}")
        
        # Get initial EE position for trajectory planning (matching Isaac Sim approach)
        context = self.simulator.get_context()
        plant_context = self.diagram.GetSubsystemContext(self.plant, context)
        
        # Get initial EE position using manipulator's method (matching Isaac Sim)
        initial_ee_pos = self.manipulator.get_ee_world_position(self.plant, plant_context)
        
        if initial_ee_pos is None:
            print("ERROR: Could not get initial EE position")
            return
        
        initial_ee_x, initial_ee_y, initial_ee_z = initial_ee_pos
        
        print(f"Initial EE position: X={initial_ee_x:.4f}, Y={initial_ee_y:.4f}, Z={initial_ee_z:.4f}")
        
        # Trajectory parameters (straight line along X) - matching Isaac Sim
        target_y = initial_ee_y  # Keep Y constant
        target_z = initial_ee_z  # Keep Z constant
        x_start = initial_ee_x
        x_range = 0.6  # Move ±0.3m along X (same as Isaac Sim)
        duration = 8.0  # 8 seconds (slower for better observation)
        
        print(f"✓ Trajectory parameters:")
        print(f"  Duration: {duration:.1f}s")
        print(f"  X range: [{x_start - x_range/2:.4f}, {x_start + x_range/2:.4f}] m")
        print(f"  Y constant: {target_y:.4f} m")
        print(f"  Z constant: {target_z:.4f} m")
        print(f"  Motion: Straight line along X-axis")
        print(f"{'='*70}\n")
        
        # Build waypoint trajectory with inverse kinematics
        print("Computing inverse kinematics for straight-line path...")
        waypoint_trajectory = []
        
        num_steps = int(duration / SIMULATOR_TIME_STEP)
        ik_failed_count = 0
        
        for i in range(num_steps):
            t = i / max(1, num_steps - 1)
            # Sinusoidal motion along X (straight line back and forth)
            target_x = x_start + x_range * np.sin(2 * np.pi * 0.5 * t)
            
            # Compute inverse kinematics using manipulator's method (matching Isaac Sim)
            joint_angles = self.manipulator.inverse_kinematics(target_x, target_y, target_z)
            
            if joint_angles is not None:
                theta1_sol, theta2_sol = joint_angles
                waypoint_trajectory.append((theta1_sol, theta2_sol))
            else:
                ik_failed_count += 1
        
        print(f"✓ Generated {len(waypoint_trajectory)} waypoints")
        if ik_failed_count > 0:
            print(f"  WARNING: IK failed for {ik_failed_count} waypoints - unreachable")
        
        # Print table header
        print(f"\n{'Time (s)':>10} | {'Joint1 (°)':>12} | {'Joint2 (°)':>12} | {'EE X (m)':>10} | {'EE Y (m)':>10} | {'EE Z (m)':>10} | {'Cart X (m)':>12} | {'Pend (°)':>12}")
        print(f"{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")
        print(f"Running simulation (physics-based EE-Cart coupling via weld joint)...")
        print(f"Visualizing in Meshcat at http://127.0.0.1:7000")
        print(f"Visualization: {'Every timestep' if VISUALIZATION_UPDATE_EVERY_STEP else f'{VISUALIZATION_FRAME_RATE} FPS'}\n")
        
        # Run simulation with physics-based EE-Cart coupling via weld joint
        dt = SIMULATOR_TIME_STEP  # Use simulator timestep
        time_s = 0.0
        frame_count = 0
        waypoint_idx = 0
        print_interval_frames = max(1, int(PRINT_INTERVAL / SIMULATOR_TIME_STEP))
        
        while time_s < SIMULATION_DURATION:
            context = self.simulator.get_mutable_context()
            plant_context = self.diagram.GetMutableSubsystemContext(self.plant, context)
            all_positions = self.plant.GetPositions(plant_context)
            
            # Get next waypoint from trajectory
            if waypoint_idx < len(waypoint_trajectory):
                theta1_target, theta2_target = waypoint_trajectory[waypoint_idx]
                waypoint_idx += 1
            else:
                # Hold final position during settling
                if waypoint_trajectory:
                    theta1_target, theta2_target = waypoint_trajectory[-1]
                else:
                    theta1_target = theta1_init
                    theta2_target = theta2_init
            
            # Use PD control with forces (allows contact dynamics)
            # Get current joint positions
            current_positions = []
            for joint_idx in manip_indices:
                joint = self.plant.get_joint(joint_idx)
                if joint.num_velocities() == 1:
                    current_positions.append(all_positions[joint.position_start()])
            
            # Get velocities
            all_velocities = self.plant.GetVelocities(plant_context)
            current_velocities = []
            for joint_idx in manip_indices:
                joint = self.plant.get_joint(joint_idx)
                if joint.num_velocities() == 1:
                    current_velocities.append(all_velocities[joint.velocity_start()])
            
            # Compute PD control torques
            kp = 10000.0  # Proportional gain
            kd = 100.0/2   # Derivative gain
            target_positions = [theta1_target, theta2_target]
            
            num_actuators = self.plant.num_actuators()
            actuation = np.zeros(num_actuators)
            
            # Apply torques to manipulator actuators
            actuator_indices = self.plant.GetJointActuatorIndices(self.manipulator_instance)
            for i, actuator_idx in enumerate(actuator_indices):
                if i < len(target_positions):
                    error = target_positions[i] - current_positions[i]
                    error_dot = 0.0 - current_velocities[i]
                    torque = kp * error + kd * error_dot
                    actuation[actuator_idx] = torque
            
            # Set actuation using the correct Drake API
            self.plant.get_actuation_input_port().FixValue(plant_context, actuation)
            
            # Log data
            self.log_data(time_s)
            
            # Print progress at specified interval
            if frame_count % print_interval_frames == 0:
                context = self.simulator.get_context()
                plant_context = self.diagram.GetSubsystemContext(self.plant, context)
                all_positions = self.plant.GetPositions(plant_context)
                
                # Get current manipulator positions
                manip_pos = []
                for joint_idx in manip_indices:
                    joint = self.plant.get_joint(joint_idx)
                    if joint.num_velocities() == 1:
                        manip_pos.append(all_positions[joint.position_start()])
                
                # Get current cart-pendulum positions
                cart_pos = []
                for joint_idx in cart_indices:
                    joint = self.plant.get_joint(joint_idx)
                    if joint.num_velocities() == 1:
                        cart_pos.append(all_positions[joint.position_start()])
                
                j1 = np.degrees(manip_pos[0]) if len(manip_pos) > 0 else 0.0
                j2 = np.degrees(manip_pos[1]) if len(manip_pos) > 1 else 0.0
                cx = cart_pos[0] if len(cart_pos) > 0 else 0.0
                pend = np.degrees(cart_pos[1]) if len(cart_pos) > 1 else 0.0
                
                # Get current end-effector position
                ee_pos = self.manipulator.get_ee_world_position(self.plant, plant_context)
                if ee_pos:
                    ee_x, ee_y, ee_z = ee_pos
                else:
                    ee_x, ee_y, ee_z = 0.0, 0.0, 0.0
                
                print(f"{time_s:10.2f} | {j1:12.2f} | {j2:12.2f} | {ee_x:10.4f} | {ee_y:10.4f} | {ee_z:10.4f} | {cx:12.4f} | {pend:12.2f}")
            
            # Publish visualization update
            if VISUALIZATION_UPDATE_EVERY_STEP:
                # Update every simulation timestep for smooth animation
                context = self.simulator.get_context()
                self.diagram.ForcedPublish(context)
            
            # Advance simulator
            self.simulator.AdvanceTo(time_s + dt)
            time_s += dt
            frame_count += 1
        
        print(f"\n{'='*70}")
        print("SIMULATION COMPLETE")
        print(f"{'='*70}\n")
        self.plot_results()
        self.save_configuration_to_json()
    
    def run_cart_toward_manipulator(self):
        """Run simulation where cart edge moves toward manipulator end-effector."""
        print(f"\n{'='*70}")
        print("CART EDGE MOVES TOWARD MANIPULATOR EE")
        print(f"{'='*70}\n")
        
        # Save configuration for this run
        self.save_configuration_to_json()
        
        self.setup_drake_system()
        self.create_simulator()
        self.set_initial_conditions()
        
        # Publish initial scene to Meshcat
        context = self.simulator.get_context()
        self.diagram.ForcedPublish(context)
        
        # Get initial state
        context = self.simulator.get_mutable_context()
        plant_context = self.diagram.GetMutableSubsystemContext(self.plant, context)
        
        # Initial manipulator and cart positions
        all_positions = self.plant.GetPositions(plant_context)
        
        # Get manipulator body positions
        manip_indices = self.plant.GetJointIndices(self.manipulator.model_instance)
        manip_pos = []
        for joint_idx in manip_indices:
            joint = self.plant.get_joint(joint_idx)
            if joint.num_velocities() == 1:
                manip_pos.append(all_positions[joint.position_start()])
        
        # Get cart body positions
        cart_indices = self.plant.GetJointIndices(self.cart_pendulum.model_instance)
        cart_pos = []
        for joint_idx in cart_indices:
            joint = self.plant.get_joint(joint_idx)
            if joint.num_velocities() == 1:
                cart_pos.append(all_positions[joint.position_start()])
        
        theta1_init = manip_pos[0] if len(manip_pos) > 0 else 0.0
        theta2_init = manip_pos[1] if len(manip_pos) > 1 else 0.0
        cart_x_init = cart_pos[0] if len(cart_pos) > 0 else 0.0
        
        # Get end-effector position (approximated from forward kinematics)
        # For 2-DOF manipulator: x = L1*cos(θ1) + L2*cos(θ1+θ2), z = 1.325 + L1*sin(θ1) + L2*sin(θ1+θ2)
        L1, L2 = 1.0, 1.0  # Link lengths
        ee_x_in_base = L1 * np.cos(theta1_init) + L2 * np.cos(theta1_init + theta2_init)
        ee_x_world = -3.0 + ee_x_in_base  # Add manipulator base offset
        
        # Cart dimensions
        cart_length_x = 0.3  # meters
        cart_half_length = cart_length_x / 2.0
        initial_cart_edge_x = cart_x_init - cart_half_length
        
        print(f"\n{'='*70}")
        print("INITIAL SETUP")
        print(f"{'='*70}")
        print(f"Manipulator initial angles: theta1={np.degrees(theta1_init):.2f}°, theta2={np.degrees(theta2_init):.2f}°")
        print(f"Cart initial position: X={cart_x_init:.4f}m")
        print(f"\nInitial cart center position: X={cart_x_init:.4f}m")
        print(f"Initial cart edge position:   X={initial_cart_edge_x:.4f}m")
        print(f"Initial EE position (X):      {ee_x_world:.4f}m")
        print(f"Initial distance (edge to EE): {abs(initial_cart_edge_x - ee_x_world):.4f}m")
        
        # Motion parameters (use global parameters)
        cart_speed = CART_SPEED  # meters per second
        dt = CART_MODE_TIME_STEP  # Time step
        cart_step = cart_speed * dt  # meters per step
        distance_threshold = CONVERGENCE_THRESHOLD  # Stop when within threshold
        target_ee_x = ee_x_world
        
        print(f"\n{'='*70}")
        print("MOVING CART EDGE TOWARD MANIPULATOR EE")
        print(f"Cart speed: {cart_speed:.2f} m/s")
        print(f"Stop threshold: {distance_threshold:.5f} m")
        print(f"Target EE X position: {target_ee_x:.4f} m")
        print(f"{'='*70}\n")
        
        # Simulation loop
        time_s = 0.0
        frame_count = 0
        converged = False
        
        print(f"{'Time (s)':>10} | {'Error (m)':>10} | {'Cart Edge X':>12} | {'Target EE X':>13} | {'Joint1 (°)':>11} | {'Joint2 (°)':>11}")
        print(f"{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*13}-+-{'-'*11}-+-{'-'*11}")
        print(f"Running simulation (visualizing in Meshcat at http://127.0.0.1:7000)...")
        print(f"Visualization: {'Every timestep' if VISUALIZATION_UPDATE_EVERY_STEP else f'{VISUALIZATION_FRAME_RATE} FPS'}\n")
        
        print_interval_frames = max(1, int(PRINT_INTERVAL / dt))  # Based on cart mode timestep
        
        while time_s < SIMULATION_DURATION and not converged:
            # Get current positions
            context = self.simulator.get_context()
            plant_context = self.diagram.GetSubsystemContext(self.plant, context)
            all_positions = self.plant.GetPositions(plant_context)
            
            # Get current cart position
            cart_indices = self.plant.GetJointIndices(self.cart_pendulum.model_instance)
            current_cart_pos = []
            for joint_idx in cart_indices:
                joint = self.plant.get_joint(joint_idx)
                if joint.num_velocities() == 1:
                    current_cart_pos.append(all_positions[joint.position_start()])
            
            current_cart_x = current_cart_pos[0] if len(current_cart_pos) > 0 else 0.0
            current_cart_edge_x = current_cart_x - cart_half_length
            
            # Compute distance
            distance = abs(current_cart_edge_x - target_ee_x)
            
            # Check if converged
            if distance < distance_threshold:
                converged = True
                print(f"\n{'='*70}")
                print("CONVERGENCE ACHIEVED!")
                print(f"{'='*70}")
                print(f"Final distance: {distance:.6f}m (threshold: {distance_threshold:.5f}m)")
                print(f"Cart center position: X={current_cart_x:.6f}m")
                print(f"Cart edge position:   X={current_cart_edge_x:.6f}m")
                print(f"EE position:          X={target_ee_x:.6f}m")
                print(f"{'='*70}\n")
                break
            
            # Move cart toward manipulator (with sign inversion bug from Isaac Sim version)
            if current_cart_edge_x > target_ee_x:
                new_cart_x = current_cart_x - cart_step  # This is the bug - should be minus
            else:
                new_cart_x = current_cart_x + cart_step  # This is the bug - should be plus
            
            # Update cart position via direct position setting
            new_positions = all_positions.copy()
            cart_count = 0
            for joint_idx in cart_indices:
                joint = self.plant.get_joint(joint_idx)
                if joint.num_velocities() == 1:
                    if cart_count == 0:
                        new_positions[joint.position_start()] = new_cart_x
                    cart_count += 1
            
            # Keep manipulator at initial position
            manip_indices = self.plant.GetJointIndices(self.manipulator.model_instance)
            manip_count = 0
            for joint_idx in manip_indices:
                joint = self.plant.get_joint(joint_idx)
                if joint.num_velocities() == 1:
                    if manip_count == 0:
                        new_positions[joint.position_start()] = theta1_init
                    elif manip_count == 1:
                        new_positions[joint.position_start()] = theta2_init
                    manip_count += 1
            
            self.plant.SetPositions(plant_context, new_positions)
            
            # Log data and print
            self.log_data(time_s)
            
            # Print progress at specified interval
            if frame_count % print_interval_frames == 0:
                manip_indices = self.plant.GetJointIndices(self.manipulator.model_instance)
                manip_pos = []
                for joint_idx in manip_indices:
                    joint = self.plant.get_joint(joint_idx)
                    if joint.num_velocities() == 1:
                        manip_pos.append(all_positions[joint.position_start()])
                
                j1 = np.degrees(manip_pos[0]) if len(manip_pos) > 0 else 0.0
                j2 = np.degrees(manip_pos[1]) if len(manip_pos) > 1 else 0.0
                
                print(f"{time_s:10.2f} | {distance:10.6f} | {current_cart_edge_x:12.6f} | {target_ee_x:13.6f} | {j1:11.2f} | {j2:11.2f}")
            
            # Publish visualization update
            if VISUALIZATION_UPDATE_EVERY_STEP:
                # Update every simulation timestep for smooth animation
                context = self.simulator.get_context()
                self.diagram.ForcedPublish(context)
            
            # Advance simulator
            self.simulator.AdvanceTo(time_s + dt)
            time_s += dt
            frame_count += 1
            
            # Safety limit
            if frame_count > 10000:
                print(colored("\nWARNING: Reached maximum iterations (10000 steps)", "yellow"))
                break
        
        print(f"\n✓ Simulation complete")
        self.plot_results()
        self.save_configuration_to_json()
    
    def run_simulation_interactive(self, mode_name: str, mode_function):
        """
        Run any simulation mode with interactive play/pause/repeat controls.
        
        Args:
            mode_name: Name of the simulation mode (for display)
            mode_function: Function reference to run (e.g., self.run_simulation)
        """
        print(f"\n{'='*70}")
        print(f"INTERACTIVE {mode_name.upper()}")
        print(f"{'='*70}\n")
        
        self.is_paused = False
        self.should_reset = False
        self.should_exit = False
        
        print(f"Starting {mode_name}...")
        print("Use Meshcat controls: Play | Pause | Repeat | Reset\n")
        
        import time
        
        try:
            # Run the simulation mode with interactive flag enabled
            self._interactive_mode = True
            mode_function()
            self._interactive_mode = False
            
        except Exception as e:
            print(colored(f"\nError in interactive simulation: {e}", "red"))
            import traceback
            traceback.print_exc()
        
        print(f"\n{'='*70}")
        print("INTERACTIVE SIMULATION COMPLETED")
        print(f"{'='*70}\n")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution flow for PyDrake simulation."""
    print("\n" + "=" * 70)
    print("PYDRAKE: Cart-Pendulum with 2-DOF Planar Manipulator")
    print("Educational Robotics Simulation")
    print("=" * 70)
    print(f"Mode: {SIMULATION_MODE}")
    print(f"Time Step: {SIMULATOR_TIME_STEP} s")
    print(f"Duration: {SIMULATION_DURATION} s")
    print("=" * 70 + "\n")
    
    try:
        # Create scene manager
        scene = DrakeSceneManager(
            cart_pendulum_params=RobotParams(
                urdf_path=CART_PENDULUM_URDF_PATH,
                initial_joint_positions=CART_PENDULUM_INITIAL_JOINT_POSITIONS,
                joint_damping=CART_PENDULUM_JOINT_DAMPING,
                joint_stiffness=CART_PENDULUM_JOINT_STIFFNESS,
                joint_friction=CART_PENDULUM_JOINT_FRICTION,
                position=CART_PENDULUM_POSITION,
            ),
            manipulator_params=RobotParams(
                urdf_path=MANIPULATOR_URDF_PATH,
                initial_joint_positions=MANIPULATOR_INITIAL_JOINT_POSITIONS,
                joint_damping=MANIPULATOR_JOINT_DAMPING,
                joint_stiffness=MANIPULATOR_JOINT_FRICTION,
                joint_friction=MANIPULATOR_JOINT_FRICTION,
                position=MANIPULATOR_POSITION,
                link_lengths=[1.0, 1.0],  # Link lengths for kinematics
            ),
        )
        
        # Run selected mode
        if INTERACTIVE:
            # Wrap with interactive controls
            if SIMULATION_MODE == "scene-viz":
                scene.run_simulation_interactive("Scene Visualization", scene.run_scene_viz)
            elif SIMULATION_MODE == "simulation":
                scene.run_simulation_interactive("Physics Simulation", scene.run_simulation)
            elif SIMULATION_MODE == "coupled-motion":
                scene.run_simulation_interactive("Coupled Motion", scene.run_coupled_motion)
            elif SIMULATION_MODE == "cart-toward-manipulator":
                scene.run_simulation_interactive("Cart Toward Manipulator", scene.run_cart_toward_manipulator)
            else:
                print(f"Unknown mode: {SIMULATION_MODE}")
        else:
            # Run without interactive controls
            if SIMULATION_MODE == "scene-viz":
                scene.run_scene_viz()
            elif SIMULATION_MODE == "simulation":
                scene.run_simulation()
            elif SIMULATION_MODE == "coupled-motion":
                scene.run_coupled_motion()
            elif SIMULATION_MODE == "cart-toward-manipulator":
                scene.run_cart_toward_manipulator()
            else:
                print(f"Unknown mode: {SIMULATION_MODE}")
    
    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR OCCURRED")
        print(f"{'='*70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
