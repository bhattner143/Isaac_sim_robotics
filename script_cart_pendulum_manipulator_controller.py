"""
Cart-Pendulum with 2-DOF Planar Manipulator

Demonstrates:
1. Custom URDF robot import (cart + pendulum system)
2. Separate 2-DOF planar manipulator
3. Scene visualization with class-based architecture

System:
- Cart: Moves along X-axis on rails at height 1.325m
- Pendulum: Hangs downward, rotates in XZ plane
- 2-DOF Manipulator: Planar manipulator in XY plane
"""

# ============================================================================
# IMPORTS: Standard Python Libraries
# ============================================================================
# These are built-in Python modules and external libraries for:
# - Application setup (SimulationApp)
# - Data structures (dataclass for clean configuration classes)
# - Type hints (Optional for better code clarity)
# - Abstract base classes (ABC for creating robot base class)
# - Command-line argument parsing (argparse for user options)
# - File operations (os, Path for file handling)
# - Math operations (math for trigonometry in kinematics)
# - Colored terminal output (termcolor for better readability)

from isaacsim import SimulationApp
from dataclasses import dataclass, asdict
from typing import Optional
from abc import ABC, abstractmethod
import argparse
import os
import json
import math
from pathlib import Path
from datetime import datetime
from termcolor import colored

# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================
# This section allows users to control the simulation behavior via command-line arguments.
# Example usage:
#   python script.py --mode coupled-motion --device cpu
#   python script.py --mode ee-trajectory --device cuda

parser = argparse.ArgumentParser()

# Device selection: CPU or GPU (CUDA)
# GPU acceleration is faster for large-scale simulations with many objects
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Simulation device")

# Simulation mode: Determines what the simulation demonstrates
# - scene-viz: Static visualization (no physics simulation)
# - simulation: Full physics simulation
# - test-simulation: Test cart motion independently
# - ee-trajectory: Move manipulator end-effector along X-axis
# - cart-toward-manipulator: Cart approaches manipulator autonomously
# - coupled-motion: Manipulator and cart connected via joint (default)
parser.add_argument(
    "--mode",
    type=str,
    choices=["scene-viz", "simulation", "test-simulation", "ee-trajectory", "cart-toward-manipulator", "coupled-motion"],
    default="coupled-motion",
    help="Mode: 'scene-viz' (static), 'simulation' (physics), 'test-simulation' (cart motion test), 'ee-trajectory' (move EE along cart direction), 'cart-toward-manipulator' (cart moves toward manipulator until they meet), 'cart-ee-aligned' (cart edge aligned with EE), 'coupled-motion' (manipulator moves cart-pendulum via joint)",
)

# Parse the arguments (parse_known_args allows unknown arguments to be ignored)
args, _ = parser.parse_known_args()

# ============================================================================
# STEP 1: LAUNCH ISAAC SIM APPLICATION
# ============================================================================
# IMPORTANT: SimulationApp must be created BEFORE importing any Isaac Sim modules!
# This initializes the Omniverse Kit framework and creates the rendering window.
#
# Configuration parameters:
# - headless: False means GUI is enabled (set to True for server/batch runs)
# - width/height: Window resolution in pixels (affects rendering quality)

simulation_app = SimulationApp({
    "headless": False,      # Show GUI window
    "width": 1280,          # Window width
    "height": 720,          # Window height
})

# ============================================================================
# STEP 2: IMPORT ISAAC SIM MODULES
# ============================================================================
# CRITICAL: These imports MUST come AFTER SimulationApp is created!
# The SimulationApp initialization sets up the Omniverse environment that
# these modules depend on.
#
# Module categories:
# - Stage/USD: Universal Scene Description (USD) format for 3D scenes
# - Physics: Articulations (multi-link robots), rigid bodies
# - Rendering: Lights, materials, geometry
# - Commands: High-level operations (URDF import, etc.)
# - Math libraries: PyTorch (tensors/GPU), Warp (parallel computing), NumPy

import isaacsim.core.experimental.utils.stage as stage_utils  # USD stage utilities
import omni.timeline  # Simulation timeline control
import omni.usd  # USD context and stage access
import omni.kit.commands  # High-level commands for operations
from isaacsim.asset.importer.urdf import _urdf  # URDF to USD converter
from pxr import UsdGeom, UsdLux, Gf, UsdShade, Sdf, UsdPhysics, PhysxSchema, Usd  # Pixar USD library
from omni.physx.scripts import utils as physx_utils  # PhysX utilities for joint creation
from omni.physx.scripts import physicsUtils  # PhysX utilities for rigid body setup
from isaacsim.core.experimental.objects import GroundPlane  # Ground plane object
from isaacsim.core.simulation_manager import SimulationManager  # Physics manager
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # Robot components
from omni.isaac.core import World  # Main simulation world
import torch  # PyTorch for tensor operations and GPU computing
import warp as wp  # NVIDIA Warp for parallel computing
import numpy as np  # NumPy for numerical computing
import matplotlib.pyplot as plt  # Matplotlib for plotting

# Helper function to check if USD needs regeneration
def needs_regeneration(urdf_path, usd_path):
    """Check if USD file needs to be regenerated based on modification times."""
    if not os.path.exists(usd_path):
        return True  # USD doesn't exist, need to generate
    
    if not os.path.exists(urdf_path):
        print(colored(f"WARNING: URDF file not found: {urdf_path}", "blue"))
        return False
    
    # Compare modification times
    urdf_mtime = os.path.getmtime(urdf_path)
    usd_mtime = os.path.getmtime(usd_path)
    
    return urdf_mtime > usd_mtime  # Regenerate if URDF is newer

# ============================================================================
# URDF TO USD CONVERSION FUNCTION
# ============================================================================

def convert_urdf_to_usd(urdf_path, output_usd_path, import_config=None):
    """
    Convert URDF file to USD format.
    
    Args:
        urdf_path: Path to input URDF file
        output_usd_path: Path to save USD file
        import_config: Import configuration dictionary
    
    Returns:
        prim_path: Path to the imported robot prim in the stage
    """
    print(f"\nConverting URDF to USD...")
    print(f"  Input:  {urdf_path}")
    print(f"  Output: {output_usd_path}")
    
    # Acquire URDF interface
    urdf_interface = _urdf.acquire_urdf_interface()
    
    # Create import config
    config = _urdf.ImportConfig()
    if import_config:
        config.convex_decomp = import_config.get("convex_decomp", False)
        config.fix_base = import_config.get("fix_base", False)
        config.make_default_prim = import_config.get("make_default_prim", True)
        config.self_collision = import_config.get("self_collision", False)
        config.distance_scale = import_config.get("distance_scale", 1.0)
        config.density = import_config.get("density", 0.0)
    
    # Parse and import URDF to USD file
    result, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=config,
        dest_path=output_usd_path
    )
    
    if not result:
        print("ERROR: Failed to convert URDF to USD")
        return None
    
    print(f"✓ URDF converted to USD")
    print(f"  Robot prim: {prim_path}")
    return prim_path


# ============================================================================
# USER CONFIGURATION
# ============================================================================

# --- Cart-Pendulum Configuration ---
CART_PENDULUM_URDF_PATH = str(Path("model/manipulators/cart_pendulum_2dof.urdf").absolute())
CART_PENDULUM_USD_PATH = str(Path("model/manipulators/cart_pendulum_2dof.usd").absolute())
CART_PENDULUM_PATH = "/World/cart_pendulum"
CART_PENDULUM_POSITION = (0.0, 0.0, 0.0)
CART_PENDULUM_ROTATION = 0.0
CART_PENDULUM_INITIAL_JOINT_POSITIONS = [0.0, 0.0]  # [cart_slider, pendulum_joint]
CART_PENDULUM_JOINT_DAMPING = [0.5, 0.05]  # Moderate damping - balances motion control and retraction
CART_PENDULUM_JOINT_STIFFNESS = [0, 0.1]  # Zero stiffness for free motion
CART_PENDULUM_JOINT_FRICTION = [0.05, 0.0]  # Very low friction to allow retraction

# --- Manipulator Configuration ---
MANIPULATOR_URDF_PATH = str(Path("model/manipulators/2dof_planar_manipulator.urdf").absolute())
MANIPULATOR_USD_PATH = str(Path("model/manipulators/2dof_planar_manipulator.usd").absolute())
MANIPULATOR_PATH = "/World/a_dof_planar_manipulator"
MANIPULATOR_POSITION = (-3.0, 0.0, 0.0)
MANIPULATOR_ROTATION = 0.0
MANIPULATOR_INITIAL_JOINT_POSITIONS = [math.radians(50), math.radians(-100.0)]  # [joint_1, joint_2] in radians
# EE position will be computed using PlanarManipulator.forward_kinematics() method
# For θ1=45°, θ2=-90°: x ≈ -1.5858m, y ≈ 0.0m, z = 1.325m
MANIPULATOR_EE_INITIAL_POSE = (-1.5858, 0.0, 1.325)  # Initial EE pose for ee-trajectory mode
MANIPULATOR_JOINT_DAMPING = [0.1, 0.1]  # Damping for both joints
MANIPULATOR_JOINT_FRICTION = [0.0, 0.0]  # Friction for both joints

# --- EE-Cart Coupling Joint Configuration ---
# Mimics hand grasping cart: compliant grip with moderate stiffness and damping
COUPLING_JOINT_TYPE = "revolute"  # Options: "fixed", "revolute", or "prismatic"

# Revolute joint impedance (for compliant hand grasp rotation)
EE_CART_COUPLING_JOINT_STIFFNESS = 500.0  # N·m/rad - hand compliance (soft grasp, not rigid)
EE_CART_COUPLING_JOINT_DAMPING  = 100.0     # N·m·s/rad - muscle damping in wrist/fingers
EE_CART_COUPLING_JOINT_FRICTION = 0.5    # N·m - friction from hand-cart contact
EE_CART_REVOLUTE_AXIS = "Z"  # Rotation axis for revolute joint: "X", "Y", or "Z"

# Prismatic joint impedance (for compliant linear coupling along movement axis)
EE_CART_PRISMATIC_STIFFNESS = 50.0  # N/m - linear spring constant (stiffer for linear motion)
EE_CART_PRISMATIC_DAMPING = 10.0      # N·s/m - linear damping in movement direction
EE_CART_PRISMATIC_FRICTION = 1.0      # N - friction force in linear motion
EE_CART_PRISMATIC_AXIS = "-X"  # Direction of prismatic joint: "X", "Y", or "Z"

# --- Scene Configuration ---
DISTANT_LIGHT_INTENSITY = 1000.0
DOME_LIGHT_INTENSITY = 300.0
DISTANT_LIGHT_ANGLE = 315.0
SIMULATION_MODE = args.mode
DEVICE = args.device

# --- Simulation Timing Configuration ---
SIMULATOR_TIME_STEP = 0.01  # seconds - physics simulation timestep
SIMULATION_DURATION = 30.0  # seconds - total simulation time
PENDULUM_SETTLING_TIME = 2.0  # seconds - time to allow pendulum to settle after trajectory


# ============================================================================
# PARAMETER CLASSES
# ============================================================================

@dataclass
class RobotParams:
    """Parameters for robot configuration."""
    urdf_path: str
    usd_path: str
    prim_path: str
    position: tuple[float, float, float]
    rotation_z: float
    initial_joint_positions: list[float]
    joint_damping: list[float]
    joint_stiffness: list[float]
    joint_friction: list[float]
    link_lengths: list[float] = None  # Link lengths for IK (optional, for manipulators)


@dataclass
class LightingParams:
    """Parameters for scene lighting."""
    distant_intensity: float = 1000.0
    dome_intensity: float = 300.0
    angle: float = 315.0


@dataclass
class RobotState:
    """Runtime state discovered from initialized robot."""
    robot: Articulation
    num_dof: int
    dof_names: list[str]


# ============================================================================
# ROBOT BASE CLASS (ABSTRACT)
# ============================================================================

class RobotBase(ABC):
    """
    Abstract Base Class for Robots
    
    DESIGN PATTERN: Template Method Pattern
    This class provides a common interface and shared functionality for all robots.
    Subclasses (CartPendulum, PlanarManipulator) inherit these methods and can
    override specific behaviors (e.g., set_joint_properties).
    
    EDUCATIONAL NOTE:
    - Abstract Base Class (ABC): Cannot be instantiated directly
    - Abstract methods (marked with @abstractmethod): MUST be implemented by subclasses
    - Concrete methods: Shared implementation used by all subclasses
    
    LIFECYCLE:
    1. Create instance with params → __init__
    2. Load to stage → load_to_stage()
    3. Reset world → world.reset() (external call)
    4. Initialize articulation → initialize_articulation()
    5. Set properties → set_joint_properties()
    6. Control robot → set_joint_positions(), get_joint_positions(), etc.
    """
    
    def __init__(self, params: RobotParams):
        """Initialize robot with configuration parameters.
        
        Args:
            params: RobotParams object containing all robot configuration
        
        Note: Robot state is None until initialize_articulation() is called.
        This is because articulations can only be created after the world is reset.
        """
        self.params = params
        self.state: Optional[RobotState] = None  # Created after world.reset()
    
    def load_to_stage(self, stage):
        """Load robot USD to stage."""
        print(f"\nLoading robot from USD: {self.params.usd_path}")
        
        # Reference the USD file in the stage
        prim = stage.OverridePrim(self.params.prim_path)
        prim.GetReferences().AddReference(self.params.usd_path)
        print(f"✓ Robot loaded at {self.params.prim_path}")
        
        # Apply transform if needed
        if prim and prim.IsValid():
            xformable = UsdGeom.Xformable(prim)
            # Get or create translate operation
            if self.params.position != (0.0, 0.0, 0.0):
                # Try to find existing translate op
                translate_op = None
                for op in xformable.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break
                # Use existing or create new
                if translate_op:
                    translate_op.Set(Gf.Vec3d(*self.params.position))
                else:
                    xformable.AddTranslateOp().Set(Gf.Vec3d(*self.params.position))
            
            # Get or create rotation operation
            if self.params.rotation_z != 0:
                # Try to find existing rotateZ op
                rotate_op = None
                for op in xformable.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                        rotate_op = op
                        break
                # Use existing or create new
                if rotate_op:
                    rotate_op.Set(self.params.rotation_z)
                else:
                    xformable.AddRotateZOp().Set(self.params.rotation_z)
    
    def print_prim_properties(self, prim_path):
        """Print all properties of a prim for debugging."""
        prim = self.get_prim(prim_path)
        if prim and prim.IsValid():
            print(f"\nProperties for {prim_path}:")
            property_names = [p.GetName() for p in prim.GetProperties()]
            for prop_name in property_names:
                print(f"  - {prop_name}")
            print()
        else:
            print(colored(f"WARNING: Prim not found at {prim_path}", "blue"))
    
    def initialize_articulation(self):
        """Initialize articulation object. Must be called after loading to stage and world.reset()."""
        if self.state is None:
            robot = Articulation(self.params.prim_path)
            
            # Discover DOF information
            num_dof = robot.num_dofs
            dof_names = robot.dof_names
            
            # Store runtime state
            self.state = RobotState(
                robot=robot,
                num_dof=num_dof,
                dof_names=dof_names
            )
            
            print(f"✓ Created articulation for {self.params.prim_path}")
            print(f"  DOFs: {num_dof}")
            for i, name in enumerate(dof_names):
                print(f"    Joint {i}: {name}")            
            
            # Call post-initialization hook for subclass-specific setup
            self._post_initialize_articulation()
        else:
            print(f"  Articulation already exists for {self.params.prim_path}")
    
    def _post_initialize_articulation(self):
        """Hook for subclasses to perform additional initialization after articulation is created.
        
        Override this method in subclasses to add robot-specific initialization logic
        that needs to run after the base articulation is set up.
        """
        pass  # Default: no additional initialization
    
    def set_initial_joint_positions(self):
        """Set initial joint positions using Articulation API."""
        if self.state is None:
            print(f"ERROR: Articulation not initialized for {self.params.prim_path}. Call initialize_articulation() first.")
            return
        
        self.state.robot.set_dof_positions(self.params.initial_joint_positions)
        print(f"✓ Set initial joint positions for {self.params.prim_path}: {self.params.initial_joint_positions}")
    
    def set_joint_positions(self, positions: list[float]):
        """
        Set joint positions using Articulation API.
        
        Args:
            positions: List of joint positions (can be partial - only first N DOFs will be set)
        """
        if self.state is None:
            print(f"ERROR: Articulation not initialized for {self.params.prim_path}. Call initialize_articulation() first.")
            return
        
        # Handle mismatched position counts gracefully
        if len(positions) != self.state.num_dof:
            if len(positions) < self.state.num_dof:
                # Get current positions and update only the provided ones
                current_positions = self.get_joint_positions()
                if current_positions is None:
                    print(f"ERROR: Could not get current joint positions")
                    return
                # Update only the first len(positions) DOFs
                for i in range(len(positions)):
                    current_positions[i] = positions[i]
                self.state.robot.set_dof_positions(current_positions)
            else:
                # More positions than DOFs - use only the first num_dof
                print(colored(f"WARNING: Expected {self.state.num_dof} joint positions, got {len(positions)}. Using first {self.state.num_dof} values", "blue"))
                self.state.robot.set_dof_positions(positions[:self.state.num_dof])
        else:
            # Exact match - use as-is
            self.state.robot.set_dof_positions(positions)
    
    def set_joint_position_targets(self, targets: list[float]):
        """
        Set joint position targets using Articulation API (for position control).
        
        This method sets target positions for position-controlled joints.
        Requires position control to be enabled on the joints.
        
        Args:
            targets: List of target joint positions (length must match num_dof)
        """
        if self.state is None:
            print(f"ERROR: Articulation not initialized for {self.params.prim_path}. Call initialize_articulation() first.")
            return
        
        if len(targets) != self.state.num_dof:
            print(f"ERROR: Expected {self.state.num_dof} joint targets, got {len(targets)}")
            return
        
        self.state.robot.set_joint_position_targets(targets)
    
    def get_joint_positions(self):
        """
        Get current joint positions using Articulation API.
        
        Returns:
            list: Current joint positions, or None if articulation not initialized
        """
        if self.state is None:
            print(f"ERROR: Articulation not initialized for {self.params.prim_path}. Call initialize_articulation() first.")
            return None
        
        # Convert warp array to numpy then to list
        positions = self.state.robot.get_dof_positions()
        return positions.numpy().flatten().tolist()
    
    def _set_joint_damping(self, joint_prim, damping_value: float, stiffness_value: float = 0.0, joint_type: str = "revolute"):
        """Helper method to set joint damping and stiffness.
        
        Args:
            joint_prim: USD prim for the joint
            damping_value: Damping coefficient
            stiffness_value: Stiffness coefficient (default: 0.0 for free motion)
            joint_type: Type of joint ('revolute' or 'prismatic')
        """
        from pxr import UsdPhysics
        
        # Use DriveAPI to set damping and stiffness
        if joint_type == "revolute":
            # For revolute joints, use angular drive
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive.CreateTypeAttr().Set("force")
            drive.CreateDampingAttr().Set(float(damping_value))
            drive.CreateStiffnessAttr().Set(float(stiffness_value))
        elif joint_type == "prismatic":
            pass
            # For prismatic joints, use linear drive
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
            
            # Print existing values if they exist
            existing_damping = drive.GetDampingAttr()
            existing_stiffness = drive.GetStiffnessAttr()
            if existing_damping:
                print(f"  Existing damping: {existing_damping.Get()}")
            if existing_stiffness:
                print(f"  Existing stiffness: {existing_stiffness.Get()}")
            
            drive.CreateTypeAttr().Set("force")
            drive.CreateDampingAttr().Set(float(damping_value))
            drive.CreateStiffnessAttr().Set(float(stiffness_value))
    
    def _set_joint_friction(self, joint_prim, friction_value: float):
        """Helper method to set joint friction.
        
        Args:
            joint_prim: USD prim for the joint
            friction_value: Friction coefficient
        """
        from pxr import PhysxSchema
        
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
        friction_attr = physx_joint.CreateJointFrictionAttr()
        friction_attr.Set(float(friction_value))
    
    def _get_joint_type(self, joint_prim) -> str:
        """Detect joint type from USD prim.
        
        Args:
            joint_prim: USD prim for the joint
            
        Returns:
            Joint type string: 'revolute', 'prismatic', or 'unknown'
        """
        from pxr import UsdPhysics
        
        if UsdPhysics.RevoluteJoint(joint_prim):
            return "revolute"
        elif UsdPhysics.PrismaticJoint(joint_prim):
            return "prismatic"
        else:
            return "unknown"
    
    # ========================================================================
    # ISAAC SIM API HELPER METHODS
    # ========================================================================
    
    def get_stage(self):
        """Get the USD stage."""
        import omni.usd
        return omni.usd.get_context().get_stage()
    
    def get_prim(self, prim_path: str):
        """
        Get prim at specified path.
        
        Args:
            prim_path: Path to the prim
            
        Returns:
            USD prim or None if not found
        """
        stage = self.get_stage()
        if not stage:
            return None
        return stage.GetPrimAtPath(prim_path)
    
    def get_world_transform(self, prim):
        """
        Get world transformation matrix of a prim.
        
        Args:
            prim: USD prim
            
        Returns:
            Gf.Matrix4d: World transformation matrix, or None if invalid
        """
        if not prim or not prim.IsValid():
            return None
        xformable = UsdGeom.Xformable(prim)
        return xformable.ComputeLocalToWorldTransform(0.0)
    
    def get_world_position(self, prim):
        """
        Get world position of a prim.
        
        Args:
            prim: USD prim
            
        Returns:
            tuple: (x, y, z) world position, or None if invalid
        """
        transform = self.get_world_transform(prim)
        if transform is None:
            return None
        translation = transform.ExtractTranslation()
        return (translation[0], translation[1], translation[2])
    
    def get_prim_world_position(self, prim_path: str):
        """
        Get world position of a prim by path.
        
        Args:
            prim_path: Path to the prim
            
        Returns:
            tuple: (x, y, z) world position, or None if not found
        """
        prim = self.get_prim(prim_path)
        return self.get_world_position(prim)
    
    @abstractmethod
    def set_joint_properties(self):
        """Set joint properties (damping, friction). Must be implemented by subclasses."""
        pass


# ============================================================================
# CART-PENDULUM CLASS
# ============================================================================

class CartPendulum(RobotBase):
    """
    Cart-Pendulum system.
    
    Manages:
    - Cart on rails with prismatic joint
    - Pendulum hanging from cart
    """
    
    def __init__(self, params: RobotParams):
        """Initialize cart-pendulum system."""
        super().__init__(params)
        # Articulation will be created after loading to stage
    
    def get_cart_world_position(self):
        """
        Get the cart's world position.
        
        Returns:
            tuple: (x, y, z) position in world coordinates, or None if not available
        """
        cart_path = f"{self.params.prim_path}/cart"
        return self.get_prim_world_position(cart_path)
    
    def print_cart_world_position(self):
        """Print the cart's world position."""
        position = self.get_cart_world_position()
        
        if position:
            print(f"\n{'='*60}")
            print(f"Cart World Position:")
            print(f"  X: {position[0]:8.4f} m")
            print(f"  Y: {position[1]:8.4f} m")
            print(f"  Z: {position[2]:8.4f} m")
            print(f"{'='*60}\n")
        else:
            print("ERROR: Could not retrieve cart position")
    
    def set_joint_properties(self):
        """Set joint properties (damping, friction) after physics is initialized."""
        if self.state is None:
            print("ERROR: Robot state not initialized. Call initialize_articulation() first.")
            return
        
        stage = self.get_stage()
        if not stage:
            print(colored("WARNING: No stage available to set joint properties", "blue"))
            return
        
        # Use discovered joint names from robot state
        for idx, joint_name in enumerate(self.state.dof_names):
            joint_path = f"{self.params.prim_path}/joints/{joint_name}"
            joint_prim = stage.GetPrimAtPath(joint_path)
            
            if joint_prim and joint_prim.IsValid():
                # Detect joint type automatically
                joint_type = self._get_joint_type(joint_prim)
                
                # Apply damping and stiffness using helper method
                if idx < len(self.params.joint_damping):
                    stiffness = self.params.joint_stiffness[idx] if idx < len(self.params.joint_stiffness) else 0.0
                    self._set_joint_damping(joint_prim, self.params.joint_damping[idx], stiffness, joint_type=joint_type)
                
                # Apply friction using helper method
                if idx < len(self.params.joint_friction):
                    self._set_joint_friction(joint_prim, self.params.joint_friction[idx])
                
                d = self.params.joint_damping[idx] if idx < len(self.params.joint_damping) else None
                s = self.params.joint_stiffness[idx] if idx < len(self.params.joint_stiffness) else None
                f = self.params.joint_friction[idx] if idx < len(self.params.joint_friction) else None
                print(f"✓ Set joint properties for {joint_name} ({joint_type}): damping={d}, stiffness={s}, friction={f}")
            else:
                print(colored(f"WARNING: Joint prim not found at {joint_path}", "blue"))


# ============================================================================
# MANIPULATOR CLASS
# ============================================================================

class PlanarManipulator(RobotBase):
    """
    2-DOF Planar Manipulator.
    
    Manages:
    - Mounting box and cylinder
    - Two revolute joints for planar motion
    - End-effector frame
    """
    
    def __init__(self, params: RobotParams):
        """Initialize planar manipulator."""
        super().__init__(params)
        # Articulation will be created after loading to stage
        self.link_lengths: list[float] = None  # Will be set from params or extracted
    
    def _post_initialize_articulation(self):
        """Extract link lengths after articulation is initialized if not provided in params."""
        # If link lengths provided in params, use them
        if self.params.link_lengths is not None and len(self.params.link_lengths) > 0:
            self.link_lengths = self.params.link_lengths
            print(f"✓ Using link lengths from params: {self.link_lengths}")
        else:
            # Otherwise, extract from geometry
            extracted_lengths = self.get_link_lengths()
            if extracted_lengths:
                self.link_lengths = extracted_lengths
                # Update params for consistency
                self.params.link_lengths = extracted_lengths
                print(f"✓ Extracted and stored link lengths: {self.link_lengths}")
            else:
                print(colored("WARNING: Could not determine link lengths", "blue"))
    
    def get_link_lengths(self):
        """Extract link lengths from robot geometry.
        
        Returns:
            list: Link lengths [L1, L2, ...] in meters
        """
        
        stage = self.get_stage()
        if not stage:
            print("ERROR: No stage available")
            return None
        
        link_lengths = []
        
        # For 2-DOF planar manipulator, extract lengths from link geometries
        # Navigate to visuals/mesh_0 to get actual geometry
        
        link_names = ["manipulator_link_1", "manipulator_link_2"]
        
        for idx, link_name in enumerate(link_names, 1):
            visual_path = f"{self.params.prim_path}/{link_name}/visuals/mesh_0"
            mesh_prim = stage.GetPrimAtPath(visual_path)
            
            if mesh_prim and mesh_prim.IsValid():
                # Activate the prim if it's inactive
                if not mesh_prim.IsActive():
                    mesh_prim.SetActive(True)
                    print(f"✓ Activated {visual_path}")
                
                # mesh_0 is typically an Xform, look for cylinder child
                cylinder_prim = None
                for child in mesh_prim.GetChildren():
                    if child.GetTypeName() == 'Cylinder':
                        cylinder_prim = child
                        break
                
                if cylinder_prim:
                    cylinder = UsdGeom.Cylinder(cylinder_prim)
                    if cylinder:
                        # Get cylinder height (along its axis)
                        height_attr = cylinder.GetHeightAttr()
                        if height_attr:
                            length = float(height_attr.Get())
                            link_lengths.append(length)
                            print(f"✓ Extracted Link {idx} length: {length:.4f} m (from cylinder height)")
                        else:
                            print(colored(f"WARNING: No height attribute on cylinder {cylinder_prim.GetPath()}", "blue"))
                    else:
                        print(colored(f"WARNING: No Cylinder child found in {visual_path}", "blue"))
                else:
                    print(colored(f"WARNING: Mesh prim not found at {visual_path}", "blue"))
        if len(link_lengths) == len(link_names):
            return link_lengths
        else:
            print(colored(f"WARNING: Could not extract all link lengths. Found {len(link_lengths)}/{len(link_names)}", "blue"))
            return None
        
    def set_joint_properties(self):
        """Set joint properties (damping, friction) after physics is initialized."""
        stage = self.get_stage()
        if not stage:
            print(colored("WARNING: No stage available to set joint properties", "blue"))
            return
        
        # Joint names for 2-DOF manipulator
        joint_names = ["manipulator_base_yaw", "manipulator_joint_2_yaw"]

        for idx, joint_name in enumerate(joint_names):
            joint_path = f"{self.params.prim_path}/joints/{joint_name}"
            joint_prim = stage.GetPrimAtPath(joint_path)

            if not joint_prim or not joint_prim.IsValid():
                print(colored(f"WARNING: Joint prim not found at {joint_path}", "blue"))
                continue

            # Apply damping using helper method
            if idx < len(self.params.joint_damping):
                self._set_joint_damping(joint_prim, self.params.joint_damping[idx], joint_type="revolute")

            # Apply friction using helper method
            if idx < len(self.params.joint_friction):
                self._set_joint_friction(joint_prim, self.params.joint_friction[idx])

            d = self.params.joint_damping[idx] if idx < len(self.params.joint_damping) else None
            f = self.params.joint_friction[idx] if idx < len(self.params.joint_friction) else None
            print(f"✓ Set joint properties for {joint_name}: damping={d}, friction={f}")
    
    def get_ee_world_position(self):
        """
        Get the end-effector frame's world position.
        
        Tries to get EE prim directly, falls back to computing from link_2 transform.
        
        Returns:
            tuple: (x, y, z) position in world coordinates, or None if not available
        """
        # Try to get EE prim directly (after _setup_ee_rigid_body creates it)
        ee_path_option1 = f"{self.params.prim_path}/manipulator_link_2/manipulator_ee"
        ee_path_option2 = f"{self.params.prim_path}/manipulator_ee"
        
        ee_position = self.get_prim_world_position(ee_path_option1)
        if ee_position is None:
            ee_position = self.get_prim_world_position(ee_path_option2)
        
        if ee_position is not None:
            return ee_position
        
        # Fallback: compute from link_2 transform
        link2_path = f"{self.params.prim_path}/manipulator_link_2"
        link2_prim = self.get_prim(link2_path)
        
        if not link2_prim:
            print(f"ERROR: Could not find prim at {link2_path}")
            return None
        
        # Get world transform of link_2
        link2_transform = self.get_world_transform(link2_prim)
        if link2_transform is None:
            print(f"ERROR: Could not get transform for {link2_path}")
            return None
        
        # EE is 1.0m along link_2's local X-axis (from URDF: <origin xyz="1.0 0 0"/>)
        ee_offset_local = Gf.Vec3d(1.0, 0.0, 0.0)
        
        # Transform local offset to world coordinates
        ee_position_world = link2_transform.Transform(ee_offset_local)
        
        return (ee_position_world[0], ee_position_world[1], ee_position_world[2])
    
    def print_ee_world_position(self):
        """Print the end-effector's world position."""
        position = self.get_ee_world_position()
        
        if position:
            print(f"\n{'='*60}")
            print(f"Manipulator End-Effector World Position:")
            print(f"  X: {position[0]:8.4f} m")
            print(f"  Y: {position[1]:8.4f} m")
            print(f"  Z: {position[2]:8.4f} m")
            print(f"{'='*60}\n")
        else:
            print("ERROR: Could not retrieve end-effector position")    
    
    def get_base_transform(self):
        """Get the base's world transformation matrix.
        
        Returns:
            Gf.Matrix4d: 4x4 transformation matrix from base to world, or None if unavailable
        """
        base_prim = self.get_prim(self.params.prim_path)
        return self.get_world_transform(base_prim)
    
    def transform_point_world_to_base(self, world_x, world_y, world_z):
        """Transform a point from world coordinates to base-relative coordinates.
        
        Uses proper transformation matrices to handle both translation and rotation.
        
        Args:
            world_x: X coordinate in world frame
            world_y: Y coordinate in world frame
            world_z: Z coordinate in world frame
            
        Returns:
            tuple: (x_base, y_base, z_base) in base frame, or None if transformation fails
        """
        # Get base's world transform
        base_to_world = self.get_base_transform()
        if base_to_world is None:
            return None
        
        # Invert to get world-to-base transform
        world_to_base = base_to_world.GetInverse()
        
        # Transform the point
        point_world = Gf.Vec3d(world_x, world_y, world_z)
        point_base = world_to_base.Transform(point_world)
        
        return (point_base[0], point_base[1], point_base[2])

    def compute_jacobian(self, theta1, theta2):
        """
        Compute Analytical Jacobian Matrix for 2-DOF Planar Manipulator
        
        WHAT IS THE JACOBIAN?
        The Jacobian is a matrix that maps joint velocities to end-effector velocities.
        It's the derivative of forward kinematics with respect to joint angles.
        
        MATHEMATICAL RELATIONSHIP:
        [v_x]   =  J  * [θ̇1]
        [v_y]          [θ̇2]
        
        where:
        - v_x, v_y: End-effector linear velocities (m/s)
        - θ̇1, θ̇2: Joint angular velocities (rad/s)
        - J: 2×2 Jacobian matrix
        
        ANALYTICAL JACOBIAN FOR 2-DOF PLANAR MANIPULATOR:
        Starting from forward kinematics:
          x = L1*cos(θ1) + L2*cos(θ1+θ2)
          y = L1*sin(θ1) + L2*sin(θ1+θ2)
        
        Taking partial derivatives:
          J = [∂x/∂θ1  ∂x/∂θ2]   =  [-L1*sin(θ1) - L2*sin(θ1+θ2),  -L2*sin(θ1+θ2)]
              [∂y/∂θ1  ∂y/∂θ2]      [ L1*cos(θ1) + L2*cos(θ1+θ2),   L2*cos(θ1+θ2)]
        
        APPLICATIONS:
        1. Velocity control: Convert desired EE velocity to joint velocities
        2. Differential IK: Iteratively move toward target position
        3. Singularity detection: Check det(J) ≈ 0 or condition number
        4. Force mapping: Relate joint torques to end-effector forces
        
        Args:
            theta1: Joint 1 angle in radians
            theta2: Joint 2 angle in radians
            
        Returns:
            np.ndarray: 2×2 Jacobian matrix, or None if link lengths unavailable
        
        EDUCATIONAL NOTE:
        Singularities occur when det(J) = 0, meaning the robot loses degrees
        of freedom in certain directions (e.g., fully extended or folded).
        """
        import numpy as np
        
        if self.link_lengths is None or len(self.link_lengths) < 2:
            print("ERROR: Link lengths not available. Ensure initialize_articulation() was called.")
            return None
        
        L1 = self.params.link_lengths[0]
        L2 = self.params.link_lengths[1]
        
        # Analytical Jacobian for 2-DOF planar manipulator
        s1 = np.sin(theta1)
        c1 = np.cos(theta1)
        s12 = np.sin(theta1 + theta2)
        c12 = np.cos(theta1 + theta2)
        
        J = np.array([
            [-L1*s1 - L2*s12, -L2*s12],
            [ L1*c1 + L2*c12,  L2*c12]
        ])
        
        return J
    
    def inverse_kinematics(self, target_x, target_y, target_z):
        """
        Compute Analytical Inverse Kinematics for 2-DOF Planar Manipulator
        
        INVERSE KINEMATICS PROBLEM:
        Given: Desired end-effector position (x, y, z) in world coordinates
        Find: Joint angles (θ1, θ2) that achieve this position
        
        ANALYTICAL SOLUTION (Geometric Method):
        For a 2-DOF planar manipulator with links L1 and L2:
        
        1. Transform world coordinates to base-relative coordinates
        2. Compute distance from base to target: r = sqrt(x² + y²)
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
            target_z: Target Z position in world frame (meters, typically constant)
            
        Returns:
            tuple: (theta1, theta2) in radians, or None if unreachable
        
        EDUCATIONAL NOTES:
        - Closed-form solution: Fast and exact (no iteration needed)
        - Multiple solutions possible (elbow-up vs elbow-down)
        - Singularities occur when arm is fully extended or folded
        """
        import numpy as np
        
        # Get link lengths from class attribute
        if self.link_lengths is None or len(self.link_lengths) < 2:
            print("ERROR: Link lengths not available. Ensure initialize_articulation() was called.")
            return None
        
        L1 = self.params.link_lengths[0]
        L2 = self.params.link_lengths[1]
        
        # Convert world coordinates to base-relative coordinates using transformation matrix
        base_coords = self.transform_point_world_to_base(target_x, target_y, target_z)
        if base_coords is None:
            print("ERROR: Failed to transform target to base frame")
            return None
        
        x_rel, y_rel, z_rel = base_coords
        
        # For planar manipulator in XY plane, only consider X and Y
        # Target distance from base
        r = np.sqrt(x_rel**2 + y_rel**2)
        
        # Check if target is reachable
        if r > (L1 + L2) or r < abs(L1 - L2):
            print(colored(f"WARNING: Target ({target_x}, {target_y}) is unreachable. Distance from base={r}, reach=[{abs(L1-L2)}, {L1+L2}]", "blue"))
            return None
        
        # Angle to target from base
        phi = np.arctan2(y_rel, x_rel)
        
        # Law of cosines to find theta2
        cos_theta2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)  # Numerical safety
        
        # Elbow-down configuration (negative theta2)
        theta2 = -np.arccos(cos_theta2)
        
        # Find theta1 using geometry
        alpha = np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
        theta1 = phi - alpha
        
        return (theta1, theta2)
    
    
    def inverse_kinematics_jacobian(self, target_x, target_y, target_z, initial_guess=None, max_iterations=100, tolerance=1e-4):
        """
        Compute inverse kinematics using numerical iterative method with Jacobian.
        
        Uses Newton-Raphson method: θ_new = θ_old + J^+ * error
        where J^+ is the pseudo-inverse of the Jacobian.
        
        Args:
            target_x: Target X position
            target_y: Target Y position
            target_z: Target Z position (not used for 2D planar manipulator)
            initial_guess: Initial joint angles [theta1, theta2] in radians (default: None, uses current positions or zeros)
            max_iterations: Maximum number of iterations (default: 100)
            tolerance: Position error tolerance in meters (default: 1e-4)
            
        Returns:
            tuple: (theta1, theta2) in radians, or None if failed to converge
        """
        import numpy as np
        
        if self.link_lengths is None or len(self.link_lengths) < 2:
            print("ERROR: Link lengths not available. Ensure initialize_articulation() was called.")
            return None
        
        # Convert world coordinates to base-relative coordinates using transformation matrix
        base_coords = self.transform_point_world_to_base(target_x, target_y, target_z)
        if base_coords is None:
            print("ERROR: Failed to transform target to base frame")
            return None
        
        x_target_rel, y_target_rel, z_target_rel = base_coords
        
        # Initialize with provided guess, current positions, or zeros
        if initial_guess is not None:
            theta1 = initial_guess[0]
            theta2 = initial_guess[1]
        else:
            current_positions = self.get_joint_positions()
            if current_positions is not None:
                theta1 = current_positions[0]
                theta2 = current_positions[1]
            else:
                theta1 = 0.0
                theta2 = 0.0
        
        # Iterative Newton-Raphson
        for iteration in range(max_iterations):
            # Compute current EE position (relative to base)
            L1 = self.params.link_lengths[0]
            L2 = self.params.link_lengths[1]
            x_current = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
            y_current = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
            
            # Compute position error
            error_x = x_target_rel - x_current
            error_y = y_target_rel - y_current
            error = np.array([error_x, error_y])
            
            # Check convergence
            error_norm = np.linalg.norm(error)
            if error_norm < tolerance:
                return (theta1, theta2)
            
            # Compute Jacobian
            J = self.compute_jacobian(theta1, theta2)
            if J is None:
                return None
            
            # Compute pseudo-inverse (for 2x2 full-rank, this is just inverse)
            try:
                J_inv = np.linalg.pinv(J)
            except np.linalg.LinAlgError:
                print(colored(f"WARNING: Jacobian is singular at iteration {iteration}", "blue"))
                return None
            
            # Update joint angles
            delta_theta = J_inv @ error
            theta1 += delta_theta[0]
            theta2 += delta_theta[1]
        
        print(colored(f"WARNING: Jacobian-based IK did not converge after {max_iterations} iterations. Final error: {error_norm:.6f}m", "blue"))
        return (theta1, theta2)
    
    def inverse_kinematics_differential(
        self,
        jacobian_end_effector: torch.Tensor,
        current_position: torch.Tensor,
        goal_position: torch.Tensor,
        method: str = "damped-least-squares",
        method_cfg: dict = None
    ) -> torch.Tensor:
        """
        Compute Differential (Velocity-Level) Inverse Kinematics
        
        CONCEPT:
        Instead of solving for exact joint angles (like analytical IK), differential IK
        computes small incremental joint angle changes (Δθ) to move the end-effector
        toward the goal. This is useful for:
        - Real-time trajectory tracking
        - Avoiding discontinuities from analytical solutions
        - Handling redundant manipulators (more DOF than needed)
        
        ALGORITHM:
        1. Compute position error: e = goal - current
        2. Compute Jacobian at current configuration: J(θ)
        3. Solve for joint velocity: θ̇ = J⁻¹ * e
        4. Update joint angles: θ_new = θ_old + Δt * θ̇
        
        METHODS FOR COMPUTING J⁻¹:
        
        1. PSEUDOINVERSE (Moore-Penrose):
           Δθ = J⁺ * e, where J⁺ = (JᵀJ)⁻¹Jᵀ
           - Minimizes ||Δθ|| (smallest joint motion)
           - Unstable near singularities
        
        2. TRANSPOSE METHOD:
           Δθ = k * Jᵀ * e
           - Simple and fast
           - Does not minimize error exactly
           - Stable near singularities
        
        3. DAMPED LEAST SQUARES (DLS) - Levenberg-Marquardt:
           Δθ = Jᵀ(JJᵀ + λ²I)⁻¹ * e
           - Adds damping term λ to avoid singularities
           - More stable than pseudoinverse
           - Trade-off: accuracy vs. stability (controlled by λ)
        
        4. SINGULAR VALUE DECOMPOSITION (SVD):
           J = UΣVᵀ, then J⁺ = VΣ⁺Uᵀ
           - Most robust method
           - Can filter small singular values
           - Computationally expensive
        
        Args:
            jacobian_end_effector: Jacobian matrix [batch, 2, num_dof] or [2, 2]
            current_position: Current EE position [batch, 2] or [2] (x, y in base frame)
            goal_position: Goal EE position [batch, 2] or [2] (x, y in base frame)
            method: IK method ("damped-least-squares" recommended, "pseudoinverse", "transpose", "singular-value-decomposition")
            method_cfg: Configuration dict:
                - scale: Step size multiplier (default: 1.0)
                - damping: Damping factor λ for DLS (default: 0.05)
                - min_singular_value: Threshold for SVD (default: 1e-5)
            
        Returns:
            torch.Tensor: Delta joint positions Δθ [batch, num_dof] or [num_dof]
        
        EDUCATIONAL NOTES:
        - Differential IK is iterative: repeat until error < threshold
        - Each iteration moves closer to goal (may not reach in one step)
        - Singularities occur when J loses rank (det(J) ≈ 0)
        - Damping helps stability but may slow convergence
        """
        if method_cfg is None:
            method_cfg = {"scale": 1.0, "damping": 0.05, "min_singular_value": 1e-5}
        
        scale = method_cfg.get("scale", 1.0)
        damping = method_cfg.get("damping", 0.05)
        min_singular_value = method_cfg.get("min_singular_value", 1e-5)
        
        # Handle batching - ensure inputs have batch dimension
        if jacobian_end_effector.dim() == 2:
            jacobian_end_effector = jacobian_end_effector.unsqueeze(0)  # [1, 2, 2]
        if current_position.dim() == 1:
            current_position = current_position.unsqueeze(0)  # [1, 2]
        if goal_position.dim() == 1:
            goal_position = goal_position.unsqueeze(0)  # [1, 2]
        
        # Compute position error
        error = (goal_position - current_position).unsqueeze(-1)  # [batch, 2, 1]
        
        # Compute delta joint angles based on method
        if method == "singular-value-decomposition":
            # Adaptive SVD
            U, S, Vh = torch.linalg.svd(jacobian_end_effector)
            inv_s = torch.where(S > min_singular_value, 1.0 / S, torch.zeros_like(S))
            pseudoinverse = torch.transpose(Vh, 1, 2) @ torch.diag_embed(inv_s) @ torch.transpose(U, 1, 2)
            delta_theta = (scale * pseudoinverse @ error).squeeze(-1)
        
        elif method == "pseudoinverse":
            # Moore-Penrose pseudoinverse
            pseudoinverse = torch.linalg.pinv(jacobian_end_effector)
            delta_theta = (scale * pseudoinverse @ error).squeeze(-1)
        
        elif method == "transpose":
            # Jacobian transpose method
            transpose = torch.transpose(jacobian_end_effector, 1, 2)
            delta_theta = (scale * transpose @ error).squeeze(-1)
        
        elif method == "damped-least-squares":
            # Damped least-squares (more stable near singularities)
            transpose = torch.transpose(jacobian_end_effector, 1, 2)
            lmbda = torch.eye(jacobian_end_effector.shape[1], device=jacobian_end_effector.device) * (damping ** 2)
            delta_theta = (scale * transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ error).squeeze(-1)
        
        else:
            raise ValueError(f"Invalid differential IK method: {method}")
        
        return delta_theta
    
    def forward_kinematics_base_frame(self, theta1, theta2):
        """
        Compute forward kinematics in base frame for 2-DOF planar manipulator.
        
        Given joint angles (theta1, theta2), compute end-effector position relative to base.
        
        Args:
            theta1: Joint 1 angle in radians
            theta2: Joint 2 angle in radians
            
        Returns:
            tuple: (x_rel, y_rel) position in base frame, or None if link lengths unavailable
        """
        if self.link_lengths is None or len(self.link_lengths) < 2:
            print("ERROR: Link lengths not available. Ensure initialize_articulation() was called.")
            return None
        
        L1 = self.link_lengths[0]
        L2 = self.link_lengths[1]
        
        # FK: Compute EE position relative to base
        x_rel = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
        y_rel = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
        
        return (x_rel, y_rel)
    
    def forward_kinematics(self, theta1, theta2):
        """
        Compute Forward Kinematics for 2-DOF Planar Manipulator
        
        FORWARD KINEMATICS PROBLEM:
        Given: Joint angles (θ1, θ2)
        Find: End-effector position (x, y, z) in world coordinates
        
        SOLUTION FOR 2-DOF PLANAR MANIPULATOR:
        For a planar manipulator with two revolute joints and link lengths L1, L2:
        
        In base frame:
          x_base = L1*cos(θ1) + L2*cos(θ1+θ2)
          y_base = L1*sin(θ1) + L2*sin(θ1+θ2)
          z_base = constant (manipulator operates in XY plane)
        
        Transform to world frame:
          x_world = x_base + base_x_offset
          y_world = y_base + base_y_offset
          z_world = z_base + base_z_offset
        
        INTUITION:
        - θ1 rotates first link from X-axis
        - θ2 rotates second link relative to first link
        - Combined angle (θ1+θ2) gives second link's absolute orientation
        - Sum of link vectors gives end-effector position
        
        Args:
            theta1: Joint 1 angle in radians (base rotation)
            theta2: Joint 2 angle in radians (elbow rotation)
            
        Returns:
            tuple: (x, y, z) position in world coordinates, or None if link lengths unavailable
        
        EDUCATIONAL NOTE:
        Forward kinematics is unique (one solution), while inverse kinematics
        can have multiple solutions (elbow-up vs elbow-down configurations).
        """
        # Compute base-frame FK
        base_frame_pos = self.forward_kinematics_base_frame(theta1, theta2)
        if base_frame_pos is None:
            return None
        
        x_rel, y_rel = base_frame_pos
        
        # Convert to world coordinates
        x_world = self.params.position[0] + x_rel
        y_world = self.params.position[1] + y_rel
        z_world = self.params.position[2] + 1.2875 + 0.0125  # Base Z + mount height + offset
        
        return (x_world, y_world, z_world)
    
    
    
    

# ============================================================================
# SCENE MANAGER CLASS
# ============================================================================

class SceneManager:
    """
    Scene Manager: Orchestrates the Entire Simulation
    
    RESPONSIBILITIES:
    1. Setup: Create USD stage, add robots, lights, ground plane
    2. Initialization: Reset physics, initialize articulations
    3. Execution: Run different simulation modes (static viz, physics, coupled motion)
    
    DESIGN PATTERN: Facade Pattern
    SceneManager provides a simple interface to complex subsystems:
    - CartPendulum robot
    - PlanarManipulator robot
    - USD stage management
    - Physics simulation (World)
    
    ARCHITECTURE:
    SceneManager (orchestrator)
      ├── CartPendulum (cart-pendulum system)
      ├── PlanarManipulator (2-DOF arm)
      ├── World (physics simulation)
      └── USD Stage (3D scene)
    """
    
    def __init__(
        self,
        cart_pendulum_params: RobotParams,
        manipulator_params: RobotParams,
        lighting_params: LightingParams,
    ):
        """Initialize scene manager with robot and lighting configurations.
        
        Args:
            cart_pendulum_params: Configuration for cart-pendulum system
            manipulator_params: Configuration for planar manipulator
            lighting_params: Scene lighting parameters
        
        Note: Actual scene creation happens in initialize_stage(), not here.
        This constructor only stores parameters and creates robot objects.
        """
        # Store configuration parameters
        self.cart_pendulum_params = cart_pendulum_params
        self.manipulator_params = manipulator_params
        self.lighting_params = lighting_params
        
        # Create robot subsystem instances
        # These objects manage their own USD prims and physics properties
        self.cart_pendulum = CartPendulum(cart_pendulum_params)
        self.manipulator = PlanarManipulator(manipulator_params)
        
        # Coupling joint reference (set when joint is created)
        self.coupling_joint_prim = None
        self.coupling_joint_prev_angle = 0.0
        self.coupling_joint_prev_time = 0.0
        
        # Data logging for plotting
        self.data_log = {
            'time': [],
            'manip_joint1_pos': [],
            'manip_joint2_pos': [],
            'manip_joint1_vel': [],
            'manip_joint2_vel': [],
            'ee_cart_joint_pos': [],
            'ee_cart_joint_vel': [],
            'cart_pos': [],
            'cart_vel': [],
            'pendulum_pos': [],
            'pendulum_vel': [],
        }
    
    
    def initialize_stage(self):
        """Initialize stage with both robots from USD files."""
        print("Initializing stage...")
        
        # Create World instance (this creates the stage)
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        
        print("✓ Ground plane added")
        
        # Add grid and lighting after world is created
        self.add_grid()
        self.add_lighting()
        
        # Load both robots to stage
        stage = self.manipulator.get_stage()
        self.cart_pendulum.load_to_stage(stage)
        self.manipulator.load_to_stage(stage)
        
        print("✓ Stage initialized")
    
    def add_grid(self):
        """Add black grid on ground plane."""
        stage = self.manipulator.get_stage()
        
        grid_size = 20
        grid_spacing = 1.0
        num_lines = int(grid_size / grid_spacing) + 1
        
        grid_path = "/World/Grid"
        grid_xform = UsdGeom.Xform.Define(stage, grid_path)
        
        material_path = "/World/Materials/GridMaterial"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        
        line_index = 0
        offset = grid_size / 2
        
        for i in range(num_lines):
            y_pos = -offset + i * grid_spacing
            line_path = f"{grid_path}/LineX_{line_index}"
            line = UsdGeom.Mesh.Define(stage, line_path)
            
            thickness = 0.01
            points = [
                (-offset, y_pos - thickness/2, 0.001),
                (offset, y_pos - thickness/2, 0.001),
                (offset, y_pos + thickness/2, 0.001),
                (-offset, y_pos + thickness/2, 0.001)
            ]
            
            line.CreatePointsAttr(points)
            line.CreateFaceVertexCountsAttr([4])
            line.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            UsdShade.MaterialBindingAPI(line).Bind(material)
            line_index += 1
        
        for i in range(num_lines):
            x_pos = -offset + i * grid_spacing
            line_path = f"{grid_path}/LineY_{line_index}"
            line = UsdGeom.Mesh.Define(stage, line_path)
            
            thickness = 0.01
            points = [
                (x_pos - thickness/2, -offset, 0.001),
                (x_pos + thickness/2, -offset, 0.001),
                (x_pos + thickness/2, offset, 0.001),
                (x_pos - thickness/2, offset, 0.001)
            ]
            
            line.CreatePointsAttr(points)
            line.CreateFaceVertexCountsAttr([4])
            line.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            UsdShade.MaterialBindingAPI(line).Bind(material)
            line_index += 1
        
        print(f"✓ Black grid added ({num_lines}x{num_lines} lines)")
    
    def add_lighting(self):
        """Add lighting to the scene."""
        print("Adding lights...")
        
        stage = self.manipulator.get_stage()
        
        distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
        distant_light.CreateIntensityAttr(self.lighting_params.distant_intensity)
        
        distant_light_prim = stage.GetPrimAtPath("/World/DistantLight")
        xformable = UsdGeom.Xformable(distant_light_prim)
        rotate_op = xformable.GetOrderedXformOps()
        if rotate_op:
            rotate_op[0].Set(Gf.Vec3d(self.lighting_params.angle, 0, 0))
        else:
            xformable.AddRotateXYZOp().Set(Gf.Vec3d(self.lighting_params.angle, 0, 0))
        
        dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(self.lighting_params.dome_intensity)
        
        print("✓ Lights added")
    
    def create_ee_cart_joint(self):
        """
        EE-Cart joint is now defined in the combined URDF as ee_to_cart_joint (fixed).
        This method delegates to specific joint type implementation if dynamic coupling is needed.
        """
        if COUPLING_JOINT_TYPE == "fixed":
            print("✓ EE-Cart joint (fixed) already defined in combined URDF as 'ee_to_cart_joint'")
            return True
        elif COUPLING_JOINT_TYPE == "revolute":
            return self.create_ee_cart_revolute_joint_v2()
        elif COUPLING_JOINT_TYPE == "prismatic":
            return self.create_ee_cart_prismatic_joint_v2()
        else:
            print(f"ERROR: Unknown coupling joint type: {COUPLING_JOINT_TYPE}")
            return False
    
    def create_ee_cart_revolute_joint_v2(self):
        """Create a compliant revolute joint using physx_utils base, then configure spring-damper."""
        print("\nCreating EE-Cart REVOLUTE coupling joint (using physx_utils)...")
        
        stage = self.manipulator.get_stage()
        if not stage:
            print("ERROR: No stage available")
            return False
        
        # Define paths
        ee_path = f"{self.manipulator.params.prim_path}/manipulator_link_2/manipulator_ee"
        cart_path = f"{self.cart_pendulum.params.prim_path}/cart"
        
        # Verify prims exist
        ee_prim = stage.GetPrimAtPath(ee_path)
        cart_prim = stage.GetPrimAtPath(cart_path)
        
        if not ee_prim or not ee_prim.IsValid():
            print(f"ERROR: EE prim not found at {ee_path}")
            return False
        
        if not cart_prim or not cart_prim.IsValid():
            print(f"ERROR: Cart prim not found at {cart_path}")
            return False
        
        print(f"✓ Found EE prim: {ee_path}")
        print(f"✓ Found cart prim: {cart_path}")
        
        # Ensure EE marker is a rigid body
        if not ee_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            print("  Adding RigidBodyAPI to EE marker...")
            UsdPhysics.RigidBodyAPI.Apply(ee_prim)
            collision_api = UsdPhysics.CollisionAPI.Apply(ee_prim)
            collision_api.CreateCollisionEnabledAttr(False)
        
        # Use physx_utils to create the revolute joint
        try:
            joint_prim = physx_utils.createJoint(stage, "Revolute", ee_prim, cart_prim)
            print(f"✓ Created revolute joint at {joint_prim.GetPath()}")
            
            # Now configure the joint properties using USD API
            joint_usd = UsdPhysics.RevoluteJoint(joint_prim)
            
            # Set rotation axis
            joint_usd.CreateAxisAttr().Set(EE_CART_REVOLUTE_AXIS)
            
            # Get current relative rotation for target position
            ee_world_transform = self.manipulator.get_world_transform(ee_prim)
            cart_world_transform = self.cart_pendulum.get_world_transform(cart_prim)
            relative_transform = cart_world_transform.GetInverse() * ee_world_transform
            relative_rotation_quat = relative_transform.ExtractRotationQuat()
            w, x, y, z = relative_rotation_quat.GetReal(), relative_rotation_quat.GetImaginary()[0], \
                         relative_rotation_quat.GetImaginary()[1], relative_rotation_quat.GetImaginary()[2]
            
            # Extract angle based on rotation axis
            if EE_CART_REVOLUTE_AXIS == "Z":
                current_angle = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            elif EE_CART_REVOLUTE_AXIS == "Y":
                current_angle = math.asin(2.0 * (w * y - z * x))
            else:  # "X"
                current_angle = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
            
            # Apply angular drive (spring-damper)
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive.CreateTypeAttr().Set("force")
            drive.CreateDampingAttr().Set(float(EE_CART_COUPLING_JOINT_DAMPING))
            drive.CreateStiffnessAttr().Set(float(EE_CART_COUPLING_JOINT_STIFFNESS))
            drive.CreateTargetPositionAttr().Set(float(current_angle))
            
            # Apply friction
            physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
            physx_joint.CreateJointFrictionAttr().Set(float(EE_CART_COUPLING_JOINT_FRICTION))
            
            # Set joint limits
            joint_usd.CreateLowerLimitAttr().Set(-90.0)
            joint_usd.CreateUpperLimitAttr().Set(90.0)
            
            # Store joint prim reference
            self.coupling_joint_prim = joint_prim
            
            print(f"  Type: Revolute (1 DOF - compliant hand grasp)")
            print(f"  Axis: {EE_CART_REVOLUTE_AXIS}")
            print(f"  Target angle: {math.degrees(current_angle):.2f}° ({current_angle:.4f} rad)")
            print(f"  Stiffness: {EE_CART_COUPLING_JOINT_STIFFNESS} N·m/rad")
            print(f"  Damping: {EE_CART_COUPLING_JOINT_DAMPING} N·m·s/rad")
            print(f"  Friction: {EE_CART_COUPLING_JOINT_FRICTION} N·m")
            print(f"  Limits: ±90°\n")
            return True
        except Exception as e:
            print(f"ERROR: Failed to create joint: {e}")
            return False
    
    def create_ee_cart_prismatic_joint_v2(self):
        """Create a compliant prismatic joint for linear cart movement along EE motion direction."""
        print("\nCreating EE-Cart PRISMATIC coupling joint (using physx_utils)...")
        
        stage = self.manipulator.get_stage()
        if not stage:
            print("ERROR: No stage available")
            return False
        
        # Define paths
        ee_path = f"{self.manipulator.params.prim_path}/manipulator_link_2/manipulator_ee"
        cart_path = f"{self.cart_pendulum.params.prim_path}/cart"
        
        # Verify prims exist
        ee_prim = stage.GetPrimAtPath(ee_path)
        cart_prim = stage.GetPrimAtPath(cart_path)
        
        if not ee_prim or not ee_prim.IsValid():
            print(f"ERROR: EE prim not found at {ee_path}")
            return False
        
        if not cart_prim or not cart_prim.IsValid():
            print(f"ERROR: Cart prim not found at {cart_path}")
            return False
        
        print(f"✓ Found EE prim: {ee_path}")
        print(f"✓ Found cart prim: {cart_path}")
        
        # Ensure EE marker is a rigid body
        if not ee_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            print("  Adding RigidBodyAPI to EE marker...")
            UsdPhysics.RigidBodyAPI.Apply(ee_prim)
            collision_api = UsdPhysics.CollisionAPI.Apply(ee_prim)
            collision_api.CreateCollisionEnabledAttr(False)
        
        # Use physx_utils to create the prismatic joint
        try:
            joint_prim = physx_utils.createJoint(stage, "Prismatic", ee_prim, cart_prim)
            print(f"✓ Created prismatic joint at {joint_prim.GetPath()}")
            
            # Now configure the joint properties using USD API
            joint_usd = UsdPhysics.PrismaticJoint(joint_prim)
            
            # Set movement axis
            if EE_CART_PRISMATIC_AXIS == "X":
                joint_usd.CreateAxisAttr().Set("X")
            elif EE_CART_PRISMATIC_AXIS == "Y":
                joint_usd.CreateAxisAttr().Set("Y")
            else:  # Default to Z
                joint_usd.CreateAxisAttr().Set("Z")
            
            # Get current relative position for target position
            ee_world_transform = self.manipulator.get_world_transform(ee_prim)
            cart_world_transform = self.cart_pendulum.get_world_transform(cart_prim)
            relative_transform = cart_world_transform.GetInverse() * ee_world_transform
            relative_translation = relative_transform.ExtractTranslation()
            
            # Extract position component along the joint axis
            if EE_CART_PRISMATIC_AXIS == "X":
                current_position = relative_translation[0]
            elif EE_CART_PRISMATIC_AXIS == "Y":
                current_position = relative_translation[1]
            else:  # Z
                current_position = relative_translation[2]
            
            # Apply linear drive (spring-damper)
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
            drive.CreateTypeAttr().Set("force")
            drive.CreateDampingAttr().Set(float(EE_CART_PRISMATIC_DAMPING))
            drive.CreateStiffnessAttr().Set(float(EE_CART_PRISMATIC_STIFFNESS))
            drive.CreateTargetPositionAttr().Set(float(current_position))
            
            # Apply friction
            physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
            physx_joint.CreateJointFrictionAttr().Set(float(EE_CART_PRISMATIC_FRICTION))
            
            # Set joint limits (reasonable range for cart motion)
            joint_usd.CreateLowerLimitAttr().Set(-0.5)  # -50cm max retraction
            joint_usd.CreateUpperLimitAttr().Set(0.5)   # +50cm max extension
            
            # Store joint prim reference
            self.coupling_joint_prim = joint_prim
            
            print(f"  Type: Prismatic (1 DOF - compliant linear coupling)")
            print(f"  Axis: {EE_CART_PRISMATIC_AXIS}")
            print(f"  Current position: {current_position:.4f} m")
            print(f"  Stiffness: {EE_CART_PRISMATIC_STIFFNESS} N/m")
            print(f"  Damping: {EE_CART_PRISMATIC_DAMPING} N·s/m")
            print(f"  Friction: {EE_CART_PRISMATIC_FRICTION} N")
            print(f"  Limits: [-0.5, 0.5] m (±50 cm)")
            print(f"  Behavior: Cart slides along {EE_CART_PRISMATIC_AXIS}-axis with compliant coupling\n")
            return True
        except Exception as e:
            print(f"ERROR: Failed to create joint: {e}")
            return False
    
    def create_ee_cart_revolute_joint(self):
        """Create a compliant revolute joint (spring-damper hinge) between EE and cart."""
        from pxr import UsdPhysics, PhysxSchema
        
        print("\nCreating EE-Cart REVOLUTE coupling joint...")
        
        stage = self.manipulator.get_stage()
        if not stage:
            print("ERROR: No stage available")
            return False
        
        # Define paths
        ee_path = f"{self.manipulator.params.prim_path}/manipulator_link_2/manipulator_ee"
        cart_path = f"{self.cart_pendulum.params.prim_path}/cart"
        joint_path = "/World/ee_cart_coupling_joint"
        
        # Verify prims exist
        ee_prim = stage.GetPrimAtPath(ee_path)
        cart_prim = stage.GetPrimAtPath(cart_path)
        
        if not ee_prim or not ee_prim.IsValid():
            print(f"ERROR: EE prim not found at {ee_path}")
            return False
        
        if not cart_prim or not cart_prim.IsValid():
            print(f"ERROR: Cart prim not found at {cart_path}")
            return False
        
        print(f"✓ Found EE prim: {ee_path}")
        print(f"✓ Found cart prim: {cart_path}")
        
        # Get current positions to set joint transform
        ee_world_transform = self.manipulator.get_world_transform(ee_prim)
        ee_translation = ee_world_transform.ExtractTranslation()
        
        cart_world_transform = self.cart_pendulum.get_world_transform(cart_prim)
        
        # Compute joint position in cart's local frame
        cart_to_world_inv = cart_world_transform.GetInverse()
        joint_pos_in_cart = cart_to_world_inv.Transform(ee_translation)
        
        print(f"  EE world position: ({ee_translation[0]:.4f}, {ee_translation[1]:.4f}, {ee_translation[2]:.4f})")
        print(f"  Joint position in cart frame: ({joint_pos_in_cart[0]:.4f}, {joint_pos_in_cart[1]:.4f}, {joint_pos_in_cart[2]:.4f})")
        
        # Create revolute joint
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        
        # Set joint bodies
        joint.CreateBody0Rel().SetTargets([ee_path])
        joint.CreateBody1Rel().SetTargets([cart_path])
        
        # Set joint anchor positions in local frames
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))  # At EE origin
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(joint_pos_in_cart[0], joint_pos_in_cart[1], joint_pos_in_cart[2]))
        
        # Set rotation axis as token (not vector)
        joint.CreateAxisAttr().Set(EE_CART_REVOLUTE_AXIS)
        
        # Compute current relative rotation angle to use as target position
        relative_transform = cart_world_transform.GetInverse() * ee_world_transform
        relative_rotation_quat = relative_transform.ExtractRotationQuat()
        w, x, y, z = relative_rotation_quat.GetReal(), relative_rotation_quat.GetImaginary()[0], \
                     relative_rotation_quat.GetImaginary()[1], relative_rotation_quat.GetImaginary()[2]
        
        # Extract angle based on rotation axis
        if EE_CART_REVOLUTE_AXIS == "Z":
            current_angle = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        elif EE_CART_REVOLUTE_AXIS == "Y":
            current_angle = math.asin(2.0 * (w * y - z * x))
        else:  # "X"
            current_angle = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        
        # Apply angular drive (spring-damper for compliance)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr().Set("force")  # Force-based drive
        drive.CreateDampingAttr().Set(float(EE_CART_COUPLING_JOINT_DAMPING))
        drive.CreateStiffnessAttr().Set(float(EE_CART_COUPLING_JOINT_STIFFNESS))
        drive.CreateTargetPositionAttr().Set(float(current_angle))  # Spring rest position at current angle
        
        # Apply friction (hand-cart contact resistance)
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
        physx_joint.CreateJointFrictionAttr().Set(float(EE_CART_COUPLING_JOINT_FRICTION))
        
        # Set joint limits (wrist rotation range ±90°)
        joint.CreateLowerLimitAttr().Set(-90.0)  # -π/2 radians
        joint.CreateUpperLimitAttr().Set(90.0)   # +π/2 radians
        
        # Store joint prim reference for later state reading
        self.coupling_joint_prim = joint.GetPrim()
        
        print(f"✓ Created revolute joint: {joint_path}")
        print(f"  Type: Revolute (1 DOF - compliant hand grasp)")
        print(f"  Axis: {EE_CART_REVOLUTE_AXIS}")
        print(f"  Target angle: {math.degrees(current_angle):.2f}° ({current_angle:.4f} rad)")
        print(f"  Stiffness: {EE_CART_COUPLING_JOINT_STIFFNESS} N·m/rad (hand compliance)")
        print(f"  Damping: {EE_CART_COUPLING_JOINT_DAMPING} N·m·s/rad (muscle damping)")
        print(f"  Friction: {EE_CART_COUPLING_JOINT_FRICTION} N·m (grip friction)")
        print(f"  Limits: ±90° (wrist rotation range)")
        print(f"  Behavior: Soft grasp with compliance and friction\n")
        
        return True
    
    def step_simulation(self, num_steps=10):
        """
        Step the simulation for a given number of steps.
        
        Args:
            num_steps: Number of simulation steps to execute (default: 10)
        """
        for _ in range(num_steps):
            self.world.step(render=True)
            simulation_app.update()
    
    def log_data(self, time_val):
        """Log current joint positions and velocities for plotting."""
        # Get manipulator joint states
        manip_positions = self.manipulator.get_joint_positions()
        manip_velocities = self.manipulator.state.robot.get_dof_velocities().numpy().flatten().tolist()
        
        # Get cart-pendulum joint states
        cart_positions = self.cart_pendulum.get_joint_positions()
        cart_velocities = self.cart_pendulum.state.robot.get_dof_velocities().numpy().flatten().tolist()
        
        if manip_positions and manip_velocities and cart_positions and cart_velocities:
            self.data_log['time'].append(time_val)
            self.data_log['manip_joint1_pos'].append(manip_positions[0])
            self.data_log['manip_joint2_pos'].append(manip_positions[1])
            self.data_log['manip_joint1_vel'].append(manip_velocities[0])
            self.data_log['manip_joint2_vel'].append(manip_velocities[1])
            self.data_log['cart_pos'].append(cart_positions[0])
            self.data_log['cart_vel'].append(cart_velocities[0])
            self.data_log['pendulum_pos'].append(cart_positions[1])
            self.data_log['pendulum_vel'].append(cart_velocities[1])
            
            # Log EE-cart coupling joint (read actual joint state if available)
            if COUPLING_JOINT_TYPE == "revolute" and self.coupling_joint_prim is not None:
                # Try to read joint state from PhysX joint
                try:
                    physx_joint_state = PhysxSchema.PhysxJointStateAPI(self.coupling_joint_prim)
                    if physx_joint_state.GetJointState(0) is not None:
                        # Joint has actual state we can read
                        joint_state = physx_joint_state.GetJointState(0)
                        angle_rad = float(joint_state[0]) if joint_state else 0.0
                        angular_vel = float(joint_state[1]) if len(joint_state) > 1 else 0.0
                    else:
                        # Fallback: compute from transforms
                        stage = self.manipulator.get_stage()
                        ee_path = f"{self.manipulator.params.prim_path}/manipulator_link_2/manipulator_ee"
                        cart_path = f"{self.cart_pendulum.params.prim_path}/cart"
                        
                        ee_prim = stage.GetPrimAtPath(ee_path)
                        cart_prim = stage.GetPrimAtPath(cart_path)
                        
                        if ee_prim and cart_prim and ee_prim.IsValid() and cart_prim.IsValid():
                            ee_transform = self.manipulator.get_world_transform(ee_prim)
                            cart_transform = self.cart_pendulum.get_world_transform(cart_prim)
                            cart_inv = cart_transform.GetInverse()
                            relative_transform = cart_inv * ee_transform
                            relative_rotation_quat = relative_transform.ExtractRotationQuat()
                            
                            w, x, y, z = relative_rotation_quat.GetReal(), relative_rotation_quat.GetImaginary()[0], \
                                         relative_rotation_quat.GetImaginary()[1], relative_rotation_quat.GetImaginary()[2]
                            
                            if EE_CART_REVOLUTE_AXIS == "Z":
                                angle_rad = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                            elif EE_CART_REVOLUTE_AXIS == "Y":
                                angle_rad = math.asin(2.0 * (w * y - z * x))
                            else:
                                angle_rad = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
                            
                            dt = time_val - self.coupling_joint_prev_time
                            if dt > 1e-6:
                                angular_vel = (angle_rad - self.coupling_joint_prev_angle) / dt
                            else:
                                angular_vel = 0.0
                            
                            self.coupling_joint_prev_angle = angle_rad
                            self.coupling_joint_prev_time = time_val
                        else:
                            angle_rad = 0.0
                            angular_vel = 0.0
                    
                    self.data_log['ee_cart_joint_pos'].append(float(angle_rad))
                    self.data_log['ee_cart_joint_vel'].append(float(angular_vel))
                except Exception as e:
                    # Fallback to geometric calculation
                    stage = self.manipulator.get_stage()
                    ee_path = f"{self.manipulator.params.prim_path}/manipulator_link_2/manipulator_ee"
                    cart_path = f"{self.cart_pendulum.params.prim_path}/cart"
                    
                    ee_prim = stage.GetPrimAtPath(ee_path)
                    cart_prim = stage.GetPrimAtPath(cart_path)
                    
                    if ee_prim and cart_prim and ee_prim.IsValid() and cart_prim.IsValid():
                        ee_transform = self.manipulator.get_world_transform(ee_prim)
                        cart_transform = self.cart_pendulum.get_world_transform(cart_prim)
                        cart_inv = cart_transform.GetInverse()
                        relative_transform = cart_inv * ee_transform
                        relative_rotation_quat = relative_transform.ExtractRotationQuat()
                        
                        w, x, y, z = relative_rotation_quat.GetReal(), relative_rotation_quat.GetImaginary()[0], \
                                     relative_rotation_quat.GetImaginary()[1], relative_rotation_quat.GetImaginary()[2]
                        
                        if EE_CART_REVOLUTE_AXIS == "Z":
                            angle_rad = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                        elif EE_CART_REVOLUTE_AXIS == "Y":
                            angle_rad = math.asin(2.0 * (w * y - z * x))
                        else:
                            angle_rad = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
                        
                        dt = time_val - self.coupling_joint_prev_time
                        if dt > 1e-6:
                            angular_vel = (angle_rad - self.coupling_joint_prev_angle) / dt
                        else:
                            angular_vel = 0.0
                        
                        self.coupling_joint_prev_angle = angle_rad
                        self.coupling_joint_prev_time = time_val
                        
                        self.data_log['ee_cart_joint_pos'].append(float(angle_rad))
                        self.data_log['ee_cart_joint_vel'].append(float(angular_vel))
                    else:
                        self.data_log['ee_cart_joint_pos'].append(0.0)
                        self.data_log['ee_cart_joint_vel'].append(0.0)
            elif COUPLING_JOINT_TYPE == "prismatic" and self.coupling_joint_prim is not None:
                # Similar logic for prismatic joint
                try:
                    physx_joint_state = PhysxSchema.PhysxJointStateAPI(self.coupling_joint_prim)
                    if physx_joint_state.GetJointState(0) is not None:
                        joint_state = physx_joint_state.GetJointState(0)
                        position = float(joint_state[0]) if joint_state else 0.0
                        velocity = float(joint_state[1]) if len(joint_state) > 1 else 0.0
                    else:
                        position = 0.0
                        velocity = 0.0
                    
                    self.data_log['ee_cart_joint_pos'].append(float(position))
                    self.data_log['ee_cart_joint_vel'].append(float(velocity))
                except Exception as e:
                    self.data_log['ee_cart_joint_pos'].append(0.0)
                    self.data_log['ee_cart_joint_vel'].append(0.0)
    
    def plot_results(self):
        """Plot joint positions and velocities after simulation."""
        if len(self.data_log['time']) == 0:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(5, 2, figsize=(14, 14))
        fig.suptitle('Joint Positions and Velocities', fontsize=16)
        
        # Manipulator Joint 1
        axes[0, 0].plot(self.data_log['time'], np.degrees(self.data_log['manip_joint1_pos']), 'b-', linewidth=2)
        axes[0, 0].set_ylabel('Angle (deg)')
        axes[0, 0].set_title('Manipulator Joint 1 Position')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.data_log['time'], np.degrees(self.data_log['manip_joint1_vel']), 'b-', linewidth=2)
        axes[0, 1].set_ylabel('Angular Vel (deg/s)')
        axes[0, 1].set_title('Manipulator Joint 1 Velocity')
        axes[0, 1].grid(True)
        
        # Manipulator Joint 2
        axes[1, 0].plot(self.data_log['time'], np.degrees(self.data_log['manip_joint2_pos']), 'r-', linewidth=2)
        axes[1, 0].set_ylabel('Angle (deg)')
        axes[1, 0].set_title('Manipulator Joint 2 Position')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(self.data_log['time'], np.degrees(self.data_log['manip_joint2_vel']), 'r-', linewidth=2)
        axes[1, 1].set_ylabel('Angular Vel (deg/s)')
        axes[1, 1].set_title('Manipulator Joint 2 Velocity')
        axes[1, 1].grid(True)
        
        # EE-Cart Coupling Joint
        axes[2, 0].plot(self.data_log['time'], np.degrees(self.data_log['ee_cart_joint_pos']), 'c-', linewidth=2)
        axes[2, 0].set_ylabel('Angle (deg)')
        axes[2, 0].set_title(f'EE-Cart Joint Position ({COUPLING_JOINT_TYPE})')
        axes[2, 0].grid(True)
        
        axes[2, 1].plot(self.data_log['time'], np.degrees(self.data_log['ee_cart_joint_vel']), 'c-', linewidth=2)
        axes[2, 1].set_ylabel('Angular Vel (deg/s)')
        axes[2, 1].set_title(f'EE-Cart Joint Velocity ({COUPLING_JOINT_TYPE})')
        axes[2, 1].grid(True)
        
        # Cart Position
        axes[3, 0].plot(self.data_log['time'], self.data_log['cart_pos'], 'g-', linewidth=2)
        axes[3, 0].set_ylabel('Position (m)')
        axes[3, 0].set_title('Cart Position')
        axes[3, 0].grid(True)
        
        axes[3, 1].plot(self.data_log['time'], self.data_log['cart_vel'], 'g-', linewidth=2)
        axes[3, 1].set_ylabel('Velocity (m/s)')
        axes[3, 1].set_title('Cart Velocity')
        axes[3, 1].grid(True)
        
        # Pendulum Angle
        axes[4, 0].plot(self.data_log['time'], np.degrees(self.data_log['pendulum_pos']), 'm-', linewidth=2)
        axes[4, 0].set_xlabel('Time (s)')
        axes[4, 0].set_ylabel('Angle (deg)')
        axes[4, 0].set_title('Pendulum Angle')
        axes[4, 0].grid(True)
        
        axes[4, 1].plot(self.data_log['time'], np.degrees(self.data_log['pendulum_vel']), 'm-', linewidth=2)
        axes[4, 1].set_xlabel('Time (s)')
        axes[4, 1].set_ylabel('Angular Vel (deg/s)')
        axes[4, 1].set_title('Pendulum Angular Velocity')
        axes[4, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('joint_positions_velocities.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to: joint_positions_velocities.png")
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
            output_path = f"configs/simulation_config_{timestamp}.json"
        
        # Automatically convert dataclasses to dicts using asdict()
        config = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "simulation_mode": SIMULATION_MODE,
                "device": DEVICE,
            },
            "simulation_parameters": {
                "time_step": SIMULATOR_TIME_STEP,
                "duration": SIMULATION_DURATION,
                "settling_time": PENDULUM_SETTLING_TIME,
            },
            "cart_pendulum_config": asdict(self.cart_pendulum.params),
            "manipulator_config": asdict(self.manipulator.params),
            "lighting_config": asdict(self.lighting_params),
            "coupling_joint_config": {
                "type": COUPLING_JOINT_TYPE,
                "revolute": {
                    "stiffness": EE_CART_COUPLING_JOINT_STIFFNESS,
                    "damping": EE_CART_COUPLING_JOINT_DAMPING,
                    "friction": EE_CART_COUPLING_JOINT_FRICTION,
                    "axis": EE_CART_REVOLUTE_AXIS,
                },
                "prismatic": {
                    "stiffness": EE_CART_PRISMATIC_STIFFNESS,
                    "damping": EE_CART_PRISMATIC_DAMPING,
                    "friction": EE_CART_PRISMATIC_FRICTION,
                    "axis": EE_CART_PRISMATIC_AXIS,
                }
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

    def run_test_scene(self):
        """Run simulation in display mode."""
        print(f"\n{'='*70}")
        print("DISPLAY MODE - Static scene visualization")
        print("Press Ctrl+C or close window to exit")
        print(f"{'='*70}\n")
        
        # Save configuration for this run
        self.save_configuration_to_json()
        
        # Reset world to initialize physics
        self.world.reset()
        
        # Initialize articulations after world reset
        print("\n" + "="*70)
        print("INITIALIZING ARTICULATIONS")
        print("="*70)
        self.cart_pendulum.initialize_articulation()
        self.manipulator.initialize_articulation()

        # Print manipulator prim properties for debugging
        print("\n" + "="*70)
        print("MANIPULATOR PRIM PROPERTIES")
        print("="*70)
        self.manipulator.print_prim_properties(f"{self.manipulator_params.prim_path}/joints/manipulator_base_yaw")
        self.manipulator.print_prim_properties(f"{self.manipulator_params.prim_path}/joints/manipulator_joint_2_yaw")
        self.manipulator.print_prim_properties(f"{self.manipulator_params.prim_path}/manipulator_link_1")
        self.manipulator.print_prim_properties(f"{self.manipulator_params.prim_path}/manipulator_link_2")
        self.manipulator.print_prim_properties(f"{self.manipulator_params.prim_path}/manipulator_link_2/manipulator_ee")
        
        # Print initial end-effector world position
        print("\n" + "="*70)
        print("INITIAL END-EFFECTOR POSITION (before setting joint properties)")
        print("="*70)
        self.manipulator.print_ee_world_position()
        
        # Set joint properties for both robots
        print("\n" + "="*70)
        print("SETTING JOINT PROPERTIES")
        print("="*70)
        self.cart_pendulum.set_joint_properties()
        self.manipulator.set_joint_properties()

        print("\n" + "="*70)
        print("SETTING INITIAL JOINT POSITIONS")
        print("="*70)
        self.cart_pendulum.set_initial_joint_positions()
        self.manipulator.set_initial_joint_positions()
        
        # Print end-effector world position after setting joint properties
        print("\n" + "="*70)
        print("END-EFFECTOR POSITION (after setting joint properties)")
        print("="*70)
        self.manipulator.print_ee_world_position()
        
        # Keep simulation running in display mode
        frame_count = 0
        while simulation_app.is_running():
            self.world.step(render=True)
            simulation_app.update()
            
            # Print positions every 60 frames (~1 second at 60 FPS)
            if frame_count % 60 == 0:
                # Get cart world position
                cart_position = self.cart_pendulum.get_cart_world_position()
                
                # Get end-effector world position
                ee_position = self.manipulator.get_ee_world_position()
                
                if cart_position and ee_position:
                    print(f"Frame {frame_count:6d} | Cart: X={cart_position[0]:8.4f}  Y={cart_position[1]:8.4f}  Z={cart_position[2]:8.4f} | EE: X={ee_position[0]:8.4f}  Y={ee_position[1]:8.4f}  Z={ee_position[2]:8.4f}")
                elif cart_position:
                    print(f"Frame {frame_count:6d} | Cart: X={cart_position[0]:8.4f}  Y={cart_position[1]:8.4f}  Z={cart_position[2]:8.4f}")
                elif ee_position:
                    print(f"Frame {frame_count:6d} | EE: X={ee_position[0]:8.4f}  Y={ee_position[1]:8.4f}  Z={ee_position[2]:8.4f}")
            
            frame_count += 1   
    
    
    
    
    
    

    
    def update_coupling_joint_drive(self):
        """Update the coupling joint drive target to follow current EE-cart relative motion."""
        if self.coupling_joint_prim is None:
            return
        
        try:
            stage = self.manipulator.get_stage()
            ee_path = f"{self.manipulator.params.prim_path}/manipulator_link_2/manipulator_ee"
            cart_path = f"{self.cart_pendulum.params.prim_path}/cart"
            
            ee_prim = stage.GetPrimAtPath(ee_path)
            cart_prim = stage.GetPrimAtPath(cart_path)
            
            if not (ee_prim and cart_prim and ee_prim.IsValid() and cart_prim.IsValid()):
                return
            
            # Get world transforms
            ee_transform = self.manipulator.get_world_transform(ee_prim)
            cart_transform = self.cart_pendulum.get_world_transform(cart_prim)
            
            # Compute relative transform
            cart_inv = cart_transform.GetInverse()
            relative_transform = cart_inv * ee_transform
            
            if COUPLING_JOINT_TYPE == "revolute":
                # Extract rotation angle for revolute joint
                relative_rotation_quat = relative_transform.ExtractRotationQuat()
                w, x, y, z = relative_rotation_quat.GetReal(), relative_rotation_quat.GetImaginary()[0], \
                             relative_rotation_quat.GetImaginary()[1], relative_rotation_quat.GetImaginary()[2]
                
                if EE_CART_REVOLUTE_AXIS == "Z":
                    angle_rad = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                elif EE_CART_REVOLUTE_AXIS == "Y":
                    angle_rad = math.asin(2.0 * (w * y - z * x))
                else:  # "X"
                    angle_rad = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
                
                # Update drive target to follow current angle
                drive_api = UsdPhysics.DriveAPI.Get(self.coupling_joint_prim, "angular")
                if drive_api:
                    drive_api.CreateTargetPositionAttr().Set(float(angle_rad))
            
            elif COUPLING_JOINT_TYPE == "prismatic":
                # Extract position component for prismatic joint
                relative_translation = relative_transform.ExtractTranslation()
                
                if EE_CART_PRISMATIC_AXIS == "X":
                    position = relative_translation[0]
                elif EE_CART_PRISMATIC_AXIS == "Y":
                    position = relative_translation[1]
                else:  # Z
                    position = relative_translation[2]
                
                # Update drive target to follow current position
                drive_api = UsdPhysics.DriveAPI.Get(self.coupling_joint_prim, "linear")
                if drive_api:
                    drive_api.CreateTargetPositionAttr().Set(float(position))
        
        except Exception as e:
            # Silently fail - coupling joint drive update is optional
            pass
    
    def run_coupled_motion(self):
        """
        Demonstrate Coupled Motion: Manipulator Drives Cart-Pendulum System
        
        CONCEPT:
        This mode demonstrates mechanical coupling between two independent robots:
        1. Planar manipulator (2-DOF arm) - actuated (moves via joint commands)
        2. Cart-pendulum system - passive (follows manipulator motion)
        3. Fixed joint connects manipulator end-effector to cart edge
        
        PHYSICS:
        - Manipulator applies forces/torques through coupling joint
        - Cart translates along rail (prismatic joint)
        - Pendulum swings due to cart acceleration (inertial forces)
        - System demonstrates multi-body dynamics with constraints
        
        EDUCATIONAL OBJECTIVES:
        1. Understand joint constraints in multi-robot systems
        2. Observe force transmission through coupling
        3. Analyze pendulum swing from cart acceleration
        4. Practice trajectory planning with coupled systems
        
        SIMULATION FLOW:
        1. Initialize both robots in aligned position
        2. Create fixed joint between EE and cart
        3. Generate manipulator trajectory (straight line motion)
        4. Execute trajectory: manipulator pulls/pushes cart
        5. Log positions
        
        REAL-WORLD APPLICATIONS:
        - Mobile manipulation (robot arm on moving base)
        - Crane systems (payload swinging during motion)
        - Humanoid robots (arm motion affects body balance)
        """
        print(f"\n{'='*70}")
        print("COUPLED MOTION: MANIPULATOR MOVES CART-PENDULUM")
        print(f"{'='*70}\n")
        
        # Save configuration for this run
        self.save_configuration_to_json()
        
        # Reset world to initialize physics
        self.world.reset()
        
        # Initialize articulations
        self.cart_pendulum.initialize_articulation()
        self.manipulator.initialize_articulation()
        
        # Set joint properties
        self.cart_pendulum.set_joint_properties()
        self.manipulator.set_joint_properties()
        
        # Set initial positions
        print(f"\n{'='*70}")
        print("INITIAL SETUP")
        print(f"{'='*70}")
        
        # Set manipulator to initial position
        theta1_init = MANIPULATOR_INITIAL_JOINT_POSITIONS[0]
        theta2_init = MANIPULATOR_INITIAL_JOINT_POSITIONS[1]
        self.manipulator.state.robot.set_dof_positions([theta1_init, theta2_init])
        print(f"Manipulator angles: theta1={math.degrees(theta1_init):.2f}°, theta2={math.degrees(theta2_init):.2f}°")
        
        # Step simulation
        self.step_simulation(10)
        
        # Get EE position
        ee_pos = self.manipulator.get_ee_world_position()
        if not ee_pos:
            print("ERROR: Could not get EE position")
            return
        
        # Position cart so edge aligns with EE
        cart_length_x = 0.3
        cart_half_length = cart_length_x / 2.0
        cart_center_x = ee_pos[0] + cart_half_length
        
        self.cart_pendulum.set_joint_positions([cart_center_x])
        self.step_simulation(10)
        
        print(f"Cart positioned at: X={cart_center_x:.4f}m")
        print(f"EE at: X={ee_pos[0]:.4f}m")
        
        # Create coupling joint
        print(f"\n{'='*70}")
        print("CREATING COUPLING JOINT")
        print(f"{'='*70}")
        joint_created = self.create_ee_cart_joint()
        
        if not joint_created:
            print("ERROR: Failed to create coupling joint")
            return
        
        # Step simulation to let joint settle
        self.step_simulation(10)
        
        # Generate manipulator trajectory (straight line in workspace)
        print(f"\n{'='*70}")
        print("GENERATING MANIPULATOR TRAJECTORY")
        print(f"{'='*70}")
        
        # Get initial EE position for trajectory planning
        initial_ee_pos = self.manipulator.get_ee_world_position()
        if not initial_ee_pos:
            print("ERROR: Could not get initial EE position")
            return
        
        print(f"Initial EE position: X={initial_ee_pos[0]:.4f}, Y={initial_ee_pos[1]:.4f}, Z={initial_ee_pos[2]:.4f}")
        
        # Trajectory parameters (straight line along X)
        target_y = initial_ee_pos[1]  # Keep Y constant
        target_z = initial_ee_pos[2]  # Keep Z constant
        x_start = initial_ee_pos[0]
        x_range = 0.6  # Move ±0.4m along X (larger motion)
        duration = 2.0  # 4 seconds (faster motion for more acceleration)
        dt = 1.0 / 60.0  # 60 FPS
        
        num_steps = int(duration / dt)
        trajectory = []
        
        for i in range(num_steps):
            t = i / (num_steps - 1)
            # Sinusoidal motion along X (straight line back and forth)
            target_x = x_start + x_range * np.sin(2 * np.pi * 0.5 * t)
            
            # Compute inverse kinematics for this target position
            joint_angles = self.manipulator.inverse_kinematics(target_x, target_y, target_z)
            
            if joint_angles is not None:
                trajectory.append((joint_angles[0], joint_angles[1]))
            else:
                print(colored(f"WARNING: IK failed for waypoint {i}: X={target_x:.4f}", "blue"))
                # Skip this waypoint if IK fails
                continue
        
        print(f"✓ Generated {len(trajectory)} waypoints")
        print(f"  Duration: {duration:.1f}s")
        print(f"  X range: [{x_start - x_range/2:.4f}, {x_start + x_range/2:.4f}] m")
        print(f"  Y constant: {target_y:.4f} m")
        print(f"  Z constant: {target_z:.4f} m")
        print(f"  Motion: Straight line along X-axis")
        print(f"{'='*70}\n")
        
        # Execute trajectory
        print(f"\n{'='*70}")
        print("EXECUTING TRAJECTORY")
        print("Manipulator will move cart, swinging pendulum")
        print(f"{'='*70}\n")
        
        # Print table header
        print(f"{'Time (s)':>10} | {'EE X':>10} | {'EE Y':>10} | {'EE Z':>10} | {'Cart X':>10} | {'Cart Y':>10} | {'Cart Z':>10} | {'Pend X':>10} | {'Pend Y':>10} | {'Pend Z':>10}")
        print(f"{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
        
        frame_count = 0
        waypoint_idx = 0
        
        while simulation_app.is_running() and waypoint_idx < len(trajectory):
            # Get current waypoint
            if waypoint_idx < len(trajectory):
                theta1, theta2 = trajectory[waypoint_idx]
                
                # Set manipulator joint positions
                self.manipulator.set_joint_positions([theta1, theta2])
                
                waypoint_idx += 1
            
            # Update coupling joint drive to follow current EE-cart motion
            self.update_coupling_joint_drive()
            
            # Step simulation
            self.world.step(render=True)
            simulation_app.update()
            
            # Log data for plotting
            time_s = frame_count * dt
            self.log_data(time_s)
            
            # Print progress every 30 frames (~0.5 seconds)
            if frame_count % 30 == 0:
                
                # Get EE position
                ee_pos = self.manipulator.get_ee_world_position()
                
                # Get cart position
                cart_pos = self.cart_pendulum.get_cart_world_position()
                
                # Get pendulum position (tip of pendulum)
                pendulum_path = f"{self.cart_pendulum.params.prim_path}/pendulum"
                pend_pos = self.cart_pendulum.get_prim_world_position(pendulum_path)
                
                if ee_pos and cart_pos:
                    ee_str = f"{ee_pos[0]:10.4f} | {ee_pos[1]:10.4f} | {ee_pos[2]:10.4f}"
                    cart_str = f"{cart_pos[0]:10.4f} | {cart_pos[1]:10.4f} | {cart_pos[2]:10.4f}"
                    if pend_pos:
                        pend_str = f"{pend_pos[0]:10.4f} | {pend_pos[1]:10.4f} | {pend_pos[2]:10.4f}"
                    else:
                        pend_str = f"{'N/A':>10} | {'N/A':>10} | {'N/A':>10}"
                    print(f"{time_s:10.2f} | {ee_str} | {cart_str} | {pend_str}")
            
            frame_count += 1
        
        # Pendulum settling period
        settling_frames = int(PENDULUM_SETTLING_TIME / dt)
        
        print(f"\n{'='*70}")
        print("TRAJECTORY COMPLETE - ALLOWING PENDULUM TO SETTLE")
        print(f"Settling time: {PENDULUM_SETTLING_TIME}s")
        print(f"{'='*70}\n")
        
        for settling_frame in range(settling_frames):
            # Step simulation (manipulator stays at final position)
            self.world.step(render=True)
            simulation_app.update()
            
            # Continue logging data during settling
            time_s = frame_count * dt
            self.log_data(time_s)
            
            # Print settling progress every 30 frames
            if settling_frame % 30 == 0:
                # Get pendulum state
                pendulum_positions = self.cart_pendulum.get_joint_positions()
                if pendulum_positions and len(pendulum_positions) > 1:
                    pend_angle = np.degrees(pendulum_positions[1])
                    print(f"Settling... t={time_s:5.2f}s | Pendulum angle: {pend_angle:6.2f}°")
            
            frame_count += 1
        
        print(f"\n{'='*70}")
        print("SETTLING COMPLETE")
        print(f"{'='*70}\n")
        
        # Plot results
        print("\nGenerating plots...")
        self.plot_results()
        
        # Keep running
        print("\nSimulation continues. Press Ctrl+C or close window to exit.\n")
        while simulation_app.is_running():
            self.world.step(render=True)
            simulation_app.update()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main Execution Flow: Setup and Run Simulation
    
    EXECUTION SEQUENCE:
    1. Check/convert URDF files to USD format
    2. Create scene manager with robot configurations
    3. Initialize USD stage (robots, lights, ground)
    4. Run selected simulation mode
    5. Clean up and close application
    
    URDF vs USD:
    - URDF (Unified Robot Description Format): XML-based, ROS standard
    - USD (Universal Scene Description): Pixar format, high-performance 3D
    - Isaac Sim converts URDF → USD for physics simulation
    
    ERROR HANDLING:
    - Try-except-finally ensures clean shutdown
    - Exceptions are logged with full traceback
    - simulation_app.close() always executes (cleanup)
    """
    print("\n" + "=" * 70)
    print("DEBUG: main() function started")
    print("Cart-Pendulum with 2-DOF Planar Manipulator")
    print("Educational Robotics Simulation")
    print("=" * 70)
    print(f"Mode: {SIMULATION_MODE}")
    print(f"Device: {DEVICE}")
    print("=" * 70 + "\n")
    
    try:
        # ====================================================================
        # STEP 1: CONVERT URDF TO USD (if needed)
        # ====================================================================
        # URDF files define robot structure (links, joints, geometry)
        # USD files are optimized for Isaac Sim physics simulation
        # Conversion happens automatically if URDF is newer than USD
        
        # Convert cart-pendulum URDF to USD if needed
        if needs_regeneration(CART_PENDULUM_URDF_PATH, CART_PENDULUM_USD_PATH):
            if os.path.exists(CART_PENDULUM_USD_PATH):
                print(f"\nCart-Pendulum URDF modified, regenerating USD...")
                print(f"  Removing existing USD: {CART_PENDULUM_USD_PATH}")
                os.remove(CART_PENDULUM_USD_PATH)
            else:
                print(f"\nCart-Pendulum USD not found, converting from URDF...")
            
            import_config = {
                "convex_decomp": False,
                "fix_base": True,  # Fix world link in space
                "make_default_prim": True,
                "self_collision": False,
                "distance_scale": 1.0,
                "density": 0.0,
            }
            convert_urdf_to_usd(
                urdf_path=CART_PENDULUM_URDF_PATH,
                output_usd_path=CART_PENDULUM_USD_PATH,
                import_config=import_config
            )
        else:
            print(f"\nUsing existing cart-pendulum USD (up to date): {CART_PENDULUM_USD_PATH}")
        
        # Convert manipulator URDF to USD if needed
        if needs_regeneration(MANIPULATOR_URDF_PATH, MANIPULATOR_USD_PATH):
            if os.path.exists(MANIPULATOR_USD_PATH):
                print(f"\nManipulator URDF modified, regenerating USD...")
                print(f"  Removing existing USD: {MANIPULATOR_USD_PATH}")
                os.remove(MANIPULATOR_USD_PATH)
            else:
                print(f"\nManipulator USD not found, converting from URDF...")
            
            import_config = {
                "convex_decomp": False,
                "fix_base": True,  # Fix world link in space
                "make_default_prim": True,
                "self_collision": False,
                "distance_scale": 1.0,
                "density": 0.0,
            }
            convert_urdf_to_usd(
                urdf_path=MANIPULATOR_URDF_PATH,
                output_usd_path=MANIPULATOR_USD_PATH,
                import_config=import_config
            )
        else:
            print(f"\nUsing existing manipulator USD (up to date): {MANIPULATOR_USD_PATH}")
        
        # Create scene manager
        scene = SceneManager(
            cart_pendulum_params=RobotParams(
                urdf_path=CART_PENDULUM_URDF_PATH,
                usd_path=CART_PENDULUM_USD_PATH,
                prim_path=CART_PENDULUM_PATH,
                position=CART_PENDULUM_POSITION,
                rotation_z=CART_PENDULUM_ROTATION,
                initial_joint_positions=CART_PENDULUM_INITIAL_JOINT_POSITIONS,
                joint_damping=CART_PENDULUM_JOINT_DAMPING,
                joint_stiffness=CART_PENDULUM_JOINT_STIFFNESS,
                joint_friction=CART_PENDULUM_JOINT_FRICTION,
            ),
            manipulator_params=RobotParams(
                urdf_path=MANIPULATOR_URDF_PATH,
                usd_path=MANIPULATOR_USD_PATH,
                prim_path=MANIPULATOR_PATH,
                position=MANIPULATOR_POSITION,
                rotation_z=MANIPULATOR_ROTATION,
                initial_joint_positions=MANIPULATOR_INITIAL_JOINT_POSITIONS,
                joint_damping=MANIPULATOR_JOINT_DAMPING,
                joint_stiffness=[0.0, 0.0],  # Zero stiffness for manipulator joints
                joint_friction=MANIPULATOR_JOINT_FRICTION,
            ),
            lighting_params=LightingParams(
                distant_intensity=DISTANT_LIGHT_INTENSITY,
                dome_intensity=DOME_LIGHT_INTENSITY,
                angle=DISTANT_LIGHT_ANGLE,
            ),
        )
        
        # Initialize stage with both robots
        try:
            scene.initialize_stage()
            print("DEBUG: Stage initialization complete")
        except Exception as stage_error:
            print(f"ERROR during stage initialization: {stage_error}")
            import traceback
            traceback.print_exc()
            raise
        
        print(f"\n{'-'*70}")
        print(f"DEBUG: Stage initialized, SIMULATION_MODE = {SIMULATION_MODE}")
        print(f"{'-'*70}\n")
        
        # Run appropriate mode
        if SIMULATION_MODE == "scene-viz":
            print("Starting scene-viz mode...")
            scene.run_test_scene()
        elif SIMULATION_MODE == "coupled-motion":
            print("Starting coupled-motion mode...")
            scene.run_coupled_motion()
        else:
            print(f"Starting simulation mode: {SIMULATION_MODE}...")
            scene.run_simulation()
    
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"FATAL ERROR OCCURRED")
        print(f"{'='*70}")
        print(f"ERROR: {e}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")
    
    finally:
        simulation_app.close()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
