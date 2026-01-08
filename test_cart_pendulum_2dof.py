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

# Import standard Python libraries
from isaacsim import SimulationApp
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
import argparse
import os
import math
from pathlib import Path
from termcolor import colored

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Simulation device")
parser.add_argument(
    "--mode",
    type=str,
    choices=["scene-viz", "simulation", "test-simulation", "ee-trajectory", "cart-toward-manipulator", "coupled-motion"],
    default="coupled-motion",
    help="Mode: 'scene-viz' (static), 'simulation' (physics), 'test-simulation' (cart motion test), 'ee-trajectory' (move EE along cart direction), 'cart-toward-manipulator' (cart moves toward manipulator until they meet), 'cart-ee-aligned' (cart edge aligned with EE), 'coupled-motion' (manipulator moves cart-pendulum via joint)",
)
args, _ = parser.parse_known_args()

# STEP 1: Launch Isaac Sim application
simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
})

# STEP 2: Import Isaac Sim modules (after SimulationApp is created)
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.timeline
import omni.usd
import omni.kit.commands
from isaacsim.asset.importer.urdf import _urdf
from pxr import UsdGeom, UsdLux, Gf, UsdShade, Sdf
from isaacsim.core.experimental.objects import GroundPlane
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from omni.isaac.core import World
import torch
import warp as wp
import numpy as np

# Video recording support
try:
    import omni.kit.viewport.utility as viewport_utils
    from omni.kit.viewport.utility import get_active_viewport
    VIDEO_CAPTURE_AVAILABLE = True
except ImportError:
    VIDEO_CAPTURE_AVAILABLE = False
    print(colored("WARNING: Video capture not available", "blue"))

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
CART_PENDULUM_JOINT_DAMPING = [0.5, 0.01]  # Low damping for pendulum to allow swinging
CART_PENDULUM_JOINT_STIFFNESS = [625, 0.1]  # Zero stiffness for free motion
CART_PENDULUM_JOINT_FRICTION = [0.1, 0.0]  # No friction on pendulum

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
EE_CART_COUPLING_JOINT_STIFFNESS = 1000  # N·m/rad - stiffness for coupling joint
EE_CART_COUPLING_JOINT_DAMPING = 20     # N·m·s/rad - damping for coupling joint

# --- Scene Configuration ---
DISTANT_LIGHT_INTENSITY = 1000.0
DOME_LIGHT_INTENSITY = 300.0
DISTANT_LIGHT_ANGLE = 315.0
SIMULATION_MODE = args.mode
DEVICE = args.device

# --- Video Recording Configuration ---
VIDEO_RECORDING_ENABLED = True  # Set to True to enable video recording
VIDEO_OUTPUT_PATH = str(Path("videos/simulation_recording.mp4").absolute())
VIDEO_RESOLUTION = (1920, 1080)  # (width, height) - Full HD
VIDEO_FPS = 60  # Frames per second


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
    Abstract base class for robots.
    
    Provides common functionality for loading, configuring, and controlling robots.
    """
    
    def __init__(self, params: RobotParams):
        """Initialize robot with parameters."""
        self.params = params
        self.state: Optional[RobotState] = None  # Will be created after loading to stage
    
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
        
        Returns:
            tuple: (x, y, z) position in world coordinates, or None if not available
        """
        ee_path = f"{self.params.prim_path}/manipulator_link_2/manipulator_ee"
        return self.get_prim_world_position(ee_path)
    
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
        Compute analytical Jacobian matrix for 2-DOF planar manipulator.
        
        The Jacobian relates joint velocities to end-effector velocities:
        [vx, vy]^T = J * [θ1_dot, θ2_dot]^T
        
        For a 2-DOF planar manipulator:
        J = [ -L1*sin(θ1) - L2*sin(θ1+θ2),  -L2*sin(θ1+θ2) ]
            [  L1*cos(θ1) + L2*cos(θ1+θ2),   L2*cos(θ1+θ2) ]
        
        Args:
            theta1: Joint 1 angle in radians
            theta2: Joint 2 angle in radians
            
        Returns:
            np.ndarray: 2x2 Jacobian matrix, or None if link lengths unavailable
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
        Compute inverse kinematics for 2-DOF planar manipulator.
        
        The manipulator is planar in XY plane at constant Z height.
        Given target (x, y, z), compute joint angles (theta1, theta2).
        
        Args:
            target_x: Target X position
            target_y: Target Y position
            target_z: Target Z position (should match manipulator base Z)
            
        Returns:
            tuple: (theta1, theta2) in radians, or None if unreachable
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
        Compute differential inverse kinematics for 2-DOF planar manipulator.
        
        Uses velocity-level control: computes delta joint positions to move toward target.
        This is useful for continuous trajectory tracking and velocity control.
        
        Args:
            jacobian_end_effector: Jacobian matrix [batch, 2, num_dof] or [2, 2]
            current_position: Current EE position [batch, 2] or [2] (x, y in base frame)
            goal_position: Goal EE position [batch, 2] or [2] (x, y in base frame)
            method: IK method ("damped-least-squares", "pseudoinverse", "transpose", "singular-value-decomposition")
            method_cfg: Configuration dict with keys: scale, damping, min_singular_value
            
        Returns:
            torch.Tensor: Delta joint positions [batch, num_dof] or [num_dof]
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
        Compute forward kinematics for 2-DOF planar manipulator.
        
        Given joint angles (theta1, theta2), compute end-effector position in world coordinates.
        
        Args:
            theta1: Joint 1 angle in radians
            theta2: Joint 2 angle in radians
            
        Returns:
            tuple: (x, y, z) position in world coordinates, or None if link lengths unavailable
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
    """Manages the overall simulation scene."""
    
    def __init__(
        self,
        cart_pendulum_params: RobotParams,
        manipulator_params: RobotParams,
        lighting_params: LightingParams,
    ):
        """Initialize scene manager."""
        self.cart_pendulum_params = cart_pendulum_params
        self.manipulator_params = manipulator_params
        self.lighting_params = lighting_params
        
        # Create subsystem instances
        self.cart_pendulum = CartPendulum(cart_pendulum_params)
        self.manipulator = PlanarManipulator(manipulator_params)
        
        # Video recording state
        self.video_writer = None
        self.is_recording = False
    
    def start_video_recording(self, output_path: str = None, resolution: tuple = None, fps: int = 60):
        """Start recording video of the simulation.
        Args:
            output_path: Path to save video file (default: from global config)
            resolution: (width, height) tuple (default: from global config)
            fps: Frames per second (default: 60)
        """
        if not VIDEO_CAPTURE_AVAILABLE:
            print("ERROR: Video capture not available. Install required packages.")
            return False
        
        output_path = output_path or VIDEO_OUTPUT_PATH
        resolution = resolution or VIDEO_RESOLUTION
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"\n{'='*70}")
        print("STARTING VIDEO RECORDING")
        print(f"{'='*70}")
        print(f"  Output: {output_path}")
        print(f"  Resolution: {resolution[0]}x{resolution[1]}")
        print(f"  FPS: {fps}")
        print(f"{'='*70}\n")
        
        try:
            # Get active viewport
            viewport_api = get_active_viewport()
            if not viewport_api:
                print("ERROR: No active viewport found")
                return False
            
            # Use omni.kit.commands to start movie capture
            # Note: file_path should NOT include the .mp4 extension - it will be added automatically
            base_path = output_path.replace('.mp4', '').replace('.avi', '')
            
            result = omni.kit.commands.execute(
                "StartMovieCapture",
                file_path=base_path,
                resolution=resolution,
                fps=fps,
                viewport_api=viewport_api
            )
            
            if result:
                self.is_recording = True
                print(f"✓ Video recording started successfully")
                print(f"  File will be saved as: {base_path}.mp4")
                return True
            else:
                print(f"ERROR: StartMovieCapture command returned False")
                return False
                
        except Exception as e:
            print(f"ERROR: Failed to start video recording: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def stop_video_recording(self):
        """Stop video recording."""
        if not self.is_recording:
            return
        
        try:
            result = omni.kit.commands.execute("StopMovieCapture")
            self.is_recording = False
            print(f"\n{'='*70}")
            print("VIDEO RECORDING STOPPED")
            print(f"{'='*70}")
            if result:
                print("✓ Video file saved successfully")
            print(f"\n")
        except Exception as e:
            print(f"ERROR: Failed to stop video recording: {e}")
            import traceback
            traceback.print_exc()
    
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
        """Create a rigid revolute joint with compliance between EE and cart edge."""
        from pxr import UsdPhysics, PhysxSchema
        
        print("\nCreating EE-Cart coupling joint...")
        
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
        
        # Create fixed joint for rigid connection
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        
        # Set joint bodies
        joint.CreateBody0Rel().SetTargets([ee_path])
        joint.CreateBody1Rel().SetTargets([cart_path])
        
        # Set joint position in local frames
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))  # At EE origin
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(joint_pos_in_cart[0], joint_pos_in_cart[1], joint_pos_in_cart[2]))
        
        # Set joint rotation (align frames)
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))  # Identity
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))  # Identity
        
        print(f"✓ Created fixed joint: {joint_path}")
        print(f"  Type: Fixed (rigid connection)")
        print(f"  Joint connects EE to cart edge")
        print(f"  Cart acts as additional link in planar motion\n")
        
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

    def run_test_scene(self):
        """Run simulation in display mode."""
        print(f"\n{'='*70}")
        print("DISPLAY MODE - Static scene visualization")
        print("Press Ctrl+C or close window to exit")
        print(f"{'='*70}\n")
        
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
    
    def run_manipulator_ee_along_cart_direction(self):
        """
        Move manipulator end-effector along X direction (cart direction).
        Keeps Y and Z constant using inverse kinematics.
        """
        import numpy as np
        
        print(f"\n{'='*70}")
        print("RUNNING MANIPULATOR END-EFFECTOR ALONG CART DIRECTION")
        print(f"{'='*70}\n")
        
        # Reset world to initialize physics
        self.world.reset()
        
        # Initialize articulations
        self.cart_pendulum.initialize_articulation()
        self.manipulator.initialize_articulation()
        
        # Set joint properties
        self.cart_pendulum.set_joint_properties()
        self.manipulator.set_joint_properties()
        
        # Set initial positions for cart
        self.cart_pendulum.set_initial_joint_positions()
        
        # STEP 1: Set manipulator using initial joint positions
        print(f"\n{'='*70}")
        print("STEP 1: SETTING INITIAL JOINT POSITIONS")
        print(f"{'='*70}")
        
        theta1_init = MANIPULATOR_INITIAL_JOINT_POSITIONS[0]
        theta2_init = MANIPULATOR_INITIAL_JOINT_POSITIONS[1]
        
        print(f"Initial joint angles: theta1={theta1_init:.4f} rad ({math.degrees(theta1_init):.2f}°), theta2={theta2_init:.4f} rad ({math.degrees(theta2_init):.2f}°)")
        
        self.manipulator.state.robot.set_dof_positions([theta1_init, theta2_init])
        print(f"✓ Set joint positions from MANIPULATOR_INITIAL_JOINT_POSITIONS")
        
        # Step simulation to apply joint positions
        self.step_simulation(10)
        
        # STEP 2: Compute EE position using Forward Kinematics
        print(f"\n{'='*70}")
        print("STEP 2: FORWARD KINEMATICS VERIFICATION")
        print(f"{'='*70}")
        
        fk_ee_pos = self.manipulator.forward_kinematics(theta1_init, theta2_init)
        if fk_ee_pos is None:
            print("ERROR: Forward kinematics failed")
            return
        
        print(f"FK computed EE position: X={fk_ee_pos[0]:.4f}, Y={fk_ee_pos[1]:.4f}, Z={fk_ee_pos[2]:.4f}")
        
        # Get actual EE position from USD
        actual_ee_pos = self.manipulator.get_ee_world_position()
        if actual_ee_pos is None:
            print("ERROR: Could not get actual EE position")
            return
        
        print(f"Actual EE position (USD): X={actual_ee_pos[0]:.4f}, Y={actual_ee_pos[1]:.4f}, Z={actual_ee_pos[2]:.4f}")
        print(f"FK error: dX={abs(fk_ee_pos[0]-actual_ee_pos[0]):.4f}, dY={abs(fk_ee_pos[1]-actual_ee_pos[1]):.4f}, dZ={abs(fk_ee_pos[2]-actual_ee_pos[2]):.4f}")
        
        # STEP 3: Compute joint angles back using Inverse Kinematics
        print(f"\n{'='*70}")
        print("STEP 3: INVERSE KINEMATICS VERIFICATION (Analytical)")
        print(f"{'='*70}")
        
        ik_joint_angles = self.manipulator.inverse_kinematics(fk_ee_pos[0], fk_ee_pos[1], fk_ee_pos[2])
        if ik_joint_angles is None:
            print("ERROR: Analytical inverse kinematics failed")
            return
        
        theta1_ik, theta2_ik = ik_joint_angles
        print(f"Analytical IK joint angles: theta1={theta1_ik:.4f} rad ({math.degrees(theta1_ik):.2f}°), theta2={theta2_ik:.4f} rad ({math.degrees(theta2_ik):.2f}°)")
        print(f"Original joint angles:       theta1={theta1_init:.4f} rad ({math.degrees(theta1_init):.2f}°), theta2={theta2_init:.4f} rad ({math.degrees(theta2_init):.2f}°)")
        print(f"Joint angle error: dTheta1={abs(theta1_ik-theta1_init):.4f} rad ({math.degrees(abs(theta1_ik-theta1_init)):.2f}°), dTheta2={abs(theta2_ik-theta2_init):.4f} rad ({math.degrees(abs(theta2_ik-theta2_init)):.2f}°)")
        
        # Check if analytical FK/IK are consistent
        angle_error_threshold = 0.01  # 0.01 rad ≈ 0.57°
        if abs(theta1_ik - theta1_init) < angle_error_threshold and abs(theta2_ik - theta2_init) < angle_error_threshold:
            print(f"✓ Analytical FK and IK are CONSISTENT (error within {math.degrees(angle_error_threshold):.2f}°)")
        else:
            print(colored(f"WARNING: Analytical FK and IK have significant error (threshold: {math.degrees(angle_error_threshold):.2f}°)", "blue"))
        
        # STEP 4: Compute joint angles using Jacobian-based IK
        print(f"\n{'='*70}")
        print("STEP 4: INVERSE KINEMATICS VERIFICATION (Jacobian-based)")
        print(f"{'='*70}")
        
        ik_jacobian_angles = self.manipulator.inverse_kinematics_jacobian(fk_ee_pos[0], 
                                          fk_ee_pos[1], 
                                          fk_ee_pos[2],
                                          max_iterations=10)
        if ik_jacobian_angles is None:
            print("ERROR: Jacobian-based inverse kinematics failed")
        else:
            theta1_ik_jac, theta2_ik_jac = ik_jacobian_angles
            print(f"Jacobian IK joint angles: theta1={theta1_ik_jac:.4f} rad ({math.degrees(theta1_ik_jac):.2f}°), theta2={theta2_ik_jac:.4f} rad ({math.degrees(theta2_ik_jac):.2f}°)")
            print(f"Original joint angles:     theta1={theta1_init:.4f} rad ({math.degrees(theta1_init):.2f}°), theta2={theta2_init:.4f} rad ({math.degrees(theta2_init):.2f}°)")
            print(f"Joint angle error: dTheta1={abs(theta1_ik_jac-theta1_init):.4f} rad ({math.degrees(abs(theta1_ik_jac-theta1_init)):.2f}°), dTheta2={abs(theta2_ik_jac-theta2_init):.4f} rad ({math.degrees(abs(theta2_ik_jac-theta2_init)):.2f}°)")
            
            if abs(theta1_ik_jac - theta1_init) < angle_error_threshold and abs(theta2_ik_jac - theta2_init) < angle_error_threshold:
                print(f"✓ Jacobian-based IK is CONSISTENT (error within {math.degrees(angle_error_threshold):.2f}°)")
            else:
                print(colored(f"WARNING: Jacobian-based IK has significant error (threshold: {math.degrees(angle_error_threshold):.2f}°)", "blue"))
            
            # Compare analytical vs Jacobian IK
            print(f"\nComparison (Analytical vs Jacobian IK):")
            print(f"  dTheta1={abs(theta1_ik-theta1_ik_jac):.4f} rad ({math.degrees(abs(theta1_ik-theta1_ik_jac)):.2f}°)")
            print(f"  dTheta2={abs(theta2_ik-theta2_ik_jac):.4f} rad ({math.degrees(abs(theta2_ik-theta2_ik_jac)):.2f}°)")
        
        # STEP 5: Display Jacobian matrix
        print(f"\n{'='*70}")
        print("STEP 5: JACOBIAN MATRIX")
        print(f"{'='*70}")
        
        J = self.manipulator.compute_jacobian(theta1_init, theta2_init)
        if J is not None:
            print(f"Jacobian at theta1={math.degrees(theta1_init):.2f}°, theta2={math.degrees(theta2_init):.2f}°:")
            print(f"  J = [{J[0,0]:8.4f}  {J[0,1]:8.4f}]")
            print(f"      [{J[1,0]:8.4f}  {J[1,1]:8.4f}]")
            
            # Compute condition number
            cond_number = np.linalg.cond(J)
            print(f"\nJacobian condition number: {cond_number:.4f}")
            if cond_number < 10:
                print(f"  ✓ Well-conditioned (good for inverse kinematics)")
            elif cond_number < 100:
                print(f"  ⚠ Moderately conditioned")
            else:
                print(f"  ⚠ Poorly conditioned (near singularity)")
        
        print(f"{'='*70}\n")
        
        # STEP 6: Differential IK verification
        print(f"\n{'='*70}")
        print("STEP 6: DIFFERENTIAL INVERSE KINEMATICS")
        print(f"{'='*70}")
        
        # Convert current position to base frame for differential IK
        fk_ee_pos_base = self.manipulator.transform_point_world_to_base(fk_ee_pos[0], fk_ee_pos[1], fk_ee_pos[2])
        if fk_ee_pos_base is not None:
            # Compute current EE position in base frame using FK method
            current_ee_pos_base = self.manipulator.forward_kinematics_base_frame(theta1_init, theta2_init)
            if current_ee_pos_base is None:
                print("ERROR: Could not compute base-frame FK")
                return
            
            # Convert to torch tensors
            current_pos = torch.tensor([current_ee_pos_base[0], current_ee_pos_base[1]], dtype=torch.float32)
            goal_pos = torch.tensor([fk_ee_pos_base[0], fk_ee_pos_base[1]], dtype=torch.float32)
            
            # Compute Jacobian tensor
            J_np = self.manipulator.compute_jacobian(theta1_init, theta2_init)
            if J_np is not None:
                J_torch = torch.tensor(J_np, dtype=torch.float32)
                
                # Test all differential IK methods
                methods = ["damped-least-squares", "pseudoinverse", "transpose", "singular-value-decomposition"]
                
                print(f"Testing differential IK methods (should give near-zero delta since current=goal):\n")
                for method in methods:
                    delta_theta = self.manipulator.inverse_kinematics_differential(
                        jacobian_end_effector=J_torch,
                        current_position=current_pos,
                        goal_position=goal_pos,
                        method=method,
                        method_cfg={"scale": 1.0, "damping": 0.05, "min_singular_value": 1e-5}
                    )
                    
                    delta_norm = torch.norm(delta_theta).item()
                    # Squeeze to handle batch dimension [1, 2] -> [2]
                    delta_squeezed = delta_theta.squeeze()
                    print(f"  {method:30s}: delta_norm = {delta_norm:.6f} rad")
                    print(f"    Δθ1={delta_squeezed[0].item():+.6f} rad, Δθ2={delta_squeezed[1].item():+.6f} rad")
                
                # Test with a small offset to show it works
                print(f"\nTesting with small offset (0.1m in X direction):")
                goal_pos_offset = torch.tensor([fk_ee_pos_base[0] + 0.1, fk_ee_pos_base[1]], dtype=torch.float32)
                
                delta_theta = self.manipulator.inverse_kinematics_differential(
                    jacobian_end_effector=J_torch,
                    current_position=current_pos,
                    goal_position=goal_pos_offset,
                    method="damped-least-squares",
                    method_cfg={"scale": 1.0, "damping": 0.05}
                )
                
                # Squeeze to handle batch dimension [1, 2] -> [2]
                delta_squeezed = delta_theta.squeeze()
                print(f"  Position error: +0.1m in X")
                print(f"  Delta: Δθ1={delta_squeezed[0].item():+.6f} rad ({math.degrees(delta_squeezed[0].item()):+.3f}°)")
                print(f"        Δθ2={delta_squeezed[1].item():+.6f} rad ({math.degrees(delta_squeezed[1].item()):+.3f}°)")
                print(f"  ✓ Differential IK computation successful")
            else:
                print("ERROR: Could not compute Jacobian")
        else:
            print("ERROR: Could not transform to base frame")
        
        print(f"{'='*70}\n")
        
        # STEP 7: Use FK result as initial EE position for trajectory
        initial_ee_pos = fk_ee_pos
        print(f"Using FK-computed EE position as trajectory starting point: X={initial_ee_pos[0]:.4f}, Y={initial_ee_pos[1]:.4f}, Z={initial_ee_pos[2]:.4f}")
        print(f"{'='*70}\n")
        
        # Generate trajectory waypoints
        print(f"\n{'='*70}")
        print("GENERATING TRAJECTORY")
        print(f"{'='*70}")
        
        target_y = initial_ee_pos[1]  # Keep Y constant
        target_z = initial_ee_pos[2]  # Keep Z constant
        x_start = initial_ee_pos[0]
        x_range = 1.5  # Move 0.75 meters along X (toward YZ plane at origin)
        duration = 5.0  # 5 seconds
        dt = 1.0 / 200.0  # 200 FPS
        
        num_waypoints = int(duration / dt)
        trajectory = []
        
        for i in range(num_waypoints):
            t = i / (num_waypoints - 1)
            # Sinusoidal motion along X (toward/away from YZ plane)
            # Negate sine so X increases (becomes less negative, closer to origin)
            target_x = x_start + x_range * np.sin(2 * np.pi/2 * t)
            
            # Compute inverse kinematics
            joint_angles = self.manipulator.inverse_kinematics(target_x, target_y, target_z)
            
            if joint_angles is not None:
                trajectory.append((target_x, target_y, target_z, joint_angles[0], joint_angles[1]))
            else:
                print(colored(f"WARNING: IK failed for waypoint {i}: X={target_x:.4f}", "blue"))
                # Skip this waypoint if IK fails
                continue
        
        print(f"✓ Generated {len(trajectory)} waypoints")
        print(f"  X range: [{x_start:.4f}, {x_start - x_range:.4f}] m")
        print(f"  Y constant: {target_y:.4f} m")
        print(f"  Z constant: {target_z:.4f} m")
        print(f"{'='*70}\n")
        
        # Execute trajectory
        print(f"\n{'='*70}")
        print("EXECUTING TRAJECTORY")
        print(f"{'='*70}\n")
        
        frame_count = 0
        waypoint_idx = 0
        
        while simulation_app.is_running() and waypoint_idx < len(trajectory):
            # Get current waypoint
            if waypoint_idx < len(trajectory):
                target_x, target_y, target_z, theta1, theta2 = trajectory[waypoint_idx]
                
                # Set joint positions using Articulation API via helper method
                self.manipulator.set_joint_positions([theta1, theta2])
                
                waypoint_idx += 1
            
            # Step simulation
            self.world.step(render=True)
            simulation_app.update()
            
            # Print progress every 10 frames
            if frame_count % 10 == 0:
                ee_position = self.manipulator.get_ee_world_position()
                if ee_position:
                    print(f"Frame {frame_count:6d} | Waypoint {waypoint_idx:4d}/{len(trajectory)} | EE: X={ee_position[0]:8.4f}  Y={ee_position[1]:8.4f}  Z={ee_position[2]:8.4f}")
            
            frame_count += 1
        
        print(f"\n{'='*70}")
        print("TRAJECTORY COMPLETE")
        print(f"{'='*70}\n")
        
        # Keep running after trajectory
        while simulation_app.is_running():
            self.world.step(render=True)
            simulation_app.update()
    
    def run_cart_toward_manipulator(self):
        """
        Move cart toward manipulator until cart edge coincides with manipulator EE origin.
        
        The cart moves along X-axis toward the manipulator EE position.
        Cart stops when its edge (not center) reaches the EE position.
        """
        print(f"\n{'='*70}")
        print("CART EDGE MOVES TOWARD MANIPULATOR EE")
        print(f"{'='*70}\n")
        
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
        
        # Set manipulator initial position
        theta1_init = MANIPULATOR_INITIAL_JOINT_POSITIONS[0]
        theta2_init = MANIPULATOR_INITIAL_JOINT_POSITIONS[1]
        self.manipulator.state.robot.set_dof_positions([theta1_init, theta2_init])
        print(f"Manipulator initial angles: theta1={math.degrees(theta1_init):.2f}°, theta2={math.degrees(theta2_init):.2f}°")
        
        # Set cart initial position (far from manipulator)
        cart_initial_pos = CART_PENDULUM_INITIAL_JOINT_POSITIONS
        self.cart_pendulum.state.robot.set_dof_positions(cart_initial_pos)
        print(f"Cart initial position: X={cart_initial_pos[0]:.2f}m")
        
        # Step simulation to apply initial positions
        self.step_simulation(10)
        
        # Get initial positions
        initial_cart_pos = self.cart_pendulum.get_cart_world_position()
        initial_ee_pos = self.manipulator.get_ee_world_position()
        
        # Cart dimensions (from URDF: box size="0.3 0.2 0.15")
        cart_length_x = 0.3  # meters
        cart_half_length = cart_length_x / 2.0  # 0.15m
        
        if not initial_cart_pos or not initial_ee_pos:
            print("ERROR: Could not get initial positions")
            return
        
        # Calculate cart edge position (left edge, which faces the manipulator)
        initial_cart_edge_x = initial_cart_pos[0] - cart_half_length
        
        print(f"\nInitial cart center position: X={initial_cart_pos[0]:.4f}, Y={initial_cart_pos[1]:.4f}, Z={initial_cart_pos[2]:.4f}")
        print(f"Initial cart edge position:   X={initial_cart_edge_x:.4f}")
        print(f"Initial EE position:          X={initial_ee_pos[0]:.4f}, Y={initial_ee_pos[1]:.4f}, Z={initial_ee_pos[2]:.4f}")
        print(f"Initial distance (edge to EE): {abs(initial_cart_edge_x - initial_ee_pos[0]):.4f}m")
        
        # Motion parameters
        cart_speed = 1  # meters per second
        dt = 1.0 / 60.0  # 60 FPS
        cart_step = cart_speed * dt  # meters per frame
        distance_threshold = 0.00005  # Stop when within 0.05mm
        
        # Target: EE position
        target_ee_x = initial_ee_pos[0]
        
        print(f"\n{'='*70}")
        print("MOVING CART EDGE TOWARD MANIPULATOR EE")
        print(f"Cart speed: {cart_speed:.2f} m/s")
        print(f"Stop threshold: {distance_threshold:.5f} m")
        print(f"Target EE X position: {target_ee_x:.4f} m")
        print(f"{'='*70}\n")
        
        frame_count = 0
        converged = False
        
        while simulation_app.is_running() and not converged:
            # Get current cart center position
            current_cart_center = self.cart_pendulum.get_cart_world_position()
            
            if not current_cart_center:
                print("ERROR: Lost position tracking")
                break
            
            # Calculate current cart edge position (left edge)
            current_cart_edge_x = current_cart_center[0] - cart_half_length
            
            # Compute distance between cart edge and EE
            distance = abs(current_cart_edge_x - target_ee_x)
            
            # Check if converged
            if distance < distance_threshold:
                converged = True
                print(f"\n{'='*70}")
                print("CONVERGENCE ACHIEVED!")
                print(f"{'='*70}")
                print(f"Final distance: {distance:.6f}m (threshold: {distance_threshold:.5f}m)")
                print(f"Cart center position: X={current_cart_center[0]:.6f}")
                print(f"Cart edge position:   X={current_cart_edge_x:.6f}")
                print(f"EE position:          X={target_ee_x:.6f}")
                print(f"{'='*70}\n")
                break
            
            # Move cart toward manipulator
            # We need the cart edge to reach target_ee_x
            # Cart edge = cart_center - cart_half_length
            # So: cart_center = target_ee_x + cart_half_length
            current_cart_x = self.cart_pendulum.get_joint_positions()[0]
            
            # Determine direction: if cart edge is to the right of EE, move left (decrease X)
            if current_cart_edge_x > target_ee_x:
                new_cart_x = current_cart_x + cart_step
            else:
                new_cart_x = current_cart_x - cart_step
            
            # Update cart position
            self.cart_pendulum.set_joint_positions([new_cart_x])
            
            # Keep manipulator fixed at initial position
            self.manipulator.set_joint_positions([theta1_init, theta2_init])
            
            # Step simulation
            self.world.step(render=True)
            simulation_app.update()
            
            # Print progress every 30 frames (~0.5 seconds) as a table
            if frame_count % 30 == 0:
                # Get pendulum bob position
                pendulum_path = f"{self.cart_pendulum.params.prim_path}/pendulum"
                pend_pos = self.cart_pendulum.get_prim_world_position(pendulum_path)
                
                if frame_count == 0:
                    print(f"{'Frame':>8} | {'Error (m)':>10} | {'Cart Edge X':>12} | {'Target EE X':>13} | {'Pend X':>10} | {'Pend Y':>10} | {'Pend Z':>10}")
                    print(f"{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*13}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
                
                if pend_pos:
                    pend_str = f"{pend_pos[0]:10.4f} | {pend_pos[1]:10.4f} | {pend_pos[2]:10.4f}"
                else:
                    pend_str = f"{'N/A':>10} | {'N/A':>10} | {'N/A':>10}"
                print(f"{frame_count:8d} | {distance:10.6f} | {current_cart_edge_x:12.6f} | {target_ee_x:13.6f} | {pend_str}")
            
            frame_count += 1
            
            # Safety limit
            if frame_count > 10000:
                print(colored("WARNING: Reached maximum iterations (10000 frames)", "blue"))
                break
        
        # Keep running after convergence
        print("\nSimulation continues. Press Ctrl+C or close window to exit.\n")
        while simulation_app.is_running():
            self.world.step(render=True)
            simulation_app.update()
    
    

    def run_coupled_motion(self):
        """
        Demonstrate manipulator moving cart-pendulum system via coupling joint.
        
        The manipulator moves through joint angles, and the coupling joint
        causes the cart to follow, swinging the pendulum.
        """
        print(f"\n{'='*70}")
        print("COUPLED MOTION: MANIPULATOR MOVES CART-PENDULUM")
        print(f"{'='*70}\n")
        
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
        
        # Start video recording if enabled
        if VIDEO_RECORDING_ENABLED:
            self.start_video_recording()
        
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
            
            # Step simulation
            self.world.step(render=True)
            simulation_app.update()
            
            # Print progress every 30 frames (~0.5 seconds)
            if frame_count % 30 == 0:
                time_s = frame_count * dt
                
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
        
        # Stop video recording if it was started
        if VIDEO_RECORDING_ENABLED:
            self.stop_video_recording()
        
        print(f"\n{'='*70}")
        print("TRAJECTORY COMPLETE")
        print(f"{'='*70}\n")
        
        # Keep running
        print("\nSimulation continues. Press Ctrl+C or close window to exit.\n")
        while simulation_app.is_running():
            self.world.step(render=True)
            simulation_app.update()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution flow."""
    print("=" * 70)
    print("Cart-Pendulum with 2-DOF Planar Manipulator")
    print("=" * 70)
    
    try:
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
        scene.initialize_stage()
        
        # Run appropriate mode
        if SIMULATION_MODE == "scene-viz":
            scene.run_test_scene()
        elif SIMULATION_MODE == "ee-trajectory":
            scene.run_manipulator_ee_along_cart_direction()
        elif SIMULATION_MODE == "cart-toward-manipulator":
            scene.run_cart_toward_manipulator()
        elif SIMULATION_MODE == "cart-ee-aligned":
            scene.set_cart_to_ee_and_joint_positions()
        elif SIMULATION_MODE == "coupled-motion":
            scene.run_coupled_motion()
        else:
            scene.run_simulation()
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
    
    finally:
        simulation_app.close()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
