"""
Franka Panda Differential IK Control with Target Sphere

Demonstrates:
1. Differential inverse kinematics using PyTorch and Jacobian matrices
2. Target tracking with visual sphere
3. Multiple trials with random target positions
4. Warp-PyTorch interoperability

Organized using OOP structure from test_manipulator_ik.py template.
"""

# Import standard Python libraries
from isaacsim import SimulationApp
from dataclasses import dataclass
from typing import Optional
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Simulation device")
parser.add_argument(
    "--ik-method",
    type=str,
    choices=["singular-value-decomposition", "pseudoinverse", "transpose", "damped-least-squares"],
    default="damped-least-squares",
    help="Differential inverse kinematics method",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["scene-viz", "robot-ik", "plate-ball"],
    default="robot-ik",
    help="Simulation mode: 'scene-viz' (just show scene), 'robot-ik' (run robot IK trials), 'plate-ball' (run plate and ball physics)",
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
import torch
import warp as wp
from pxr import UsdGeom, UsdLux, Gf
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import Sphere, GroundPlane
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path


# ============================================================================
# DIFFERENTIAL IK HELPER FUNCTIONS (PyTorch-based)
# ============================================================================

# ============================================================================
# DIFFERENTIAL IK HELPER FUNCTIONS (PyTorch-based)
# ============================================================================

def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions (WXYZ format)."""
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion conjugate (WXYZ format)."""
    return torch.cat((q[:, :1], -q[:, 1:]), dim=-1)


def differential_inverse_kinematics(
    jacobian_end_effector: torch.Tensor,
    current_position: torch.Tensor,
    current_orientation: torch.Tensor,
    goal_position: torch.Tensor,
    goal_orientation: torch.Tensor | None = None,
    method: str = "damped-least-squares",
    method_cfg: dict[str, float] = {"scale": 1.0, "damping": 0.05, "min_singular_value": 1e-5},
) -> torch.Tensor:
    """
    Compute differential inverse kinematics to reach a goal pose.
    
    Args:
        jacobian_end_effector: Jacobian matrix for end-effector [batch, 6, num_dof]
        current_position: Current EE position [batch, 3]
        current_orientation: Current EE orientation quaternion [batch, 4] (WXYZ)
        goal_position: Goal EE position [batch, 3]
        goal_orientation: Goal EE orientation quaternion [batch, 4] (WXYZ), or None to keep current
        method: IK method ("singular-value-decomposition", "pseudoinverse", "transpose", "damped-least-squares")
        method_cfg: Configuration dictionary with scale, damping, min_singular_value
    
    Returns:
        Delta DOF positions [batch, num_dof]
    """
    scale = method_cfg.get("scale", 1.0)
    # Compute velocity error
    goal_orientation = current_orientation if goal_orientation is None else goal_orientation
    q = quat_mul(goal_orientation, quat_conjugate(current_orientation))
    error = torch.cat([goal_position - current_position, q[:, 1:] * torch.sign(q[:, [0]])], dim=-1).unsqueeze(-1)
    # Compute delta DOF positions
    # - Adaptive Singular Value Decomposition (SVD)
    if method == "singular-value-decomposition":
        min_singular_value = method_cfg.get("min_singular_value", 1e-5)
        U, S, Vh = torch.linalg.svd(jacobian_end_effector)
        inv_s = torch.where(S > min_singular_value, 1.0 / S, torch.zeros_like(S))
        pseudoinverse = torch.transpose(Vh, 1, 2)[:, :, :6] @ torch.diag_embed(inv_s) @ torch.transpose(U, 1, 2)
        return (scale * pseudoinverse @ error).squeeze(-1)
    # - Moore-Penrose pseudoinverse
    elif method == "pseudoinverse":
        pseudoinverse = torch.linalg.pinv(jacobian_end_effector)
        return (scale * pseudoinverse @ error).squeeze(-1)
    # - Transpose of matrix
    elif method == "transpose":
        transpose = torch.transpose(jacobian_end_effector, 1, 2)
        return (scale * transpose @ error).squeeze(-1)
    # - Damped Least-Squares
    elif method == "damped-least-squares":
        damping = method_cfg.get("damping", 0.05)
        transpose = torch.transpose(jacobian_end_effector, 1, 2)
        lmbda = torch.eye(jacobian_end_effector.shape[1], device=jacobian_end_effector.device) * (damping**2)
        return (scale * transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ error).squeeze(-1)
    else:
        raise ValueError(f"Invalid IK method: {method}")


# ============================================================================
# USER CONFIGURATION - EDIT THESE PARAMETERS
# ============================================================================

# --- Robot Configuration ---
ROBOT_USD_PATH = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"  # Cloud-hosted USD
ROBOT_PATH = "/World/robot"
ROBOT_POSITION = (0.0, 1.2, 0.0)  # (x, y, z) - positioned away from plate/ball area
ROBOT_BASE_ROTATION = 180  # Rotation in degrees around Z-axis
ROBOT_SCALE = 1.0  # Scale factor (1.0 = normal size, 2.0 = double size, 0.5 = half size)
ROBOT_VARIANTS = [("Gripper", "AlternateFinger"), ("Mesh", "Performance")]
DEFAULT_DOF_POSITIONS = [0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.0, 0.0]
EE_LINK_NAME = "panda_hand"
NUM_ARM_DOF = 7  # Franka Panda has 7 arm DOFs (+ 2 gripper DOFs)

# --- Target Sphere Configuration ---
SPHERE_PATH = "/World/sphere"
SPHERE_RADIUS = 0.05
SPHERE_COLOR = [1.0, 0.0, 0.0]  # Red
TARGET_REGION_CENTER = [0.5, 0.0, 0.5]  # Center of random target region
TARGET_REGION_SIZE = 0.2  # Cube side length for random targets

# --- Horizontal Line Trajectory Configuration ---
LINE_START_POINT = [0.1, 0.7, 0.5]  # Starting point - in front of robot, away from plate
LINE_END_POINT   = [0.7, 0.7, 0.5]  # Ending point - horizontal line in robot workspace
USE_LINE_TRAJECTORY = True  # True: follow line, False: random targets
TRAJECTORY_DURATION = 10.0  # Total duration (seconds) to complete the line trajectory
TRAJECTORY_TRACKING_MODE = True  # True: continuous tracking, False: discrete waypoints

# --- Simulation Configuration ---
NUM_TRIALS = 1  # Number of random target trials
STEPS_PER_TRIAL = 200  # IK steps per trial
DEVICE = args.device  # "cpu" or "cuda"
IK_METHOD = args.ik_method
SIMULATION_MODE = args.mode  # "display", "robot-ik", or "plate-ball"

# --- Scene Configuration ---
DISTANT_LIGHT_INTENSITY = 1000.0
DOME_LIGHT_INTENSITY = 300.0
DISTANT_LIGHT_ANGLE = 315.0

# --- Plate and Ball Configuration ---
from pathlib import Path
PLATE_USD_FILE = str(Path("model/plate_dips/part_dips_coarse_rot.usd").absolute())
PLATE_POSITION = (0.5, 0.5, 0.1)
PLATE_SCALE_FACTOR = 0.25  # Scaling factor for plate (1.0 = normal size, 2.0 = double size, 0.5 = half size)
PLATE_SCALE = tuple(s * PLATE_SCALE_FACTOR for s in (5.0, 5.0, 5.0))

PLATE_ROTATION_X = 0
PLATE_COLOR = (0.0, 1.0, 0.0)  # Green
PLATE_STATIC_FRICTION = 0.6
PLATE_DYNAMIC_FRICTION = 0.5
PLATE_RESTITUTION = 0.1
PLATE_MECHANICS = "kinematic"  # "static", "kinematic", or "dynamic"

BALL_POSITION = (0.5, 0.5, 2.0)
BALL_RADIUS = 0.2
BALL_SCALE_FACTOR = 0.25  # Scaling factor for ball (1.0 = normal size, 2.0 = double size, 0.5 = half size)
BALL_SCALE = tuple(s * BALL_SCALE_FACTOR for s in (1.0, 1.0, 1.0))
BALL_COLOR = (1.0, 0.0, 0.0)  # Red
BALL_MASS = 0.5
BALL_STATIC_FRICTION = 0.6
BALL_DYNAMIC_FRICTION = 0.5
BALL_RESTITUTION = 0.3
ENABLE_CCD = True
CONTACT_OFFSET = 0.02
REST_OFFSET = 0.0

PLATE_MOTION_ENABLED = True
PLATE_MOTION_AMPLITUDE = 1.0
PLATE_MOTION_FREQUENCY = 0.2
PLATE_MOTION_AXIS = 0  # 0=X, 1=Y, 2=Z
SETTLE_TIME = 3.0


# ============================================================================
# PARAMETER CLASSES (Dataclasses)
# ============================================================================

@dataclass
class RobotParams:
    """Parameters for robot configuration."""
    usd_path: str
    prim_path: str
    position: tuple[float, float, float]
    rotation_z: float  # Rotation in degrees around Z-axis
    scale: float  # Scale factor
    variants: list
    default_dof_positions: list
    ee_link_name: str
    num_arm_dof: int = 7


@dataclass
class TargetParams:
    """Parameters for target sphere."""
    prim_path: str
    radius: float
    color: list
    region_center: list
    region_size: float


@dataclass
class SimulationParams:
    """Parameters controlling simulation behavior."""
    num_trials: int = 10
    steps_per_trial: int = 100
    device: str = "cpu"
    ik_method: str = "damped-least-squares"


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
    end_effector: RigidPrim
    ee_link_index: int
    num_dof: int
    dof_names: list[str]


@dataclass
class PhysicsMaterialParams:
    """Parameters controlling how surfaces interact."""
    static_friction: float
    dynamic_friction: float
    restitution: float


@dataclass
class BallPlateVisualMaterialParams:
    """Parameters controlling visual appearance."""
    color: tuple[float, float, float]
    roughness: float = 0.4
    metallic: float = 0.0


@dataclass
class PhysicsBodyParams:
    """Parameters for physics body."""
    is_dynamic: bool = True
    enable_ccd: bool = True
    mass: Optional[float] = None
    contact_offset: float = 0.02
    rest_offset: float = 0.0


@dataclass
class CollisionParams:
    """Parameters for collision detection."""
    approximation: str = "none"
    enable_collision: bool = True


@dataclass
class TransformParams:
    """Parameters for positioning and scaling."""
    position: tuple[float, float, float]
    scale: tuple[float, float, float]
    rotateX: Optional[float] = None


@dataclass
class PlateParams:
    """Parameters for plate configuration."""
    prim_path: str
    model_file: str
    transform: TransformParams
    physics_material: PhysicsMaterialParams
    visual_material: BallPlateVisualMaterialParams
    physics_body: PhysicsBodyParams
    collision: CollisionParams
    mechanics_mode: str


@dataclass
class BallParams:
    """Parameters for ball configuration."""
    prim_path: str
    radius: float
    transform: TransformParams
    physics_material: PhysicsMaterialParams
    visual_material: BallPlateVisualMaterialParams
    physics_body: PhysicsBodyParams


# ============================================================================
# FRANKA ROBOT CLASS
# ============================================================================

class Franka:
    """
    Franka Panda manipulator robot model.
    
    Manages:
    - Robot articulation and end-effector state
    - Joint positions and end-effector pose
    - DOF information
    """
    
    def __init__(self, robot_params: RobotParams):
        """Initialize Franka robot."""
        self.params = robot_params
        self.state: Optional[RobotState] = None
    
    def initialize(self):
        """Initialize robot articulation and end-effector (call after timeline is playing)."""
        print("\nInitializing Franka robot...")
        
        # Create articulation and end-effector
        robot = Articulation(self.params.prim_path)
        end_effector = RigidPrim(f"{self.params.prim_path}/{self.params.ee_link_name}")

        assert end_effector.get_world_poses()  # Ensure EE is initialized

        ee_link_index = robot.get_link_indices(self.params.ee_link_name).list()[0]
        
        # Set default state
        robot.set_default_state(dof_positions=self.params.default_dof_positions)
        print(f"✓ Robot initialized with {self.params.num_arm_dof} arm DOFs")
        print(f"  End-effector link: {self.params.ee_link_name} (index: {ee_link_index})")
        
        # Discover DOF information
        num_dof = robot.num_dofs
        dof_names = robot.dof_names
        print(f"✓ Robot has {num_dof} DOF:")
        for i, name in enumerate(dof_names):
            print(f"  Joint {i}: {name}")
        
        # Store runtime state
        self.state = RobotState(
            robot=robot,
            end_effector=end_effector,
            ee_link_index=ee_link_index,
            num_dof=num_dof,
            dof_names=dof_names
        )
        print("✓ Robot state initialized")
    
    def reset_to_default(self):
        """Reset robot to default state."""
        if self.state:
            self.state.robot.reset_to_default_state()
    
    def get_ee_pose(self):
        """Get current end-effector world pose."""
        if self.state is None:
            return None, None
        return self.state.end_effector.get_world_poses()
    
    def get_dof_positions(self):
        """Get current DOF positions."""
        if self.state is None:
            return None
        return self.state.robot.get_dof_positions()
    
    def get_jacobian_matrices(self):
        """Get Jacobian matrices for all links."""
        if self.state is None:
            return None
        return self.state.robot.get_jacobian_matrices()
    
    def set_dof_position_targets(self, dof_positions, dof_indices: list = None):
        """Set DOF position targets."""
        if self.state:
            self.state.robot.set_dof_position_targets(dof_positions, dof_indices=dof_indices)


# ============================================================================
# FRANKA CONTROLLER CLASS
# ============================================================================

class FrankaController:
    """
    Differential IK controller for Franka Panda manipulator.
    
    Implements:
    - Differential inverse kinematics
    - Target tracking
    - Multiple IK methods (SVD, pseudoinverse, transpose, damped least-squares)
    """
    
    def __init__(self, robot: Franka, device: str = "cpu"):
        """Initialize controller with Franka robot instance."""
        self.robot = robot
        self.device = device
    
    def compute_differential_ik(
        self,
        target_position: torch.Tensor,
        method: str = "damped-least-squares",
        method_cfg: dict = None
    ) -> torch.Tensor:
        """
        Compute differential IK to reach target position.
        
        Args:
            target_position: Target EE position [3] (torch tensor)
            method: IK method
            method_cfg: Method configuration
        
        Returns:
            Delta DOF positions for arm joints
        """
        if method_cfg is None:
            method_cfg = {"scale": 1.0, "damping": 0.05, "min_singular_value": 1e-5}
        
        if self.robot.state is None:
            return None
        
        # Get current state
        current_ee_pos, current_ee_ori = self.robot.get_ee_pose()
        if current_ee_pos is None:
            return None
        
        # Convert current state to torch tensors
        current_dof_positions = wp.to_torch(self.robot.get_dof_positions())
        current_ee_position = wp.to_torch(current_ee_pos)
        current_ee_orientation = wp.to_torch(current_ee_ori)
        
        # Get Jacobian matrix
        jacobian_matrices = wp.to_torch(self.robot.get_jacobian_matrices())
        # Extract Jacobian for end-effector and arm DOFs only
        jacobian_ee = jacobian_matrices[:, self.robot.state.ee_link_index - 1, :, :self.robot.params.num_arm_dof]
        
        # Compute delta DOF positions
        delta_dof = differential_inverse_kinematics(
            jacobian_end_effector=jacobian_ee,
            current_position=current_ee_position,
            current_orientation=current_ee_orientation,
            goal_position=target_position.unsqueeze(0) if target_position.dim() == 1 else target_position,
            goal_orientation=None,
            method=method,
            method_cfg=method_cfg
        )
        
        return delta_dof
    
    def apply_joint_command(self, dof_positions: torch.Tensor, dof_indices: list = None):
        """Apply joint position command to robot."""
        dof_targets = wp.from_torch(dof_positions)
        self.robot.set_dof_position_targets(dof_targets, dof_indices=dof_indices)


# ============================================================================
# PLATE CLASS
# ============================================================================

class Plate:
    """
    Represents a plate object imported from USD file.
    
    Manages:
    - USD model import
    - Transform (position, rotation, scale)
    - Visual materials (color, appearance)
    - Physics (collision, dynamics)
    - Physics materials (friction, bounce)
    """
    
    def __init__(self, plate_params: PlateParams):
        """Initialize Plate with parameters."""
        self.params = plate_params
        self.prim_path = plate_params.prim_path
        self.model_file = plate_params.model_file
        self.transform_params = plate_params.transform
        self.physics_material_params = plate_params.physics_material
        self.visual_material_params = plate_params.visual_material
        self.physics_body_params = plate_params.physics_body
        self.collision_params = plate_params.collision
        self.mechanics_mode = plate_params.mechanics_mode
        self._prim = None
    
    def get_prim(self):
        """Get USD prim with caching."""
        if self._prim is None:
            stage = omni.usd.get_context().get_stage()
            self._prim = stage.GetPrimAtPath(self.prim_path)
        return self._prim
    
    def import_model(self):
        """Import the USD model file into the scene."""
        print(f"Importing plate model from: {self.model_file}")
        
        # Import using reference command
        usd_context = omni.usd.get_context()
        omni.kit.commands.execute(
            'CreateReferenceCommand',
            usd_context=usd_context,
            path_to=self.prim_path,
            asset_path=self.model_file,
            instanceable=False
        )
        print(f"✓ Plate model imported at {self.prim_path}")
    
    def apply_transform(self):
        """Apply position, rotation, and scale transforms."""
        prim = self.get_prim()
        xformable = UsdGeom.Xformable(prim)
        
        # Clear existing transforms
        xformable.ClearXformOpOrder()
        
        # Apply translate
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*self.transform_params.position))
        
        # Apply rotation if specified
        if self.transform_params.rotateX is not None:
            rotate_op = xformable.AddRotateXOp()
            rotate_op.Set(self.transform_params.rotateX)
        
        # Apply scale
        scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3f(*self.transform_params.scale))
    
    def create_visual_material(self):
        """Create and apply PBR visual material."""
        from pxr import UsdShade, Sdf
        
        stage = omni.usd.get_context().get_stage()
        
        # Create material
        material_path = f"{self.prim_path}/Looks/Material"
        material = UsdShade.Material.Define(stage, material_path)
        
        # Create PBR shader
        shader_path = f"{material_path}/Shader"
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        # Set shader parameters
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*self.visual_material_params.color)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            self.visual_material_params.roughness
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            self.visual_material_params.metallic
        )
        
        # Connect shader to material
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.GetOutput("surface"))
        
        # Bind to all meshes
        self._bind_material_to_meshes(self.get_prim(), material)
    
    def _bind_material_to_meshes(self, prim, material):
        """Recursively bind material to all mesh children."""
        from pxr import UsdShade
        
        if prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI(prim).Bind(material)
        
        for child in prim.GetChildren():
            self._bind_material_to_meshes(child, material)
    
    def apply_physics_material(self):
        """Apply physics material (friction and restitution)."""
        from pxr import UsdPhysics, Sdf
        
        stage = omni.usd.get_context().get_stage()
        material_path = f"{self.prim_path}/PhysicsMaterial"
        physics_material = UsdPhysics.MaterialAPI.Apply(
            stage.DefinePrim(material_path, "Material")
        )
        
        physics_material.CreateStaticFrictionAttr(
            self.physics_material_params.static_friction
        )
        physics_material.CreateDynamicFrictionAttr(
            self.physics_material_params.dynamic_friction
        )
        physics_material.CreateRestitutionAttr(
            self.physics_material_params.restitution
        )
    
    def apply_physics(self):
        """Apply physics properties using mesh collision."""
        from pxr import UsdPhysics, PhysxSchema, Sdf
        
        prim = self.get_prim()
        
        # Apply rigid body API
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
            
            # Set kinematic based on mechanics mode
            if self.mechanics_mode == 'static':
                rigid_body.CreateRigidBodyEnabledAttr(False)
            elif self.mechanics_mode == 'kinematic':
                rigid_body.CreateKinematicEnabledAttr(True)
            else:  # dynamic
                rigid_body.CreateRigidBodyEnabledAttr(True)
                rigid_body.CreateKinematicEnabledAttr(False)
            
            # Apply PhysX API
            physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            
            # Lock all angular axes to prevent rotation
            physx_rigid_body.CreateAngularDampingAttr(1000.0)
            prim.CreateAttribute("physx:lockFlags", Sdf.ValueTypeNames.Int).Set(56)
        
        # Recursively apply collision to all meshes
        self._apply_physics_recursive(prim)
    
    def _apply_physics_recursive(self, prim):
        """Recursively apply physics collision to all mesh children."""
        from pxr import UsdPhysics, PhysxSchema
        
        if prim.IsA(UsdGeom.Mesh):
            # Apply collision API
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            
            # Apply mesh collision API
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision.CreateApproximationAttr(self.collision_params.approximation)
            
            # Apply PhysX collision API
            if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                physx_collision.CreateContactOffsetAttr(self.physics_body_params.contact_offset)
                physx_collision.CreateRestOffsetAttr(self.physics_body_params.rest_offset)
        
        # Recursively process children
        for child in prim.GetChildren():
            self._apply_physics_recursive(child)
    
    def setup(self):
        """Complete setup: import, transform, materials, physics."""
        self.import_model()
        self.apply_transform()
        self.create_visual_material()
        self.apply_physics()
        self.apply_physics_material()


# ============================================================================
# BALL CLASS
# ============================================================================

class Ball:
    """
    Represents a spherical ball created procedurally.
    
    Manages:
    - Sphere geometry creation
    - Visual materials (color, appearance)
    - Physics (collision, dynamics, mass)
    - Physics materials (friction, bounce)
    """
    
    def __init__(self, ball_params: BallParams):
        """Initialize Ball with parameters."""
        self.params = ball_params
        self.prim_path = ball_params.prim_path
        self.radius = ball_params.radius
        self.transform_params = ball_params.transform
        self.physics_material_params = ball_params.physics_material
        self.visual_material_params = ball_params.visual_material
        self.physics_body_params = ball_params.physics_body
        self._prim = None
    
    def get_prim(self):
        """Get USD prim with caching."""
        if self._prim is None:
            stage = omni.usd.get_context().get_stage()
            self._prim = stage.GetPrimAtPath(self.prim_path)
        return self._prim
    
    def create_geometry(self):
        """Create sphere geometry procedurally."""
        stage = omni.usd.get_context().get_stage()
        sphere = UsdGeom.Sphere.Define(stage, self.prim_path)
        
        # Set radius
        sphere.CreateRadiusAttr(self.radius)
        
        # Apply transforms
        sphere.AddTranslateOp().Set(Gf.Vec3d(*self.transform_params.position))
        sphere.AddScaleOp().Set(Gf.Vec3f(*self.transform_params.scale))
        
        # Set extent (bounding box)
        extent = [
            (-self.radius, -self.radius, -self.radius),
            (self.radius, self.radius, self.radius)
        ]
        sphere.CreateExtentAttr(extent)
    
    def apply_visual_material(self):
        """Create and apply PBR visual material."""
        from pxr import UsdShade, Sdf
        
        stage = omni.usd.get_context().get_stage()
        
        # Create material
        material_path = f"{self.prim_path}/Looks/Material"
        material = UsdShade.Material.Define(stage, material_path)
        
        # Create PBR shader
        shader_path = f"{material_path}/Shader"
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        # Set shader parameters
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*self.visual_material_params.color)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            self.visual_material_params.roughness
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            self.visual_material_params.metallic
        )
        
        # Connect shader to material
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.GetOutput("surface"))
        
        # Bind material to sphere
        UsdShade.MaterialBindingAPI(self.get_prim()).Bind(material)
    
    def apply_physics(self):
        """Apply physics using analytic sphere collision."""
        from pxr import UsdPhysics, PhysxSchema, Sdf
        
        prim = self.get_prim()
        
        # Apply rigid body and collision APIs
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        
        # Set physics attributes
        prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        
        # Apply PhysX APIs
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
        
        # Enable CCD if requested
        if self.physics_body_params.enable_ccd:
            physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI(prim)
            physx_rigid_body.CreateEnableCCDAttr(True)
        
        # Set mass if specified
        if self.physics_body_params.mass is not None:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(self.physics_body_params.mass)
    
    def apply_physics_material(self):
        """Apply physics material (friction and restitution)."""
        from pxr import UsdPhysics, Sdf
        
        stage = omni.usd.get_context().get_stage()
        material_path = f"{self.prim_path}/PhysicsMaterial"
        physics_material = UsdPhysics.MaterialAPI.Apply(
            stage.DefinePrim(material_path, "Material")
        )
        
        physics_material.CreateStaticFrictionAttr(
            self.physics_material_params.static_friction
        )
        physics_material.CreateDynamicFrictionAttr(
            self.physics_material_params.dynamic_friction
        )
        physics_material.CreateRestitutionAttr(
            self.physics_material_params.restitution
        )
    
    def setup(self):
        """Complete setup: geometry, materials, physics."""
        self.create_geometry()
        self.apply_visual_material()
        self.apply_physics()
        self.apply_physics_material()


# ============================================================================
# SCENE MANAGER CLASS
# ============================================================================

class SceneManager:
    """
    Manages the overall simulation scene.
    
    Responsibilities:
    - Initialize stage
    - Setup robot and target
    - Run simulation trials
    """
    
    def __init__(
        self,
        robot_params: RobotParams,
        target_params: TargetParams,
        simulation_params: SimulationParams,
        lighting_params: LightingParams,
        plate_params: PlateParams,
        ball_params: BallParams,
    ):
        """Initialize scene manager."""
        self.robot_params = robot_params
        self.target_params = target_params
        self.simulation_params = simulation_params
        self.lighting_params = lighting_params
        self.plate_params = plate_params
        self.ball_params = ball_params
        
        # Create robot and controller
        self.robot = Franka(robot_params)
        self.controller = FrankaController(self.robot, simulation_params.device)
        
        # Create plate and ball
        self.plate = Plate(plate_params)
        self.ball = Ball(ball_params)
        
        self.target_sphere: Optional[Sphere] = None
        self.objects = []  # List to store all physics objects
    
    def initialize_stage(self):
        """Initialize stage with robot and target sphere."""
        print("Initializing stage...")
        
        # Configure simulation device
        SimulationManager.set_physics_sim_device(self.simulation_params.device)
        simulation_app.update()
        
        # Create stage
        stage_utils.create_new_stage()
        
        # Add ground plane
        GroundPlane("/World/groundPlane")
        print("✓ Ground plane added")
        
        # Add grid visualization
        self.add_grid()
        
        # Add lighting
        self.add_lighting()
        
        # Add robot
        print(f"Adding robot from: {self.robot_params.usd_path}")
        stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path() + self.robot_params.usd_path,
            path=self.robot_params.prim_path,
            variants=self.robot_params.variants,
        )
        
        # Apply robot transform (position, rotation, and scale)
        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath(self.robot_params.prim_path)
        xformable = UsdGeom.Xformable(robot_prim)
        
        # Clear and recreate transform operations
        xformable.ClearXformOpOrder()
        
        # Add translate operation
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*self.robot_params.position))
        
        # Add rotation operation
        rotate_op = xformable.AddRotateZOp()
        rotate_op.Set(self.robot_params.rotation_z)
        
        # Add scale operation - only if scale != 1.0
        if self.robot_params.scale != 1.0:
            scale_op = xformable.AddScaleOp()
            scale_op.Set(Gf.Vec3d(self.robot_params.scale, self.robot_params.scale, self.robot_params.scale))
        
        print(f"✓ Robot positioned at {self.robot_params.position} with {self.robot_params.rotation_z}° rotation, scale={self.robot_params.scale}x")
        
        # Add target sphere
        print(f"Adding target sphere at: {self.target_params.prim_path}")
        visual_material = PreviewSurfaceMaterial("/Visual_materials/red")
        visual_material.set_input_values("diffuseColor", self.target_params.color)
        self.target_sphere = Sphere(
            self.target_params.prim_path,
            radii=[self.target_params.radius],
            reset_xform_op_properties=True
        )
        self.target_sphere.apply_visual_materials(visual_material)
        
        print("✓ Stage initialized")
    
    def add_grid(self):
        """Add black grid on ground plane."""
        from pxr import UsdGeom, Gf, UsdShade, Sdf
        
        stage = omni.usd.get_context().get_stage()
        
        # Grid parameters
        grid_size = 20  # Total grid size in meters
        grid_spacing = 1.0  # Spacing between grid lines
        num_lines = int(grid_size / grid_spacing) + 1
        
        # Create a parent Xform for all grid lines
        grid_path = "/World/Grid"
        grid_xform = UsdGeom.Xform.Define(stage, grid_path)
        
        # Create material for black lines
        material_path = "/World/Materials/GridMaterial"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        
        line_index = 0
        offset = grid_size / 2
        
        # Create lines parallel to X-axis (varying Y)
        for i in range(num_lines):
            y_pos = -offset + i * grid_spacing
            line_path = f"{grid_path}/LineX_{line_index}"
            line = UsdGeom.Mesh.Define(stage, line_path)
            
            # Create thin rectangle for line on XY plane at Z=0
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
        
        # Create lines parallel to Y-axis (varying X)
        for i in range(num_lines):
            x_pos = -offset + i * grid_spacing
            line_path = f"{grid_path}/LineY_{line_index}"
            line = UsdGeom.Mesh.Define(stage, line_path)
            
            # Create thin rectangle for line on XY plane at Z=0
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
        
        print(f"✓ Black grid added on XY plane ({num_lines}x{num_lines} lines)")
    
    def add_lighting(self):
        """Add lighting to the scene."""
        print("Adding lights...")
        
        stage = omni.usd.get_context().get_stage()
        
        # Distant light (sun)
        distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
        distant_light.CreateIntensityAttr(self.lighting_params.distant_intensity)
        
        # Get or create rotation operation
        distant_light_prim = stage.GetPrimAtPath("/World/DistantLight")
        xformable = UsdGeom.Xformable(distant_light_prim)
        rotate_op = xformable.GetOrderedXformOps()
        if rotate_op:
            rotate_op[0].Set(Gf.Vec3d(self.lighting_params.angle, 0, 0))
        else:
            xformable.AddRotateXYZOp().Set(Gf.Vec3d(self.lighting_params.angle, 0, 0))
        
        # Dome light (ambient)
        dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(self.lighting_params.dome_intensity)
        
        print("✓ Lights added")
    
    def add_object(self, obj):
        """Add a physics object to the scene.
        
        Args:
            obj: A Plate or Ball object to add
        """
        print(f"Setting up object at {obj.prim_path}...")
        obj.setup()
        self.objects.append(obj)
        
        # Store reference for specific object types
        if isinstance(obj, Plate):
            self.plate = obj
        elif isinstance(obj, Ball):
            self.ball = obj
        
        print(f"✓ Object {obj.prim_path} setup complete")
    
    def setup_plate_and_ball(self):
        """Setup plate and ball (call after stage is initialized)."""
        print("\nSetting up plate and ball...")
        self.plate.setup()
        self.ball.setup()
        print("✓ Plate and ball setup complete")
    
    def initialize_robot(self):
        """Initialize robot state (call after timeline is playing)."""
        self.robot.initialize()
    
    def run_simulation(self, run_trials: bool = True):
        """Run simulation with multiple random target trials or just display scene."""
        # Play simulation
        print("\nStarting simulation...")
        omni.timeline.get_timeline_interface().play()
        simulation_app.update()
        
        # Initialize robot after timeline is playing
        self.robot.initialize()
        
        if not run_trials:
            print(f"\n{'='*70}")
            print("Scene display mode - GUI will stay open")
            print("Press Ctrl+C or close window to exit")
            print(f"{'='*70}\n")
            
            # Keep the simulation running to display the scene
            while simulation_app.is_running():
                simulation_app.update()
            return
        
        print(f"\n{'='*70}")
        print(f"Running {self.simulation_params.num_trials} trials with differential IK")
        print(f"IK Method: {self.simulation_params.ik_method}")
        print(f"Device: {self.simulation_params.device}")
        if USE_LINE_TRAJECTORY:
            mode_str = "Continuous Tracking" if TRAJECTORY_TRACKING_MODE else "Discrete Waypoints"
            print(f"Trajectory: {mode_str} - Line from {LINE_START_POINT} to {LINE_END_POINT}")
            if TRAJECTORY_TRACKING_MODE:
                print(f"Duration: {TRAJECTORY_DURATION}s")
        else:
            print(f"Trajectory: Random targets")
        print(f"{'='*70}\n")
        
        if USE_LINE_TRAJECTORY and TRAJECTORY_TRACKING_MODE:
            # Continuous trajectory tracking mode
            line_start = torch.tensor(LINE_START_POINT, device=self.simulation_params.device)
            line_end = torch.tensor(LINE_END_POINT, device=self.simulation_params.device)
            
            # Reset robot to starting position
            self.robot.reset_to_default()
            simulation_app.update()
            
            print(f"Starting continuous trajectory tracking...")
            print(f"Moving from {LINE_START_POINT} to {LINE_END_POINT} over {TRAJECTORY_DURATION}s\n")
            
            # Estimate physics timestep (typically 1/60 or 1/120)
            dt = 1.0 / 60.0  # Assuming 60 FPS
            total_steps = int(TRAJECTORY_DURATION / dt)
            
            for step in range(total_steps):
                # Compute current time and trajectory parameter
                t = step * dt / TRAJECTORY_DURATION  # 0 to 1
                t = min(t, 1.0)  # Clamp to [0, 1]
                
                # Compute target position along the line
                target_position = line_start + t * (line_end - line_start)
                
                # Update sphere position
                self.target_sphere.set_world_poses(positions=wp.from_torch(target_position))
                
                # Compute differential IK
                delta_dof = self.controller.compute_differential_ik(
                    target_position=target_position,
                    method=self.simulation_params.ik_method
                )
                
                # Get current DOF positions
                current_dof = wp.to_torch(self.robot.get_dof_positions())
                
                # Get current end-effector position
                current_ee_pos, _ = self.robot.get_ee_pose()
                current_ee_pos_torch = wp.to_torch(current_ee_pos)
                
                # Compute position error
                position_error = torch.norm(target_position - current_ee_pos_torch.squeeze())
                
                # Print debug info every 60 steps (approximately every second)
                if step % 60 == 0:
                    time_elapsed = step * dt
                    print(f"  t={time_elapsed:5.2f}s: Target=({target_position[0]:.3f},{target_position[1]:.3f},{target_position[2]:.3f}) "
                          f"EE=({current_ee_pos_torch[0,0]:.3f},{current_ee_pos_torch[0,1]:.3f},{current_ee_pos_torch[0,2]:.3f}) "
                          f"Error={position_error:.4f}m")
                
                # Apply delta to arm DOFs only
                new_dof_targets = current_dof[:, :self.robot_params.num_arm_dof] + delta_dof
                self.controller.apply_joint_command(new_dof_targets, dof_indices=list(range(self.robot_params.num_arm_dof)))
                
                simulation_app.update()
            
            # Final position check
            final_ee_pos, _ = self.robot.get_ee_pose()
            final_ee_pos_torch = wp.to_torch(final_ee_pos)
            final_target = line_end
            final_error = torch.norm(final_target - final_ee_pos_torch.squeeze())
            print(f"\n✓ Trajectory tracking completed - Final error: {final_error:.4f}m")
            
        else:
            # Discrete waypoint mode (original behavior)
            for trial in range(self.simulation_params.num_trials):
                # Generate target position
                if USE_LINE_TRAJECTORY:
                    # Move along horizontal line from start to end
                    t = trial / max(1, self.simulation_params.num_trials - 1)  # 0 to 1
                    line_start = torch.tensor(LINE_START_POINT, device=self.simulation_params.device)
                    line_end = torch.tensor(LINE_END_POINT, device=self.simulation_params.device)
                    target_position = line_start + t * (line_end - line_start)
                else:
                    # Generate random target position
                    random_sample = 2 * (torch.rand((3,), device=self.simulation_params.device) - 0.5)
                    target_position = torch.tensor(
                        self.target_params.region_center,
                        device=self.simulation_params.device
                    ) + (self.target_params.region_size / 2) * random_sample
                
                # Set sphere to target position
                self.target_sphere.set_world_poses(positions=wp.from_torch(target_position))
                
                # Reset robot
                self.robot.reset_to_default()
                simulation_app.update()
                
                trajectory_type = "Line" if USE_LINE_TRAJECTORY else "Random"
                print(f"Trial {trial + 1}/{self.simulation_params.num_trials} ({trajectory_type}): "
                      f"Target = ({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")
                
                # Run IK steps
                for step in range(self.simulation_params.steps_per_trial):
                    # Compute differential IK
                    delta_dof = self.controller.compute_differential_ik(
                        target_position=target_position,
                        method=self.simulation_params.ik_method
                    )
                    
                    # Get current DOF positions
                    current_dof = wp.to_torch(self.robot.get_dof_positions())
                    
                    # Get current end-effector position for debugging
                    current_ee_pos, _ = self.robot.get_ee_pose()
                    current_ee_pos_torch = wp.to_torch(current_ee_pos)
                    
                    # Compute position error
                    position_error = torch.norm(target_position - current_ee_pos_torch.squeeze())
                    
                    # Print debug info every 20 steps
                    if step % 20 == 0:
                        print(f"    Step {step:3d}: EE=({current_ee_pos_torch[0,0]:.3f}, {current_ee_pos_torch[0,1]:.3f}, {current_ee_pos_torch[0,2]:.3f}) "
                              f"Error={position_error:.4f} DOF_delta_norm={torch.norm(delta_dof):.4f}")
                    
                    # Apply delta to arm DOFs only
                    new_dof_targets = current_dof[:, :self.robot_params.num_arm_dof] + delta_dof
                    self.controller.apply_joint_command(new_dof_targets, dof_indices=list(range(self.robot_params.num_arm_dof)))
                    
                    simulation_app.update()
                
                # Final position check
                final_ee_pos, _ = self.robot.get_ee_pose()
                final_ee_pos_torch = wp.to_torch(final_ee_pos)
                final_error = torch.norm(target_position - final_ee_pos_torch.squeeze())
                print(f"  ✓ Completed {self.simulation_params.steps_per_trial} IK steps - Final error: {final_error:.4f}m")
    
    def run_plate_ball_simulation(self):
        """Run plate and ball physics simulation."""
        import math
        from pxr import UsdGeom, Gf, Usd
        
        print("\nStarting plate and ball simulation...")
        print(f"Plate will start moving after {SETTLE_TIME}s settling time")
        if PLATE_MOTION_ENABLED:
            axis_names = ['X', 'Y', 'Z']
            print(f"Sinusoidal motion on {axis_names[PLATE_MOTION_AXIS]}-axis:")
            print(f"  Amplitude: {PLATE_MOTION_AMPLITUDE} m")
            print(f"  Frequency: {PLATE_MOTION_FREQUENCY} Hz")
        print("Close window to stop.\n")
        
        # Play timeline
        omni.timeline.get_timeline_interface().play()
        simulation_app.update()
        
        step_count = 0
        motion_started = False
        plate_initial_pos = None
        is_kinematic = (PLATE_MECHANICS == 'kinematic')
        
        # Store initial plate position
        if is_kinematic and self.plate:
            plate_prim = self.plate.get_prim()
            xformable = UsdGeom.Xformable(plate_prim)
            world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            plate_initial_pos = list(world_transform.ExtractTranslation())
        
        # Simulation loop
        while simulation_app.is_running():
            simulation_app.update()
            step_count += 1
            
            # Estimate time (assuming ~60 FPS)
            t = step_count / 60.0
            
            # Apply plate motion after settling
            if PLATE_MOTION_ENABLED and self.plate and t >= SETTLE_TIME:
                if not motion_started:
                    print(f"\n{'='*70}")
                    mode = "KINEMATIC" if is_kinematic else "DYNAMIC"
                    print(f"STARTING PLATE MOTION at t={t:.2f}s - {mode}")
                    print(f"{'='*70}\n")
                    motion_started = True
                
                motion_time = t - SETTLE_TIME
                amplitude = PLATE_MOTION_AMPLITUDE
                frequency = PLATE_MOTION_FREQUENCY
                axis = PLATE_MOTION_AXIS
                angular_freq = 2 * math.pi * frequency
                
                plate_prim = self.plate.get_prim()
                
                if is_kinematic:
                    # Kinematic: position control
                    offset = amplitude * math.sin(angular_freq * motion_time)
                    new_pos = plate_initial_pos.copy()
                    new_pos[axis] = plate_initial_pos[axis] + offset
                    
                    xformable = UsdGeom.Xformable(plate_prim)
                    xformable.ClearXformOpOrder()
                    xformable.AddTranslateOp().Set(Gf.Vec3d(*new_pos))
                    if PLATE_ROTATION_X is not None:
                        xformable.AddRotateXOp().Set(float(PLATE_ROTATION_X))
                    xformable.AddScaleOp().Set(Gf.Vec3f(*PLATE_SCALE))
                else:
                    # Dynamic: velocity control
                    from pxr import UsdPhysics
                    velocity_magnitude = amplitude * angular_freq * math.cos(angular_freq * motion_time)
                    velocity = [0.0, 0.0, 0.0]
                    velocity[axis] = velocity_magnitude
                    
                    rigid_body = UsdPhysics.RigidBodyAPI(plate_prim)
                    rigid_body.GetVelocityAttr().Set(Gf.Vec3f(*velocity))
                    rigid_body.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            
            # Print positions periodically
            if step_count % 100 == 0:
                if self.plate:
                    plate_prim = self.plate.get_prim()
                    xformable = UsdGeom.Xformable(plate_prim)
                    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    plate_pos = world_transform.ExtractTranslation()
                    print(f"Step {step_count:5d} | Plate: ({plate_pos[0]:6.3f}, {plate_pos[1]:6.3f}, {plate_pos[2]:6.3f})")
                
                if self.ball:
                    ball_prim = self.ball.get_prim()
                    xformable = UsdGeom.Xformable(ball_prim)
                    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    ball_pos = world_transform.ExtractTranslation()
                    print(f"Step {step_count:5d} | Ball: ({ball_pos[0]:6.3f}, {ball_pos[1]:6.3f}, {ball_pos[2]:6.3f})")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main execution flow.
    
    Steps:
    1. Create scene manager
    2. Initialize stage with robot and target
    3. Initialize robot controller
    4. Run simulation trials
    """
    print("=" * 70)
    print("Franka Panda Differential IK Control")
    print("=" * 70)
    
    try:
        # ===== STEP 1: Create Scene Manager =====
        scene = SceneManager(
            robot_params=RobotParams(
                usd_path=ROBOT_USD_PATH,
                prim_path=ROBOT_PATH,
                position=ROBOT_POSITION,
                rotation_z=ROBOT_BASE_ROTATION,
                scale=ROBOT_SCALE,
                variants=ROBOT_VARIANTS,
                default_dof_positions=DEFAULT_DOF_POSITIONS,
                ee_link_name=EE_LINK_NAME,
                num_arm_dof=NUM_ARM_DOF,
            ),
            target_params=TargetParams(
                prim_path=SPHERE_PATH,
                radius=SPHERE_RADIUS,
                color=SPHERE_COLOR,
                region_center=TARGET_REGION_CENTER,
                region_size=TARGET_REGION_SIZE,
            ),
            simulation_params=SimulationParams(
                num_trials=NUM_TRIALS,
                steps_per_trial=STEPS_PER_TRIAL,
                device=DEVICE,
                ik_method=IK_METHOD,
            ),
            lighting_params=LightingParams(
                distant_intensity=DISTANT_LIGHT_INTENSITY,
                dome_intensity=DOME_LIGHT_INTENSITY,
                angle=DISTANT_LIGHT_ANGLE,
            ),
            plate_params=PlateParams(
                prim_path="/World/Plate",
                model_file=PLATE_USD_FILE,
                transform=TransformParams(
                    position=PLATE_POSITION,
                    scale=PLATE_SCALE,
                    rotateX=PLATE_ROTATION_X,
                ),
                physics_material=PhysicsMaterialParams(
                    static_friction=PLATE_STATIC_FRICTION,
                    dynamic_friction=PLATE_DYNAMIC_FRICTION,
                    restitution=PLATE_RESTITUTION,
                ),
                visual_material=BallPlateVisualMaterialParams(
                    color=PLATE_COLOR,
                ),
                physics_body=PhysicsBodyParams(
                    enable_ccd=ENABLE_CCD,
                    contact_offset=CONTACT_OFFSET,
                    rest_offset=REST_OFFSET,
                ),
                collision=CollisionParams(
                    approximation="convexDecomposition",
                    enable_collision=True,
                ),
                mechanics_mode=PLATE_MECHANICS,
            ),
            ball_params=BallParams(
                prim_path="/World/PlateBall",
                radius=BALL_RADIUS,
                transform=TransformParams(
                    position=BALL_POSITION,
                    scale=BALL_SCALE,
                ),
                physics_material=PhysicsMaterialParams(
                    static_friction=BALL_STATIC_FRICTION,
                    dynamic_friction=BALL_DYNAMIC_FRICTION,
                    restitution=BALL_RESTITUTION,
                ),
                visual_material=BallPlateVisualMaterialParams(
                    color=BALL_COLOR,
                ),
                physics_body=PhysicsBodyParams(
                    is_dynamic=True,
                    enable_ccd=ENABLE_CCD,
                    mass=BALL_MASS,
                    contact_offset=CONTACT_OFFSET,
                    rest_offset=REST_OFFSET,
                ),
            ),
        )
        
        # ===== STEP 2: Initialize Stage =====
        scene.initialize_stage()
        
        # ===== STEP 3: Setup Plate and Ball =====
        scene.setup_plate_and_ball()
        
        # ===== STEP 4: Run Simulation Based on Mode =====
        if SIMULATION_MODE == "scene-viz":
            # Just display the scene without running any simulation (timeline stopped)
            print(f"\n{'='*70}")
            print("DISPLAY MODE - Static scene visualization")
            print("Timeline is STOPPED - no physics simulation running")
            print("Press Ctrl+C or close window to exit")
            print(f"{'='*70}\n")
            
            # Do NOT play the timeline - keep it stopped for static visualization
            simulation_app.update()
            
            while simulation_app.is_running():
                simulation_app.update()
        
        elif SIMULATION_MODE == "robot-ik":
            # Run robot IK simulation
            scene.initialize_robot()
            scene.run_simulation(run_trials=True)
            print(f"\n{'='*70}")
            print("All trials completed successfully!")
            print(f"{'='*70}\n")
        
        elif SIMULATION_MODE == "plate-ball":
            # Run plate and ball physics simulation
            scene.run_plate_ball_simulation()
    
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
