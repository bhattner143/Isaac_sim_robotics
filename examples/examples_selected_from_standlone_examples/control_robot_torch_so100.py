"""
SO100 Manipulator Differential IK Control with Target Sphere

Demonstrates:
1. Differential inverse kinematics using PyTorch and Jacobian matrices
2. Target tracking with visual sphere
3. Multiple trials with random target positions
4. Warp-PyTorch interoperability
5. SO100 6-DOF manipulator control

Organized using OOP structure from test_manipulator_ik.py template.
"""

# Import standard Python libraries
from isaacsim import SimulationApp
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
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
args, _ = parser.parse_known_args()

# STEP 1: Launch Isaac Sim application
simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
    "renderer": "RayTracedLighting",
    "active_gpu": 0,
})

# STEP 2: Import Isaac Sim modules (after SimulationApp is created)
from pxr import UsdGeom, UsdLux, Gf, UsdShade, Sdf
import omni.usd
import omni.kit.commands
import torch
import warp as wp
import numpy as np

# STEP 2: Import Isaac Sim modules (after SimulationApp is created)
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.timeline
import torch
import warp as wp
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import Sphere, GroundPlane
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path


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
# NOTE: Run test_so100_links.py first to verify correct link names for your robot model
ROBOT_USD_FILE = str(Path("manipulators/so101.usd").absolute())
ROBOT_PATH = "/World/so101_new_calib"
DEFAULT_DOF_POSITIONS = [0.0, -1.3, 0.0, -2.87, 0.0, 2.0]
ROBOT_POSITION = (0.0, 0.0, 0.5)
ROBOT_SCALE = (1.0, 1.0, 1.0)
ROBOT_ROTATION_X = 0
EE_LINK_NAME = "moving_jaw_so101_v1"  # Adjust based on test_so100_links.py output
# EE_LINK_PATH = "/World/SO100/so101/gripper/gripperframe"  # Full path to end-effector link
NUM_ARM_DOF = 6  # SO100 has 6 DOF arm

# --- Target Sphere Configuration ---
SPHERE_PATH = "/World/sphere"
SPHERE_RADIUS = 0.03
SPHERE_COLOR = (1.0, 0.0, 0.0)  # Red
TARGET_REGION_CENTER = [0.3, 0.0, 0.8]  # Center of random target region
TARGET_REGION_SIZE = 0.15  # Cube side length for random targets

# --- Simulation Configuration ---
NUM_TRIALS = 10  # Number of random target trials
STEPS_PER_TRIAL = 100  # IK steps per trial
DEVICE = args.device  # "cpu" or "cuda"
IK_METHOD = args.ik_method
RUN_SIMULATION = True  # Set to True to run IK trials, False to just show scene

# --- Scene Configuration ---
DISTANT_LIGHT_INTENSITY = 1000.0
DOME_LIGHT_INTENSITY = 300.0
DISTANT_LIGHT_ANGLE = 315.0


# ============================================================================
# PARAMETER CLASSES (Dataclasses)
# ============================================================================

@dataclass
class TransformParams:
    """Parameters for positioning, rotating, and scaling objects in 3D space."""
    position: tuple[float, float, float]
    scale: tuple[float, float, float]
    rotateX: Optional[float] = None


@dataclass
class RobotParams:
    """Parameters for robot configuration."""
    usd_file: str
    prim_path: str
    transform_params: TransformParams
    ee_link_name: str
    default_dof_positions: list
    num_arm_dof: int = 6


@dataclass
class TargetParams:
    """Parameters for target sphere."""
    prim_path: str
    radius: float
    color: tuple
    region_center: list
    region_size: float


@dataclass
class LightingParams:
    """Parameters for scene lighting."""
    distant_intensity: float = 1000.0
    dome_intensity: float = 300.0
    angle: float = 315.0


@dataclass
class SimulationParams:
    """Parameters controlling simulation behavior."""
    num_trials: int = 10
    steps_per_trial: int = 100
    device: str = "cpu"
    ik_method: str = "damped-least-squares"


@dataclass
class RobotState:
    """Runtime state discovered from initialized robot."""
    robot: Articulation
    end_effector: RigidPrim
    ee_link_index: int
    num_dof: int
    dof_names: list[str]


# ============================================================================
# SO100 ROBOT CLASS
# ============================================================================

class SO100:
    """
    SO100 manipulator robot model.
    
    Manages:
    - Robot model import and transforms
    - Articulation and end-effector state
    - Joint positions and end-effector pose
    """
    
    def __init__(self, robot_params: RobotParams):
        """Initialize SO100 robot."""
        self.params = robot_params
        self.state: Optional[RobotState] = None
    
    def import_model(self):
        """Import the SO100 USD file into the scene."""
        print(f"Importing SO100 from: {self.params.usd_file}")
        
        usd_context = omni.usd.get_context()
        omni.kit.commands.execute(
            'CreateReferenceCommand',
            usd_context=usd_context,
            path_to=self.params.prim_path,
            asset_path=self.params.usd_file,
            instanceable=False
        )
        print(f"✓ Model imported at {self.params.prim_path}")
    
    def apply_transform(self):
        """Apply position, rotation, and scale to the robot."""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.params.prim_path)
        xformable = UsdGeom.Xformable(prim)
        
        # Clear existing transforms
        xformable.ClearXformOpOrder()
        
        # Apply transforms: translate, rotate, scale
        xformable.AddTranslateOp().Set(Gf.Vec3d(*self.params.transform_params.position))
        if self.params.transform_params.rotateX is not None:
            xformable.AddRotateXOp().Set(self.params.transform_params.rotateX)
        # xformable.AddScaleOp().Set(Gf.Vec3d(*self.params.transform_params.scale))
        
        print(f"✓ Transform applied: pos={self.params.transform_params.position}")
    
    def initialize(self):
        """Initialize robot articulation and end-effector (call after world.reset())."""
        print("\nInitializing SO100 robot...")
        
        # Create articulation and end-effector
        robot = Articulation(self.params.prim_path)
        end_effector = RigidPrim(f"{self.params.prim_path}/{self.params.ee_link_name}")
        ee_link_index = robot.get_link_indices(self.params.ee_link_name).list()[0]

        # Set default state
        if self.params.default_dof_positions is not None:
            robot.set_default_state(dof_positions=self.params.default_dof_positions)
            print(f"✓ Robot initialized with {self.params.num_arm_dof} arm DOFs")
        print(f"  End-effector link: {self.params.ee_link_name} (index: {ee_link_index})")
        
        # Discover DOF information after initialization
        num_dof = robot.num_dofs
        dof_names = robot.dof_names
        print(f"✓ Robot has {num_dof} DOF:")
        for i, name in enumerate(dof_names):
            print(f"  Joint {i}: {name}")
        
        # Store all runtime state in RobotState dataclass
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
        """Get current end-effector world pose (position and orientation in WXYZ format)."""
        if self.state is None:
            return None, None
        
        # Get pose from end-effector RigidPrim
        position, orientation = self.state.end_effector.get_world_poses()
        return position, orientation
    
    def get_dof_positions(self):
        """Get current DOF positions."""
        if self.state is None:
            return None
        return self.state.robot.get_dof_positions()
    
    def set_joint_positions(self, positions):
        """Set robot joint positions."""
        if self.state is not None:
            self.state.robot.set_dof_position_targets(positions)
    
    def get_jacobian_matrices(self):
        """Get Jacobian matrices for all links."""
        if self.state is None:
            return None
        return self.state.robot.get_jacobian_matrices()


# ============================================================================
# SO100 CONTROLLER CLASS
# ============================================================================

class SO100Controller:
    """
    Differential IK controller for SO100 manipulator.
    
    Implements:
    - Differential inverse kinematics
    - Target tracking
    - Multiple IK methods (SVD, pseudoinverse, transpose, damped least-squares)
    """
    
    def __init__(self, robot: SO100, device: str = "cpu"):
        """Initialize controller with SO100 robot instance."""
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
    
    def apply_joint_command(self, positions):
        """Apply joint position command to robot."""
        self.robot.set_joint_positions(positions)


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
        lighting_params: LightingParams,
        simulation_params: SimulationParams,
    ):
        """Initialize scene manager."""
        self.robot_params = robot_params
        self.target_params = target_params
        self.lighting_params = lighting_params
        self.simulation_params = simulation_params
        
        # Create robot and controller
        self.robot = SO100(robot_params)
        self.controller = SO100Controller(self.robot, simulation_params.device)
        
        self.stage = None
        self.target_sphere: Optional[Sphere] = None
    
    def initialize_world(self):
        """Initialize the stage."""
        print("Initializing stage...")
        
        # Configure simulation device
        SimulationManager.set_physics_sim_device(self.simulation_params.device)
        simulation_app.update()
        
        # Create stage
        stage_utils.create_new_stage()
        
        # Add ground plane
        GroundPlane("/World/groundPlane")
        print("✓ Ground plane added")
        
        self.stage = omni.usd.get_context().get_stage()
        
        # Add grid visualization
        self.add_grid()
    
    def add_grid(self):
        """Add black grid on ground plane."""
        from pxr import UsdGeom, Gf, UsdShade, Sdf
        
        # Grid parameters
        grid_size = 20  # Total grid size in meters
        grid_spacing = 1.0  # Spacing between grid lines
        num_lines = int(grid_size / grid_spacing) + 1
        
        # Create a parent Xform for all grid lines
        grid_path = "/World/Grid"
        grid_xform = UsdGeom.Xform.Define(self.stage, grid_path)
        
        # Create material for black lines
        material_path = "/World/Materials/GridMaterial"
        material = UsdShade.Material.Define(self.stage, material_path)
        shader = UsdShade.Shader.Define(self.stage, material_path + "/Shader")
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
            line = UsdGeom.Mesh.Define(self.stage, line_path)
            
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
            line = UsdGeom.Mesh.Define(self.stage, line_path)
            
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
        
        # Distant light (sun)
        distant_light = UsdLux.DistantLight.Define(self.stage, "/World/DistantLight")
        distant_light.CreateIntensityAttr(self.lighting_params.distant_intensity)
        
        # Get or create rotation operation
        distant_light_prim = self.stage.GetPrimAtPath("/World/DistantLight")
        xformable = UsdGeom.Xformable(distant_light_prim)
        rotate_op = xformable.GetOrderedXformOps()
        if rotate_op:
            rotate_op[0].Set(Gf.Vec3d(self.lighting_params.angle, 0, 0))
        else:
            xformable.AddRotateXYZOp().Set(Gf.Vec3d(self.lighting_params.angle, 0, 0))
        
        # Dome light (ambient)
        dome_light = UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(self.lighting_params.dome_intensity)
        
        print("✓ Lights added")
    
    def add_target_sphere(self):
        """Add target sphere to the scene."""
        print(f"Adding target sphere at: {self.target_params.prim_path}")
        
        # Create visual material
        visual_material = PreviewSurfaceMaterial("/Visual_materials/red")
        visual_material.set_input_values("diffuseColor", self.target_params.color)
        
        # Create sphere
        self.target_sphere = Sphere(
            self.target_params.prim_path,
            radii=[self.target_params.radius],
            reset_xform_op_properties=True
        )
        self.target_sphere.apply_visual_materials(visual_material)
        
        print("✓ Target sphere added")
    
    def setup_robot(self):
        """Import robot model and apply transforms (but don't initialize yet)."""
        # Import model and apply transforms
        self.robot.import_model()
        self.robot.apply_transform()
        
        print("✓ Robot model imported and transformed")
        
        # Note: robot.initialize() will be called after timeline plays
    
    def run_simulation(self, run_trials: bool = True):
        """Run simulation with multiple random target trials or just display scene."""
        # Play simulation
        print("\nStarting simulation...")
        omni.timeline.get_timeline_interface().play()
        simulation_app.update()
        
        # Initialize robot state after timeline is playing
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
        print(f"{'='*70}\n")
        
        # Run trials
        for trial in range(self.simulation_params.num_trials):
            # Generate random target position
            random_sample = 2 * (torch.rand((3,), device=self.simulation_params.device) - 0.5)
            target_position = torch.tensor(
                self.target_params.region_center,
                device=self.simulation_params.device,
                dtype=torch.float32
            ) + (self.target_params.region_size / 2) * random_sample
            
            # Set sphere to target position
            self.target_sphere.set_world_poses(positions=wp.from_torch(target_position))
            
            # Reset robot to default position
            if self.robot.state:
                self.robot.reset_to_default()
            
            simulation_app.update()
            
            print(f"Trial {trial + 1}/{self.simulation_params.num_trials}: "
                  f"Target = ({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")
            
            # Run IK steps
            for step in range(self.simulation_params.steps_per_trial):
                
                # Compute differential IK
                delta_dof = self.controller.compute_differential_ik(
                    target_position=target_position,
                    method=self.simulation_params.ik_method
                )
                
                if delta_dof is not None:
                    # Get current DOF positions
                    current_dof = wp.to_torch(self.robot.get_dof_positions())
                    
                    # Apply delta to arm DOFs only
                    new_dof_targets = current_dof[:, :self.robot_params.num_arm_dof] + delta_dof
                    
                    # Set joint positions
                    new_positions = current_dof.cpu().numpy()[0]
                    new_positions[:self.robot_params.num_arm_dof] = new_dof_targets.cpu().numpy()[0]
                    self.controller.apply_joint_command(new_positions)
                
                simulation_app.update()
            
            print(f"  ✓ Completed {self.simulation_params.steps_per_trial} IK steps")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main execution flow.
    
    Steps:
    1. Create scene manager
    2. Initialize world with lighting
    3. Add robot and target sphere
    4. Run simulation trials
    """
    print("=" * 70)
    print("SO100 Manipulator Differential IK Control")
    print("=" * 70)
    
    try:
        # ===== STEP 1: Create Scene Manager =====
        scene = SceneManager(
            robot_params=RobotParams(
                usd_file=ROBOT_USD_FILE,
                prim_path=ROBOT_PATH,
                transform_params=TransformParams(
                    position=ROBOT_POSITION,
                    scale=ROBOT_SCALE,
                    rotateX=ROBOT_ROTATION_X,
                ),
                ee_link_name=EE_LINK_NAME,
                num_arm_dof=NUM_ARM_DOF,
                default_dof_positions=DEFAULT_DOF_POSITIONS,
            ),
            target_params=TargetParams(
                prim_path=SPHERE_PATH,
                radius=SPHERE_RADIUS,
                color=SPHERE_COLOR,
                region_center=TARGET_REGION_CENTER,
                region_size=TARGET_REGION_SIZE,
            ),
            lighting_params=LightingParams(
                distant_intensity=DISTANT_LIGHT_INTENSITY,
                dome_intensity=DOME_LIGHT_INTENSITY,
                angle=DISTANT_LIGHT_ANGLE,
            ),
            simulation_params=SimulationParams(
                num_trials=NUM_TRIALS,
                steps_per_trial=STEPS_PER_TRIAL,
                device=DEVICE,
                ik_method=IK_METHOD,
            ),
        )
        
        # ===== STEP 2: Initialize World =====
        # scene.initialize_world()
        scene.initialize_world()
        
        # ===== STEP 3: Add Lighting =====
        scene.add_lighting()
        
        # ===== STEP 4: Setup Robot =====
        scene.setup_robot()
        
        # ===== STEP 5: Add Target Sphere =====
        scene.add_target_sphere()
        
        # ===== STEP 6: Run Simulation or Display Scene =====
        if RUN_SIMULATION:
            scene.run_simulation(run_trials=True)
            print(f"\n{'='*70}")
            print("All trials completed successfully!")
            print(f"{'='*70}\n")
        else:
            scene.run_simulation(run_trials=False)
    
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
