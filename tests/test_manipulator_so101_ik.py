"""
Test Script: Import and Control Manipulator in Isaac Sim (with IK option)

Demonstrates:
1. Importing a robot manipulator from USD
2. Robot articulation control (joint-space sinusoid or IK-driven EE motion)
3. End-effector state monitoring
4. Scene management with OOP structure
"""

# Import standard Python libraries
from isaacsim import SimulationApp
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# STEP 1: Launch Isaac Sim application
simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
    "renderer": "RayTracedLighting",
    "active_gpu": 0,
})

# STEP 2: Import Isaac Sim modules (after SimulationApp is created)
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.motion_generation import ArticulationKinematicsSolver
from pxr import UsdGeom, UsdLux, Gf
import omni.usd
import omni.kit.commands
import math
import numpy as np
import torch
import warp as wp


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

# --- Manipulator Configuration ---
MANIPULATOR_USD_FILE = str(Path("manipulators/so101_physics.usd").absolute())
MANIPULATOR_POSITION = (0.0, 0.0, 0.5)
MANIPULATOR_SCALE = (1.0, 1.0, 1.0)
MANIPULATOR_ROTATION_X = 0

# --- End-Effector Configuration ---
EE_LINK_PATH = "/World/Manipulator/so101/tcp_link"
EE_FRAME_NAME = "tcp_link"  # end-effector frame for IK solver

# --- Joint Motion Configuration ---
TRAJECTORY_AMPLITUDE = 0.5  # Joint motion amplitude (radians)
TRAJECTORY_FREQUENCY = 0.2  # Frequency in Hz
TRAJECTORY_ENABLED = True

# --- IK Configuration ---
IK_ENABLED = True                    # Toggle IK solver usage
IK_USE_DIFFERENTIAL = True           # Use differential IK (Jacobian-based) instead of ArticulationKinematicsSolver
IK_METHOD = "damped-least-squares"   # Options: "singular-value-decomposition", "pseudoinverse", "transpose", "damped-least-squares"
IK_SCALE = 1.0                       # IK solution scale factor
IK_DAMPING = 0.05                    # Damping factor for damped-least-squares method
IK_MIN_SINGULAR_VALUE = 1e-5         # Minimum singular value for SVD method
IK_TARGET_RADIUS = 0.05              # Horizontal circle radius (meters)
IK_HEIGHT_OFFSET = 0.0               # Vertical offset relative to initial EE height

# --- Scene Configuration ---
DISTANT_LIGHT_INTENSITY = 1000.0
DOME_LIGHT_INTENSITY = 300.0
DISTANT_LIGHT_ANGLE = 315.0

# --- Simulation Configuration ---
PRINT_INTERVAL = 60  # Print state every N steps (~1 second at 60 FPS)
RUN_INTERACTIVE = True


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
class LightingParams:
    """Parameters for scene lighting."""
    distant_intensity: float = 1000.0
    dome_intensity: float = 300.0
    angle: float = 315.0


@dataclass
class SimulationParams:
    """Parameters controlling simulation behavior."""
    print_interval: int = 60
    run_interactive: bool = True


@dataclass
class TrajectoryParams:
    """Parameters for robot trajectory control."""
    amplitude: float = 0.5
    frequency: float = 0.2
    enabled: bool = True


@dataclass
class IKParams:
    """Parameters for inverse kinematics control."""
    enabled: bool = False
    use_differential: bool = False
    method: str = "damped-least-squares"
    scale: float = 1.0
    damping: float = 0.05
    min_singular_value: float = 1e-5
    frame_name: str = "tcp_link"
    radius: float = 0.05
    height_offset: float = 0.0


# ============================================================================
# MANIPULATOR CLASS
# ============================================================================

class Manipulator:
    """
    Represents a robot manipulator imported from USD.
    
    Provides methods for:
    - Importing USD model
    - Setting up articulation controller
    - Controlling joint positions
    - Monitoring end-effector state
    - Solving IK targets
    """
    
    def __init__(
        self,
        prim_path: str,
        model_file: str,
        transform_params: TransformParams,
        ee_link_path: str,
        ee_frame_name: str,
    ):
        """Initialize manipulator with model and transform parameters."""
        self.prim_path = prim_path
        self.model_file = model_file
        self.transform_params = transform_params
        self.ee_link_path = ee_link_path
        self.ee_frame_name = ee_frame_name
        self.stage = omni.usd.get_context().get_stage()
        self.robot = None
        self.ee_xformable = None
        self._prim = None
        self.ik_solver: Optional[ArticulationKinematicsSolver] = None
        self.initial_ee_pos: Optional[np.ndarray] = None
        self.initial_ee_orientation: Optional[np.ndarray] = None
        self.ee_link_index: Optional[int] = None
        self.num_arm_dof: int = 6  # SO100 has 6 DOF arm
    
    def get_prim(self):
        """Get the USD prim for this manipulator (with caching)."""
        if self._prim is None:
            self._prim = self.stage.GetPrimAtPath(self.prim_path)
        return self._prim
    
    def import_model(self):
        """Import the manipulator USD file into the scene."""
        print(f"Importing manipulator from: {self.model_file}")
        
        usd_context = omni.usd.get_context()
        omni.kit.commands.execute(
            'CreateReferenceCommand',
            usd_context=usd_context,
            path_to=self.prim_path,
            asset_path=self.model_file,
            instanceable=False
        )
        print(f"✓ Model imported at {self.prim_path}")
    
    def apply_transform(self):
        """Apply position, rotation, and scale to the manipulator."""
        prim = self.get_prim()
        xformable = UsdGeom.Xformable(prim)
        
        # Clear existing transforms
        xformable.ClearXformOpOrder()
        
        # Apply transforms: translate, rotate, scale
        xformable.AddTranslateOp().Set(Gf.Vec3d(*self.transform_params.position))
        if self.transform_params.rotateX is not None:
            xformable.AddRotateXOp().Set(self.transform_params.rotateX)
        xformable.AddScaleOp().Set(Gf.Vec3f(*self.transform_params.scale))
        
        print(f"✓ Transform applied: pos={self.transform_params.position}, scale={self.transform_params.scale}")
    
    def initialize_robot(self, world: World):
        """Initialize robot articulation controller."""
        print("\nInitializing robot controller...")
        
        # Create Robot object to control articulation
        self.robot = Robot(prim_path=self.prim_path, name="manipulator")
        world.scene.add(self.robot)
        self.robot.initialize()
        
        # Print DOF information
        num_dof = self.robot.num_dof
        dof_names = self.robot.dof_names
        print(f"✓ Robot has {num_dof} DOF:")
        for i, name in enumerate(dof_names):
            print(f"  Joint {i}: {name}")
        
        # Get initial joint positions
        initial_positions = self.robot.get_joint_positions()
        print(f"Initial joint positions: {initial_positions}")
        
        return initial_positions
    
    def initialize_end_effector(self):
        """Initialize end-effector monitoring."""
        print(f"\nLooking for end-effector at: {self.ee_link_path}")
        ee_prim = self.stage.GetPrimAtPath(self.ee_link_path)
        
        if ee_prim.IsValid():
            print(f"✓ End-effector found!")
            self.ee_xformable = UsdGeom.Xformable(ee_prim)
            
            # Get end-effector link index for Jacobian extraction
            if self.robot:
                # Debug: print all available link names
                all_links = self.robot.get_link_names()
                print(f"  Available link names: {all_links}")
                
                ee_indices = self.robot.get_link_indices(self.ee_frame_name)
                print(f"  Searching for link name: '{self.ee_frame_name}'")
                print(f"  Result from get_link_indices: {ee_indices}")
                
                if ee_indices is not None and len(ee_indices.list()) > 0:
                    self.ee_link_index = ee_indices.list()[0]
                    print(f"  ✓ End-effector link index: {self.ee_link_index}")
                else:
                    print(f"  ✗ Could not get link index for '{self.ee_frame_name}'")
            
            # Get initial transform
            initial_xform = self.ee_xformable.ComputeLocalToWorldTransform(0)
            initial_pos = initial_xform.ExtractTranslation()
            self.initial_ee_pos = np.array([initial_pos[0], initial_pos[1], initial_pos[2]], dtype=np.float32)
            
            rotation = initial_xform.ExtractRotation().GetQuat()
            # Store in WXYZ format for PyTorch compatibility
            self.initial_ee_orientation = np.array([
                rotation.GetReal(),
                rotation.GetImaginary()[0],
                rotation.GetImaginary()[1],
                rotation.GetImaginary()[2]
            ], dtype=np.float32)
            
            print(
                f"  ✓ Initial EE world position: "
                f"({self.initial_ee_pos[0]:.3f}, {self.initial_ee_pos[1]:.3f}, {self.initial_ee_pos[2]:.3f})"
            )
            print(
                f"  ✓ Initial EE orientation (WXYZ): "
                f"({self.initial_ee_orientation[0]:.3f}, {self.initial_ee_orientation[1]:.3f}, "
                f"{self.initial_ee_orientation[2]:.3f}, {self.initial_ee_orientation[3]:.3f})"
            )
        else:
            print(f"✗ End-effector NOT found at {self.ee_link_path}")
            self.ee_xformable = None
            self.initial_ee_pos = None
            self.initial_ee_orientation = None
            self.ee_link_index = None
    
    def initialize_ik_solver(self):
        """Initialize inverse kinematics solver."""
        if not self.robot:
            print("✗ Cannot initialize IK solver: robot not initialized")
            return False
        
        try:
            print(f"\nInitializing IK solver for frame '{self.ee_frame_name}'...")
            self.ik_solver = ArticulationKinematicsSolver(
                robot_articulation=self.robot,
                end_effector_frame_name=self.ee_frame_name
            )
            print("✓ IK solver initialized successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize IK solver: {e}")
            print("  Falling back to joint-space trajectory control.")
            self.ik_solver = None
            return False
    
    def compute_inverse_kinematics(self, target_position, target_orientation=None):
        """Compute joint positions for a desired EE pose."""
        if not self.ik_solver:
            return None
        
        if target_orientation is None:
            target_orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        
        try:
            success, joint_positions = self.ik_solver.compute_inverse_kinematics(
                target_position=np.array(target_position, dtype=np.float32),
                target_orientation=np.array(target_orientation, dtype=np.float32),
            )
            if success:
                return joint_positions
        except Exception as e:
            print(f"✗ IK solve failed: {e}")
        
        return None
    
    def set_joint_positions(self, positions: np.ndarray):
        """Set robot joint positions."""
        if self.robot is not None:
            self.robot.set_joint_positions(positions)
    
    def get_joint_state(self):
        """Get current joint positions and velocities."""
        if self.robot is not None:
            positions = self.robot.get_joint_positions()
            velocities = self.robot.get_joint_velocities()
            return positions, velocities
        return None, None
    
    def get_ee_position(self):
        """Get current end-effector world position."""
        if self.ee_xformable:
            ee_world_xform = self.ee_xformable.ComputeLocalToWorldTransform(0)
            ee_world_pos = ee_world_xform.ExtractTranslation()
            return np.array([ee_world_pos[0], ee_world_pos[1], ee_world_pos[2]])
        return None
    
    def get_ee_pose(self):
        """Get current end-effector world pose (position and orientation in WXYZ format)."""
        if self.ee_xformable:
            ee_world_xform = self.ee_xformable.ComputeLocalToWorldTransform(0)
            ee_world_pos = ee_world_xform.ExtractTranslation()
            position = np.array([ee_world_pos[0], ee_world_pos[1], ee_world_pos[2]], dtype=np.float32)
            
            rotation = ee_world_xform.ExtractRotation().GetQuat()
            # WXYZ format
            orientation = np.array([
                rotation.GetReal(),
                rotation.GetImaginary()[0],
                rotation.GetImaginary()[1],
                rotation.GetImaginary()[2]
            ], dtype=np.float32)
            
            return position, orientation
        return None, None
    
    def compute_differential_ik(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        method: str = "damped-least-squares",
        method_cfg: dict = None
    ) -> Optional[np.ndarray]:
        """Compute differential IK using Jacobian matrices."""
        if self.robot is None or self.ee_link_index is None:
            return None
        
        if method_cfg is None:
            method_cfg = {"scale": 1.0, "damping": 0.05, "min_singular_value": 1e-5}
        
        # Get current state
        current_pos, current_ori = self.get_ee_pose()
        if current_pos is None:
            return None
        
        # Get current DOF positions
        current_dof_positions = wp.to_torch(self.robot.get_dof_positions())
        
        # Get Jacobian matrix
        jacobian_matrices = wp.to_torch(self.robot.get_jacobian_matrices())
        # Extract Jacobian for end-effector and arm DOFs only
        jacobian_ee = jacobian_matrices[:, self.ee_link_index - 1, :, :self.num_arm_dof]
        
        # Convert to torch tensors
        device = jacobian_ee.device
        current_pos_t = torch.from_numpy(current_pos).unsqueeze(0).to(device)
        current_ori_t = torch.from_numpy(current_ori).unsqueeze(0).to(device)
        target_pos_t = torch.from_numpy(target_position).unsqueeze(0).to(device)
        target_ori_t = (
            torch.from_numpy(target_orientation).unsqueeze(0).to(device)
            if target_orientation is not None
            else None
        )
        
        # Compute delta DOF positions
        delta_dof = differential_inverse_kinematics(
            jacobian_end_effector=jacobian_ee,
            current_position=current_pos_t,
            current_orientation=current_ori_t,
            goal_position=target_pos_t,
            goal_orientation=target_ori_t,
            method=method,
            method_cfg=method_cfg
        )
        
        # Return new DOF positions (only for arm joints)
        new_positions = current_dof_positions[:, :self.num_arm_dof] + delta_dof
        return new_positions.cpu().numpy()
    
    def setup(self):
        """Complete setup: import model and apply transforms."""
        self.import_model()
        self.apply_transform()


# ============================================================================
# SCENE MANAGER CLASS
# ============================================================================

class SceneManager:
    """
    Manages the overall simulation scene.
    
    Responsibilities:
    - Initialize physics world
    - Add lighting
    - Manage manipulator
    - Run simulation loop
    """
    
    def __init__(
        self,
        lighting_params: LightingParams,
        simulation_params: SimulationParams,
    ):
        """Initialize scene manager with configuration."""
        self.lighting_params = lighting_params
        self.simulation_params = simulation_params
        self.world = None
        self.stage = None
        self.manipulator = None
    
    def initialize(self):
        """Initialize the physics world."""
        print("Initializing world...")
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        self.stage = omni.usd.get_context().get_stage()
        print("✓ World initialized with ground plane")
    
    def add_lighting(self):
        """Add lighting to the scene."""
        print("Adding lights...")
        
        # Distant light (sun)
        distant_light = UsdLux.DistantLight.Define(self.stage, "/World/DistantLight")
        distant_light.CreateIntensityAttr(self.lighting_params.distant_intensity)
        distant_light.AddRotateXYZOp().Set(Gf.Vec3f(self.lighting_params.angle, 0, 0))
        
        # Dome light (ambient)
        dome_light = UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(self.lighting_params.dome_intensity)
        
        print("✓ Lights added")
    
    def add_manipulator(self, manipulator: Manipulator):
        """Add manipulator to scene and set it up."""
        self.manipulator = manipulator
        manipulator.setup()
    
    def run_simulation(self, trajectory_params: TrajectoryParams, ik_params: IKParams):
        """
        Run the simulation loop with trajectory control.
        
        Args:
            trajectory_params: Parameters for joint trajectory generation
            ik_params: Parameters for IK-driven EE control
        """
        # Reset world to initialize physics
        print("\nResetting world...")
        self.world.reset()
        
        # Initialize robot controller
        initial_positions = self.manipulator.initialize_robot(self.world)
        
        # Step simulation once to update all transforms
        self.world.step(render=False)
        
        # Initialize end-effector monitoring (after stepping to ensure transforms are updated)
        self.manipulator.initialize_end_effector()
        
        # Initialize IK solver if requested
        ik_active = False
        use_differential_ik = False
        if ik_params.enabled:
            if ik_params.use_differential:
                # Use differential IK (Jacobian-based)
                if self.manipulator.ee_link_index is not None and self.manipulator.initial_ee_pos is not None:
                    ik_active = True
                    use_differential_ik = True
                    print("✓ Using differential IK (Jacobian-based)")
                    print(f"  Method: {ik_params.method}")
                    print(f"  Scale: {ik_params.scale}, Damping: {ik_params.damping}")
                else:
                    print("✗ Differential IK disabled: end-effector link index or pose unavailable.")
                    if self.manipulator.ee_link_index is None:
                        print("  - End-effector link index is None")
                    if self.manipulator.initial_ee_pos is None:
                        print("  - Initial EE position is None")
            else:
                # Use ArticulationKinematicsSolver
                ik_active = self.manipulator.initialize_ik_solver()
                if ik_active and self.manipulator.initial_ee_pos is None:
                    print("✗ IK target disabled: end-effector pose unavailable.")
                    ik_active = False
        
        print("\n" + "="*70)
        print("SUCCESS! Manipulator loaded with articulation controller.")
        if ik_active:
            ik_type = "Differential IK (Jacobian)" if use_differential_ik else "ArticulationKinematicsSolver"
            print(f"IK Method: {ik_type}")
            print(
                f"EE motion: horizontal IK circle (radius={ik_params.radius:.3f} m, "
                f"freq={trajectory_params.frequency} Hz)"
            )
        elif trajectory_params.enabled:
            print(
                f"Joint motion: Sinusoidal (amplitude={trajectory_params.amplitude} rad, "
                f"freq={trajectory_params.frequency} Hz)"
            )
        else:
            print("No automated motion enabled.")
        print("Interact with the 3D view. Close window when done.")
        print("="*70 + "\n")
        
        # Simulation loop
        step_count = 0
        dt = 1.0 / 60.0
        num_dof = self.manipulator.robot.num_dof if self.manipulator.robot else 0
        
        while simulation_app.is_running():
            self.world.step(render=True)
            time = step_count * dt
            
            if ik_active and self.manipulator.initial_ee_pos is not None:
                # Horizontal circular trajectory for EE
                horizontal_offset = np.array([
                    ik_params.radius * math.cos(2 * math.pi * trajectory_params.frequency * time),
                    ik_params.radius * math.sin(2 * math.pi * trajectory_params.frequency * time),
                    ik_params.height_offset
                ], dtype=np.float32)
                target_pos = self.manipulator.initial_ee_pos + horizontal_offset
                target_ori = (
                    self.manipulator.initial_ee_orientation
                    if self.manipulator.initial_ee_orientation is not None
                    else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # WXYZ format
                )
                
                if use_differential_ik:
                    # Use differential IK (Jacobian-based)
                    method_cfg = {
                        "scale": ik_params.scale,
                        "damping": ik_params.damping,
                        "min_singular_value": ik_params.min_singular_value
                    }
                    ik_solution = self.manipulator.compute_differential_ik(
                        target_position=target_pos,
                        target_orientation=target_ori,
                        method=ik_params.method,
                        method_cfg=method_cfg
                    )
                    if ik_solution is not None:
                        # Only set arm DOF positions (first num_arm_dof joints)
                        current_positions = self.manipulator.robot.get_joint_positions()
                        new_positions = np.array(current_positions)
                        new_positions[:self.manipulator.num_arm_dof] = ik_solution[0]
                        self.manipulator.set_joint_positions(new_positions)
                else:
                    # Use ArticulationKinematicsSolver
                    ik_solution = self.manipulator.compute_inverse_kinematics(target_pos, target_ori)
                    if ik_solution is not None:
                        self.manipulator.set_joint_positions(ik_solution)
            elif trajectory_params.enabled and num_dof > 0:
                # Apply sinusoidal joint motion
                target_positions = np.zeros(num_dof)
                for i in range(num_dof):
                    phase_offset = i * (2 * math.pi / num_dof)
                    target_positions[i] = initial_positions[i] + trajectory_params.amplitude * math.sin(
                        2 * math.pi * trajectory_params.frequency * time + phase_offset
                    )
                self.manipulator.set_joint_positions(target_positions)
            
            # Print state at regular intervals
            if step_count % self.simulation_params.print_interval == 0:
                current_positions, current_velocities = self.manipulator.get_joint_state()
                ee_position = self.manipulator.get_ee_position()
                
                print(f"\n--- Step {step_count} (t={time:.2f}s) ---")
                if current_positions is not None:
                    print(f"  Joint positions: {np.round(current_positions, 3)}")
                if current_velocities is not None:
                    print(f"  Joint velocities: {np.round(current_velocities, 3)}")
                if ee_position is not None:
                    print(f"  EE position: ({ee_position[0]:.3f}, {ee_position[1]:.3f}, {ee_position[2]:.3f})")
            
            step_count += 1


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main execution flow - Import and control manipulator.
    
    Steps:
    1. Create scene manager
    2. Initialize world and lighting
    3. Import manipulator with transforms
    4. Run simulation with trajectory/IK control
    """
    print("=" * 70)
    print("Manipulator Control Test")
    print("=" * 70)
    
    try:
        # ===== STEP 1: Create Scene Manager =====
        scene = SceneManager(
            lighting_params=LightingParams(
                distant_intensity=DISTANT_LIGHT_INTENSITY,
                dome_intensity=DOME_LIGHT_INTENSITY,
                angle=DISTANT_LIGHT_ANGLE,
            ),
            simulation_params=SimulationParams(
                print_interval=PRINT_INTERVAL,
                run_interactive=RUN_INTERACTIVE,
            ),
        )
        
        # ===== STEP 2: Initialize World =====
        scene.initialize()
        
        # ===== STEP 3: Add Lighting =====
        scene.add_lighting()
        
        # ===== STEP 4: Create Manipulator =====
        manipulator = Manipulator(
            prim_path="/World/Manipulator",
            model_file=MANIPULATOR_USD_FILE,
            transform_params=TransformParams(
                position=MANIPULATOR_POSITION,
                scale=MANIPULATOR_SCALE,
                rotateX=MANIPULATOR_ROTATION_X,
            ),
            ee_link_path=EE_LINK_PATH,
            ee_frame_name=EE_FRAME_NAME,
        )
        scene.add_manipulator(manipulator)
        
        # ===== STEP 5: Run Simulation =====
        trajectory_params = TrajectoryParams(
            amplitude=TRAJECTORY_AMPLITUDE,
            frequency=TRAJECTORY_FREQUENCY,
            enabled=TRAJECTORY_ENABLED,
        )
        ik_params = IKParams(
            enabled=IK_ENABLED,
            use_differential=IK_USE_DIFFERENTIAL,
            method=IK_METHOD,
            scale=IK_SCALE,
            damping=IK_DAMPING,
            min_singular_value=IK_MIN_SINGULAR_VALUE,
            frame_name=EE_FRAME_NAME,
            radius=IK_TARGET_RADIUS,
            height_offset=IK_HEIGHT_OFFSET,
        )
        scene.run_simulation(trajectory_params, ik_params)
    
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