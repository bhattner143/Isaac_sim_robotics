"""
Combined Scene: Plate Dips, Manipulator, and Ball
Object-Oriented Implementation using Isaac Sim

This script demonstrates:
1. Importing plate_dips USD model
2. Importing manipulator (SO101) USD model  
3. Creating a dynamic ball with physics
4. Scene management with lighting
5. Independent object control and positioning

CONFIGURATION:
- All parameters are at the top in the USER CONFIGURATION section
- Easy to modify positions, scales, and physics properties
"""

# Import standard Python libraries
from isaacsim import SimulationApp
from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path
from abc import ABC, abstractmethod

# STEP 1: Launch Isaac Sim application
simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
    "renderer": "RayTracedLighting",
    "active_gpu": 0,
})

# STEP 2: Import Isaac Sim modules (after SimulationApp)
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from pxr import Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema, Sdf, UsdShade, UsdLux
import omni.usd
import omni.kit.commands
import math
import numpy as np

# Import IK solver
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver


# ============================================================================
# USER CONFIGURATION - EDIT THESE PARAMETERS
# ============================================================================

# --- Plate Configuration ---
PLATE_USD_FILE = str(Path("plate_dips/part_dips_coarse_rot.usd").absolute())
PLATE_POSITION = (0.0, 0.0, 0.1)
PLATE_SCALING_FACTOR = 1.0
PLATE_SCALE = (PLATE_SCALING_FACTOR * 10.0, PLATE_SCALING_FACTOR * 10.0, PLATE_SCALING_FACTOR * 10.0)
PLATE_ROTATION_X = 0
PLATE_COLOR = (0.0, 1.0, 0.0)  # Green
PLATE_STATIC_FRICTION = 0.6
PLATE_DYNAMIC_FRICTION = 0.5
PLATE_RESTITUTION = 0.1
PLATE_MECHANICS = "kinematic"  # "static", "kinematic", or "dynamic"
PLATE_COLLISION_TYPE = "convexDecomposition"

# --- Ball Configuration ---
BALL_POSITION = (0.0, 0.0, 2.0)  # Position above the plate center
BALL_SCALING_FACTOR = 1.0
BALL_SCALE = (BALL_SCALING_FACTOR * 1.0, BALL_SCALING_FACTOR * 1.0, BALL_SCALING_FACTOR * 1.0)
BALL_RADIUS = 0.25  # 5 cm radius
BALL_COLOR = (2.0, 0.0, 0.0)  # Red
BALL_STATIC_FRICTION = 0.6
BALL_DYNAMIC_FRICTION = 0.5
BALL_RESTITUTION = 0.3
BALL_MASS = 0.1
BALL_IS_DYNAMIC = True

# --- Manipulator Configuration ---
MANIPULATOR_USD_FILE = str(Path("manipulators/so101_physics.usd").absolute())
MANIPULATOR_POSITION = (0, 2.5, 0.1)
MANIPULATOR_SCALE_FACTOR = 5.0
MANIPULATOR_SCALE = (MANIPULATOR_SCALE_FACTOR * 1.0, MANIPULATOR_SCALE_FACTOR * 1.0, MANIPULATOR_SCALE_FACTOR * 1.0)
MANIPULATOR_ROTATION_X = 0
MANIPULATOR_COLOR = (0.0, 0.5, 1.0)  # Blue
MANIPULATOR_STATIC_FRICTION = 0.6
MANIPULATOR_DYNAMIC_FRICTION = 0.5
MANIPULATOR_RESTITUTION = 0.1
MANIPULATOR_MECHANICS = "static"
MANIPULATOR_COLLISION_TYPE = "convexDecomposition"

# --- IK Trajectory Configuration ---
IK_TRAJECTORY_AMPLITUDE = 1.0  # Horizontal trajectory amplitude in meters
IK_TRAJECTORY_FREQUENCY = 0.1  # Hz
IK_FALLBACK_JOINT_AMPLITUDE = 0.3  # Fallback joint amplitude if IK fails

# --- Scene Configuration ---
DISTANT_LIGHT_INTENSITY = 1000.0
DOME_LIGHT_INTENSITY = 300.0
DISTANT_LIGHT_ANGLE = -45.0

# --- Simulation Configuration ---
PRINT_INTERVAL = 100
ENABLE_CCD = True
CONTACT_OFFSET = 0.02
REST_OFFSET = 0.0

# --- Plate Motion Configuration ---
SETTLE_TIME = 3.0
PLATE_MOTION_AMPLITUDE = 2.0
PLATE_MOTION_FREQUENCY = 0.1
PLATE_MOTION_AXIS = 0  # 0=X, 1=Y, 2=Z
PLATE_MOTION_ENABLED = True

# --- Visual Material Properties ---
ROUGHNESS = 0.4
METALLIC = 0.0


# ============================================================================
# PARAMETER CLASSES (Dataclasses)
# ============================================================================

@dataclass
class TransformParams:
    """Parameters for positioning, rotating, and scaling objects in 3D space."""
    position: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    rotateX: Optional[float] = None


@dataclass
class PhysicsMaterialParams:
    """Parameters controlling surface interactions (friction and bounce)."""
    static_friction: float
    dynamic_friction: float
    restitution: float


@dataclass
class VisualMaterialParams:
    """Parameters controlling visual appearance (color and surface properties)."""
    color: Tuple[float, float, float]
    roughness: float = 0.4
    metallic: float = 0.0


@dataclass
class PhysicsBodyParams:
    """Parameters defining physics body behavior."""
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
class LightingParams:
    """Parameters for scene lighting."""
    distant_intensity: float = 1000.0
    dome_intensity: float = 300.0
    angle: float = -45.0


@dataclass
class SimulationParams:
    """Parameters controlling simulation behavior."""
    print_interval: int = 100
    run_interactive: bool = True
    settle_time: float = 3.0
    plate_motion_amplitude: float = 2.0
    plate_motion_frequency: float = 0.1
    plate_motion_axis: int = 0
    plate_motion_enabled: bool = True


# ============================================================================
# BASE CLASS: PhysicsObject
# ============================================================================

class PhysicsObject:
    """
    Base class for all physics objects in the scene.
    Provides common functionality for transforms, materials, and physics.
    """
    
    def __init__(
        self,
        prim_path: str,
        transform_params: TransformParams,
        physics_material_params: PhysicsMaterialParams,
        visual_material_params: VisualMaterialParams,
        physics_body_params: PhysicsBodyParams,
    ):
        self.prim_path = prim_path
        self.transform_params = transform_params
        self.physics_material_params = physics_material_params
        self.visual_material_params = visual_material_params
        self.physics_body_params = physics_body_params
        self.stage = omni.usd.get_context().get_stage()
        self._prim = None
    
    def get_prim(self):
        """Get the USD prim for this object (with caching)."""
        if self._prim is None:
            self._prim = self.stage.GetPrimAtPath(self.prim_path)
        return self._prim
    
    def apply_transform(self):
        """Apply position, rotation, and scale to the object."""
        prim = self.get_prim()
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        
        # Position
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*self.transform_params.position))
        
        # Rotation
        if self.transform_params.rotateX is not None:
            rotate_op = xformable.AddRotateXOp()
            rotate_op.Set(self.transform_params.rotateX)
        
        # Scale
        scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3f(*self.transform_params.scale))
    
    def create_visual_material(self):
        """Create and apply a PBR material."""
        material_path = f"{self.prim_path}/Looks/Material"
        material = UsdShade.Material.Define(self.stage, material_path)
        
        shader_path = f"{material_path}/Shader"
        shader = UsdShade.Shader.Define(self.stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*self.visual_material_params.color)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            self.visual_material_params.roughness
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            self.visual_material_params.metallic
        )
        
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.GetOutput("surface"))
        
        self._bind_material_to_meshes(self.get_prim(), material)
    
    def _bind_material_to_meshes(self, prim, material):
        """Recursively bind material to all mesh children."""
        if prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI(prim).Bind(material)
        
        for child in prim.GetChildren():
            self._bind_material_to_meshes(child, material)
    
    def apply_physics_material(self):
        """Apply physics material (friction and restitution)."""
        material_path = f"{self.prim_path}/PhysicsMaterial"
        physics_material = UsdPhysics.MaterialAPI.Apply(
            self.stage.DefinePrim(material_path, "Material")
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
        """Apply physics properties - must be implemented by child classes."""
        raise NotImplementedError("Child classes must implement apply_physics()")
    
    def setup(self):
        """Complete setup process for the object."""
        self.apply_transform()
        self.create_visual_material()
        self.apply_physics()
        self.apply_physics_material()


# ============================================================================
# USD PHYSICS OBJECT CLASS (abstract base for USD-imported objects)
# ============================================================================

class USDPhysicsObject(PhysicsObject, ABC):
    """
    Abstract base class for physics objects imported from USD files.
    
    This class contains common functionality for both Plate and Manipulator:
    - USD file import
    - Mesh collision physics
    - Configurable mechanics modes (static, kinematic, dynamic)
    
    Child classes MUST implement get_object_type() to customize logging.
    """
    
    def __init__(
        self,
        prim_path: str,
        model_file: str,
        transform_params: TransformParams,
        physics_material_params: PhysicsMaterialParams,
        visual_material_params: VisualMaterialParams,
        physics_body_params: PhysicsBodyParams,
        collision_params: CollisionParams,
        mechanics_mode: str = 'static',
    ):
        super().__init__(
            prim_path,
            transform_params,
            physics_material_params,
            visual_material_params,
            physics_body_params,
        )
        self.model_file = model_file
        self.collision_params = collision_params
        self.mechanics_mode = mechanics_mode
    
    @abstractmethod
    def get_object_type(self) -> str:
        """Return the object type name for logging. MUST be implemented by child classes."""
        pass
    
    def import_model(self):
        """Import the USD model file into the scene."""
        object_type = self.get_object_type()
        print(f"Importing {object_type} from: {self.model_file}")
        
        usd_context = omni.usd.get_context()
        omni.kit.commands.execute(
            'CreateReferenceCommand',
            usd_context=usd_context,
            path_to=self.prim_path,
            asset_path=self.model_file,
            instanceable=False
        )
        print(f"✓ {object_type.capitalize()} imported at {self.prim_path}")
    
    def apply_physics(self):
        """Apply physics using mesh collision with configurable mechanics mode."""
        prim = self.get_prim()
        
        # Apply rigid body API
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
            
            if self.mechanics_mode == 'static':
                rigid_body.CreateRigidBodyEnabledAttr().Set(False)
            elif self.mechanics_mode == 'kinematic':
                rigid_body.CreateKinematicEnabledAttr().Set(True)
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                prim.CreateAttribute("physxRigidBody:lockFlags", Sdf.ValueTypeNames.Int).Set(56)
            else:  # dynamic
                prim.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(10000.0)
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                prim.CreateAttribute("physxRigidBody:lockFlags", Sdf.ValueTypeNames.Int).Set(56)
        
        # Recursively apply collision to all meshes
        self._apply_physics_recursive(prim)
    
    def _apply_physics_recursive(self, prim):
        """Recursively apply physics collision to all mesh children."""
        if prim.IsA(UsdGeom.Mesh):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(prim)
                
                if self.collision_params.approximation != "none":
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                    mesh_collision_api.CreateApproximationAttr().Set(
                        self.collision_params.approximation
                    )
            
            if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                physx_collision.CreateContactOffsetAttr().Set(
                    self.physics_body_params.contact_offset
                )
                physx_collision.CreateRestOffsetAttr().Set(
                    self.physics_body_params.rest_offset
                )
        
        for child in prim.GetChildren():
            self._apply_physics_recursive(child)
    
    def setup(self):
        """Override setup to import model first, then apply standard setup."""
        self.import_model()
        super().setup()


# ============================================================================
# PLATE CLASS (inherits from USDPhysicsObject)
# ============================================================================

class Plate(USDPhysicsObject):
    """
    Represents the plate_dips object imported from USD file.
    Inherits all USD import and physics functionality from USDPhysicsObject.
    """
    
    def get_object_type(self) -> str:
        """Return object type for logging."""
        return "plate"


# ============================================================================
# MANIPULATOR CLASS (inherits from USDPhysicsObject)
# ============================================================================

class Manipulator(USDPhysicsObject):
    """
    Represents the manipulator (robot arm) imported from USD file.
    Inherits all USD import and physics functionality from USDPhysicsObject.
    Adds articulation control and inverse kinematics for end-effector trajectory following.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.robot = None
        self.initial_joint_positions = None
        self.ee_xformable = None
        self.initial_ee_position = None
        self.ik_solver = None
    
    def get_object_type(self) -> str:
        """Return object type for logging."""
        return "manipulator"
    
    def initialize_robot(self, world: World):
        """Initialize robot articulation controller."""
        print(f"\nInitializing robot controller for {self.prim_path}...")
        
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
        self.initial_joint_positions = self.robot.get_joint_positions()
        print(f"Initial joint positions: {self.initial_joint_positions}")
        
        return self.initial_joint_positions
    
    def initialize_ik_solver(self):
        """Initialize inverse kinematics solver."""
        try:
            print(f"\nInitializing IK solver for {self.prim_path}...")
            self.ik_solver = ArticulationKinematicsSolver(
                robot_articulation=self.robot,
                end_effector_frame_name="tcp_link"  # Adjust based on your robot
            )
            print("✓ IK solver initialized successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize IK solver: {e}")
            print("  Will use coordinated joint motion instead")
            return False
    
    def initialize_end_effector(self):
        """Initialize end-effector monitoring."""
        # Try to find the end effector prim
        prim = self.get_prim()
        for child in Usd.PrimRange(prim):
            path_str = str(child.GetPath()).lower()
            if "wrist" in path_str or "tcp" in path_str or "tool" in path_str or "ee" in path_str:
                print(f"✓ Found end-effector at: {child.GetPath()}")
                self.ee_xformable = UsdGeom.Xformable(child)
                
                # Get initial transform
                initial_xform = self.ee_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                initial_pos = initial_xform.ExtractTranslation()
                self.initial_ee_position = np.array([initial_pos[0], initial_pos[1], initial_pos[2]])
                print(f"  Initial EE world position: ({initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f})")
                return
        
        print(f"✗ End-effector not found")
    
    def compute_ik(self, target_position, target_orientation=None):
        """Compute inverse kinematics for target end-effector position."""
        if self.ik_solver is None:
            return None
        
        try:
            # Solve IK for position (and orientation if provided)
            action = self.ik_solver.compute_inverse_kinematics(
                target_position=target_position,
                target_orientation=target_orientation
            )
            return action.joint_positions if action else None
        except Exception as e:
            return None
    
    def set_joint_positions(self, positions):
        """Set robot joint positions."""
        if self.robot:
            self.robot.set_joint_positions(positions)
    
    def get_joint_state(self):
        """Get current joint positions and velocities."""
        if self.robot:
            positions = self.robot.get_joint_positions()
            velocities = self.robot.get_joint_velocities()
            return positions, velocities
        return None, None
    
    def get_ee_position(self):
        """Get current end-effector world position."""
        if self.ee_xformable:
            ee_world_xform = self.ee_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            ee_world_pos = ee_world_xform.ExtractTranslation()
            return np.array([ee_world_pos[0], ee_world_pos[1], ee_world_pos[2]])
        return None


# ============================================================================
# BALL CLASS (inherits from PhysicsObject)
# ============================================================================

class Ball(PhysicsObject):
    """
    Represents a spherical ball created procedurally.
    Uses analytic sphere collision for accurate physics.
    """
    
    def __init__(
        self,
        prim_path: str,
        radius: float,
        transform_params: TransformParams,
        physics_material_params: PhysicsMaterialParams,
        visual_material_params: VisualMaterialParams,
        physics_body_params: PhysicsBodyParams,
    ):
        super().__init__(
            prim_path,
            transform_params,
            physics_material_params,
            visual_material_params,
            physics_body_params,
        )
        self.radius = radius
    
    def create_geometry(self):
        """Create the sphere geometry procedurally."""
        sphere = UsdGeom.Sphere.Define(self.stage, self.prim_path)
        sphere.CreateRadiusAttr(self.radius)
        sphere.AddTranslateOp().Set(Gf.Vec3d(*self.transform_params.position))
        sphere.AddScaleOp().Set(Gf.Vec3f(*self.transform_params.scale))
        
        extent = [
            (-self.radius, -self.radius, -self.radius),
            (self.radius, self.radius, self.radius)
        ]
        sphere.CreateExtentAttr(extent)
    
    def apply_physics(self):
        """Apply physics using analytic sphere collision."""
        prim = self.get_prim()
        
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        
        prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
        
        if self.physics_body_params.enable_ccd:
            prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
            prim.CreateAttribute("physxRigidBody:contactOffset", Sdf.ValueTypeNames.Float).Set(0.01)
            prim.CreateAttribute("physxRigidBody:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
        
        if self.physics_body_params.mass is not None:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(self.physics_body_params.mass)
    
    def apply_transform(self):
        """Skip transform (already applied in create_geometry)."""
        pass
    
    def setup(self):
        """Override setup to create geometry first."""
        self.create_geometry()
        self.create_visual_material()
        self.apply_physics()
        self.apply_physics_material()


# ============================================================================
# SCENE MANAGER CLASS
# ============================================================================

class SceneManager:
    """Manages the overall simulation scene."""
    
    def __init__(
        self,
        lighting_params: LightingParams,
        simulation_params: SimulationParams,
    ):
        self.lighting_params = lighting_params
        self.simulation_params = simulation_params
        self.world = None
        self.stage = None
        self.objects = []
    
    def initialize(self):
        """Initialize the physics world and USD stage."""
        print("Initializing world...")
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        self.stage = omni.usd.get_context().get_stage()
        print("World initialized with ground plane")
    
    def add_lighting(self):
        """Add lighting to the scene."""
        print("Adding lights...")
        
        # Distant light
        distant_light = UsdLux.DistantLight.Define(self.stage, "/World/DistantLight")
        distant_light.CreateIntensityAttr().Set(self.lighting_params.distant_intensity)
        distant_light.CreateAngleAttr().Set(0.5)
        
        xformable = UsdGeom.Xformable(distant_light)
        xformable.ClearXformOpOrder()
        rotate_op = xformable.AddRotateXOp()
        rotate_op.Set(self.lighting_params.angle)
        
        # Dome light
        dome_light = UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr().Set(self.lighting_params.dome_intensity)
        
        print("Lights added")
    
    def add_object(self, obj: PhysicsObject):
        """Add a physics object to the scene."""
        print(f"Setting up object at {obj.prim_path}...")
        obj.setup()
        self.objects.append(obj)
        print(f"Object {obj.prim_path} setup complete")
    
    def run_simulation(self):
        """Run the simulation loop with kinematic plate motion and manipulator joint control."""
        print("\nStarting simulation...")
        if self.simulation_params.plate_motion_enabled:
            print(f"Plate will start moving after {self.simulation_params.settle_time}s settling time")
            axis_names = ['X', 'Y', 'Z']
            print(f"Sinusoidal motion on {axis_names[self.simulation_params.plate_motion_axis]}-axis:")
            print(f"  Amplitude: {self.simulation_params.plate_motion_amplitude} m")
            print(f"  Frequency: {self.simulation_params.plate_motion_frequency} Hz")
            period = 1.0 / self.simulation_params.plate_motion_frequency
            print(f"  Period: {period:.2f} s")
        print("Close the window to stop.\n")
        
        self.world.reset()
        
        step_count = 0
        motion_started = False
        plate_obj = None
        plate_initial_pos = None
        is_kinematic = None
        manipulator_obj = None
        initial_joint_positions = None
        num_dof = 0
        
        # Find the plate and manipulator objects
        for obj in self.objects:
            if "Plate" in obj.prim_path:
                plate_obj = obj
                mechanics_mode = getattr(obj, 'mechanics_mode', 'static')
                is_kinematic = (mechanics_mode == 'kinematic')
                # Store initial position for kinematic motion
                if is_kinematic:
                    plate_prim = obj.get_prim()
                    xformable = UsdGeom.Xformable(plate_prim)
                    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    plate_initial_pos = list(world_transform.ExtractTranslation())
            elif "Manipulator" in obj.prim_path:
                manipulator_obj = obj
                # Initialize robot articulation
                if hasattr(manipulator_obj, 'initialize_robot'):
                    initial_joint_positions = manipulator_obj.initialize_robot(self.world)
                    manipulator_obj.initialize_end_effector()
                    manipulator_obj.initialize_ik_solver()
                    num_dof = manipulator_obj.robot.num_dof if manipulator_obj.robot else 0
        
        try:
            while simulation_app.is_running():
                self.world.step(render=True)
                step_count += 1
                t = step_count * self.world.get_physics_dt()
                
                # Apply sinusoidal motion to plate after settling
                if self.simulation_params.plate_motion_enabled and plate_obj is not None:
                    if t >= self.simulation_params.settle_time:
                        if not motion_started:
                            print(f"\n{'='*70}")
                            mode = "KINEMATIC (position control)" if is_kinematic else "DYNAMIC (velocity control)"
                            print(f"STARTING SINUSOIDAL PLATE MOTION at t={t:.2f}s - {mode}")
                            print(f"{'='*70}\n")
                            motion_started = True
                        
                        motion_time = t - self.simulation_params.settle_time
                        amplitude = self.simulation_params.plate_motion_amplitude
                        frequency = self.simulation_params.plate_motion_frequency
                        axis = self.simulation_params.plate_motion_axis
                        angular_freq = 2 * math.pi * frequency
                        plate_prim = plate_obj.get_prim()
                        
                        if is_kinematic:
                            # KINEMATIC MODE: Direct position control
                            offset = amplitude * math.sin(angular_freq * motion_time)
                            new_pos = plate_initial_pos.copy()
                            new_pos[axis] = plate_initial_pos[axis] + offset
                            
                            xformable = UsdGeom.Xformable(plate_prim)
                            xformable.ClearXformOpOrder()
                            
                            # STEP 1: Apply translation
                            translate_op = xformable.AddTranslateOp()
                            translate_op.Set(Gf.Vec3d(*new_pos))

                            # STEP 2: Apply fixed 90 degree rotation around X-axis (to make plate horizontal)
                            rotate_op = xformable.AddRotateXOp()
                            rotate_op.Set(90.0)
                             
                            # STEP 3: Reapply scale
                            scale_vec = (
                                tuple(plate_obj.transform_params.scale)
                                if hasattr(plate_obj, "transform_params") and plate_obj.transform_params.scale
                                else (1.0, 1.0, 1.0)
                            )
                            scale_op = xformable.AddScaleOp()
                            scale_op.Set(Gf.Vec3f(*scale_vec))
                            
                            # CRITICAL: Set velocity for kinematic object so it can push other objects
                            # Velocity = A * ω * cos(ωt) - derivative of position
                            computed_velocity = [0.0, 0.0, 0.0]
                            computed_velocity[axis] = amplitude * angular_freq * math.cos(angular_freq * motion_time)
                            
                            rigid_body = UsdPhysics.RigidBodyAPI(plate_prim)
                            rigid_body.GetVelocityAttr().Set(Gf.Vec3f(*computed_velocity))
                            rigid_body.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                        else:
                            # DYNAMIC MODE: Velocity control
                            velocity_magnitude = amplitude * angular_freq * math.cos(angular_freq * motion_time)
                            velocity = [0.0, 0.0, 0.0]
                            velocity[axis] = velocity_magnitude
                            
                            rigid_body = UsdPhysics.RigidBodyAPI(plate_prim)
                            rigid_body.GetVelocityAttr().Set(Gf.Vec3f(*velocity))
                            rigid_body.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                        
                        # Move manipulator end effector using inverse kinematics
                        if manipulator_obj is not None and manipulator_obj.initial_ee_position is not None:
                            motion_time = t - self.simulation_params.settle_time
                            
                            # Compute desired end-effector position (horizontal trajectory)
                            # Move left-right along X-axis
                            amplitude = IK_TRAJECTORY_AMPLITUDE
                            frequency = IK_TRAJECTORY_FREQUENCY
                            
                            # Horizontal sinusoidal trajectory
                            x_offset = amplitude * math.sin(2 * math.pi * frequency * motion_time)
                            
                            target_ee_position = manipulator_obj.initial_ee_position.copy()
                            target_ee_position[0] += x_offset  # Move along X-axis (left-right)
                            
                            # Try to use IK if solver is available
                            if manipulator_obj.ik_solver is not None:
                                joint_positions = manipulator_obj.compute_ik(target_ee_position)
                                
                                if joint_positions is not None:
                                    manipulator_obj.set_joint_positions(joint_positions)
                                else:
                                    # IK failed, use fallback joint motion
                                    if initial_joint_positions is not None and num_dof > 0:
                                        target_positions = np.zeros(num_dof)
                                        for i in range(num_dof):
                                            phase_offset = i * (2 * math.pi / num_dof)
                                            target_positions[i] = initial_joint_positions[i] + IK_FALLBACK_JOINT_AMPLITUDE * math.sin(
                                                2 * math.pi * frequency * motion_time + phase_offset
                                            )
                                        manipulator_obj.set_joint_positions(target_positions)
                            else:
                                # No IK solver, use coordinated joint space motion
                                if initial_joint_positions is not None and num_dof > 0:
                                    target_positions = np.zeros(num_dof)
                                    for i in range(num_dof):
                                        phase_offset = i * (2 * math.pi / num_dof)
                                        target_positions[i] = initial_joint_positions[i] + IK_FALLBACK_JOINT_AMPLITUDE * math.sin(
                                            2 * math.pi * frequency * motion_time + phase_offset
                                        )
                                    manipulator_obj.set_joint_positions(target_positions)
                
                # Print positions periodically
                if step_count % self.simulation_params.print_interval == 0:
                    for obj in self.objects:
                        prim = obj.get_prim()
                        xformable = UsdGeom.Xformable(prim)
                        world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                        position = world_transform.ExtractTranslation()
                        
                        print(f"Step {step_count:5d} | Time {t:6.2f}s | "
                              f"{obj.prim_path}: ({position[0]:6.3f}, "
                              f"{position[1]:6.3f}, {position[2]:6.3f})")
                    
                    # Print manipulator state if available
                    if manipulator_obj is not None and hasattr(manipulator_obj, 'get_joint_state'):
                        joint_pos, joint_vel = manipulator_obj.get_joint_state()
                        ee_pos = manipulator_obj.get_ee_position()
                        if joint_pos is not None:
                            print(f"  Manipulator joints: {np.round(joint_pos, 3)}")
                        if ee_pos is not None:
                            print(f"  End-effector: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
        
        except KeyboardInterrupt:
            print("\nStopping simulation...")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main execution flow - Complete scene with plate, manipulator, and ball.
    
    This script:
    1. Creates a scene manager
    2. Sets up the world and lighting
    3. Imports the plate_dips model
    4. Imports the manipulator model
    5. Creates a dynamic ball that falls onto the plate
    6. Runs interactive simulation
    """
    print("=" * 70)
    print("Combined Scene: Plate Dips and Manipulator")
    print("=" * 70)
    
    try:
        # ===== STEP 1: Initialize Scene Manager =====
        scene = SceneManager(
            lighting_params=LightingParams(
                distant_intensity=DISTANT_LIGHT_INTENSITY,
                dome_intensity=DOME_LIGHT_INTENSITY,
                angle=DISTANT_LIGHT_ANGLE,
            ),
            simulation_params=SimulationParams(
                print_interval=PRINT_INTERVAL,
                run_interactive=True,
                settle_time=SETTLE_TIME,
                plate_motion_amplitude=PLATE_MOTION_AMPLITUDE,
                plate_motion_frequency=PLATE_MOTION_FREQUENCY,
                plate_motion_axis=PLATE_MOTION_AXIS,
                plate_motion_enabled=PLATE_MOTION_ENABLED,
            ),
        )
        
        # ===== STEP 2: Initialize World =====
        scene.initialize()
        
        # ===== STEP 3: Add Lighting =====
        scene.add_lighting()
        
        # ===== STEP 4: Create the Plate =====
        plate = Plate(
            prim_path="/World/Plate",
            model_file=PLATE_USD_FILE,
            transform_params=TransformParams(
                position=PLATE_POSITION,
                scale=PLATE_SCALE,
                rotateX=PLATE_ROTATION_X,
            ),
            physics_material_params=PhysicsMaterialParams(
                static_friction=PLATE_STATIC_FRICTION,
                dynamic_friction=PLATE_DYNAMIC_FRICTION,
                restitution=PLATE_RESTITUTION,
            ),
            visual_material_params=VisualMaterialParams(
                color=PLATE_COLOR,
                roughness=ROUGHNESS,
                metallic=METALLIC,
            ),
            physics_body_params=PhysicsBodyParams(
                enable_ccd=ENABLE_CCD,
                contact_offset=CONTACT_OFFSET,
                rest_offset=REST_OFFSET,
            ),
            collision_params=CollisionParams(
                approximation=PLATE_COLLISION_TYPE,
                enable_collision=True,
            ),
            mechanics_mode=PLATE_MECHANICS,
        )
        scene.add_object(plate)
        
        # ===== STEP 5: Create the Manipulator =====
        manipulator = Manipulator(
            prim_path="/World/Manipulator",
            model_file=MANIPULATOR_USD_FILE,
            transform_params=TransformParams(
                position=MANIPULATOR_POSITION,
                scale=MANIPULATOR_SCALE,
                rotateX=MANIPULATOR_ROTATION_X,
            ),
            physics_material_params=PhysicsMaterialParams(
                static_friction=MANIPULATOR_STATIC_FRICTION,
                dynamic_friction=MANIPULATOR_DYNAMIC_FRICTION,
                restitution=MANIPULATOR_RESTITUTION,
            ),
            visual_material_params=VisualMaterialParams(
                color=MANIPULATOR_COLOR,
                roughness=ROUGHNESS,
                metallic=METALLIC,
            ),
            physics_body_params=PhysicsBodyParams(
                enable_ccd=ENABLE_CCD,
                contact_offset=CONTACT_OFFSET,
                rest_offset=REST_OFFSET,
            ),
            collision_params=CollisionParams(
                approximation=MANIPULATOR_COLLISION_TYPE,
                enable_collision=True,
            ),
            mechanics_mode=MANIPULATOR_MECHANICS,
        )
        scene.add_object(manipulator)
        
        # ===== STEP 6: Create the Ball =====
        ball = Ball(
            prim_path="/World/Ball",
            radius=BALL_RADIUS,
            transform_params=TransformParams(
                position=BALL_POSITION,
                scale=BALL_SCALE,
                rotateX=None,
            ),
            physics_material_params=PhysicsMaterialParams(
                static_friction=BALL_STATIC_FRICTION,
                dynamic_friction=BALL_DYNAMIC_FRICTION,
                restitution=BALL_RESTITUTION,
            ),
            visual_material_params=VisualMaterialParams(
                color=BALL_COLOR,
                roughness=ROUGHNESS,
                metallic=METALLIC,
            ),
            physics_body_params=PhysicsBodyParams(
                is_dynamic=BALL_IS_DYNAMIC,
                enable_ccd=ENABLE_CCD,
                mass=BALL_MASS,
                contact_offset=CONTACT_OFFSET,
                rest_offset=REST_OFFSET,
            ),
        )
        scene.add_object(ball)
        
        # ===== STEP 7: Run the Simulation =====
        print("\n" + "=" * 70)
        print("Complete scene: Plate, Manipulator, and Ball loaded successfully!")
        print("Close the window when done.")
        print("=" * 70 + "\n")
        
        scene.run_simulation()
    
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nClosing simulation...")
        simulation_app.close()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
