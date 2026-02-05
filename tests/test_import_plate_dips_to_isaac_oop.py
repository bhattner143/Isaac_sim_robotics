"""
Object-Oriented Implementation: Import plate_dips and ball into Isaac Sim
Uses class-based architecture with parameter dataclasses and inheritance

This script demonstrates:
1. Parameter management using dataclasses
2. Base class for physics objects with inheritance
3. Specialized Plate and Ball classes
4. Scene manager for orchestration

BEGINNER'S GUIDE:
- All configuration parameters are at the top (see USER CONFIGURATION section)
- Classes organize related code together (like folders for different topics)
- Inheritance means child classes (Plate, Ball) inherit methods from parent (PhysicsObject)
- Dataclasses are simple containers for grouping related parameters
"""

# Import standard Python libraries
from isaacsim import SimulationApp  # Main Isaac Sim application
from dataclasses import dataclass  # For creating simple parameter classes
from typing import Tuple, Optional  # Type hints for better code clarity
from pathlib import Path  # For working with file paths

# STEP 1: Launch Isaac Sim application
# This creates a window where the simulation will run
simulation_app = SimulationApp({
    "headless": False,  # False = show GUI window, True = run without display
    "width": 1280,      # Window width in pixels
    "height": 720,      # Window height in pixels
    "renderer": "RayTracedLighting",  # Use ray tracing for realistic graphics
    "active_gpu": 0,    # Use first GPU (0-indexed)
})

# STEP 2: Import Isaac Sim modules
# These imports must happen AFTER SimulationApp is created
from omni.isaac.core import World  # Manages physics simulation
from pxr import Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema, Sdf, UsdShade, UsdLux  # USD scene description
import omni.usd  # USD context management
import omni.kit.commands  # For executing Isaac Sim commands
import math  # Mathematical functions
import carb  # NVIDIA Carbonite settings framework


# ============================================================================
# USER CONFIGURATION - EDIT THESE PARAMETERS
# ============================================================================
# All the values you can easily change are here at the top!
# No need to dig through code - just modify these constants.

# --- Plate Configuration ---
# The plate_dips is a plate with dips/indentations for the ball to roll into
PLATE_POSITION = (0.0, 0.0, 0.1)  # (x, y, z) in meters - plate at origin
PLATE_SCALE = (10.0, 10.0, 10.0)  # Scale multiplier - 10x larger
PLATE_ROTATION_X = 0           # Rotation around X-axis in degrees to make plate horizontal (-90° counter-rotates the mesh)
PLATE_COLOR = (0.0, 1.0, 0.0)      # RGB values from 0.0 to 1.0 (Green)
PLATE_STATIC_FRICTION = 0.6        # Friction when object starts to move (0=slippery, 1=sticky)
PLATE_DYNAMIC_FRICTION = 0.5       # Friction when object is sliding
PLATE_RESTITUTION = 0.1            # Bounciness (0=no bounce, 1=perfectly bouncy)
PLATE_MECHANICS = "kinematic"      # "static" (fixed), "kinematic" (position control), "dynamic" (velocity control)
PLATE_COLLISION_TYPE = "convexDecomposition"  # How to calculate collisions (accurate but slower)

# --- Manipulator Configuration ---
# The manipulator is an imported USD object (e.g., robot arm, gripper)
MANIPULATOR_NAME = "so101_new_calib"  # Name of the manipulator
MANIPULATOR_USD_FILE = str(Path(f"manipulator_models/{MANIPULATOR_NAME}/{MANIPULATOR_NAME}.usd").absolute())  # USD file to import
MANIPULATOR_POSITION = (1.5, 0.0, 0.1)  # (x, y, z) in meters - positioned next to the plate
MANIPULATOR_SCALE = (1.0, 1.0, 1.0)     # Scale multiplier
MANIPULATOR_ROTATION_X = 0              # Rotation around X-axis in degrees
MANIPULATOR_COLOR = (0.0, 0.0, 1.0)     # RGB values from 0.0 to 1.0 (Blue)
MANIPULATOR_STATIC_FRICTION = 0.6       # Friction when object starts to move
MANIPULATOR_DYNAMIC_FRICTION = 0.5      # Friction when object is sliding
MANIPULATOR_RESTITUTION = 0.1           # Bounciness (0=no bounce, 1=perfectly bouncy)
MANIPULATOR_MECHANICS = "static"        # "static" (fixed), "kinematic" (position control), "dynamic" (velocity control)
MANIPULATOR_COLLISION_TYPE = "convexDecomposition"  # How to calculate collisions

# --- Ball Configuration ---
# The ball will fall from above and land on the plate
BALL_POSITION = (0.0, 0.0, 3)    # Size multiplier - 10x larger
BALL_SCALE = (1.0, 1.0, 1.0)       # Scale multiplier - no scaling
BALL_RADIUS = 0.5                  # Base radius in meters (before scaling)
BALL_COLOR = (2.0, 0.0, 0.0)       # RGB values (R0.15
BALL_STATIC_FRICTION = 0.6         # Starting friction
BALL_DYNAMIC_FRICTION = 0.5        # Sliding friction
BALL_RESTITUTION = 0.3             # How much it bounces (0.3 = slight bounce)
BALL_MASS = 0.1                    # Mass in kilograms
BALL_IS_DYNAMIC = True             # True = affected by gravity and forces

# --- Scene Configuration ---
# Lighting makes objects visible and look realistic
DISTANT_LIGHT_INTENSITY = 1000.0   # Brightness of directional sun-like light
DOME_LIGHT_INTENSITY = 300.0       # Brightness of ambient environmental light
DISTANT_LIGHT_ANGLE = -45.0        # Angle of sun light (negative = from above)

# --- Simulation Configuration ---
PRINT_INTERVAL = 100               # Print object positions every N simulation steps
ENABLE_CCD = True                  # Continuous Collision Detection prevents objects passing through each other
CONTACT_OFFSET = 0.02              # How close objects get before collision (meters)
REST_OFFSET = 0.0                  # Minimum separation when at rest (meters)

# --- Plate Motion Configuration ---
SETTLE_TIME = 3.0                  # Time to wait for ball and plate to settle (seconds)
PLATE_MOTION_AMPLITUDE = 2.0       # Amplitude of sinusoidal motion (meters)
PLATE_MOTION_FREQUENCY = 0.1       # Frequency of sinusoidal motion (Hz) - 0.5 Hz = 2 second period
PLATE_MOTION_AXIS = 0              # Axis for motion: 0=X, 1=Y, 2=Z
PLATE_MOTION_ENABLED = True        # Enable horizontal plate motion after settling

# --- Visual Material Properties ---
# These affect how shiny o0.15ugh objects look
ROUGHNESS = 0.4                    # Surface roughness (0=mirror, 1=rough/matte)
METALLIC = 0.0                     # Metallic look (0=plastic/wood, 1=metal)


# ============================================================================
# PARAMETER CLASSES (Dataclasses)
# ============================================================================
# Dataclasses are like organized containers that group related parameters.
# Think of them as forms with labeled fields - they make code cleaner and easier to understand.

@dataclass
class TransformParams:
    """
    Parameters for positioning, rotating, and scaling objects in 3D space.
    
    Think of this like giving directions: where to place it (position),
    how to rotate it (rotateX), and how big to make it (scale).
    """
    position: Tuple[float, float, float]  # (x, y, z) coordinates in meters
    scale: Tuple[float, float, float]     # (x, y, z) scale multipliers
    rotateX: Optional[float] = None       # Rotation around X-axis in degrees


@dataclass
class PhysicsMaterialParams:
    """
    Parameters controlling how surfaces interact (friction and bounce).
    
    Real-world analogy:
    - Ice has low friction (slippery), rubber has high friction (sticky)
    - Basketball has high restitution (bouncy), clay has low (no bounce)
    """
    static_friction: float   # Resistance when object starts moving (0.0 to 1.0+)
    dynamic_friction: float  # Resistance while object is sliding (0.0 to 1.0+)
    restitution: float       # Bounciness (0.0 = no bounce, 1.0 = perfect bounce)


@dataclass
class VisualMaterialParams:
    """
    Parameters controlling how an object looks (color and surface properties).
    
    These create the visual appearance using Physically Based Rendering (PBR):
    - color: What color the object appears
    - roughness: How rough or smooth the surface looks
    - metallic: How metallic vs plastic/wood the material appears
    """
    color: Tuple[float, float, float]  # RGB color (0.0 to 1.0 for each channel)
    roughness: float = 0.4             # Surface roughness (0=mirror, 1=matte)
    metallic: float = 0.0              # Metalness (0=dielectric, 1=metal)


@dataclass
class PhysicsBodyParams:
    """
    Parameters defining physics body behavior.
    
    - is_dynamic: Can the object move? (True) or is it fixed in place? (False)
    - enable_ccd: Use Continuous Collision Detection to prevent fast objects from passing through
    - mass: How heavy the object is (only for dynamic objects)
    - contact_offset: Distance threshold for collision detection
    - rest_offset: Minimum separation when objects are resting
    """
    is_dynamic: bool = True
    enable_ccd: bool = True
    mass: Optional[float] = None
    contact_offset: float = 0.02
    rest_offset: float = 0.0


@dataclass
class CollisionParams:
    """
    Parameters for collision detection.
    
    - approximation: How to calculate collisions
      * "convexDecomposition": Break complex shape into simpler pieces (accurate but slow)
      * "convexHull": Wrap object in simplest convex shape (fast but less accurate)
      * "none": Analytic shapes like sphere (fastest, for simple geometry)
    - enable_collision: Should this object participate in collisions?
    """
    approximation: str = "none"
    enable_collision: bool = True


@dataclass
class LightingParams:
    """
    Parameters for scene lighting.
    
    - distant_intensity: Brightness of directional light (like the sun)
    - dome_intensity: Brightness of ambient environment light (like sky)
    - angle: Angle of distant light in degrees
    """
    distant_intensity: float = 1000.0
    dome_intensity: float = 300.0
    angle: float = -45.0


@dataclass
class SimulationParams:
    """
    Parameters controlling simulation behavior.
    
    - print_interval: How often to print object positions (every N steps)
    - run_interactive: If True, simulation runs until you close the window
    - settle_time: Time to wait before applying plate motion (seconds)
    - plate_motion_amplitude: Amplitude of sinusoidal motion (meters)
    - plate_motion_frequency: Frequency of sinusoidal motion (Hz)
    - plate_motion_axis: Axis for motion (0=X, 1=Y, 2=Z)
    - plate_motion_enabled: Whether to enable plate motion
    """
    print_interval: int = 100
    run_interactive: bool = True
    settle_time: float = 3.0
    plate_motion_amplitude: float = 5.0
    plate_motion_frequency: float = 0.5
    plate_motion_axis: int = 0
    plate_motion_enabled: bool = True


# ============================================================================
# BASE CLASS: PhysicsObject
# ============================================================================
# This is the parent class that contains common functionality for all physics objects.
# Both Plate and Ball will inherit from this, avoiding code duplication.

class PhysicsObject:
    """
    Base class for all physics objects in the scene.
    
    WHY USE A BASE CLASS?
    Instead of repeating the same code in Plate and Ball classes, we put common
    functionality here. Both child classes inherit these methods automatically.
    
    Common functionality includes:
    - Applying position, rotation, and scale (transforms)
    - Creating visual materials (colors and appearance)
    - Applying physics materials (friction and bounce)
    - Setting up the complete object
    
    Each child class (Plate, Ball) will override the apply_physics() method
    to define their own specific physics behavior.
    """
    
    def __init__(
        self,
        prim_path: str,
        transform_params: TransformParams,
        physics_material_params: PhysicsMaterialParams,
        visual_material_params: VisualMaterialParams,
        physics_body_params: PhysicsBodyParams,
    ):
        """
        Initialize a physics object with all its parameters.
        
        Args:
            prim_path: USD path where object will be created (e.g., "/World/Plate")
            transform_params: Position, scale, rotation settings
            physics_material_params: Friction and restitution settings
            visual_material_params: Color and appearance settings
            physics_body_params: Mass, dynamic/static, collision settings
        """
        self.prim_path = prim_path
        self.transform_params = transform_params
        self.physics_material_params = physics_material_params
        self.visual_material_params = visual_material_params
        self.physics_body_params = physics_body_params
        self.stage = omni.usd.get_context().get_stage()  # Get the USD stage
        self._prim = None  # Cache for the USD prim
    
    def get_prim(self):
        """
        Get the USD prim for this object (with caching for efficiency).
        
        A "prim" is a primitive element in USD - think of it as a node in the scene tree.
        We cache it so we don't have to look it up multiple times.
        """
        if self._prim is None:
            self._prim = self.stage.GetPrimAtPath(self.prim_path)
        return self._prim
    
    def apply_transform(self):
        """
        Apply position, rotation, and scale to the object.
        
        In USD, transforms are applied using "xformOps" (transform operations).
        The order matters: we do translate (position), rotate, then scale.
        """
        prim = self.get_prim()
        xformable = UsdGeom.Xformable(prim)
        
        # Clear any existing transform operations to start fresh
        xformable.ClearXformOpOrder()
        
        # STEP 1: Set position (translate)
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*self.transform_params.position))
        
        # STEP 2: Set rotation around X-axis (if specified)
        if self.transform_params.rotateX is not None:
            rotate_op = xformable.AddRotateXOp()
            rotate_op.Set(self.transform_params.rotateX)
        
        # STEP 3: Set scale
        scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3f(*self.transform_params.scale))
    
    def create_visual_material(self):
        """
        Create and apply a PBR (Physically Based Rendering) material.
        
        PBR materials simulate how light interacts with surfaces in the real world.
        We set:
        - Base color (diffuse color)
        - Roughness (how rough vs smooth the surface is)
        - Metallic (how metallic vs dielectric the material is)
        """
        # Create a material prim in the Looks scope
        material_path = f"{self.prim_path}/Looks/Material"
        material = UsdShade.Material.Define(self.stage, material_path)
        
        # Create a PBR shader
        shader_path = f"{material_path}/Shader"
        shader = UsdShade.Shader.Define(self.stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")  # Use USD's standard PBR shader
        
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
        
        # Connect shader output to material surface
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.GetOutput("surface"))
        
        # Bind material to all mesh geometry in this object
        self._bind_material_to_meshes(self.get_prim(), material)
    
    def _bind_material_to_meshes(self, prim, material):
        """
        Recursively bind material to all mesh children.
        
        Some objects (like imported models) have many mesh pieces.
        We walk through the hierarchy and bind the material to each mesh.
        """
        if prim.IsA(UsdGeom.Mesh):
            # This is a mesh - bind the material
            UsdShade.MaterialBindingAPI(prim).Bind(material)
        
        # Recursively process all children
        for child in prim.GetChildren():
            self._bind_material_to_meshes(child, material)
    
    def apply_physics_material(self):
        """
        Apply physics material (friction and restitution).
        
        This defines how the surface behaves in physics simulations:
        - Static friction: resistance when starting to move
        - Dynamic friction: resistance while sliding
        - Restitution: how much energy is retained in a bounce
        """
        # Create physics material container
        material_path = f"{self.prim_path}/PhysicsMaterial"
        physics_material = UsdPhysics.MaterialAPI.Apply(
            self.stage.DefinePrim(material_path, "Material")
        )
        
        # Set friction coefficients
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
        """
        Apply physics properties - MUST be implemented by child classes.
        
        This is an abstract method. Each child class (Plate, Ball) will implement
        this differently because they have different physics requirements.
        
        For example:
        - Plate: Uses mesh collision with convex decomposition
        - Ball: Uses analytic sphere collision
        """
        raise NotImplementedError("Child classes must implement apply_physics()")
    
    def setup(self):
        """
        Complete setup process for the object.
        
        This is a "template method" that orchestrates all the setup steps:
        1. Apply transforms (position, rotation, scale)
        2. Create visual material (appearance)
        3. Apply physics (collision and dynamics)
        4. Apply physics material (friction and bounce)
        
        Child classes can override this if they need a different setup sequence.
        """
        self.apply_transform()
        self.create_visual_material()
        self.apply_physics()
        self.apply_physics_material()


# ============================================================================
# PLATE CLASS (inherits from PhysicsObject)
# ============================================================================

class Plate(PhysicsObject):
    """
    Represents the plate_dips object imported from USD file.
    
    WHY A SEPARATE CLASS?
    The plate is different from the ball:
    - It's loaded from an external USD file (not created procedurally)
    - It uses mesh collision (the ball uses a sphere)
    - It needs to recursively apply physics to all mesh children
    
    By inheriting from PhysicsObject, we get all the common functionality
    (transforms, materials) for free, and only implement what's unique.
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
        mechanics_mode: str = 'dynamic',
    ):
        """
        Initialize the Plate with additional model file and collision parameters.
        
        Args:
            model_file: Path to the USD model file to import
            collision_params: How to calculate collisions (convex decomposition, etc.)
            mechanics_mode: 'static' (fixed), 'kinematic' (position control), or 'dynamic' (velocity control)
            (other args inherited from parent class)
        """
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
    
    def import_model(self):
        """
        Import the model file into the scene.
        
        The plate_dips folder contains a pre-converted USD file (plate_dips.usd)
        which we'll use as a reference. This is much simpler and more reliable
        than converting OBJ at runtime.
        """
        print(f"Importing model from: {self.model_file}")
        
        # Use the USD file instead of OBJ (it's already in the folder)
        usd_file = self.model_file.replace('.obj', '.usd')
        
        # Import using reference command (same as working script)
        usd_context = omni.usd.get_context()
        omni.kit.commands.execute(
            'CreateReferenceCommand',
            usd_context=usd_context,
            path_to=self.prim_path,
            asset_path=usd_file,
            instanceable=False
        )
        print(f"✓ Model imported at {self.prim_path}")
    
    def apply_physics(self):
        """
        Apply physics to the plate using mesh collision.
        
        The plate is a complex mesh, so we:
        1. Apply rigid body dynamics
        2. Apply collision to all mesh children
        3. Use the specified collision approximation (convex decomposition)
        4. Lock all angular axes to prevent rotation
        """
        prim = self.get_prim()
        
        # Apply rigid body API
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
            
            # Set mechanics mode based on PLATE_MECHANICS setting
            mechanics_mode = getattr(self, 'mechanics_mode', 'dynamic')
            
            if mechanics_mode == 'static':
                # Static: completely fixed in place, no motion
                rigid_body.CreateRigidBodyEnabledAttr().Set(False)
            elif mechanics_mode == 'kinematic':
                # Kinematic: position-controlled, doesn't respond to forces
                rigid_body.CreateKinematicEnabledAttr().Set(True)
                # Apply angular locks to prevent rotation even in kinematic mode
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                prim.CreateAttribute("physxRigidBody:lockFlags", Sdf.ValueTypeNames.Int).Set(56)  # Lock X,Y,Z rotation
            else:  # dynamic
                # Dynamic: velocity-controlled with angular locks to prevent rotation
                prim.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(10000.0)
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                prim.CreateAttribute("physxRigidBody:lockFlags", Sdf.ValueTypeNames.Int).Set(56)  # Lock X,Y,Z rotation
        
        # Recursively apply collision to all meshes
        self._apply_physics_recursive(prim)
    
    def _apply_physics_recursive(self, prim):
        """
        Recursively apply physics collision to all mesh children.
        
        OBJ files often contain multiple mesh objects. We walk through
        the hierarchy and apply collision settings to each mesh we find.
        """
        # If this is a mesh, apply collision
        if prim.IsA(UsdGeom.Mesh):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(prim)
                
                # Apply collision approximation
                if self.collision_params.approximation != "none":
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                    mesh_collision_api.CreateApproximationAttr().Set(
                        self.collision_params.approximation
                    )
            
            # Apply PhysX collision settings
            if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                physx_collision.CreateContactOffsetAttr().Set(
                    self.physics_body_params.contact_offset
                )
                physx_collision.CreateRestOffsetAttr().Set(
                    self.physics_body_params.rest_offset
                )
        
        # Recursively process all children
        for child in prim.GetChildren():
            self._apply_physics_recursive(child)
    
    def setup(self):
        """
        Override setup to import model first, then do standard setup.
        
        The plate needs to be imported before we can apply transforms,
        materials, and physics to it.
        """
        self.import_model()  # Import OBJ first
        super().setup()      # Then do standard setup (transform, materials, physics)


# ============================================================================
# BALL CLASS (inherits from PhysicsObject)
# ============================================================================

class Ball(PhysicsObject):
    """
    Represents a spherical ball created procedurally.
    
    WHY A SEPARATE CLASS?
    The ball is different from the plate:
    - It's created procedurally (not loaded from a file)
    - It uses a simple sphere shape (not a complex mesh)
    - It uses analytic collision (exact sphere math, not mesh approximation)
    - It has a mass (the plate might be static)
    
    By inheriting from PhysicsObject, we reuse common code and only
    implement what makes the ball unique.
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
        """
        Initialize the Ball with a radius parameter.
        
        Args:
            radius: Radius of the sphere in meters (before scaling)
            (other args inherited from parent class)
        """
        super().__init__(
            prim_path,
            transform_params,
            physics_material_params,
            visual_material_params,
            physics_body_params,
        )
        self.radius = radius
    
    def create_geometry(self):
        """
        Create the sphere geometry procedurally.
        
        We use UsdGeom.Sphere to create a perfect mathematical sphere.
        The radius and transform are set directly on the sphere prim.
        """
        # Create sphere prim
        sphere = UsdGeom.Sphere.Define(self.stage, self.prim_path)
        
        # Set radius
        sphere.CreateRadiusAttr(self.radius)
        
        # Apply transforms directly to sphere
        sphere.AddTranslateOp().Set(Gf.Vec3d(*self.transform_params.position))
        sphere.AddScaleOp().Set(Gf.Vec3f(*self.transform_params.scale))
        
        # Set extent (bounding box) for rendering optimization
        extent = [
            (-self.radius, -self.radius, -self.radius),
            (self.radius, self.radius, self.radius)
        ]
        sphere.CreateExtentAttr(extent)
    
    def apply_physics(self):
        """
        Apply physics to the ball using analytic sphere collision.
        
        For a simple sphere, we can use exact mathematical collision detection
        instead of mesh approximation. This is faster and more accurate.
        """
        prim = self.get_prim()
        
        # Apply rigid body and collision APIs
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        
        # Set physics attributes
        prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        
        # Apply PhysX specific settings
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
        
        # Enable Continuous Collision Detection if requested
        if self.physics_body_params.enable_ccd:
            prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
            prim.CreateAttribute("physxRigidBody:contactOffset", Sdf.ValueTypeNames.Float).Set(0.01)
            prim.CreateAttribute("physxRigidBody:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
        
        # Set mass if specified
        if self.physics_body_params.mass is not None:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(self.physics_body_params.mass)
    
    def apply_transform(self):
        """
        Override to skip transform application (already done in create_geometry).
        
        For the ball, we apply transforms when creating the geometry,
        so we don't need to do it again here.
        """
        pass  # Transform already applied in create_geometry
    
    def setup(self):
        """
        Override setup to create geometry first, then do standard setup.
        
        The ball needs to be created before we can apply materials and physics.
        Transform is handled in create_geometry, so we skip apply_transform.
        """
        self.create_geometry()  # Create sphere first
        # Skip apply_transform (already done in create_geometry)
        self.create_visual_material()
        self.apply_physics()
        self.apply_physics_material()


# ============================================================================
# SCENE MANAGER CLASS
# ============================================================================

class SceneManager:
    """
    Manages the overall simulation scene.
    
    WHAT DOES THIS CLASS DO?
    Think of SceneManager as a movie director:
    - It sets up the world (the stage)
    - It adds lighting (so we can see)
    - It creates and manages all objects (actors)
    - It runs the simulation (action!)
    
    WHY SEPARATE FROM OBJECTS?
    We separate scene management from individual objects because:
    - Scene setup (world, lights) is different from object behavior
    - It's easier to understand and modify
    - We can reuse this manager for different simulations
    """
    
    def __init__(
        self,
        lighting_params: LightingParams,
        simulation_params: SimulationParams,
    ):
        """
        Initialize the scene manager.
        
        Args:
            lighting_params: Settings for scene lighting
            simulation_params: Settings for simulation behavior
        """
        self.lighting_params = lighting_params
        self.simulation_params = simulation_params
        self.world = None
        self.stage = None
        self.objects = []  # List to store all objects we create
    
    def initialize(self):
        """
        Initialize the physics world and get the USD stage.
        
        The "World" manages:
        - Physics simulation (gravity, collisions, forces)
        - Simulation time stepping
        - Ground plane (floor)
        """
        print("Initializing world...")
        
        # Create physics world with gravity and ground plane
        # Units are in meters (matching USD model units)
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()  # Add a floor
        
        # Get the USD stage (the scene graph)
        self.stage = omni.usd.get_context().get_stage()
        
        print("World initialized with ground plane")
    
    def add_lighting(self):
        """
        Add lighting to the scene.
        
        We create two types of lights:
        1. Distant Light: Directional light like the sun
        2. Dome Light: Ambient environmental lighting like the sky
        
        Together they create realistic lighting conditions.
        """
        print("Adding lights...")
        
        # Create a distant light (directional, like the sun)
        distant_light = UsdLux.DistantLight.Define(self.stage, "/World/DistantLight")
        distant_light.CreateIntensityAttr().Set(self.lighting_params.distant_intensity)
        distant_light.CreateAngleAttr().Set(0.5)  # Soft shadows
        
        # Rotate light to shine from above at an angle
        xformable = UsdGeom.Xformable(distant_light)
        xformable.ClearXformOpOrder()
        rotate_op = xformable.AddRotateXOp()
        rotate_op.Set(self.lighting_params.angle)
        
        # Create dome light (ambient environment light)
        dome_light = UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr().Set(self.lighting_params.dome_intensity)
        
        print("Lights added")
    
    def add_object(self, obj: PhysicsObject):
        """
        Add a physics object to the scene.
        
        Args:
            obj: A PhysicsObject (Plate or Ball) to add
        """
        print(f"Setting up object at {obj.prim_path}...")
        obj.setup()  # Call the object's setup method
        self.objects.append(obj)  # Add to our list
        print(f"Object {obj.prim_path} setup complete")
    
    def enable_collision_visualization(self):
        """
        Enable visualization of collision meshes.
        
        This helps debug collision problems by showing the actual collision
        shapes (which might be different from the visual mesh).
        """
        settings = carb.settings.get_settings()
        settings.set("/persistent/physics/visualizationDisplayCollisionMeshes", True)
    
    def run_simulation(self):
        """
        Run the simulation loop.
        
        This is the main simulation loop that:
        1. Resets the world to initial state
        2. Steps through physics simulation
        3. Waits for objects to settle
        4. Applies sinusoidal motion to the plate
        5. Prints object positions periodically
        6. Continues until user closes the window (if interactive)
        
        UNDERSTANDING THE SIMULATION LOOP:
        - Each "step" advances simulation by a small time increment (usually 1/60 second)
        - Physics engine calculates forces, collisions, and movements
        - Sinusoidal motion: position = amplitude * sin(2π * frequency * time)
        """
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
        
        # Reset world to initial state
        self.world.reset()
        
        step_count = 0
        motion_started = False
        plate_obj = None
        plate_initial_pos = None
        is_kinematic = None
        
        # Find the plate object
        for obj in self.objects:
            if "Plate" in obj.prim_path:
                plate_obj = obj
                mechanics_mode = getattr(obj, 'mechanics_mode', 'dynamic')
                is_kinematic = (mechanics_mode == 'kinematic')
                # Store initial position for kinematic motion
                if is_kinematic:
                    plate_prim = obj.get_prim()
                    xformable = UsdGeom.Xformable(plate_prim)
                    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    plate_initial_pos = list(world_transform.ExtractTranslation())
                break
        
        try:
            # Main simulation loop
            while simulation_app.is_running():
                # Step the physics simulation forward
                self.world.step(render=True)
                
                step_count += 1
                
                # Get simulation time
                t = step_count * self.world.get_physics_dt()
                
                # Apply sinusoidal motion to plate after settling
                if self.simulation_params.plate_motion_enabled and plate_obj is not None:
                    #Motion is delayed until after a settling period (settle_time), so the system stabilizes before moving the plate.
                    if t >= self.simulation_params.settle_time:
                        if not motion_started:
                            print(f"\n{'='*70}")
                            mode = "KINEMATIC (position control)" if is_kinematic else "DYNAMIC (velocity control)"
                            print(f"STARTING SINUSOIDAL PLATE MOTION at t={t:.2f}s - {mode}")
                            print(f"{'='*70}\n")
                            motion_started = True
                        
                        # Calculate sinusoidal offset for x-axis motion 
                        motion_time = t - self.simulation_params.settle_time
                        amplitude = self.simulation_params.plate_motion_amplitude
                        frequency = self.simulation_params.plate_motion_frequency
                        axis = self.simulation_params.plate_motion_axis
                        angular_freq = 2 * math.pi * frequency
                        # Get plate prim
                        plate_prim = plate_obj.get_prim()
                        
                        if is_kinematic:
                            # KINEMATIC MODE: Direct position control
                            # position = initial + A * sin(2π * f * t)
                            offset = amplitude * math.sin(angular_freq * motion_time)
                            
                            new_pos = plate_initial_pos.copy()
                            new_pos[axis] = plate_initial_pos[axis] + offset
                            
                            # Apply position directly via transform
                            xformable = UsdGeom.Xformable(plate_prim)
                            xformable.ClearXformOpOrder()
                            
                            # STEP 1: Apply translation
                            translate_op = xformable.AddTranslateOp()
                            translate_op.Set(Gf.Vec3d(*new_pos))

                            # STEP 2: Apply fixed 90 degree rotation around X-axis
                            rotate_op = xformable.AddRotateXOp()
                            rotate_op.Set(90.0)
                            
                            # Alternative: Apply rotation using quaternion (commented out)
                            # orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
                            # half_angle = math.pi / 4  # 90°/2 in radians
                            # quat = Gf.Quatf(
                            #     math.cos(half_angle),  # w
                            #     math.sin(half_angle),  # x
                            #     0.0,                   # y
                            #     0.0                    # z
                            # )
                            # orient_op.Set(quat)
                            
                            # STEP 3: Reapply scale
                            scale_vec = (
                                tuple(plate_obj.transform_params.scale)
                                if hasattr(plate_obj, "transform_params") and plate_obj.transform_params.scale
                                else (1.0, 1.0, 1.0)
                            )
                            scale_op = xformable.AddScaleOp()
                            scale_op.Set(Gf.Vec3f(*scale_vec))

                            # Print plate physics parameters (KINEMATIC MODE)
                            # Compute rotation from simulation data (read from xformOps applied)
                            xformable = UsdGeom.Xformable(plate_prim)
                            xform_ops = xformable.GetOrderedXformOps()
                            
                            # Extract rotation from xformOps
                            rotation_x_deg = 0.0
                            rotation_y_deg = 0.0
                            rotation_z_deg = 0.0
                            for op in xform_ops:
                                op_type = op.GetOpType()
                                if op_type == UsdGeom.XformOp.TypeRotateX:
                                    rotation_x_deg = op.Get()
                                elif op_type == UsdGeom.XformOp.TypeRotateY:
                                    rotation_y_deg = op.Get()
                                elif op_type == UsdGeom.XformOp.TypeRotateZ:
                                    rotation_z_deg = op.Get()
                            
                            # Compute velocity from position derivative: v = A * ω * cos(ωt)
                            computed_velocity = [0.0, 0.0, 0.0]
                            computed_velocity[axis] = amplitude * angular_freq * math.cos(angular_freq * motion_time)
                            
                            # Get physics properties
                            rigid_body = UsdPhysics.RigidBodyAPI(plate_prim)
                            try:
                                is_kinematic = rigid_body.GetKinematicEnabledAttr().Get() if rigid_body.GetKinematicEnabledAttr() else False
                            except:
                                is_kinematic = False
                            
                            # Get collision properties
                            try:
                                physx_collision = PhysxSchema.PhysxCollisionAPI(plate_prim)
                                contact_offset = physx_collision.GetContactOffsetAttr().Get() if physx_collision.GetContactOffsetAttr() else "N/A"
                                rest_offset = physx_collision.GetRestOffsetAttr().Get() if physx_collision.GetRestOffsetAttr() else "N/A"
                            except:
                                contact_offset = "N/A"
                                rest_offset = "N/A"
                            
                            print(f"[Step {step_count}] PLATE KINEMATIC PARAMS:")
                            print(f"  Translation: ({new_pos[0]:.4f}, {new_pos[1]:.4f}, {new_pos[2]:.4f}) m")
                            print(f"  Rotation (XYZ): ({rotation_x_deg:.2f}°, {rotation_y_deg:.2f}°, {rotation_z_deg:.2f}°)")
                            print(f"  Computed Velocity: ({computed_velocity[0]:.4f}, {computed_velocity[1]:.4f}, {computed_velocity[2]:.4f}) m/s")
                            print(f"  Kinematic: {is_kinematic}, Contact Offset: {contact_offset}, Rest Offset: {rest_offset}")
                            
                            mesh_path = "/World/Plate/part_dips_coarse_rot/mesh32"
                            mesh_prim = self.stage.GetPrimAtPath(mesh_path)

                            if mesh_prim and mesh_prim.IsValid():
                                xformable = UsdGeom.Xformable(mesh_prim)
                                time_code = Usd.TimeCode.Default()  # or Usd.TimeCode(t)

                                world_xform = xformable.ComputeLocalToWorldTransform(time_code)
                                world_pos = world_xform.ExtractTranslation()

                                world_rot_quat = world_xform.ExtractRotationQuat()
                                world_rot = Gf.Rotation(world_rot_quat)
                                world_rot_euler = world_rot.Decompose(
                                    Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis()
                                )

                                world_scale = Gf.Vec3d(
                                    world_xform.GetRow3(0).GetLength(),
                                    world_xform.GetRow3(1).GetLength(),
                                    world_xform.GetRow3(2).GetLength()
                                )

                                print(f"[{mesh_path}] WORLD:")
                                print(f"  Pos: ({world_pos[0]:.4f}, {world_pos[1]:.4f}, {world_pos[2]:.4f})")
                                print(f"  Rot XYZ: ({world_rot_euler[0]:.2f}°, {world_rot_euler[1]:.2f}°, {world_rot_euler[2]:.2f}°)")
                                print(f"  Quat (wxyz): ({world_rot_quat.GetReal():.4f}, "
                                    f"{world_rot_quat.GetImaginary()[0]:.4f}, "
                                    f"{world_rot_quat.GetImaginary()[1]:.4f}, "
                                    f"{world_rot_quat.GetImaginary()[2]:.4f})")
                                print(f"  Scale: ({world_scale[0]:.4f}, {world_scale[1]:.4f}, {world_scale[2]:.4f})")
                            
                        else:
                            # DYNAMIC MODE: Velocity control
                            # velocity = A * 2π * f * cos(2π * f * t)
                            velocity_magnitude = amplitude * angular_freq * math.cos(angular_freq * motion_time)
                            
                            # Create velocity vector (only on specified axis)
                            velocity = [0.0, 0.0, 0.0]
                            velocity[axis] = velocity_magnitude
                            
                            # Apply velocity to plate rigid body
                            rigid_body = UsdPhysics.RigidBodyAPI(plate_prim)
                            rigid_body.GetVelocityAttr().Set(Gf.Vec3f(*velocity))
                            
                            # Keep angular velocity at zero to prevent rotation
                            rigid_body.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                
                # Print positions periodically
                if step_count % self.simulation_params.print_interval == 0:
                    # Print status for each object
                    for obj in self.objects:
                        prim = obj.get_prim()
                        xformable = UsdGeom.Xformable(prim)
                        
                        # Get current world transform
                        world_transform = xformable.ComputeLocalToWorldTransform(
                            Usd.TimeCode.Default()
                        )
                        position = world_transform.ExtractTranslation()
                        
                        print(f"Step {step_count:5d} | Time {t:6.2f}s | "
                              f"{obj.prim_path}: ({position[0]:6.3f}, "
                              f"{position[1]:6.3f}, {position[2]:6.3f})")
        
        except KeyboardInterrupt:
            # User pressed Ctrl+C - gracefully stop
            print("\nStopping simulation...")


# ============================================================================
# MAIN FUNCTION - This is where everything starts!
# ============================================================================

def main():
    """
    Main execution flow - Import and visualize plate_dips with a ball on top.
    
    This version:
    1. Creates a scene manager
    2. Sets up the world and lighting
    3. Imports the part_dips_coarse model
    4. Adds a ball positioned just above the plate
    5. Lets you explore the 3D view
    
    All configuration comes from the constants at the top of the file.
    """
    print("=" * 70)
    print("Plate and Ball Visualization (No Dropping)")
    print("=" * 70)
    
    try:
        # ===== STEP 1: Initialize Scene Manager =====
        # Create the "director" that will manage everything
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
        # Set up the physics world and ground plane
        scene.initialize()
        
        # ===== STEP 3: Add Lighting =====
        # Make the scene visible with proper lighting
        scene.add_lighting()
        
        # ===== STEP 4: Create the Plate =====
        # Import the part_dips_coarse_rot model (pre-rotated version)
        plate_usd_path = str(Path("plate_dips/part_dips_coarse_rot.usd").absolute())
        
        plate = Plate(
            prim_path="/World/Plate",
            model_file=plate_usd_path,
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
        # Import the manipulator USD model (robot arm, gripper, etc.)
        manipulator_usd_path = MANIPULATOR_USD_FILE
        
        manipulator = Plate(  # Reusing Plate class for USD import
            prim_path="/World/Manipulator",
            model_file=manipulator_usd_path,
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
        # Create a small ball positioned just above the plate center
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
        
        # ===== STEP 7: Enable Collision Visualization (Optional) =====
        # Uncomment to see collision meshes (helpful for debugging)
        # scene.enable_collision_visualization()
        
        # ===== STEP 8: Run the Simulation =====
        # Start the physics simulation loop
        print("\n" + "=" * 70)
        print("Plate and Ball loaded successfully! Explore the 3D view.")
        print("Close the window when done.")
        print("=" * 70 + "\n")

        # Run the simulation (interactive mode)
        scene.run_simulation()
    
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up and close
        print("\nClosing simulation...")
        simulation_app.close()


# ============================================================================
# ENTRY POINT
# ============================================================================
# This is what Python runs when you execute this script

if __name__ == "__main__":
    main()
