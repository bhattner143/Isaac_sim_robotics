"""
Object-Oriented Implementation: Import plate and ball into Isaac Sim
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
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema, Sdf, UsdShade, UsdLux  # USD scene description
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
# The plate is the flat surface the ball will fall onto
PLATE_POSITION = (0.0, 0.0, 0.0)  # (x, y, z) in meters - plate at origin
PLATE_SCALE = (0.1, 0.1, 0.1)     # Scale multiplier (1.0 = original size)
PLATE_ROTATION_X = None            # Rotation around X-axis in degrees, or None
PLATE_COLOR = (0.0, 1.0, 0.0)      # RGB values from 0.0 to 1.0 (Green)
PLATE_STATIC_FRICTION = 0.6        # Friction when object starts to move (0=slippery, 1=sticky)
PLATE_DYNAMIC_FRICTION = 0.5       # Friction when object is sliding
PLATE_RESTITUTION = 0.1            # Bounciness (0=no bounce, 1=perfectly bouncy)
PLATE_IS_DYNAMIC = True            # True=can move, False=stays fixed in place
PLATE_COLLISION_TYPE = "convexDecomposition"  # How to calculate collisions (accurate but slower)

# --- Ball Configuration ---
# The ball will fall from above and land on the plate
BALL_POSITION = (0.0, 0.0, 1.0)    # Start position (x, y, z) - 1 meter above ground
BALL_SCALE = (0.25, 0.25, 0.25)    # Size multiplier
BALL_RADIUS = 0.5                  # Base radius in meters (before scaling)
BALL_COLOR = (1.0, 0.0, 0.0)       # RGB values (Red)
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
REST_OFFSET = 0.0                  # Minimum separation when at rest

# --- Visual Material Properties ---
# These affect how shiny or rough objects look
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
    Parameters for object positioning, rotation, and scaling in 3D space.
    
    Think of this like: Where is it? How big is it? Is it rotated?
    """
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, z) location in meters
    scale: Tuple[float, float, float] = (10.0, 10.0, 10.0)  # Size multiplier for each axis
    rotateX: Optional[float] = None  # Rotation angle in degrees (or None for no rotation)


@dataclass
class PhysicsMaterialParams:
    """
    Parameters that control how objects interact physically (friction, bounce).
    
    These determine: How slippery? How bouncy?
    """
    static_friction: float = 0.6     # Resistance to start moving (higher = harder to push)
    dynamic_friction: float = 0.5    # Resistance while sliding (higher = slows down faster)
    restitution: float = 0.1         # Bounciness (0=no bounce, 1=super bouncy)


@dataclass
class VisualMaterialParams:
    """
    Parameters for how objects look (color, shininess).
    
    Controls the appearance: What color? Shiny or matte?
    """
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # RGB color (each 0.0 to 1.0)
    roughness: float = 0.4  # Surface roughness (0=mirror smooth, 1=very rough)
    metallic: float = 0.0   # How metallic it looks (0=plastic, 1=metal)


@dataclass
class PhysicsBodyParams:
    """
    Parameters for rigid body dynamics (mass, movement type).
    
    Controls: Can it move? Does gravity affect it? How heavy is it?
    """
    is_dynamic: bool = True          # True=can move and be pushed, False=stays fixed
    enable_ccd: bool = True          # Continuous Collision Detection (prevents tunneling)
    contact_offset: float = 0.02     # Distance before collision is detected (meters)
    rest_offset: float = 0.0         # Minimum gap when objects are resting
    mass: Optional[float] = None     # Mass in kg (None=auto-calculated from size)


@dataclass
class CollisionParams:
    """
    Parameters for collision shape calculation.
    
    Determines: How to detect when objects touch?
    """
    approximation: str = "convexDecomposition"  # Method to calculate collision shape
    enable_collision: bool = True               # Whether this object can collide


@dataclass
class LightingParams:
    """
    Parameters for scene lighting setup.
    
    Controls: How bright? What angle?
    """
    distant_intensity: float = 1000.0  # Brightness of directional (sun-like) light
    dome_intensity: float = 300.0      # Brightness of ambient (sky) light
    distant_angle: float = -45.0       # Angle of directional light in degrees


@dataclass
class SimulationParams:
    """
    Parameters for running the simulation.
    
    Controls: How fast? When to print info?
    """
    print_interval: int = 100       # Print status every N simulation steps
    run_interactive: bool = True    # Whether to run with GUI interaction

@dataclass
class SimulationParams:
    """Parameters for simulation execution"""
    print_interval: int = 100
    run_interactive: bool = True


# ============================================================================
# BASE PHYSICS OBJECT CLASS
# ============================================================================
# This is the "parent" class that both Plate and Ball will inherit from.
# It contains common functionality that all physics objects need.

class PhysicsObject:
    """
    Base class for physics-enabled objects in Isaac Sim.
    
    Think of this as a template that defines what ALL physics objects must have:
    - A position in 3D space (transform)
    - How they look (visual material)
    - How they interact physically (physics material)
    - Physics properties (mass, collision, etc.)
    
    Child classes (Plate, Ball) inherit these abilities and add their own specific features.
    """
    
    def __init__(
        self,
        stage,                          # The USD stage (like a canvas where we place objects)
        prim_path: str,                 # Path/address of this object in the scene (like /World/Ball)
        transform_params: TransformParams,          # Where it is, how big, rotation
        visual_material_params: VisualMaterialParams,  # Color, shininess
        physics_material_params: PhysicsMaterialParams,  # Friction, bounce
        physics_body_params: PhysicsBodyParams      # Mass, dynamics
    ):
        # Store all parameters as instance variables (self.variable_name)
        # so we can use them later in other methods
        self.stage = stage
        self.prim_path = prim_path
        self.transform_params = transform_params
        self.visual_material_params = visual_material_params
        self.physics_material_params = physics_material_params
        self.physics_body_params = physics_body_params
        self.prim = None  # Will store the actual USD prim (object) once created
        
    def get_prim(self):
        """
        Get the USD prim (primitive) for this object.
        
        A "prim" is like the actual object instance in the scene.
        This method retrieves it from the stage using the path.
        It caches the result so we don't look it up repeatedly.
        """
        if self.prim is None:  # First time? Look it up
            self.prim = self.stage.GetPrimAtPath(self.prim_path)
        return self.prim  # Return cached prim
    
    def apply_transform(self):
        """
        Apply transformation (position, rotation, scale) to the object.
        
        This is like placing an object on a table:
        1. WHERE to put it (translate/position)
        2. HOW to turn it (rotate)
        3. HOW BIG to make it (scale)
        """
        prim = self.get_prim()
        if not prim or not prim.IsValid():
            print(f"Warning: Could not access prim at {self.prim_path}")
            return False
        
        # Make this prim "transformable" (can be moved, rotated, scaled)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()  # Clear any existing transforms
        
        # Add transforms in specific order (order matters!)
        # 1. TRANSLATE: Move to position
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*self.transform_params.position))
        
        # 2. ORIENT: Set base orientation (identity = no rotation)
        orient_op = xform.AddOrientOp()
        orient_op.Set(Gf.Quatf(1, 0, 0, 0))  # Quaternion for "no rotation"
        
        # 3. SCALE: Set to 1 initially (will be overridden later)
        scale_op = xform.AddScaleOp()
        scale_op.Set(Gf.Vec3f(1, 1, 1))
        
        # 4. ROTATE X: Optional rotation around X-axis
        if self.transform_params.rotateX is not None:
            rotate_x_op = xform.AddRotateXOp(opSuffix="unitsResolve")
            rotate_x_op.Set(self.transform_params.rotateX)  # Degrees
        
        # 5. FINAL SCALE: Apply actual scale values
        scale_units_op = xform.AddScaleOp(opSuffix="unitsResolve")
        scale_units_op.Set(Gf.Vec3d(*self.transform_params.scale))
        
        print(f"✓ Transform applied to {self.prim_path}")
        return True
    
    def create_visual_material(self):
        """
        Create and apply visual material (color, shininess, etc.).
        
        This is like painting an object:
        - What color is it?
        - Is it shiny or matte?
        - Does it look like metal or plastic?
        
        Uses PBR (Physically Based Rendering) for realistic appearance.
        """
        # Create material container at path like /World/Ball/Material
        material_path = f"{self.prim_path}/Material"
        material = UsdShade.Material.Define(self.stage, material_path)
        
        # Create shader (the "paint" that defines appearance)
        shader_path = f"{material_path}/Shader"
        shader = UsdShade.Shader.Define(self.stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")  # Use standard PBR shader
        
        # Set color (RGB values from 0.0 to 1.0)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            self.visual_material_params.color
        )
        # Set roughness (0=mirror, 1=matte)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            self.visual_material_params.roughness
        )
        # Set metallic (0=plastic/wood, 1=metal)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            self.visual_material_params.metallic
        )
        
        # Connect shader to material output
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        
        # Apply this material to all mesh parts of the object
        self._bind_material_to_meshes(self.get_prim(), material)
        print(f"✓ Visual material applied to {self.prim_path}")
        return material
    
    def _bind_material_to_meshes(self, prim, material):
        """
        Recursively bind material to all mesh parts.
        
        Some objects have multiple mesh parts (like a car has body, wheels, etc.).
        This function walks through all parts and applies the material to each.
        The underscore prefix (_) indicates this is a "private" helper method.
        """
        # If this prim is a mesh, bind the material to it
        if prim.IsA(UsdGeom.Mesh):
            binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
            binding_api.Bind(material)  # "Paint" this mesh
        
        # Check all children (parts) and bind material to them too
        for child in prim.GetChildren():
            self._bind_material_to_meshes(child, material)  # Recursive call
    
    def apply_physics_material(self):
        """
        Apply physics material (friction and bounciness).
        
        This is separate from visual material - it controls how objects
        interact when they touch:
        - How slippery is the surface?
        - How much does it bounce?
        """
        # Create physics material container
        material_path = f"{self.prim_path}/PhysicsMaterial"
        physics_material = UsdPhysics.MaterialAPI.Apply(
            self.stage.DefinePrim(material_path, "Material")
        )
        
        # Set static friction (resistance to start sliding)
        physics_material.CreateStaticFrictionAttr(
            self.physics_material_params.static_friction
        )
        # Set dynamic friction (resistance while sliding)
        physics_material.CreateDynamicFrictionAttr(
            self.physics_material_params.dynamic_friction
        )
        # Set restitution (bounciness: 0=no bounce, 1=perfect bounce)
        physics_material.CreateRestitutionAttr(
            self.physics_material_params.restitution
        )
        
        # Link this physics material to the collision system
        prim = self.get_prim()
        collision_api = UsdPhysics.CollisionAPI.Get(self.stage, prim.GetPath())
        if collision_api:
            # Create relationship between collision and physics material
            collision_api.GetPrim().CreateRelationship("material:binding:physics").AddTarget(material_path)
        
        print(f"✓ Physics material applied to {self.prim_path}")
    
    def setup(self):
        """
        Complete setup - calls all necessary methods in order.
        
        This is like a checklist:
        1. Position the object (transform)
        2. Make it look nice (visual material)
        3. Make it obey physics (rigid body, collision)
        4. Set interaction properties (friction, bounce)
        """
        self.apply_transform()
        self.create_visual_material()
        self.apply_physics()  # Defined by child classes (Plate/Ball)
        self.apply_physics_material()
    
    def apply_physics(self):
        """
        Apply physics - MUST be implemented by child classes.
        
        This is an "abstract method" - it's like a promise that child classes
        (Plate, Ball) will provide their own version of this method.
        Each object type applies physics differently.
        """
        raise NotImplementedError("Subclasses must implement apply_physics()")


# ============================================================================
# PLATE CLASS (Inherits from PhysicsObject)
# ============================================================================
# The Plate class is a specialized type of PhysicsObject.
# It knows how to:
# 1. Load a 3D model from a file
# 2. Apply complex collision shapes (for accurate physics with non-simple shapes)

class Plate(PhysicsObject):
    """
    Plate object with convex decomposition collision.
    
    INHERITANCE: This class "inherits" from PhysicsObject, meaning it automatically
    gets all the methods from the parent class (like create_visual_material, etc.).
    We only need to add/override methods specific to plates.
    
    CONVEX DECOMPOSITION: The plate has a complex shape, so we break it into
    simpler convex pieces for accurate collision detection.
    """
    
    def __init__(
        self,
        stage,
        prim_path: str,
        model_file: Path,                       # NEW: Path to the 3D model file
        transform_params: TransformParams,
        visual_material_params: VisualMaterialParams,
        physics_material_params: PhysicsMaterialParams,
        physics_body_params: PhysicsBodyParams,
        collision_params: CollisionParams       # NEW: How to handle collisions
    ):
        # Call parent class constructor using super()
        # This sets up all the common physics object properties
        super().__init__(
            stage, prim_path, transform_params, visual_material_params,
            physics_material_params, physics_body_params
        )
        # Store plate-specific parameters
        self.model_file = model_file
        self.collision_params = collision_params
        self.transform_params = transform_params
        self.visual_material_params = visual_material_params
        self.physics_material_params = physics_material_params
        self.collision_params = collision_params
    
    def import_model(self):
        """
        Import the 3D model file into the scene.
        
        Think of this like placing a pre-made model (from a file) into your scene.
        The model was created in a 3D modeling program and saved as a USD file.
        
        We use a "reference" (not a copy) so if the original file changes,
        the scene updates too.
        """
        usd_context = omni.usd.get_context()  # Get the USD system
        omni.kit.commands.execute(
            'CreateReferenceCommand',  # Command to link a file
            usd_context=usd_context,
            path_to=self.prim_path,    # Where to place it in our scene
            asset_path=str(self.model_file.absolute()),  # File to load
            instanceable=False         # Each instance is independent
        )
        print(f"✓ Plate model imported at {self.prim_path}")
    
    def apply_physics(self):
        """
        Apply physics properties to the plate.
        
        This is the plate's version of the apply_physics method.
        It overrides (replaces) the parent class's version.
        
        For complex meshes like plates, we need to apply physics to
        each mesh part separately (recursive approach).
        """
        prim = self.get_prim()
        if not prim or not prim.IsValid():
            print(f"Warning: Could not access prim at {self.prim_path}")
            return False
        
        # Apply physics recursively to all mesh children
        self._apply_physics_recursive(prim)
        print(f"✓ Physics applied to plate with {self.collision_params.approximation}")
        return True
    
    def _apply_physics_recursive(self, p):
        """
        Recursively apply physics to all mesh parts of the plate.
        
        RECURSIVE: This function calls itself to handle nested structures.
        Think of it like exploring a folder tree - check this folder, then
        check all subfolders by calling yourself again.
        
        WHY: The plate model might have multiple mesh parts, and we need
        to set up physics for each one.
        """
        # If this prim is a mesh (actual 3D geometry), set up physics
        if p.IsA(UsdGeom.Mesh):
            # --- STEP 1: Make it a rigid body (can be pushed, affected by gravity) ---
            UsdPhysics.RigidBodyAPI.Apply(p)
            # Enable rigid body simulation
            p.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(True)
            # Set whether it's kinematic (position-controlled) or dynamic (force-controlled)
            p.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(
                not self.physics_body_params.is_dynamic  # is_dynamic=True means kinematic=False
            )
            
            # --- STEP 2: Enable collision detection ---
            UsdPhysics.CollisionAPI.Apply(p)
            p.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(
                self.collision_params.enable_collision
            )
            
            # --- STEP 3: Set collision shape approximation ---
            # For complex shapes, we use convexDecomposition (breaks into simple convex pieces)
            UsdPhysics.MeshCollisionAPI.Apply(p)
            mesh_collision = UsdPhysics.MeshCollisionAPI(p)
            mesh_collision.CreateApproximationAttr(self.collision_params.approximation)
            
            # --- STEP 4: PhysX-specific settings ---
            PhysxSchema.PhysxRigidBodyAPI.Apply(p)
            # Enable CCD (Continuous Collision Detection) to prevent fast objects from tunneling
            if self.physics_body_params.enable_ccd:
                p.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
            
            PhysxSchema.PhysxCollisionAPI.Apply(p)
            # Contact offset: objects "touch" when this close
            p.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(
                self.physics_body_params.contact_offset
            )
            # Rest offset: minimum gap when resting
            p.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(
                self.physics_body_params.rest_offset
            )
            
            print(f"  ✓ Physics applied to mesh: {p.GetPath()}")
        
        # Check all children (parts) and apply physics to them too (RECURSION)
        for child in p.GetChildren():
            self._apply_physics_recursive(child)  # Call itself on each child
    
    def setup(self):
        """
        Complete setup for the plate - including model import.
        
        This OVERRIDES the parent class's setup() method to add an extra step:
        1. Import the 3D model file (plate-specific)
        2. Do all the normal setup (call parent's setup using super())
        """
        self.import_model()  # Load the 3D model first
        super().setup()      # Then do standard setup (transform, materials, physics)


# ============================================================================
# BALL CLASS (Inherits from PhysicsObject)
# ============================================================================
# The Ball class is another specialized type of PhysicsObject.
# Unlike Plate (which loads a file), Ball creates its own simple sphere geometry.

class Ball(PhysicsObject):
    """
    Spherical ball object with analytic collision.
    
    ANALYTIC COLLISION: Spheres are mathematically simple, so we can use
    exact formulas for collision (faster and more accurate than mesh collision).
    
    This class creates the geometry from scratch rather than loading a file.
    """
    
    def __init__(
        self,
        stage,
        prim_path: str,
        radius: float,  # NEW: Size of the sphere
        transform_params: TransformParams,
        visual_material_params: VisualMaterialParams,
        physics_material_params: PhysicsMaterialParams,
        physics_body_params: PhysicsBodyParams
    ):
        # Call parent constructor to set up common properties
        super().__init__(
            stage, prim_path, transform_params, visual_material_params,
            physics_material_params, physics_body_params
        )
        # Store ball-specific parameter
        self.radius = radius
        self.transform_params = transform_params
        self.visual_material_params = visual_material_params
        self.physics_material_params = physics_material_params
        self.physics_body_params = physics_body_params
    
    def create_geometry(self):
        """
        Create a sphere geometry programmatically.
        
        Instead of loading a file, we tell USD to create a perfect sphere.
        This is simpler and more efficient for basic shapes like spheres,
        cubes, cylinders, etc.
        """
        # Define a sphere at our prim path
        sphere = UsdGeom.Sphere.Define(self.stage, self.prim_path)
        sphere.CreateRadiusAttr(self.radius)  # Set size
        
        # Apply transforms directly to the sphere
        sphere.AddTranslateOp().Set(Gf.Vec3d(*self.transform_params.position))
        sphere.AddScaleOp().Set(Gf.Vec3f(*self.transform_params.scale))
        
        # Set extent (bounding box) for rendering optimization
        # This tells the renderer "the sphere fits in this box"
        extent = [(-self.radius, -self.radius, -self.radius),
                  (self.radius, self.radius, self.radius)]
        sphere.CreateExtentAttr(extent)
        
        print(f"✓ Ball geometry created at {self.prim_path}")
        return sphere
    
    def apply_physics(self):
        """
        Apply physics to the ball using analytic sphere collision.
        
        ANALYTIC: For spheres, we can use exact mathematical formulas
        instead of approximating the shape. This is faster and more accurate.
        
        This is the ball's version of apply_physics (overrides parent's abstract method).
        """
        prim = self.get_prim()
        
        # --- STEP 1: Make it a rigid body ---
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        
        # --- STEP 2: Set physics attributes ---
        # Enable rigid body simulation (can move and be pushed)
        prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        # Not kinematic (force-controlled, not position-controlled)
        prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        # Enable collision detection
        prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        
        # --- STEP 3: PhysX settings ---
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
        
        # Enable CCD (prevents ball from passing through objects when moving fast)
        if self.physics_body_params.enable_ccd:
            prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
            prim.CreateAttribute("physxRigidBody:contactOffset", Sdf.ValueTypeNames.Float).Set(0.01)
            prim.CreateAttribute("physxRigidBody:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
        
        # --- STEP 4: Set mass (weight) ---
        if self.physics_body_params.mass is not None:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(self.physics_body_params.mass)  # Set mass in kg
        
        print(f"✓ Physics applied to ball (analytic sphere)")
        return True
    
    def apply_transform(self):
        """
        Override apply_transform - ball handles transforms differently.
        
        The ball's transforms are applied directly when creating geometry
        (in create_geometry method), so we don't need the complex transform
        setup from the parent class. This method intentionally does nothing.
        """
        # Transform already applied in create_geometry
        pass  # "pass" means "do nothing" in Python
    
    def setup(self):
        """
        Complete setup for the ball.
        
        This OVERRIDES the parent's setup() to follow a different order:
        1. Create geometry (ball-specific)
        2. Skip apply_transform (already done in step 1)
        3. Apply visual and physics materials
        """
        self.create_geometry()       # Create the sphere
        # Skip apply_transform (it's handled in create_geometry)
        self.create_visual_material()  # Make it look nice
        self.apply_physics()          # Enable physics
        self.apply_physics_material()  # Set friction/bounce


# ============================================================================
# SCENE MANAGER CLASS
# ============================================================================
# SceneManager is like the "director" that coordinates everything.
# It sets up the world, adds lighting, manages objects, and runs the simulation.

class SceneManager:
    """
    Manages the complete Isaac Sim scene setup and simulation.
    
    This class is responsible for:
    - Creating the physics world
    - Adding lights so we can see things
    - Managing all physics objects (plate, ball, etc.)
    - Running the simulation loop (step-by-step physics updates)
    """
    
    def __init__(
        self,
        lighting_params: LightingParams,      # How bright are the lights?
        simulation_params: SimulationParams   # How to run the simulation?
    ):
        # Store parameters
        self.lighting_params = lighting_params
        self.simulation_params = simulation_params
        # These will be initialized later
        self.world = None   # The physics world (gravity, time, etc.)
        self.stage = None   # The USD stage (scene container)
        self.objects = {}   # Dictionary to store all objects by name
        
    def initialize(self):
        """
        Initialize the physics world and get the USD stage.
        
        WORLD: Contains physics settings (gravity, timestep, etc.)
        STAGE: The container where all 3D objects live
        GROUND PLANE: A flat surface at z=0 for objects to rest on
        """
        # Create physics world with 1 meter = 1 unit (standard scale)
        self.world = World(stage_units_in_meters=1.0)
        # Add a default ground plane (infinite flat surface at bottom)
        self.world.scene.add_default_ground_plane()
        # Get the USD stage to add objects to
        self.stage = omni.usd.get_context().get_stage()
        print("✓ Scene initialized with physics world and ground plane")
    
    def add_lighting(self):
        """
        Add lights to the scene so we can see objects.
        
        TWO TYPES OF LIGHTS:
        1. DISTANT LIGHT: Like the sun - directional light from far away
        2. DOME LIGHT: Like the sky - soft ambient light from all directions
        """
        # --- DISTANT LIGHT (Directional, like the sun) ---
        light_path = "/World/DistantLight"
        distant_light = UsdLux.DistantLight.Define(self.stage, light_path)
        # Set brightness
        distant_light.CreateIntensityAttr(self.lighting_params.distant_intensity)
        # Set angle (how wide the light beam is)
        distant_light.CreateAngleAttr(0.5)
        # Rotate the light to shine from above at an angle
        distant_light.AddRotateXYZOp().Set(
            Gf.Vec3f(self.lighting_params.distant_angle, 0, 0)
        )
        
        # --- DOME LIGHT (Ambient, like the sky) ---
        dome_light_path = "/World/DomeLight"
        dome_light = UsdLux.DomeLight.Define(self.stage, dome_light_path)
        # Set brightness of ambient light
        dome_light.CreateIntensityAttr(self.lighting_params.dome_intensity)
        
        print(f"✓ Lighting added: Distant({self.lighting_params.distant_intensity}) + "
              f"Dome({self.lighting_params.dome_intensity})")
    
    def add_object(self, name: str, obj: PhysicsObject):
        """
        Add a physics object to the scene and set it up.
        
        NAME: A label to identify this object (like 'ball' or 'plate')
        OBJ: The PhysicsObject instance (Plate or Ball)
        
        This stores the object in our dictionary and calls its setup() method.
        """
        self.objects[name] = obj  # Store in dictionary for later access
        obj.setup()  # Complete object setup (geometry, materials, physics)
    
    def enable_collision_visualization(self):
        """
        Turn on visual indicators for collisions.
        
        This shows:
        - Collision meshes (the invisible shapes used for physics)
        - Contact points (where objects touch)
        
        Helpful for debugging - you can see exactly what the physics engine sees.
        """
        physics_settings = carb.settings.get_settings()
        # Show collision mesh outlines
        physics_settings.set("/physics/visualizationDisplayCollisionMeshes", True)
        # Show contact points where objects touch
        physics_settings.set("/physics/visualizationDisplayContacts", True)
        # Simplify collision mesh visualization
        physics_settings.set("/physics/visualizationSimplificationDisplayCollisionMeshes", True)
        print("✓ Collision visualization enabled")
    
    def run_simulation(self):
        """
        Run the physics simulation loop.
        
        This is the main loop that:
        1. Steps the physics forward in time
        2. Renders the scene
        3. Prints positions periodically
        4. Repeats until user closes the window
        
        SIMULATION LOOP: Like animation frames - each step calculates
        new positions/velocities based on forces and collisions.
        """
        print("\n" + "=" * 70)
        print("Simulation Running - Ball Falling onto Plate")
        print("=" * 70)
        print("Controls:")
        print("  - Mouse drag: Rotate view")
        print("  - Middle mouse: Pan")
        print("  - Scroll: Zoom")
        print("  - Ctrl+C to exit")
        print("=" * 70)
        
        # Reset physics to initial state
        self.world.reset()
        
        # Get our ball and plate objects from the dictionary
        self.ball = self.objects.get('ball')
        self.plate = self.objects.get('plate')
        
        if not self.ball or not self.plate:
            print("Warning: Ball or plate not found in scene")
            return
        
        # Get the USD prims (actual objects in scene)
        self.ball_prim = self.ball.get_prim()
        self.plate_prim = self.plate.get_prim()
        
        # Calculate actual ball radius (base radius × scale)
        ball_geom = UsdGeom.Sphere(self.ball_prim)
        ball_radius = self.ball.radius * self.ball.transform_params.scale[0]
        
        # Initialize counters
        step_count = 0  # How many simulation steps
        dt = self.world.get_physics_dt()  # Time per step (delta time)
        
        try:
            # MAIN LOOP: Keep running while Isaac Sim window is open
            while simulation_app.is_running():
                # STEP 1: Advance physics by one timestep
                self.world.step(render=True)  # render=True shows the updated scene
                step_count += 1
                
                # STEP 2: Print positions at regular intervals (not every frame - too much!)
                if step_count % self.simulation_params.print_interval == 0:
                    t = step_count * dt  # Calculate current time
                    
                    # Get ball position (center of sphere)
                    ball_xform = UsdGeom.Xformable(self.ball_prim)
                    ball_matrix = ball_xform.ComputeLocalToWorldTransform(0)  # Get transform matrix
                    ball_center = ball_matrix.ExtractTranslation()  # Extract position from matrix
                    ball_bottom = ball_center[2] - ball_radius  # Bottom of ball (center minus radius)
                    
                    # Get plate position
                    plate_xform = UsdGeom.Xformable(self.plate_prim)
                    plate_matrix = plate_xform.ComputeLocalToWorldTransform(0)
                    plate_pos = plate_matrix.ExtractTranslation()
                    
                    # Calculate gap between ball and plate
                    separation = ball_bottom - plate_pos[2]
                    
                    # Print formatted status
                    print(f"Step {step_count:5d} | Time {t:6.2f}s | "
                          f"Plate: ({plate_pos[0]:6.3f}, {plate_pos[1]:6.3f}, {plate_pos[2]:6.3f}) | "
                          f"Ball: ({ball_center[0]:6.3f}, {ball_center[1]:6.3f}, {ball_center[2]:6.3f}) | "
                          f"Gap: {separation:6.3f}")
        
        except KeyboardInterrupt:
            # User pressed Ctrl+C - gracefully stop
            print("\nStopping simulation...")


# ============================================================================
# MAIN FUNCTION - This is where everything starts!
# ============================================================================

def main():
    """
    Main execution flow using Object-Oriented Programming architecture.
    
    This function orchestrates the entire simulation:
    1. Create a scene manager
    2. Set up the world and lighting
    3. Create the plate object
    4. Create the ball object
    5. Run the simulation
    
    All configuration comes from the constants at the top of the file.
    """
    print("=" * 70)
    print("Object-Oriented Ball and Plate Simulation")
    print("=" * 70)
    
    # Print plate size information
    from pxr import Usd, UsdGeom
    try:
        stage = Usd.Stage.Open('plate/plate.usdc')
        bbox_cache = UsdGeom.BboxCache(Usd.TimeCode.Default(), ['default'])
        bbox = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot())
        bbox_range = bbox.ComputeAlignedRange()
        dimensions = bbox_range.GetMax() - bbox_range.GetMin()
        
        print(f"\nPlate Model Information:")
        print(f"  Original dimensions: {dimensions[0]:.3f} × {dimensions[1]:.3f} × {dimensions[2]:.3f} meters")
        print(f"  With PLATE_SCALE = {PLATE_SCALE}:")
        print(f"  Actual size: {dimensions[0]*PLATE_SCALE[0]:.3f} × {dimensions[1]*PLATE_SCALE[1]:.3f} × {dimensions[2]*PLATE_SCALE[2]:.3f} meters")
        print(f"  That's {dimensions[0]*PLATE_SCALE[0]*100:.1f} × {dimensions[1]*PLATE_SCALE[1]*100:.1f} × {dimensions[2]*PLATE_SCALE[2]*100:.1f} cm")
    except Exception as e:
        print(f"\nCould not read plate dimensions: {e}")
    print("=" * 70)
    
    try:
        # ===== STEP 1: Initialize Scene Manager =====
        # Create the "director" that will manage everything
        scene = SceneManager(
            lighting_params=LightingParams(
                distant_intensity=DISTANT_LIGHT_INTENSITY,
                dome_intensity=DOME_LIGHT_INTENSITY,
                distant_angle=DISTANT_LIGHT_ANGLE
            ),
            simulation_params=SimulationParams(
                print_interval=PRINT_INTERVAL,
                run_interactive=True
            )
        )
        scene.initialize()  # Create world and ground plane
        scene.add_lighting()  # Add lights
        
        # ===== STEP 2: Load Plate Model =====
        # Find the 3D model file for the plate
        model_file = Path(__file__).parent / "plate" / "plate.usdc"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # ===== STEP 3: Create Plate Object =====
        # Instantiate a Plate with all its parameters
        plate = Plate(
            stage=scene.stage,              # Where to place it
            prim_path="/World/Plate",       # Address in scene
            model_file=model_file,          # 3D model file
            transform_params=TransformParams(
                position=PLATE_POSITION,    # Where to put it
                scale=PLATE_SCALE,          # How big
                rotateX=PLATE_ROTATION_X    # Any rotation
            ),
            visual_material_params=VisualMaterialParams(
                color=PLATE_COLOR,          # What color
                roughness=ROUGHNESS,        # How shiny
                metallic=METALLIC           # Looks like metal?
            ),
            physics_material_params=PhysicsMaterialParams(
                static_friction=PLATE_STATIC_FRICTION,    # Friction when still
                dynamic_friction=PLATE_DYNAMIC_FRICTION,  # Friction when sliding
                restitution=PLATE_RESTITUTION             # Bounciness
            ),
            physics_body_params=PhysicsBodyParams(
                is_dynamic=PLATE_IS_DYNAMIC,      # Can it move?
                enable_ccd=ENABLE_CCD,            # Prevent tunneling
                contact_offset=CONTACT_OFFSET,    # Collision distance
                rest_offset=REST_OFFSET           # Resting gap
            ),
            collision_params=CollisionParams(
                approximation=PLATE_COLLISION_TYPE,  # How to calculate collisions
                enable_collision=True
            )
        )
        scene.add_object('plate', plate)  # Add to scene and set up
        
        # ===== STEP 4: Create Ball Object =====
        # Instantiate a Ball with all its parameters
        ball = Ball(
            stage=scene.stage,
            prim_path="/World/Ball",
            radius=BALL_RADIUS,             # Size of sphere
            transform_params=TransformParams(
                position=BALL_POSITION,     # Start position (above plate)
                scale=BALL_SCALE            # Size multiplier
            ),
            visual_material_params=VisualMaterialParams(
                color=BALL_COLOR,           # Red color
                roughness=ROUGHNESS,
                metallic=METALLIC
            ),
            physics_material_params=PhysicsMaterialParams(
                static_friction=BALL_STATIC_FRICTION,
                dynamic_friction=BALL_DYNAMIC_FRICTION,
                restitution=BALL_RESTITUTION  # Slight bounce
            ),
            physics_body_params=PhysicsBodyParams(
                is_dynamic=BALL_IS_DYNAMIC,  # Affected by gravity
                enable_ccd=ENABLE_CCD,
                mass=BALL_MASS               # Weight in kg
            )
        )
        scene.add_object('ball', ball)  # Add to scene and set up
        
        # ===== STEP 5: Run Simulation =====
        scene.enable_collision_visualization()  # Show collision meshes
        scene.run_simulation()  # Start the physics loop!
        
    except Exception as e:
        # If anything goes wrong, print detailed error info
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()  # Show full error traceback
    
    finally:
        # Always close Isaac Sim cleanly, even if there was an error
        simulation_app.close()
        print("Done!")


# Python convention: only run main() if this file is run directly
# (not if it's imported as a module)
if __name__ == "__main__":
    main()
