"""
Import cube and ball into Isaac Sim with physics
Creates a kinematic cube that moves horizontally with a ball on top

This script demonstrates:
1. Launching Isaac Sim programmatically
2. Creating a physics world with ground plane
3. Creating a cube with kinematic motion
4. Creating a ball with dynamic physics
5. Running the interactive simulation with horizontal motion
"""

from isaacsim import SimulationApp

# STEP 1: Launch Isaac Sim application
simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
    "renderer": "RayTracedLighting",
    "active_gpu": 0,
})

# STEP 2: Import Isaac Sim modules
from omni.isaac.core import World
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema, Sdf, UsdShade, UsdLux
import omni.usd
import omni.kit.commands
from pathlib import Path
import math
import carb


def get_usd_context():
    """Get the USD context for the current Isaac Sim session"""
    return omni.usd.get_context()


def create_world_with_ground():
    """Create the physics world with ground plane"""
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    print("✓ Physics world created with ground plane")
    return world


def add_scene_lighting(distant_intensity=1000.0, dome_intensity=300.0, distant_angle=-45.0):
    """Add lighting to the scene"""
    stage = get_usd_context().get_stage()
    
    # Add distant light (directional, like sun)
    light_path = "/World/DistantLight"
    distant_light = UsdLux.DistantLight.Define(stage, light_path)
    distant_light.CreateIntensityAttr(distant_intensity)
    distant_light.CreateAngleAttr(0.5)
    distant_light.AddRotateXYZOp().Set(Gf.Vec3f(distant_angle, 0, 0))
    
    # Add dome light for ambient lighting
    dome_light_path = "/World/DomeLight"
    dome_light = UsdLux.DomeLight.Define(stage, dome_light_path)
    dome_light.CreateIntensityAttr(dome_intensity)
    
    print(f"✓ Lights added: Distant({distant_intensity}) + Dome({dome_intensity})")
    return distant_light, dome_light


def create_dynamic_cube(position=(0.0, 0.0, 0.25), size=0.5, color=(0.0, 1.0, 0.0)):
    """
    Create a dynamic cube that responds to forces
    
    Args:
        position (tuple): (x, y, z) position in meters
        size (float): Size of the cube in meters
        color (tuple): (r, g, b) color values
        
    Returns:
        UsdGeom.Cube: The created cube prim
    """
    stage = get_usd_context().get_stage()
    cube_path = "/World/Cube"
    
    # Create cube geometry
    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.CreateSizeAttr(size)
    cube.AddTranslateOp().Set(Gf.Vec3d(*position))
    
    # Set extent (bounding box)
    half_size = size / 2.0
    extent = [(-half_size, -half_size, -half_size), (half_size, half_size, half_size)]
    cube.CreateExtentAttr(extent)
    
    # Physics: Make it dynamic rigid body
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    
    # Add mass - heavy cube to be stable platform
    mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
    mass_api.CreateMassAttr(10.0)  # 10 kg (100x heavier than ball)
    
    # Lock Y and Z translation, and all rotations to keep it as moving platform
    rigid_body.CreateAngularVelocityAttr(Gf.Vec3f(0, 0, 0))
    
    # Set PhysX properties
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim())
    if physx_rb:
        prim = cube.GetPrim()
        prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("physxRigidBody:contactOffset", Sdf.ValueTypeNames.Float).Set(0.02)
        prim.CreateAttribute("physxRigidBody:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
        # Lock rotation and Y/Z translation with high damping
        prim.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(10000.0)
    
    # Collision API with contact offset
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
    if physx_collision:
        cube.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.02)
        cube.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.001)
    
    # Create and apply material with color
    material_path = f"{cube_path}/Material"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    
    # Bind material
    binding_api = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
    binding_api.Bind(material)
    
    print(f"✓ Dynamic cube created: position={position}, size={size}, mass=10kg, color={color}")
    return cube


def add_physics_material(prim_path, static_friction=0.6, dynamic_friction=0.5, restitution=0.1):
    """Add physics material for friction and bounce properties"""
    stage = get_usd_context().get_stage()
    
    # Create physics material
    material_path = f"{prim_path}/PhysicsMaterial"
    physics_material = UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(material_path, "Material"))
    
    # Set friction and restitution
    physics_material.CreateStaticFrictionAttr(static_friction)
    physics_material.CreateDynamicFrictionAttr(dynamic_friction)
    physics_material.CreateRestitutionAttr(restitution)
    
    # Bind to collision
    prim = stage.GetPrimAtPath(prim_path)
    collision_api = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
    if collision_api:
        collision_api.GetPrim().CreateRelationship("material:binding:physics").AddTarget(material_path)
    
    print(f"✓ Physics material added: friction={static_friction}/{dynamic_friction}, restitution={restitution}")


def create_ball(position=(0.0, 0.0, 1.0), radius=0.1, color=(1.0, 0.0, 0.0)):
    """
    Create a dynamic sphere (ball) with physics
    
    Args:
        position (tuple): (x, y, z) position in meters
        radius (float): Radius of the ball in meters
        color (tuple): (r, g, b) color values
        
    Returns:
        UsdGeom.Sphere: The created ball prim
    """
    stage = get_usd_context().get_stage()
    ball_path = "/World/Ball"
    
    # Create sphere geometry
    sphere = UsdGeom.Sphere.Define(stage, ball_path)
    sphere.CreateRadiusAttr(radius)
    sphere.AddTranslateOp().Set(Gf.Vec3d(*position))
    
    # Set extent (bounding box)
    extent = [(-radius, -radius, -radius), (radius, radius, radius)]
    sphere.CreateExtentAttr(extent)
    
    # Physics: Dynamic rigid body with analytic sphere collision
    UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    # NOTE: Sphere uses analytic collider automatically - no MeshCollisionAPI needed
    
    # Enable CCD on the ball to prevent tunneling
    physx_ball = PhysxSchema.PhysxRigidBodyAPI.Apply(sphere.GetPrim())
    if physx_ball:
        prim_ball = sphere.GetPrim()
        prim_ball.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
        prim_ball.CreateAttribute("physxRigidBody:contactOffset", Sdf.ValueTypeNames.Float).Set(0.01)
        prim_ball.CreateAttribute("physxRigidBody:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    
    # Add mass
    mass_api = UsdPhysics.MassAPI.Apply(sphere.GetPrim())
    mass_api.CreateMassAttr(0.1)  # 100 grams
    
    # Create and apply material with color
    material_path = f"{ball_path}/Material"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    
    # Bind material
    binding_api = UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim())
    binding_api.Bind(material)
    
    print(f"✓ Ball created: position={position}, radius={radius}, color={color}")
    return sphere


def enable_collision_visualization():
    """Enable collision mesh visualization in Isaac Sim"""
    physics_settings = carb.settings.get_settings()
    
    # Enable collision mesh visualization
    physics_settings.set("/physics/visualizationDisplayCollisionMeshes", True)
    physics_settings.set("/physics/visualizationDisplayContacts", True)
    physics_settings.set("/physics/visualizationSimplificationDisplayCollisionMeshes", True)
    
    print("✓ Collision visualization enabled")
    print("  - Collision meshes will show as wireframes")
    print("  - Contact points will show as colored markers")


def set_prim_transform(prim_path, position, scale=(1.0, 1.0, 1.0), rotation=None):
    """Set the position, scale, and rotation of a USD prim"""
    stage = get_usd_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    
    if not prim or not prim.IsValid():
        return False
    
    # Make it transformable
    xform = UsdGeom.Xformable(prim)
    
    # Clear any existing transforms
    xform.ClearXformOpOrder()
    
    # Add translation
    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*position))
    
    # Add rotation if provided
    if rotation is not None:
        rotate_op = xform.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3f(*rotation))
    
    # Add scale
    scale_op = xform.AddScaleOp()
    scale_op.Set(Gf.Vec3f(*scale))
    
    return True


def run_simulation(world, cube_path="/World/Cube", ball_path="/World/Ball"):
    """
    Run the interactive simulation loop with horizontal cube motion
    
    Args:
        world (World): The Isaac Sim world object
        cube_path (str): Path to the cube prim
        ball_path (str): Path to the ball prim
    """
    print("\n" + "=" * 70)
    print("Simulation Running - Dynamic Cube with Velocity Control")
    print("=" * 70)
    print("Controls:")
    print("  - Mouse drag: Rotate view")
    print("  - Middle mouse: Pan")
    print("  - Scroll: Zoom")
    print("  - Ctrl+C to exit")
    print("=" * 70)
    
    # Reset physics to initialize all states
    world.reset()
    
    # Get stage and prims for position tracking
    stage = get_usd_context().get_stage()
    cube_prim = stage.GetPrimAtPath(cube_path)
    ball_prim = stage.GetPrimAtPath(ball_path)
    
    # Get ball radius for surface calculation
    ball_geom = UsdGeom.Sphere(ball_prim)
    ball_radius = ball_geom.GetRadiusAttr().Get()
    
    # Get rigid body API for velocity control
    cube_rb = UsdPhysics.RigidBodyAPI(cube_prim)
    
    # Motion parameters
    A = 1.0   # amplitude in meters
    f = 0.5    # frequency in Hz
    
    step_count = 0
    dt = world.get_physics_dt()
    
    try:
        while simulation_app.is_running():
            t = step_count * dt
            
            # Calculate desired velocity: v(t) = dx/dt = A * 2πf * cos(2πft)
            velocity_x = A * 2.0 * math.pi * f * math.cos(2.0 * math.pi * f * t)
            
            # Apply velocity to cube (only X direction, keep Y and Z at zero)
            cube_rb.GetVelocityAttr().Set(Gf.Vec3f(velocity_x, 0.0, 0.0))
            
            # Lock angular velocity to prevent rotation
            cube_rb.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            
            world.step(render=True)
            step_count += 1
            
            # Print positions every 100 steps
            if step_count % 100 == 0:
                # Get ball position (center)
                ball_xform = UsdGeom.Xformable(ball_prim)
                ball_matrix = ball_xform.ComputeLocalToWorldTransform(0)
                ball_center = ball_matrix.ExtractTranslation()
                ball_bottom = ball_center[2] - ball_radius
                
                # Get cube position
                cube_xform = UsdGeom.Xformable(cube_prim)
                cube_matrix = cube_xform.ComputeLocalToWorldTransform(0)
                cube_pos = cube_matrix.ExtractTranslation()
                
                # Calculate separation distance
                separation = ball_bottom - cube_pos[2]
                
                print(f"Step {step_count:5d} | Time {t:6.2f}s | "
                      f"Cube X: {cube_pos[0]:6.3f} | "
                      f"Ball center: ({ball_center[0]:6.3f}, {ball_center[1]:6.3f}, {ball_center[2]:6.3f}) | "
                      f"Ball bottom: {ball_bottom:6.3f} | Gap: {separation:6.3f}")
    
    except KeyboardInterrupt:
        print("\nStopping...")


def main():
    """Main execution flow"""
    print("=" * 70)
    print("Cube and Ball Simulation in Isaac Sim")
    print("=" * 70)
    
    try:
        # Step 1: Create physics world
        world = create_world_with_ground()
        
        # Step 2: Add lighting to scene
        add_scene_lighting(distant_intensity=1000.0, dome_intensity=300.0)
        
        # Step 3: Create dynamic cube
        create_dynamic_cube(position=(0.0, 0.0, 0.25), size=0.5, color=(0.0, 1.0, 0.0))
        
        # Step 4: Add physics material to cube
        add_physics_material("/World/Cube", static_friction=0.6, dynamic_friction=0.5, restitution=0.1)
        
        # Step 5: Create a ball above the cube
        create_ball(position=(0.0, 0.0, 2.0), radius=0.1, color=(1.0, 0.0, 0.0))
        
        # Step 6: Add physics material to ball
        add_physics_material("/World/Ball", static_friction=0.6, dynamic_friction=0.5, restitution=0.3)
        
        # Step 7: Enable collision visualization
        enable_collision_visualization()
        
        # Step 8: Run simulation with position tracking
        run_simulation(world, cube_path="/World/Cube", ball_path="/World/Ball")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean shutdown
        simulation_app.close()
        print("Done!")


if __name__ == "__main__":
    main()
