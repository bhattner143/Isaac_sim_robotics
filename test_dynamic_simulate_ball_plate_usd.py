"""
Import and simulate ball_plate_2.usd directly in Isaac Sim
Loads the complete scene with plate and ball already configured

This script demonstrates:
1. Launching Isaac Sim programmatically
2. Importing a complete USD scene with physics
3. Running the simulation with position tracking
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
from pxr import UsdGeom, Gf
import omni.usd
from pathlib import Path
import carb


def get_usd_context():
    """Get the USD context for the current Isaac Sim session"""
    return omni.usd.get_context()


def import_scene(usd_file_path):
    """
    Import a USD scene file as the root layer
    
    Args:
        usd_file_path (Path or str): Path to the USD file to import
        
    Returns:
        bool: True if successful
    """
    usd_context = get_usd_context()
    success = usd_context.open_stage(str(usd_file_path))
    
    if success:
        print(f"✓ Scene loaded from: {usd_file_path}")
        return True
    else:
        print(f"✗ Failed to load scene from: {usd_file_path}")
        return False


def enable_collision_visualization():
    """Enable collision mesh visualization in Isaac Sim"""
    physics_settings = carb.settings.get_settings()
    
    # Enable collision mesh visualization
    physics_settings.set("/physics/visualizationDisplayCollisionMeshes", True)
    physics_settings.set("/physics/visualizationDisplayContacts", True)
    physics_settings.set("/physics/visualizationSimplificationDisplayCollisionMeshes", True)
    
    print("✓ Collision visualization enabled")


def run_simulation(world, ball_path="/World/Sphere", plate_path="/World/plate"):
    """
    Run the interactive simulation loop
    
    Args:
        world (World): The Isaac Sim world object
        ball_path (str): Path to the ball prim
        plate_path (str): Path to the plate prim
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
    
    # Reset physics to initialize all states
    world.reset()
    
    # Get stage and prims for position tracking
    stage = get_usd_context().get_stage()
    ball_prim = stage.GetPrimAtPath(ball_path)
    plate_prim = stage.GetPrimAtPath(plate_path)
    
    if not ball_prim or not ball_prim.IsValid():
        print(f"Warning: Ball prim not found at {ball_path}")
        ball_prim = None
    
    if not plate_prim or not plate_prim.IsValid():
        print(f"Warning: Plate prim not found at {plate_path}")
        plate_prim = None
    
    # Get ball radius for surface calculation (if available)
    ball_radius = 0.125  # Default: 0.5 * 0.25 scale from USD
    if ball_prim:
        try:
            ball_geom = UsdGeom.Sphere(ball_prim)
            base_radius = ball_geom.GetRadiusAttr().Get()
            # Get scale from xform
            xformable = UsdGeom.Xformable(ball_prim)
            xform_ops = xformable.GetOrderedXformOps()
            for op in xform_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale = op.Get()
                    ball_radius = base_radius * scale[0]  # Assume uniform scale
                    break
            print(f"✓ Ball radius: {ball_radius}m")
        except:
            print(f"Using default ball radius: {ball_radius}m")
    
    step_count = 0
    dt = world.get_physics_dt()
    
    try:
        while simulation_app.is_running():
            world.step(render=True)
            step_count += 1
            
            # Print positions every 100 steps
            if step_count % 100 == 0:
                if ball_prim:
                    # Get ball position (center)
                    ball_xform = UsdGeom.Xformable(ball_prim)
                    ball_matrix = ball_xform.ComputeLocalToWorldTransform(0)
                    ball_center = ball_matrix.ExtractTranslation()
                    ball_bottom = ball_center[2] - ball_radius
                    
                    if plate_prim:
                        # Get plate position
                        plate_xform = UsdGeom.Xformable(plate_prim)
                        plate_matrix = plate_xform.ComputeLocalToWorldTransform(0)
                        plate_pos = plate_matrix.ExtractTranslation()
                        
                        # Calculate separation distance
                        separation = ball_bottom - plate_pos[2]
                        
                        print(f"Step {step_count:5d} | Time {step_count * dt:6.2f}s | "
                              f"Plate: ({plate_pos[0]:6.3f}, {plate_pos[1]:6.3f}, {plate_pos[2]:6.3f}) | "
                              f"Ball center: ({ball_center[0]:6.3f}, {ball_center[1]:6.3f}, {ball_center[2]:6.3f}) | "
                              f"Ball bottom: {ball_bottom:6.3f} | Gap: {separation:6.3f}")
                    else:
                        print(f"Step {step_count:5d} | Time {step_count * dt:6.2f}s | "
                              f"Ball: ({ball_center[0]:6.3f}, {ball_center[1]:6.3f}, {ball_center[2]:6.3f}) | "
                              f"Bottom: {ball_bottom:6.3f}")
    
    except KeyboardInterrupt:
        print("\nStopping...")


def main():
    """Main execution flow"""
    print("=" * 70)
    print("Ball and Plate Simulation - Direct USD Import")
    print("=" * 70)
    
    try:
        # Find the ball_plate_2.usd file
        script_dir = Path(__file__).parent
        usd_file = script_dir / "ball_plate_2.usd"
        
        if not usd_file.exists():
            # Try .usda extension
            usd_file = script_dir / "ball_plate_2.usda"
            
        if not usd_file.exists():
            raise FileNotFoundError(f"Could not find ball_plate_2.usd or ball_plate_2.usda in {script_dir}")
        
        print(f"Found USD file: {usd_file}")
        
        # Import the complete scene
        import_scene(usd_file)
        
        # Create physics world (this will use the scene's existing physics setup)
        world = World(stage_units_in_meters=1.0)
        print("✓ Physics world initialized")
        
        # Enable collision visualization
        enable_collision_visualization()
        
        # Run simulation
        run_simulation(world, ball_path="/World/Sphere", plate_path="/World/plate")
        
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
