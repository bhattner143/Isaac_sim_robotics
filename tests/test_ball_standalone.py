"""
Standalone Isaac Sim script - run from terminal with VS Code as editor
This launches Isaac Sim, creates physics scene, adds ball, and runs simulation
"""

from isaacsim import SimulationApp

# Launch Isaac Sim
simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
})

# Now import other modules (must be after SimulationApp creation)
from pxr import UsdGeom, UsdPhysics, Gf
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicSphere
import numpy as np

def main():
    # Create world with physics
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    print("=" * 70)
    print("BOUNCING BALL TEST - Standalone Script")
    print("=" * 70)
    
    # Create ball using Isaac Sim's DynamicSphere
    ball = DynamicSphere(
        prim_path="/World/Ball",
        name="red_ball",
        position=np.array([0.0, 0.0, 2.0]),
        radius=0.12,
        color=np.array([0.9, 0.1, 0.1]),
        mass=0.5
    )
    
    print("✓ Ball created at (0, 0, 2.0) with radius 0.12m, mass 0.5kg")
    print("\nStarting simulation for 10 seconds...")
    print("=" * 70)
    
    # Reset physics
    world.reset()
    
    # Run simulation
    simulation_time = 0.0
    max_time = 10.0
    step_count = 0
    
    while simulation_app.is_running() and simulation_time < max_time:
        world.step(render=True)
        simulation_time += world.get_physics_dt()
        step_count += 1
        
        # Print ball height every second
        if step_count % 100 == 0:
            pos, _ = ball.get_world_pose()
            print(f"t={simulation_time:.1f}s: Ball height = {pos[2]:.3f}m")
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    
    simulation_app.close()

if __name__ == "__main__":
    main()
