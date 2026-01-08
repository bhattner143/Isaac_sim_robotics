"""
Test script for pip-installed ISAAC Sim (isaacsim package)
This uses the newer isaacsim.core.api instead of deprecated omni.isaac.core
"""

from isaacsim import SimulationApp

# Configure and launch the simulation app
config = {
    "headless": False,  # Set to True for headless mode
}

simulation_app = SimulationApp(config)

# Import ISAAC Sim modules after SimulationApp initialization
from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCuboid
import numpy as np

def main():
    print("=" * 60)
    print("Testing pip-installed ISAAC Sim")
    print("=" * 60)
    
    # Create a new world
    world = World(stage_units_in_meters=1.0)
    print("✓ World created")
    
    # Add a ground plane
    world.scene.add_default_ground_plane()
    print("✓ Ground plane added")
    
    # Create a dynamic cube that will fall
    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cube",
            name="dynamic_cube",
            position=np.array([0.0, 0.0, 2.0]),
            scale=np.array([0.5, 0.5, 0.5]),
            color=np.array([0.0, 0.5, 1.0]),  # Blue color
        )
    )
    print("✓ Dynamic cube created at position [0, 0, 2.0]")
    
    # Reset the world to initialize physics
    world.reset()
    print("✓ World reset - physics initialized")
    
    print("\nRunning simulation for 500 steps...")
    print("The cube should fall and bounce on the ground plane")
    
    # Run simulation
    for i in range(500):
        world.step(render=True)
        
        # Print cube position every 100 steps
        if i % 100 == 0:
            position, _ = cube.get_world_pose()
            print(f"  Step {i:3d}: Cube position = [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
    
    print("\n" + "=" * 60)
    print("Simulation completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
