"""
Test script to verify PyDrake works with ISAAC Sim
This creates a simple PyDrake system and runs it alongside ISAAC Sim
"""

# Test PyDrake import first
print("Testing PyDrake installation...")
try:
    from pydrake.systems.framework import DiagramBuilder
    from pydrake.systems.primitives import ConstantVectorSource, LogVectorOutput
    from pydrake.systems.analysis import Simulator
    import numpy as np
    print("✓ PyDrake imported successfully")
except ImportError as e:
    print(f"✗ PyDrake import failed: {e}")
    exit(1)

# Test ISAAC Sim
print("\nTesting ISAAC Sim with PyDrake...")
from isaacsim import SimulationApp

config = {
    "headless": False,
}

simulation_app = SimulationApp(config)

from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCuboid
import numpy as np

def create_pydrake_system():
    """Create a simple PyDrake system"""
    builder = DiagramBuilder()
    
    # Create a constant source
    source = builder.AddSystem(ConstantVectorSource([1.0, 2.0, 3.0]))
    
    # Add a logger
    logger = LogVectorOutput(source.get_output_port(), builder)
    
    # Build the diagram
    diagram = builder.Build()
    
    return diagram, logger

def main():
    print("=" * 60)
    print("Testing PyDrake + ISAAC Sim Integration")
    print("=" * 60)
    
    # Create PyDrake system
    print("\n[PyDrake] Creating system...")
    diagram, logger = create_pydrake_system()
    simulator = Simulator(diagram)
    simulator.Initialize()
    print("✓ PyDrake system created")
    
    # Create ISAAC Sim world
    print("\n[ISAAC Sim] Creating world...")
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    # Add a cube
    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cube",
            name="dynamic_cube",
            position=np.array([0.0, 0.0, 2.0]),
            scale=np.array([0.5, 0.5, 0.5]),
            color=np.array([0.0, 0.5, 1.0]),
        )
    )
    print("✓ ISAAC Sim world created")
    
    # Reset the world
    world.reset()
    print("✓ Physics initialized")
    
    print("\n" + "=" * 60)
    print("Running integrated simulation...")
    print("=" * 60)
    
    # Run simulation
    for i in range(200):
        # Advance PyDrake simulation
        simulator.AdvanceTo(i * 0.01)
        
        # Advance ISAAC Sim
        world.step(render=True)
        
        if i % 50 == 0:
            # Get PyDrake logger data
            pydrake_data = logger.FindLog(simulator.get_context()).data()
            
            # Get ISAAC Sim cube position
            position, _ = cube.get_world_pose()
            
            print(f"\nStep {i:3d}:")
            print(f"  PyDrake output: {pydrake_data[:, -1]}")
            print(f"  Cube position:  [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
    
    print("\n" + "=" * 60)
    print("Integration test completed successfully!")
    print("PyDrake and ISAAC Sim are working together!")
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
