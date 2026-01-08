#!/usr/bin/env python3
"""
Minimal ISAAC Sim headless test - no GUI, just verifies everything loads
Run this with: ./python.sh simple_headless_test.py
"""

from isaacsim import SimulationApp

# Create simulation app in headless mode (no GUI, less resource intensive)
simulation_app = SimulationApp({"headless": True})

print("=" * 60)
print("ISAAC Sim Headless Test")
print("=" * 60)

# Import after SimulationApp is created
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
import numpy as np

print("✓ SimulationApp created successfully")
print("✓ Imports successful")

# Create world
world = World()
print("✓ World created")

# Add ground plane
world.scene.add_default_ground_plane()
print("✓ Ground plane added")

# Add a simple cube
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/TestCube",
        name="test_cube",
        position=np.array([0, 0, 1.0]),
        scale=np.array([0.2, 0.2, 0.2]),
    )
)
print("✓ Cube added at position [0, 0, 1.0]")

# Reset the world
world.reset()
print("✓ World reset complete")

# Run just a few physics steps to verify simulation works
print("\nRunning 10 physics steps...")
for i in range(10):
    world.step(render=False)  # render=False for headless
    
# Get cube position after simulation
cube_position = cube.get_world_pose()[0]
print(f"✓ Simulation complete - Cube position: [{cube_position[0]:.3f}, {cube_position[1]:.3f}, {cube_position[2]:.3f}]")

print("\n" + "=" * 60)
print("✅ All tests passed! ISAAC Sim is working correctly.")
print("=" * 60)

# Cleanup
simulation_app.close()
print("Closed successfully")
