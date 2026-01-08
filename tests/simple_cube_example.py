#!/usr/bin/env python3
"""
Simple ISAAC Sim example - creates a cube and ground plane
Run this with: ./python.sh simple_cube_example.py
"""

from isaacsim import SimulationApp

# Create simulation app
simulation_app = SimulationApp({"headless": False})

# Import after SimulationApp is created
import omni
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
import numpy as np

# Create world
world = World()
print("World created")

# Add ground plane
world.scene.add_default_ground_plane()
print("Ground plane added")

# Add a cube
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Cube",
        name="my_cube",
        position=np.array([0, 0, 1.0]),
        scale=np.array([0.5, 0.5, 0.5]),
        color=np.array([0.0, 0.5, 1.0]),
    )
)
print("Blue cube added at position [0, 0, 1.0]")

# Reset the world
world.reset()
print("World reset - simulation starting")

# Run for 500 steps then exit
print("Running simulation for 500 steps (then auto-closing)...")
for i in range(500):
    world.step(render=True)
    if i % 100 == 0:
        print(f"Step {i}/500")

print("Simulation complete, closing...")
simulation_app.close()
print("Done!")
