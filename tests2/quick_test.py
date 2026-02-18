#!/usr/bin/env python3
"""
Quick ISAAC Sim verification test
"""
import sys

print("\n" + "="*60)
print("STARTING ISAAC SIM TEST")
print("="*60 + "\n")

try:
    from isaacsim import SimulationApp
    print("✅ Step 1: Imported SimulationApp")
    
    simulation_app = SimulationApp({"headless": True})
    print("✅ Step 2: Created SimulationApp (headless)")
    
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicCuboid
    import numpy as np
    print("✅ Step 3: Imported Isaac Core modules")
    
    world = World()
    print("✅ Step 4: Created World")
    
    world.scene.add_default_ground_plane()
    print("✅ Step 5: Added ground plane")
    
    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/TestCube",
            name="test_cube",
            position=np.array([0, 0, 2.0]),
            scale=np.array([0.3, 0.3, 0.3]),
        )
    )
    print("✅ Step 6: Added cube at position [0, 0, 2.0]")
    
    world.reset()
    print("✅ Step 7: World reset complete")
    
    print("\n🔄 Running 5 physics steps...")
    for i in range(5):
        world.step(render=False)
    
    pos = cube.get_world_pose()[0]
    print(f"✅ Step 8: Simulation complete - Final cube position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    simulation_app.close()
    print("✅ Step 9: Closed SimulationApp\n")
    
    print("="*60)
    print("🎉 ALL TESTS PASSED! ISAAC SIM IS WORKING! 🎉")
    print("="*60 + "\n")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
