#!/usr/bin/env python3
"""
ISAAC Sim Installation Checker and Setup Instructions

The 'isaacsim' pip package is just a launcher stub. To use ISAAC Sim, you need to:

1. Download ISAAC Sim from NVIDIA:
   https://developer.nvidia.com/isaac-sim

2. Install options:
   a) Via Omniverse Launcher (recommended for beginners)
   b) Direct download (for advanced users)
   c) Docker container

3. After installation, set the ISAAC_SIM_PATH environment variable:
   export ISAAC_SIM_PATH="/path/to/isaac-sim"

4. Then you can use this script to test the installation.
"""

import sys
import os


def check_installation():
    """Check if ISAAC Sim is properly installed."""
    
    print("=" * 60)
    print("ISAAC Sim Installation Checker")
    print("=" * 60)
    
    # Check if isaacsim package is installed
    try:
        import isaacsim
        print("✓ isaacsim package found")
        print(f"  Location: {isaacsim.__file__}")
    except ImportError:
        print("✗ isaacsim package not found")
        print("  Install with: pip install isaacsim")
        return False
    
    # Check if SimulationApp is available
    from isaacsim import SimulationApp
    if SimulationApp is None:
        print("✗ SimulationApp is not available")
        print("\n" + "=" * 60)
        print("ISAAC Sim full installation required!")
        print("=" * 60)
        print("\nThe pip package 'isaacsim' is just a stub/launcher.")
        print("You need to install the full ISAAC Sim application:\n")
        print("Option 1 - Omniverse Launcher (Easiest):")
        print("  1. Download from: https://www.nvidia.com/en-us/omniverse/download/")
        print("  2. Install Omniverse Launcher")
        print("  3. From launcher, install 'Isaac Sim'")
        print("\nOption 2 - Direct Installation:")
        print("  1. Visit: https://developer.nvidia.com/isaac-sim")
        print("  2. Download ISAAC Sim for your platform")
        print("  3. Extract and follow installation instructions")
        print("\nOption 3 - Docker:")
        print("  docker pull nvcr.io/nvidia/isaac-sim:2023.1.1")
        print("\nAfter installation, set environment variable:")
        print("  export ISAAC_SIM_PATH='/path/to/isaac-sim'")
        print("\nSystem Requirements:")
        print("  - GPU: NVIDIA RTX series or better")
        print("  - RAM: 32GB+ recommended")
        print("  - OS: Ubuntu 20.04/22.04 or Windows 10/11")
        print("=" * 60)
        return False
    
    print("✓ SimulationApp is available")
    
    # Check for ISAAC_SIM_PATH
    isaac_path = os.environ.get('ISAAC_SIM_PATH')
    if isaac_path:
        print(f"✓ ISAAC_SIM_PATH set to: {isaac_path}")
        if os.path.exists(isaac_path):
            print("✓ ISAAC Sim directory exists")
        else:
            print("✗ ISAAC Sim directory not found at specified path")
            return False
    else:
        print("⚠ ISAAC_SIM_PATH environment variable not set")
        print("  Set it with: export ISAAC_SIM_PATH='/path/to/isaac-sim'")
    
    print("\n" + "=" * 60)
    print("Installation check complete!")
    print("=" * 60)
    
    return True


def test_simulation():
    """Test launching ISAAC Sim if properly installed."""
    
    try:
        from isaacsim import SimulationApp
        
        if SimulationApp is None:
            print("\nCannot test simulation - ISAAC Sim not fully installed")
            return False
        
        print("\nAttempting to launch ISAAC Sim...")
        print("(This may take a minute on first launch)")
        
        # Launch with minimal configuration
        simulation_app = SimulationApp({"headless": False})
        
        # If we get here, it worked!
        print("✓ ISAAC Sim launched successfully!")
        
        # Import Isaac modules
        from omni.isaac.core import World
        from omni.isaac.core.objects import DynamicCuboid
        import numpy as np
        
        # Create a simple scene
        world = World()
        world.scene.add_default_ground_plane()
        
        cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube",
                name="test_cube",
                position=np.array([0, 0, 1.0]),
                scale=np.array([0.5, 0.5, 0.5]),
                color=np.array([0.0, 0.5, 1.0]),
            )
        )
        
        world.reset()
        
        print("\nScene created with:")
        print("  - Ground plane")
        print("  - Blue cube at [0, 0, 1.0]")
        print("\nRunning for 100 steps then closing...")
        
        # Run for a few steps
        for i in range(100):
            world.step(render=True)
        
        simulation_app.close()
        print("\n✓ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error during simulation test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n")
    
    # Check installation first
    if check_installation():
        # If check passes, offer to test
        print("\nWould you like to test launching the simulator? (requires full ISAAC Sim)")
        response = input("Enter 'y' to test (or any key to skip): ").lower()
        
        if response == 'y':
            test_simulation()
    else:
        print("\nPlease install ISAAC Sim following the instructions above.")
        sys.exit(1)
