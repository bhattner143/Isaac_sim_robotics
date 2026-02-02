"""
Test script to verify JSON configuration saving functionality.

This script tests the automatic configuration serialization without running
the full simulation, so you can quickly verify the JSON output.
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# ============================================================================
# TEST CONFIGURATION CLASSES (Same as in main scripts)
# ============================================================================

@dataclass
class RobotParams:
    """Parameters for robot configuration."""
    urdf_path: str
    usd_path: str
    prim_path: str
    position: tuple
    rotation_z: float
    initial_joint_positions: list
    joint_damping: list
    joint_stiffness: list
    joint_friction: list
    link_lengths: list = None


@dataclass
class LightingParams:
    """Parameters for scene lighting."""
    distant_intensity: float = 1000.0
    dome_intensity: float = 300.0
    angle: float = 315.0


# ============================================================================
# CREATE TEST CONFIGURATION
# ============================================================================

def create_test_config():
    """Create a test configuration dictionary similar to actual simulations."""
    
    # Create sample robot parameters
    cart_pendulum_params = RobotParams(
        urdf_path="model/manipulators/cart_pendulum_2dof.urdf",
        usd_path="model/manipulators/cart_pendulum_2dof.usd",
        prim_path="/World/cart_pendulum",
        position=(0.0, 0.0, 0.0),
        rotation_z=0.0,
        initial_joint_positions=[0.0, 0.0],
        joint_damping=[0.5, 0.05],
        joint_stiffness=[0, 0.1],
        joint_friction=[0.05, 0.0]
    )
    
    manipulator_params = RobotParams(
        urdf_path="model/manipulators/2dof_planar_manipulator.urdf",
        usd_path="model/manipulators/2dof_planar_manipulator.usd",
        prim_path="/World/a_dof_planar_manipulator",
        position=(-3.0, 0.0, 0.0),
        rotation_z=0.0,
        initial_joint_positions=[0.8727, -1.7453],  # ~50°, -100°
        joint_damping=[0.1, 0.1],
        joint_stiffness=[0.0, 0.0],
        joint_friction=[0.0, 0.0],
        link_lengths=[1.0, 1.0]
    )
    
    lighting_params = LightingParams(
        distant_intensity=1000.0,
        dome_intensity=300.0,
        angle=315.0
    )
    
    # Build the complete configuration dictionary
    config = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "simulation_mode": "coupled-motion",
            "device": "cpu",
            "framework": "Isaac Sim",
        },
        "simulation_parameters": {
            "time_step": 0.001,
            "duration": 4.0,
            "settling_time": 2.0,
        },
        "cart_pendulum_config": asdict(cart_pendulum_params),
        "manipulator_config": asdict(manipulator_params),
        "lighting_config": asdict(lighting_params),
        "coupling_joint_config": {
            "type": "revolute",
            "revolute": {
                "stiffness": 500.0,
                "damping": 100.0,
                "friction": 0.5,
                "axis": "Z",
            },
            "prismatic": {
                "stiffness": 50.0,
                "damping": 10.0,
                "friction": 1.0,
                "axis": "-X",
            }
        },
        "video_recording": {
            "enabled": True,
            "output_path": "videos/simulation_recording.mp4",
            "resolution": [1920, 1080],
            "fps": 60,
        }
    }
    
    return config


# ============================================================================
# SAVE AND DISPLAY CONFIGURATION
# ============================================================================

def main():
    """Test the configuration saving functionality."""
    
    print("\n" + "="*70)
    print("TESTING CONFIGURATION SAVING")
    print("="*70 + "\n")
    
    # Create test configuration
    print("1. Creating configuration dictionary using dataclasses.asdict()...")
    config = create_test_config()
    print("   ✓ Configuration created\n")
    
    # Create output directory
    print("2. Creating output directory 'configs/'...")
    os.makedirs("configs", exist_ok=True)
    print("   ✓ Directory created\n")
    
    # Save to JSON file
    print("3. Saving configuration to JSON file...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"configs/simulation_config_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"   ✓ Configuration saved to: {output_path}\n")
    
    # Display the JSON content
    print("4. JSON file contents:\n")
    print("-" * 70)
    with open(output_path, 'r') as f:
        print(f.read())
    print("-" * 70 + "\n")
    
    # Verify file was created
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"5. Verification:")
        print(f"   ✓ File exists: {output_path}")
        print(f"   ✓ File size: {file_size} bytes\n")
        
        # Reload and verify contents
        with open(output_path, 'r') as f:
            loaded_config = json.load(f)
        
        print(f"   ✓ JSON is valid and parseable")
        print(f"   ✓ Configuration keys: {list(loaded_config.keys())}\n")
        
        print("="*70)
        print("TEST PASSED: Configuration saving works correctly!")
        print("="*70 + "\n")
        
        return True
    else:
        print(f"ERROR: File not created at {output_path}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
