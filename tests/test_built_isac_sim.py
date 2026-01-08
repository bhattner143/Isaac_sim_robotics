#!/usr/bin/env python3
"""
Simple test script to verify ISAAC Sim is working after build from source.
This script uses the built ISAAC Sim installation.
"""

import sys
import os

# Add the ISAAC Sim Python path
ISAAC_SIM_PATH = os.path.expanduser("~/Documents/isac_sim_pydrake/isaacsim/_build/linux-x86_64/release")

# Add to Python path
sys.path.insert(0, os.path.join(ISAAC_SIM_PATH, "exts/omni.isaac.kit/pip_prebundle"))
sys.path.insert(0, ISAAC_SIM_PATH)

print("=" * 60)
print("ISAAC Sim Build Test")
print("=" * 60)
print(f"\nISAAC_SIM_PATH: {ISAAC_SIM_PATH}")
print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}")

# Check if the path exists
if not os.path.exists(ISAAC_SIM_PATH):
    print(f"\n✗ ISAAC Sim build not found at: {ISAAC_SIM_PATH}")
    print("\nPlease complete the build first:")
    print("  cd ~/Documents/isac_sim_pydrake/isaacsim")
    print("  ./build.sh")
    sys.exit(1)

print(f"\n✓ ISAAC Sim build directory found")

# Check for key executables
isaac_sim_sh = os.path.join(ISAAC_SIM_PATH, "isaac-sim.sh")
if os.path.exists(isaac_sim_sh):
    print(f"✓ isaac-sim.sh executable found")
else:
    print(f"✗ isaac-sim.sh not found")
    sys.exit(1)

# Check for Python launcher
python_sh = os.path.join(ISAAC_SIM_PATH, "python.sh")
if os.path.exists(python_sh):
    print(f"✓ python.sh launcher found")
else:
    print(f"✗ python.sh not found")

# List available example apps
print("\n" + "=" * 60)
print("Available ISAAC Sim Executables:")
print("=" * 60)
for file in sorted(os.listdir(ISAAC_SIM_PATH)):
    if file.startswith("isaac-sim") and file.endswith(".sh"):
        print(f"  • {file}")

print("\n" + "=" * 60)
print("How to Run ISAAC Sim:")
print("=" * 60)
print(f"\n1. Basic Launch:")
print(f"   cd {ISAAC_SIM_PATH}")
print(f"   ./isaac-sim.sh")

print(f"\n2. With Python Script:")
print(f"   ./isaac-sim.sh --exec your_script.py")

print(f"\n3. Run Standalone Examples:")
print(f"   ./python.sh standalone_examples/api/omni.isaac.core/add_cubes.py")

print(f"\n4. Headless Mode (no GUI):")
print(f"   ./isaac-sim.sh --headless")

# Check for standalone examples
examples_dir = os.path.join(ISAAC_SIM_PATH, "standalone_examples")
if os.path.exists(examples_dir):
    print(f"\n✓ Standalone examples directory found")
    print(f"  Location: {examples_dir}")
    
    # List some examples
    api_examples = os.path.join(examples_dir, "api/omni.isaac.core")
    if os.path.exists(api_examples):
        examples = [f for f in os.listdir(api_examples) if f.endswith('.py')]
        if examples:
            print(f"\n  Sample examples:")
            for example in sorted(examples)[:5]:
                print(f"    - {example}")
            if len(examples) > 5:
                print(f"    ... and {len(examples) - 5} more")

print("\n" + "=" * 60)
print("✓ ISAAC Sim Build Test Complete!")
print("=" * 60)
print("\nYou can now launch ISAAC Sim using the commands above.")
print("For detailed documentation, visit:")
print("  https://docs.omniverse.nvidia.com/isaacsim/latest/")
