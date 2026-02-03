#!/usr/bin/env python3
"""
Convert all STL files in simple_pendulum assets to OBJ format.
This makes them compatible with Drake's Meshcat visualization.
"""

from pathlib import Path
import trimesh

ASSETS_DIR = Path("model_using_onshape_to_robot/cup_manipulator/assets")

print(f"Converting STL files in {ASSETS_DIR} to OBJ format...\n")

stl_files = list(ASSETS_DIR.glob("*.stl"))
if not stl_files:
    print("No STL files found!")
    exit(1)

for stl_file in stl_files:
    obj_file = stl_file.with_suffix(".obj")
    
    print(f"Converting {stl_file.name}...")
    try:
        mesh = trimesh.load(stl_file)
        mesh.export(obj_file)
        print(f"  ✓ Created {obj_file.name}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print(f"\n✓ Conversion complete! {len(stl_files)} files processed.")
print("\nNext step: Update URDF to reference .obj instead of .stl files")