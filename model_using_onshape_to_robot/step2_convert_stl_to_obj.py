#!/usr/bin/env python3
"""
Convert all STL files in model assets to OBJ format.
This makes them compatible with Drake's Meshcat visualization.

Usage:
    python step2_convert_stl_to_obj.py ball
    python step2_convert_stl_to_obj.py cup_manipulator
"""

import argparse
import sys
from pathlib import Path
import trimesh


def convert_stl_to_obj(model_name):
    """Convert STL files to OBJ for a specific model."""

    script_dir = Path(__file__).parent
    assets_dir = script_dir / model_name / "assets"
    
    if not assets_dir.exists():
        print(f"Error: Assets directory not found: {assets_dir}")
        return 1
    
    print(f"Converting STL files in {assets_dir} to OBJ format...\n")
    
    stl_files = list(assets_dir.glob("*.stl"))
    if not stl_files:
        print(f"No STL files found in {assets_dir}!")
        return 1
    
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
    print(f"\nNext step: Run step3_urdf_stl_to_obj.py {model_name}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert STL mesh files to OBJ format for Drake compatibility"
    )
    parser.add_argument(
        "model_name",
        type=str,
        nargs='?',
        default="manipulator_cable",
        help="Name of the model (e.g., 'ball', 'cup', 'cup_manipulator', 'cup_manipulator2', 'manipulator_cable')"
    )
    
    args = parser.parse_args()
    
    return convert_stl_to_obj(args.model_name)


if __name__ == "__main__":
    sys.exit(main())