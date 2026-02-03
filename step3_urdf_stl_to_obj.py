#!/usr/bin/env python3
"""
Convert URDF file to use .obj mesh files instead of .stl files.

This script reads a URDF file and replaces all references to .stl mesh files
with .obj mesh files, then saves the result as a new URDF file.
"""

import os
from pathlib import Path


def convert_urdf_stl_to_obj(input_urdf_path, output_urdf_path):
    """
    Convert all .stl mesh references in a URDF file to .obj references.
    
    Args:
        input_urdf_path: Path to the input URDF file
        output_urdf_path: Path where the converted URDF will be saved
    """
    # Read the input URDF file
    with open(input_urdf_path, 'r') as f:
        urdf_content = f.read()
    
    # Replace all .stl extensions with .obj
    converted_content = urdf_content.replace('.stl', '.obj')
    
    # Write the converted URDF to the output file
    with open(output_urdf_path, 'w') as f:
        f.write(converted_content)
    
    print(f"✓ Converted URDF saved to: {output_urdf_path}")
    
    # Count how many replacements were made
    stl_count = urdf_content.count('.stl')
    print(f"  Replaced {stl_count} .stl references with .obj")


def main():
    # Define paths
    script_dir = Path(__file__).parent
    simple_pendulum_dir = script_dir / "model_using_onshape_to_robot" / "cup_manipulator"
    
    input_urdf = simple_pendulum_dir / "cup_manipulator.urdf"
    output_urdf = simple_pendulum_dir / "cup_manipulator_obj.urdf"
    
    # Check if input file exists
    if not input_urdf.exists():
        print(f"Error: Input URDF file not found: {input_urdf}")
        return 1
    
    print(f"Converting URDF: {input_urdf.name}")
    print(f"Output file: {output_urdf.name}")
    print()
    
    # Perform conversion
    convert_urdf_stl_to_obj(input_urdf, output_urdf)
    
    return 0


if __name__ == "__main__":
    exit(main())