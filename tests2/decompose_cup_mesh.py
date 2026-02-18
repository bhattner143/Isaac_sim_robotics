"""
Convex Decomposition for Cup Mesh

This script uses VHACD (Volumetric Hierarchical Approximate Convex Decomposition)
to break down the hollow cup mesh into multiple convex pieces for better collision detection.
"""

import trimesh
import numpy as np
from pathlib import Path

def decompose_mesh(obj_path, output_dir, max_convex_hulls=32, resolution=100000):
    """
    Decompose a mesh into convex hulls using VHACD.
    
    Args:
        obj_path: Path to input .obj file
        output_dir: Directory to save decomposed meshes
        max_convex_hulls: Maximum number of convex hulls to generate
        resolution: VHACD resolution parameter (higher = more detail)
    """
    print(f"\n{'='*70}")
    print(f"CONVEX DECOMPOSITION: {obj_path.name}")
    print(f"{'='*70}\n")
    
    # Load mesh
    print(f"Loading mesh from: {obj_path}")
    mesh = trimesh.load(obj_path)
    print(f"✓ Loaded mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Run VHACD decomposition
    print(f"\nRunning VHACD decomposition...")
    print(f"  Max convex hulls: {max_convex_hulls}")
    print(f"  Resolution: {resolution}")
    
    try:
        # Use trimesh's convex decomposition (calls VHACD)
        convex_pieces = trimesh.decomposition.convex_decomposition(
            mesh, 
            maxhulls=max_convex_hulls,
            resolution=resolution
        )
        
        print(f"✓ Decomposition complete: {len(convex_pieces)} convex pieces generated")
        
        # Save individual pieces
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, piece in enumerate(convex_pieces):
            piece_path = output_dir / f"cup_collision_{i:03d}.obj"
            piece.export(piece_path)
            print(f"  ✓ Saved piece {i+1}/{len(convex_pieces)}: {piece_path.name}")
        
        # Generate URDF collision snippet
        generate_urdf_snippet(convex_pieces, output_dir)
        
        return convex_pieces
        
    except Exception as e:
        print(f"\n⚠ VHACD decomposition failed: {e}")
        print("\nTrying simpler convex hull approach...")
        
        # Fallback: just compute convex hull
        convex_hull = mesh.convex_hull
        hull_path = output_dir / "cup_collision_hull.obj"
        convex_hull.export(hull_path)
        print(f"✓ Saved convex hull: {hull_path}")
        
        return [convex_hull]

def generate_urdf_snippet(convex_pieces, output_dir):
    """Generate URDF XML snippet for multiple collision geometries."""
    
    snippet_path = output_dir / "collision_urdf_snippet.xml"
    
    with open(snippet_path, 'w') as f:
        f.write("<!-- Convex decomposition collision geometries -->\n")
        f.write("<!-- Replace the single collision block with these multiple collision blocks -->\n\n")
        
        for i in range(len(convex_pieces)):
            f.write(f"<collision name=\"cup_collision_{i}\">\n")
            f.write(f"  <origin xyz=\"-0.3015 -8.32667e-17 -0.15\" rpy=\"3.14159 -0 0\"/>\n")
            f.write(f"  <geometry>\n")
            f.write(f"    <mesh filename=\"package://cup_assets/collision/cup_collision_{i:03d}.obj\"/>\n")
            f.write(f"  </geometry>\n")
            f.write(f"</collision>\n")
            if i < len(convex_pieces) - 1:
                f.write("\n")
    
    print(f"\n✓ URDF snippet saved to: {snippet_path}")
    print(f"\nCopy the contents of this file into cup_manipulator_obj.urdf")
    print(f"to replace the current collision block.")

def main():
    # Paths
    obj_path = Path("model_using_onshape_to_robot/cup_manipulator/assets/part_1.obj")
    output_dir = Path("model_using_onshape_to_robot/cup_manipulator/assets/collision")
    
    if not obj_path.exists():
        print(f"Error: Could not find {obj_path}")
        return
    
    # Run decomposition
    convex_pieces = decompose_mesh(
        obj_path, 
        output_dir,
        max_convex_hulls=16,  # Fewer pieces for performance
        resolution=100000
    )
    
    print(f"\n{'='*70}")
    print(f"DECOMPOSITION COMPLETE")
    print(f"{'='*70}")
    print(f"\nGenerated {len(convex_pieces)} convex collision meshes")
    print(f"Output directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Review the generated collision meshes")
    print(f"2. Copy the URDF snippet into cup_manipulator_obj.urdf")
    print(f"3. Test the simulation with the new collision geometries")

if __name__ == "__main__":
    main()
