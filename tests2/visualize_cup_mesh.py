"""
Visualize Cup Mesh Geometry

This script loads and visualizes the cup .obj file to understand its geometry
and determine appropriate collision shapes.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def load_obj(filepath):
    """Load vertices and faces from .obj file."""
    vertices = []
    faces = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Vertex: v x y z
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                # Face: f v1 v2 v3 (or v1/vt1/vn1 format)
                parts = line.strip().split()
                face = []
                for part in parts[1:]:
                    # Handle formats like "v/vt/vn" or just "v"
                    vertex_idx = int(part.split('/')[0]) - 1  # OBJ is 1-indexed
                    face.append(vertex_idx)
                faces.append(face)
    
    return np.array(vertices), faces

def analyze_mesh(vertices):
    """Analyze mesh geometry."""
    print("\n" + "="*70)
    print("MESH GEOMETRY ANALYSIS")
    print("="*70)
    
    print(f"\nNumber of vertices: {len(vertices)}")
    print(f"\nBounding box:")
    print(f"  X: [{vertices[:, 0].min():.4f}, {vertices[:, 0].max():.4f}] (range: {vertices[:, 0].max() - vertices[:, 0].min():.4f})")
    print(f"  Y: [{vertices[:, 1].min():.4f}, {vertices[:, 1].max():.4f}] (range: {vertices[:, 1].max() - vertices[:, 1].min():.4f})")
    print(f"  Z: [{vertices[:, 2].min():.4f}, {vertices[:, 2].max():.4f}] (range: {vertices[:, 2].max() - vertices[:, 2].min():.4f})")
    
    center = vertices.mean(axis=0)
    print(f"\nCenter of mass (approximate): [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    
    # Estimate cup dimensions
    radius = (vertices[:, 0].max() - vertices[:, 0].min()) / 2
    height = vertices[:, 2].max() - vertices[:, 2].min()
    
    print(f"\nEstimated dimensions:")
    print(f"  Diameter: {2*radius:.4f} m ({2*radius*100:.2f} cm)")
    print(f"  Height: {height:.4f} m ({height*100:.2f} cm)")
    
    return center, radius, height

def plot_mesh(vertices, faces):
    """Plot mesh in 3D."""
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='blue', s=1, alpha=0.5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Cup Mesh - 3D View')
    ax1.set_box_aspect([1,1,1])
    
    # Plot 2: Top view (X-Y plane)
    ax2 = fig.add_subplot(132)
    ax2.scatter(vertices[:, 0], vertices[:, 1], c='blue', s=1, alpha=0.5)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Cup Mesh - Top View (X-Y)')
    ax2.axis('equal')
    ax2.grid(True)
    
    # Plot 3: Side view (X-Z plane)
    ax3 = fig.add_subplot(133)
    ax3.scatter(vertices[:, 0], vertices[:, 2], c='blue', s=1, alpha=0.5)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Cup Mesh - Side View (X-Z)')
    ax3.axis('equal')
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("plots/cup_mesh_visualization.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    plt.show()

def suggest_collision_geometry(center, radius, height):
    """Suggest collision geometry based on mesh analysis."""
    print("\n" + "="*70)
    print("SUGGESTED COLLISION GEOMETRY")
    print("="*70)
    
    print("\nFor a hollow cup, use these collision shapes in URDF:")
    print("\n1. Bottom disk (to catch ball):")
    print(f"   <collision>")
    print(f"     <origin xyz=\"{center[0]:.4f} {center[1]:.4f} {center[2] - height/2:.4f}\" rpy=\"0 0 0\"/>")
    print(f"     <geometry>")
    print(f"       <cylinder radius=\"{radius*0.9:.4f}\" length=\"0.01\"/>")
    print(f"     </geometry>")
    print(f"   </collision>")
    
    print("\n2. Outer wall (cylinder ring):")
    print(f"   <collision>")
    print(f"     <origin xyz=\"{center[0]:.4f} {center[1]:.4f} {center[2]:.4f}\" rpy=\"0 0 0\"/>")
    print(f"     <geometry>")
    print(f"       <cylinder radius=\"{radius:.4f}\" length=\"{height:.4f}\"/>")
    print(f"     </geometry>")
    print(f"   </collision>")

def main():
    obj_path = Path("model_using_onshape_to_robot/cup_manipulator/assets/part_1.obj")
    
    if not obj_path.exists():
        print(f"Error: Could not find {obj_path}")
        return
    
    print(f"\nLoading mesh from: {obj_path}")
    vertices, faces = load_obj(obj_path)
    
    center, radius, height = analyze_mesh(vertices)
    plot_mesh(vertices, faces)
    suggest_collision_geometry(center, radius, height)

if __name__ == "__main__":
    main()
