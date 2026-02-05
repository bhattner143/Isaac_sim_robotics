"""
Visualize Cup Mesh and Manual Convex Decomposition

Shows the original mesh alongside the primitive collision shapes.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

def load_obj(filepath):
    """Load vertices and faces from .obj file."""
    vertices = []
    faces = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.strip().split()
                face = []
                for part in parts[1:]:
                    vertex_idx = int(part.split('/')[0]) - 1
                    face.append(vertex_idx)
                faces.append(face)
    
    return np.array(vertices), faces

def create_cylinder_mesh(radius, height, center, resolution=20):
    """Create cylinder mesh vertices."""
    theta = np.linspace(0, 2*np.pi, resolution)
    z = np.array([center[2] - height/2, center[2] + height/2])
    
    vertices = []
    # Bottom circle
    for t in theta:
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        vertices.append([x, y, z[0]])
    # Top circle
    for t in theta:
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        vertices.append([x, y, z[1]])
    
    return np.array(vertices)

def create_box_mesh(size, center, rpy):
    """Create box mesh vertices with rotation."""
    # Box corners (local)
    sx, sy, sz = size[0]/2, size[1]/2, size[2]/2
    corners = np.array([
        [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
        [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]
    ])
    
    # Rotation matrix from RPY
    r, p, y = rpy
    # Simplified: just yaw rotation for now
    Rz = np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y), np.cos(y), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(p), 0, np.sin(p)],
        [0, 1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r), np.cos(r)]
    ])
    
    R = Rz @ Ry @ Rx
    
    # Rotate and translate
    rotated = (R @ corners.T).T
    translated = rotated + center
    
    return translated

def plot_collision_shapes(ax, alpha=0.3):
    """Plot the primitive collision shapes."""
    
    # Bottom disk
    disk = create_cylinder_mesh(0.23, 0.01, [0.15, 0.0, -0.215])
    ax.scatter(disk[:, 0], disk[:, 1], disk[:, 2], c='red', s=2, alpha=alpha, label='Bottom disk')
    
    # Wall segments
    wall_configs = [
        ([0.38, 0.0, -0.12], [0, 0.3, 0], [0.02, 0.4, 0.18]),
        ([0.15, 0.23, -0.12], [0.3, 0, 1.571], [0.02, 0.4, 0.18]),
        ([-0.08, 0.23, -0.12], [0.3, 0, 1.571], [0.02, 0.4, 0.18]),
        ([-0.08, -0.23, -0.12], [0.3, 0, 1.571], [0.02, 0.4, 0.18]),
        ([0.15, -0.23, -0.12], [0.3, 0, 1.571], [0.02, 0.4, 0.18]),
    ]
    
    for i, (center, rpy, size) in enumerate(wall_configs):
        box = create_box_mesh(size, center, rpy)
        ax.scatter(box[:, 0], box[:, 1], box[:, 2], c='orange', s=3, alpha=alpha, 
                  label='Wall segments' if i == 0 else '')

def main():
    obj_path = Path("model_using_onshape_to_robot/cup_manipulator/assets/part_1.obj")
    
    if not obj_path.exists():
        print(f"Error: Could not find {obj_path}")
        return
    
    print(f"Loading mesh from: {obj_path}")
    vertices, faces = load_obj(obj_path)
    print(f"✓ Loaded {len(vertices)} vertices, {len(faces)} faces")
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 7))
    
    # Plot 1: Original mesh
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='blue', s=1, alpha=0.3)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Original Cup Mesh', fontsize=14, fontweight='bold')
    ax1.set_box_aspect([1,1,1])
    
    # Plot 2: Mesh + Collision shapes overlay
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='blue', s=1, alpha=0.15, label='Original mesh')
    plot_collision_shapes(ax2, alpha=0.6)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('Manual Convex Decomposition\n(Primitives: 1 disk + 5 wall boxes)', 
                  fontsize=14, fontweight='bold')
    ax2.set_box_aspect([1,1,1])
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("plots/cup_collision_decomposition.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
