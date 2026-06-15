"""Matplotlib visualization for MHP cable routing."""
from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from cable.path_mhp import compute_cable_path
from cable.types_mhp import CableRouteConfig, MHPKinematics

def disc_3d(ax, center: np.ndarray, radius: float, color: str,
             alpha: float = 0.25, n: int = 60):
    """Draw a horizontal (Z-normal) filled disc in 3D at *center*."""
    theta = np.linspace(0, 2 * np.pi, n)
    xs = center[0] + radius * np.cos(theta)
    ys = center[1] + radius * np.sin(theta)
    zs = np.full_like(xs, center[2])
    ax.plot(xs, ys, zs, color=color, linewidth=1.2, alpha=alpha + 0.2)
    ax.plot_surface(
        xs.reshape(1, -1), ys.reshape(1, -1), zs.reshape(1, -1),
        color=color, alpha=alpha
    )


def circle_xy(ax, center: np.ndarray, radius: float, color: str,
               alpha: float = 0.3, lw: float = 1.5, zorder: int = 3):
    """Draw a filled circle in the XY top-view axes."""
    patch = mpatches.Circle(
        (center[0], center[1]), radius,
        color=color, alpha=alpha, linewidth=lw,
        edgecolor=color, fill=True, zorder=zorder,
    )
    ax.add_patch(patch)


def draw_joint_frame(ax3d, ax_top, pos: np.ndarray, R: np.ndarray,
                      label: str, scale: float = 0.025, ax_top_3d=None):
    """Draw XYZ frame trident at *pos* with rotation *R* in all provided axes."""
    colors = ('red', 'green', 'blue')
    for i, c in enumerate(colors):
        end = pos + scale * R[:, i]
        ax3d.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]],
                  color=c, linewidth=1.8, alpha=0.9, zorder=8)
        if ax_top_3d is not None:
            ax_top_3d.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]],
                          color=c, linewidth=1.8, alpha=0.9, zorder=8)
    ax3d.scatter(*pos, s=60, color='blue', marker='+', zorder=10, linewidths=2)
    ax3d.text(pos[0] + 0.005, pos[1] + 0.005, pos[2] + 0.005,
              label, fontsize=7.5, color='navy', fontweight='bold')

    if ax_top_3d is not None:
        ax_top_3d.scatter(*pos, s=60, color='blue', marker='+', zorder=10, linewidths=2)
        ax_top_3d.text(pos[0] + 0.005, pos[1] + 0.005, pos[2] + 0.005,
                      label, fontsize=7.5, color='navy', fontweight='bold')

    ax_top.scatter(pos[0], pos[1], s=80, color='blue', marker='+',
                   zorder=10, linewidths=2)
    ax_top.text(pos[0] + 0.005, pos[1] + 0.004,
                label, fontsize=7.5, color='navy', fontweight='bold')


def helical_wrap_xy(ax, center, radius: float, z_start: float, z_end: float,
                     T_start, T_end, branch: int = +1,
                     color: str = "#333", lw: float = 2.5, n_turns: int = 8,
                     pts_per_turn: int = 32, zorder: int = 7) -> None:
    """Draw a helical wrap XY projection from T_start to T_end."""
    cx, cy = float(center[0]), float(center[1])
    a_start = np.arctan2(float(T_start[1]) - cy, float(T_start[0]) - cx)
    a_target = np.arctan2(float(T_end[1]) - cy, float(T_end[0]) - cx)
    n_extra = max(0, int(round(n_turns)) - 1)
    if branch > 0:
        delta = (a_target - a_start) % (2.0 * np.pi)
        if delta < 1e-6:
            delta += 2.0 * np.pi
        a_end = a_start + delta + n_extra * 2.0 * np.pi
    else:
        delta = (a_start - a_target) % (2.0 * np.pi)
        if delta < 1e-6:
            delta += 2.0 * np.pi
        a_end = a_start - delta - n_extra * 2.0 * np.pi
    n_pts = max(2, int(abs(a_end - a_start) / (2.0 * np.pi) * pts_per_turn))
    angles = np.linspace(a_start, a_end, n_pts)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    ax.plot(xs, ys, '-', color=color, linewidth=lw, zorder=zorder,
            solid_capstyle='round', solid_joinstyle='round', antialiased=False)


def helical_wrap_3d(ax3d, center, radius: float, z_start: float, z_end: float,
                     T_start, T_end, branch: int = +1,
                     color: str = "#333", lw: float = 2.5, n_turns: int = 8,
                     pts_per_turn: int = 32, zorder: int = 7) -> None:
    """Draw a 3-D helix around a spool, terminating exactly on T_end."""
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    a_start = np.arctan2(float(T_start[1]) - cy, float(T_start[0]) - cx)
    a_target = np.arctan2(float(T_end[1]) - cy, float(T_end[0]) - cx)
    n_extra = max(0, int(round(n_turns)) - 1)
    if branch > 0:
        delta = (a_target - a_start) % (2.0 * np.pi)
        if delta < 1e-6:
            delta += 2.0 * np.pi
        a_end = a_start + delta + n_extra * 2.0 * np.pi
    else:
        delta = (a_start - a_target) % (2.0 * np.pi)
        if delta < 1e-6:
            delta += 2.0 * np.pi
        a_end = a_start - delta - n_extra * 2.0 * np.pi
    n_pts = max(2, int(abs(a_end - a_start) / (2.0 * np.pi) * pts_per_turn))
    angles = np.linspace(a_start, a_end, n_pts)
    zs = np.linspace(z_start, z_end, n_pts)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    ax3d.plot(xs, ys, zs, '-', color=color, linewidth=lw, zorder=zorder,
              solid_capstyle='round', solid_joinstyle='round')


def wrap_arc_xy(ax, center, radius: float, T_in, T_out,
                color: str, lw: float = 3.0, n: int = 56, zorder: int = 7,
                direction: str = 'auto') -> None:
    """Draw a wrap arc from T_in to T_out around center in the XY plane."""
    cx, cy = float(center[0]), float(center[1])
    a_in = np.arctan2(float(T_in[1]) - cy, float(T_in[0]) - cx)
    a_out = np.arctan2(float(T_out[1]) - cy, float(T_out[0]) - cx)
    if direction == 'ccw':
        if a_out < a_in:
            a_out += 2.0 * np.pi
    elif direction == 'cw':
        if a_out > a_in:
            a_out -= 2.0 * np.pi
    else:
        cross_z = (
            (float(T_in[0]) - cx) * (float(T_out[1]) - cy)
            - (float(T_in[1]) - cy) * (float(T_out[0]) - cx)
        )
        if cross_z > 0:
            if a_out < a_in:
                a_out += 2.0 * np.pi
        else:
            if a_out > a_in:
                a_out -= 2.0 * np.pi
    ang = np.linspace(a_in, a_out, n)
    ax.plot(cx + radius * np.cos(ang), cy + radius * np.sin(ang),
            color=color, linewidth=lw, zorder=zorder, solid_capstyle='round',
            solid_joinstyle='round', antialiased=False)


def wrap_arc_3d(ax3d, center, radius: float, T_in, T_out,
                color: str, lw: float = 3.0, n: int = 56, zorder: int = 7,
                direction: str = 'auto') -> None:
    """Draw a wrap arc in 3D at Z = center[2]."""
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    a_in = np.arctan2(float(T_in[1]) - cy, float(T_in[0]) - cx)
    a_out = np.arctan2(float(T_out[1]) - cy, float(T_out[0]) - cx)
    if direction == 'ccw':
        if a_out < a_in:
            a_out += 2.0 * np.pi
    elif direction == 'cw':
        if a_out > a_in:
            a_out -= 2.0 * np.pi
    else:
        cross_z = (
            (float(T_in[0]) - cx) * (float(T_out[1]) - cy)
            - (float(T_in[1]) - cy) * (float(T_out[0]) - cx)
        )
        if cross_z > 0:
            if a_out < a_in:
                a_out += 2.0 * np.pi
        else:
            if a_out > a_in:
                a_out -= 2.0 * np.pi
    ang = np.linspace(a_in, a_out, n)
    xs, ys = cx + radius * np.cos(ang), cy + radius * np.sin(ang)
    ax3d.plot(xs, ys, np.full_like(xs, cz), color=color, linewidth=lw, zorder=zorder,
              solid_capstyle='round', solid_joinstyle='round')


def seg_xy(ax, P1, P2, color: str, lw: float = 3.0, zorder: int = 7) -> None:
    ax.plot([P1[0], P2[0]], [P1[1], P2[1]], '-',
            color=color, linewidth=lw, zorder=zorder, solid_capstyle='round',
            solid_joinstyle='round', antialiased=False)


def seg_3d(ax3d, P1, P2, color: str, lw: float = 3.0, zorder: int = 7) -> None:
    ax3d.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]], '-',
              color=color, linewidth=lw, zorder=zorder, solid_capstyle='round',
              solid_joinstyle='round')


def load_obj(filename: str) -> tuple:
    """Load vertices and faces from an OBJ file.

    Returns (vertices, faces) where:
      vertices : (N, 3) array of vertex coordinates
      faces    : list of face tuples, each containing vertex indices (0-indexed)
    """
    vertices = []
    faces = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if parts[0] == 'v':
                    vertices.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'f':
                    face = []
                    for vertex_str in parts[1:]:
                        vertex_idx = int(vertex_str.split('/')[0]) - 1  # OBJ uses 1-indexing
                        face.append(vertex_idx)
                    faces.append(tuple(face))
    except Exception as e:
        print(f"Warning: Could not load OBJ file {filename}: {e}")
        return np.array([]), []
    return np.array(vertices) if vertices else np.array([]), faces


def plot_mesh_3d(ax3d, vertices: np.ndarray, faces: list,
                  color: str, alpha: float = 0.25) -> None:
    """Plot a mesh (vertices + faces) in 3D axes using Poly3DCollection."""
    if len(vertices) == 0 or len(faces) == 0:
        return
    
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Build list of 3D polygon vertices
    face_verts = []
    for face in faces:
        if len(face) >= 3:
            # Convert face indices to vertex coordinates
            face_verts.append(vertices[list(face[:3])])  # Use first 3 vertices of polygon
    
    if face_verts:
        # Create and add the polygon collection
        poly_collection = Poly3DCollection(
            face_verts,
            alpha=alpha,
            facecolor=color,
            edgecolor='none',
            linewidth=0
        )
        ax3d.add_collection3d(poly_collection)



# ─── FK helpers ───────────────────────────────────────────────────────────────
def plot_cable_routing(
    lower: CableRouteConfig,
    upper: CableRouteConfig,
    q1: float = 0.0,
    q2: float = 0.0,
    view_elev: float = 28,
    view_azim: float = -55,
) -> plt.Figure:
    """Render triple-panel figure: 3D perspective (left) + XY top-down (center) + XZ side 3D (right).

    Parameters
    ----------
    lower, upper : CableRouteConfig
        Built by :func:`build_lower_cable_config` / :func:`build_upper_cable_config`.
    q1, q2 : float
        Joint angles in radians.
    """
    kin = MHPKinematics(q1, q2)

    fig = plt.figure(figsize=(36, 8.5), facecolor='white')
    fig.suptitle(
        f"MHP Cable Routing — World Frame   "
        f"q1 = {np.rad2deg(q1):.1f}°   q2 = {np.rad2deg(q2):.1f}°",
        fontsize=13, fontweight='bold', y=0.98,
    )

    ax3d = fig.add_subplot(1, 4, 1, projection='3d')
    ax_top = fig.add_subplot(1, 4, 2)
    ax_side_3d = fig.add_subplot(1, 4, 3, projection='3d')
    ax_top_3d = fig.add_subplot(1, 4, 4, projection='3d')

    # ── Arm structure ─────────────────────────────────────────────────────────
    base_origin = np.array([0., 0., 0.])
    J1 = kin.J1
    J2 = kin.J2
    # Lower arm end-effector direction: 0.25 m along lower_arm X axis
    EE = kin.to_world(np.array([0.25, 0., 0.]), "lower_arm")

    arm_pts = np.array([base_origin, J1, J2, EE])
    ax3d.plot(arm_pts[:, 0], arm_pts[:, 1], arm_pts[:, 2],
              'k-', linewidth=2.0, alpha=0.35, label='Arm centreline', zorder=1)
    ax_top.plot(arm_pts[:, 0], arm_pts[:, 1],
                'k-', linewidth=2.0, alpha=0.35, label='Arm centreline', zorder=1)
    ax_side_3d.plot(arm_pts[:, 0], arm_pts[:, 1], arm_pts[:, 2],
                    'k-', linewidth=2.0, alpha=0.35, label='Arm centreline', zorder=1)
    ax_top_3d.plot(arm_pts[:, 0], arm_pts[:, 1], arm_pts[:, 2],
                   'k-', linewidth=2.0, alpha=0.35, label='Arm centreline', zorder=1)

    # World origin marker
    ax3d.scatter(0, 0, 0, s=60, color='black', marker='x', zorder=8)
    ax3d.text(0.004, 0.004, 0.005, 'World', fontsize=7, color='black')
    ax_top.scatter(0, 0, s=60, color='black', marker='x', zorder=8)
    ax_top.text(0.005, 0.004, 'World', fontsize=7, color='black')
    ax_side_3d.scatter(0, 0, 0, s=60, color='black', marker='x', zorder=8)
    ax_side_3d.text(0.004, 0.004, 0.005, 'World', fontsize=7, color='black')
    ax_top_3d.scatter(0, 0, 0, s=60, color='black', marker='x', zorder=8)
    ax_top_3d.text(0.004, 0.004, 0.005, 'World', fontsize=7, color='black')

    # Joint frames
    draw_joint_frame(ax3d, ax_top, J1, kin.R1, "J1 (shoulder)", ax_top_3d=ax_top_3d)
    draw_joint_frame(ax3d, ax_top, J2, kin.R2, "J2 (elbow)", ax_top_3d=ax_top_3d)

    # ── Cables ────────────────────────────────────────────────────────────────
    # Both cables wrap the SAME 40 mm shoulder spool (centre = lower.physical[0]).
    # The upper cable lists a small anchor instead, so capture the real spool here.
    shoulder_spool_w = kin.to_world(lower.physical[0].pos_in_link, lower.physical[0].link)
    shoulder_spool_r = lower.physical[0].diameter_mm * 0.5e-3
    for route in (lower, upper):
        # 1. Physical components (spools, guide pulleys, elbow roller)
        # Get the base path for OBJ files (same directory as this script)
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(script_dir, 'assets')

        for comp in route.physical:
            pw = kin.to_world(comp.pos_in_link, comp.link)
            r_m = comp.diameter_mm * 0.5e-3

            # Try to load and render OBJ mesh
            obj_path = os.path.join(assets_dir, comp.obj_filename)
            if os.path.exists(obj_path):
                vertices, faces = load_obj(obj_path)
                if len(vertices) > 0:
                    # Transform vertices to world frame: apply FK rotation then translate
                    R = kin.link_frame_rotation(comp.link)
                    verts_rotated = vertices @ R.T
                    verts_world = verts_rotated + pw
                    plot_mesh_3d(ax3d, verts_world, faces, color=comp.color, alpha=0.20)
                    plot_mesh_3d(ax_top_3d, verts_world, faces, color=comp.color, alpha=0.20)
                    # Skip the fallback drawing below if mesh loaded successfully
                    continue

            # Fallback: draw simple geometric shapes if OBJ not available
            if comp.role == "elbow_roller":
                disc_3d(ax3d, pw, r_m, comp.color, alpha=0.20)
                disc_3d(ax_top_3d, pw, r_m, comp.color, alpha=0.20)
                circle_xy(ax_top, pw, r_m, comp.color, alpha=0.18, lw=1.5)
                ax_top.text(pw[0] + r_m + 0.003, pw[1],
                            "Elbow roller", fontsize=6.5, color="#555555",
                            va="center")

            elif comp.role == "spool":
                ax3d.scatter(*pw, s=250, color=comp.color, marker='o',
                             alpha=0.85, zorder=6,
                             edgecolors='black', linewidths=0.6)
                ax_top_3d.scatter(*pw, s=250, color=comp.color, marker='o',
                                  alpha=0.85, zorder=6,
                                  edgecolors='black', linewidths=0.6)
                disc_3d(ax3d, pw, r_m, comp.color, alpha=0.30)
                disc_3d(ax_top_3d, pw, r_m, comp.color, alpha=0.30)
                circle_xy(ax_top, pw, r_m, comp.color, alpha=0.40)
                ax_top.text(pw[0], pw[1] + r_m + 0.004,
                            comp.name, fontsize=6.5, color=comp.color,
                            ha='center', fontweight='bold')

            elif comp.role == "guide_pulley":
                ax3d.scatter(*pw, s=55, color=comp.color, marker='o',
                             alpha=0.9, zorder=6,
                             edgecolors='white', linewidths=0.5)
                ax_top_3d.scatter(*pw, s=55, color=comp.color, marker='o',
                                  alpha=0.9, zorder=6,
                                  edgecolors='white', linewidths=0.5)
                circle_xy(ax_top, pw, r_m, comp.color, alpha=0.55, lw=1.2)

        # 2. Cable path — computed once via shared compute_cable_path(), then drawn.
        cp = compute_cable_path(route, kin, cable_loc=route.path[0].cable)
        phys_w       = cp.phys_w
        phys_r       = cp.phys_r
        path_w       = cp.path_w
        T_spool_exit = cp.T_spool_exit
        T_gp1_entry  = cp.T_gp1_entry
        T_gp1_exit   = cp.T_gp1_exit
        T_gp2_entry  = cp.T_gp2_entry
        T_gp2_exit   = cp.T_gp2_exit
        T_gp3_entry  = cp.T_gp3_entry
        T_gp3_exit   = cp.T_gp3_exit
        T_roller_in  = cp.T_roller_in
        T_roller_out = cp.T_roller_out
        helix_branch = cp.helix_branch

        # ── Step 1: Spool helix (path_w[0] → T_spool_exit) ──────────────────
        helical_wrap_xy(ax_top, phys_w[0], phys_r[0],
                         float(path_w[0][2]), float(T_spool_exit[2]),
                         path_w[0], T_spool_exit, branch=helix_branch,
                         color=route.color, n_turns=2, pts_per_turn=48)
        helical_wrap_3d(ax3d, phys_w[0], phys_r[0],
                         float(path_w[0][2]), float(T_spool_exit[2]),
                         path_w[0], T_spool_exit, branch=helix_branch,
                         color=route.color, n_turns=2, pts_per_turn=48)
        helical_wrap_3d(ax_side_3d, phys_w[0], phys_r[0],
                         float(path_w[0][2]), float(T_spool_exit[2]),
                         path_w[0], T_spool_exit, branch=helix_branch,
                         color=route.color, n_turns=2, pts_per_turn=48)
        helical_wrap_3d(ax_top_3d, phys_w[0], phys_r[0],
                         float(path_w[0][2]), float(T_spool_exit[2]),
                         path_w[0], T_spool_exit, branch=helix_branch,
                         color=route.color, n_turns=2, pts_per_turn=48)

        # ── Step 2: spool exit → GP1 entry ──────────────────────────────────
        seg_xy(ax_top,     T_spool_exit, T_gp1_entry, route.color)
        seg_3d(ax3d,       T_spool_exit, T_gp1_entry, route.color)
        seg_3d(ax_side_3d, T_spool_exit, T_gp1_entry, route.color)
        seg_3d(ax_top_3d,  T_spool_exit, T_gp1_entry, route.color)

        # ── Step 3: GP1 arc ──────────────────────────────────────────────────
        wrap_arc_xy(ax_top,     phys_w[1], phys_r[1], T_gp1_entry, T_gp1_exit, route.color)
        wrap_arc_3d(ax3d,       phys_w[1], phys_r[1], T_gp1_entry, T_gp1_exit, route.color)
        wrap_arc_3d(ax_side_3d, phys_w[1], phys_r[1], T_gp1_entry, T_gp1_exit, route.color)
        wrap_arc_3d(ax_top_3d,  phys_w[1], phys_r[1], T_gp1_entry, T_gp1_exit, route.color)

        # ── Step 4: GP1 exit → GP2 entry ────────────────────────────────────
        seg_xy(ax_top,     T_gp1_exit, T_gp2_entry, route.color)
        seg_3d(ax3d,       T_gp1_exit, T_gp2_entry, route.color)
        seg_3d(ax_side_3d, T_gp1_exit, T_gp2_entry, route.color)
        seg_3d(ax_top_3d,  T_gp1_exit, T_gp2_entry, route.color)

        # ── Step 5: GP2 arc ──────────────────────────────────────────────────
        wrap_arc_xy(ax_top,     phys_w[2], phys_r[2], T_gp2_entry, T_gp2_exit, route.color)
        wrap_arc_3d(ax3d,       phys_w[2], phys_r[2], T_gp2_entry, T_gp2_exit, route.color)
        wrap_arc_3d(ax_side_3d, phys_w[2], phys_r[2], T_gp2_entry, T_gp2_exit, route.color)
        wrap_arc_3d(ax_top_3d,  phys_w[2], phys_r[2], T_gp2_entry, T_gp2_exit, route.color)

        # ── Step 6: GP2 exit → GP3 entry ────────────────────────────────────
        seg_xy(ax_top,     T_gp2_exit, T_gp3_entry, route.color)
        seg_3d(ax3d,       T_gp2_exit, T_gp3_entry, route.color)
        seg_3d(ax_side_3d, T_gp2_exit, T_gp3_entry, route.color)
        seg_3d(ax_top_3d,  T_gp2_exit, T_gp3_entry, route.color)

        # ── Step 7: GP3 arc ──────────────────────────────────────────────────
        wrap_arc_xy(ax_top,     phys_w[3], phys_r[3], T_gp3_entry, T_gp3_exit, route.color)
        wrap_arc_3d(ax3d,       phys_w[3], phys_r[3], T_gp3_entry, T_gp3_exit, route.color)
        wrap_arc_3d(ax_side_3d, phys_w[3], phys_r[3], T_gp3_entry, T_gp3_exit, route.color)
        wrap_arc_3d(ax_top_3d,  phys_w[3], phys_r[3], T_gp3_entry, T_gp3_exit, route.color)

        # ── Step 8: GP3 exit → roller entry ─────────────────────────────────
        seg_xy(ax_top,     T_gp3_exit, T_roller_in, route.color)
        seg_3d(ax3d,       T_gp3_exit, T_roller_in, route.color)
        seg_3d(ax_side_3d, T_gp3_exit, T_roller_in, route.color)
        seg_3d(ax_top_3d,  T_gp3_exit, T_roller_in, route.color)

        # ── Step 9: Roller arc (CCW for lower cable, CW for upper cable) ─────
        _roller_dir = 'ccw' if cp.route.path[0].cable == 'lower' else 'cw'
        wrap_arc_xy(ax_top,     phys_w[4], phys_r[4], T_roller_in, T_roller_out, route.color, direction=_roller_dir)
        wrap_arc_3d(ax3d,       phys_w[4], phys_r[4], T_roller_in, T_roller_out, route.color, direction=_roller_dir)
        wrap_arc_3d(ax_side_3d, phys_w[4], phys_r[4], T_roller_in, T_roller_out, route.color, direction=_roller_dir)
        wrap_arc_3d(ax_top_3d,  phys_w[4], phys_r[4], T_roller_in, T_roller_out, route.color, direction=_roller_dir)

        # ── Step 10: Roller exit → cable endpoint (direct tangent) ───────────
        seg_xy(ax_top,     T_roller_out, phys_w[5], route.color)
        seg_3d(ax3d,       T_roller_out, phys_w[5], route.color)
        seg_3d(ax_side_3d, T_roller_out, phys_w[5], route.color)
        seg_3d(ax_top_3d,  T_roller_out, phys_w[5], route.color)

        # Proxy artists for all axes
        ax_top.plot(    [], [], '-', color=route.color, linewidth=1.9, label=route.name)
        ax3d.plot(      [], [], '-', color=route.color, linewidth=1.9, label=route.name)
        ax_side_3d.plot([], [], '-', color=route.color, linewidth=1.9, label=route.name)
        ax_top_3d.plot( [], [], '-', color=route.color, linewidth=1.9, label=route.name)

        # Labels on top view
        for pt, lbl, dy_sign in [
            (path_w[0],    "Spool start",    +1),
            (T_spool_exit, "Spool exit",      +1),
            (path_w[-2],   "Lower arm entry", -1),
            (path_w[-1],   "Lower arm exit",  -1),
        ]:
            dy = 0.006 * dy_sign
            ax_top.annotate(
                lbl,
                xy=(pt[0], pt[1]),
                xytext=(pt[0] + 0.012, pt[1] + dy),
                fontsize=6.0, color=route.color,
                arrowprops=dict(arrowstyle='->', color=route.color, lw=0.6),
                zorder=9,
            )




    # ── 3D axes ───────────────────────────────────────────────────────────────
    ax3d.set_xlabel('X [m]', fontsize=9, labelpad=6)
    ax3d.set_ylabel('Y [m]', fontsize=9, labelpad=6)
    ax3d.set_zlabel('Z [m]', fontsize=9, labelpad=6)
    ax3d.set_title('3-D View\n(Frames: X=red  Y=green  Z=blue)', fontsize=9)
    ax3d.view_init(elev=view_elev, azim=view_azim)
    ax3d.legend(loc='upper left', fontsize=7, framealpha=0.7)
    ax3d.tick_params(labelsize=7)

    # ── Top-view axes ─────────────────────────────────────────────────────────
    ax_top.set_xlabel('X [m]', fontsize=9)
    ax_top.set_ylabel('Y [m]', fontsize=9)
    ax_top.set_title('Top View (XY plane)', fontsize=10)
    ax_top.set_aspect('equal')
    ax_top.grid(True, alpha=0.3, linestyle='--')
    ax_top.legend(loc='lower right', fontsize=7, framealpha=0.8)
    ax_top.tick_params(labelsize=8)

    # ── Side 3D view axes (XZ plane) ──────────────────────────────────────────
    ax_side_3d.set_xlabel('X [m]', fontsize=9, labelpad=6)
    ax_side_3d.set_ylabel('Y [m]', fontsize=9, labelpad=6)
    ax_side_3d.set_zlabel('Z [m]', fontsize=9, labelpad=6)
    ax_side_3d.set_title('XZ Side View\n(3D perspective)', fontsize=9)
    ax_side_3d.view_init(elev=0, azim=90)  # XZ plane view
    ax_side_3d.legend(loc='upper left', fontsize=7, framealpha=0.7)
    ax_side_3d.tick_params(labelsize=7)

    # Text box summarising FK state
    info = (
        f"q1 = {np.rad2deg(q1):.1f}°  q2 = {np.rad2deg(q2):.1f}°\n"
        f"J1 = ({J1[0]:.3f}, {J1[1]:.3f}, {J1[2]:.3f}) m\n"
        f"J2 = ({J2[0]:.3f}, {J2[1]:.3f}, {J2[2]:.3f}) m"
    )
    ax_side_3d.text2D(
        0.02, 0.05, info,
        transform=ax_side_3d.transAxes,
        fontsize=6, color='#333333',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85, ec='#cccccc'),
        verticalalignment='bottom',
    )

    # ── Top-down 3D view with meshes ───────────────────────────────────────────
    ax_top_3d.set_xlabel('X [m]', fontsize=9, labelpad=6)
    ax_top_3d.set_ylabel('Y [m]', fontsize=9, labelpad=6)
    ax_top_3d.set_zlabel('Z [m]', fontsize=9, labelpad=6)
    ax_top_3d.set_title('XY Top-Down View\n(with OBJ meshes)', fontsize=9)
    ax_top_3d.view_init(elev=85, azim=0)  # Nearly top-down view
    ax_top_3d.legend(loc='upper left', fontsize=7, framealpha=0.7)
    ax_top_3d.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

