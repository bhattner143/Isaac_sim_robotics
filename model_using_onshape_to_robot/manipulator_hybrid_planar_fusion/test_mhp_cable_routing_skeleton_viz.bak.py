#!/usr/bin/env python3
"""
test_mhp_cable_routing_viz.py
─────────────────────────────
Config dataclass objects describing the full lower and upper cable routing
for the manipulator_hybrid_planar_fusion (MHP) robot.

Includes a dual-panel matplotlib figure:
  Left  — 3-D perspective view
  Right — XY top-down view

All positions are expressed in world frame via FK at the given (q1, q2).
No Drake / Isaac Sim required — pure numpy + matplotlib.

Usage:
    cd /Volumes/Data/Isaac_sim_robotics
    conda activate pydrake   # any env with numpy + matplotlib
    python model_using_onshape_to_robot/manipulator_hybrid_planar_fusion/test_mhp_cable_routing_viz.py
    python ...                   --q1 30  --q2 -20
    python ...  --save out.png
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)

try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False

# ─── Drawing helpers ──────────────────────────────────────────────────────────

def _disc_3d(ax, center: np.ndarray, radius: float, color: str,
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


def _draw_joint_frame(ax3d, ax_top_3d, pos: np.ndarray, R: np.ndarray, label: str, scale: float = 0.025):
    """Draw XYZ frame trident at *pos* with rotation *R* in both 3D axes."""
    colors = ('red', 'green', 'blue')
    for i, c in enumerate(colors):
        end = pos + scale * R[:, i]
        ax3d.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]],
                  color=c, linewidth=1.8, alpha=0.9, zorder=8)
        ax_top_3d.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]],
                      color=c, linewidth=1.8, alpha=0.9, zorder=8)
    ax3d.scatter(*pos, s=60, color='blue', marker='+', zorder=10, linewidths=2)
    ax3d.text(pos[0] + 0.005, pos[1] + 0.005, pos[2] + 0.005,
              label, fontsize=7.5, color='navy', fontweight='bold')
    ax_top_3d.scatter(*pos, s=60, color='blue', marker='+', zorder=10, linewidths=2)
    ax_top_3d.text(pos[0] + 0.005, pos[1] + 0.005, pos[2] + 0.005,
                  label, fontsize=7.5, color='navy', fontweight='bold')


# ─── Tangent & wrap-arc helpers (pure numpy, no Drake) ───────────────────────

def _compute_tangent(
    c1, r1: float,
    c2, r2: float,
    branch: int = +1 or -1,
    kind: str = "external" or "internal",
) -> tuple:
    """One external/internal tangent between circles (c1, r1) and (c2, r2).

    Works in the XY plane; Z of each centre is preserved in the output points.

    Parameters
    ----------
    branch : +1 or -1  — selects one of the two parallel tangent branches.
    kind   : 'external' (both circles on same side) or 'internal' (crosses between).

    Returns (T1, T2) — tangent contact points on circle 1 and circle 2.
    """
    c1 = np.asarray(c1, float)
    c2 = np.asarray(c2, float)
    p1, p2 = c1[:2], c2[:2]
    d = p2 - p1
    D = np.linalg.norm(d)
    if D < 1e-9:
        return c1.copy(), c2.copy()
    d_hat = d / D
    perp  = np.array([-d_hat[1], d_hat[0]])
    branch = 1 if branch >= 0 else -1

    if kind == "internal":
        cos_a = (r1 + r2) / D
        sin_a = np.sqrt(max(0.0, 1.0 - cos_a ** 2))
        n  = cos_a * d_hat + branch * sin_a * perp
        T1 = np.array([p1[0] + r1 * n[0], p1[1] + r1 * n[1], c1[2]])
        T2 = np.array([p2[0] - r2 * n[0], p2[1] - r2 * n[1], c2[2]])
    else:  # external
        cos_a = (r1 - r2) / D
        sin_a = np.sqrt(max(0.0, 1.0 - cos_a ** 2))
        n  = cos_a * d_hat + branch * sin_a * perp
        T1 = np.array([p1[0] + r1 * n[0], p1[1] + r1 * n[1], c1[2]])
        T2 = np.array([p2[0] + r2 * n[0], p2[1] + r2 * n[1], c2[2]])
    return T1, T2





def _helical_wrap_3d(ax3d, center, radius: float, z_start: float, z_end: float,
                     T_start, T_end, branch: int = +1,
                     color: str = "#333", lw: float = 2.5, n_turns: int = 8,
                     pts_per_turn: int = 32, zorder: int = 7) -> None:
    """Draw a 3-D helix around a spool, terminating exactly on T_end."""
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    a_start  = np.arctan2(float(T_start[1]) - cy, float(T_start[0]) - cx)
    a_target = np.arctan2(float(T_end[1])   - cy, float(T_end[0])   - cx)
    n_extra  = max(0, int(round(n_turns)) - 1)

    if branch > 0:                                # CCW
        delta = (a_target - a_start) % (2.0 * np.pi)
        if delta < 1e-6:
            delta += 2.0 * np.pi
        a_end = a_start + delta + n_extra * 2.0 * np.pi
    else:                                         # CW
        delta = (a_start - a_target) % (2.0 * np.pi)
        if delta < 1e-6:
            delta += 2.0 * np.pi
        a_end = a_start - delta - n_extra * 2.0 * np.pi

    n_pts  = max(2, int(abs(a_end - a_start) / (2.0 * np.pi) * pts_per_turn))
    angles = np.linspace(a_start, a_end, n_pts)
    zs = np.linspace(z_start, z_end, n_pts)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)

    ax3d.plot(xs, ys, zs, '-', color=color, linewidth=lw, zorder=zorder,
              solid_capstyle='round', solid_joinstyle='round')





def _wrap_arc_3d(ax3d, center, radius: float, T_in, T_out,
                 color: str, lw: float = 3.0, n: int = 56, zorder: int = 7,
                 direction: str = 'auto') -> None:
    """Draw a wrap arc in 3D — arc lives in the XY plane at Z = center[2].

    direction : 'auto' (cross-product), 'ccw' (force anticlockwise), 'cw' (force clockwise).
    """
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    a_in  = np.arctan2(float(T_in[1])  - cy, float(T_in[0])  - cx)
    a_out = np.arctan2(float(T_out[1]) - cy, float(T_out[0]) - cx)
    if direction == 'ccw':
        if a_out < a_in:
            a_out += 2.0 * np.pi
    elif direction == 'cw':
        if a_out > a_in:
            a_out -= 2.0 * np.pi
    else:
        cross_z = (
            (float(T_in[0])  - cx) * (float(T_out[1]) - cy)
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





def _seg_3d(ax3d, P1, P2, color: str, lw: float = 3.0, zorder: int = 7) -> None:
    """Draw a straight cable segment in 3D."""
    ax3d.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]], '-',
              color=color, linewidth=lw, zorder=zorder, solid_capstyle='round',
              solid_joinstyle='round')


# ─── Multi-axis drawing consolidators (for 2 remaining axes: ax3d, ax_top_3d) ────

def _draw_segment_multi_axes(ax3d, ax_top_3d, P1, P2, color: str, lw: float = 3.0) -> None:
    """Draw a straight cable segment on both 3D axes."""
    _seg_3d(ax3d,      P1, P2, color, lw)
    _seg_3d(ax_top_3d, P1, P2, color, lw)


def _draw_arc_multi_axes(ax3d, ax_top_3d, center, radius: float, T_in, T_out, 
                          color: str, direction: str = 'auto') -> None:
    """Draw a wrap arc on both 3D axes."""
    _wrap_arc_3d(ax3d,      center, radius, T_in, T_out, color, direction=direction)
    _wrap_arc_3d(ax_top_3d, center, radius, T_in, T_out, color, direction=direction)


def _draw_helix_multi_axes(ax3d, ax_top_3d, center, radius: float, 
                            z_start: float, z_end: float,
                            P_start, P_exit, color: str, branch: int = -1,
                            n_turns: int = 3, pts_per_turn: int = 48) -> None:
    """Draw a helical wrap on both 3D axes."""
    _helical_wrap_3d(ax3d,      center, radius, z_start, z_end, P_start, P_exit,
                     branch=branch, color=color, n_turns=n_turns, pts_per_turn=pts_per_turn)
    _helical_wrap_3d(ax_top_3d, center, radius, z_start, z_end, P_start, P_exit,
                     branch=branch, color=color, n_turns=n_turns, pts_per_turn=pts_per_turn)


def _draw_arm_skeleton_multi_axes(ax3d, ax_top_3d, arm_pts: np.ndarray) -> None:
    """Draw arm skeleton on both 3D axes."""
    ax3d.plot(arm_pts[:, 0], arm_pts[:, 1], arm_pts[:, 2],
              'k-', linewidth=2.0, alpha=0.35, label='Arm centreline', zorder=1)
    ax_top_3d.plot(arm_pts[:, 0], arm_pts[:, 1], arm_pts[:, 2],
                   'k-', linewidth=2.0, alpha=0.35, label='Arm centreline', zorder=1)


def _draw_world_origin_multi_axes(ax3d, ax_top_3d) -> None:
    """Draw world origin marker on both 3D axes."""
    ax3d.scatter(0, 0, 0, s=60, color='black', marker='x', zorder=8)
    ax3d.text(0.004, 0.004, 0.005, 'World', fontsize=7, color='black')
    
    ax_top_3d.scatter(0, 0, 0, s=60, color='black', marker='x', zorder=8)
    ax_top_3d.text(0.004, 0.004, 0.005, 'World', fontsize=7, color='black')


def _add_legend_proxy_multi_axes(ax3d, ax_top_3d, color: str, label: str) -> None:
    """Add legend proxy artist to both 3D axes."""
    ax3d.plot(      [], [], '-', color=color, linewidth=1.9, label=label)
    ax_top_3d.plot( [], [], '-', color=color, linewidth=1.9, label=label)


# ─── Meshcat visualization ────────────────────────────────────────────────────

def _helix_pts_3d(center, radius: float, z_start: float, z_end: float,
                  T_start, T_end, branch: int = +1,
                  n_turns: int = 2, pts_per_turn: int = 48) -> np.ndarray:
    """Return (N,3) helix points — same math as _helical_wrap_3d but returns array."""
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    a_start  = np.arctan2(float(T_start[1]) - cy, float(T_start[0]) - cx)
    a_target = np.arctan2(float(T_end[1])   - cy, float(T_end[0])   - cx)
    n_extra  = max(0, int(round(n_turns)) - 1)
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
    n_pts  = max(2, int(abs(a_end - a_start) / (2.0 * np.pi) * pts_per_turn))
    angles = np.linspace(a_start, a_end, n_pts)
    zs     = np.linspace(z_start, z_end, n_pts)
    xs     = cx + radius * np.cos(angles)
    ys     = cy + radius * np.sin(angles)
    return np.column_stack([xs, ys, zs]).astype(np.float32)


def _arc_pts_3d(center, radius: float, T_in, T_out,
                n: int = 56, direction: str = 'auto') -> np.ndarray:
    """Return (N,3) arc points in the XY plane at Z=center[2] — same math as _wrap_arc_3d.

    direction : 'auto' (cross-product), 'ccw' (force anticlockwise), 'cw' (force clockwise).
    """
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    a_in  = np.arctan2(float(T_in[1])  - cy, float(T_in[0])  - cx)
    a_out = np.arctan2(float(T_out[1]) - cy, float(T_out[0]) - cx)
    if direction == 'ccw':
        if a_out < a_in:
            a_out += 2.0 * np.pi
    elif direction == 'cw':
        if a_out > a_in:
            a_out -= 2.0 * np.pi
    else:  # auto
        cross_z = ((float(T_in[0]) - cx) * (float(T_out[1]) - cy)
                   - (float(T_in[1]) - cy) * (float(T_out[0]) - cx))
        if cross_z > 0:
            if a_out < a_in:
                a_out += 2.0 * np.pi
        else:
            if a_out > a_in:
                a_out -= 2.0 * np.pi
    ang = np.linspace(a_in, a_out, n)
    xs  = cx + radius * np.cos(ang)
    ys  = cy + radius * np.sin(ang)
    return np.column_stack([xs, ys, np.full(n, cz)]).astype(np.float32)

# ─── OBJ loading utilities ─────────────────────────────────────────────────────

def _load_obj(filename: str) -> tuple:
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


def _plot_mesh_3d(ax3d, vertices: np.ndarray, faces: list,
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

def _Rz(theta: float) -> np.ndarray:
    """3×3 rotation matrix about the Z axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class JointFrameConfig:
    """URDF revolute joint — position expressed in the *parent* link frame."""
    name: str
    parent_link: str
    child_link: str
    xyz: np.ndarray                                           # (3,) [m]
    axis: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1.]))

    def __post_init__(self):
        self.xyz  = np.asarray(self.xyz,  float)
        self.axis = np.asarray(self.axis, float)


@dataclass
class LinkConfig:
    """URDF link with mass / inertia properties."""
    name: str
    mass_kg: float
    com_xyz: np.ndarray   # (3,) CoM in link frame [m]
    ixx: float
    iyy: float
    izz: float

    def __post_init__(self):
        self.com_xyz = np.asarray(self.com_xyz, float)


@dataclass
class CableComponent:
    """One element in the cable routing path (spool / guide pulley / roller / ball marker)."""
    name: str
    obj_filename: str
    link: str                  # 'base_link' | 'upper_arm' | 'lower_arm'
    pos_in_link: np.ndarray    # (3,) routing/geometric centre in link frame [m]
    diameter_mm: float         # outer diameter [mm]
    color: str                 # matplotlib colour
    role: str                  # 'spool' | 'guide_pulley' | 'elbow_roller' | 'ball_marker' | 'cable_anchor'
    cable: str                 # 'lower' | 'upper' | 'shared'
    note: str = ""
    # Optional OBJ mesh placement override (from URDF <visual> <origin>).
    # When set, used instead of pos_in_link for Meshcat OBJ rendering.
    visual_pos_in_link: object = None   # (3,) URDF visual origin xyz [m] or None
    visual_rpy: object = None           # (3,) URDF visual rpy [rad] or None

    def __post_init__(self):
        self.pos_in_link = np.asarray(self.pos_in_link, float)
        if self.visual_pos_in_link is not None:
            self.visual_pos_in_link = np.asarray(self.visual_pos_in_link, float)
        if self.visual_rpy is not None:
            self.visual_rpy = np.asarray(self.visual_rpy, float)





# ─── Robot constants ──────────────────────────────────────────────────────────

JOINTS: List[JointFrameConfig] = [
    JointFrameConfig(
        name="jt_upper_base",
        parent_link="base_link",
        child_link="upper_arm",
        xyz=[0.045, -1.90933e-09, 0.1249],
    ),
    JointFrameConfig(
        name="jt_lower_upper",
        parent_link="upper_arm",
        child_link="lower_arm",
        xyz=[0.4, 0.0470711, 0.0141],
    ),
]

LINKS: List[LinkConfig] = [
    LinkConfig(
        name="base_link_aka_shoulder_transmission",
        mass_kg=2.0,
        com_xyz=[0.08, 0.0, 0.13],
        ixx=0.0170, iyy=0.0170, izz=0.0050,
    ),
    LinkConfig(
        name="upper_arm",
        mass_kg=0.8,
        com_xyz=[0.325, 0.015, 0.065],
        ixx=0.0006, iyy=0.0137, izz=0.0137,
    ),
    LinkConfig(
        name="lower_arm",
        mass_kg=0.4,
        com_xyz=[0.125, 0.0, 0.01],
        ixx=0.0004, iyy=0.0021, izz=0.0023,
    ),
]


# ─── FK ───────────────────────────────────────────────────────────────────────

class MHPKinematics:
    """Forward kinematics for the 2-DOF MHP robot.

    Joints both rotate about their local Z axis.
    All positions in world frame.

    Parameters
    ----------
    q1 : float  — jt_upper_base angle  [rad]
    q2 : float  — jt_lower_upper angle [rad]
    """

    # fixed from URDF
    _J1_IN_WORLD = np.array([0.045, 0.,       0.1249])
    _J2_IN_UA    = np.array([0.4,   0.0470711, 0.0141])

    def __init__(self, q1: float = 0.0, q2: float = 0.0):
        self.q1 = q1
        self.q2 = q2
        R1 = _Rz(q1)
        R2 = _Rz(q1 + q2)
        J1 = self._J1_IN_WORLD.copy()
        J2 = J1 + R1 @ self._J2_IN_UA

        self.R1 = R1
        self.R2 = R2
        self.J1 = J1
        self.J2 = J2

    def to_world(self, pos_in_link: np.ndarray, link: str) -> np.ndarray:
        """Transform *pos_in_link* (expressed in *link* frame) to world frame."""
        p = np.asarray(pos_in_link, float)
        if link in ("base_link", "world"):
            return p.copy()
        if link == "upper_arm":
            return self.J1 + self.R1 @ p
        if link == "lower_arm":
            return self.J2 + self.R2 @ p
        raise ValueError(f"Unknown link: {link!r}")

    def link_frame_rotation(self, link: str) -> np.ndarray:
        """Get the 3×3 rotation matrix from *link* frame to world frame."""
        if link in ("base_link", "world"):
            return np.eye(3)
        if link == "upper_arm":
            return self.R1
        if link == "lower_arm":
            return self.R2
        raise ValueError(f"Unknown link: {link!r}")



@dataclass
class CableRouteConfig:
    """Full tendon cable description: physical components + ordered ball markers.

    branch : +1 → external tangent on the +Y side (lower cable),
             -1 → external tangent on the −Y side (upper cable).
    """
    name: str
    color: str
    physical: List[CableComponent]   # spools, guide pulleys, elbow roller
    path: List[CableComponent]       # ordered transition markers (cable path)
    # Cable routing geometry 
    branch_sign_seq: List[int]      # tangent branch directions [+1/-1, ...]
    kind_seq: List[str]             # tangent types ['external'/'internal', ...]
    elbow_roller_arc_dir: str       # 'auto'/'cw'/'ccw' for wrap arc around elbow roller
    n_spool_turns: int = 2          # number of cable wraps around the drive spool
    spool_pitch_mm: float = 3.0     # axial advance per wrap [mm] (≈ cable diameter)


# ─── Shared cable path computation ───────────────────────────────────────────

@dataclass
class CablePathData:
    """All computed cable path geometry for one route in the world frame.

    Single source of truth consumed by both the matplotlib renderer
    (``plot_cable_routing``) and the Meshcat renderer
    (``visualize_cable_routing_meshcat``).  Computed once per route per
    pose by :func:`compute_cable_path`.
    """
    route:         "CableRouteConfig"   # back-reference to config
    # FK-resolved positions
    path_w:        list                 # world-frame path anchor points
    phys_w:        list                 # world-frame physical component centres
    phys_r:        list                 # component radii [m]
    # Tangent contact points (world frame, all (3,) arrays)
    T_spool_exit:  np.ndarray
    T_gp1_entry:   np.ndarray
    T_gp1_exit:    np.ndarray
    T_gp2_entry:   np.ndarray
    T_gp2_exit:    np.ndarray
    T_gp3_entry:   np.ndarray
    T_gp3_exit:    np.ndarray
    T_roller_in:   np.ndarray
    T_roller_out:  np.ndarray
    # Spool helix direction (±1)
    helix_branch:  int
    # Ordered polyline pieces [(N,3) arrays] — 10 pieces matching matplotlib steps 1-11
    pieces:        list

# ─── Lower cable config (shoulder, mostly +Y) ─────────────────────────────────

def build_lower_cable_config() -> CableRouteConfig:
    """Config objects for the lower cable (actuates jt_upper_base / shoulder joint).

    Routing order:
      Elbow spool → guide pulley 1 → guide pulley 2 → guide pulley 3 → elbow roller
    Cable path ball markers (transition points) span from spool start to lower arm exit.
    """
    _C_LINE  = "#E37629"   # cable line — orange
    _C_SPOOL = "#D45F00"   # drive spool
    _C_GUIDE = "#333333"   # guide pulleys
    _C_ROLLER= "#888888"   # elbow roller
    _C_BALL  = "#87CEEB"   # sky-blue ball markers
    _C_LO_ARM= "#FFA500"   # lower-arm entry/exit

    # ─────────────────────────────────────────────────────────────────────────────
    # SPOOL POSITION CALCULATION (in upper_arm frame)
    #
    # Two different spool positions exist in the system:
    #
    # (1) URDF PHYSICAL POSITION [0.225, 0, 0.1268] (m)
    #     → From: manipulator_hybrid_planar_fusion_obj.urdf line ~391
    #     → Origin: Onshape CAD model geometry (mhp_arm_00_elbow_spool_v2.obj location)
    #     → Purpose: 3D visualization in Meshcat/Drake
    #     → Represents: Actual spool drum location inside the shoulder transmission housing
    #     → Note: High Z (0.1268 m) because spool sits near top of housing
    #
    # (2) CABLE ROUTING POSITION [-0.0795, 0, 0.0155] (m) ← USED HERE
    #
    #     EXTRACTION PROCESS (NOT CALCULATED):
    #     ─────────────────────────────────────
    #     Source: URDF file line 664, ball_cable_spool_upper_arm_start marker
    #
    #     URDF snippet:
    #       <!-- Part ball_cable_spool_upper_arm_start_2 -->
    #       <visual>
    #         <origin xyz="-0.0795 1.76602e-13 0.0155" rpy="..."/>
    #         <geometry>
    #           <mesh filename="package://assets/ball_cable_spool_upper_arm_start.obj"/>
    #         </geometry>
    #       </visual>
    #
    #     Value breakdown:
    #       X = -0.0795 m  ← backward from shoulder joint (negative direction)
    #       Y = 1.76602e-13 m ≈ 0  ← numerical precision artifact (Onshape export)
    #       Z = 0.0155 m   ← at lower arm channel height (18.4 mm groove level)
    #
    #     This position marks where the cable EXITS the housing into the arm channel.
    #     It was placed by the Onshape CAD designer as a reference point for cable routing.
    #
    # Why different positions?
    #   The physical spool (0.225, 0, 0.1268) is INSIDE the transmission housing.
    #   For cable routing visualization, we show the CABLE EXIT POINT (cable emerges from
    #   housing at -0.0795, 0, 0.0155) where the cable actually starts routing through
    #   pulleys. This makes the 2D visualization show the cable path from its actual
    #   routing starting point, not from the deep spool drum location.
    #
    # Coordinate system (upper_arm frame):
    #   Origin: jt_upper_base joint center
    #   X: along upper arm length (+ toward J2)
    #   Y: perpendicular (+ in original arm orientation)
    #   Z: vertical (+ upward)
    # ─────────────────────────────────────────────────────────────────────────────

    physical = [
        CableComponent(
            name="Cable anchor (clamp on spool rim)",
            obj_filename="ball_cable_spool_upper_arm_start.obj",
            link="upper_arm",
            pos_in_link=[-0.0795, -0.020, 0.0155],  # on spool rim at -Y, r=20mm from centre
            diameter_mm=0.0,
            color=_C_SPOOL,
            role="cable_anchor",
            cable="lower",
            note="Physical clamp point where lower cable is fixed to spool drum, -Y side",
        ),
        CableComponent(
            name="Shoulder drive spool",
            obj_filename="mhp_arm_00_elbow_spool_v2__2.obj",
            link="upper_arm",
            pos_in_link=[-0.0795, 1.76602e-13, 0.0155],
            diameter_mm=40.0,
            color=_C_SPOOL,
            role="spool",
            cable="lower",
            note="Drive spool drum — lower cable groove, Z=15.5 mm",
            visual_pos_in_link=[0.225, 0.0, 0.1268],
            visual_rpy=[0.0, 0.0, 3.14159],
        ),
        CableComponent(
            name="Guide pulley 1 (+Y)",
            obj_filename="steel_v_groove_guide_pulley___4x13x6mm.obj",
            link="upper_arm",
            pos_in_link=[-0.0409243, 0.03445, 0.0325],
            diameter_mm=10.0,
            color=_C_GUIDE,
            role="guide_pulley",
            cable="lower",
            note="First guide pulley, +Y side",
        ),
        CableComponent(
            name="Guide pulley 2 (+Y)",
            obj_filename="steel_v_groove_guide_pulley___4x13x6mm.obj",
            link="upper_arm",
            pos_in_link=[0.33, 0.03445, 0.0324],
            diameter_mm=10.0,
            color=_C_GUIDE,
            role="guide_pulley",
            cable="lower",
            note="Mid-span guide pulley, +Y side",
        ),
        CableComponent(
            name="Guide pulley 3 (+Y)",
            obj_filename="steel_v_groove_guide_pulley___4x13x6mm.obj",
            link="upper_arm",
            pos_in_link=[0.353129, 0.0165902, 0.0324],
            diameter_mm=10.0,
            color=_C_GUIDE,
            role="guide_pulley",
            cable="lower",
            note="Pre-elbow guide pulley",
        ),
        CableComponent(
            name="Elbow roller groove (+Y)",
            obj_filename="mhp_arm_00_elbow_roller_v1.obj",
            link="lower_arm",
            pos_in_link=[6.10623e-16, 6.93889e-18, 0.0184],#[6.10623e-16, 6.93889e-18, 0.0259],
            diameter_mm=78.8,
            color=_C_ROLLER,
            role="elbow_roller",
            cable="shared",
            note="Driven elbow roller, OD=85.44 mm (grove dia 78.8 mm)",
        ),
         CableComponent(
            name="Cable end point (leaves the roller and enters lower arm)",
            obj_filename="ball_cable_mount_lower_arm_enter.obj",
            link="lower_arm",
            pos_in_link=[0.0333444, -0.0372498, 0.0183897],
            diameter_mm=5.0, color=_C_LO_ARM, role="ball_marker", cable="lower",
            note="End point where cable enters the lower arm mount Z=18.4 mm (lower groove)",
        ),
    ]

    path = [
        CableComponent(
            name="Cable start point (spool start)",
            obj_filename="ball_cable_spool_upper_arm_start.obj",
            link="upper_arm",
            pos_in_link=[-0.0795, -0.020, 0.0155],  # on spool rim at -Y (anchor clamp, opp. to +Y pulleys)
            diameter_mm=5.0, color=_C_SPOOL, role="ball_marker", cable="lower",
            note="Cable clamp on spool rim — -Y side, angle=270°, r=20mm",
        ),
        CableComponent(
            name="Spool exit",
            obj_filename="ball_cable_spool_upper_arm_exit.obj",
            link="upper_arm",
            pos_in_link=[-0.0718787, 0.0159976, 0.0324999],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        ),
        CableComponent(
            name="Pulley A3",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[-0.0438212, 0.0387601, 0.0324999],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        ),
        CableComponent(
            name="Pulley A4",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[-0.0403245, 0.0400001, 0.0324999],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        ),
        CableComponent(
            name="Pulley A5",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[0.3306, 0.0400001, 0.0324999],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        ),
        CableComponent(
            name="Pulley A6",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[0.335406, 0.0372251, 0.0324999],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        ),
        CableComponent(
            name="Pulley A7",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[0.348922, 0.0138153, 0.0324999],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        ),
        CableComponent(
            name="Pulley A8",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[0.353245, 0.0110614, 0.0324999],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        ),
        CableComponent(
            name="Elbow roller enter",
            obj_filename="ball_cable_elbow_roller_enter.obj",
            link="upper_arm",
            pos_in_link=[0.397114, 0.00722337, 0.0324999],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        ),
        CableComponent(
            name="Cable end point (leaves the roller and enters lower arm)",
            obj_filename="ball_cable_mount_lower_arm_enter.obj",
            link="lower_arm",
            pos_in_link=[0.0333444, -0.0372498, 0.0183897],
            diameter_mm=5.0, color=_C_LO_ARM, role="ball_marker", cable="lower",
            note="Z=18.4 mm (lower groove)",
        ),
        # CableComponent(
        #     name="Lower arm exit",
        #     obj_filename="ball_cable_mount_lower_arm_exit.obj",
        #     link="lower_arm",
        #     pos_in_link=[0.0613398, -0.012071, 0.0184],
        #     diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="lower",
        #     note="Z=18.4 mm (lower groove)",
        # ),
    ]

    return CableRouteConfig(
        name="Lower Cable — shoulder (+Y)",
        color=_C_LINE,
        physical=physical,
        path=path,
        branch_sign_seq=[+1, +1, +1, -1, -1],
        kind_seq=['external', 'external', 'internal', 'external', 'external'],
        elbow_roller_arc_dir='ccw',
        n_spool_turns=3,
    )


# ─── Upper cable config (elbow, mostly -Y) ────────────────────────────────────

def build_upper_cable_config() -> CableRouteConfig:
    """Config objects for the upper cable (actuates jt_lower_upper / elbow joint).

    Routing order:
      Spool anchor → guide pulley 1 → guide pulley 2 → guide pulley 3 → elbow roller
    Cable path ball markers span from spool start to lower arm exit.
    """
    _C_LINE  = "#8B34C4"   # cable line — purple
    _C_GUIDE = "#333333"   # guide pulleys
    _C_ROLLER= "#888888"   # elbow roller
    _C_BALL  = "#87CEEB"   # sky-blue ball markers
    _C_LO_ARM= "#FFA500"   # lower-arm entry/exit

    physical = [
        CableComponent(
            name="Cable anchor (clamp on spool rim)",
            obj_filename="ball_cable_spool_upper_arm_start.obj",
            link="upper_arm",
            pos_in_link=[-0.0795, 0.020, 0.0645],  # on spool rim at +Y, r=20mm from centre
            diameter_mm=0.0,
            color=_C_LINE,
            role="cable_anchor",
            cable="upper",
            note="Physical clamp point where upper cable is fixed to spool drum, +Y side",
        ),
        CableComponent(
            name="Shoulder drive spool",
            obj_filename="mhp_arm_00_elbow_spool_v2.obj",
            link="upper_arm",
            pos_in_link=[-0.0795, 1.76602e-13, 0.0645],
            diameter_mm=40.0,
            color=_C_LINE,
            role="spool",
            cable="upper",
            note="Drive spool drum — upper cable groove, Z=64.5 mm",
            visual_pos_in_link=[-0.0795, 1.76602e-13, 0.0645],
            visual_rpy=[0.0, 0.0, 3.14159],
        ),
        CableComponent(
            name="Guide pulley 1 (-Y)",
            obj_filename="steel_v_groove_guide_pulley___4x13x6mm.obj",
            link="upper_arm",
            pos_in_link=[-0.0409243, -0.03445, 0.0475],
            diameter_mm=10.0,
            color=_C_GUIDE,
            role="guide_pulley",
            cable="upper",
            note="First guide pulley, -Y side",
        ),
        CableComponent(
            name="Guide pulley 2 (-Y)",
            obj_filename="steel_v_groove_guide_pulley___4x13x6mm.obj",
            link="upper_arm",
            pos_in_link=[0.349567, -0.03445, 0.0476],
            diameter_mm=10.0,
            color=_C_GUIDE,
            role="guide_pulley",
            cable="upper",
            note="Mid-span guide pulley, -Y side",
        ),
        CableComponent(
            name="Guide pulley 3 (-Y)",
            obj_filename="steel_v_groove_guide_pulley___4x13x6mm.obj",
            link="upper_arm",
            pos_in_link=[0.360376, -0.0226536, 0.0476],
            diameter_mm=10.0,
            color=_C_GUIDE,
            role="guide_pulley",
            cable="upper",
            note="Pre-elbow guide pulley",
        ),
        CableComponent(
            name="Elbow roller groove (-Y)",
            obj_filename="mhp_arm_00_elbow_roller_v1.obj",
            link="lower_arm",
            pos_in_link=[6.10623e-16, 6.93889e-18, 0.0334],
            diameter_mm=78.8,
            color=_C_ROLLER,
            role="elbow_roller",
            cable="shared",
            note="Driven elbow roller, OD=85.44 mm (grove dia 78.8 mm)",
        ),
        CableComponent(
            name="Cable end point (leaves the roller and enters lower arm)",
            obj_filename="ball_cable_mount_lower_arm_enter.obj",
            link="lower_arm",
            pos_in_link=[0.037261, -0.0333478, 0.033401],
            diameter_mm=5.0, color=_C_LO_ARM, role="ball_marker", cable="upper",
            note="End point where cable leaves elbow roller and enters the lower arm mount Z=33.4 mm (upper groove)",
        )
    ]

    path = [
        CableComponent(
            name="Cable start point (spool anchor)",
            obj_filename="ball_cable_spool_upper_arm_start.obj",
            link="upper_arm",
            pos_in_link=[-0.0795, 0.020, 0.0645],  # on spool rim at +Y (anchor clamp, opp. to -Y pulleys)
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
            note="Cable clamp on spool rim — +Y side, angle=90°, r=20mm",
        ),
        CableComponent(
            name="Spool exit",
            obj_filename="ball_cable_spool_upper_arm_exit.obj",
            link="upper_arm",
            pos_in_link=[-0.0718787, -0.0159976, 0.0475001],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        ),
        CableComponent(
            name="Pulley B3",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[-0.0438212, -0.0387601, 0.0475001],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        ),
        CableComponent(
            name="Pulley B4",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[-0.0403245, -0.0400001, 0.0475001],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        ),
        CableComponent(
            name="Pulley B5",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[0.350166, -0.0400001, 0.0475001],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        ),
        CableComponent(
            name="Pulley B6",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[0.354258, -0.0381996, 0.0475001],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        ),
        CableComponent(
            name="Pulley B7",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[0.365068, -0.0264032, 0.0475001],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        ),
        CableComponent(
            name="Pulley B8",
            obj_filename="ball_cable_pulleys_upper_arm.obj",
            link="upper_arm",
            pos_in_link=[0.366505, -0.0221699, 0.0475001],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        ),
        CableComponent(
            name="Elbow roller enter",
            obj_filename="ball_cable_elbow_roller_enter.obj",
            link="upper_arm",
            pos_in_link=[0.360752, 0.0435847, 0.0475001],
            diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        ),
        CableComponent(
            name="Cable end point (leaves the roller and enters lower arm)",
            obj_filename="ball_cable_mount_lower_arm_enter.obj",
            link="lower_arm",
            pos_in_link=[0.037261, -0.0333478, 0.033401],
            diameter_mm=5.0, color=_C_LO_ARM, role="ball_marker", cable="upper",
            note="End point where cable leaves elbow roller and enters the lower arm mount Z=33.4 mm (upper groove)",
        ),
        # CableComponent(
        #     name="Lower arm exit",
        #     obj_filename="ball_cable_mount_lower_arm_exit.obj",
        #     link="lower_arm",
        #     pos_in_link=[0.0655164, -0.0660097, 0.0334],
        #     diameter_mm=5.0, color=_C_BALL, role="ball_marker", cable="upper",
        #     note="Z=33.4 mm (upper groove)",
        # ),
    ]

    return CableRouteConfig(
        name="Upper Cable — elbow (-Y)",
        color=_C_LINE,
        physical=physical,
        path=path,
        branch_sign_seq = [-1, -1, -1, -1, +1],
        kind_seq        = ['external', 'external', 'external', 'internal', 'external'],
        elbow_roller_arc_dir   = 'cw',
        n_spool_turns          = 3,
    )



def compute_cable_path(route: "CableRouteConfig",
                       kin: "MHPKinematics",
                       cable_loc: Literal['lower', 'upper']) -> CablePathData:
    """Compute all cable path geometry for *route* at the current FK pose.

    Returns a :class:`CablePathData` consumed identically by the matplotlib
    and Meshcat renderers — there is no other place where these tangents are
    computed.

    Tangent kinds match the physical geometry:
      * Spool → GP1, GP1 → GP2  : external tangent (both pulleys same side)
      * GP2 → GP3, GP3 → Roller : internal tangent (cable crosses between them)
    """
    path_w = [kin.to_world(c.pos_in_link, c.link) for c in route.path]
    phys_w = [kin.to_world(c.pos_in_link, c.link) for c in route.physical]
    phys_r = [c.diameter_mm * 0.5e-3              for c in route.physical]
    helix_branch = +1 if cable_loc == 'upper' else -1

    branch_sign_seq      = route.branch_sign_seq # [+1/-1 for each tangent segment]
    kind_seq             = route.kind_seq# ['external'/'internal' for each tangent segment]
    elbow_roller_arc_dir = route.elbow_roller_arc_dir# 'cw' or 'ccw' arc direction for wrap around elbow roller

    # physical indices:
    #   [0] cable_anchor  — clamp point on spool rim (no radius)
    #   [1] spool drum    — rotation axis, r=20 mm
    #   [2] GP1, [3] GP2, [4] GP3
    #   [5] elbow roller
    #   [6] cable endpoint

    # Compute tangents between spool and pulleys.
    T_spool_exit, T_gp1_entry = _compute_tangent(
        phys_w[1], phys_r[1], phys_w[2], phys_r[2], branch=branch_sign_seq[0],  kind=kind_seq[0])

    # Override T_spool_exit Z so the helix has a real axial span.
    # _compute_tangent preserves Z of the centre (phys_w[1][2]), making z_start==z_end
    # which produces a flat circle instead of a helix.
    # We advance Z by (n_turns × pitch) in the direction toward the guide pulleys:
    #   lower cable (helix_branch=-1) exits upward  → z_end = spool_Z + span
    #   upper cable (helix_branch=+1) exits downward → z_end = spool_Z - span
    _spool_pitch = route.spool_pitch_mm * 1e-3
    _z_helix_end = float(phys_w[1][2]) - helix_branch * route.n_spool_turns * _spool_pitch
    T_spool_exit = np.array([T_spool_exit[0], T_spool_exit[1], _z_helix_end], dtype=float)
    T_gp1_exit,   T_gp2_entry = _compute_tangent(
        phys_w[2], phys_r[2], phys_w[3], phys_r[3], branch=branch_sign_seq[1],  kind=kind_seq[1])
    T_gp2_exit,   T_gp3_entry = _compute_tangent(
        phys_w[3], phys_r[3], phys_w[4], phys_r[4], branch=branch_sign_seq[2],  kind=kind_seq[2])
    T_gp3_exit,   T_roller_in = _compute_tangent(
        phys_w[4], phys_r[4], phys_w[5], phys_r[5], branch=branch_sign_seq[3],  kind=kind_seq[3])
    T_roller_out, _            = _compute_tangent(
        phys_w[5], phys_r[5], phys_w[6], 0.0,       branch=branch_sign_seq[4], kind=kind_seq[4])

    # Build the full piecewise path as a list of (N,3) arrays for each segment.
    pieces = [
        # 1. Cable anchor (phys_w[0]) is fixed on the spool rim.
        #    Helix wraps around spool centre (phys_w[1]) starting from that anchor angle.
        np.vstack([
            phys_w[0].reshape(1, 3),
            _helix_pts_3d(phys_w[1], phys_r[1],
                          float(phys_w[0][2]), float(T_spool_exit[2]),
                          phys_w[0], T_spool_exit,
                          branch=helix_branch, n_turns=route.n_spool_turns, pts_per_turn=48),
        ]).astype(np.float32),
        # 2. Spool exit → GP1 entry
        np.array([T_spool_exit, T_gp1_entry], dtype=np.float32),
        # 3. GP1 arc
        _arc_pts_3d(phys_w[2], phys_r[2], T_gp1_entry, T_gp1_exit),
        # 4. GP1 exit → GP2 entry
        np.array([T_gp1_exit, T_gp2_entry], dtype=np.float32),
        # 5. GP2 arc
        _arc_pts_3d(phys_w[3], phys_r[3], T_gp2_entry, T_gp2_exit),
        # 6. GP2 exit → GP3 entry
        np.array([T_gp2_exit, T_gp3_entry], dtype=np.float32),
        # 7. GP3 arc
        _arc_pts_3d(phys_w[4], phys_r[4], T_gp3_entry, T_gp3_exit),
        # 8. GP3 exit → roller entry
        np.array([T_gp3_exit, T_roller_in], dtype=np.float32),
        # 9. Roller arc (direction is CCW for lower, CW for upper)
        _arc_pts_3d(phys_w[5], phys_r[5], T_roller_in, T_roller_out, direction=elbow_roller_arc_dir),
        # 10. Roller exit → cable endpoint (direct tangent)
        np.array([T_roller_out, phys_w[6]], dtype=np.float32),
    ]

    return CablePathData(
        route=route, path_w=path_w, phys_w=phys_w, phys_r=phys_r,
        T_spool_exit=T_spool_exit, T_gp1_entry=T_gp1_entry,
        T_gp1_exit=T_gp1_exit,    T_gp2_entry=T_gp2_entry,
        T_gp2_exit=T_gp2_exit,    T_gp3_entry=T_gp3_entry,
        T_gp3_exit=T_gp3_exit,    T_roller_in=T_roller_in,
        T_roller_out=T_roller_out, helix_branch=helix_branch,
        pieces=pieces,
    )


def _rpy_to_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Extrinsic RPY (roll=Rx, pitch=Ry, yaw=Rz) → 3×3 rotation matrix."""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    Rx = np.array([[1, 0,   0  ], [0,  cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp  ], [0,   1,   0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0 ], [sy,  cy,  0], [0,   0,  1]])
    return Rz @ Ry @ Rx


def _make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4×4 homogeneous transform from a 3×3 rotation and 3-vec translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _cylinder_transform(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """4×4 transform that maps a Z-aligned unit cylinder to the segment p1→p2.
    
    Meshcat's Cylinder geometry is centred at origin, aligned with Y-axis.
    """
    mid = (p1 + p2) / 2.0
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-9:
        return _make_transform(np.eye(3), mid)
    direction = direction / length

    # Rotate Y-axis → direction
    y_axis = np.array([0., 1., 0.])
    dot = np.clip(np.dot(y_axis, direction), -1.0, 1.0)
    if 1.0 - dot < 1e-8:          # already aligned
        R = np.eye(3)
    elif 1.0 + dot < 1e-8:        # anti-aligned
        R = np.diag([1., -1., -1.]).astype(float)
    else:
        axis = np.cross(y_axis, direction)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(dot)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return _make_transform(R, mid)


def visualize_cable_routing_meshcat(
    lower: CableRouteConfig,
    upper: CableRouteConfig,
    q1: float = 0.0,
    q2: float = 0.0,
) -> None:
    """Visualize cable routing in Meshcat browser viewer (real OBJ meshes).

    Shows 3D robot model with physical OBJ meshes, arm skeleton, and cables.
    Prints the URL to open in a browser.
    """
    if not MESHCAT_AVAILABLE:
        print("⚠️  Meshcat not available. Install: pip install meshcat")
        return

    import os

    vis = meshcat.Visualizer()
    kin = MHPKinematics(q1, q2)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, 'assets')

    # ── OBJ meshes for spools, guide pulleys, elbow roller ─────────────────────
    seen_names: set = set()   # deduplicate shared components (e.g. elbow roller used by both routes)
    for route in (lower, upper):
        for comp in route.physical:
            node_name = f"{comp.role}_{comp.name}"
            if node_name in seen_names:
                continue
            seen_names.add(node_name)

            # For cable routing, spool uses routing pos (where helix centers).
            # Visual override only for guide pulleys, elbow roller, etc.
            if comp.visual_pos_in_link is not None and comp.role != "spool":
                pw = kin.to_world(comp.visual_pos_in_link, comp.link)
                R_rpy = _rpy_to_mat(*comp.visual_rpy) if comp.visual_rpy is not None else np.eye(3)
                R  = kin.link_frame_rotation(comp.link) @ R_rpy
            else:
                pw = kin.to_world(comp.pos_in_link, comp.link)
                R  = kin.link_frame_rotation(comp.link)
            T  = _make_transform(R, pw)

            obj_path = os.path.join(assets_dir, comp.obj_filename)
            color_val = int(comp.color.lstrip('#'), 16)
            material  = g.MeshPhongMaterial(color=color_val, opacity=0.35,
                                            wireframe=False)

            if os.path.exists(obj_path):
                try:
                    geometry = g.ObjMeshGeometry.from_file(obj_path)
                    vis[node_name].set_object(geometry, material)
                    vis[node_name].set_transform(T)
                    continue
                except Exception as e:
                    print(f"  ⚠ OBJ load failed for {comp.obj_filename}: {e}")

            # Fallback: sphere sized to component radius (skip zero-size)
            r = comp.diameter_mm * 0.5e-3
            if r < 1e-6:
                continue
            vis[node_name].set_object(g.Sphere(r), material)
            vis[node_name].set_transform(T)

    # ── Arm skeleton (cylinders along link segments) ───────────────────────────
    base  = np.array([0., 0., 0.])
    J1    = kin.J1
    J2    = kin.J2
    EE    = kin.to_world(np.array([0.2, 0., 0.]), "lower_arm")
    bone_material = g.MeshPhongMaterial(color=0x2266aa, opacity=0.15)

    for i, (p1, p2) in enumerate([(base, J1), (J1, J2), (J2, EE)]):
        length = np.linalg.norm(p2 - p1)
        cyl    = g.Cylinder(length, 0.008)   # Cylinder(height, radius)
        vis[f"arm_{i}"].set_object(cyl, bone_material)
        vis[f"arm_{i}"].set_transform(_cylinder_transform(p1, p2))

    # Joint spheres
    joint_mat = g.MeshPhongMaterial(color=0xff8800, opacity=0.9)
    for i, pt in enumerate([J1, J2]):
        vis[f"joint_{i}"].set_object(g.Sphere(0.012), joint_mat)
        vis[f"joint_{i}"].set_transform(_make_transform(np.eye(3), pt))

    # ── Cable paths: rendered as tube (cylinders) for visible thickness ────────
    CABLE_RADIUS = 0.0005   # 0.5 mm tube radius — adjust for thicker/thinner appearance
    for route in (lower, upper):
        cp        = compute_cable_path(route, kin, cable_loc=route.path[0].cable)
        color_val = int(route.color.lstrip('#'), 16)
        cable_mat = g.MeshPhongMaterial(color=color_val, opacity=1.0)
        # Collect all consecutive point pairs across all pieces
        seg_idx = 0
        for piece in cp.pieces:
            pts = piece  # (N, 3) float32
            for k in range(len(pts) - 1):
                p1, p2 = pts[k].astype(float), pts[k + 1].astype(float)
                seg_len = np.linalg.norm(p2 - p1)
                if seg_len < 1e-6:
                    continue
                cyl = g.Cylinder(seg_len, CABLE_RADIUS)
                vis[f"cable_{route.name}/seg_{seg_idx}"].set_object(cyl, cable_mat)
                vis[f"cable_{route.name}/seg_{seg_idx}"].set_transform(
                    _cylinder_transform(p1, p2)
                )
                seg_idx += 1

    # ── Set default camera view centred on the robot ───────────────────────────
    # Meshcat's orthographic camera default frustum spans ~20 world-units.
    # The robot is ~0.5 m wide, so zoom ≈ 30 fills the view nicely.
    robot_cx = float((kin.J1[0] + kin.J2[0]) / 2)
    robot_cz = float((kin.J1[2] + kin.J2[2]) / 2)
    vis["/Cameras/default"].set_transform(
        tf.translation_matrix([robot_cx, 0.0, robot_cz])
    )
    vis["/Cameras/default"].set_property("zoom", 30)

    # ── Print URL and return vis so the server stays alive ─────────────────────
    try:
        url = vis.url()
    except Exception:
        url = "http://127.0.0.1:7001/static/"

    print(f"\n✓ Meshcat visualization ready!")
    print(f"  Open browser → {url}")
    print(f"  q1={np.rad2deg(q1):.1f}°  q2={np.rad2deg(q2):.1f}°")
    return vis


# ─── Main plot function ───────────────────────────────────────────────────────

def plot_cable_routing(
    lower: CableRouteConfig,
    upper: CableRouteConfig,
    q1: float = 0.0,
    q2: float = 0.0,
    view_elev: float = 28,
    view_azim: float = -55,
) -> plt.Figure:
    """Render dual-panel figure: 3D perspective (left) + XY top-down 3D (right).

    Parameters
    ----------
    lower, upper : CableRouteConfig
        Built by :func:`build_lower_cable_config` / :func:`build_upper_cable_config`.
    q1, q2 : float
        Joint angles in radians.
    """
    kin = MHPKinematics(q1, q2)

    fig = plt.figure(figsize=(24, 10), facecolor='white')
    fig.suptitle(
        f"MHP Cable Routing — World Frame   "
        f"q1 = {np.rad2deg(q1):.1f}°   q2 = {np.rad2deg(q2):.1f}°",
        fontsize=13, fontweight='bold', y=0.98,
    )

    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax_top_3d = fig.add_subplot(1, 2, 2, projection='3d')

    # ── Arm structure ─────────────────────────────────────────────────────────
    base_origin = np.array([0., 0., 0.])
    J1 = kin.J1
    J2 = kin.J2
    # Lower arm end-effector direction: 0.25 m along lower_arm X axis
    EE = kin.to_world(np.array([0.25, 0., 0.]), "lower_arm")

    arm_pts = np.array([base_origin, J1, J2, EE])
    _draw_arm_skeleton_multi_axes(ax3d, ax_top_3d, arm_pts)

    # World origin marker
    _draw_world_origin_multi_axes(ax3d, ax_top_3d)

    # Joint frames
    _draw_joint_frame(ax3d, ax_top_3d, J1, kin.R1, "J1 (shoulder)")
    _draw_joint_frame(ax3d, ax_top_3d, J2, kin.R2, "J2 (elbow)")

    # ── Cables ────────────────────────────────────────────────────────────────
    # physical[0]=cable_anchor, physical[1]=spool drum — capture spool from lower[1].
    shoulder_spool_w = kin.to_world(lower.physical[1].pos_in_link, lower.physical[1].link)
    shoulder_spool_r = lower.physical[1].diameter_mm * 0.5e-3
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
                vertices, faces = _load_obj(obj_path)
                if len(vertices) > 0:
                    # Transform vertices to world frame: apply FK rotation then translate
                    R = kin.link_frame_rotation(comp.link)
                    verts_rotated = vertices @ R.T
                    verts_world = verts_rotated + pw
                    _plot_mesh_3d(ax3d, verts_world, faces, color=comp.color, alpha=0.20)
                    _plot_mesh_3d(ax_top_3d, verts_world, faces, color=comp.color, alpha=0.20)
                    # Skip the fallback drawing below if mesh loaded successfully
                    continue

            # Fallback: draw simple geometric shapes if OBJ not available
            if comp.role == "cable_anchor":
                # Draw a small cross marker at the clamp point
                ax3d.scatter(*pw, s=60, color=comp.color, marker='x',
                             alpha=1.0, zorder=9, linewidths=1.5)
                ax_top_3d.scatter(*pw, s=60, color=comp.color, marker='x',
                                  alpha=1.0, zorder=9, linewidths=1.5)

            elif comp.role == "elbow_roller":
                _disc_3d(ax3d, pw, r_m, comp.color, alpha=0.20)
                _disc_3d(ax_top_3d, pw, r_m, comp.color, alpha=0.20)

            elif comp.role == "spool":
                ax3d.scatter(*pw, s=250, color=comp.color, marker='o',
                             alpha=0.85, zorder=6,
                             edgecolors='black', linewidths=0.6)
                ax_top_3d.scatter(*pw, s=250, color=comp.color, marker='o',
                                  alpha=0.85, zorder=6,
                                  edgecolors='black', linewidths=0.6)
                _disc_3d(ax3d, pw, r_m, comp.color, alpha=0.30)
                _disc_3d(ax_top_3d, pw, r_m, comp.color, alpha=0.30)

            elif comp.role == "guide_pulley":
                ax3d.scatter(*pw, s=55, color=comp.color, marker='o',
                             alpha=0.9, zorder=6,
                             edgecolors='white', linewidths=0.5)
                ax_top_3d.scatter(*pw, s=55, color=comp.color, marker='o',
                                  alpha=0.9, zorder=6,
                                  edgecolors='white', linewidths=0.5)

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

        # ── Step 1: Helix from cable anchor (phys_w[0]) wrapping spool (phys_w[1]) ────
        _draw_helix_multi_axes(ax3d, ax_top_3d,
                               phys_w[1], phys_r[1],
                               float(phys_w[0][2]), float(T_spool_exit[2]),
                               phys_w[0], T_spool_exit, route.color,
                               branch=helix_branch, n_turns=route.n_spool_turns, pts_per_turn=48)

        # ── Step 2: spool exit → GP1 entry ──────────────────────────────────
        _draw_segment_multi_axes(ax3d, ax_top_3d,
                                 T_spool_exit, T_gp1_entry, route.color)

        # ── Step 3: GP1 arc ──────────────────────────────────────────────────
        _draw_arc_multi_axes(ax3d, ax_top_3d,
                             phys_w[2], phys_r[2], T_gp1_entry, T_gp1_exit, route.color)

        # ── Step 4: GP1 exit → GP2 entry ────────────────────────────────────
        _draw_segment_multi_axes(ax3d, ax_top_3d,
                                 T_gp1_exit, T_gp2_entry, route.color)

        # ── Step 5: GP2 arc ──────────────────────────────────────────────────
        _draw_arc_multi_axes(ax3d, ax_top_3d,
                             phys_w[3], phys_r[3], T_gp2_entry, T_gp2_exit, route.color)

        # ── Step 6: GP2 exit → GP3 entry ────────────────────────────────────
        _draw_segment_multi_axes(ax3d, ax_top_3d,
                                 T_gp2_exit, T_gp3_entry, route.color)

        # ── Step 7: GP3 arc ──────────────────────────────────────────────────
        _draw_arc_multi_axes(ax3d, ax_top_3d,
                             phys_w[4], phys_r[4], T_gp3_entry, T_gp3_exit, route.color)

        # ── Step 8: GP3 exit → roller entry ─────────────────────────────────
        _draw_segment_multi_axes(ax3d, ax_top_3d,
                                 T_gp3_exit, T_roller_in, route.color)

        # ── Step 9: Roller arc (CCW for lower cable, CW for upper cable) ─────
        _roller_dir = 'ccw' if cp.route.path[0].cable == 'lower' else 'cw'
        _draw_arc_multi_axes(ax3d, ax_top_3d,
                             phys_w[5], phys_r[5], T_roller_in, T_roller_out,
                             route.color, direction=_roller_dir)

        # ── Step 10: Roller exit → cable endpoint (direct tangent) ───────────
        _draw_segment_multi_axes(ax3d, ax_top_3d,
                                 T_roller_out, phys_w[6], route.color)

        # Legend proxy for all axes
        _add_legend_proxy_multi_axes(ax3d, ax_top_3d,
                                     route.color, route.name)




    # ──────────────────────────────────────────────────────────────────────────
    ax3d.set_xlabel('X [m]', fontsize=9, labelpad=6)
    ax3d.set_ylabel('Y [m]', fontsize=9, labelpad=6)
    ax3d.set_zlabel('Z [m]', fontsize=9, labelpad=6)
    ax3d.set_title('3D Perspective View\n(Frames: X=red  Y=green  Z=blue)', fontsize=10)
    ax3d.view_init(elev=view_elev, azim=view_azim)
    ax3d.legend(loc='upper left', fontsize=7, framealpha=0.7)
    ax3d.tick_params(labelsize=7)

    # ── Top-down 3D view ──────────────────────────────────────────────────────
    ax_top_3d.set_xlabel('X [m]', fontsize=9, labelpad=6)
    ax_top_3d.set_ylabel('Y [m]', fontsize=9, labelpad=6)
    ax_top_3d.set_zlabel('Z [m]', fontsize=9, labelpad=6)
    ax_top_3d.set_title('XY Top-Down 3D View\n(with OBJ meshes)', fontsize=10)
    ax_top_3d.view_init(elev=85, azim=0)  # Nearly top-down view
    ax_top_3d.legend(loc='upper left', fontsize=7, framealpha=0.7)
    ax_top_3d.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="MHP cable routing visualisation — 3D + XY top view"
    )
    ap.add_argument("--q1",   type=float, default=0.0,
                    help="Shoulder joint angle q1 [deg]  (default 0)")
    ap.add_argument("--q2",   type=float, default=150.0,
                    help="Elbow joint angle q2 [deg]     (default 0)")
    ap.add_argument("--elev", type=float, default=28,
                    help="3D view elevation angle [deg] (default 28)")
    ap.add_argument("--azim", type=float, default=-55,
                    help="3D view azimuth angle [deg] (default -55)")
    ap.add_argument("--save", type=str,   default=None,
                    help="Save figure to this path (PNG/PDF) instead of showing")
    ap.add_argument("--show", action='store_true',
                    help="Show interactive plot window (rotatable with mouse) - only if not saving")
    args = ap.parse_args()

    q1_rad = np.deg2rad(args.q1)
    q2_rad = np.deg2rad(args.q2)

    lower_cable = build_lower_cable_config()
    upper_cable = build_upper_cable_config()

    print(f"\nLower cable ({lower_cable.name}): "
          f"{len(lower_cable.physical)} physical components, "
          f"{len(lower_cable.path)} path markers")
    print(f"Upper cable ({upper_cable.name}): "
          f"{len(upper_cable.physical)} physical components, "
          f"{len(upper_cable.path)} path markers")

    kin = MHPKinematics(q1_rad, q2_rad)
    print(f"\nFK at q1={args.q1:.1f}°, q2={args.q2:.1f}°:")
    print(f"  J1 (shoulder) = {kin.J1}")
    print(f"  J2 (elbow)    = {kin.J2}")

    fig = plot_cable_routing(lower_cable, upper_cable, q1=q1_rad, q2=q2_rad,
                            view_elev=args.elev, view_azim=args.azim)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"\nSaved → {args.save}")
        if args.show:
            plt.show()
    else:
        print("\n💡 TIP: The 3D plot is interactive! Use your mouse to rotate/zoom:")
        print("   • Left-click + drag to rotate")
        print("   • Right-click + drag (or scroll) to zoom")
        print("   • Use --elev and --azim to set specific view angles")
        print("   • Use --save FILE to save to PNG/PDF instead of showing\n")
        plt.show()
    
    # Meshcat visualization — keep reference alive so the server stays up
    vis = visualize_cable_routing_meshcat(lower_cable, upper_cable, q1=q1_rad, q2=q2_rad)
    if vis is not None:
        import sys, time
        if sys.stdin.isatty():
            try:
                input("\nPress Enter to stop the Meshcat server...\n")
            except (EOFError, KeyboardInterrupt):
                pass
        else:
            print("\nMeshcat server running. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
