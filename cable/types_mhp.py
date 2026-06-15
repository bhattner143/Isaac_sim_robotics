"""MHP robot types, FK, and cable route dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from cable.geometry_mhp import Rz

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
        R1 = Rz(q1)
        R2 = Rz(q1 + q2)
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
    spool_pitch_mm: float = 2.5     # axial pitch between cable wraps [mm]


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
