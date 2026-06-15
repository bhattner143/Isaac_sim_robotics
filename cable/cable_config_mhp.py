"""MHP cable route configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

@dataclass
class CableComponent:
    """One element in the cable routing path (spool / guide pulley / roller / ball marker)."""
    name: str
    obj_filename: str
    link: str                  # 'base_link' | 'upper_arm' | 'lower_arm'
    pos_in_link: np.ndarray    # (3,) URDF visual origin xyz [m]
    diameter_mm: float         # outer diameter [mm]
    color: str                 # matplotlib colour
    role: str                  # 'spool' | 'guide_pulley' | 'elbow_roller' | 'ball_marker'
    cable: str                 # 'lower' | 'upper' | 'shared'
    note: str = ""
    visual_pos_in_link: object = None  # optional override for visual origin xyz
    visual_rpy: object = None          # optional override for visual rpy [rad]

    def __post_init__(self):
        self.pos_in_link = np.asarray(self.pos_in_link, float)

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
