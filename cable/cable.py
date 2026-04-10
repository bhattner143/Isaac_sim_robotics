#!/usr/bin/env python3
"""
test_cable_routing_viz.py
─────────────────────────
Visualizes the tendon/cable routing of the cable manipulator in Meshcat.

Cable route (from URDF visual origin xyz values):
  ① link1_base_pulley  — drive pulley, on pulley_htd_5m_60t body  [0.0142, 0, 0.2660]
  ② 623zz  (side A)   — idler bearing top,    on pulley_htd_5m_60t [0.2531,+0.0165, 0.1982]
  ③ 623zz_2 (side B)  — idler bearing bottom, on pulley_htd_5m_60t [0.2569,-0.0150, 0.2018]
  ④ pulley_big         — driven pulley, on link2_tendon body        [0, 0, 0.0045]

The cable exits ① on one side, wraps around ② then ③ (opposite Y — "the other
side"), and drives ④ on link2.  Because ④ is on link2_tendon (q2 body) while
①②③ are on pulley_htd_5m_60t (q1 body), CalcPointsPositions must be called on
the correct body each time.

Interactive: type  q1 q2 [deg]  at the prompt → manipulator moves + cable redraws.

Usage:
    python test_cable_routing_viz.py
"""

import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from pydrake.geometry import Rgba, Cylinder

sys.path.append(str(Path(__file__).parent))

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    SceneGraph,
    MeshcatVisualizer,
    StartMeshcat,
    Simulator,
    Parser,
    RigidTransform,
    RotationMatrix,
    RevoluteJoint,
    PrismaticJoint,
)
from pydrake.multibody.tree import FixedOffsetFrame
from pydrake.math import RollPitchYaw
from termcolor import colored

from robots.cup_manipulator import RobotBase
from configs.robot.robot_types import ManipulatorConfig, JointConfig, Pose
from project_utils.viz_cables import (
    print_cable_routing_points,
    _Xw,
    draw_cables,
    visualize_cable_routing_top_view,
    visualize_cable_routing_3d,
)



# ─── URDF part-origin parser ─────────────────────────────────────────────────

def _parse_urdf_part_origins(urdf_path: str) -> dict:
    """Return ``{part_name: (xyz_tuple, rpy_tuple)}`` from the URDF.

    Scans for the pattern that ``onshape-to-robot`` emits for every visual::

        <!-- Part NAME -->
        <visual>
          <origin xyz="x y z" rpy="r p y"/>

    All part names are unique in a URDF produced by that tool.
    """
    text    = Path(urdf_path).read_text()
    pattern = re.compile(
        r'<!--\s*Part\s+(\S+?)\s*-->\s*'
        r'<visual>\s*'
        r'<origin\s+xyz="([^"]+)"\s+rpy="([^"]+)"'
    )
    return {
        m.group(1): (
            tuple(float(v) for v in m.group(2).split()),
            tuple(float(v) for v in m.group(3).split()),
        )
        for m in pattern.finditer(text)
    }


# ─── PulleyBase: base class for cable-routing objects ────────────────────────

class PulleyBase:
    """Base class for cable-routing objects (pulleys and idler bearings).

    Subclasses declare their geometry as class attributes.  centroid and radius
    are computed lazily from the OBJ mesh on first property access and cached.

    Set PulleyBase.assets_dir once at startup, before accessing any property::

        PulleyBase.assets_dir = "path/to/manipulator_cable/assets"

    Body-frame position convention (URDF <visual><origin>):
        p_body = Rz(yaw) @ Ry(pitch) @ Rx(roll) @ p_local + vis_xyz
    """

    # ── Subclasses must define ────────────────────────────────────────────────
    obj_name:   str    # OBJ mesh filename in assets_dir
    body_name:  str    # Drake/URDF rigid-body link name
    vis_xyz:    tuple  # URDF visual origin xyz [x, y, z] (metres)
    vis_rpy:    tuple  # URDF visual origin rpy [roll, pitch, yaw] (radians)
    face_color: str    # matplotlib hex colour for 3-D mesh rendering
    label:      str    # human-readable name used in plots and logs

    # ── Subclasses may override ───────────────────────────────────────────────
    mesh_alpha:        float = 0.45    # transparency for 3-D mesh rendering [0–1]
    pulley_axis_local: tuple = (0, 0, 1)
    # Pulley rotation axis in OBJ local space:
    #   • Z (default): DRIVE_PULLEY, IDLER_L, IDLER_R  (rpy = 0, OBJ Z → body Z)
    #   • X          : PULLEY_BIG  (Ry(90°) maps OBJ X → body Z)

    # ── Set once at startup (before first property access) ────────────────────
    assets_dir:     str  = ""  # path to directory containing OBJ files
    urdf_part_name: str  = ""  # <!-- Part NAME --> key in URDF; set in each subclass
    _urdf_origins:  dict = {}  # populated at module level via _parse_urdf_part_origins()
                               # __init__ reads this to set vis_xyz/vis_rpy per-instance

    def __init__(self) -> None:
        """Eagerly compute all context-free transforms at instantiation.

        If ``PulleyBase._urdf_origins`` has been populated (by assigning
        ``PulleyBase._urdf_origins = _parse_urdf_part_origins(urdf_path)``
        at module level), and this subclass declares ``urdf_part_name``,
        then ``vis_xyz`` / ``vis_rpy`` are overridden as **instance** attributes
        from the URDF before any transforms are computed.  The class-level
        literals become fallbacks only.

        Computed only when ``assets_dir`` is set (OBJ mesh required):
          • ``centroid``   — mesh geometric centroid in body frame
          • ``radius``     — max cable-groove radius
        """
        # ── Override vis_xyz / vis_rpy from URDF if available ────────────────
        name = self.urdf_part_name
        if name and name in PulleyBase._urdf_origins:
            self.vis_xyz, self.vis_rpy = PulleyBase._urdf_origins[name]

        # ── Conditional: mesh-derived — skip if assets_dir not yet set ───────
        if not self.assets_dir:
            return  # centroid / radius filled lazily on first access
        self._centroid_cache = self._compute_centroid()
        self._radius_cache   = self._compute_radius()

    # ── Lazily cached, mesh-derived properties ────────────────────────────────

    def _compute_centroid(self) -> np.ndarray:
        """Load OBJ and return mesh geometric centroid transformed into body frame.

        Applies the URDF <visual><origin> transform:
            p_body = Rz(yaw) @ Ry(pitch) @ Rx(roll) @ p_local + vis_xyz
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError("Install trimesh first:  pip install trimesh")

        mesh_path = Path(self.assets_dir) / self.obj_name
        if not mesh_path.exists():
            raise FileNotFoundError(f"{self.obj_name} not found at {mesh_path}")

        mesh           = trimesh.load(str(mesh_path), force="mesh")
        centroid_local = np.array(mesh.centroid)          # in OBJ local space

        # URDF RPY: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        roll, pitch, yaw = self.vis_rpy
        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)
        Rx = np.array([[1,   0,    0], [0,  cr, -sr], [0,  sr, cr]])
        Ry = np.array([[cp,  0,   sp], [0,   1,   0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy,   0], [sy,  cy,  0], [0,   0,  1]])
        R  = Rz @ Ry @ Rx

        xyz_offset    = np.array(self.vis_xyz, dtype=float)
        centroid_body = R @ centroid_local + xyz_offset

        print(colored("─" * 60, "cyan"))
        print(colored(f"⚙  {self.obj_name} centroid computation", "cyan", attrs=["bold"]))
        print(colored(f"   OBJ file              : {mesh_path}", "yellow"))
        print(colored(f"   Mesh centroid (local) : [{centroid_local[0]:.6f},"
                      f" {centroid_local[1]:.6f}, {centroid_local[2]:.6f}]", "yellow"))
        print(colored(f"   URDF xyz offset       : {xyz_offset.tolist()}", "yellow"))
        print(colored(f"   URDF rpy              : [{roll:.4f}, {pitch:.4f}, {yaw:.4f}]", "yellow"))
        print(colored(f"   → Centroid in body frame:", "green"))
        print(colored(f"     [{centroid_body[0]:.6f}, {centroid_body[1]:.6f},"
                      f" {centroid_body[2]:.6f}]", "green", attrs=["bold"]))
        print(colored("─" * 60, "cyan"))

        return centroid_body

    @property
    def centroid(self) -> np.ndarray | None:
        """Mesh geometric centroid in body frame (lazy, cached on first access).

        Returns None if ``PulleyBase.assets_dir`` has not been set yet so that
        debugger hover / variable inspection does not trigger OBJ loading and
        produce a traceback in the Variables panel.
        """
        if not hasattr(self, "_centroid_cache"):
            if not self.assets_dir:
                return None   # assets_dir not set yet — silent no-op
            self._centroid_cache = self._compute_centroid()
        return self._centroid_cache

    @property
    def radius(self) -> float | None:
        """Outer cable-groove radius in metres (lazy, cached on first access).

        Returns None if ``PulleyBase.assets_dir`` has not been set yet.
        """
        if not hasattr(self, "_radius_cache"):
            if not self.assets_dir:
                return None   # assets_dir not set yet — silent no-op
            self._radius_cache = self._compute_radius()
        return self._radius_cache

    @property
    def is_resolved(self) -> bool:
        """True once centroid and radius have been computed and cached.

        Use this guard instead of accessing ``.centroid`` or ``.radius``
        directly when you only want to *check* readiness without triggering
        OBJ loading (which requires ``assets_dir`` to be set first).
        """
        return hasattr(self, "_centroid_cache") and hasattr(self, "_radius_cache")

    def _compute_radius(self) -> float:
        """Load OBJ and return max radial distance from centroid ⊥ to shaft axis."""
        try:
            import trimesh
        except ImportError:
            raise ImportError("Install trimesh first:  pip install trimesh")

        mesh_path = Path(self.assets_dir) / self.obj_name
        if not mesh_path.exists():
            raise FileNotFoundError(f"{self.obj_name} not found at {mesh_path}")

        mesh  = trimesh.load(str(mesh_path), force="mesh")
        c     = np.array(mesh.centroid)                   # OBJ local centroid
        v     = mesh.vertices - c                         # centred vertices (N × 3)
        ax    = np.array(self.pulley_axis_local, dtype=float)
        ax   /= np.linalg.norm(ax)
        v_rad = v - (v @ ax)[:, None] * ax               # axial component removed
        r     = float(np.linalg.norm(v_rad, axis=1).max())

        print(colored(f"   {self.label:<22} radius = {r * 1e3:.2f} mm "
                      f"(axis = {self.pulley_axis_local})", "cyan"))
        return r

    @property
    def waypoint(self) -> np.ndarray:
        """Body-frame position for cable routing (mesh geometric centroid)."""
        return self.centroid

    def centroid_world_frame(self, plant, plant_context, manipulator
                             ) -> tuple:
        """Return the centroid position and body-frame orientation in world frame.

        The frame axes are identical to the joint (body) frame axes — the
        origin is simply shifted to the mesh centroid.

        Parameters
        ----------
        plant, plant_context, manipulator :
            Standard Drake objects (same signature as world_points).

        Returns
        -------
        origin_world : np.ndarray, shape (3,)
            Centroid position expressed in the world frame.
        R_WB : np.ndarray, shape (3, 3)
            Rotation matrix from body frame to world frame (columns = X, Y, Z
            body axes expressed in the world frame).
        """
        body = plant.GetBodyByName(self.body_name, manipulator.model_instance)
        X    = plant.CalcRelativeTransform(plant_context, plant.world_frame(),
                                           body.body_frame())
        R_WB, t_WB = X.rotation().matrix(), X.translation()
        origin_world = R_WB @ self.centroid + t_WB
        return origin_world, R_WB

    # ── Helper ────────────────────────────────────────────────────────────────

    def _R_vis(self) -> np.ndarray:
        """3×3 rotation matrix from URDF visual-origin RPY: R = Rz(yaw)@Ry(pitch)@Rx(roll)."""
        roll, pitch, yaw = self.vis_rpy
        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)
        Rx = np.array([[1,    0,   0], [0,  cr, -sr], [0,  sr, cr]])
        Ry = np.array([[cp,   0,  sp], [0,   1,   0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy,   0], [sy, cy,   0], [0,   0,  1]])
        return Rz @ Ry @ Rx

    # ── Context-free convenience ──────────────────────────────────────────────

    @property
    def shaft_axis_body(self) -> np.ndarray:
        """Pulley shaft direction expressed in the **body frame**.

        ``pulley_axis_local`` is defined in OBJ local space.  The URDF visual
        origin rotation ``_R_vis()`` maps it into the body frame::

            shaft_body = Rz(yaw) @ Ry(pitch) @ Rx(roll) @ pulley_axis_local

        For Drive / Idler (rpy = 0): shaft_body = (0, 0, 1)  (no change).
        For BigPulley (Ry(π/2)):    shaft_body = (0, 0, 1)  (OBJ X → body Z).
        """
        ax = self._R_vis() @ np.asarray(self.pulley_axis_local, dtype=float)
        return ax / np.linalg.norm(ax)

    @property
    def centroid_in_joint_frame(self) -> np.ndarray | None:
        """Mesh geometric centroid expressed in the body / joint frame.

        Alias for :attr:`centroid`; the explicit name makes the intent clearer
        when assigning cable anchor points (B_L / B_R) in subclass ``__init__``.
        Returns ``None`` if the mesh assets have not been resolved yet.
        """
        return self.centroid

    # ── Pure geometry ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_tangent(c1, r1, c2, r2, branch: int = +1,
                kind: str = "external") -> tuple[np.ndarray, np.ndarray]:
        """Compute one tangent line between circles (c1, r1) and (c2, r2).

        Works in the XY plane; Z is preserved from each input centroid.

        Parameters
        ----------
        c1, c2 : array-like, shape (3,)
            Circle centres (world frame). Only XY is used; Z is restored in output.
        r1, r2 : float
            Circle radii.
        branch : int
            +1 or -1, selects which of the two parallel tangent branches to return.
        kind : str
            ``"external"`` (default) — tangent line does not pass between the
            circles; both circles lie on the same side.  Exists when
            ``D >= |r1 - r2|``.

            ``"internal"`` — tangent line passes between the circles; each
            circle lies on the opposite side.  Exists only when ``D >= r1 + r2``
            (circles do not overlap).

        Returns
        -------
        T1, T2 : np.ndarray, shape (3,)
            Tangency points on circle 1 and circle 2 in their respective
            input z-planes.
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
            if D < r1 + r2:
                raise ValueError(
                    "No internal tangent exists: circles overlap (D < r1 + r2).")
            cos_a = (r1 + r2) / D
            sin_a = np.sqrt(max(0.0, 1.0 - cos_a**2))
            n = cos_a * d_hat + branch * sin_a * perp
            T1 = np.array([p1[0] + r1 * n[0], p1[1] + r1 * n[1], c1[2]])
            T2 = np.array([p2[0] - r2 * n[0], p2[1] - r2 * n[1], c2[2]])  # opposite side
        else:  # "external"
            if D < abs(r1 - r2):
                raise ValueError(
                    "No external tangent exists: one circle lies too far inside the other.")
            cos_a = (r1 - r2) / D
            sin_a = np.sqrt(max(0.0, 1.0 - cos_a**2))
            n = cos_a * d_hat + branch * sin_a * perp
            T1 = np.array([p1[0] + r1 * n[0], p1[1] + r1 * n[1], c1[2]])
            T2 = np.array([p2[0] + r2 * n[0], p2[1] + r2 * n[1], c2[2]])
        return T1, T2

    @staticmethod
    def tangent_in_world_frame(plant, plant_context, manipulator, cfg1, cfg2,
                               branch: int, kind: str = "external") -> tuple[np.ndarray, np.ndarray]:
        """Compute an external tangent contact pair for two pulleys via Drake FK.

        Both pulleys may live in different body frames (cross-frame pairs such as
        Idler ↔ BigPulley).  Each centroid is first mapped to the world frame via
        FK, the 2-D external tangent is solved there using
        :meth:`compute_tangent`, then T1 / T2 are mapped back into their
        respective body frames.

        Parameters
        ----------
        plant, plant_context, manipulator :
            Standard Drake objects.
        cfg1, cfg2 : PulleyBase
            The two pulleys.  Must have ``body_name``, ``centroid``, and
            ``radius`` set (i.e. :attr:`is_resolved` is True).
        branch : int
            +1 or -1 passed to :meth:`external_tangent`.

        Returns
        -------
        T1_body, T2_body : np.ndarray, shape (3,)
            Tangent contact points in cfg1’s and cfg2’s respective body frames.
        """
        def _Xw(body_name):
            body = plant.GetBodyByName(body_name, manipulator.model_instance)
            X    = plant.CalcRelativeTransform(plant_context, plant.world_frame(),
                                               body.body_frame())
            return X.rotation().matrix(), X.translation()

        R1, t1 = _Xw(cfg1.body_name)
        R2, t2 = _Xw(cfg2.body_name)

        c1_w   = R1 @ np.asarray(cfg1.centroid) + t1
        c2_w   = R2 @ np.asarray(cfg2.centroid) + t2

        T1_w, T2_w = PulleyBase.compute_tangent(c1_w, cfg1.radius, c2_w, cfg2.radius, branch, kind)
        return R1.T @ (T1_w - t1), R2.T @ (T2_w - t2)

class CableStartPointR(PulleyBase):
    """Right cable anchor ball on pulley_htd_5m_60t (green cable start, routes to IDLER_R).

    Fixed attachment point where the green cable is clamped on the back of the q1 body.
    URDF: Part simple_ball_2 on pulley_htd_5m_60t
        xyz="-0.0238116  0  0.203849"  rpy="0 0 0"
    Radius is zero — fixed anchor, not a wrapping pulley.
    """
    obj_name        = "simple_ball.obj"
    body_name       = "pulley_htd_5m_60t"
    urdf_part_name  = "simple_ball_2"      # <!-- Part simple_ball_2 --> in URDF
    vis_xyz         = (-0.0238116,  0.0,  0.203849)
    vis_rpy         = (0.0,         0.0,  0.0     )
    face_color = "#dd9999"      # light red — red cable start anchor
    label      = "Start-point R"
    mesh_alpha = 0.80
    A_R        = None  # None cause the cable start here — no entering tangent point
    B_R        = None|np.ndarray  # exiting tangent point, left/red cable

    def __init__(self):
        super().__init__() 
        self.B_R = self.centroid_in_joint_frame  # fixed anchor, so exiting tangent point is the centroid


    def _compute_radius(self) -> float:
        """Fixed anchor — no pulley wrapping, radius is zero."""
        return 0.0


class CableStartPointL(PulleyBase):
    """Left cable anchor ball on pulley_htd_5m_60t (red cable start, routes to IDLER_L).

    Fixed attachment point where the red cable is clamped on the back of the q1 body.
    URDF: Part simple_ball on pulley_htd_5m_60t
        xyz="-0.0238116  0  0.200249"  rpy="0 0 0"
    Radius is zero — fixed anchor, not a wrapping pulley.
    """
    obj_name        = "simple_ball.obj"
    body_name       = "pulley_htd_5m_60t"
    urdf_part_name  = "simple_ball"        # <!-- Part simple_ball --> in URDF
    vis_xyz         = (-0.0238116,  0.0,  0.200249)
    vis_rpy         = (0.0,         0.0,  0.0     )
    face_color = "#99dd99"      # light green — green cable start anchor
    label      = "Start-point L"
    mesh_alpha = 0.80
    A_L        = None  # None cause the cable start here — no entering tangent point
    B_L        = None|np.ndarray  # exiting tangent point, left/red cable

    def __init__(self):
        super().__init__() 
        self.B_L = self.centroid_in_joint_frame  # fixed anchor, so exiting tangent point is the centroid

    def _compute_radius(self) -> float:
        """Fixed anchor — no pulley wrapping, radius is zero."""
        return 0.0

# ─── Concrete pulley / idler-bearing subclasses ───────────────────────────────

class DrivePulley(PulleyBase):
    """HTD 5M 60T drive pulley mounted on the q1 body (pulley_htd_5m_60t)."""
    obj_name       = "link1_base_pulley.obj"
    body_name      = "pulley_htd_5m_60t"
    urdf_part_name = "link1_base_pulley"  # <!-- Part link1_base_pulley --> in URDF
    vis_xyz        = (0.0141884,  0.0,  0.266049)
    vis_rpy        = (0.0,        0.0,  0.0     )
    face_color    = "#d4843a"
    label         = "Drive pulley"
    mesh_alpha    = 0.50
    # Belt spec: HTD 5M 60T  (5 mm pitch, 60 teeth)
    belt_teeth:   int   = 60
    belt_pitch_m: float = 0.005            # [m]  5 mm for HTD-5M profile
    pitch_radius: float = belt_teeth * belt_pitch_m / (2 * np.pi)  # ≈ 47.746 mm [m]
    A_R          = None|np.ndarray  # entering tangent point, right/green cable
    A_L        = None|np.ndarray  # entering tangent point, left/red cable
    B_R        = None|np.ndarray  # exiting tangent point, right/green cable
    B_L        = None|np.ndarray  # exiting tangent point, left/red cable

    


class IdlerL(PulleyBase):
    """Left idler bearing (623ZZ) on the q1 body, +Y side of the drive pulley."""
    obj_name       = "623zz.obj"
    body_name      = "pulley_htd_5m_60t"
    urdf_part_name = "623zz"              # <!-- Part 623zz --> in URDF (+Y idler)
    vis_xyz        = (0.253146,   0.0165029, 0.198249)
    vis_rpy        = (0.0,        0.0,       0.0     )
    face_color = "#c0c0c0"
    label      = "Idler 623zz-L"
    mesh_alpha = 0.40
    A_L        = None|np.ndarray  # entering tangent point, left/red cable
    B_L        = None|np.ndarray  # exiting tangent point, left/red cable



class IdlerR(PulleyBase):
    """Right idler bearing (623ZZ) on the q1 body, −Y side of the drive pulley."""
    obj_name       = "623zz.obj"
    body_name      = "pulley_htd_5m_60t"
    urdf_part_name = "623zz_2"            # <!-- Part 623zz_2 --> in URDF (−Y idler)
    vis_xyz        = (0.256914,  -0.015,  0.201849)
    vis_rpy        = (0.0,        0.0,    0.0     )
    face_color = "#909090"
    label      = "Idler 623zz-R"
    mesh_alpha = 0.40
    A_R        = None|np.ndarray  # entering tangent point, right/green cable
    B_R        = None|np.ndarray  # exiting tangent point, right/green cable


class BigPulley(PulleyBase):
    """Large driven pulley on the q2 body (link2_tendon).

    The OBJ mesh X-axis is the pulley shaft.  The URDF visual origin applies
    Ry(90°) which maps OBJ X → body Z, so pulley_axis_local = (1, 0, 0).
    """
    obj_name          = "pulley_big.obj"
    body_name         = "link2_tendon"
    urdf_part_name    = "pulley_big"       # <!-- Part pulley_big --> in URDF
    vis_xyz           = (0.0,  0.0,  0.00450711)
    vis_rpy           = (0.0,  np.pi / 2,  0.0)
    face_color        = "#606060"
    label             = "Driven pulley-big"
    mesh_alpha        = 0.45
    # Belt spec: HTD 5M 60T  (5 mm pitch, 60 teeth)
    belt_teeth:   int   = 60
    belt_pitch_m: float = 0.005            # [m]  5 mm for HTD-5M profile
    pitch_radius: float = belt_teeth * belt_pitch_m / (2 * np.pi)  # ≈ 47.746 mm [m]
    pulley_axis_local = (1, 0, 0)   # OBJ X = shaft; Ry(90°) maps it to body Z
    A_L               = None|np.ndarray  # entering tangent point, left/red cable
    A_R               = None|np.ndarray  # entering tangent point, right/green cable
    B_L               = None|np.ndarray  # exiting tangent point, left/red cable
    B_R               = None|np.ndarray  # exiting tangent point, right/green cable


class CableEndPointL(PulleyBase):
    """Left cable attachment ball on link2_tendon (+Y side, same side as IDLER_L).

    This is the terminal anchor where the red (left) cable is fixed on link2.
    The ball mesh centroid gives the attachment point; radius is zero (no wrapping).
    URDF: Part simple_ball_3  xyz="0.12 0.05 -0.0182"  rpy="-π/2 0 π/2"
    """
    obj_name        = "simple_ball.obj"
    body_name       = "link2_tendon"
    urdf_part_name  = "simple_ball_3"      # <!-- Part simple_ball_3 --> in URDF (+Y end)
    vis_xyz         = (0.12,  0.05,  -0.0182)
    vis_rpy         = (-np.pi / 2,  0.0,  np.pi / 2)
    face_color = "#ff6666"      # light red — marks the left-side anchor
    label      = "End-point L"
    mesh_alpha = 0.70
    A_L        = None|np.ndarray  # entering tangent point, left/red cable
    B_L        = None             # None cause the cable terminate here

    def __init__(self):
        super().__init__() 
        self.B_L = self.centroid_in_joint_frame  # fixed anchor, so exiting tangent point is the centroid

    def _compute_radius(self) -> float:
        """Cable terminates here; no pulley wrapping — radius is zero."""
        return 0.0


class CableEndPointR(PulleyBase):
    """Right cable attachment ball on link2_tendon (−Y side, same side as IDLER_R).

    This is the terminal anchor where the green (right) cable is fixed on link2.
    The ball mesh centroid gives the attachment point; radius is zero (no wrapping).
    URDF: Part simple_ball_4  xyz="0.12 -0.05 -0.0218"  rpy="-π/2 0 π/2"
    """
    obj_name        = "simple_ball.obj"
    body_name       = "link2_tendon"
    urdf_part_name  = "simple_ball_4"      # <!-- Part simple_ball_4 --> in URDF (−Y end)
    vis_xyz         = (0.12,  -0.05,  -0.0218)
    vis_rpy         = (-np.pi / 2,  0.0,  np.pi / 2)
    face_color = "#66cc66"      # light green — marks the right-side anchor
    label      = "End-point R"
    mesh_alpha = 0.70
    A_R        = None|np.ndarray  # entering tangent point, right/green cable
    B_R        = None             # None cause the cable terminate here

    def __init__(self):
        super().__init__() 
        self.B_R = self.centroid_in_joint_frame  # fixed anchor, so exiting tangent point is the centroid

    def _compute_radius(self) -> float:
        """Cable terminates here; no pulley wrapping — radius is zero."""
        return 0.0


@dataclass
class CableRoute:
    """One tendon cable: an ordered sequence of waypoint configs.

    Each segment is a ``FixedBodyPoint`` holding a pre-computed body-frame
    contact position (tangent point or anchor centroid).  ``world_points()``
    applies FK to transform them into the world frame for drawing.
    """
    segments:             list         # list[tuple[waypoint_cfg, ignored_offset]]
    meshcat_path:         str          # Meshcat scene-tree path for this cable line
    meshcat_color:        object       # Rgba from pydrake.geometry
    mpl_color:            str          # matplotlib colour string
    label:                str          # human-readable route name
    skip_chord_segments:  frozenset    # segment indices whose chord is replaced by a wrap arc

    def world_points(self, plant, plant_context, manipulator) -> np.ndarray:
        """Return (N, 3) world-frame cable contact points.

        All contact positions are pre-computed by ``rig.compute_tangents()``
        and stored in ``FixedBodyPoint`` waypoints as body-frame points.
        This method applies FK to bring them into the world frame.
        """
        pts = []
        for cfg, _ in self.segments:
            body = plant.GetBodyByName(cfg.body_name, manipulator.model_instance)
            X    = plant.CalcRelativeTransform(plant_context, plant.world_frame(),
                                               body.body_frame())
            pts.append(X.rotation().matrix() @ cfg.waypoint + X.translation())
        return np.array(pts)


# ─── FixedBodyPoint: zero-radius explicit cable waypoint ────────────────────

class FixedBodyPoint:
    """A fixed body-frame point used as an explicit, pre-calculated cable waypoint.

    Unlike PulleyBase subclasses, this holds a pre-computed body-frame position
    (e.g. a tangent contact point) rather than deriving its location from an OBJ
    mesh centroid.  Its radius is zero and ``waypoint`` returns its stored
    body-frame position directly for ``world_points()`` to FK into world frame.
    """
    radius      = 0.0
    is_resolved = False   # not a mesh-derived object; tangent point is pre-computed

    def __init__(self, body_name: str, tangent_point: np.ndarray, label: str) -> None:
        self.body_name    = body_name
        self._tangent_point = np.asarray(tangent_point, float)
        self.label        = label

    @property
    def tangent_point(self) -> np.ndarray:
        return self._tangent_point

    @property
    def waypoint(self) -> np.ndarray:
        if not np.any(self._tangent_point):
            raise RuntimeError(
                f"FixedBodyPoint '{self.label}' is still a zero placeholder — "
                "call rig.compute_tangents() before accessing waypoints."
            )
        return self._tangent_point


# ─── CableSpring: optional compliant element at cable endpoints ──────────────

@dataclass
class CableSpring:
    """Spring element inserted between a cable endpoint attachment and the cable.

    When enabled, the last cable segment (BigPulley exit → EndPoint) is drawn
    as a zigzag spring instead of a straight line.

    Attributes:
        stiffness:        Spring stiffness [N/m].
        rest_length:      Natural (unstretched) length [m].
        n_coils:          Number of helical coils for visualization.
        amplitude:        Helix radius perpendicular to spring axis [m].
        spring_fraction:  Fraction of the last cable segment occupied by the
                          spring (0–1).  E.g. 0.30 means the spring is 30% of
                          the total segment length.
        spring_position:  Centre position of the spring along the segment,
                          measured from the endpoint side (0 = at endpoint,
                          1 = at pulley).  E.g. 0.5 centres the spring;
                          0.3 places it closer to the endpoint.
        enabled:          Whether this spring is active.
        label:            Human-readable name for logging.
    """
    stiffness:        float = 100.0
    rest_length:      float = 0.002
    n_coils:          int   = 6
    amplitude:        float = 0.004
    spring_fraction:  float = 0.30
    spring_position:  float = 0.50
    enabled:          bool  = True
    label:            str   = ""


def spring_zigzag_points(p_start: np.ndarray, p_end: np.ndarray,
                         n_coils: int = 6, amplitude: float = 0.004,
                         lead_fraction: float = 0.1,
                         pts_per_coil: int = 16) -> np.ndarray:
    """Generate 3-D helical coil points between two endpoints to visualize a spring.

    Parameters
    ----------
    p_start, p_end : array-like, shape (3,)
        Spring attachment points in whatever frame (world / body).
    n_coils : int
        Number of full helical turns.
    amplitude : float
        Helix radius perpendicular to the spring axis [m].
    lead_fraction : float
        Fraction of total length reserved for straight lead-in/lead-out segments
        at each end (so the spring doesn't start abruptly at the anchor).
    pts_per_coil : int
        Number of sample points per coil turn (higher = smoother helix).

    Returns
    -------
    pts : np.ndarray, shape (N, 3)
        Ordered 3-D points tracing the helical spring path.
    """
    p0 = np.asarray(p_start, float)
    p1 = np.asarray(p_end, float)
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-9:
        return np.vstack([p0, p1])
    ax = axis / length

    # Build two perpendicular directions spanning the plane normal to the axis
    tmp = np.array([0.0, 1.0, 0.0]) if abs(ax[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    perp1 = np.cross(ax, tmp)
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(ax, perp1)  # already unit length

    lead_len = length * lead_fraction
    coil_len = length - 2 * lead_len
    if coil_len <= 0:
        return np.vstack([p0, p1])

    pts = [p0, p0 + ax * lead_len]

    # Helical coil: parametric helix along the axis
    n_total = n_coils * pts_per_coil
    for i in range(n_total + 1):
        frac = i / n_total               # 0 → 1 along coil region
        t = lead_len + coil_len * frac    # distance along axis
        theta = 2.0 * np.pi * n_coils * frac  # angle swept
        pts.append(p0 + ax * t
                   + amplitude * np.cos(theta) * perp1
                   + amplitude * np.sin(theta) * perp2)

    pts.append(p0 + ax * (length - lead_len))
    pts.append(p1)

    return np.array(pts)

# ──────────────────────────────────────────────────────────────────────────────

class CableRig:
    """All cable-routing state in one place: pulleys, tangent waypoints, and routes.

    Create AFTER setting PulleyBase.assets_dir (so mesh centroids resolve eagerly).
    Then call rig.compute_tangents(plant, ctx, manipulator) to compute FK tangents.
    """
    def __init__(self, springs_enabled: bool = True):
        _Q1 = "pulley_htd_5m_60t"
        _Q2 = "link2_tendon"

        # ── Pulleys and end-balls ─────────────────────────────────────────────
        self.drive_pulley  = DrivePulley()
        self.idler_l       = IdlerL()
        self.idler_r       = IdlerR()
        self.pulley_big    = BigPulley()
        self.cable_start_l = CableStartPointL()
        self.cable_start_r = CableStartPointR()
        self.cable_end_l   = CableEndPointL()
        self.cable_end_r   = CableEndPointR()

        # Drive pulley A contacts alias the start-ball exit points
        self.drive_pulley.A_R = self.cable_start_r.B_R
        self.drive_pulley.A_L = self.cable_start_l.B_L

        # Tangent contact placeholders (mutated in-place by compute_tangents)
        self.drive_pulley.B_R = np.zeros(3)
        self.idler_r.A_R      = np.zeros(3)
        self.drive_pulley.B_L = np.zeros(3)
        self.idler_l.A_L      = np.zeros(3)
        self.idler_r.B_R      = np.zeros(3)
        self.pulley_big.A_L   = np.zeros(3)
        self.idler_l.B_L      = np.zeros(3)
        self.pulley_big.A_R   = np.zeros(3)
        self.pulley_big.B_L   = np.zeros(3)
        self.pulley_big.B_R   = np.zeros(3)
        self.cable_end_l.A_L  = np.zeros(3)
        self.cable_end_r.A_R  = np.zeros(3)

        # ── FixedBodyPoint wrappers ───────────────────────────────────────────
        self.tangent_drive_b_r = FixedBodyPoint(_Q1, self.drive_pulley.B_R, "Drive exit B_R")
        self.tangent_idler_a_r = FixedBodyPoint(_Q1, self.idler_r.A_R,      "IdlerR entry A_R")
        self.tangent_drive_b_l = FixedBodyPoint(_Q1, self.drive_pulley.B_L, "Drive exit B_L")
        self.tangent_idler_a_l = FixedBodyPoint(_Q1, self.idler_l.A_L,      "IdlerL entry A_L")
        self.tangent_idler_b_r = FixedBodyPoint(_Q1, self.idler_r.B_R,      "IdlerR exit B_R")
        self.tangent_big_a_l   = FixedBodyPoint(_Q2, self.pulley_big.A_L,   "BigPulley entry A_L")
        self.tangent_idler_b_l = FixedBodyPoint(_Q1, self.idler_l.B_L,      "IdlerL exit B_L")
        self.tangent_big_a_r   = FixedBodyPoint(_Q2, self.pulley_big.A_R,   "BigPulley entry A_R")
        self.tangent_big_b_l   = FixedBodyPoint(_Q2, self.pulley_big.B_L,   "BigPulley exit B_L")
        self.tangent_big_b_r   = FixedBodyPoint(_Q2, self.pulley_big.B_R,   "BigPulley exit B_R")

        # ── Two cable routes ──────────────────────────────────────────────────
        self.cable_green = CableRoute(
            segments=[
                (self.cable_start_r,     np.zeros(3)),  # ① start ball (−Y)
                (self.tangent_drive_b_r, np.zeros(3)),  # ② drive exit → IdlerR
                (self.tangent_idler_a_r, np.zeros(3)),  # ③ IdlerR entry
                (self.tangent_idler_b_r, np.zeros(3)),  # ④ IdlerR exit → BigPulley
                (self.tangent_big_a_l,   np.zeros(3)),  # ⑤ BigPulley entry
                (self.tangent_big_b_l,   np.zeros(3)),  # ⑥ BigPulley exit → EndL
                (self.cable_end_l,       np.zeros(3)),  # ⑦ anchor on link2 (+Y)
            ],
            meshcat_path        = "/cable/green",
            meshcat_color       = Rgba(0.1, 0.85, 0.1, 1.0),
            mpl_color           = "limegreen",
            label               = "Green (Drive→IdlerR→Big→EndL)",
            skip_chord_segments = frozenset({2, 4}),
        )
        self.cable_red = CableRoute(
            segments=[
                (self.cable_start_l,     np.zeros(3)),  # ① start ball (+Y)
                (self.tangent_drive_b_l, np.zeros(3)),  # ② drive exit → IdlerL
                (self.tangent_idler_a_l, np.zeros(3)),  # ③ IdlerL entry
                (self.tangent_idler_b_l, np.zeros(3)),  # ④ IdlerL exit → BigPulley
                (self.tangent_big_a_r,   np.zeros(3)),  # ⑤ BigPulley entry
                (self.tangent_big_b_r,   np.zeros(3)),  # ⑥ BigPulley exit → EndR
                (self.cable_end_r,       np.zeros(3)),  # ⑦ anchor on link2 (−Y)
            ],
            meshcat_path        = "/cable/red",
            meshcat_color       = Rgba(0.9, 0.1, 0.1, 1.0),
            mpl_color           = "red",
            label               = "Red   (Drive→IdlerL→Big→EndR)",
            skip_chord_segments = frozenset({2, 4}),
        )

        self.routes    = [self.cable_green, self.cable_red]
        self.waypoints = [self.drive_pulley, self.idler_l, self.idler_r,
                          self.pulley_big,
                          self.cable_start_l, self.cable_start_r,
                          self.cable_end_l,   self.cable_end_r]

        # ── Optional springs at cable endpoints ───────────────────────────────
        self.springs_enabled = springs_enabled
        self.spring_L = CableSpring(label="Spring End-L", enabled=springs_enabled)
        self.spring_R = CableSpring(label="Spring End-R", enabled=springs_enabled)

    def compute_tangents(self, plant, plant_context, manipulator) -> None:
        """Compute all inter-pulley tangent contacts using Drake FK.

        Call once after the Drake plant is built with joint positions set,
        and again whenever joint angles change (cross-frame pairs like
        Idler<->BigPulley move relative to each other with q2).
        """
        twf = PulleyBase.tangent_in_world_frame

        # ── Green cable ─────────────────────────────────────────────────────────
        self.drive_pulley.B_R, self.idler_r.A_R    = twf(plant, plant_context, manipulator, self.drive_pulley, self.idler_r,    kind="external", branch=-1)
        self.idler_r.B_R,      self.pulley_big.A_L = twf(plant, plant_context, manipulator, self.idler_r,      self.pulley_big, kind="internal", branch=-1)
        self.pulley_big.B_L,   self.cable_end_l.A_L = twf(plant, plant_context, manipulator, self.pulley_big,  self.cable_end_l, kind="external", branch=+1)

        # ── Red cable ───────────────────────────────────────────────────────────
        self.drive_pulley.B_L, self.idler_l.A_L    = twf(plant, plant_context, manipulator, self.drive_pulley, self.idler_l,    kind="external", branch=+1)
        self.idler_l.B_L,      self.pulley_big.A_R = twf(plant, plant_context, manipulator, self.idler_l,      self.pulley_big, kind="internal", branch=+1)
        self.pulley_big.B_R,   self.cable_end_r.A_R = twf(plant, plant_context, manipulator, self.pulley_big,  self.cable_end_r, kind="external", branch=-1)

        # ── Propagate to FixedBodyPoint wrappers (mutate _tangent_point in place) ──
        self.tangent_drive_b_r._tangent_point = self.drive_pulley.B_R
        self.tangent_idler_a_r._tangent_point = self.idler_r.A_R
        self.tangent_idler_b_r._tangent_point = self.idler_r.B_R
        self.tangent_big_a_l._tangent_point   = self.pulley_big.A_L
        self.tangent_drive_b_l._tangent_point = self.drive_pulley.B_L
        self.tangent_idler_a_l._tangent_point = self.idler_l.A_L
        self.tangent_idler_b_l._tangent_point = self.idler_l.B_L
        self.tangent_big_a_r._tangent_point   = self.pulley_big.A_R
        self.tangent_big_b_l._tangent_point   = self.pulley_big.B_L
        self.tangent_big_b_r._tangent_point   = self.pulley_big.B_R


# ═══════════════════════════════════════════════════════════════════════════════
# CUP MANIPULATOR — cable-driven 2-DOF robot adapter for Drake
# ═══════════════════════════════════════════════════════════════════════════════
# Lives here (not in robots/cup_manipulator_tendon.py) so that test_drive_pulley
# is fully self-contained.  robots/cup_manipulator_tendon.py re-exports this class
# as CupManipulatorTendon for backward compatibility with other scripts.

class CupManipulator(RobotBase):
    """Cable-driven (tendon) 2-DOF manipulator for Drake.

    Wraps manipulator_cable.urdf which uses a belt/pulley transmission.
    Joint names:
        JT1_NAME = "link1_base"   (q1)
        JT2_NAME = "link2_link1"  (q2)
    """

    JT1_NAME  = "link1_base"
    JT2_NAME  = "link2_link1"
    ACT1_NAME = f"tau_{JT1_NAME}"
    ACT2_NAME = f"tau_{JT2_NAME}"

    BASE_LINK_NAME = "base_mate"
    LINK2_NAME     = "link2_tendon"

    EE_XYZ_LINK2  = np.array([0.19, 0.0, 0.0515])
    EE_RPY_LINK2  = np.array([0.0, 0.0, 0.0])
    EE_FRAME_NAME = "tendon_ee"
    EE_OFFSET     = EE_XYZ_LINK2

    def __init__(self, config: ManipulatorConfig, enable_visualization: bool = True):
        super().__init__(config)
        self.joint_names: List[str]    = [self.JT1_NAME, self.JT2_NAME]
        self.actuator_names: List[str] = []
        self.enable_visualization      = enable_visualization
        self.rig                       = None  # CableRig — set via init_cable_rig()

    # ── URDF loading ────────────────────────────────────────────────────────

    def load_urdf_to_plant(self, plant: MultibodyPlant, parser: Parser) -> int:
        model_instance = super().load_urdf_to_plant(plant, parser)
        self.JT1_NAME    = "link1_base"
        self.JT2_NAME    = "link2_link1"
        self.ACT1_NAME   = f"tau_{self.JT1_NAME}"
        self.ACT2_NAME   = f"tau_{self.JT2_NAME}"
        self.joint_names = [self.JT1_NAME, self.JT2_NAME]
        print(colored(
            f"✓ CupManipulator: joints confirmed: [{self.JT1_NAME}, {self.JT2_NAME}]",
            'green'
        ))
        return model_instance

    # ── End-effector frame ──────────────────────────────────────────────────

    def add_end_effector_frame(self, plant: MultibodyPlant):
        if plant.is_finalized():
            raise RuntimeError("Cannot add EE frame after plant is finalized")
        link2_body = plant.GetBodyByName(self.LINK2_NAME, self.model_instance)
        X_L2_EE    = RigidTransform(RollPitchYaw(self.EE_RPY_LINK2), self.EE_XYZ_LINK2)
        try:
            return plant.GetFrameByName(self.EE_FRAME_NAME, self.model_instance)
        except Exception:
            pass
        return plant.AddFrame(
            FixedOffsetFrame(
                self.EE_FRAME_NAME,
                link2_body.body_frame(),
                X_L2_EE,
                self.model_instance,
            )
        )

    def get_end_effector_frame(self, plant: MultibodyPlant):
        return plant.GetFrameByName(self.EE_FRAME_NAME, self.model_instance)

    # ── Joint actuators ─────────────────────────────────────────────────────

    def add_joint_actuators(self, plant: MultibodyPlant):
        if plant.is_finalized():
            raise RuntimeError("Cannot add actuators after plant is finalized")
        jt1 = self.get_joint_by_name(plant, self.JT1_NAME)
        jt2 = self.get_joint_by_name(plant, self.JT2_NAME)
        plant.AddJointActuator(self.ACT1_NAME, jt1)
        plant.AddJointActuator(self.ACT2_NAME, jt2)
        self.actuator_names = [self.ACT1_NAME, self.ACT2_NAME]
        print(colored(f"✓ Added actuators: {self.ACT1_NAME}, {self.ACT2_NAME}", 'green'))

    # ── EE kinematics ───────────────────────────────────────────────────────

    def get_end_effector_position(self, plant: MultibodyPlant, context) -> np.ndarray:
        ee_frame = self.get_end_effector_frame(plant)
        X_WE     = plant.CalcRelativeTransform(context, plant.world_frame(), ee_frame)
        return X_WE.translation()

    def CalcPosition(self, plant: MultibodyPlant, context) -> np.ndarray:
        return self.get_end_effector_position(plant, context)

    # ── State helpers ───────────────────────────────────────────────────────

    def get_state_from_plant(self, plant: MultibodyPlant, context) -> np.ndarray:
        return plant.GetPositionsAndVelocities(context, self.model_instance)

    def set_state_in_plant(self, plant: MultibodyPlant, context, user_state: np.ndarray):
        q1, q2, q1_dot, q2_dot = user_state
        self.set_jt([self.JT1_NAME, self.JT2_NAME], plant, context, [q1, q2])
        self.set_jt_velocity([self.JT1_NAME, self.JT2_NAME], plant, context, [q1_dot, q2_dot])

    def get_positions_user_order(self, plant: MultibodyPlant, context) -> np.ndarray:
        return np.array(self.get_jt([self.JT1_NAME, self.JT2_NAME], plant, context))

    def set_positions_user_order(self, plant: MultibodyPlant, context, user_positions):
        if isinstance(user_positions, dict):
            for joint_name, angle in user_positions.items():
                self.set_jt([joint_name], plant, context, [angle])
        else:
            q1, q2 = user_positions
            self.set_jt([self.JT1_NAME, self.JT2_NAME], plant, context, [q1, q2])

    def get_velocities_user_order(self, plant: MultibodyPlant, context) -> np.ndarray:
        return self.get_jt_velocity([self.JT1_NAME, self.JT2_NAME], plant, context)

    def set_velocities_user_order(self, plant: MultibodyPlant, context, user_velocities):
        if isinstance(user_velocities, dict):
            for joint_name, velocity in user_velocities.items():
                self.set_jt_velocity([joint_name], plant, context, [velocity])
        else:
            q1_dot, q2_dot = user_velocities
            self.set_jt_velocity([self.JT1_NAME, self.JT2_NAME], plant, context, [q1_dot, q2_dot])

    def get_joint_positions(self, plant: MultibodyPlant, context) -> dict:
        positions = {}
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0:
                if isinstance(joint, RevoluteJoint):
                    positions[joint.name()] = joint.get_angle(context)
                elif isinstance(joint, PrismaticJoint):
                    positions[joint.name()] = joint.get_translation(context)
        return positions

    def get_joint_velocities(self, plant: MultibodyPlant, context) -> dict:
        velocities = {}
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0:
                if isinstance(joint, RevoluteJoint):
                    velocities[joint.name()] = joint.get_angular_rate(context)
                elif isinstance(joint, PrismaticJoint):
                    velocities[joint.name()] = joint.get_translation_rate(context)
        return velocities

    # ── Joint helpers ───────────────────────────────────────────────────────

    def get_joint_by_name(self, plant: MultibodyPlant, joint_name: str):
        return plant.GetJointByName(joint_name, self.model_instance)

    def get_jt(self, joint_name, plant: MultibodyPlant, context):
        if isinstance(joint_name, list):
            return np.array([
                self.get_joint_by_name(plant, n).get_angle(context) for n in joint_name
            ])
        return self.get_joint_by_name(plant, joint_name).get_angle(context)

    def set_jt(self, joint_name, plant: MultibodyPlant, context, angle):
        if isinstance(joint_name, list):
            angles = np.atleast_1d(angle)
            for name, ang in zip(joint_name, angles):
                self.get_joint_by_name(plant, name).set_angle(context, float(ang))
        else:
            self.get_joint_by_name(plant, joint_name).set_angle(context, float(angle))

    def get_jt_velocity(self, joint_name, plant: MultibodyPlant, context):
        if isinstance(joint_name, list):
            return np.array([
                self.get_joint_by_name(plant, n).get_angular_rate(context) for n in joint_name
            ])
        return self.get_joint_by_name(plant, joint_name).get_angular_rate(context)

    def set_jt_velocity(self, joint_name, plant: MultibodyPlant, context, velocity):
        if isinstance(joint_name, list):
            velocities = np.atleast_1d(velocity)
            for name, vel in zip(joint_name, velocities):
                self.get_joint_by_name(plant, name).set_angular_rate(context, float(vel))
        else:
            self.get_joint_by_name(plant, joint_name).set_angular_rate(context, float(velocity))

    # ── Inverse kinematics ──────────────────────────────────────────────────

    def solve_initial_pose_via_ik(
        self,
        plant,
        target_xy,
        q_seed,
        pos_tol: float = 1e-3,
        verbose: bool = False,
        ee_frame_name: Optional[str] = None,
        target_z: Optional[float] = None,
    ):
        from pydrake.multibody.inverse_kinematics import InverseKinematics
        from pydrake.solvers import Solve

        target_xy = np.asarray(target_xy).reshape(2,)
        q_seed    = np.asarray(q_seed).reshape(2,)
        ik         = InverseKinematics(plant)
        ik_context = ik.context()
        self.set_positions_user_order(plant, ik_context, q_seed)
        world = plant.world_frame()

        if ee_frame_name is None:
            ee_frame_name = self.EE_FRAME_NAME
        try:
            ee_frame = plant.GetFrameByName(ee_frame_name, self.model_instance)
            p_BQ = np.zeros(3)
        except Exception:
            link2_body = plant.GetBodyByName(self.LINK2_NAME, self.model_instance)
            ee_frame   = link2_body.body_frame()
            p_BQ       = np.asarray(self.EE_XYZ_LINK2).reshape(3,)

        ee_pos_seed = plant.CalcPointsPositions(
            ik_context, ee_frame, p_BQ.reshape(3, 1), world
        ).ravel()
        z_target = target_z if target_z is not None else ee_pos_seed[2]

        if verbose:
            print(f"  Seed EE: ({ee_pos_seed[0]:.3f}, {ee_pos_seed[1]:.3f}, {ee_pos_seed[2]:.3f})")
            print(f"  Target:  ({target_xy[0]:.3f}, {target_xy[1]:.3f}, {z_target:.3f})")
            print(f"  Tol:     ±{pos_tol:.6f} m")

        lower = np.array([target_xy[0], target_xy[1], z_target]) - pos_tol
        upper = np.array([target_xy[0], target_xy[1], z_target]) + pos_tol
        ik.AddPositionConstraint(
            frameB=ee_frame, p_BQ=p_BQ,
            frameA=world, p_AQ_lower=lower, p_AQ_upper=upper,
        )
        prog   = ik.prog()
        q_vars = ik.q()
        q0_all = plant.GetPositions(ik_context)
        prog.AddQuadraticErrorCost(1000.0 * np.eye(len(q0_all)), q0_all, q_vars)
        prog.SetInitialGuess(q_vars, q0_all)
        result = Solve(prog)

        if verbose:
            print(f"  Solver: {result.get_solver_id().name()}, success={result.is_success()}")
        if not result.is_success():
            return q_seed, False

        q_sol_all    = result.GetSolution(q_vars)
        temp_context = plant.CreateDefaultContext()
        plant.SetPositions(temp_context, q_sol_all)
        q_sol_user   = plant.GetPositions(temp_context, self.model_instance)
        return np.asarray(q_sol_user), True

    # ── Cable rig ───────────────────────────────────────────────────────────

    def init_cable_rig(self, urdf_path: str = None, assets_dir: str = None,
                       springs_enabled: bool = True) -> None:
        """Initialize the cable rig.  Call after the plant is built.

        Args:
            springs_enabled: If True, add compliant springs at End-point L/R.
        """
        if urdf_path is None:
            urdf_path = self.config.urdf_path
        if assets_dir is None:
            assets_dir = str(Path(urdf_path).parent / "assets")
        PulleyBase._urdf_origins = _parse_urdf_part_origins(urdf_path)
        PulleyBase.assets_dir    = assets_dir
        self.rig = CableRig(springs_enabled=springs_enabled)

    def compute_tangents(self, plant, plant_context) -> None:
        """Recompute all cable tangent contacts at the current joint configuration."""
        if self.rig is None:
            raise RuntimeError("init_cable_rig() must be called before compute_tangents()")
        self.rig.compute_tangents(plant, plant_context, self)

    # ── Weld base ───────────────────────────────────────────────────────────

    def weld_base_to_world(
        self,
        plant: MultibodyPlant,
        position:    np.ndarray = np.array([0.0, 0.0, 0.0]),
        orientation: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ):
        if plant.is_finalized():
            raise RuntimeError("Cannot weld base after plant is finalized")
        base_body = plant.GetBodyByName(self.BASE_LINK_NAME, self.model_instance)
        X_WB      = RigidTransform(RollPitchYaw(orientation), position)
        plant.WeldFrames(plant.world_frame(), base_body.body_frame(), X_WB)
        print(colored(
            f"✓ Welded '{self.BASE_LINK_NAME}' to world at pos={position}, rpy={orientation}",
            'green'
        ))


def create_cable_manipulator_config(
    urdf_path: str = "model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf",
    joint_angles: Optional[dict] = None,
    damping:    tuple = (0.1, 0.1),
    stiffness:  tuple = (0.0, 0.0),
    friction:   tuple = (0.0, 0.0),
) -> ManipulatorConfig:
    """Factory for the cable (tendon) manipulator configuration."""
    urdf_dir    = str(Path(urdf_path).parent)
    joint_names = ["link1_base", "link2_link1"]
    if joint_angles is None:
        joint_angles = {n: 0.0 for n in joint_names}
    joint_configs = {}
    for i, name in enumerate(joint_names):
        joint_configs[name] = JointConfig(
            position=joint_angles.get(name, 0.0),
            damping=damping[i],
            stiffness=stiffness[i],
            friction=friction[i],
        )
    return ManipulatorConfig(
        name="manipulator_cable",
        urdf_path=urdf_path,
        joint_configs=joint_configs,
        base_pose=Pose(),
        package_map={"assets": urdf_dir + "/assets/"},
    )


# ──────────────────────────────────────────────────────────────────────────────
def build_plant(manipulator_config):
    """Build DiagramBuilder + MultibodyPlant containing only the manipulator."""
    builder     = DiagramBuilder()
    plant       = MultibodyPlant(time_step=0.0)
    scene_graph = builder.AddSystem(SceneGraph())
    plant.RegisterAsSourceForSceneGraph(scene_graph)

    manipulator = CupManipulator(manipulator_config, enable_visualization=True)
    parser_urdf = Parser(plant)
    manipulator.load_urdf_to_plant(plant, parser_urdf)
    manipulator.weld_base_to_world(plant, position=np.zeros(3), orientation=np.zeros(3))
    manipulator.add_joint_actuators(plant)
    manipulator.add_end_effector_frame(plant)
    plant.Finalize()

    builder.AddSystem(plant)
    builder.Connect(
        plant.get_geometry_pose_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()),
    )
    builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port(),
    )

    return builder, plant, scene_graph, manipulator



# ──────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Cable routing visualization.")
    ap.add_argument("--no-springs", action="store_true",
                    help="Disable endpoint springs (default: springs enabled)")
    args = ap.parse_args()
    springs_enabled = not args.no_springs

    # ── Configuration ─────────────────────────────────────────────────────────
    config = create_cable_manipulator_config(
        urdf_path="model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf",
        joint_angles={"link1_base": 0.0, "link2_link1": 0.0},
        damping=(0.1, 0.1),
    )

    # ── Meshcat ───────────────────────────────────────────────────────────────
    meshcat = StartMeshcat()
    print(colored(f"\n🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))

    # ── Plant ─────────────────────────────────────────────────────────────────
    builder, plant, scene_graph, manipulator = build_plant(config)

    # ── Cable rig — owned by manipulator, mirrors physical assembly ───────────
    manipulator.init_cable_rig(springs_enabled=springs_enabled)
    rig = manipulator.rig  # local alias for draw_cables / viz helpers

    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram   = builder.Build()
    simulator = Simulator(diagram)
    context   = simulator.get_mutable_context()
    plant_ctx = plant.GetMyMutableContextFromRoot(context)

    # ── Home pose ─────────────────────────────────────────────────────────────
    current_q = np.array([0.0, 0.0])
    manipulator.set_positions_user_order(plant, plant_ctx, {
        "link1_base":  current_q[0],
        "link2_link1": current_q[1],
    })
    plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))
    diagram.ForcedPublish(context)
    manipulator.compute_tangents(plant, plant_ctx)  # FK-based, all pairs
    draw_cables(meshcat, plant, plant_ctx, manipulator, rig)    # straight segments + wrap arcs
    print_cable_routing_points(plant, plant_ctx, manipulator, rig)

    # Figure 1 — top view (XY)
    _top_fig, _ = visualize_cable_routing_top_view(plant, plant_ctx, manipulator, 0.0, 0.0, rig)
    plt.show(block=False)
    plt.pause(0.05)

    # # Figure 2 — 3-D view with OBJ meshes
    # _viz_fig, _ = visualize_cable_routing_3d(
    #     plant, plant_ctx, manipulator, PulleyBase.assets_dir, 0.0, 0.0
    # )
    # plt.show(block=False)
    # plt.pause(0.1)
    _viz_fig = None  # created on first interactive update

    ee = manipulator.get_end_effector_position(plant, plant_ctx)
    print(colored("Cable route: drive_pulley → 623zz (A) → 623zz_2 (B, other side) → pulley_big", "yellow"))
    print(colored(f"Home:  q1=0°  q2=0°  →  EE=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) m\n", "cyan"))
    print(colored("Enter joint angles in degrees  (e.g.  30  -15)  or Ctrl+C to exit.\n", "yellow"))

    # ── Interactive loop ───────────────────────────────────────────────────────
    try:
        while True:
            raw = input(colored("q1  q2 [deg]: ", "cyan")).strip()
            if not raw:
                continue
            try:
                parts = raw.split()
                if len(parts) != 2:
                    print(colored("  ✗ Expected exactly two values: q1 q2", "red"))
                    continue
                q1_deg, q2_deg = float(parts[0]), float(parts[1])
                current_q = np.deg2rad([q1_deg, q2_deg])

                manipulator.set_positions_user_order(plant, plant_ctx, {
                    "link1_base":  current_q[0],
                    "link2_link1": current_q[1],
                })
                plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))

                # Update robot geometry
                diagram.ForcedPublish(context)

                # Recompute all tangents in world frame (q1 and q2 may have changed)
                manipulator.compute_tangents(plant, plant_ctx)

                # Recompute and redraw cable at new pose
                draw_cables(meshcat, plant, plant_ctx, manipulator, rig)
                # Update Figure 1 (top view) and Figure 2 (3-D)
                plt.close(_top_fig)
                _top_fig, _ = visualize_cable_routing_top_view(plant, plant_ctx, manipulator, q1_deg, q2_deg, rig)
                plt.show(block=False)
                plt.pause(0.05)

                if _viz_fig is not None:
                    plt.close(_viz_fig)
                _viz_fig, _ = visualize_cable_routing_3d(plant, plant_ctx, manipulator, PulleyBase.assets_dir, q1_deg, q2_deg, rig)
                plt.show(block=False)
                plt.pause(0.05)

                ee = manipulator.get_end_effector_position(plant, plant_ctx)
                print(colored(
                    f"  ✓  q1={q1_deg:.1f}°  q2={q2_deg:.1f}°  "
                    f"→  EE=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) m",
                    "green",
                ))
            except ValueError:
                print(colored("  ✗ Invalid numbers. Enter two floats: q1 q2", "red"))
    except KeyboardInterrupt:
        print(colored("\n✓ Stopped.", "green"))


# ============================================================================
# DRAKE CABLE PLANT — headless FK wrapper for cable tangent computation
# ============================================================================

class DrakeCablePlant:
    """Headless Drake plant used solely for cable FK / tangent computation.

    No rendering — just MultibodyPlant + CableRig.  Joint angles are
    synchronised from Isaac Sim (or any caller), then compute_tangents()
    produces world-frame cable waypoints that can be drawn as USD prims.

    Usage::

        from cable import DrakeCablePlant
        dc = DrakeCablePlant(drake_urdf, q1=q1_rad, q2=q2_rad)
        dc.update(new_q1, new_q2)
        for route, pts in dc.get_cable_world_points():
            ...
    """

    def __init__(self, drake_urdf: str, q1: float = 0.0, q2: float = 0.0,
                 springs_enabled: bool = True):
        config = create_cable_manipulator_config(
            urdf_path=drake_urdf,
            joint_angles={"link1_base": q1, "link2_link1": q2},
        )
        self.builder, self.plant, self._sg, self.manipulator = build_plant(config)
        self.manipulator.init_cable_rig(urdf_path=drake_urdf,
                                        springs_enabled=springs_enabled)
        self.rig = self.manipulator.rig
        self.diagram = self.builder.Build()
        self._root_ctx = self.diagram.CreateDefaultContext()
        self.plant_ctx = self.plant.GetMyMutableContextFromRoot(self._root_ctx)
        self._set_angles(q1, q2)
        self.manipulator.compute_tangents(self.plant, self.plant_ctx)

    def _set_angles(self, q1: float, q2: float):
        self.manipulator.set_positions_user_order(
            self.plant, self.plant_ctx,
            {"link1_base": q1, "link2_link1": q2},
        )
        self.plant.SetVelocities(self.plant_ctx,
                                 np.zeros(self.plant.num_velocities()))

    def update(self, q1: float, q2: float):
        """Sync joint angles and recompute cable tangents."""
        self._set_angles(q1, q2)
        self.manipulator.compute_tangents(self.plant, self.plant_ctx)

    def get_cable_world_points(self):
        """Return list of (route, world_pts) for each cable."""
        return [
            (route, route.world_points(self.plant, self.plant_ctx, self.manipulator))
            for route in self.rig.routes
        ]

    def get_wrap_arcs(self, n_arc_pts: int = 24):
        """Return wrap-arc world-frame points for each pulley wrap segment.

        Yields ``(label, color_rgb, arc_pts_Nx3)`` tuples.
        """
        dp = self.rig.drive_pulley
        ir = self.rig.idler_r
        il = self.rig.idler_l
        pb = self.rig.pulley_big
        _G = (0.1, 0.85, 0.1)   # green
        _R = (0.9, 0.1,  0.1)   # red
        wraps = [
            (dp, dp.A_R, dp.B_R, "drive_green",  _G),
            (ir, ir.A_R, ir.B_R, "idlerR_green", _G),
            (pb, pb.A_L, pb.B_L, "big_green",    _G),
            (dp, dp.A_L, dp.B_L, "drive_red",    _R),
            (il, il.A_L, il.B_L, "idlerL_red",   _R),
            (pb, pb.A_R, pb.B_R, "big_red",      _R),
        ]

        def _Xw(body_name):
            body = self.plant.GetBodyByName(body_name, self.manipulator.model_instance)
            X = self.plant.CalcRelativeTransform(
                self.plant_ctx, self.plant.world_frame(), body.body_frame())
            return X.rotation().matrix(), X.translation()

        results = []
        for pulley, A_body, B_body, label, color in wraps:
            if A_body is None or B_body is None:
                continue
            R_wb, t_wb = _Xw(pulley.body_name)
            center_w = R_wb @ np.asarray(pulley.centroid) + t_wb
            shaft_w  = R_wb @ pulley.shaft_axis_body
            A_w = R_wb @ np.asarray(A_body) + t_wb
            B_w = R_wb @ np.asarray(B_body) + t_wb
            ax = shaft_w / np.linalg.norm(shaft_w)
            dA = A_w - center_w
            dA -= np.dot(dA, ax) * ax
            dA /= np.linalg.norm(dA)
            dB = B_w - center_w
            dB -= np.dot(dB, ax) * ax
            dB /= np.linalg.norm(dB)
            cos_ab = float(np.clip(np.dot(dA, dB), -1.0, 1.0))
            angle = np.sign(np.dot(np.cross(dA, dB), ax)) * np.arccos(cos_ab)
            ax_cross_dA = np.cross(ax, dA)
            arc_pts = np.array([
                center_w + pulley.radius * (dA * np.cos(th) + ax_cross_dA * np.sin(th))
                for th in np.linspace(0.0, angle, n_arc_pts)
            ])
            results.append((label, color, arc_pts))
        return results


if __name__ == "__main__":
    main()