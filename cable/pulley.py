"""
cable/pulley.py

Cable routing pulley hierarchy and URDF origin parser.

Classes: PulleyBase (abstract), CableStartPointR/L, DrivePulley,
         IdlerL, IdlerR, BigPulley, CableEndPointL/R.

Functions: _parse_urdf_part_origins()
"""

import re
import numpy as np
from pathlib import Path
from termcolor import colored


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

