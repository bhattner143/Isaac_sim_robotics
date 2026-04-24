#!/usr/bin/env python3
"""
cable_with_exo_springs.py
─────────────────────────
Extends cable.py with two exosuit spring-cable routes for the
manipulator_cable_exo_springs URDF.

On top of the existing belt-drive routing (green/red cables via
DrivePulley → Idler → BigPulley), this adds:

  Exo RIGHT cable (−Y side, orange):
    ExoStartRight  →  ExoLink1PulleyRight  →  ExoElbowPulleyRight  →  ExoEndRight

  Exo LEFT cable (+Y side, magenta):
    ExoStartLeft  →  ExoLink1PulleyLeft  →  ExoElbowPulleyLeft  →  ExoEndLeft

Two cable-wrap routing modes are supported via ``ExoRouting``:

  CW_CCW  — link1 pulley CW, internal→external tangent, elbow pulley CCW
  CCW_CW  — link1 pulley CCW, external→internal tangent, elbow pulley CW

All classes imported from cable.cable (PulleyBase, CableRig, CupManipulator, …)
operate on the exo-springs URDF by setting ``PulleyBase._urdf_origins`` and
``PulleyBase.assets_dir`` at runtime via ``CupManipulator.init_cable_rig()``.

Interactive: type  q1 q2 [deg]  at the prompt → manipulator moves + cable redraws.

Usage:
    python cable/cable_with_exo_springs.py
    python cable/cable_with_exo_springs.py --no-springs
    python cable/cable_with_exo_springs.py --routing ccw_cw
"""

import enum
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root

from pydrake.all import (
    MeshcatVisualizer,
    StartMeshcat,
    Simulator,
    RigidTransform,
    RotationMatrix,
)
from pydrake.geometry import Rgba
from termcolor import colored

# ── Import shared classes from  cable.cable  ─────────────────────────────────
from cable.cable import (
    _parse_urdf_part_origins,
    PulleyBase,
    CableRoute,
    CableSpring,
    FixedBodyPoint,
    CableRig,
    CupManipulator,
    create_cable_manipulator_config,
    build_plant,
    DrakeCablePlant,
    spring_zigzag_points,
)
from project_utils.viz_cables import (
    print_cable_routing_points,
    draw_cables,
    visualize_cable_routing_top_view,
    visualize_cable_routing_3d,
)

# ─── Exo-springs URDF ────────────────────────────────────────────────────────
_EXO_URDF = ("model_using_onshape_to_robot/manipulator_cable_exo_springs/"
             "manipulator_cable_exo_springs_obj.urdf")
_Q1_EXO = "pulley_htd_5m_60t"
_Q2_EXO = "link2_tendon"


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry helper
# ═══════════════════════════════════════════════════════════════════════════════

def _radial_project(center, pt, target_r):
    """Scale *pt* radially outward from *center* to radius *target_r* (XY only)."""
    dx, dy = pt[0] - center[0], pt[1] - center[1]
    d = np.hypot(dx, dy)
    if d < 1e-12:
        return np.array(pt, float)
    s = target_r / d
    return np.array([center[0] + dx * s, center[1] + dy * s, pt[2]], float)


# ═══════════════════════════════════════════════════════════════════════════════
# EXOSUIT CABLE CLASSES — spring-cable waypoints on manipulator_cable_exo_springs
# ═══════════════════════════════════════════════════════════════════════════════
# Part positions are in the pulley_htd_5m_60t body frame (q1) or link2_tendon (q2).
# All parts are fixed positional waypoints; radius = 0 for anchor balls.

class ExoStartRight(PulleyBase):
    """Right exo cable start anchor on q1 body (−Y side).

    URDF: Part simple_ball_spring_cable_start on pulley_htd_5m_60t
      xyz=(-0.0238116, -0.108, 0.235049)  rpy≈(0, 0, 0)
      Mesh centroid offset: (0, +0.02, 0) in body frame.
    Zero radius — fixed anchor ball, not a wrapping pulley.
    """
    obj_name       = "simple_ball.obj"
    body_name      = _Q1_EXO
    urdf_part_name = "simple_ball_spring_cable_start"
    vis_xyz        = (-0.0238116, -0.088,  0.235049)   # ball center (origin + mesh offset)
    vis_rpy        = (0.0,         0.0,    0.0)
    face_color     = "#ff8800"     # orange — right exo cable
    label          = "Exo Start R"
    mesh_alpha     = 0.85

    def __init__(self):
        super().__init__()
        # Use class-level vis_xyz (corrected ball center), NOT self.vis_xyz
        # which PulleyBase.__init__ overrides to raw URDF origin.
        self._centroid_cache = np.array(type(self).vis_xyz, dtype=float)
        self._radius_cache   = 0.0

    def _compute_radius(self) -> float:
        return 0.0


class ExoStartLeft(PulleyBase):
    """Left exo cable start anchor on q1 body (+Y side).

    URDF: Part simple_ball_spring_cable_start_2 on pulley_htd_5m_60t
      xyz=(-0.0238116, 0.068, 0.235049)  rpy≈(0, 0, 0)
      Mesh centroid offset: (0, +0.02, 0) in body frame.
    Zero radius — fixed anchor ball.
    """
    obj_name       = "simple_ball.obj"
    body_name      = _Q1_EXO
    urdf_part_name = "simple_ball_spring_cable_start_2"
    vis_xyz        = (-0.0238116,  0.088,  0.235049)   # ball center (origin + mesh offset)
    vis_rpy        = (0.0,         0.0,    0.0)
    face_color     = "#cc00cc"     # magenta — left exo cable
    label          = "Exo Start L"
    mesh_alpha     = 0.85

    def __init__(self):
        super().__init__()
        self._centroid_cache = np.array(type(self).vis_xyz, dtype=float)
        self._radius_cache   = 0.0

    def _compute_radius(self) -> float:
        return 0.0


class ExoLink1PulleyRight(PulleyBase):
    """Right exo link-1 spring cable pulley on q1 body (−Y side).

    URDF: Part link1_spring_cable_pulley on pulley_htd_5m_60t
      xyz=(0.0141884, -0.088, 0.0990487)  rpy≈(0, 0, 0)
    Radius = 0.041 m (outer mesh rim — visual/drawing radius).
    Tangent radius = 0.035 m (groove floor — used for tangent computation
    because Start ball is only 0.038 m away and sits inside the outer rim).
    After computing tangent points at groove radius, they are projected
    radially to the outer rim so cable arcs are visible outside the mesh.
    Cable routing Z overridden to 0.235049 (cable plane).
    """
    obj_name       = "link1_spring_cable_pulley.obj"
    body_name      = _Q1_EXO
    urdf_part_name = "link1_spring_cable_pulley"
    vis_xyz        = (0.0141884, -0.088,  0.235049)   # Z = cable plane, not pulley centre
    vis_rpy        = (0.0,        0.0,    0.0)
    face_color     = "#ff6600"     # darker orange
    label          = "Exo Link1 Pulley R"
    mesh_alpha     = 0.55
    tangent_radius = 0.035          # groove floor — for tangent geometry

    def __init__(self):
        super().__init__()
        self._centroid_cache = np.array(type(self).vis_xyz, dtype=float)
        self._radius_cache   = 0.041   # outer mesh rim — for drawing

    def _compute_radius(self) -> float:
        return 0.041


class ExoLink1PulleyLeft(PulleyBase):
    """Left exo link-1 spring cable pulley on q1 body (+Y side).

    URDF: Part link1_spring_cable_pulley_2 on pulley_htd_5m_60t
      xyz=(0.0141884, 0.088, 0.0990487)  rpy≈(0, 0, 0)
    Radius = 0.041 m (outer mesh rim — visual/drawing radius).
    Tangent radius = 0.035 m (groove floor — same as Right side).
    Cable routing Z overridden to 0.235049 (cable plane).
    """
    obj_name       = "link1_spring_cable_pulley.obj"
    body_name      = _Q1_EXO
    urdf_part_name = "link1_spring_cable_pulley_2"
    vis_xyz        = (0.0141884,  0.088,  0.235049)   # Z = cable plane, not pulley centre
    vis_rpy        = (0.0,        0.0,    0.0)
    face_color     = "#aa00aa"     # darker magenta
    label          = "Exo Link1 Pulley L"
    mesh_alpha     = 0.55
    tangent_radius = 0.035          # groove floor — for tangent geometry

    def __init__(self):
        super().__init__()
        self._centroid_cache = np.array(type(self).vis_xyz, dtype=float)
        self._radius_cache   = 0.041   # outer mesh rim — for drawing

    def _compute_radius(self) -> float:
        return 0.041


class ExoElbowPulleyRight(PulleyBase):
    """Right exo elbow spring-cable pulley on q1 body (−Y side).

    URDF: Part pulley_springs_cables on pulley_htd_5m_60t
      xyz=(0.232043, -0.0375, 0.236049)  rpy≈(0, 0, 0)
    Radius = 0.032 m (mesh outer rim, same approach as cable.py).
    """
    obj_name       = "pulley_springs_cables.obj"
    body_name      = _Q1_EXO
    urdf_part_name = "pulley_springs_cables"
    vis_xyz        = (0.232043, -0.0375,  0.236049)
    vis_rpy        = (0.0,       0.0,     0.0)
    face_color     = "#ff4400"     # red-orange
    label          = "Exo Elbow Pulley R"
    mesh_alpha     = 0.55

    def __init__(self):
        super().__init__()
        self._centroid_cache = np.array(type(self).vis_xyz, dtype=float)
        self._radius_cache   = 0.032

    def _compute_radius(self) -> float:
        return 0.032


class ExoElbowPulleyLeft(PulleyBase):
    """Left exo elbow spring-cable pulley on q1 body (+Y side).

    URDF: Part pulley_springs_cables_2 on pulley_htd_5m_60t
      xyz=(0.232043, 0.0375, 0.235049)  rpy≈(0, 0, 0)
    Radius = 0.032 m (mesh outer rim, same approach as cable.py).
    """
    obj_name       = "pulley_springs_cables.obj"
    body_name      = _Q1_EXO
    urdf_part_name = "pulley_springs_cables_2"
    vis_xyz        = (0.232043,  0.0375,  0.235049)
    vis_rpy        = (0.0,       0.0,     0.0)
    face_color     = "#880088"     # purple
    label          = "Exo Elbow Pulley L"
    mesh_alpha     = 0.55

    def __init__(self):
        super().__init__()
        self._centroid_cache = np.array(type(self).vis_xyz, dtype=float)
        self._radius_cache   = 0.032

    def _compute_radius(self) -> float:
        return 0.032


class ExoEndRight(PulleyBase):
    """Right exo cable end anchor on link2 (−Y side).

    URDF: Part simple_ball_spring_cable_end_2 on link2_tendon
      xyz=(0.16, -0.018, 0.013)  rpy=(-π/2, 0, π/2)
      Mesh centroid offset after rotation: (-0.04, 0, 0) in body frame.
    Zero radius — fixed anchor ball.
    """
    obj_name       = "simple_ball.obj"
    body_name      = _Q2_EXO
    urdf_part_name = "simple_ball_spring_cable_end_2"
    vis_xyz        = (0.12, -0.018,  0.013)   # ball center (origin + rotated mesh offset)
    vis_rpy        = (-1.5708, 0.0,  1.5708)
    face_color     = "#ff8800"     # orange
    label          = "Exo End R"
    mesh_alpha     = 0.85

    def __init__(self):
        super().__init__()
        self._centroid_cache = np.array(type(self).vis_xyz, dtype=float)
        self._radius_cache   = 0.0

    def _compute_radius(self) -> float:
        return 0.0


class ExoEndLeft(PulleyBase):
    """Left exo cable end anchor on link2 (+Y side).

    URDF: Part simple_ball_spring_cable_end on link2_tendon
      xyz=(0.16, 0.018, 0.013)  rpy=(-π/2, 0, π/2)
      Mesh centroid offset after rotation: (-0.04, 0, 0) in body frame.
    Zero radius — fixed anchor ball.
    """
    obj_name       = "simple_ball.obj"
    body_name      = _Q2_EXO
    urdf_part_name = "simple_ball_spring_cable_end"
    vis_xyz        = (0.12,  0.018,  0.013)   # ball center (origin + rotated mesh offset)
    vis_rpy        = (-1.5708, 0.0,  1.5708)
    face_color     = "#cc00cc"     # magenta
    label          = "Exo End L"
    mesh_alpha     = 0.85

    def __init__(self):
        super().__init__()
        self._centroid_cache = np.array(type(self).vis_xyz, dtype=float)
        self._radius_cache   = 0.0

    def _compute_radius(self) -> float:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Routing mode enum
# ═══════════════════════════════════════════════════════════════════════════════

class ExoRouting(enum.Enum):
    """Cable-wrap mode for exosuit spring cables.

    CW_CCW
        Right cable: Start → **CW** wrap on Link1 Pulley →
        *internal* tangent to *external* tangent on Elbow Pulley →
        **CCW** wrap → End.  Left cable is symmetric (mirrored Y).

    CCW_CW
        Right cable: Start → **CCW** wrap on Link1 Pulley →
        *external* tangent to *internal* tangent on Elbow Pulley →
        **CW** wrap → End.  Left cable is symmetric (mirrored Y).
    """
    CW_CCW  = "cw_ccw"
    CCW_CW  = "ccw_cw"


# ═══════════════════════════════════════════════════════════════════════════════
# ExoCableRig — two exosuit spring cables
# ═══════════════════════════════════════════════════════════════════════════════

class ExoCableRig:
    """Exosuit cable routing: two spring cables acting on the elbow joint.

    Mirrors :class:`CableRig` structure: stores pulley instances as named
    attributes, a ``waypoints`` list, and a ``compute_tangents()`` method.

    Each route has **6 waypoints** when tangent routing is active::

      ①  Start anchor (zero-radius)
      ②  Tangent entry onto Link1 Pulley  (A)
      ③  Tangent exit  off  Link1 Pulley  (B)
      ④  Tangent entry onto Elbow Pulley  (A)
      ⑤  Tangent exit  off  Elbow Pulley  (B)
      ⑥  End   anchor (zero-radius)

    Segments ②→③ and ④→⑤ are pulley-wrap arcs (drawn separately);
    their chords are skipped via ``skip_chord_segments``.

    Parameters
    ----------
    routing : ExoRouting
        Selects wrapping direction & tangent kind (default CW_CCW).
    """

    def __init__(self, routing: ExoRouting = ExoRouting.CW_CCW,
                 springs_enabled: bool = True) -> None:
        self.routing = routing
        self.springs_enabled = springs_enabled

        # ── Pulley / anchor instances ─────────────────────────────────────────
        self.exo_start_r        = ExoStartRight()
        self.exo_link1_pulley_r = ExoLink1PulleyRight()
        self.exo_elbow_pulley_r = ExoElbowPulleyRight()
        self.exo_end_r          = ExoEndRight()

        self.exo_start_l        = ExoStartLeft()
        self.exo_link1_pulley_l = ExoLink1PulleyLeft()
        self.exo_elbow_pulley_l = ExoElbowPulleyLeft()
        self.exo_end_l          = ExoEndLeft()

        # ── FixedBodyPoint wrappers — tangent contact points ─────────────────
        # Start / End anchors are pre-set from vis_xyz (zero-radius);
        # the pulley entry/exit points (A/B) are zero placeholders, filled by
        # compute_tangents().
        self.wp_start_r = FixedBodyPoint(_Q1_EXO, np.array(ExoStartRight.vis_xyz), "Exo Start R")
        self.wp_end_r   = FixedBodyPoint(_Q2_EXO, np.array(ExoEndRight.vis_xyz),   "Exo End R")
        self.wp_start_l = FixedBodyPoint(_Q1_EXO, np.array(ExoStartLeft.vis_xyz),  "Exo Start L")
        self.wp_end_l   = FixedBodyPoint(_Q2_EXO, np.array(ExoEndLeft.vis_xyz),    "Exo End L")

        # Link1 pulley tangent contacts
        self.wp_link1_A_r = FixedBodyPoint(_Q1_EXO, np.zeros(3), "Exo Link1 Pulley R (A)")
        self.wp_link1_B_r = FixedBodyPoint(_Q1_EXO, np.zeros(3), "Exo Link1 Pulley R (B)")
        self.wp_link1_A_l = FixedBodyPoint(_Q1_EXO, np.zeros(3), "Exo Link1 Pulley L (A)")
        self.wp_link1_B_l = FixedBodyPoint(_Q1_EXO, np.zeros(3), "Exo Link1 Pulley L (B)")

        # Elbow pulley tangent contacts
        self.wp_elbow_A_r = FixedBodyPoint(_Q1_EXO, np.zeros(3), "Exo Elbow Pulley R (A)")
        self.wp_elbow_B_r = FixedBodyPoint(_Q1_EXO, np.zeros(3), "Exo Elbow Pulley R (B)")
        self.wp_elbow_A_l = FixedBodyPoint(_Q1_EXO, np.zeros(3), "Exo Elbow Pulley L (A)")
        self.wp_elbow_B_l = FixedBodyPoint(_Q1_EXO, np.zeros(3), "Exo Elbow Pulley L (B)")

        # ── Two exo cable routes ──────────────────────────────────────────────
        # Segments: Start → A_link1 → B_link1 → A_elbow → B_elbow → End
        # skip_chord_segments {1, 3}: chord A→B on each pulley is an arc.
        self.exo_cable_right = CableRoute(
            segments=[
                (self.wp_start_r,   np.zeros(3)),  # ① start anchor
                (self.wp_link1_A_r, np.zeros(3)),  # ② link1 entry
                (self.wp_link1_B_r, np.zeros(3)),  # ③ link1 exit
                (self.wp_elbow_A_r, np.zeros(3)),  # ④ elbow entry
                (self.wp_elbow_B_r, np.zeros(3)),  # ⑤ elbow exit
                (self.wp_end_r,     np.zeros(3)),  # ⑥ end anchor
            ],
            meshcat_path        = "/exo_cable/right",
            meshcat_color       = Rgba(1.0, 0.55, 0.0, 1.0),   # orange
            mpl_color           = "darkorange",
            label               = "Exo Right",
            skip_chord_segments = frozenset({1, 3}),
        )
        self.exo_cable_left = CableRoute(
            segments=[
                (self.wp_start_l,   np.zeros(3)),  # ① start anchor
                (self.wp_link1_A_l, np.zeros(3)),  # ② link1 entry
                (self.wp_link1_B_l, np.zeros(3)),  # ③ link1 exit
                (self.wp_elbow_A_l, np.zeros(3)),  # ④ elbow entry
                (self.wp_elbow_B_l, np.zeros(3)),  # ⑤ elbow exit
                (self.wp_end_l,     np.zeros(3)),  # ⑥ end anchor
            ],
            meshcat_path        = "/exo_cable/left",
            meshcat_color       = Rgba(0.8, 0.0, 0.8, 1.0),   # magenta
            mpl_color           = "mediumorchid",
            label               = "Exo Left",
            skip_chord_segments = frozenset({1, 3}),
        )

        self.routes = [self.exo_cable_right, self.exo_cable_left]

        # Expose all pulleys that have geometry (for visualization)
        self.waypoints = [
            self.exo_start_r,  self.exo_link1_pulley_r,
            self.exo_elbow_pulley_r, self.exo_end_r,
            self.exo_start_l,  self.exo_link1_pulley_l,
            self.exo_elbow_pulley_l, self.exo_end_l,
        ]

        # Wrap arc descriptors — filled after compute_tangents()
        # Each entry: (pulley_instance, A_body, B_body)
        self.wrap_arcs: list[tuple] = []

        # ── Optional springs at exo cable endpoints (elbow exit → end) ────────
        # Compact rest pose: short fraction so coils are tight (not stretched).
        # spring_fraction will grow when cable is pulled (spring_extension > 0).
        self.spring_R = CableSpring(label="Exo Spring R",
                                    enabled=springs_enabled,
                                    spring_fraction=0.12,
                                    n_coils=8,
                                    amplitude=0.005,
                                    spring_position=0.50)
        self.spring_L = CableSpring(label="Exo Spring L",
                                    enabled=springs_enabled,
                                    spring_fraction=0.12,
                                    n_coils=8,
                                    amplitude=0.005,
                                    spring_position=0.50)

    # ── Tangent computation ───────────────────────────────────────────────────

    def compute_tangents(self, plant, plant_context, manipulator) -> None:
        """Compute tangent entry/exit points on link1 and elbow pulleys.

        Start, Link1Pulley, and ElbowPulley all live on q1 body — their
        body-frame coordinates can be used directly with compute_tangent().
        ElbowPulley→End crosses from q1 to q2, so we transform End's position
        into q1's frame via FK for the tangent solve, then keep the End
        FixedBodyPoint in its native q2 frame (radius = 0 ⇒ T2 = centre).

        Routing topology (right cable, CW_CCW):
          Start (r=0) → CW on Link1Pulley: branch=-1, ext
          Link1Pulley → ElbowPulley:       branch=-1, int (internal)
          ElbowPulley → End (r=0):         branch=+1, ext  (CCW exit)

        CCW_CW flips all branches and swaps internal↔external.
        Left cable mirrors by negating branch signs.
        """
        ct = PulleyBase.compute_tangent

        if self.routing == ExoRouting.CW_CCW:
            #       right cable                  left cable (mirror)
            b_start_link1_R, b_start_link1_L = -1, +1   # CW right, CCW left
            kind_link1_elbow                  = "internal"
            b_link1_elbow_R, b_link1_elbow_L  = -1, +1
            b_elbow_end_R,   b_elbow_end_L    = +1, -1   # CCW right, CW left
        else:  # CCW_CW
            b_start_link1_R, b_start_link1_L = -1, +1   # same entry as CW_CCW
            kind_link1_elbow                  = "internal"
            b_link1_elbow_R, b_link1_elbow_L  = +1, -1   # flipped → CCW right, CW left
            b_elbow_end_R,   b_elbow_end_L    = -1, +1   # flipped → CW right, CCW left

        r_link1_tangent = ExoLink1PulleyRight.tangent_radius   # groove floor for ct()
        r_link1_draw    = self.exo_link1_pulley_r.radius       # outer rim for drawing
        r_elbow = self.exo_elbow_pulley_r.radius

        # FK: transform End points (q2 body) into q1 body frame for tangent solve
        body_q1 = plant.GetBodyByName(_Q1_EXO, manipulator.model_instance)
        body_q2 = plant.GetBodyByName(_Q2_EXO, manipulator.model_instance)
        X_q1_q2 = plant.CalcRelativeTransform(
            plant_context, body_q1.body_frame(), body_q2.body_frame())
        R12, t12 = X_q1_q2.rotation().matrix(), X_q1_q2.translation()

        c_end_r_in_q1 = R12 @ np.array(ExoEndRight.vis_xyz, float) + t12
        c_end_l_in_q1 = R12 @ np.array(ExoEndLeft.vis_xyz,  float) + t12

        # --- Right cable (all in q1 body frame) ---
        c_start_r = np.array(ExoStartRight.vis_xyz, float)
        c_link1_r = np.array(ExoLink1PulleyRight.vis_xyz, float)
        c_elbow_r = np.array(ExoElbowPulleyRight.vis_xyz, float)

        # Start (r=0) → Link1 Pulley (tangent at groove radius)
        _, A_link1_r = ct(c_start_r, 0.0, c_link1_r, r_link1_tangent,
                          branch=b_start_link1_R, kind="external")
        # Link1 Pulley → Elbow Pulley (tangent at groove radius)
        B_link1_r, A_elbow_r = ct(c_link1_r, r_link1_tangent, c_elbow_r, r_elbow,
                                   branch=b_link1_elbow_R, kind=kind_link1_elbow)
        # Project Link1 tangent points from groove to outer rim
        A_link1_r = _radial_project(c_link1_r, A_link1_r, r_link1_draw)
        B_link1_r = _radial_project(c_link1_r, B_link1_r, r_link1_draw)
        # Elbow Pulley → End (r=0, End position in q1 frame)
        B_elbow_r, _ = ct(c_elbow_r, r_elbow, c_end_r_in_q1, 0.0,
                          branch=b_elbow_end_R, kind="external")

        # --- Left cable (all in q1 body frame) ---
        c_start_l = np.array(ExoStartLeft.vis_xyz, float)
        c_link1_l = np.array(ExoLink1PulleyLeft.vis_xyz, float)
        c_elbow_l = np.array(ExoElbowPulleyLeft.vis_xyz, float)

        _, A_link1_l = ct(c_start_l, 0.0, c_link1_l, r_link1_tangent,
                          branch=b_start_link1_L, kind="external")
        B_link1_l, A_elbow_l = ct(c_link1_l, r_link1_tangent, c_elbow_l, r_elbow,
                                   branch=b_link1_elbow_L, kind=kind_link1_elbow)
        # Project Link1 tangent points from groove to outer rim
        A_link1_l = _radial_project(c_link1_l, A_link1_l, r_link1_draw)
        B_link1_l = _radial_project(c_link1_l, B_link1_l, r_link1_draw)
        B_elbow_l, _ = ct(c_elbow_l, r_elbow, c_end_l_in_q1, 0.0,
                          branch=b_elbow_end_L, kind="external")

        # --- Propagate to FixedBodyPoint wrappers ---
        self.wp_link1_A_r._tangent_point = A_link1_r
        self.wp_link1_B_r._tangent_point = B_link1_r
        self.wp_elbow_A_r._tangent_point = A_elbow_r
        self.wp_elbow_B_r._tangent_point = B_elbow_r

        self.wp_link1_A_l._tangent_point = A_link1_l
        self.wp_link1_B_l._tangent_point = B_link1_l
        self.wp_elbow_A_l._tangent_point = A_elbow_l
        self.wp_elbow_B_l._tangent_point = B_elbow_l

        # --- Store tangent points on pulley instances for wrap-arc drawing ---
        self.exo_link1_pulley_r.A_R = A_link1_r
        self.exo_link1_pulley_r.B_R = B_link1_r
        self.exo_elbow_pulley_r.A_R = A_elbow_r
        self.exo_elbow_pulley_r.B_R = B_elbow_r

        self.exo_link1_pulley_l.A_L = A_link1_l
        self.exo_link1_pulley_l.B_L = B_link1_l
        self.exo_elbow_pulley_l.A_L = A_elbow_l
        self.exo_elbow_pulley_l.B_L = B_elbow_l

        # Wrap arcs for draw_exo_cables
        self.wrap_arcs = [
            (self.exo_link1_pulley_r, A_link1_r, B_link1_r,
             "/exo_cable/right/wrap/link1", Rgba(1.0, 0.55, 0.0, 1.0)),
            (self.exo_elbow_pulley_r, A_elbow_r, B_elbow_r,
             "/exo_cable/right/wrap/elbow", Rgba(1.0, 0.55, 0.0, 1.0)),
            (self.exo_link1_pulley_l, A_link1_l, B_link1_l,
             "/exo_cable/left/wrap/link1",  Rgba(0.8, 0.0, 0.8, 1.0)),
            (self.exo_elbow_pulley_l, A_elbow_l, B_elbow_l,
             "/exo_cable/left/wrap/elbow",  Rgba(0.8, 0.0, 0.8, 1.0)),
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# Exo cable visualization helpers
# ═══════════════════════════════════════════════════════════════════════════════

def print_exo_cable_routing_points(plant, plant_context, manipulator,
                                   exo_rig: ExoCableRig) -> None:
    """Print all exo cable waypoints in world frame — mirrors print_cable_routing_points."""
    print("\n" + "=" * 64)
    print("  Exo-suit cable routing points (world frame)")
    print("=" * 64)
    for route in exo_rig.routes:
        pts = route.world_points(plant, plant_context, manipulator)  # (N, 3)
        print(f"\n  [{route.label}]")
        seg_names = [cfg.label for cfg, _ in route.segments]
        col_w = max(len(n) for n in seg_names) + 2
        print(f"  {'Waypoint':<{col_w}}  {'x':>10}  {'y':>10}  {'z':>10}  (m)")
        print(f"  {'-' * col_w}  {'----------'}  {'----------'}  {'----------'}")
        for name, pt in zip(seg_names, pts):
            print(f"  {name:<{col_w}}  {pt[0]:>10.6f}  {pt[1]:>10.6f}  {pt[2]:>10.6f}")
    print("=" * 64 + "\n")


def visualize_exo_cable_routing_top_view(
    plant, plant_context, manipulator, exo_rig: ExoCableRig,
    q1_deg: float = 0.0, q2_deg: float = 0.0,
) -> tuple:
    """Top-view (XY plane, looking down Z) schematic of exosuit cable routing.

    Same style as visualize_cable_routing_top_view (Figure 1):
      - Filled circles for pulleys (scaled to actual radius)
      - Small filled dots for zero-radius anchor balls
      - Cable polylines connecting waypoints
      - Joint-frame origin crosses (J1, J2)
      - Labels on every waypoint

    Returns (fig, ax).
    """

    def _body_pt_world(body_name, p_body):
        body = plant.GetBodyByName(body_name, manipulator.model_instance)
        return plant.CalcPointsPositions(
            plant_context,
            body.body_frame(),
            np.array(p_body, float).reshape(3, 1),
            plant.world_frame(),
        ).flatten()

    def _Xw(body_name):
        body = plant.GetBodyByName(body_name, manipulator.model_instance)
        X    = plant.CalcRelativeTransform(plant_context, plant.world_frame(),
                                           body.body_frame())
        return X.rotation().matrix(), X.translation()

    fig, ax = plt.subplots(figsize=(10, 7),
                           num="Exo Cable — Top View (XY)")
    ax.set_title(
        f"Exo Cable Routing — Top View (XY plane)   "
        f"q1 = {q1_deg:.1f}°   q2 = {q2_deg:.1f}°",
        fontsize=11,
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── Pulley circles & anchor dots ─────────────────────────────────────────
    for cfg in exo_rig.waypoints:
        wx, wy, _ = _body_pt_world(cfg.body_name, cfg.centroid)
        if cfg.radius > 0.001:
            # Real pulley — draw filled circle
            circle = plt.Circle(
                (wx, wy), cfg.radius,
                color=cfg.face_color, fill=True,
                alpha=max(cfg.mesh_alpha, 0.35),
                linewidth=1.5, edgecolor='k', zorder=3,
            )
            ax.add_patch(circle)
            ax.text(wx, wy, cfg.label, fontsize=6, ha='center', va='center',
                    color='k', fontweight='bold', zorder=5)
        else:
            # Zero-radius anchor ball — small filled dot
            ax.plot(wx, wy, 'o', color=cfg.face_color, markersize=8,
                    markeredgecolor='k', markeredgewidth=0.8, zorder=4)
            ax.text(wx, wy + 0.004, cfg.label, fontsize=6.5,
                    ha='center', va='bottom',
                    color=cfg.face_color, fontweight='bold', zorder=5)

    # ── Cable straight segments + wrap arcs ─────────────────────────────────
    springs_on = getattr(exo_rig, "springs_enabled", False)

    for ri, route in enumerate(exo_rig.routes):
        pts = route.world_points(plant, plant_context, manipulator)  # (N, 3)
        skip = getattr(route, "skip_chord_segments", frozenset())
        last_seg_idx = len(pts) - 2
        # Straight chord segments
        for i, (p0, p1) in enumerate(zip(pts[:-1], pts[1:])):
            if i in skip:
                continue
            if springs_on and i == last_seg_idx:
                spring = exo_rig.spring_R if ri == 0 else exo_rig.spring_L
                if spring.enabled:
                    sf = spring.spring_fraction
                    sp = np.clip(spring.spring_position, sf / 2, 1.0 - sf / 2)
                    t0 = sp - sf / 2
                    t1 = sp + sf / 2
                    p_ss = p1 + t0 * (p0 - p1)  # spring start (near end)
                    p_se = p1 + t1 * (p0 - p1)  # spring end (near pulley)
                    # Cable segments either side
                    ax.plot([p0[0], p_se[0]], [p0[1], p_se[1]], '-',
                            color=route.mpl_color, linewidth=2, zorder=6)
                    ax.plot([p_ss[0], p1[0]], [p_ss[1], p1[1]], '-',
                            color=route.mpl_color, linewidth=2, zorder=6)
                    # Zigzag spring symbol (2-D)
                    zz = spring_zigzag_points(p_se, p_ss,
                                              n_coils=spring.n_coils,
                                              amplitude=spring.amplitude)
                    ax.plot(zz[:, 0], zz[:, 1], '-', color='goldenrod',
                            linewidth=2, zorder=7)
                    continue
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], '-', color=route.mpl_color,
                    linewidth=2, zorder=6)
        # Tangent contact dots
        for pt in pts:
            ax.plot(pt[0], pt[1], 'o', color=route.mpl_color, markersize=4, zorder=6)
        # Legend entry
        ax.plot([], [], '-o', color=route.mpl_color, linewidth=2, markersize=4,
                label=f"{route.label} ({exo_rig.routing.value})")

    # ── Wrap arcs on pulleys (matplotlib) ─────────────────────────────────────
    for pulley, A_body, B_body, _, _ in exo_rig.wrap_arcs:
        # Transform A, B, centre into world frame
        R_wb, t_wb = _Xw(pulley.body_name)
        c_w  = R_wb @ np.asarray(pulley.centroid) + t_wb
        A_w  = R_wb @ np.asarray(A_body) + t_wb
        B_w  = R_wb @ np.asarray(B_body) + t_wb
        shaft_w = R_wb @ pulley.shaft_axis_body

        # Project to plane perp to shaft
        ax_s = shaft_w / np.linalg.norm(shaft_w)
        dA = A_w - c_w;  dA -= np.dot(dA, ax_s) * ax_s;  dA /= np.linalg.norm(dA)
        dB = B_w - c_w;  dB -= np.dot(dB, ax_s) * ax_s;  dB /= np.linalg.norm(dB)
        cos_ab = float(np.clip(np.dot(dA, dB), -1.0, 1.0))
        angle = np.sign(np.dot(np.cross(dA, dB), ax_s)) * np.arccos(cos_ab)
        ax_cross_dA = np.cross(ax_s, dA)
        arc_pts = np.array([
            c_w + pulley.radius * (dA * np.cos(th) + ax_cross_dA * np.sin(th))
            for th in np.linspace(0.0, angle, 40)
        ])
        mpl_col = pulley.face_color
        ax.plot(arc_pts[:, 0], arc_pts[:, 1], '-', color=mpl_col,
                linewidth=2.5, alpha=0.8, zorder=6)

    # ── Body-frame origin crosses ────────────────────────────────────────────
    for body_name, color, lbl in [
        (_Q1_EXO, "royalblue",  "J1"),
        (_Q2_EXO, "seagreen",   "J2"),
    ]:
        ox, oy, _ = _body_pt_world(body_name, np.zeros(3))
        ax.plot(ox, oy, '+', color=color, markersize=10,
                markeredgewidth=2, zorder=8)
        ax.text(ox, oy, f"  {lbl}", fontsize=8, color=color,
                fontweight='bold', zorder=8)

    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig, ax


def draw_exo_cables(meshcat, plant, plant_context, manipulator,
                    exo_rig: ExoCableRig, cable_radius: float = 0.0008,
                    n_arc_pts: int = 32) -> None:
    """Draw exo cable routes as straight cylinders + wrap arcs in Meshcat."""
    from pydrake.geometry import Cylinder as _Cyl

    def _body_pt_world(body_name, p_body):
        body = plant.GetBodyByName(body_name, manipulator.model_instance)
        return plant.CalcPointsPositions(
            plant_context,
            body.body_frame(),
            np.array(p_body, float).reshape(3, 1),
            plant.world_frame(),
        ).flatten()

    def _Xw(body_name):
        body = plant.GetBodyByName(body_name, manipulator.model_instance)
        X    = plant.CalcRelativeTransform(plant_context, plant.world_frame(),
                                           body.body_frame())
        return X.rotation().matrix(), X.translation()

    def _place_seg(path, p0, p1, rgba):
        diff   = p1 - p0
        length = float(np.linalg.norm(diff))
        if length < 1e-9:
            return
        mid   = (p0 + p1) * 0.5
        z_hat = diff / length
        tmp   = np.array([0., 1., 0.]) if abs(z_hat[0]) > 0.9 else np.array([1., 0., 0.])
        x_hat = np.cross(tmp, z_hat);  x_hat /= np.linalg.norm(x_hat)
        y_hat = np.cross(z_hat, x_hat)
        R_mat = RotationMatrix(np.column_stack([x_hat, y_hat, z_hat]))
        meshcat.SetObject(path, _Cyl(cable_radius, length), rgba)
        meshcat.SetTransform(path, RigidTransform(R_mat, mid))

    def _arc_world_pts(center_w, radius, shaft_w, A_w, B_w):
        """Arc points from A to B around center, in the plane perp to shaft."""
        ax = shaft_w / np.linalg.norm(shaft_w)
        dA = A_w - center_w;  dA -= np.dot(dA, ax) * ax;  dA /= np.linalg.norm(dA)
        dB = B_w - center_w;  dB -= np.dot(dB, ax) * ax;  dB /= np.linalg.norm(dB)
        cos_ab = float(np.clip(np.dot(dA, dB), -1.0, 1.0))
        angle  = np.sign(np.dot(np.cross(dA, dB), ax)) * np.arccos(cos_ab)
        ax_cross_dA = np.cross(ax, dA)
        return np.array([
            center_w + radius * (dA * np.cos(th) + ax_cross_dA * np.sin(th))
            for th in np.linspace(0.0, angle, n_arc_pts)
        ])

    # ── Straight cable segments (skipping wrap chords) ────────────────────────
    springs_on = getattr(exo_rig, "springs_enabled", False)

    for ri, route in enumerate(exo_rig.routes):
        pts = np.array([
            _body_pt_world(cfg.body_name, cfg._tangent_point)
            for cfg, _ in route.segments
        ])
        skip = getattr(route, "skip_chord_segments", frozenset())
        last_seg_idx = len(pts) - 2  # elbow exit → end anchor
        for i, (p0, p1) in enumerate(zip(pts[:-1], pts[1:])):
            if i in skip:
                continue
            # Spring visualization on the last segment (elbow exit → end)
            if springs_on and i == last_seg_idx:
                spring = exo_rig.spring_R if ri == 0 else exo_rig.spring_L
                if spring.enabled:
                    seg_len = float(np.linalg.norm(p0 - p1))
                    sf = spring.spring_fraction
                    sp = np.clip(spring.spring_position, sf / 2, 1.0 - sf / 2)
                    t0 = sp - sf / 2
                    t1 = sp + sf / 2
                    # p0 = elbow exit (far end), p1 = endpoint (near end)
                    # t measured from p1 toward p0
                    p_spring_start = p1 + t0 * (p0 - p1)
                    p_spring_end   = p1 + t1 * (p0 - p1)
                    # Cable: elbow exit → spring end
                    _place_seg(f"{route.meshcat_path}/seg{i:02d}_a",
                               p0, p_spring_end, route.meshcat_color)
                    # Helical spring (gold/orange)
                    zz = spring_zigzag_points(p_spring_end, p_spring_start,
                                              n_coils=spring.n_coils,
                                              amplitude=spring.amplitude)
                    spring_rgba = Rgba(0.9, 0.6, 0.0, 1.0)
                    for j, (z0, z1) in enumerate(zip(zz[:-1], zz[1:])):
                        _place_seg(f"{route.meshcat_path}/spring{j:02d}",
                                   z0, z1, spring_rgba)
                    # Cable: spring start → endpoint
                    _place_seg(f"{route.meshcat_path}/seg{i:02d}_b",
                               p_spring_start, p1, route.meshcat_color)
                    continue
            _place_seg(f"{route.meshcat_path}/seg{i:02d}", p0, p1, route.meshcat_color)

    # ── Wrap arcs on pulleys ──────────────────────────────────────────────────
    for pulley, A_body, B_body, path_prefix, rgba in exo_rig.wrap_arcs:
        R_wb, t_wb = _Xw(pulley.body_name)
        center_w   = R_wb @ np.asarray(pulley.centroid) + t_wb
        shaft_w    = R_wb @ pulley.shaft_axis_body
        A_w        = R_wb @ np.asarray(A_body) + t_wb
        B_w        = R_wb @ np.asarray(B_body) + t_wb
        arc_pts    = _arc_world_pts(center_w, pulley.radius, shaft_w, A_w, B_w)
        for i, (p0, p1) in enumerate(zip(arc_pts[:-1], arc_pts[1:])):
            _place_seg(f"{path_prefix}/arc{i:02d}", p0, p1, rgba)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Exo-springs cable routing visualization.")
    ap.add_argument("--no-springs", action="store_true",
                    help="Disable drive-cable endpoint springs (default: springs enabled)")
    ap.add_argument("--no-exo-springs", action="store_true",
                    help="Disable exo-cable endpoint springs (default: springs enabled)")
    ap.add_argument("--routing", type=str, default="cw_ccw",
                    choices=["cw_ccw", "ccw_cw"],
                    help="Exo cable routing mode (default: cw_ccw)")
    args = ap.parse_args()
    springs_enabled = not args.no_springs
    exo_springs_enabled = not args.no_exo_springs
    routing = ExoRouting(args.routing)

    # ── Configuration (exo-springs URDF) ─────────────────────────────────────
    config = create_cable_manipulator_config(
        urdf_path=_EXO_URDF,
        joint_angles={"link1_base": 0.0, "link2_link1": 0.0},
        damping=(0.1, 0.1),
    )

    # ── Meshcat ───────────────────────────────────────────────────────────────
    meshcat = StartMeshcat()
    print(colored(f"\n🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))

    # ── Plant (imported build_plant uses CupManipulator from cable.cable) ────
    builder, plant, scene_graph, manipulator = build_plant(config)

    # ── Cable rigs ────────────────────────────────────────────────────────────
    # Drive-belt rig: PulleyBase._urdf_origins + assets_dir are set to the
    # exo-springs URDF by init_cable_rig(), so imported classes (DrivePulley,
    # IdlerL/R, BigPulley, etc.) all resolve from the new URDF.
    manipulator.init_cable_rig(springs_enabled=springs_enabled)
    rig     = manipulator.rig           # drive-belt rig (green/red cables)
    exo_rig = ExoCableRig(routing=routing,
                          springs_enabled=exo_springs_enabled)

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
    manipulator.compute_tangents(plant, plant_ctx)               # drive-cable FK tangents
    exo_rig.compute_tangents(plant, plant_ctx, manipulator)      # exo-cable tangents
    draw_cables(meshcat, plant, plant_ctx, manipulator, rig)     # drive cables in Meshcat
    draw_exo_cables(meshcat, plant, plant_ctx, manipulator, exo_rig)  # exo cables in Meshcat
    print_cable_routing_points(plant, plant_ctx, manipulator, rig)
    print_exo_cable_routing_points(plant, plant_ctx, manipulator, exo_rig)

    # Figure 1 — top view (XY) of drive cables
    _top_fig, _ = visualize_cable_routing_top_view(plant, plant_ctx, manipulator, 0.0, 0.0, rig)
    plt.show(block=False)
    plt.pause(0.05)

    # Exo cable top view (XY) — same style as Figure 1
    _exo_fig, _ = visualize_exo_cable_routing_top_view(plant, plant_ctx, manipulator, exo_rig, 0.0, 0.0)
    plt.show(block=False)
    plt.pause(0.05)

    _viz_fig = None  # 3-D view — created on first interactive update

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

                # Recompute all tangents in world frame
                manipulator.compute_tangents(plant, plant_ctx)
                exo_rig.compute_tangents(plant, plant_ctx, manipulator)

                # Redraw cables
                draw_cables(meshcat, plant, plant_ctx, manipulator, rig)
                draw_exo_cables(meshcat, plant, plant_ctx, manipulator, exo_rig)

                # Update Figure 1 (drive-cable top view)
                plt.close(_top_fig)
                _top_fig, _ = visualize_cable_routing_top_view(plant, plant_ctx, manipulator, q1_deg, q2_deg, rig)
                plt.show(block=False)
                plt.pause(0.05)

                # Update exo cable top view
                plt.close(_exo_fig)
                _exo_fig, _ = visualize_exo_cable_routing_top_view(plant, plant_ctx, manipulator, exo_rig, q1_deg, q2_deg)
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


if __name__ == "__main__":
    main()
