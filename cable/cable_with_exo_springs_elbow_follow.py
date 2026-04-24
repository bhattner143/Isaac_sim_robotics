#!/usr/bin/env python3
"""
cable_with_exo_springs_elbow_follow.py
──────────────────────────────────────
Exosuit cable routing for the *centred-elbow-pulley* variant (Method B).

URDF: manipulator_cable_exo_springs_elbow_follow

In this design a **single big pulley** (``exo_elbow_pulley_big``) is centred
on the elbow joint axis, replacing the two small offset pulleys of the
original manipulator_cable_exo_springs model.  Each cable stays on its own
side and wraps around its own groove on the shared elbow pulley:

  Exo RIGHT cable (orange)  —  upper groove  Z_world = 286.55 mm:
    ExoStartRight (−Y)  →  L1PulleyR (−Y)  →  ElbowBig  →  End−Y

  Exo LEFT cable (magenta)  —  lower groove  Z_world = 283.55 mm:
    ExoStartLeft (+Y)  →  L1PulleyL (+Y)  →  ElbowBig  →  End+Y

Key differences from the offset-pulley exo (cable_with_exo_springs.py):

  1. ONE shared elbow pulley with TWO grooves (3 mm apart) on the joint axis
  2. Cables do NOT cross sides — each stays on its own Y-side (agonist/antagonist)
  3. All tangents are external (no internal crossing)
  4. ElbowBig→End tangent crosses q1→q2 bodies (FK required)

Interactive: type  q1 q2 [deg]  at the prompt → manipulator moves + cable redraws.

Usage:
    python cable/cable_with_exo_springs_elbow_follow.py
    python cable/cable_with_exo_springs_elbow_follow.py --no-springs
    python cable/cable_with_exo_springs_elbow_follow.py --no-exo-springs
"""

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
from cable.pulley import (
    _parse_urdf_part_origins,
    PulleyBase,
)
from cable.routing import (
    CableRoute,
    CableSpring,
    FixedBodyPoint,
    CableRig,
    spring_zigzag_points,
)
from cable.drake_plant import DrakeCablePlant
from cable.cable import (
    CupManipulator,
    create_cable_manipulator_config,
    build_plant,
)
from project_utils.viz_cables import (
    print_cable_routing_points,
    draw_cables,
    visualize_cable_routing_top_view,
    visualize_cable_routing_3d,
)

# ─── Elbow-follow URDF ───────────────────────────────────────────────────────
_EF_URDF = ("model_using_onshape_to_robot/manipulator_cable_exo_springs_elbow_follow/"
            "manipulator_cable_exo_springs_elbow_follow_obj.urdf")
_Q1_EF = "pulley_htd_5m_60t"
_Q2_EF = "link2_tendon"


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
# EXOSUIT CABLE CLASSES — Method B (centred elbow pulley)
# ═══════════════════════════════════════════════════════════════════════════════
# Part positions are in the pulley_htd_5m_60t body frame (q1) or link2_tendon (q2).

# ── Start anchors (on q1 body) ────────────────────────────────────────────────

class ExoStartRight(PulleyBase):
    """Right exo cable start anchor on q1 body (−Y side).

    URDF: Part simple_ball_spring_cable_start on pulley_htd_5m_60t
      xyz=(-0.0238116, -0.108, 0.235049)  rpy≈(0, 0, 0)
      Mesh centroid offset: (0, +0.02, 0) in body frame.
    Zero radius — fixed anchor ball.
    Cable-plane Z: 0.23855 (upper groove → world 286.55 mm).
    """
    obj_name       = "simple_ball.obj"
    body_name      = _Q1_EF
    urdf_part_name = "simple_ball_spring_cable_start"
    vis_xyz        = (-0.0238116, -0.088,  0.23855)
    vis_rpy        = (0.0,         0.0,    0.0)
    face_color     = "#ff8800"
    label          = "Exo Start R"
    mesh_alpha     = 0.85

    def __init__(self):
        super().__init__()
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
    Cable-plane Z: 0.23555 (lower groove → world 283.55 mm).
    """
    obj_name       = "simple_ball.obj"
    body_name      = _Q1_EF
    urdf_part_name = "simple_ball_spring_cable_start_2"
    vis_xyz        = (-0.0238116,  0.088,  0.23555)
    vis_rpy        = (0.0,         0.0,    0.0)
    face_color     = "#cc00cc"
    label          = "Exo Start L"
    mesh_alpha     = 0.85

    def __init__(self):
        super().__init__()
        self._centroid_cache = np.array(type(self).vis_xyz, dtype=float)
        self._radius_cache   = 0.0

    def _compute_radius(self) -> float:
        return 0.0


# ── Link-1 pulleys (on q1 body) ──────────────────────────────────────────────

class ExoLink1PulleyRight(PulleyBase):
    """Right exo link-1 spring-cable pulley on q1 body (−Y side).

    URDF: Part link1_spring_cable_pulley on pulley_htd_5m_60t
      xyz=(0.0141884, -0.088, 0.0990487)  rpy≈(0, 0, 0)
    Radius = 0.041 m (outer mesh rim — visual/drawing radius).
    Tangent radius = 0.035 m (groove floor — for tangent computation).
    Cable-plane Z: 0.23855 (upper groove → world 286.55 mm).
    """
    obj_name       = "link1_spring_cable_pulley.obj"
    body_name      = _Q1_EF
    urdf_part_name = "link1_spring_cable_pulley"
    vis_xyz        = (0.0141884, -0.088,  0.23855)
    vis_rpy        = (0.0,        0.0,    0.0)
    face_color     = "#ff6600"
    label          = "Exo Link1 Pulley R"
    mesh_alpha     = 0.55
    tangent_radius = 0.035

    def __init__(self):
        super().__init__()
        self._centroid_cache = np.array(type(self).vis_xyz, dtype=float)
        self._radius_cache   = 0.041

    def _compute_radius(self) -> float:
        return 0.041


class ExoLink1PulleyLeft(PulleyBase):
    """Left exo link-1 spring-cable pulley on q1 body (+Y side).

    URDF: Part link1_spring_cable_pulley_2 on pulley_htd_5m_60t
      xyz=(0.0141884, 0.088, 0.0990487)  rpy≈(0, 0, 0)
    Radius = 0.041 m (outer mesh rim — visual/drawing radius).
    Tangent radius = 0.035 m (groove floor).
    Cable-plane Z: 0.23555 (lower groove → world 283.55 mm).
    """
    obj_name       = "link1_spring_cable_pulley.obj"
    body_name      = _Q1_EF
    urdf_part_name = "link1_spring_cable_pulley_2"
    vis_xyz        = (0.0141884,  0.088,  0.23555)
    vis_rpy        = (0.0,        0.0,    0.0)
    face_color     = "#aa00aa"
    label          = "Exo Link1 Pulley L"
    mesh_alpha     = 0.55
    tangent_radius = 0.035

    def __init__(self):
        super().__init__()
        self._centroid_cache = np.array(type(self).vis_xyz, dtype=float)
        self._radius_cache   = 0.041

    def _compute_radius(self) -> float:
        return 0.041


# ── Big elbow pulley (on q1 body, centred on joint axis) ─────────────────────

class ExoElbowPulleyBig(PulleyBase):
    """Exo elbow pulley centred on the joint axis (Method B).

    URDF: Part exo_elbow_pulley_big on pulley_htd_5m_60t
      xyz=(0.334804, ~0, 0.222649)  rpy≈(0, -π/2, 0)
    This is at the elbow joint axis (x = 0.334804 mm), so the cable
    kinematics are exact: l_c = r_cp * q2.

    SHARED by both right and left exo cables — like the drive BigPulley.
    Radius is computed from the mesh (expected ~47.75 mm = HTD 5M 60T pitch radius).

    The pulley has TWO grooves (3 mm apart in Z):
      - Upper groove z=0.23855 in q1 frame (world 286.55 mm) → orange/right cable
      - Lower groove z=0.23555 in q1 frame (world 283.55 mm) → purple/left cable
    ``vis_xyz`` is set to the midpoint; compute_tangents() uses per-cable Z.
    """
    obj_name          = "exo_elbow_pulley_big.obj"
    body_name         = _Q1_EF
    urdf_part_name    = "exo_elbow_pulley_big"
    vis_xyz           = (0.334804,  0.0,  0.23705)
    vis_rpy           = (-6.93889e-17, -1.5708, 0.0)
    face_color        = "#cc4400"
    label             = "Exo Elbow Big"
    mesh_alpha        = 0.55
    pulley_axis_local = (1, 0, 0)   # OBJ X = shaft; Ry(−90°) maps to body Z

    # Tangent contact points — filled by ExoCableRig.compute_tangents()
    # Right cable wraps on −Y side (upper groove), left on +Y side (lower groove)
    A_R = None  # right cable entry (−Y side, upper groove)
    B_R = None  # right cable exit  (−Y side → End−Y)
    A_L = None  # left cable entry  (+Y side, lower groove)
    B_L = None  # left cable exit   (+Y side → End+Y)

    # Per-groove Z heights in q1 body frame
    Z_UPPER = 0.23855   # orange / right cable groove
    Z_LOWER = 0.23555   # purple / left  cable groove

    def __init__(self):
        super().__init__()
        self._centroid_cache = np.array(type(self).vis_xyz, dtype=float)
        # _radius_cache intentionally NOT set here — let the lazy property
        # in PulleyBase compute it from the OBJ mesh when first accessed.


# ── End anchors (on q2 body / link2_tendon) ──────────────────────────────────
# Cables CROSS sides at the big elbow pulley (matching manipulator drive cables):
#   Right/orange cable (starts −Y) → internal tangent → exits +Y → ExoEndPlusY
#   Left/purple  cable (starts +Y) → internal tangent → exits −Y → ExoEndMinusY

class ExoEndPlusY(PulleyBase):
    """End anchor on link2 (+Y side) — destination of the RIGHT (orange) cable.

    URDF: Part simple_ball_spring_cable_end on link2_tendon
      xyz=(0.16, 0.018, 0.0165)  rpy=(-π/2, 0, π/2)
      Mesh centroid offset after rotation: (-0.04, 0, 0) in body frame.
    Zero radius — fixed anchor ball.
    q2-frame z=0.0165 → world Z ≈ 286.55 mm (upper groove, same as orange start).
    """
    obj_name       = "simple_ball_spring_cable_end.obj"
    body_name      = _Q2_EF
    urdf_part_name = "simple_ball_spring_cable_end"
    vis_xyz        = (0.12,  0.018,  0.0165)
    vis_rpy        = (-1.5708, 0.0,  1.5708)
    face_color     = "#ff8800"
    label          = "Exo End +Y (←R)"
    mesh_alpha     = 0.85

    def __init__(self):
        super().__init__()
        self._centroid_cache = np.array(type(self).vis_xyz, dtype=float)
        self._radius_cache   = 0.0

    def _compute_radius(self) -> float:
        return 0.0


class ExoEndMinusY(PulleyBase):
    """End anchor on link2 (−Y side) — destination of the LEFT (purple) cable.

    URDF: Part simple_ball_spring_cable_end_2 on link2_tendon
      xyz=(0.16, -0.018, 0.0135)  rpy=(-π/2, 0, π/2)
      Mesh centroid offset after rotation: (-0.04, 0, 0) in body frame.
    Zero radius — fixed anchor ball.
    q2-frame z=0.0135 → world Z ≈ 283.55 mm (lower groove, same as purple start).
    """
    obj_name       = "simple_ball_spring_cable_end.obj"
    body_name      = _Q2_EF
    urdf_part_name = "simple_ball_spring_cable_end_2"
    vis_xyz        = (0.12, -0.018,  0.0135)
    vis_rpy        = (-1.5708, 0.0,  1.5708)
    face_color     = "#cc00cc"
    label          = "Exo End −Y (←L)"
    mesh_alpha     = 0.85

    def __init__(self):
        super().__init__()
        self._centroid_cache = np.array(type(self).vis_xyz, dtype=float)
        self._radius_cache   = 0.0

    def _compute_radius(self) -> float:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ExoCableRig — two exosuit spring cables through a shared centred elbow pulley
# ═══════════════════════════════════════════════════════════════════════════════

class ExoCableRig:
    """Exosuit cable routing (Method B): centred big elbow pulley.

    Mirrors :class:`CableRig` structure: stores pulley instances as named
    attributes, a ``waypoints`` list, and a ``compute_tangents()`` method.

    Routing matches manipulator drive cables (cables CROSS sides at elbow):
      - Right/orange cable (−Y start) → internal at ElbowBig → End +Y (crosses)
      - Left/purple  cable (+Y start) → internal at ElbowBig → End −Y (crosses)

    Each route has **6 waypoints** when tangent routing is active::

      ①  Start anchor (zero-radius)
      ②  Tangent entry onto Link1 Pulley  (A)
      ③  Tangent exit  off  Link1 Pulley  (B)
      ④  Tangent entry onto Elbow Big     (A)
      ⑤  Tangent exit  off  Elbow Big     (B)
      ⑥  End   anchor (zero-radius, on link2)

    Segments ②→③ and ④→⑤ are pulley-wrap arcs (drawn separately);
    their chords are skipped via ``skip_chord_segments``.
    """

    def __init__(self, springs_enabled: bool = True) -> None:
        self.springs_enabled = springs_enabled

        # ── Pulley / anchor instances ─────────────────────────────────────────
        self.exo_start_r        = ExoStartRight()
        self.exo_link1_pulley_r = ExoLink1PulleyRight()
        self.exo_start_l        = ExoStartLeft()
        self.exo_link1_pulley_l = ExoLink1PulleyLeft()

        self.exo_elbow_big      = ExoElbowPulleyBig()   # SHARED by both cables

        self.exo_end_plus_y     = ExoEndPlusY()          # right cable → +Y (crosses)
        self.exo_end_minus_y    = ExoEndMinusY()         # left cable  → −Y (crosses)

        # ── FixedBodyPoint wrappers — tangent contact points ─────────────────
        # Start / End anchors use vis_xyz; pulley entry/exit filled by compute_tangents().

        # Right cable waypoints  (−Y start → crosses → End +Y)
        self.wp_start_r   = FixedBodyPoint(_Q1_EF, np.array(ExoStartRight.vis_xyz),  "Exo Start R")
        self.wp_end_r     = FixedBodyPoint(_Q2_EF, np.array(ExoEndPlusY.vis_xyz),    "Exo End +Y")
        self.wp_link1_A_r = FixedBodyPoint(_Q1_EF, np.zeros(3), "L1 Pulley R (A)")
        self.wp_link1_B_r = FixedBodyPoint(_Q1_EF, np.zeros(3), "L1 Pulley R (B)")
        self.wp_elbow_A_r = FixedBodyPoint(_Q1_EF, np.zeros(3), "Elbow Big (A,R)")
        self.wp_elbow_B_r = FixedBodyPoint(_Q1_EF, np.zeros(3), "Elbow Big (B,R)")

        # Left cable waypoints  (+Y start → crosses → End −Y)
        self.wp_start_l   = FixedBodyPoint(_Q1_EF, np.array(ExoStartLeft.vis_xyz),   "Exo Start L")
        self.wp_end_l     = FixedBodyPoint(_Q2_EF, np.array(ExoEndMinusY.vis_xyz),   "Exo End −Y")
        self.wp_link1_A_l = FixedBodyPoint(_Q1_EF, np.zeros(3), "L1 Pulley L (A)")
        self.wp_link1_B_l = FixedBodyPoint(_Q1_EF, np.zeros(3), "L1 Pulley L (B)")
        self.wp_elbow_A_l = FixedBodyPoint(_Q1_EF, np.zeros(3), "Elbow Big (A,L)")
        self.wp_elbow_B_l = FixedBodyPoint(_Q1_EF, np.zeros(3), "Elbow Big (B,L)")

        # ── Cable routes ──────────────────────────────────────────────────────
        # Cables cross sides at elbow (internal tangent → exit opposite Y-side).
        # skip_chord_segments {1, 3}: chords A→B on L1 and ElbowBig are arcs.
        self.exo_cable_right = CableRoute(
            segments=[
                (self.wp_start_r,   np.zeros(3)),  # ① start anchor (−Y, q1)
                (self.wp_link1_A_r, np.zeros(3)),  # ② L1 Pulley R entry
                (self.wp_link1_B_r, np.zeros(3)),  # ③ L1 Pulley R exit
                (self.wp_elbow_A_r, np.zeros(3)),  # ④ Elbow Big entry (crosses)
                (self.wp_elbow_B_r, np.zeros(3)),  # ⑤ Elbow Big exit  (+Y side)
                (self.wp_end_r,     np.zeros(3)),  # ⑥ end anchor (+Y, q2)
            ],
            meshcat_path        = "/exo_cable/right",
            meshcat_color       = Rgba(1.0, 0.55, 0.0, 1.0),   # orange
            mpl_color           = "darkorange",
            label               = "Exo Right (StartR→L1R→ElbowBig→End+Y)",
            skip_chord_segments = frozenset({1, 3}),
        )
        self.exo_cable_left = CableRoute(
            segments=[
                (self.wp_start_l,   np.zeros(3)),  # ① start anchor (+Y, q1)
                (self.wp_link1_A_l, np.zeros(3)),  # ② L1 Pulley L entry
                (self.wp_link1_B_l, np.zeros(3)),  # ③ L1 Pulley L exit
                (self.wp_elbow_A_l, np.zeros(3)),  # ④ Elbow Big entry (crosses)
                (self.wp_elbow_B_l, np.zeros(3)),  # ⑤ Elbow Big exit  (−Y side)
                (self.wp_end_l,     np.zeros(3)),  # ⑥ end anchor (−Y, q2)
            ],
            meshcat_path        = "/exo_cable/left",
            meshcat_color       = Rgba(0.8, 0.0, 0.8, 1.0),   # magenta
            mpl_color           = "mediumorchid",
            label               = "Exo Left  (StartL→L1L→ElbowBig→End−Y)",
            skip_chord_segments = frozenset({1, 3}),
        )

        self.routes = [self.exo_cable_right, self.exo_cable_left]

        # All pulleys with geometry (for visualization)
        self.waypoints = [
            self.exo_start_r,  self.exo_link1_pulley_r,
            self.exo_elbow_big,
            self.exo_end_plus_y,
            self.exo_start_l,  self.exo_link1_pulley_l,
            self.exo_end_minus_y,
        ]

        # Wrap arc descriptors — filled after compute_tangents()
        self.wrap_arcs: list[tuple] = []

        # ── Optional springs (elbow exit → end) ──────────────────────────────
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
        """Compute tangent entry/exit points on L1 pulleys and the big elbow pulley.

        Routing matches manipulator drive cables (cables cross sides):

        Right cable (−Y start, upper groove):
          Start (r=0) → CW on L1PulleyR: branch=-1, external
          L1PulleyR → ElbowBig:           branch=-1, **internal** (crosses to +Y)
          ElbowBig → End+Y (r=0):         branch=+1, external  (exit +Y side)

        Left cable (+Y start, lower groove):
          Start (r=0) → CCW on L1PulleyL: branch=+1, external
          L1PulleyL → ElbowBig:           branch=+1, **internal** (crosses to −Y)
          ElbowBig → End−Y (r=0):         branch=-1, external  (exit −Y side)

        The ElbowBig→End tangent crosses from q1 body to q2 body.
        We FK-transform the End position into q1's frame for the tangent solve.
        """
        ct = PulleyBase.compute_tangent

        # Branch signs — matching manipulator drive cables (cross sides at elbow)
        #       right cable (−Y)             left cable (+Y)
        b_start_link1_R, b_start_link1_L = -1, +1   # CW right, CCW left
        b_link1_elbow_R, b_link1_elbow_L = -1, +1   # internal tangent → crosses!
        b_elbow_end_R,   b_elbow_end_L   = +1, -1   # external, exit opposite side

        r_link1_tangent = ExoLink1PulleyRight.tangent_radius   # groove floor for ct()
        r_link1_draw    = self.exo_link1_pulley_r.radius       # outer rim for drawing
        r_elbow = self.exo_elbow_big.radius

        # Per-cable Z heights for the two elbow-pulley grooves
        z_upper = ExoElbowPulleyBig.Z_UPPER   # orange / right
        z_lower = ExoElbowPulleyBig.Z_LOWER   # purple / left

        # FK: transform End points (q2 body) into q1 body frame for tangent solve
        body_q1 = plant.GetBodyByName(_Q1_EF, manipulator.model_instance)
        body_q2 = plant.GetBodyByName(_Q2_EF, manipulator.model_instance)
        X_q1_q2 = plant.CalcRelativeTransform(
            plant_context, body_q1.body_frame(), body_q2.body_frame())
        R12, t12 = X_q1_q2.rotation().matrix(), X_q1_q2.translation()

        c_end_minus_y_in_q1 = R12 @ np.array(ExoEndMinusY.vis_xyz, float) + t12
        c_end_plus_y_in_q1  = R12 @ np.array(ExoEndPlusY.vis_xyz, float)  + t12

        # --- Right cable (Start R → L1 Pulley R → Elbow Big → End +Y) ---
        # Entire cable stays on upper groove (Z=286.55 mm).
        c_start_r = np.array(ExoStartRight.vis_xyz, float)
        c_link1_r = np.array(ExoLink1PulleyRight.vis_xyz, float)
        c_elbow_r = np.array([ExoElbowPulleyBig.vis_xyz[0],
                              ExoElbowPulleyBig.vis_xyz[1], z_upper])

        # Start (r=0) → L1 Pulley R (tangent at groove radius)
        _, A_link1_r = ct(c_start_r, 0.0, c_link1_r, r_link1_tangent,
                          branch=b_start_link1_R, kind="external")
        # L1 Pulley R → Elbow Big (internal tangent — cable crosses to +Y)
        B_link1_r, A_elbow_r = ct(c_link1_r, r_link1_tangent, c_elbow_r, r_elbow,
                                   branch=b_link1_elbow_R, kind="internal")
        # Project L1 tangent points from groove to outer rim for drawing
        A_link1_r = _radial_project(c_link1_r, A_link1_r, r_link1_draw)
        B_link1_r = _radial_project(c_link1_r, B_link1_r, r_link1_draw)
        # Elbow Big → End +Y (same upper groove Z)
        B_elbow_r, _ = ct(c_elbow_r, r_elbow, c_end_plus_y_in_q1, 0.0,
                          branch=b_elbow_end_R, kind="external")

        # --- Left cable (Start L → L1 Pulley L → Elbow Big → End −Y) ---
        # Entire cable stays on lower groove (Z=283.55 mm).
        c_start_l = np.array(ExoStartLeft.vis_xyz, float)
        c_link1_l = np.array(ExoLink1PulleyLeft.vis_xyz, float)
        c_elbow_l = np.array([ExoElbowPulleyBig.vis_xyz[0],
                              ExoElbowPulleyBig.vis_xyz[1], z_lower])

        _, A_link1_l = ct(c_start_l, 0.0, c_link1_l, r_link1_tangent,
                          branch=b_start_link1_L, kind="external")
        # L1 Pulley L → Elbow Big (internal tangent — cable crosses to −Y)
        B_link1_l, A_elbow_l = ct(c_link1_l, r_link1_tangent, c_elbow_l, r_elbow,
                                   branch=b_link1_elbow_L, kind="internal")
        A_link1_l = _radial_project(c_link1_l, A_link1_l, r_link1_draw)
        B_link1_l = _radial_project(c_link1_l, B_link1_l, r_link1_draw)
        # Elbow Big → End −Y (same lower groove Z)
        B_elbow_l, _ = ct(c_elbow_l, r_elbow, c_end_minus_y_in_q1, 0.0,
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
        self.exo_link1_pulley_l.A_L = A_link1_l
        self.exo_link1_pulley_l.B_L = B_link1_l

        self.exo_elbow_big.A_R = A_elbow_r
        self.exo_elbow_big.B_R = B_elbow_r
        self.exo_elbow_big.A_L = A_elbow_l
        self.exo_elbow_big.B_L = B_elbow_l

        # Wrap arcs for drawing
        # Tuple: (pulley, A_body, B_body, path, rgba, center_override_body)
        # center_override_body: if not None, use this as the arc center in body
        # frame instead of pulley.centroid (needed for dual-groove elbow).
        self.wrap_arcs = [
            (self.exo_link1_pulley_r, A_link1_r, B_link1_r,
             "/exo_cable/right/wrap/link1", Rgba(1.0, 0.55, 0.0, 1.0), None),
            (self.exo_elbow_big, A_elbow_r, B_elbow_r,
             "/exo_cable/right/wrap/elbow", Rgba(1.0, 0.55, 0.0, 1.0),
             c_elbow_r),  # upper groove center
            (self.exo_link1_pulley_l, A_link1_l, B_link1_l,
             "/exo_cable/left/wrap/link1",  Rgba(0.8, 0.0, 0.8, 1.0), None),
            (self.exo_elbow_big, A_elbow_l, B_elbow_l,
             "/exo_cable/left/wrap/elbow",  Rgba(0.8, 0.0, 0.8, 1.0),
             c_elbow_l),  # lower groove center
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# Exo cable visualization helpers
# ═══════════════════════════════════════════════════════════════════════════════

def print_exo_cable_routing_points(plant, plant_context, manipulator,
                                   exo_rig: ExoCableRig) -> None:
    """Print all exo cable waypoints in world frame."""
    print("\n" + "=" * 64)
    print("  Exo-suit cable routing points — centred elbow (world frame)")
    print("=" * 64)
    for route in exo_rig.routes:
        pts = route.world_points(plant, plant_context, manipulator)
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
    """Top-view (XY plane) schematic of exo cable routing (Method B)."""

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
                           num="Exo Cable (Elbow Follow) — Top View (XY)")
    ax.set_title(
        f"Exo Cable Routing — Centred Elbow Pulley (Method B)   "
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
            ax.plot(wx, wy, 'o', color=cfg.face_color, markersize=8,
                    markeredgecolor='k', markeredgewidth=0.8, zorder=4)
            ax.text(wx, wy + 0.004, cfg.label, fontsize=6.5,
                    ha='center', va='bottom',
                    color=cfg.face_color, fontweight='bold', zorder=5)

    # ── Cable straight segments + wrap arcs ──────────────────────────────────
    springs_on = exo_rig.springs_enabled

    for ri, route in enumerate(exo_rig.routes):
        pts = route.world_points(plant, plant_context, manipulator)
        skip = getattr(route, "skip_chord_segments", frozenset())
        last_seg_idx = len(pts) - 2
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
                    p_ss = p1 + t0 * (p0 - p1)
                    p_se = p1 + t1 * (p0 - p1)
                    ax.plot([p0[0], p_se[0]], [p0[1], p_se[1]], '-',
                            color=route.mpl_color, linewidth=2, zorder=6)
                    ax.plot([p_ss[0], p1[0]], [p_ss[1], p1[1]], '-',
                            color=route.mpl_color, linewidth=2, zorder=6)
                    zz = spring_zigzag_points(p_se, p_ss,
                                              n_coils=spring.n_coils,
                                              amplitude=spring.amplitude)
                    ax.plot(zz[:, 0], zz[:, 1], '-', color='goldenrod',
                            linewidth=2, zorder=7)
                    continue
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], '-', color=route.mpl_color,
                    linewidth=2, zorder=6)
        for pt in pts:
            ax.plot(pt[0], pt[1], 'o', color=route.mpl_color, markersize=4, zorder=6)
        ax.plot([], [], '-o', color=route.mpl_color, linewidth=2, markersize=4,
                label=route.label)

    # ── Wrap arcs on pulleys (matplotlib) ────────────────────────────────────
    for pulley, A_body, B_body, _, _, center_override in exo_rig.wrap_arcs:
        R_wb, t_wb = _Xw(pulley.body_name)
        if center_override is not None:
            c_w = R_wb @ np.asarray(center_override) + t_wb
        else:
            c_w = R_wb @ np.asarray(pulley.centroid) + t_wb
        A_w  = R_wb @ np.asarray(A_body) + t_wb
        B_w  = R_wb @ np.asarray(B_body) + t_wb
        shaft_w = R_wb @ pulley.shaft_axis_body

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
        ax.plot(arc_pts[:, 0], arc_pts[:, 1], '-', color=pulley.face_color,
                linewidth=2.5, alpha=0.8, zorder=6)

    # ── Body-frame origin crosses ────────────────────────────────────────────
    for body_name, color, lbl in [
        (_Q1_EF, "royalblue",  "J1"),
        (_Q2_EF, "seagreen",   "J2"),
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
    springs_on = exo_rig.springs_enabled

    for ri, route in enumerate(exo_rig.routes):
        pts = np.array([
            _body_pt_world(cfg.body_name, cfg._tangent_point)
            for cfg, _ in route.segments
        ])
        skip = getattr(route, "skip_chord_segments", frozenset())
        last_seg_idx = len(pts) - 2
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
                    p_spring_start = p1 + t0 * (p0 - p1)
                    p_spring_end   = p1 + t1 * (p0 - p1)
                    _place_seg(f"{route.meshcat_path}/seg{i:02d}_a",
                               p0, p_spring_end, route.meshcat_color)
                    zz = spring_zigzag_points(p_spring_end, p_spring_start,
                                              n_coils=spring.n_coils,
                                              amplitude=spring.amplitude)
                    spring_rgba = Rgba(0.9, 0.6, 0.0, 1.0)
                    for j, (z0, z1) in enumerate(zip(zz[:-1], zz[1:])):
                        _place_seg(f"{route.meshcat_path}/spring{j:02d}",
                                   z0, z1, spring_rgba)
                    _place_seg(f"{route.meshcat_path}/seg{i:02d}_b",
                               p_spring_start, p1, route.meshcat_color)
                    continue
            _place_seg(f"{route.meshcat_path}/seg{i:02d}", p0, p1, route.meshcat_color)

    # ── Wrap arcs on pulleys ──────────────────────────────────────────────────
    for pulley, A_body, B_body, path_prefix, rgba, center_override in exo_rig.wrap_arcs:
        R_wb, t_wb = _Xw(pulley.body_name)
        if center_override is not None:
            center_w = R_wb @ np.asarray(center_override) + t_wb
        else:
            center_w = R_wb @ np.asarray(pulley.centroid) + t_wb
        shaft_w    = R_wb @ pulley.shaft_axis_body
        A_w        = R_wb @ np.asarray(A_body) + t_wb
        B_w        = R_wb @ np.asarray(B_body) + t_wb
        arc_pts    = _arc_world_pts(center_w, pulley.radius, shaft_w, A_w, B_w)
        for i, (p0, p1) in enumerate(zip(arc_pts[:-1], arc_pts[1:])):
            _place_seg(f"{path_prefix}/arc{i:02d}", p0, p1, rgba)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Exo-springs cable routing (Method B — centred elbow pulley).")
    ap.add_argument("--no-springs", action="store_true",
                    help="Disable drive-cable endpoint springs")
    ap.add_argument("--no-exo-springs", action="store_true",
                    help="Disable exo-cable endpoint springs")
    args = ap.parse_args()
    springs_enabled     = not args.no_springs
    exo_springs_enabled = not args.no_exo_springs

    # ── Configuration (elbow-follow URDF) ─────────────────────────────────────
    config = create_cable_manipulator_config(
        urdf_path=_EF_URDF,
        joint_angles={"link1_base": 0.0, "link2_link1": 0.0},
        damping=(0.1, 0.1),
    )

    # ── Meshcat ───────────────────────────────────────────────────────────────
    meshcat = StartMeshcat()
    print(colored(f"\n🌐 Meshcat: {meshcat.web_url()}\n", "green", attrs=["bold"]))

    # ── Plant ─────────────────────────────────────────────────────────────────
    builder, plant, scene_graph, manipulator = build_plant(config)

    # ── Cable rigs ────────────────────────────────────────────────────────────
    manipulator.init_cable_rig(springs_enabled=springs_enabled)
    rig     = manipulator.rig           # drive-belt rig (green/red cables)

    # Sync cable.pulley.PulleyBase class attributes from cable.cable.PulleyBase
    # (init_cable_rig sets assets_dir/_urdf_origins on cable.cable's copy;
    #  the exo pulley classes inherit from cable.pulley's copy).
    from cable.cable import PulleyBase as _CablePB
    PulleyBase.assets_dir    = _CablePB.assets_dir
    PulleyBase._urdf_origins = _CablePB._urdf_origins

    exo_rig = ExoCableRig(springs_enabled=exo_springs_enabled)

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

    manipulator.compute_tangents(plant, plant_ctx)
    exo_rig.compute_tangents(plant, plant_ctx, manipulator)

    draw_cables(meshcat, plant, plant_ctx, manipulator, rig)
    draw_exo_cables(meshcat, plant, plant_ctx, manipulator, exo_rig)
    print_cable_routing_points(plant, plant_ctx, manipulator, rig)
    print_exo_cable_routing_points(plant, plant_ctx, manipulator, exo_rig)

    # Top-view figures
    _top_fig, _ = visualize_cable_routing_top_view(plant, plant_ctx, manipulator, 0.0, 0.0, rig)
    plt.show(block=False)
    plt.pause(0.05)

    _exo_fig, _ = visualize_exo_cable_routing_top_view(plant, plant_ctx, manipulator, exo_rig, 0.0, 0.0)
    plt.show(block=False)
    plt.pause(0.05)

    _viz_fig = None

    ee = manipulator.get_end_effector_position(plant, plant_ctx)
    print(colored(
        f"Exo cables: Method B (centred elbow pulley, r_cp ≈ {exo_rig.exo_elbow_big.radius*1000:.1f} mm)",
        "yellow"))
    print(colored(f"Home:  q1=0°  q2=0°  →  EE=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) m\n", "cyan"))
    print(colored("Enter joint angles in degrees  (e.g.  30  -15)  or Ctrl+C to exit.\n", "yellow"))

    # ── Interactive loop ──────────────────────────────────────────────────────
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
                diagram.ForcedPublish(context)

                manipulator.compute_tangents(plant, plant_ctx)
                exo_rig.compute_tangents(plant, plant_ctx, manipulator)

                draw_cables(meshcat, plant, plant_ctx, manipulator, rig)
                draw_exo_cables(meshcat, plant, plant_ctx, manipulator, exo_rig)

                plt.close(_top_fig)
                _top_fig, _ = visualize_cable_routing_top_view(
                    plant, plant_ctx, manipulator, q1_deg, q2_deg, rig)
                plt.show(block=False)
                plt.pause(0.05)

                plt.close(_exo_fig)
                _exo_fig, _ = visualize_exo_cable_routing_top_view(
                    plant, plant_ctx, manipulator, exo_rig, q1_deg, q2_deg)
                plt.show(block=False)
                plt.pause(0.05)

                if _viz_fig is not None:
                    plt.close(_viz_fig)
                _viz_fig, _ = visualize_cable_routing_3d(
                    plant, plant_ctx, manipulator, PulleyBase.assets_dir,
                    q1_deg, q2_deg, rig)
                plt.show(block=False)
                plt.pause(0.05)

                ee = manipulator.get_end_effector_position(plant, plant_ctx)
                print(colored(
                    f"  ✓  q1={q1_deg:.1f}°  q2={q2_deg:.1f}°  "
                    f"→  EE=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) m",
                    "green",
                ))
            except ValueError as e:
                print(colored(f"  ✗ {e}. Enter two floats: q1 q2", "red"))
    except (KeyboardInterrupt, EOFError):
        print(colored("\n✓ Stopped.", "green"))


if __name__ == "__main__":
    main()
