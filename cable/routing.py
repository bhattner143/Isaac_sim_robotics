"""
cable/routing.py

Cable route, spring, and rig assembly classes.

Classes: CableRoute, FixedBodyPoint, CableSpring, CableRig.
Functions: spring_zigzag_points()
"""

import numpy as np
from dataclasses import dataclass
from pydrake.geometry import Rgba

from cable.pulley import (
    PulleyBase,
    CableStartPointR,
    CableStartPointL,
    DrivePulley,
    IdlerL,
    IdlerR,
    BigPulley,
    CableEndPointL,
    CableEndPointR,
)



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


