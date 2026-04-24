"""
project_utils/viz_cables_isaacsim.py
─────────────────────────────────────
Cable (tendon) visualization for Isaac Sim using ``DrakeCablePlant`` from
``cable.py`` for accurate cable routing + tangent computation.

Uses the same headless Drake MultibodyPlant FK that the PyDrake and Newton
scripts rely on — guaranteeing identical cable geometry across all viewers.

This is the Isaac Sim counterpart of ``project_utils/viz_cables.py``
(which provides Meshcat + matplotlib visualization for PyDrake).

IMPORTANT: When using ``CableVisualizerIsaac``, call ``create_prims()``
BEFORE ``world.reset()`` to pre-allocate USD cylinder prims.  Then call
``update()`` during the sim loop — it only modifies existing prim
transforms, never creates or removes prims (which would invalidate
Isaac Sim's ArticulationView / physics view).

Usage (class-based — recommended for simulation loops)::

    from project_utils.viz_cables_isaacsim import CableVisualizerIsaac

    cable_viz = CableVisualizerIsaac(stage, drake_urdf_path)
    cable_viz.create_prims(q1_init, q2_init)   # BEFORE world.reset()
    world.reset()
    # In sim loop:
    cable_viz.update(q1, q2)

Usage (function-based — for scene-viz / interactive)::

    from project_utils.viz_cables_isaacsim import draw_cables_usd, update_cables_usd

    draw_cables_usd(stage, drake_cable)     # initial draw
    update_cables_usd(stage, drake_cable)   # after joint change
"""

import math
import numpy as np

# USD imports (available after SimulationApp)
from pxr import UsdGeom, Gf

# Cable FK — headless Drake plant (same as scene-viz and PyDrake scripts)
from cable import DrakeCablePlant


# ============================================================================
# USD CYLINDER RENDERING
# ============================================================================

_CABLE_ROOT = "/World/Cables"
_CABLE_RADIUS = 0.0005  # 0.5 mm


def _usd_cylinder(stage, path: str, p0: np.ndarray, p1: np.ndarray, color_rgb):
    """Create or update a thin USD cylinder between two 3D points."""
    diff = p1 - p0
    length = float(np.linalg.norm(diff))
    if length < 1e-9:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Cylinder(prim).GetHeightAttr().Set(0.0)
        return
    mid = (p0 + p1) * 0.5

    # Orientation: cylinder default axis is Z; rotate to align with diff
    z_hat = diff / length
    tmp = np.array([0., 1., 0.]) if abs(z_hat[0]) > 0.9 else np.array([1., 0., 0.])
    x_hat = np.cross(tmp, z_hat)
    x_hat /= np.linalg.norm(x_hat)
    y_hat = np.cross(z_hat, x_hat)
    # 4×4 row-major for USD (rows = basis vectors + translation)
    mat = Gf.Matrix4d(
        float(x_hat[0]), float(x_hat[1]), float(x_hat[2]), 0.0,
        float(y_hat[0]), float(y_hat[1]), float(y_hat[2]), 0.0,
        float(z_hat[0]), float(z_hat[1]), float(z_hat[2]), 0.0,
        float(mid[0]),   float(mid[1]),   float(mid[2]),   1.0,
    )

    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        # Update existing
        cyl = UsdGeom.Cylinder(prim)
        cyl.GetHeightAttr().Set(length)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(mat)
    else:
        # Create new
        cyl = UsdGeom.Cylinder.Define(stage, path)
        cyl.GetRadiusAttr().Set(_CABLE_RADIUS)
        cyl.GetHeightAttr().Set(length)
        cyl.GetDisplayColorAttr().Set([Gf.Vec3f(*[float(c) for c in color_rgb])])
        xf = UsdGeom.Xformable(cyl.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(mat)


def _route_color(route):
    """Extract RGB tuple from a CableRoute's mpl_color string."""
    if "green" in route.mpl_color.lower():
        return (0.1, 0.85, 0.1)
    return (0.9, 0.1, 0.1)


# ============================================================================
# FUNCTION-BASED API  (scene-viz / interactive use)
# ============================================================================

def draw_cables_usd(stage, drake_cable: DrakeCablePlant):
    """Draw all cable segments and wrap arcs as USD Cylinder prims.

    Creates new USD prims.  For the first draw or full rebuild.
    """
    # Straight segments between waypoints
    for route, pts in drake_cable.get_cable_world_points():
        skip = getattr(route, "skip_chord_segments", frozenset())
        color = _route_color(route)
        base = f"{_CABLE_ROOT}/{route.meshcat_path.replace('/', '_').strip('_')}"
        for i, (p0, p1) in enumerate(zip(pts[:-1], pts[1:])):
            if i in skip:
                continue
            _usd_cylinder(stage, f"{base}/seg{i:02d}", p0, p1, color)

    # Wrap arcs
    for label, color, arc_pts in drake_cable.get_wrap_arcs():
        base = f"{_CABLE_ROOT}/wrap_{label}"
        for i, (p0, p1) in enumerate(zip(arc_pts[:-1], arc_pts[1:])):
            _usd_cylinder(stage, f"{base}/arc{i:02d}", p0, p1, color)


def update_cables_usd(stage, drake_cable: DrakeCablePlant):
    """Update existing cable USD prims after joint angles changed.

    Removes old cable prim tree and redraws.
    """
    cable_prim = stage.GetPrimAtPath(_CABLE_ROOT)
    if cable_prim.IsValid():
        stage.RemovePrim(_CABLE_ROOT)
    draw_cables_usd(stage, drake_cable)


# ============================================================================
# CLASS-BASED API  (simulation loops — pre-allocate before world.reset)
# ============================================================================

class CableVisualizerIsaac:
    """Cable visualization for Isaac Sim using DrakeCablePlant from cable.py.

    Uses a headless Drake MultibodyPlant for cable FK and tangent computation,
    then renders as USD cylinders in Isaac Sim.

    IMPORTANT: Call ``create_prims()`` BEFORE ``world.reset()`` to pre-allocate
    all USD cylinder prims. Then call ``update()`` during the sim loop — it only
    modifies existing prim transforms, never creates or removes prims (which
    would invalidate Isaac Sim's ArticulationView / physics view).

    Usage::

        viz = CableVisualizerIsaac(stage, drake_urdf_path)
        viz.create_prims(q1_init, q2_init)   # BEFORE world.reset()
        world.reset()
        # ... in sim loop:
        viz.update(q1, q2)
    """

    def __init__(self, stage, drake_urdf_path: str):
        self._stage = stage
        self._drake_urdf = drake_urdf_path
        self._drake_cable = None
        print(f"[CableViz] Using DrakeCablePlant with {drake_urdf_path}")

    @property
    def drake_cable(self) -> DrakeCablePlant:
        """Access the underlying DrakeCablePlant (for direct queries)."""
        return self._drake_cable

    def create_prims(self, q1: float = 0.0, q2: float = 0.0):
        """Pre-allocate all cable USD cylinder prims.

        Call this BEFORE ``world.reset()`` so prims are part of the initial
        stage composition and don't invalidate the physics view later.
        """
        self._drake_cable = DrakeCablePlant(self._drake_urdf, q1=q1, q2=q2)
        draw_cables_usd(self._stage, self._drake_cable)
        print(f"[CableViz] Pre-allocated cable prims at q=({math.degrees(q1):.1f}°, {math.degrees(q2):.1f}°)")

    def update(self, q1: float, q2: float):
        """Recompute cable routing and update existing USD cylinder transforms.

        Only updates transforms of existing prims — never creates or removes
        prims. Safe to call during simulation without invalidating physics.
        """
        self._drake_cable.update(q1, q2)
        draw_cables_usd(self._stage, self._drake_cable)


# ============================================================================
# EXO-CABLE VISUALISER  (Method B — centred elbow pulley)
# ============================================================================
# Uses a headless PyDrake plant to compute exo-cable routing (two
# antagonistic cables wrapping through a shared centred elbow pulley) and
# renders them as USD cylinders in Isaac Sim.  Independent of the drive
# CableVisualizerIsaac — both can be used simultaneously.

_EXO_CABLE_ROOT = "/World/ExoCables"
_EXO_CABLE_RADIUS = 0.0005          # 0.5 mm
# Per-route display colours matching the PyDrake Meshcat visualisation
# (see ExoCableRig.exo_cable_right / exo_cable_left in
# ``cable/cable_with_exo_springs_elbow_follow.py``).
_EXO_COLOR_RIGHT = (1.0, 0.55, 0.0)   # orange — right cable (upper groove)
_EXO_COLOR_LEFT  = (0.8, 0.0, 0.8)    # magenta — left  cable (lower groove)


def _exo_route_color(route_index: int):
    """Return RGB tuple for exo-cable route index (0=right, 1=left)."""
    return _EXO_COLOR_RIGHT if route_index == 0 else _EXO_COLOR_LEFT


class _DrakeExoCablePlant:
    """Headless PyDrake plant for exo-cable FK + tangent computation.

    Builds a MultibodyPlant using the Exo URDF + CupManipulatorTendonWithExo
    wrapper so that BOTH drive cables and exo cables share the same joint
    state.  Only the exo rig is exposed — drive cables are handled by the
    regular :class:`CableVisualizerIsaac`.
    """

    def __init__(self, drake_urdf: str, q1: float = 0.0, q2: float = 0.0,
                 springs_enabled: bool = True):
        # Local import keeps Drake off the Isaac-Sim import hot path when
        # only the drive visualiser is used.
        from pydrake.all import (
            DiagramBuilder, MultibodyPlant, SceneGraph, Parser,
        )
        from robots.cup_manipulator_tendon_with_exo import (
            CupManipulatorTendonWithExo,
        )
        from robots.cup_manipulator_tendon import create_cable_manipulator_config

        config = create_cable_manipulator_config(
            urdf_path=drake_urdf,
            joint_angles={"link1_base": q1, "link2_link1": q2},
        )
        builder = DiagramBuilder()
        plant = MultibodyPlant(time_step=0.001)
        scene_graph = builder.AddSystem(SceneGraph())
        plant.RegisterAsSourceForSceneGraph(scene_graph)

        manipulator = CupManipulatorTendonWithExo(
            config, enable_visualization=False,
        )
        parser = Parser(plant)
        manipulator.load_urdf_to_plant(plant, parser)
        manipulator.weld_base_to_world(plant)
        manipulator.add_end_effector_frame(plant)
        plant.Finalize()
        builder.AddSystem(plant)

        self.plant = plant
        self.manipulator = manipulator
        self.diagram = builder.Build()
        self._root_ctx = self.diagram.CreateDefaultContext()
        self.plant_ctx = plant.GetMyMutableContextFromRoot(self._root_ctx)

        # Initialise both cable rigs (drive + exo).  Drive rig is built so
        # that callers who want drive cables rendered can reuse the same
        # plant for both — but in practice we only return the exo rig here.
        manipulator.init_cable_rig(drake_urdf, springs_enabled=springs_enabled)
        manipulator.init_exo_cable_rig(drake_urdf, springs_enabled=springs_enabled)

        self.rig = manipulator.exo_rig
        self._set_angles(q1, q2)
        manipulator.compute_exo_tangents(plant, self.plant_ctx)

    def _set_angles(self, q1: float, q2: float):
        import numpy as _np
        self.manipulator.set_positions_user_order(
            self.plant, self.plant_ctx,
            {"link1_base": q1, "link2_link1": q2},
        )
        self.plant.SetVelocities(
            self.plant_ctx, _np.zeros(self.plant.num_velocities()),
        )

    def update(self, q1: float, q2: float):
        """Sync joint angles and recompute exo-cable tangents."""
        self._set_angles(q1, q2)
        self.manipulator.compute_exo_tangents(self.plant, self.plant_ctx)

    def get_cable_world_points(self):
        """Return list of (route, world_pts) for each exo cable."""
        return [
            (route, route.world_points(self.plant, self.plant_ctx, self.manipulator))
            for route in self.rig.routes
        ]

    def get_wrap_arcs(self, n_arc_pts: int = 24):
        """Return wrap-arc world-frame points for each exo pulley wrap.

        Yields ``(label, color_rgb, arc_pts_Nx3)`` tuples.  Uses the
        per-route ``wrap_arcs`` metadata built by
        ``ExoCableRig.compute_tangents`` (see
        ``cable/cable_with_exo_springs_elbow_follow.py``).
        """
        import numpy as _np

        def _Xw(body_name):
            body = self.plant.GetBodyByName(
                body_name, self.manipulator.model_instance,
            )
            X = self.plant.CalcRelativeTransform(
                self.plant_ctx, self.plant.world_frame(), body.body_frame(),
            )
            return X.rotation().matrix(), X.translation()

        results = []
        for entry in getattr(self.rig, "wrap_arcs", []):
            # Tuple format: (pulley, A_body, B_body, path, rgba, center_override_body)
            if len(entry) >= 6:
                pulley, A_body, B_body, path, rgba, center_override = entry
            else:
                pulley, A_body, B_body, path, rgba = entry[:5]
                center_override = None
            if A_body is None or B_body is None:
                continue
            R_wb, t_wb = _Xw(pulley.body_name)
            centroid_body = (center_override if center_override is not None
                             else pulley.centroid)
            center_w = R_wb @ _np.asarray(centroid_body) + t_wb
            shaft_w  = R_wb @ pulley.shaft_axis_body
            A_w = R_wb @ _np.asarray(A_body) + t_wb
            B_w = R_wb @ _np.asarray(B_body) + t_wb
            ax = shaft_w / _np.linalg.norm(shaft_w)
            dA = A_w - center_w
            dA -= _np.dot(dA, ax) * ax
            dA /= _np.linalg.norm(dA)
            dB = B_w - center_w
            dB -= _np.dot(dB, ax) * ax
            dB /= _np.linalg.norm(dB)
            cos_ab = float(_np.clip(_np.dot(dA, dB), -1.0, 1.0))
            angle = _np.sign(_np.dot(_np.cross(dA, dB), ax)) * _np.arccos(cos_ab)
            ax_cross_dA = _np.cross(ax, dA)
            arc_pts = _np.array([
                center_w + pulley.radius * (dA * _np.cos(th)
                                            + ax_cross_dA * _np.sin(th))
                for th in _np.linspace(0.0, angle, n_arc_pts)
            ])
            # Convert meshcat Rgba to RGB tuple for USD.
            try:
                color_rgb = (float(rgba.r()), float(rgba.g()), float(rgba.b()))
            except Exception:
                color_rgb = _EXO_COLOR_RIGHT
            # Extract a short label from the full meshcat path.
            label = path.strip("/").replace("/", "_")
            results.append((label, color_rgb, arc_pts))
        return results


def _draw_exo_cables_usd(stage, drake_exo: "_DrakeExoCablePlant"):
    """Draw exo-cable segments + wrap arcs as USD Cylinder prims."""
    # Straight segments between waypoints
    for ri, (route, pts) in enumerate(drake_exo.get_cable_world_points()):
        skip = getattr(route, "skip_chord_segments", frozenset())
        color = _exo_route_color(ri)
        base = f"{_EXO_CABLE_ROOT}/{route.meshcat_path.replace('/', '_').strip('_')}"
        for i, (p0, p1) in enumerate(zip(pts[:-1], pts[1:])):
            if i in skip:
                continue
            _usd_cylinder_exo(stage, f"{base}/seg{i:02d}", p0, p1, color)

    # Wrap arcs around pulleys
    for label, color, arc_pts in drake_exo.get_wrap_arcs():
        base = f"{_EXO_CABLE_ROOT}/wrap_{label}"
        for i, (p0, p1) in enumerate(zip(arc_pts[:-1], arc_pts[1:])):
            _usd_cylinder_exo(stage, f"{base}/arc{i:02d}", p0, p1, color)


def _usd_cylinder_exo(stage, path: str, p0: np.ndarray, p1: np.ndarray, color_rgb):
    """Create or update a thin USD cylinder for an exo-cable segment.

    Same as ``_usd_cylinder`` but uses ``_EXO_CABLE_RADIUS`` so exo cables
    can be styled independently of drive cables.
    """
    diff = p1 - p0
    length = float(np.linalg.norm(diff))
    if length < 1e-9:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Cylinder(prim).GetHeightAttr().Set(0.0)
        return
    mid = (p0 + p1) * 0.5

    z_hat = diff / length
    tmp = np.array([0., 1., 0.]) if abs(z_hat[0]) > 0.9 else np.array([1., 0., 0.])
    x_hat = np.cross(tmp, z_hat)
    x_hat /= np.linalg.norm(x_hat)
    y_hat = np.cross(z_hat, x_hat)
    mat = Gf.Matrix4d(
        float(x_hat[0]), float(x_hat[1]), float(x_hat[2]), 0.0,
        float(y_hat[0]), float(y_hat[1]), float(y_hat[2]), 0.0,
        float(z_hat[0]), float(z_hat[1]), float(z_hat[2]), 0.0,
        float(mid[0]),   float(mid[1]),   float(mid[2]),   1.0,
    )

    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        cyl = UsdGeom.Cylinder(prim)
        cyl.GetHeightAttr().Set(length)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(mat)
    else:
        cyl = UsdGeom.Cylinder.Define(stage, path)
        cyl.GetRadiusAttr().Set(_EXO_CABLE_RADIUS)
        cyl.GetHeightAttr().Set(length)
        cyl.GetDisplayColorAttr().Set([Gf.Vec3f(*[float(c) for c in color_rgb])])
        xf = UsdGeom.Xformable(cyl.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(mat)


class ExoCableVisualizerIsaac:
    """Exosuit-cable visualisation for Isaac Sim (Method B — centred elbow).

    Uses a headless PyDrake plant via :class:`_DrakeExoCablePlant` for
    accurate cable routing + tangent computation, then renders each cable
    segment and pulley wrap arc as USD cylinders in Isaac Sim.

    IMPORTANT: Call ``create_prims()`` BEFORE ``world.reset()`` to pre-
    allocate USD cylinder prims.  Then call ``update()`` during the sim
    loop — it only modifies existing prim transforms, never creates or
    removes prims (which would invalidate Isaac Sim's ArticulationView).

    Usage::

        viz = ExoCableVisualizerIsaac(stage, exo_urdf_path)
        viz.create_prims(q1_init, q2_init)   # BEFORE world.reset()
        world.reset()
        # In sim loop:
        viz.update(q1, q2)
    """

    def __init__(self, stage, drake_urdf_path: str,
                 springs_enabled: bool = True):
        self._stage = stage
        self._drake_urdf = drake_urdf_path
        self._springs_enabled = bool(springs_enabled)
        self._drake_exo: "_DrakeExoCablePlant | None" = None
        print(f"[ExoCableViz] Using headless PyDrake plant with {drake_urdf_path}")

    @property
    def drake_exo(self) -> "_DrakeExoCablePlant":
        """Access the underlying headless PyDrake exo plant."""
        return self._drake_exo

    def create_prims(self, q1: float = 0.0, q2: float = 0.0):
        """Pre-allocate all exo-cable USD cylinder prims.

        Call BEFORE ``world.reset()``.
        """
        self._drake_exo = _DrakeExoCablePlant(
            self._drake_urdf, q1=q1, q2=q2,
            springs_enabled=self._springs_enabled,
        )
        _draw_exo_cables_usd(self._stage, self._drake_exo)
        print(f"[ExoCableViz] Pre-allocated exo-cable prims at "
              f"q=({math.degrees(q1):.1f}°, {math.degrees(q2):.1f}°)")

    def update(self, q1: float, q2: float):
        """Recompute exo-cable routing and update existing prim transforms."""
        if self._drake_exo is None:
            return
        self._drake_exo.update(q1, q2)
        _draw_exo_cables_usd(self._stage, self._drake_exo)


# ============================================================================
# EXO-CABLE VISUALISER  (Method B — centred elbow pulley)
# ============================================================================
# Uses a headless PyDrake plant to compute exo-cable routing (two
# antagonistic cables wrapping through a shared centred elbow pulley) and
# renders them as USD cylinders in Isaac Sim.  Independent of the drive
# CableVisualizerIsaac — both can be used simultaneously.

_EXO_CABLE_ROOT = "/World/ExoCables"
_EXO_CABLE_RADIUS = 0.0005          # 0.5 mm
# Per-route display colours matching the PyDrake Meshcat visualisation
# (see ExoCableRig.exo_cable_right / exo_cable_left in
# ``cable/cable_with_exo_springs_elbow_follow.py``).
_EXO_COLOR_RIGHT = (1.0, 0.55, 0.0)   # orange — right cable (upper groove)
_EXO_COLOR_LEFT  = (0.8, 0.0, 0.8)    # magenta — left  cable (lower groove)


def _exo_route_color(route_index: int):
    """Return RGB tuple for exo-cable route index (0=right, 1=left)."""
    return _EXO_COLOR_RIGHT if route_index == 0 else _EXO_COLOR_LEFT


class _DrakeExoCablePlant:
    """Headless PyDrake plant for exo-cable FK + tangent computation.

    Builds a MultibodyPlant using the Exo URDF + CupManipulatorTendonWithExo
    wrapper so that BOTH drive cables and exo cables share the same joint
    state.  Only the exo rig is exposed — drive cables are handled by the
    regular :class:`CableVisualizerIsaac`.
    """

    def __init__(self, drake_urdf: str, q1: float = 0.0, q2: float = 0.0,
                 springs_enabled: bool = True):
        # Local import keeps Drake off the Isaac-Sim import hot path when
        # only the drive visualiser is used.
        from pydrake.all import (
            DiagramBuilder, MultibodyPlant, SceneGraph, Parser,
        )
        from robots.cup_manipulator_tendon_with_exo import (
            CupManipulatorTendonWithExo,
        )
        from robots.cup_manipulator_tendon import create_cable_manipulator_config

        config = create_cable_manipulator_config(
            urdf_path=drake_urdf,
            joint_angles={"link1_base": q1, "link2_link1": q2},
        )
        builder = DiagramBuilder()
        plant = MultibodyPlant(time_step=0.001)
        scene_graph = builder.AddSystem(SceneGraph())
        plant.RegisterAsSourceForSceneGraph(scene_graph)

        manipulator = CupManipulatorTendonWithExo(
            config, enable_visualization=False,
        )
        parser = Parser(plant)
        manipulator.load_urdf_to_plant(plant, parser)
        manipulator.weld_base_to_world(plant)
        manipulator.add_end_effector_frame(plant)
        plant.Finalize()
        builder.AddSystem(plant)

        self.plant = plant
        self.manipulator = manipulator
        self.diagram = builder.Build()
        self._root_ctx = self.diagram.CreateDefaultContext()
        self.plant_ctx = plant.GetMyMutableContextFromRoot(self._root_ctx)

        # Initialise both cable rigs (drive + exo).  Drive rig is built so
        # that callers who want drive cables rendered can reuse the same
        # plant for both — but in practice we only return the exo rig here.
        manipulator.init_cable_rig(drake_urdf, springs_enabled=springs_enabled)
        manipulator.init_exo_cable_rig(drake_urdf, springs_enabled=springs_enabled)

        self.rig = manipulator.exo_rig
        self._set_angles(q1, q2)
        manipulator.compute_exo_tangents(plant, self.plant_ctx)

    def _set_angles(self, q1: float, q2: float):
        import numpy as _np
        self.manipulator.set_positions_user_order(
            self.plant, self.plant_ctx,
            {"link1_base": q1, "link2_link1": q2},
        )
        self.plant.SetVelocities(
            self.plant_ctx, _np.zeros(self.plant.num_velocities()),
        )

    def update(self, q1: float, q2: float):
        """Sync joint angles and recompute exo-cable tangents."""
        self._set_angles(q1, q2)
        self.manipulator.compute_exo_tangents(self.plant, self.plant_ctx)

    def get_cable_world_points(self):
        """Return list of (route, world_pts) for each exo cable."""
        return [
            (route, route.world_points(self.plant, self.plant_ctx, self.manipulator))
            for route in self.rig.routes
        ]

    def get_wrap_arcs(self, n_arc_pts: int = 24):
        """Return wrap-arc world-frame points for each exo pulley wrap.

        Yields ``(label, color_rgb, arc_pts_Nx3)`` tuples.  Uses the
        per-route ``wrap_arcs`` metadata built by
        ``ExoCableRig.compute_tangents`` (see
        ``cable/cable_with_exo_springs_elbow_follow.py``).
        """
        import numpy as _np

        def _Xw(body_name):
            body = self.plant.GetBodyByName(
                body_name, self.manipulator.model_instance,
            )
            X = self.plant.CalcRelativeTransform(
                self.plant_ctx, self.plant.world_frame(), body.body_frame(),
            )
            return X.rotation().matrix(), X.translation()

        results = []
        for entry in getattr(self.rig, "wrap_arcs", []):
            # Tuple format: (pulley, A_body, B_body, path, rgba, center_override_body)
            if len(entry) >= 6:
                pulley, A_body, B_body, path, rgba, center_override = entry
            else:
                pulley, A_body, B_body, path, rgba = entry[:5]
                center_override = None
            if A_body is None or B_body is None:
                continue
            R_wb, t_wb = _Xw(pulley.body_name)
            centroid_body = (center_override if center_override is not None
                             else pulley.centroid)
            center_w = R_wb @ _np.asarray(centroid_body) + t_wb
            shaft_w  = R_wb @ pulley.shaft_axis_body
            A_w = R_wb @ _np.asarray(A_body) + t_wb
            B_w = R_wb @ _np.asarray(B_body) + t_wb
            ax = shaft_w / _np.linalg.norm(shaft_w)
            dA = A_w - center_w
            dA -= _np.dot(dA, ax) * ax
            dA /= _np.linalg.norm(dA)
            dB = B_w - center_w
            dB -= _np.dot(dB, ax) * ax
            dB /= _np.linalg.norm(dB)
            cos_ab = float(_np.clip(_np.dot(dA, dB), -1.0, 1.0))
            angle = _np.sign(_np.dot(_np.cross(dA, dB), ax)) * _np.arccos(cos_ab)
            ax_cross_dA = _np.cross(ax, dA)
            arc_pts = _np.array([
                center_w + pulley.radius * (dA * _np.cos(th)
                                            + ax_cross_dA * _np.sin(th))
                for th in _np.linspace(0.0, angle, n_arc_pts)
            ])
            # Convert meshcat Rgba to RGB tuple for USD.
            try:
                color_rgb = (float(rgba.r()), float(rgba.g()), float(rgba.b()))
            except Exception:
                color_rgb = _EXO_COLOR_RIGHT
            # Extract a short label from the full meshcat path.
            label = path.strip("/").replace("/", "_")
            results.append((label, color_rgb, arc_pts))
        return results


def _draw_exo_cables_usd(stage, drake_exo: "_DrakeExoCablePlant"):
    """Draw exo-cable segments + wrap arcs as USD Cylinder prims."""
    # Straight segments between waypoints
    for ri, (route, pts) in enumerate(drake_exo.get_cable_world_points()):
        skip = getattr(route, "skip_chord_segments", frozenset())
        color = _exo_route_color(ri)
        base = f"{_EXO_CABLE_ROOT}/{route.meshcat_path.replace('/', '_').strip('_')}"
        for i, (p0, p1) in enumerate(zip(pts[:-1], pts[1:])):
            if i in skip:
                continue
            _usd_cylinder_exo(stage, f"{base}/seg{i:02d}", p0, p1, color)

    # Wrap arcs around pulleys
    for label, color, arc_pts in drake_exo.get_wrap_arcs():
        base = f"{_EXO_CABLE_ROOT}/wrap_{label}"
        for i, (p0, p1) in enumerate(zip(arc_pts[:-1], arc_pts[1:])):
            _usd_cylinder_exo(stage, f"{base}/arc{i:02d}", p0, p1, color)


def _usd_cylinder_exo(stage, path: str, p0: np.ndarray, p1: np.ndarray, color_rgb):
    """Create or update a thin USD cylinder for an exo-cable segment.

    Same as ``_usd_cylinder`` but uses ``_EXO_CABLE_RADIUS`` so exo cables
    can be styled independently of drive cables.
    """
    diff = p1 - p0
    length = float(np.linalg.norm(diff))
    if length < 1e-9:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Cylinder(prim).GetHeightAttr().Set(0.0)
        return
    mid = (p0 + p1) * 0.5

    z_hat = diff / length
    tmp = np.array([0., 1., 0.]) if abs(z_hat[0]) > 0.9 else np.array([1., 0., 0.])
    x_hat = np.cross(tmp, z_hat)
    x_hat /= np.linalg.norm(x_hat)
    y_hat = np.cross(z_hat, x_hat)
    mat = Gf.Matrix4d(
        float(x_hat[0]), float(x_hat[1]), float(x_hat[2]), 0.0,
        float(y_hat[0]), float(y_hat[1]), float(y_hat[2]), 0.0,
        float(z_hat[0]), float(z_hat[1]), float(z_hat[2]), 0.0,
        float(mid[0]),   float(mid[1]),   float(mid[2]),   1.0,
    )

    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        cyl = UsdGeom.Cylinder(prim)
        cyl.GetHeightAttr().Set(length)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(mat)
    else:
        cyl = UsdGeom.Cylinder.Define(stage, path)
        cyl.GetRadiusAttr().Set(_EXO_CABLE_RADIUS)
        cyl.GetHeightAttr().Set(length)
        cyl.GetDisplayColorAttr().Set([Gf.Vec3f(*[float(c) for c in color_rgb])])
        xf = UsdGeom.Xformable(cyl.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(mat)


class ExoCableVisualizerIsaac:
    """Exosuit-cable visualisation for Isaac Sim (Method B — centred elbow).

    Uses a headless PyDrake plant via :class:`_DrakeExoCablePlant` for
    accurate cable routing + tangent computation, then renders each cable
    segment and pulley wrap arc as USD cylinders in Isaac Sim.

    IMPORTANT: Call ``create_prims()`` BEFORE ``world.reset()`` to pre-
    allocate USD cylinder prims.  Then call ``update()`` during the sim
    loop — it only modifies existing prim transforms, never creates or
    removes prims (which would invalidate Isaac Sim's ArticulationView).

    Usage::

        viz = ExoCableVisualizerIsaac(stage, exo_urdf_path)
        viz.create_prims(q1_init, q2_init)   # BEFORE world.reset()
        world.reset()
        # In sim loop:
        viz.update(q1, q2)
    """

    def __init__(self, stage, drake_urdf_path: str,
                 springs_enabled: bool = True):
        self._stage = stage
        self._drake_urdf = drake_urdf_path
        self._springs_enabled = bool(springs_enabled)
        self._drake_exo: "_DrakeExoCablePlant | None" = None
        print(f"[ExoCableViz] Using headless PyDrake plant with {drake_urdf_path}")

    @property
    def drake_exo(self) -> "_DrakeExoCablePlant":
        """Access the underlying headless PyDrake exo plant."""
        return self._drake_exo

    def create_prims(self, q1: float = 0.0, q2: float = 0.0):
        """Pre-allocate all exo-cable USD cylinder prims.

        Call BEFORE ``world.reset()``.
        """
        self._drake_exo = _DrakeExoCablePlant(
            self._drake_urdf, q1=q1, q2=q2,
            springs_enabled=self._springs_enabled,
        )
        _draw_exo_cables_usd(self._stage, self._drake_exo)
        print(f"[ExoCableViz] Pre-allocated exo-cable prims at "
              f"q=({math.degrees(q1):.1f}°, {math.degrees(q2):.1f}°)")

    def update(self, q1: float, q2: float):
        """Recompute exo-cable routing and update existing prim transforms."""
        if self._drake_exo is None:
            return
        self._drake_exo.update(q1, q2)
        _draw_exo_cables_usd(self._stage, self._drake_exo)


# ============================================================================
# EXO SPRING HELIX VISUALIZER
# ============================================================================
#
# Mirrors the Meshcat helix in the PyDrake exo script.  Each exo cable carries
# a physical Bowden-housed spring whose extension δ_R / δ_L (returned by the
# SEA-exo actuator every step) drives how compressed or stretched the helix
# should look in the scene.  We pre-allocate a fixed number of short USD
# cylinder segments per spring and only update their transforms each step —
# never creating or deleting prims after world.reset() (which would corrupt
# Isaac Sim's PhysX view).
#
# Mapping δ → visual spring fraction of the last exo cable segment:
#   • At δ = 0  (slack / just contact) the coil takes ≈ _REST_FRAC of segment.
#   • At δ = δ_max it takes ≈ _MAX_FRAC.
#   • Clamped [0.05, 0.70] so coils never explode off the geometry.
#
# This is purely cosmetic — it has no effect on physics.

from cable import spring_zigzag_points as _spring_zigzag_points

_EXO_SPRING_ROOT       = "/World/ExoSprings"
_EXO_SPRING_RADIUS     = 0.0010            # coil wire radius [m]
_EXO_SPRING_N_COILS    = 8
_EXO_SPRING_AMPLITUDE  = 0.0055            # helix radius [m]
_EXO_SPRING_COLOR_R    = (1.00, 0.55, 0.00)   # right — warm orange
_EXO_SPRING_COLOR_L    = (0.80, 0.20, 0.80)   # left  — magenta
_EXO_REST_FRAC         = 0.20
_EXO_MAX_FRAC          = 0.75


def _usd_helix_cylinder(stage, path: str, p0: np.ndarray, p1: np.ndarray,
                        color_rgb, radius: float):
    """Create or update a thin USD cylinder for one helix segment."""
    diff = p1 - p0
    length = float(np.linalg.norm(diff))
    if length < 1e-9:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Cylinder(prim).GetHeightAttr().Set(0.0)
        return
    mid = (p0 + p1) * 0.5
    z_hat = diff / length
    tmp = np.array([0., 1., 0.]) if abs(z_hat[0]) > 0.9 else np.array([1., 0., 0.])
    x_hat = np.cross(tmp, z_hat)
    n = np.linalg.norm(x_hat)
    if n > 1e-9:
        x_hat /= n
    y_hat = np.cross(z_hat, x_hat)
    mat = Gf.Matrix4d(
        float(x_hat[0]), float(x_hat[1]), float(x_hat[2]), 0.0,
        float(y_hat[0]), float(y_hat[1]), float(y_hat[2]), 0.0,
        float(z_hat[0]), float(z_hat[1]), float(z_hat[2]), 0.0,
        float(mid[0]),   float(mid[1]),   float(mid[2]),   1.0,
    )
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        cyl = UsdGeom.Cylinder(prim)
        cyl.GetHeightAttr().Set(length)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(mat)
    else:
        cyl = UsdGeom.Cylinder.Define(stage, path)
        cyl.GetRadiusAttr().Set(radius)
        cyl.GetHeightAttr().Set(length)
        cyl.GetDisplayColorAttr().Set([Gf.Vec3f(*[float(c) for c in color_rgb])])
        xf = UsdGeom.Xformable(cyl.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(mat)


class ExoSpringVisualizerIsaac:
    """Helix visualisation for the two exo springs (right + left).

    Animates each spring's visual length from the SEA-exo spring extension
    δ_R, δ_L produced by :class:`actuators.sea_exo_isaacsim.SEAExoActuatorNP`.
    Mirrors the Meshcat helices in the PyDrake exo script.

    Attach to an existing :class:`ExoCableVisualizerIsaac` — we read its
    current exo-cable world points (last segment per route) and lay the
    coil along that segment.

    Pre-allocation order::

        exo_viz        = ExoCableVisualizerIsaac(stage, URDF)
        exo_viz.create_prims(q1, q2)
        exo_spring_viz = ExoSpringVisualizerIsaac(stage, exo_viz,
                                                  k_exo, r_exo, tau_max)
        exo_spring_viz.create_prims()
        world.reset()
        # in sim loop:
        exo_spring_viz.update(delta_R, delta_L)
    """

    def __init__(self, stage, exo_cable_viz: "ExoCableVisualizerIsaac",
                 k_exo: float = 8000.0, r_exo: float = 0.04775,
                 tau_max: float = 25.0):
        self._stage = stage
        self._exo_cable_viz = exo_cable_viz
        self._n_coils = _EXO_SPRING_N_COILS
        self._amp = _EXO_SPRING_AMPLITUDE
        # δ_max = max spring force / k_exo.  max force for one side when it
        # produces torque τ_max is F = τ_max / r_exo → δ = τ_max/(k_exo·r_exo).
        self._delta_max = float(tau_max) / max(float(k_exo) * float(r_exo), 1e-9)
        self._prims_ready = False

    def _last_segment_endpoints(self):
        """Return list of (right_pts, left_pts) tuples — per route.

        We use the segment just before the final endpoint on link2 as the
        "spring region" of each exo cable.  Matches the PyDrake helix
        placement convention.
        """
        if self._exo_cable_viz is None or self._exo_cable_viz.drake_exo is None:
            return []
        routes = list(self._exo_cable_viz.drake_exo.get_cable_world_points())
        out = []
        for ri, (route, pts) in enumerate(routes):
            if len(pts) < 2:
                continue
            out.append((ri, route, pts[-2], pts[-1]))
        return out

    def create_prims(self):
        """Pre-allocate USD helix cylinder prims — call BEFORE world.reset()."""
        entries = self._last_segment_endpoints()
        if not entries:
            print("[ExoSpringViz] No exo routes found — helix viz disabled.")
            return
        for ri, route, p0, p1 in entries:
            color = _EXO_SPRING_COLOR_R if ri == 0 else _EXO_SPRING_COLOR_L
            label = "right" if ri == 0 else "left"
            seg_dir = p1 - p0
            seg_len = float(np.linalg.norm(seg_dir))
            if seg_len < 1e-9:
                continue
            mid = 0.5
            half = _EXO_REST_FRAC / 2.0
            # lay helix between (mid - half) and (mid + half) of segment
            ps = p0 + (mid - half) * seg_dir
            pe = p0 + (mid + half) * seg_dir
            zz = _spring_zigzag_points(
                ps, pe, n_coils=self._n_coils, amplitude=self._amp,
            )
            base = f"{_EXO_SPRING_ROOT}/{label}"
            for j in range(len(zz) - 1):
                _usd_helix_cylinder(
                    self._stage, f"{base}/coil{j:03d}",
                    zz[j], zz[j + 1], color, _EXO_SPRING_RADIUS,
                )
        self._prims_ready = True
        print(
            f"[ExoSpringViz] Pre-allocated helix prims for {len(entries)} exo "
            f"spring(s); δ_max = {self._delta_max*1e3:.2f} mm.")

    def update(self, delta_R: float = 0.0, delta_L: float = 0.0):
        """Re-lay each helix along its current exo-cable segment.

        ``delta_R`` / ``delta_L`` are the exo spring extensions in metres
        (``exo_diag.delta_R`` / ``exo_diag.delta_L``).  The helix length
        grows linearly with |δ|.
        """
        if not self._prims_ready:
            return
        entries = self._last_segment_endpoints()
        deltas = (delta_R, delta_L)
        for ri, route, p0, p1 in entries:
            color = _EXO_SPRING_COLOR_R if ri == 0 else _EXO_SPRING_COLOR_L
            label = "right" if ri == 0 else "left"
            d = float(deltas[ri]) if ri < len(deltas) else 0.0
            seg_dir = p1 - p0
            seg_len = float(np.linalg.norm(seg_dir))
            if seg_len < 1e-9:
                continue
            norm = min(abs(d) / max(self._delta_max, 1e-9), 1.0)
            frac = _EXO_REST_FRAC + norm * (_EXO_MAX_FRAC - _EXO_REST_FRAC)
            frac = float(np.clip(frac, 0.05, 0.80))
            half = frac / 2.0
            mid = 0.5
            ps = p0 + (mid - half) * seg_dir
            pe = p0 + (mid + half) * seg_dir
            zz = _spring_zigzag_points(
                ps, pe, n_coils=self._n_coils, amplitude=self._amp,
            )
            base = f"{_EXO_SPRING_ROOT}/{label}"
            for j in range(len(zz) - 1):
                _usd_helix_cylinder(
                    self._stage, f"{base}/coil{j:03d}",
                    zz[j], zz[j + 1], color, _EXO_SPRING_RADIUS,
                )
