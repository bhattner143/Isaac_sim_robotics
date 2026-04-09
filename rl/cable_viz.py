"""
rl/cable_viz.py
───────────────
Cable (tendon) visualization for Isaac Sim using ``DrakeCablePlant`` from
``cable.py`` for accurate cable routing + tangent computation.

Uses the same headless Drake MultibodyPlant FK that the PyDrake and Newton
scripts rely on — guaranteeing identical cable geometry across all viewers.

IMPORTANT: Call ``create_prims()`` BEFORE ``world.reset()`` to pre-allocate
USD cylinder prims.  Then call ``update()`` during the sim loop — it only
modifies existing prim transforms, never creates or removes prims (which
would invalidate Isaac Sim's ArticulationView / physics view).

Usage::

    from rl.cable_viz import CableVisualizerIsaac
    cable_viz = CableVisualizerIsaac(stage, drake_urdf_path)
    cable_viz.create_prims(q1_init, q2_init)   # BEFORE world.reset()
    world.reset()
    # In sim loop:
    cable_viz.update(q1, q2)
"""

import math
import numpy as np
from pathlib import Path

# USD imports (available after SimulationApp)
from pxr import UsdGeom, Gf

# Cable FK — headless Drake plant (same as scene-viz and PyDrake scripts)
from cable import DrakeCablePlant


# ============================================================================
# USD CYLINDER RENDERING — same as scene-viz Isaac Sim script
# ============================================================================

_CABLE_ROOT = "/World/Cables"
_CABLE_RADIUS = 0.0005


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


def _draw_cables_usd(stage, drake_cable: DrakeCablePlant):
    """Draw all cable segments and wrap arcs as USD Cylinder prims.

    Identical to the draw_cables_usd() in the scene-viz Isaac Sim script.
    """
    for route, pts in drake_cable.get_cable_world_points():
        skip = getattr(route, "skip_chord_segments", frozenset())
        color = _route_color(route)
        base = f"{_CABLE_ROOT}/{route.meshcat_path.replace('/', '_').strip('_')}"
        for i, (p0, p1) in enumerate(zip(pts[:-1], pts[1:])):
            if i in skip:
                continue
            _usd_cylinder(stage, f"{base}/seg{i:02d}", p0, p1, color)

    for label, color, arc_pts in drake_cable.get_wrap_arcs():
        base = f"{_CABLE_ROOT}/wrap_{label}"
        for i, (p0, p1) in enumerate(zip(arc_pts[:-1], arc_pts[1:])):
            _usd_cylinder(stage, f"{base}/arc{i:02d}", p0, p1, color)


# ============================================================================
# HIGH-LEVEL VISUALIZER CLASS
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

    def create_prims(self, q1: float = 0.0, q2: float = 0.0):
        """Pre-allocate all cable USD cylinder prims.

        Call this BEFORE ``world.reset()`` so prims are part of the initial
        stage composition and don't invalidate the physics view later.
        """
        self._drake_cable = DrakeCablePlant(self._drake_urdf, q1=q1, q2=q2)
        _draw_cables_usd(self._stage, self._drake_cable)
        print(f"[CableViz] Pre-allocated cable prims at q=({math.degrees(q1):.1f}°, {math.degrees(q2):.1f}°)")

    def update(self, q1: float, q2: float):
        """Recompute cable routing and update existing USD cylinder transforms.

        Only updates transforms of existing prims — never creates or removes
        prims. Safe to call during simulation without invalidating physics.
        """
        self._drake_cable.update(q1, q2)
        _draw_cables_usd(self._stage, self._drake_cable)
