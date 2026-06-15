"""Meshcat / Drake 3D scene for MHP cable routing."""
from __future__ import annotations

import os

import numpy as np

try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False

try:
    from pydrake.all import (
        StartMeshcat as _DrakeStartMeshcat,
        MeshcatVisualizer,
        MeshcatVisualizerParams,
        DiagramBuilder,
        AddMultibodyPlantSceneGraph,
        Parser,
        Role,
    )
    from pydrake.math import RigidTransform, RotationMatrix
    from pydrake.geometry import Cylinder as _DrakeCylinder, Rgba as _DrakeRgba
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False

from cable.path_mhp import compute_cable_path
from cable.types_mhp import CableRouteConfig, MHPKinematics

_DEFAULT_VIS_PREFIX = "visualizer"


def _configure_urdf_alpha(params: "MeshcatVisualizerParams", alpha: float) -> None:
    """Enable Drake's built-in alpha handling for URDF illustration geometry.

    OBJ meshes loaded from URDF use Phong materials with alpha=1. Drake's
    MeshcatVisualizer only calls ``SetAlphas()`` (per-geometry ``modulated_opacity``)
    when ``enable_alpha_slider`` is True.  A bare ``SetProperty`` on the visualizer
    prefix does not affect those meshes.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha >= 1.0:
        return
    params.enable_alpha_slider = True
    params.initial_alpha_slider_value = alpha

def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4×4 homogeneous transform from a 3×3 rotation and 3-vec translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def cylinder_transform(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """4×4 transform that maps a Y-aligned cylinder to the segment p1→p2.
    
    For raw meshcat Cylinder(length, radius), which is Y-axis aligned.
    """
    mid = (p1 + p2) / 2.0
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-9:
        return make_transform(np.eye(3), mid)
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
    return make_transform(R, mid)


def cylinder_transform_drake(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """4×4 transform that maps a Z-aligned Drake cylinder to the segment p1→p2.
    
    For Drake's Cylinder(radius, length), which is Z-axis aligned (height along Z).
    """
    mid = (p1 + p2) / 2.0
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-9:
        return make_transform(np.eye(3), mid)
    direction = direction / length

    # Rotate Z-axis → direction (Drake cylinder is Z-aligned)
    z_axis = np.array([0., 0., 1.])
    dot = np.clip(np.dot(z_axis, direction), -1.0, 1.0)
    if 1.0 - dot < 1e-8:          # already aligned
        R = np.eye(3)
    elif 1.0 + dot < 1e-8:        # anti-aligned (pointing down)
        R = np.diag([-1., -1., 1.]).astype(float)
    else:
        axis = np.cross(z_axis, direction)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(dot)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return make_transform(R, mid)


def _draw_cables_drake(scene: dict, kin: "MHPKinematics",
                       lower: CableRouteConfig, upper: CableRouteConfig) -> None:
    """(Re)draw the cable tubes on the Drake Meshcat server.

    Physical components (spool, guide pulleys, elbow roller, ball markers) are
    NOT overlaid here — they are already rendered correctly by Drake's
    MeshcatVisualizer straight from the URDF, so overlaying them again at the
    cable-routing positions would place them at the wrong location/orientation.
    Only the cable centre-line tubes are drawn on top of the real robot.
    """
    drake_mc = scene["drake_mc"]
    # Clear any cable tubes from the previous pose so stale segments disappear.
    drake_mc.Delete("cables")

    CABLE_RADIUS = 0.0008
    for route in (lower, upper):
        cp = compute_cable_path(route, kin, cable_loc=route.path[0].cable)
        hex_str = route.color.lstrip('#')
        cable_rgba = _DrakeRgba(
            int(hex_str[0:2], 16) / 255.0,
            int(hex_str[2:4], 16) / 255.0,
            int(hex_str[4:6], 16) / 255.0,
            1.0,
        )
        seg_idx = 0
        for piece in cp.pieces:
            pts = piece
            for k in range(len(pts) - 1):
                p1, p2  = pts[k].astype(float), pts[k + 1].astype(float)
                seg_len = np.linalg.norm(p2 - p1)
                if seg_len < 1e-6:
                    continue
                T44 = cylinder_transform_drake(p1, p2)  # Drake cylinders are Z-aligned
                X   = RigidTransform(RotationMatrix(T44[:3, :3]), T44[:3, 3])
                path = f"cables/{route.name}/seg_{seg_idx}"
                drake_mc.SetObject(path, _DrakeCylinder(CABLE_RADIUS, seg_len), cable_rgba)
                drake_mc.SetTransform(path, X)
                seg_idx += 1
        print(f"  Cable '{route.name}': {seg_idx} segments")


def _draw_fallback_full(scene: dict, kin: "MHPKinematics",
                        lower: CableRouteConfig, upper: CableRouteConfig) -> None:
    """(Re)draw the full approximate scene on a raw meshcat server (no Drake).

    Because there is no articulated URDF robot in this mode, the arm skeleton,
    physical components and cable tubes are all redrawn from scratch each pose.
    """
    import os
    vis        = scene["vis"]
    assets_dir = scene["assets_dir"]

    # ── OBJ meshes for spools, guide pulleys, elbow roller ─────────────────────
    seen_names: set = set()   # deduplicate shared components
    for route in (lower, upper):
        for comp in route.physical:
            node_name = f"{comp.role}_{comp.name}"
            if node_name in seen_names:
                continue
            seen_names.add(node_name)

            pw = kin.to_world(comp.pos_in_link, comp.link)
            R  = kin.link_frame_rotation(comp.link)
            T  = make_transform(R, pw)

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

            # Fallback: sphere sized to component radius
            r = comp.diameter_mm * 0.5e-3
            vis[node_name].set_object(g.Sphere(r), material)
            vis[node_name].set_transform(T)

    # ── Arm skeleton (approximate, since no URDF/Drake available) ─────────────
    base  = np.array([0., 0., 0.])
    J1    = kin.J1
    J2    = kin.J2
    EE    = kin.to_world(np.array([0.2, 0., 0.]), "lower_arm")
    bone_material = g.MeshPhongMaterial(color=0x2266aa, opacity=0.15)

    for i, (p1, p2) in enumerate([(base, J1), (J1, J2), (J2, EE)]):
        length = np.linalg.norm(p2 - p1)
        cyl    = g.Cylinder(length, 0.008)
        vis[f"arm_{i}"].set_object(cyl, bone_material)
        vis[f"arm_{i}"].set_transform(cylinder_transform(p1, p2))

    joint_mat = g.MeshPhongMaterial(color=0xff8800, opacity=0.9)
    for i, pt in enumerate([J1, J2]):
        vis[f"joint_{i}"].set_object(g.Sphere(0.012), joint_mat)
        vis[f"joint_{i}"].set_transform(make_transform(np.eye(3), pt))

    # ── Cable tubes ───────────────────────────────────────────────────────────
    CABLE_RADIUS = 0.0005
    for route in (lower, upper):
        cp        = compute_cable_path(route, kin, cable_loc=route.path[0].cable)
        color_val = int(route.color.lstrip('#'), 16)
        cable_mat = g.MeshPhongMaterial(color=color_val, opacity=1.0)
        # Clear stale cable segments from the previous pose.
        vis[f"cable_{route.name}"].delete()
        seg_idx = 0
        for piece in cp.pieces:
            pts = piece
            for k in range(len(pts) - 1):
                p1, p2 = pts[k].astype(float), pts[k + 1].astype(float)
                seg_len = np.linalg.norm(p2 - p1)
                if seg_len < 1e-6:
                    continue
                cyl = g.Cylinder(seg_len, CABLE_RADIUS)
                vis[f"cable_{route.name}/seg_{seg_idx}"].set_object(cyl, cable_mat)
                vis[f"cable_{route.name}/seg_{seg_idx}"].set_transform(
                    cylinder_transform(p1, p2)
                )
                seg_idx += 1


def build_meshcat_scene(
    lower: CableRouteConfig,
    upper: CableRouteConfig,
    urdf_alpha: float = 0.3,
) -> dict | None:
    """Build the Meshcat scene ONCE and return a reusable handle dict.

    When pydrake is available the real URDF robot is loaded via Drake's
    MeshcatVisualizer (mode='drake'); the returned handle keeps the diagram,
    context and plant so the robot can be re-posed interactively.  Otherwise a
    raw meshcat server is created (mode='meshcat') and the scene is rebuilt on
    each pose update.

    Returns ``None`` if neither backend is available.
    """
    if not DRAKE_AVAILABLE and not MESHCAT_AVAILABLE:
        print("⚠️  Neither pydrake nor meshcat available. Install either.")
        return None

    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assets and URDF live in the MHP folder, one level up from cable/
    mhp_dir    = os.path.join(os.path.dirname(script_dir),
                               'model_using_onshape_to_robot',
                               'manipulator_hybrid_planar_fusion')
    assets_dir = os.path.join(mhp_dir, 'assets')
    urdf_path  = os.path.join(mhp_dir, 'manipulator_hybrid_planar_fusion_obj.urdf')

    # ── Drake path: real URDF robot ───────────────────────────────────────────
    if DRAKE_AVAILABLE and os.path.exists(urdf_path):
        drake_mc = _DrakeStartMeshcat()
        print(f"\n✓ Meshcat: {drake_mc.web_url()}")

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        parser = Parser(plant)
        parser.package_map().Add("assets", assets_dir)
        model = parser.AddModels(urdf_path)[0]
        plant.WeldFrames(
            plant.world_frame(),
            plant.GetFrameByName("base_link_aka_shoulder_transmission", model),
        )
        plant.Finalize()

        vis_params = MeshcatVisualizerParams(role=Role.kIllustration)
        vis_params.prefix = _DEFAULT_VIS_PREFIX
        _configure_urdf_alpha(vis_params, urdf_alpha)
        MeshcatVisualizer.AddToBuilder(
            builder, scene_graph.get_query_output_port(), drake_mc, vis_params
        )

        diagram   = builder.Build()
        context   = diagram.CreateDefaultContext()
        plant_ctx = plant.GetMyMutableContextFromRoot(context)

        # Render once to populate meshcat tree (SetAlphas runs inside visualizer)
        diagram.ForcedPublish(context)

        if urdf_alpha < 1.0:
            print(f"  ✓ URDF transparency set to {urdf_alpha * 100:.0f}%"
                  f"  (Meshcat slider: '{vis_params.prefix} α')")

        return {
            "mode": "drake", "drake_mc": drake_mc, "diagram": diagram,
            "context": context, "plant": plant, "plant_ctx": plant_ctx,
            "model": model, "assets_dir": assets_dir,
            "urdf_alpha": urdf_alpha,
            "vis_prefix": vis_params.prefix,
        }

    # ── Fallback: raw meshcat (no articulated robot) ──────────────────────────
    if not MESHCAT_AVAILABLE:
        print("⚠️  Meshcat not available. Install: pip install meshcat")
        return None

    vis = meshcat.Visualizer()
    try:
        url = vis.url()
    except Exception:
        url = "http://127.0.0.1:7001/static/"
    print(f"\n✓ Meshcat (approx skeleton): {url}")
    return {"mode": "meshcat", "vis": vis, "assets_dir": assets_dir}


def update_meshcat_pose(
    scene: dict,
    lower: CableRouteConfig,
    upper: CableRouteConfig,
    q1: float,
    q2: float,
) -> None:
    """Re-pose the robot to (q1, q2) [rad] and redraw the cables.

    In Drake mode the articulated URDF robot is moved via ``SetPositions`` and
    the cable tubes are redrawn on top.  In fallback mode the whole approximate
    scene is rebuilt.
    """
    if scene is None:
        return
    kin = MHPKinematics(q1, q2)

    if scene["mode"] == "drake":
        plant, plant_ctx = scene["plant"], scene["plant_ctx"]
        diagram, context = scene["diagram"], scene["context"]
        model = scene["model"]
        # Drake position order (verified empirically):
        #   position[0] = jt_upper_base  (shoulder = q1)
        #   position[1] = jt_lower_upper (elbow    = q2)
        plant.SetPositions(plant_ctx, model, np.array([q1, q2]))
        diagram.ForcedPublish(context)
        print(f"  Robot at q1={np.rad2deg(q1):.1f}°  q2={np.rad2deg(q2):.1f}°")
        _draw_cables_drake(scene, kin, lower, upper)
    else:
        _draw_fallback_full(scene, kin, lower, upper)
        vis = scene["vis"]
        robot_cx = float((kin.J1[0] + kin.J2[0]) / 2)
        robot_cz = float((kin.J1[2] + kin.J2[2]) / 2)
        vis["/Cameras/default"].set_transform(
            tf.translation_matrix([robot_cx, 0.0, robot_cz])
        )
        vis["/Cameras/default"].set_property("zoom", 30)
        print(f"  q1={np.rad2deg(q1):.1f}°  q2={np.rad2deg(q2):.1f}°")


def update_mhp_cable_tubes(
    drake_mc,
    lower: CableRouteConfig,
    upper: CableRouteConfig,
    q1: float,
    q2: float,
) -> None:
    """Redraw MHP cable centre-line tubes on an existing Drake Meshcat server."""
    kin = MHPKinematics(q1, q2)
    _draw_cables_drake({"drake_mc": drake_mc, "mode": "drake"}, kin, lower, upper)


def visualize_cable_routing_meshcat(
    lower: CableRouteConfig,
    upper: CableRouteConfig,
    q1: float = 0.0,
    q2: float = 0.0,
):
    """Build the Meshcat scene and render a single pose (backward-compatible).

    Thin wrapper around :func:`build_meshcat_scene` + :func:`update_meshcat_pose`.
    Returns the scene handle dict (or ``None``) so callers can keep the server
    alive and re-pose the robot interactively via :func:`update_meshcat_pose`.
    """
    scene = build_meshcat_scene(lower, upper)
    if scene is None:
        return None
    update_meshcat_pose(scene, lower, upper, q1, q2)
    backend_mc = scene.get("drake_mc") or scene.get("vis")
    url = backend_mc.web_url() if scene["mode"] == "drake" else scene["vis"].url()
    print(f"\n  Open browser → {url}")
    return scene


