#!/usr/bin/env python3
"""
viz_cables.py
─────────────
Drawing and visualization functions for the cable manipulator.

All data (cable_routes, cable_waypoints, pulley instances) is passed as explicit
arguments — no module-level state is defined here.
Classes and module-level instances live in test_drive_pulley.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pydrake.all import (
    RigidTransform,
    RotationMatrix,
)
from pydrake.geometry import Rgba, Cylinder
from termcolor import colored


def print_cable_routing_points(plant, plant_context, manipulator, rig) -> None:
    """Print all cable contact points in world frame for each route."""
    print("\n" + "=" * 64)
    print("  Cable routing points (world frame)")
    print("=" * 64)
    for route in rig.routes:
        pts = route.world_points(plant, plant_context, manipulator)  # (N, 3)
        print(f"\n  [{route.label}]")
        seg_names = [cfg.label for cfg, _ in route.segments]
        col_w = max(len(n) for n in seg_names) + 2
        print(f"  {'Waypoint':<{col_w}}  {'x':>10}  {'y':>10}  {'z':>10}  (m)")
        print(f"  {'-' * col_w}  {'----------'}  {'----------'}  {'----------'}")
        for name, pt in zip(seg_names, pts):
            print(f"  {name:<{col_w}}  {pt[0]:>10.6f}  {pt[1]:>10.6f}  {pt[2]:>10.6f}")
    print("=" * 64 + "\n")

# ── Shared FK helper ──────────────────────────────────────────────────────
def _Xw(plant, manipulator, plant_context, body_name):
    body = plant.GetBodyByName(body_name, manipulator.model_instance)
    X    = plant.CalcRelativeTransform(plant_context, plant.world_frame(),
                                        body.body_frame())
    return X.rotation().matrix(), X.translation()

def draw_cables(meshcat, plant, plant_context, manipulator, rig,
                cable_radius: float = 0.0005, n_arc_pts: int = 32,
                spring_extension: float = 0.0) -> None:
    """Draw both tendon cables in Meshcat: straight segments and pulley wrap arcs.

    Parameters
    ----------
    spring_extension : float
        Current spring extension δ [m] from the SEA controller.  When non-zero
        the helical spring visualization stretches/compresses proportionally:
            visual_length = rest_length + |δ|
        so the coil spacing in Meshcat matches the actual physics state.
    """

    # ── Shared cylinder-placement helper ─────────────────────────────────────
    def _place_cylinder(path, p0, p1, rgba):
        diff   = p1 - p0
        length = float(np.linalg.norm(diff))
        if length < 1e-9:
            return
        mid   = (p0 + p1) * 0.5
        z_hat = diff / length
        tmp   = np.array([0., 1., 0.]) if abs(z_hat[0]) > 0.9 else np.array([1., 0., 0.])
        x_hat = np.cross(tmp, z_hat);  x_hat /= np.linalg.norm(x_hat)
        y_hat = np.cross(z_hat, x_hat)
        R     = RotationMatrix(np.column_stack([x_hat, y_hat, z_hat]))
        meshcat.SetObject(path, Cylinder(cable_radius, length), rgba)
        meshcat.SetTransform(path, RigidTransform(R, mid))

    # ── 1. Straight cable segments ────────────────────────────────────────────
    #      When springs are enabled, the last segment of each route is drawn
    #      as a zigzag spring instead of a straight cylinder.
    from cable import spring_zigzag_points  # local to avoid circular at module level
    springs_on = getattr(rig, "springs_enabled", False)

    for ri, route in enumerate(rig.routes):
        pts  = route.world_points(plant, plant_context, manipulator)  # (N, 3)
        skip = getattr(route, "skip_chord_segments", frozenset())
        last_seg_idx = len(pts) - 2   # index of the final segment
        for i, (p0, p1) in enumerate(zip(pts[:-1], pts[1:])):
            if i in skip:
                continue
            # Spring visualization for the last segment: cable—spring—cable
            if springs_on and i == last_seg_idx:
                spring = rig.spring_L if ri == 0 else rig.spring_R
                if spring.enabled:
                    seg_len = float(np.linalg.norm(p0 - p1))
                    # Dynamic spring length: only the taut cable stretches.
                    # ri==0 → green cable (taut when δ > 0, F_raw > 0)
                    # ri==1 → red   cable (taut when δ < 0, F_raw < 0)
                    rest_len   = spring.rest_length
                    route_ext  = max(spring_extension, 0.0) if ri == 0 \
                                 else max(-spring_extension, 0.0)
                    actual_len = rest_len + route_ext
                    # Fraction of segment occupied by the spring (clamped)
                    if seg_len > 1e-6:
                        sf = np.clip(actual_len / seg_len, 0.05, 0.90)
                    else:
                        sf = 0.30
                    sp = np.clip(spring.spring_position, sf/2, 1.0 - sf/2)
                    t0 = sp - sf / 2   # spring start (fraction from endpoint)
                    t1 = sp + sf / 2   # spring end
                    # p0 = pulley exit (far end), p1 = endpoint (near end)
                    # t measured from p1 (endpoint) toward p0 (pulley)
                    p_spring_start = p1 + t0 * (p0 - p1)
                    p_spring_end   = p1 + t1 * (p0 - p1)
                    # Cable: pulley exit → spring end
                    _place_cylinder(f"{route.meshcat_path}/seg{i:02d}_a",
                                    p0, p_spring_end, route.meshcat_color)
                    # Helical spring
                    zz = spring_zigzag_points(p_spring_end, p_spring_start,
                                              n_coils=spring.n_coils,
                                              amplitude=spring.amplitude)
                    spring_rgba = Rgba(0.9, 0.6, 0.0, 1.0)  # gold/orange
                    for j, (z0, z1) in enumerate(zip(zz[:-1], zz[1:])):
                        _place_cylinder(f"{route.meshcat_path}/spring{j:02d}",
                                        z0, z1, spring_rgba)
                    # Cable: spring start → endpoint
                    _place_cylinder(f"{route.meshcat_path}/seg{i:02d}_b",
                                    p_spring_start, p1, route.meshcat_color)
                    continue
            _place_cylinder(f"{route.meshcat_path}/seg{i:02d}", p0, p1,
                            route.meshcat_color)

    # ── 2. Pulley wrap arcs (Rodrigues sweep in the plane ⊥ to shaft) ─────────
    def _arc_world_pts(center_w, radius, shaft_w, A_w, B_w):
        """Arc points from A to B around center, in the plane ⊥ to shaft_w."""
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

    _G = Rgba(0.1, 0.85, 0.1, 1.0)   # green cable
    _R = Rgba(0.9, 0.1,  0.1, 1.0)   # red   cable
    dp, ir, il, pb = rig.drive_pulley, rig.idler_r, rig.idler_l, rig.pulley_big
    wraps = [
        # Green cable — Drive → IdlerR → BigPulley
        (dp, dp.A_R, dp.B_R, "/wrap/drive/green",  _G),
        (ir, ir.A_R, ir.B_R, "/wrap/idlerR/green", _G),
        (pb, pb.A_L, pb.B_L, "/wrap/big/green",    _G),
        # Red cable — Drive → IdlerL → BigPulley
        (dp, dp.A_L, dp.B_L, "/wrap/drive/red",    _R),
        (il, il.A_L, il.B_L, "/wrap/idlerL/red",   _R),
        (pb, pb.A_R, pb.B_R, "/wrap/big/red",      _R),
    ]
    for pulley, A_body, B_body, path_prefix, rgba in wraps:
        if A_body is None or B_body is None:
            continue
        R_wb, t_wb = _Xw(plant, manipulator, plant_context, pulley.body_name)
        center_w   = R_wb @ np.asarray(pulley.centroid) + t_wb
        shaft_w    = R_wb @ pulley.shaft_axis_body
        A_w        = R_wb @ np.asarray(A_body) + t_wb
        B_w        = R_wb @ np.asarray(B_body) + t_wb
        arc_pts    = _arc_world_pts(center_w, pulley.radius, shaft_w, A_w, B_w)
        for i, (p0, p1) in enumerate(zip(arc_pts[:-1], arc_pts[1:])):
            _place_cylinder(f"{path_prefix}/arc{i:02d}", p0, p1, rgba)


# ──────────────────────────────────────────────────────────────────────────────

def visualize_cable_routing_3d(plant, plant_context, manipulator, assets_dir: str,
                               q1_deg: float = 0.0, q2_deg: float = 0.0, rig=None):
    """
    3-D matplotlib plot showing:
      • Coordinate frame triads: World / Joint-1 / Joint-2 / EE  (X=red Y=green Z=blue)
      • Cable routing objects rendered from their OBJ meshes:
            link1_base_pulley, 623zz ×2 (idler bearings), pulley_big
      • Cable path as a polyline through the four FK-computed waypoints

    Uses Drake FK so the plot is correct for any (q1, q2) already in plant_context.

    Args:
        plant         : Finalized MultibodyPlant
        plant_context : Drake context with joint angles already set
        manipulator   : CupManipulatorTendon instance
        assets_dir    : Path to directory containing .obj mesh files
        q1_deg, q2_deg: Joint angles for the plot title only

    Returns:
        (fig, ax): matplotlib Figure and Axes3D objects
    """
    from mpl_toolkits.mplot3d import Axes3D              # noqa — registers projection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import trimesh

    assets = Path(assets_dir)

    # ── RPY → rotation matrix (URDF: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)) ─────
    def rpy_to_R(r, p, y):
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    # ── Drake FK: body name → (R_world, t_world) ──────────────────────────────
    def body_Xw(body_name):
        body = plant.GetBodyByName(body_name, manipulator.model_instance)
        X    = plant.CalcRelativeTransform(plant_context, plant.world_frame(),
                                           body.body_frame())
        return X.rotation().matrix(), X.translation()

    # ── Draw one RGB frame triad as quiver arrows ──────────────────────────────
    def draw_frame(ax, origin, R, length=0.035, label=""):
        for i, (color, axis_lbl) in enumerate(zip(['red', 'lime', 'blue'], ['X', 'Y', 'Z'])):
            d = R[:, i] * length
            ax.quiver(origin[0], origin[1], origin[2],
                      d[0], d[1], d[2],
                      color=color, linewidth=2, arrow_length_ratio=0.35)
            tip = origin + d * 1.55
            ax.text(tip[0], tip[1], tip[2], f"{label}{axis_lbl}",
                    fontsize=7, color=color, fontweight='bold')

    # ── Load OBJ, apply URDF visual transform, apply FK, return world verts ────
    def load_mesh_world(obj_name, body_name, vis_xyz, vis_rpy=(0.0, 0.0, 0.0)):
        mesh_path = assets / obj_name
        if not mesh_path.exists():
            print(colored(f"  ⚠ {obj_name} not found, skipping", "yellow"))
            return None, None
        mesh    = trimesh.load(str(mesh_path), force="mesh")
        R_vis   = rpy_to_R(*vis_rpy)
        xyz_off = np.array(vis_xyz)
        # vertex in body frame: p_body = R_vis @ p_local + xyz_off
        v_body  = (R_vis @ mesh.vertices.T).T + xyz_off      # (N,3)
        # vertex in world frame: p_world = R_body @ p_body + t_body
        R_b, t_b = body_Xw(body_name)
        v_world = (R_b @ v_body.T).T + t_b                   # (N,3)
        return v_world, mesh.faces

    # ── Both cable routes in world frame ────────────────────────────────────────
    route_pts = [
        (route, route.world_points(plant, plant_context, manipulator))
        for route in rig.routes
    ]

    # ── Key frame world-frame transforms ─────────────────────────────────────
    R_w,  t_w  = np.eye(3), np.zeros(3)                # world
    R_j1, t_j1 = body_Xw("pulley_htd_5m_60t")          # joint-1 child body
    R_j2, t_j2 = body_Xw("link2_tendon")               # joint-2 child body
    ee_frame    = plant.GetFrameByName(manipulator.EE_FRAME_NAME, manipulator.model_instance)
    X_WEE       = plant.CalcRelativeTransform(plant_context, plant.world_frame(), ee_frame)
    R_ee, t_ee  = X_WEE.rotation().matrix(), X_WEE.translation()

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title(
        f"Cable Routing — q1 = {q1_deg:.1f}°   q2 = {q2_deg:.1f}°\n"
        "Frames: X = red   Y = green   Z = blue",
        fontsize=11
    )

    # ── Render meshes ─────────────────────────────────────────────────────────
    all_world_verts = [pts for _, pts in route_pts]   # seed bounding box with cable pts
    for cfg in rig.waypoints:
        verts, faces = load_mesh_world(cfg.obj_name, cfg.body_name, cfg.vis_xyz, cfg.vis_rpy)
        if verts is None:
            continue
        all_world_verts.append(verts)
        tris = [[verts[f[0]], verts[f[1]], verts[f[2]]] for f in faces]
        poly = Poly3DCollection(tris, alpha=cfg.mesh_alpha, linewidth=0,
                                facecolor=cfg.face_color, edgecolor='none')
        ax.add_collection3d(poly)
        centroid = verts.mean(axis=0)
        ax.text(centroid[0], centroid[1], centroid[2],
                f"  {cfg.label}", fontsize=7, color='k')

    # ── Render cable paths (one per route) ────────────────────────────────────
    from cable import spring_zigzag_points
    springs_on = getattr(rig, "springs_enabled", False)
    pt_icons = ['①', '②', '③', '④']
    for ri, (route, pts) in enumerate(route_pts):
        spring = (rig.spring_L if ri == 0 else rig.spring_R) if springs_on else None
        if spring and spring.enabled:
            sf = np.clip(spring.spring_fraction, 0.05, 0.90)
            sp = np.clip(spring.spring_position, sf/2, 1.0 - sf/2)
            t0 = sp - sf / 2
            t1 = sp + sf / 2
            p_end = pts[-1]
            p_pul = pts[-2]
            p_spring_start = p_end + t0 * (p_pul - p_end)
            p_spring_end   = p_end + t1 * (p_pul - p_end)
            # Cable: all waypoints up to pulley exit + cable to spring end
            cable_a = np.vstack([pts[:-1], p_spring_end.reshape(1, 3)])
            ax.plot(cable_a[:, 0], cable_a[:, 1], cable_a[:, 2],
                    'o-', color=route.mpl_color, linewidth=2.5, markersize=6,
                    label=route.label, zorder=6)
            # Helical spring
            zz = spring_zigzag_points(p_spring_end, p_spring_start,
                                      n_coils=spring.n_coils,
                                      amplitude=spring.amplitude)
            ax.plot(zz[:, 0], zz[:, 1], zz[:, 2],
                    '-', color='darkorange', linewidth=2.5, zorder=6,
                    label=f"Spring ({spring.label})" if ri == 0 else None)
            # Cable: spring start → endpoint
            ax.plot([p_spring_start[0], p_end[0]],
                    [p_spring_start[1], p_end[1]],
                    [p_spring_start[2], p_end[2]],
                    '-o', color=route.mpl_color, linewidth=2.5, markersize=6,
                    zorder=6)
        else:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    'o-', color=route.mpl_color, linewidth=2.5, markersize=6,
                    label=route.label, zorder=6)
        for pt, icon, (cfg, _) in zip(pts, pt_icons, route.segments):
            ax.text(pt[0], pt[1], pt[2], f"  {icon}{cfg.label}",
                    fontsize=7, color=route.mpl_color, zorder=7)

    # ── Render coordinate frames + origin dots ────────────────────────────────
    fl = 0.03   # arrow length [m]
    for origin, R, prefix, dot_color, name in [
        (t_w,  R_w,  "W_",  'k',          "World"),
        (t_j1, R_j1, "J1_", 'royalblue',  "Joint-1"),
        (t_j2, R_j2, "J2_", 'seagreen',   "Joint-2"),
        (t_ee, R_ee, "EE_", 'darkorchid', "EE"),
    ]:
        draw_frame(ax, origin, R, length=fl, label=prefix)
        ax.scatter(*origin, s=55, c=dot_color, zorder=10, depthshade=False)
        ax.text(origin[0], origin[1], origin[2] + fl * 0.4,
                name, fontsize=8, color=dot_color, fontweight='bold')

    # ── Equal-aspect bounding box ─────────────────────────────────────────────
    all_pts = np.vstack([v if v.ndim == 2 else v.reshape(1, 3)
                         for v in all_world_verts])
    mid  = all_pts.mean(axis=0)
    span = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2, 0.05)
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    return fig, ax


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def visualize_cable_routing_top_view(plant, plant_context, manipulator,
                                     q1_deg: float = 0.0, q2_deg: float = 0.0,
                                     rig=None):
    """Figure 1 — 2-D top view (XY plane, looking down Z) of cable routing.

    Draws every pulley as a filled circle scaled to its radius,
    cable routes as polylines, and annotates each contact-point label.

    Args:
        plant, plant_context, manipulator : Drake objects with FK already set.
        q1_deg, q2_deg : Joint angles used in the plot title only.
        cable_routes : list of CableRoute.
        cable_waypoints : list of PulleyBase.
        drive_pulley, idler_r, idler_l, pulley_big : pulley instances for wrap arcs.

    Returns:
        (fig, ax): matplotlib Figure and Axes objects.
    """
    # ── Drake FK helper: body-frame point → world frame ──────────────────────
    def body_world_pt(body_name, p_body):
        body = plant.GetBodyByName(body_name, manipulator.model_instance)
        return plant.CalcPointsPositions(
            plant_context,
            body.body_frame(),
            np.array(p_body, float).reshape(3, 1),
            plant.world_frame(),
        ).flatten()

    fig, ax = plt.subplots(figsize=(10, 7), num="Figure 1 — Top View (XY)")
    ax.set_title(
        f"Cable Routing — Top View (XY plane)   "
        f"q1 = {q1_deg:.1f}°   q2 = {q2_deg:.1f}°",
        fontsize=11,
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── Pulley circles ───────────────────────────────────────────────────────
    for cfg in rig.waypoints:
        if cfg.radius <= 0:
            continue   # skip zero-radius anchor balls
        wx, wy, _ = body_world_pt(cfg.body_name, cfg.centroid)
        circle = plt.Circle((wx, wy), cfg.radius,
                             color=cfg.face_color, fill=True,
                             alpha=max(cfg.mesh_alpha, 0.35),
                             linewidth=1.5, edgecolor='k', zorder=3)
        ax.add_patch(circle)
        ax.text(wx, wy, cfg.label, fontsize=7, ha='center', va='center',
                color='k', fontweight='bold', zorder=5)

    # ── Cable polylines + waypoint labels ───────────────────────────────────
    from cable import spring_zigzag_points
    springs_on = getattr(rig, "springs_enabled", False)

    for ri, route in enumerate(rig.routes):
        pts = route.world_points(plant, plant_context, manipulator)
        last_seg_idx = len(pts) - 2  # index of final segment
        spring = (rig.spring_L if ri == 0 else rig.spring_R) if springs_on else None

        # Draw cable—spring—cable layout for the last segment
        if spring and spring.enabled:
            sf = np.clip(spring.spring_fraction, 0.05, 0.90)
            sp = np.clip(spring.spring_position, sf/2, 1.0 - sf/2)
            t0 = sp - sf / 2   # spring start (fraction from endpoint)
            t1 = sp + sf / 2   # spring end
            p_end = pts[-1]    # endpoint
            p_pul = pts[-2]    # pulley exit
            p_spring_start = p_end + t0 * (p_pul - p_end)
            p_spring_end   = p_end + t1 * (p_pul - p_end)
            # Cable: all waypoints up to pulley exit + cable to spring end
            cable_a = np.vstack([pts[:-1], p_spring_end.reshape(1, 3)])
            ax.plot(cable_a[:, 0], cable_a[:, 1], 'o-', color=route.mpl_color,
                    linewidth=2, markersize=5, label=route.label, zorder=6)
            # Helical spring
            zz = spring_zigzag_points(p_spring_end, p_spring_start,
                                      n_coils=spring.n_coils,
                                      amplitude=spring.amplitude)
            ax.plot(zz[:, 0], zz[:, 1], '-', color='darkorange', linewidth=2.0,
                    zorder=6, label=f"Spring ({spring.label})" if ri == 0 else None)
            # Cable: spring start → endpoint
            ax.plot([p_spring_start[0], p_end[0]],
                    [p_spring_start[1], p_end[1]], '-o', color=route.mpl_color,
                    linewidth=2, markersize=5, zorder=6)
        else:
            ax.plot(pts[:, 0], pts[:, 1], 'o-', color=route.mpl_color,
                    linewidth=2, markersize=5, label=route.label, zorder=6)

        for pt, (cfg, _) in zip(pts, route.segments):
            ax.annotate(f"  {cfg.label}", (pt[0], pt[1]),
                        fontsize=6.5, color=route.mpl_color, zorder=7)

    # ── Cable wrap arcs (A → B on each pulley rim) ───────────────────────────
    # Uses matplotlib.patches.Arc (mathematical arc stroke) instead of a
    # discrete polyline so the arc is always clearly curved even when the
    # sagitta is sub-pixel (e.g. small idler bearings with a short wrap angle).
    from matplotlib.patches import Arc as _MplArc

    dp, ir, il, pb = rig.drive_pulley, rig.idler_r, rig.idler_l, rig.pulley_big
    wrap_pairs = [
        # (pulley,  A tangent pt,  B tangent pt,  mpl color)
        (dp, dp.A_R, dp.B_R, "limegreen"),
        (ir, ir.A_R, ir.B_R, "limegreen"),
        (pb, pb.A_L, pb.B_L, "limegreen"),
        (dp, dp.A_L, dp.B_L, "red"),
        (il, il.A_L, il.B_L, "red"),
        (pb, pb.A_R, pb.B_R, "red"),
    ]
    for pulley, A_body, B_body, color in wrap_pairs:
        if A_body is None or B_body is None:
            continue
        center_w = body_world_pt(pulley.body_name, pulley.centroid)
        A_w      = body_world_pt(pulley.body_name, A_body)
        B_w      = body_world_pt(pulley.body_name, B_body)
        cx, cy   = center_w[0], center_w[1]
        R        = pulley.radius

        # 2-D unit vectors from centre to A and B (XY plane only)
        dA = np.array([A_w[0] - cx, A_w[1] - cy])
        dB = np.array([B_w[0] - cx, B_w[1] - cy])
        if np.linalg.norm(dA) < 1e-9 or np.linalg.norm(dB) < 1e-9:
            continue
        dA /= np.linalg.norm(dA)
        dB /= np.linalg.norm(dB)

        # Signed 2-D cross product: positive → CCW from A to B
        cross_z = float(dA[0] * dB[1] - dA[1] * dB[0])

        theta_A = np.degrees(np.arctan2(A_w[1] - cy, A_w[0] - cx))
        theta_B = np.degrees(np.arctan2(B_w[1] - cy, B_w[0] - cx))

        if cross_z >= 0:          # CCW from A → B
            t1 = theta_A
            t2 = theta_A + (theta_B - theta_A) % 360
        else:                     # CW from A → B = CCW from B → A
            t1 = theta_B
            t2 = theta_B + (theta_A - theta_B) % 360

        arc_patch = _MplArc(
            (cx, cy), 2 * R, 2 * R,
            angle=0, theta1=t1, theta2=t2,
            color=color, linewidth=3.5, zorder=8,
        )
        ax.add_patch(arc_patch)

    # ── Body-frame origin crosses ────────────────────────────────────────────
    for body_name, color, lbl in [
        ("pulley_htd_5m_60t", "royalblue",  "J1"),
        ("link2_tendon",      "seagreen",    "J2"),
    ]:
        ox, oy, _ = body_world_pt(body_name, np.zeros(3))
        ax.plot(ox, oy, '+', color=color, markersize=10, markeredgewidth=2, zorder=8)
        ax.text(ox, oy, f"  {lbl}", fontsize=8, color=color,
                fontweight='bold', zorder=8)

    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig, ax


