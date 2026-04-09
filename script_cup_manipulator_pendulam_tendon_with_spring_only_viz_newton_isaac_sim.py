#!/usr/bin/env python3
"""
Cup Manipulator with Spring-Damper — Newton Physics + Isaac Sim Rendering
=========================================================================

Newton (Featherstone solver) runs the physics for the 2R cup-manipulator
with configurable joint spring-damper properties. Isaac Sim provides the
3D rendering. The cable routing is still computed via DrakeCablePlant
(headless Drake FK) for the USD cylinder visualization.

Architecture:
    Newton (Featherstone) — rigid-body physics (metre scale)
    Isaac Sim (PhysX disabled) — rendering only
    Drake (headless) — cable FK for visualization

Interactive CLI commands:
    e <x> <y>     — move EE to (x, y) via Newton IK
    j <q1> <q2>   — set joint target angles [degrees]
    s <K1> <K2>   — set joint spring stiffness [Nm/rad]
    d <D1> <D2>   — set joint damping [Nm·s/rad]
    p             — print state
    r             — release (zero spring stiffness, let arm swing freely)
    q / Ctrl+C    — quit

Usage:
    source /path/to/isaacsim/_build/linux-x86_64/release/setup_conda_env.sh
    conda activate env_isaacsim
    python script_cup_manipulator_pendulam_with_spring_damper_newton_isaac_sim.py
    python script_cup_manipulator_pendulam_with_spring_damper_newton_isaac_sim.py --render websocket
    python script_cup_manipulator_pendulam_with_spring_damper_newton_isaac_sim.py --stiffness 5.0 5.0
    python script_cup_manipulator_pendulam_with_spring_damper_newton_isaac_sim.py --q1 20 --q2 -30
"""

from __future__ import annotations

import sys
import os

# ── PRE-PARSE --render BEFORE SimulationApp ──────────────────────────────
_RENDER_CHOICES = ("native", "websocket", "headless")
_render_mode = "native"
for _i, _arg in enumerate(sys.argv):
    if _arg == "--render" and _i + 1 < len(sys.argv):
        _render_mode = sys.argv[_i + 1]
        if _render_mode not in _RENDER_CHOICES:
            print(f"[ERROR] --render must be one of {_RENDER_CHOICES}, got '{_render_mode}'")
            sys.exit(1)
        break

os.environ.setdefault("CARB_LOG_LEVEL", "error")
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Import warp BEFORE SimulationApp ─────────────────────────────────────
# Isaac Sim 5.1 bundles Warp 1.8.2, but Newton requires Warp ≥1.12.1.
# Import from conda env first to cache in sys.modules.
import warp as wp
_warp_conda_path = wp.__path__[:]

# ── Isaac Sim startup ────────────────────────────────────────────────────
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": _render_mode != "native",
    "width": 1920,
    "height": 1080,
    "hide_ui": False,
})

if _render_mode == "websocket":
    from isaacsim.core.utils.extensions import enable_extension
    simulation_app.set_setting("/app/window/drawMouse", True)
    simulation_app.set_setting("/app/livestream/port", 49100)
    simulation_app.set_setting("/app/livestream/proto", "websocket")
    enable_extension("omni.kit.livestream.webrtc")

# ── Fix Warp path after SimulationApp ────────────────────────────────────
wp.__path__[:] = _warp_conda_path
_bad_warp_mods = [k for k in sys.modules if k.startswith("warp.") and "isaacsim" in str(getattr(sys.modules[k], "__file__", ""))]
for _mod in _bad_warp_mods:
    del sys.modules[_mod]

# ── Post-SimulationApp imports ───────────────────────────────────────────
import argparse
import queue
import threading
import numpy as np
from pathlib import Path

import omni.usd
from omni.isaac.core import World
from pxr import UsdGeom, UsdLux, Gf, Usd

import newton
from newton import ModelBuilder, eval_fk
from newton.solvers import SolverFeatherstone

# Add project root to path for cable import
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cable import DrakeCablePlant


# ── Constants ────────────────────────────────────────────────────────────
URDF_PATH = "model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf"
URDF_ISAAC_PATH = "model_using_onshape_to_robot/manipulator_cable_isaac/manipulator_cable_obj.urdf"

# Link geometry (metres)
L1 = 0.335  # base joint to elbow joint
L2 = 0.19   # elbow joint to EE
EE_OFFSET_LINK2 = np.array([0.19, 0.0, 0.0515])  # EE in link2 frame


def _urdf_resolve_package_paths(urdf_path: str) -> str:
    """Read a URDF file and resolve ``package://assets/`` to absolute paths.

    Newton's built-in package:// fallback only works when the package name
    appears in the URDF folder's path string, which is not the case for our
    ``manipulator_cable`` layout.  We sidestep the issue by returning the
    URDF XML with absolute mesh paths so it can be fed as a string to
    ``ModelBuilder.add_urdf()``.
    """
    urdf_abs = str(Path(urdf_path).resolve())
    assets_dir = str(Path(urdf_abs).parent / "assets")
    with open(urdf_abs) as f:
        xml_text = f.read()
    xml_text = xml_text.replace("package://assets/", assets_dir + "/")
    return xml_text


# ── Newton Physics ───────────────────────────────────────────────────────

class NewtonCupManipulator:
    """Newton Featherstone physics for the 2R cup manipulator."""

    def __init__(self, urdf_path: str, q1_rad: float, q2_rad: float,
                 stiffness: tuple[float, float] = (0.0, 0.0),
                 damping: tuple[float, float] = (0.05, 0.05)):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # Build model — resolve package:// paths and pass XML string
        builder = ModelBuilder()
        urdf_xml = _urdf_resolve_package_paths(urdf_path)

        builder.add_urdf(
            urdf_xml,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            scale=1.0,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
        )

        # Discover joint labels and map to DOF offsets.
        # FIXED base joint has 0 DOFs; each revolute has 1 DOF.
        self._jt_labels = list(builder.joint_label)
        print(f"[Newton] Joints ({builder.joint_count}): {self._jt_labels}")

        self._q1_dof = -1  # DOF offset for link1_base
        self._q2_dof = -1  # DOF offset for link2_link1
        self._link2_body_idx = -1

        for j_idx in range(builder.joint_count):
            label = builder.joint_label[j_idx]
            dof_start = builder.joint_qd_start[j_idx]
            q_start = builder.joint_q_start[j_idx]
            if label == "link1_base":
                self._q1_dof = dof_start
                self._q1_q = q_start
            elif label == "link2_link1":
                self._q2_dof = dof_start
                self._q2_q = q_start

        assert self._q1_dof >= 0 and self._q2_dof >= 0, (
            f"Could not find expected joints. Labels: {self._jt_labels}"
        )
        print(f"[Newton] link1_base → DOF {self._q1_dof}, "
              f"link2_link1 → DOF {self._q2_dof}")

        # Set initial joint coordinates (joint_q is per-coordinate)
        builder.joint_q[self._q1_q] = q1_rad
        builder.joint_q[self._q2_q] = q2_rad

        # Spring-damper: target_ke/kd are per-DOF
        builder.joint_target_ke[self._q1_dof] = stiffness[0]
        builder.joint_target_ke[self._q2_dof] = stiffness[1]
        builder.joint_target_kd[self._q1_dof] = damping[0]
        builder.joint_target_kd[self._q2_dof] = damping[1]

        # Target position (equilibrium angle) = initial position
        builder.joint_target_pos[self._q1_dof] = q1_rad
        builder.joint_target_pos[self._q2_dof] = q2_rad

        self._stiffness = list(stiffness)
        self._damping = list(damping)

        # Identify link2 body for EE computation
        for b_idx, lbl in enumerate(builder.body_label):
            if "link2" in lbl:
                self._link2_body_idx = b_idx
        print(f"[Newton] Bodies ({len(builder.body_label)}): {list(builder.body_label)}")

        # Finalize
        self.model = builder.finalize(requires_grad=False)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # FK eval to initialize body transforms
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Solver
        self.solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)

        print(f"[Newton] Bodies: {self.model.body_count}, Joints: {self.model.joint_count}")

    def step(self):
        """Advance one frame of physics."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt

    def get_joint_positions(self) -> tuple[float, float]:
        """Return (q1, q2) in radians."""
        q = self.state_0.joint_q.numpy()
        return float(q[self._q1_q]), float(q[self._q2_q])

    def get_joint_velocities(self) -> tuple[float, float]:
        """Return (q1_dot, q2_dot) in rad/s."""
        qd = self.state_0.joint_qd.numpy()
        return float(qd[self._q1_dof]), float(qd[self._q2_dof])

    def set_target_positions(self, q1_rad: float, q2_rad: float):
        """Set spring equilibrium (target) positions."""
        target = self.control.joint_target_pos.numpy()
        target[self._q1_dof] = q1_rad
        target[self._q2_dof] = q2_rad
        self.control.joint_target_pos.assign(target)

    def set_joint_positions_direct(self, q1_rad: float, q2_rad: float):
        """Teleport joints to given angles (reset velocities)."""
        q = self.state_0.joint_q.numpy()
        q[self._q1_q] = q1_rad
        q[self._q2_q] = q2_rad
        self.state_0.joint_q.assign(q)

        qd = self.state_0.joint_qd.numpy()
        qd[self._q1_dof] = 0.0
        qd[self._q2_dof] = 0.0
        self.state_0.joint_qd.assign(qd)

        eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def set_stiffness(self, k1: float, k2: float):
        """Update joint spring stiffness."""
        ke = self.model.joint_target_ke.numpy()
        ke[self._q1_dof] = k1
        ke[self._q2_dof] = k2
        self.model.joint_target_ke.assign(ke)
        self._stiffness = [k1, k2]

    def set_damping(self, d1: float, d2: float):
        """Update joint damping."""
        kd = self.model.joint_target_kd.numpy()
        kd[self._q1_dof] = d1
        kd[self._q2_dof] = d2
        self.model.joint_target_kd.assign(kd)
        self._damping = [d1, d2]

    def get_body_transforms(self) -> np.ndarray:
        """Return body transforms as array of shape (N, 7): [px,py,pz, qx,qy,qz,qw]."""
        return self.state_0.body_q.numpy()

    def compute_ee_position(self) -> np.ndarray:
        """Compute EE position from body FK transforms."""
        from scipy.spatial.transform import Rotation

        body_xforms = self.get_body_transforms()
        if self._link2_body_idx >= 0 and self._link2_body_idx < len(body_xforms):
            xf = body_xforms[self._link2_body_idx]
            pos = xf[:3]
            quat = xf[3:]  # (qx, qy, qz, qw)
            rot = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]])
            return pos + rot.apply(EE_OFFSET_LINK2)

        # Fallback: analytical 2R FK
        q1, q2 = self.get_joint_positions()
        base_z = 0.048
        x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
        z = base_z + L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
        return np.array([x, 0.0, z])


# ── Analytical IK for 2R planar ──────────────────────────────────────────

def solve_ik_2r(target_x: float, target_z: float,
                base_z: float = 0.048) -> tuple[np.ndarray, bool]:
    """Solve 2R planar IK. Returns ([q1, q2], success)."""
    # Target relative to shoulder
    dx = target_x
    dz = target_z - base_z

    dist_sq = dx**2 + dz**2
    dist = np.sqrt(dist_sq)

    if dist > (L1 + L2) or dist < abs(L1 - L2):
        return np.array([0.0, 0.0]), False

    cos_q2 = (dist_sq - L1**2 - L2**2) / (2 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    q2 = np.arctan2(-np.sqrt(1 - cos_q2**2), cos_q2)  # elbow-up

    k1 = L1 + L2 * np.cos(q2)
    k2 = L2 * np.sin(q2)
    q1 = np.arctan2(dz, dx) - np.arctan2(k2, k1)

    return np.array([q1, q2]), True


# ── Isaac Sim Scene ──────────────────────────────────────────────────────

def setup_lighting(stage):
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(500.0)
    dist = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    dist.CreateIntensityAttr(2000.0)
    dist.CreateAngleAttr(0.53)
    xf = UsdGeom.Xformable(dist)
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))


def load_manipulator_urdf(stage, urdf_path: str) -> str:
    """Load manipulator URDF into Isaac Sim stage for visuals."""
    from isaacsim.asset.importer.urdf import _urdf

    abs_path = str(Path(urdf_path).resolve())
    config = _urdf.ImportConfig()
    config.convex_decomp = False
    config.fix_base = True
    config.make_default_prim = False
    config.self_collision = False
    config.distance_scale = 1.0
    config.create_physics_scene = False

    result, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=abs_path,
        import_config=config,
        dest_path="",
    )
    print(f"[Isaac Sim] Manipulator loaded at: {prim_path}")
    return prim_path


def add_ee_marker(stage):
    sphere = UsdGeom.Sphere.Define(stage, "/World/EE_Marker")
    sphere.GetRadiusAttr().Set(0.008)
    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.2, 0.2)])
    return sphere


def update_ee_marker(stage, pos: np.ndarray):
    prim = stage.GetPrimAtPath("/World/EE_Marker")
    if prim.IsValid():
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))


# ── Cable rendering (same as existing Isaac Sim script) ──────────────────

_CABLE_ROOT = "/World/Cables"
_CABLE_RADIUS = 0.0005


def _usd_cylinder(stage, path: str, p0: np.ndarray, p1: np.ndarray, color_rgb):
    diff = p1 - p0
    length = float(np.linalg.norm(diff))
    if length < 1e-9:
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
    if "green" in route.mpl_color.lower():
        return (0.1, 0.85, 0.1)
    return (0.9, 0.1, 0.1)


def draw_cables_usd(stage, drake_cable: DrakeCablePlant):
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


def update_cables_usd(stage, drake_cable: DrakeCablePlant):
    cable_prim = stage.GetPrimAtPath(_CABLE_ROOT)
    if cable_prim.IsValid():
        stage.RemovePrim(_CABLE_ROOT)
    draw_cables_usd(stage, drake_cable)


# ── Sync Newton body transforms → Isaac Sim ─────────────────────────────

def sync_robot_transforms(stage, prim_path: str, physics: NewtonCupManipulator):
    """Sync Newton body transforms to Isaac Sim visual prims.

    Newton stores body labels in ``model.body_label`` (a list[str] indexed
    by body index).  Isaac Sim's URDF importer creates USD prims whose
    names match the URDF link names.  We build a mapping once, then update
    transforms every frame.
    """
    body_transforms = physics.get_body_transforms()
    model = physics.model

    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return

    # Build prim name map (USD child prim name → Usd.Prim)
    prim_map: dict[str, "Usd.Prim"] = {}
    for desc in Usd.PrimRange(root_prim):
        if desc.IsA(UsdGeom.Xform) or desc.IsA(UsdGeom.Mesh):
            prim_map[desc.GetName()] = desc

    for body_idx in range(model.body_count):
        body_name = model.body_label[body_idx] if body_idx < len(model.body_label) else None
        if body_name is None:
            continue

        # Try full name first, then short (after last '/')
        target_prim = prim_map.get(body_name)
        if target_prim is None:
            short = body_name.split("/")[-1] if "/" in body_name else body_name
            target_prim = prim_map.get(short)
        if target_prim is None:
            continue

        xf = body_transforms[body_idx]
        pos = Gf.Vec3d(float(xf[0]), float(xf[1]), float(xf[2]))
        # Newton quat: (qx, qy, qz, qw)
        quat = Gf.Quatd(float(xf[6]), float(xf[3]), float(xf[4]), float(xf[5]))

        xformable = UsdGeom.Xformable(target_prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(pos)
        xformable.AddOrientOp().Set(Gf.Quatf(quat))


# ── Interactive CLI ──────────────────────────────────────────────────────

_cmd_queue: queue.Queue = queue.Queue()


def _input_reader(sim_app):
    print("\n" + "=" * 60)
    print("  Newton Cup-Manipulator Spring-Damper — Commands:")
    print("    e <x> <z>     — move EE via IK (set spring target)")
    print("    j <q1> <q2>   — set target angles [deg]")
    print("    J <q1> <q2>   — teleport to angles [deg] (no dynamics)")
    print("    s <K1> <K2>   — set spring stiffness [Nm/rad]")
    print("    d <D1> <D2>   — set damping [Nm·s/rad]")
    print("    r             — release (zero stiffness)")
    print("    p             — print state")
    print("    q             — quit")
    print("=" * 60 + "\n")
    while sim_app.is_running():
        try:
            line = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            _cmd_queue.put("q")
            break
        if line:
            _cmd_queue.put(line)


def process_commands(physics: NewtonCupManipulator, stage, drake_cable):
    """Drain command queue. Returns False to quit."""
    while not _cmd_queue.empty():
        try:
            text = _cmd_queue.get_nowait()
        except queue.Empty:
            break

        parts = text.split()
        cmd = parts[0]

        if cmd == 'q':
            return False

        elif cmd == 'p':
            q1, q2 = physics.get_joint_positions()
            v1, v2 = physics.get_joint_velocities()
            ee = physics.compute_ee_position()
            print(f"  q1={np.rad2deg(q1):+7.2f}°  q2={np.rad2deg(q2):+7.2f}°")
            print(f"  v1={v1:+.4f}  v2={v2:+.4f} rad/s")
            print(f"  EE=({ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}) m")
            print(f"  stiffness=({physics._stiffness[0]:.2f}, {physics._stiffness[1]:.2f})")
            print(f"  damping=({physics._damping[0]:.4f}, {physics._damping[1]:.4f})")

        elif cmd == 'e' and len(parts) >= 3:
            try:
                tx, tz = float(parts[1]), float(parts[2])
                sol, ok = solve_ik_2r(tx, tz)
                if ok:
                    physics.set_target_positions(sol[0], sol[1])
                    print(f"  ✓ IK target → q1={np.rad2deg(sol[0]):+.2f}°  q2={np.rad2deg(sol[1]):+.2f}°")
                else:
                    print(f"  ✗ IK failed for ({tx}, {tz})")
            except ValueError:
                print("  Usage: e <x> <z>")

        elif cmd == 'j' and len(parts) >= 3:
            try:
                q1d, q2d = float(parts[1]), float(parts[2])
                physics.set_target_positions(np.deg2rad(q1d), np.deg2rad(q2d))
                print(f"  ✓ Target → q1={q1d:+.2f}°  q2={q2d:+.2f}°")
            except ValueError:
                print("  Usage: j <q1_deg> <q2_deg>")

        elif cmd == 'J' and len(parts) >= 3:
            try:
                q1d, q2d = float(parts[1]), float(parts[2])
                q1r, q2r = np.deg2rad(q1d), np.deg2rad(q2d)
                physics.set_joint_positions_direct(q1r, q2r)
                physics.set_target_positions(q1r, q2r)
                if drake_cable:
                    drake_cable.update(q1r, q2r)
                    update_cables_usd(stage, drake_cable)
                print(f"  ✓ Teleport → q1={q1d:+.2f}°  q2={q2d:+.2f}°")
            except ValueError:
                print("  Usage: J <q1_deg> <q2_deg>")

        elif cmd == 's' and len(parts) >= 3:
            try:
                k1, k2 = float(parts[1]), float(parts[2])
                physics.set_stiffness(k1, k2)
                print(f"  ✓ Stiffness → ({k1}, {k2}) Nm/rad")
            except ValueError:
                print("  Usage: s <K1> <K2>")

        elif cmd == 'd' and len(parts) >= 3:
            try:
                d1, d2 = float(parts[1]), float(parts[2])
                physics.set_damping(d1, d2)
                print(f"  ✓ Damping → ({d1}, {d2}) Nm·s/rad")
            except ValueError:
                print("  Usage: d <D1> <D2>")

        elif cmd == 'r':
            physics.set_stiffness(0.0, 0.0)
            print("  ✓ Released — zero stiffness (free swing)")

        else:
            print("  Unknown command. Try: e, j, J, s, d, r, p, q")

    return True


# ── Args ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cup manipulator with spring-damper (Newton + Isaac Sim)",
    )
    parser.add_argument("--render", choices=_RENDER_CHOICES, default=_render_mode)
    parser.add_argument("--q1", type=float, default=10.0, help="Initial q1 [deg]")
    parser.add_argument("--q2", type=float, default=-10.0, help="Initial q2 [deg]")
    parser.add_argument(
        "--stiffness", type=float, nargs=2, default=[5.0, 5.0],
        metavar=("K1", "K2"), help="Joint spring stiffness [Nm/rad]",
    )
    parser.add_argument(
        "--damping", type=float, nargs=2, default=[0.5, 0.5],
        metavar=("D1", "D2"), help="Joint damping [Nm·s/rad]",
    )
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  Cup Manipulator Spring-Damper (Newton + Isaac Sim)")
    print("=" * 60)

    q1_rad = np.deg2rad(args.q1)
    q2_rad = np.deg2rad(args.q2)

    # ── 1. Newton physics ────────────────────────────────────────────────
    print("[Newton] Building physics scene...")
    physics = NewtonCupManipulator(
        urdf_path=os.path.join(PROJECT_ROOT, URDF_PATH),
        q1_rad=q1_rad,
        q2_rad=q2_rad,
        stiffness=tuple(args.stiffness),
        damping=tuple(args.damping),
    )

    # ── 2. Isaac Sim stage ───────────────────────────────────────────────
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    setup_lighting(stage)

    # Ground plane
    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    s = 2.0
    ground.CreatePointsAttr([
        Gf.Vec3f(-s, -s, 0), Gf.Vec3f(s, -s, 0),
        Gf.Vec3f(s, s, 0), Gf.Vec3f(-s, s, 0),
    ])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.3, 0.3, 0.3)])

    # Robot visual
    manipulator_prim_path = load_manipulator_urdf(
        stage, os.path.join(PROJECT_ROOT, URDF_ISAAC_PATH),
    )

    # EE marker
    add_ee_marker(stage)

    # ── 3. World init ────────────────────────────────────────────────────
    world.reset()
    for _ in range(5):
        world.step(render=True)
        simulation_app.update()

    # ── 4. Cable visualization (headless Drake FK) ───────────────────────
    drake_urdf = os.path.join(PROJECT_ROOT, URDF_PATH)
    drake_cable = DrakeCablePlant(drake_urdf, q1=q1_rad, q2=q2_rad)
    draw_cables_usd(stage, drake_cable)

    # Initial sync
    sync_robot_transforms(stage, manipulator_prim_path, physics)
    ee = physics.compute_ee_position()
    update_ee_marker(stage, ee)

    print(f"\n[Ready] q1={args.q1:+.1f}° q2={args.q2:+.1f}°  "
          f"K=({args.stiffness[0]}, {args.stiffness[1]})  "
          f"D=({args.damping[0]}, {args.damping[1]})")

    # ── 5. CLI thread ────────────────────────────────────────────────────
    cli_thread = threading.Thread(target=_input_reader, args=(simulation_app,), daemon=True)
    cli_thread.start()

    # ── 6. Main loop ─────────────────────────────────────────────────────
    frame = 0
    cable_update_interval = 5  # update cables every N frames (expensive)
    try:
        while simulation_app.is_running():
            if not process_commands(physics, stage, drake_cable):
                break

            # Step Newton physics
            physics.step()

            # Sync robot visual
            sync_robot_transforms(stage, manipulator_prim_path, physics)

            # Update EE marker
            ee = physics.compute_ee_position()
            update_ee_marker(stage, ee)

            # Update cable visualization periodically
            if frame % cable_update_interval == 0:
                q1, q2 = physics.get_joint_positions()
                drake_cable.update(q1, q2)
                update_cables_usd(stage, drake_cable)

            # Render
            world.step(render=True)
            simulation_app.update()

            frame += 1

    except KeyboardInterrupt:
        pass

    simulation_app.close()
    print("[Done]")


if __name__ == "__main__":
    main()
