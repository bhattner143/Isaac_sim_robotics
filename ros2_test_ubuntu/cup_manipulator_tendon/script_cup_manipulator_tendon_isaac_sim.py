#!/usr/bin/env python3
"""
Cup Manipulator Tendon — Isaac Sim ROS 2 Bridge
================================================
Isaac Sim subscribes to commands from PyDrake via ROS 2 and publishes
feedback (EE position or joint state) back to Drake.

Two modes (set via --mode):
  joint_command — Subscribe /manip/joint_command (JointState)
                  Publish   /manip/ee_position   (Point)
  ee_command    — Subscribe /manip/ee_command     (Point)
                  Publish   /manip/joint_state    (JointState)

Architecture (follows cube_commander/isaac_sim.py pattern):
  isaacsim.ros2.bridge extension → rclpy in same process
  rclpy.spin() in daemon thread, world.step() on main thread
  queue.Queue bridges the two threads

Standalone — run directly (no shell wrapper needed):
    python script_cup_manipulator_tendon_isaac_sim.py
    python script_cup_manipulator_tendon_isaac_sim.py --mode ee_command --render headless

Then start the Drake commander in another terminal:
    bash ros2_test_ubuntu/cup_manipulator_tendon/run_drake_commander.sh
"""

# ============================================================================
# ENVIRONMENT SETUP — must run BEFORE any isaacsim imports
# ============================================================================
# Isaac Sim uses Python 3.11, but the system ROS 2 Jazzy was built for 3.12.
# If ros2_jazzy paths leak into PYTHONPATH, the bridge extension's internal
# rclpy (built for 3.11) cannot load.  This block cleans the environment and
# re-execs once so that LD_LIBRARY_PATH changes take effect for the dynamic
# linker.
import os
import sys

_SENTINEL = "_ISAAC_MANIP_REEXEC"

if os.environ.get(_SENTINEL) != "1":
    # ── Strip ros2_jazzy from PYTHONPATH ─────────────────────────────────
    _pp = os.environ.get("PYTHONPATH", "")
    _clean = ":".join(p for p in _pp.split(":") if p and "ros2_jazzy" not in p)
    os.environ["PYTHONPATH"] = _clean

    # ── Set ROS 2 env vars ───────────────────────────────────────────────
    os.environ["ROS_DISTRO"] = "jazzy"
    os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"

    # ── Find bridge extension and prepend jazzy/lib to LD_LIBRARY_PATH ───
    import importlib.util
    _spec = importlib.util.find_spec("isaacsim")
    if _spec and _spec.origin:
        import pathlib
        _bridge_ext = pathlib.Path(_spec.origin).parent / "exts" / "isaacsim.ros2.bridge"
        _jazzy_lib = str(_bridge_ext / "jazzy" / "lib")
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        if _jazzy_lib not in _ld:
            os.environ["LD_LIBRARY_PATH"] = f"{_jazzy_lib}:{_ld}" if _ld else _jazzy_lib

    # ── Re-exec so LD_LIBRARY_PATH takes effect for the dynamic linker ───
    os.environ[_SENTINEL] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# After re-exec (or if already re-execed): ensure sys.path is also clean
sys.path[:] = [p for p in sys.path if "ros2_jazzy" not in p]

# ============================================================================
# PRE-PARSE flags BEFORE SimulationApp
# ============================================================================

_RENDER_CHOICES = ("native", "websocket", "headless")
_render_mode = "native"  # default: native rendering with no streaming
_mode = "joint_command"

for _i, _arg in enumerate(sys.argv):
    if _arg == "--render" and _i + 1 < len(sys.argv):
        _render_mode = sys.argv[_i + 1]
    if _arg == "--mode" and _i + 1 < len(sys.argv):
        _mode = sys.argv[_i + 1]

# ============================================================================
# SUPPRESS ISAAC SIM STARTUP WARNINGS
# ============================================================================
# CARB_LOG_LEVEL=error silences carb-level warnings (crashdumps, PCIe,
# deprecated omni.isaac.* namespaces).  Set to "warn" to re-enable.
os.environ.setdefault("CARB_LOG_LEVEL", "error")
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================================
# MUST BE FIRST — Isaac Sim requirement
# ============================================================================
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": _render_mode != "native",
    "width":    1280,
    "height":   720,
    "hide_ui":  True,
})

# Enable WebRTC streaming extension when mode is 'websocket'
if _render_mode == "websocket":
    import subprocess
    from isaacsim.core.utils.extensions import enable_extension as _enable_ext

    # Detect Tailscale IP so remote clients (Mac) can connect
    _tailscale_ip = ""
    try:
        _tailscale_ip = subprocess.check_output(
            ["tailscale", "ip", "-4"], text=True, timeout=3
        ).strip()
    except Exception:
        pass

    simulation_app.set_setting("/app/window/drawMouse", True)
    simulation_app.set_setting("/app/livestream/port", 49100)
    simulation_app.set_setting("/app/livestream/proto", "websocket")
    if _tailscale_ip:
        simulation_app.set_setting("/app/livestream/publicEndpointAddress", _tailscale_ip)
    _enable_ext("omni.kit.livestream.webrtc")

    _connect_ip = _tailscale_ip if _tailscale_ip else "localhost"
    print("\n" + "=" * 60)
    print("  WebRTC streaming enabled")
    print(f"  Port          : 49100")
    if _tailscale_ip:
        print(f"  Tailscale IP  : {_tailscale_ip}")
    print(f"  Mac client    : connect to  {_connect_ip} : 49100")
    print("=" * 60 + "\n")

# ============================================================================
# ENABLE ROS 2 BRIDGE — must happen before importing rclpy
# ============================================================================
import omni.kit.app
_ext_manager = omni.kit.app.get_app().get_extension_manager()
_ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

# ============================================================================
# IMPORTS (after SimulationApp + extension enable)
# ============================================================================
import argparse
import queue
import threading
import numpy as np
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState

import omni.usd
from omni.isaac.core import World
from pxr import UsdGeom, UsdLux, Gf
from termcolor import colored

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robots.cup_manipulator_tendon_isaac import (
    CupManipulatorTendonIsaac,
    create_cable_manipulator_config,
)

# ── Cable FK (headless Drake plant for cable tangent computation) ─────────────
from cable import DrakeCablePlant

# ── QoS: must match the Drake publisher side ─────────────────────────────────
_CONTROL_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


# ============================================================================
# ROS 2 NODE — subscribes to commands, publishes feedback
# ============================================================================
class ManipulatorBridgeNode(Node):
    """
    Isaac Sim ↔ ROS 2 bridge for the cup manipulator tendon.

    mode=joint_command:
        Subscribe /manip/joint_command (JointState) → queue
        Publish   /manip/ee_position   (Point)

    mode=ee_command:
        Subscribe /manip/ee_command    (Point) → queue
        Publish   /manip/joint_state   (JointState)
    """

    def __init__(self, mode: str, cmd_queue: queue.Queue):
        super().__init__('isaac_manip_bridge')
        self._mode = mode
        self._queue = cmd_queue

        if mode == "joint_command":
            self.create_subscription(
                JointState, '/manip/joint_command',
                self._on_joint_command, _CONTROL_QOS,
            )
            self._ee_pub = self.create_publisher(
                Point, '/manip/ee_position', _CONTROL_QOS,
            )
            self.get_logger().info(
                "Mode: joint_command — sub /manip/joint_command, pub /manip/ee_position"
            )
        else:
            self.create_subscription(
                Point, '/manip/ee_command',
                self._on_ee_command, _CONTROL_QOS,
            )
            self._joint_pub = self.create_publisher(
                JointState, '/manip/joint_state', _CONTROL_QOS,
            )
            self.get_logger().info(
                "Mode: ee_command — sub /manip/ee_command, pub /manip/joint_state"
            )

    def _on_joint_command(self, msg: JointState):
        if len(msg.position) >= 2:
            self._queue.put(("joint", np.array(msg.position[:2])))

    def _on_ee_command(self, msg: Point):
        self._queue.put(("ee", np.array([msg.x, msg.y])))

    def publish_ee_position(self, ee: np.ndarray):
        msg = Point()
        msg.x, msg.y, msg.z = float(ee[0]), float(ee[1]), float(ee[2])
        self._ee_pub.publish(msg)

    def publish_joint_state(self, q: np.ndarray, v: np.ndarray):
        msg = JointState()
        msg.name = ['link1_base', 'link2_link1']
        msg.position = [float(q[0]), float(q[1])]
        msg.velocity = [float(v[0]), float(v[1])]
        self._joint_pub.publish(msg)


# ============================================================================
# SCENE SETUP
# ============================================================================
def add_lighting(stage):
    distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant.CreateIntensityAttr(1000.0)
    xformable = UsdGeom.Xformable(stage.GetPrimAtPath("/World/DistantLight"))
    xformable.AddRotateXYZOp().Set(Gf.Vec3d(315.0, 0, 0))
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(300.0)


def add_ee_marker(stage, manipulator):
    ee_pos = manipulator.get_end_effector_position()
    sphere = UsdGeom.Sphere.Define(stage, "/World/EE_Marker")
    sphere.GetRadiusAttr().Set(0.008)
    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.2, 0.2)])
    xformable = UsdGeom.Xformable(sphere.GetPrim())
    xformable.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in ee_pos]))


def update_ee_marker(stage, position: np.ndarray):
    prim = stage.GetPrimAtPath("/World/EE_Marker")
    if prim.IsValid():
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))


# ============================================================================
# USD CABLE RENDERING
# ============================================================================

_CABLE_ROOT = "/World/Cables"
_CABLE_RADIUS = 0.0005


def _usd_cylinder(stage, path: str, p0: np.ndarray, p1: np.ndarray, color_rgb):
    """Create or update a thin cylinder prim between two world-frame points."""
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
    """Draw all cable segments and wrap arcs as USD Cylinder prims."""
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
    print(colored("✓ Cables drawn", 'green'), flush=True)


def update_cables_usd(stage, drake_cable: DrakeCablePlant):
    """Remove and redraw cable prims after joint angles changed."""
    cable_prim = stage.GetPrimAtPath(_CABLE_ROOT)
    if cable_prim.IsValid():
        stage.RemovePrim(_CABLE_ROOT)
    draw_cables_usd(stage, drake_cable)


# ============================================================================
# ARGS
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Cup manipulator tendon — Isaac Sim ROS 2 bridge',
    )
    parser.add_argument('--render', choices=_RENDER_CHOICES, default=_render_mode)
    parser.add_argument(
        '--mode', choices=('joint_command', 'ee_command'),
        default=_mode,
        help='joint_command: receive joints, publish EE | ee_command: receive EE, publish joints',
    )
    parser.add_argument('--q1', type=float, default=10.0, help='Initial q1 [deg]')
    parser.add_argument('--q2', type=float, default=-10.0, help='Initial q2 [deg]')
    parser.add_argument('--damping', type=float, nargs=2, default=[0.05, 0.05])
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================
def main():
    args = parse_args()

    print("=" * 60, flush=True)
    print(f"  Cup Manipulator Tendon — Isaac Sim ROS 2 ({args.mode})", flush=True)
    print("=" * 60, flush=True)

    # ── ROS 2 init ───────────────────────────────────────────────────────────
    rclpy.init()
    cmd_queue = queue.Queue()
    ros_node = ManipulatorBridgeNode(args.mode, cmd_queue)

    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()
    print(colored("✓ ROS 2 node started", 'green'), flush=True)

    # ── Robot config ─────────────────────────────────────────────────────────
    config = create_cable_manipulator_config(
        urdf_path="model_using_onshape_to_robot/manipulator_cable_isaac/manipulator_cable_obj.urdf",
        joint_angles={
            'link1_base':  np.deg2rad(args.q1),
            'link2_link1': np.deg2rad(args.q2),
        },
        damping=tuple(args.damping),
    )

    manipulator = CupManipulatorTendonIsaac(config, enable_visualization=True)
    manipulator.prepare_usd()

    # ── World ────────────────────────────────────────────────────────────────
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1 / 500,
        rendering_dt=1 / 60,
    )
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()

    add_lighting(stage)
    manipulator.load_urdf()
    manipulator.weld_base_to_world(
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0]),
    )
    manipulator.set_joint_properties()
    manipulator.add_joint_actuators()
    manipulator.add_end_effector_frame()

    world.reset()
    manipulator.initialize_state()
    manipulator.set_initial_positions()

    for _ in range(10):
        world.step(render=True)
        simulation_app.update()

    add_ee_marker(stage, manipulator)

    # ── Drake cable rig (headless FK for cable tangent computation) ───────────
    drake_urdf = "model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf"
    q = manipulator.get_positions_user_order()
    drake_cable = DrakeCablePlant(drake_urdf, q1=float(q[0]), q2=float(q[1]))
    draw_cables_usd(stage, drake_cable)

    ee = manipulator.get_end_effector_position()
    print(colored(
        f"\n✓ Scene ready  mode={args.mode}\n"
        f"  q1 = {np.rad2deg(q[0]):+.2f}°  q2 = {np.rad2deg(q[1]):+.2f}°\n"
        f"  EE = ({ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}) m",
        'green', attrs=['bold'],
    ))
    print("Waiting for ROS 2 commands...\n")

    step_count = 0

    # ── Simulation loop ──────────────────────────────────────────────────────
    try:
        while simulation_app.is_running():
            world.step(render=True)

            # Drain queue — keep only the latest command
            latest = None
            while not cmd_queue.empty():
                try:
                    latest = cmd_queue.get_nowait()
                except queue.Empty:
                    break

            if latest is None:
                continue

            cmd_type, data = latest
            step_count += 1

            if cmd_type == "joint":
                # Mode 1: receive joint positions, apply them, publish EE
                q1, q2 = data
                manipulator.set_positions_user_order([q1, q2])
                world.step(render=True)
                simulation_app.update()

                ee = manipulator.get_end_effector_position()
                update_ee_marker(stage, ee)
                drake_cable.update(q1, q2)
                update_cables_usd(stage, drake_cable)
                ros_node.publish_ee_position(ee)

                if step_count % 50 == 0:
                    print(
                        f"[Isaac] #{step_count} joint_cmd → "
                        f"q=({np.rad2deg(q1):+.1f}°, {np.rad2deg(q2):+.1f}°)  "
                        f"EE=({ee[0]:.3f}, {ee[1]:.3f})",
                        flush=True,
                    )

            elif cmd_type == "ee":
                # Mode 2: receive EE target, solve IK, apply, publish joints
                target_xy = data
                q_now = manipulator.get_positions_user_order()
                q_sol, ok = manipulator.compute_ik_analytical(target_xy, q_now)

                if ok:
                    manipulator.set_positions_user_order([q_sol[0], q_sol[1]])
                    world.step(render=True)
                    simulation_app.update()

                    q = manipulator.get_positions_user_order()
                    v = manipulator.get_velocities_user_order()
                    ee = manipulator.get_end_effector_position()
                    update_ee_marker(stage, ee)
                    drake_cable.update(q[0], q[1])
                    update_cables_usd(stage, drake_cable)
                    ros_node.publish_joint_state(q, v)

                    if step_count % 50 == 0:
                        print(
                            f"[Isaac] #{step_count} ee_cmd → "
                            f"target=({target_xy[0]:.3f}, {target_xy[1]:.3f})  "
                            f"q=({np.rad2deg(q[0]):+.1f}°, {np.rad2deg(q[1]):+.1f}°)",
                            flush=True,
                        )
                else:
                    if step_count % 50 == 0:
                        print(
                            colored(
                                f"[Isaac] #{step_count} IK failed for "
                                f"({target_xy[0]:.3f}, {target_xy[1]:.3f})",
                                'red',
                            ),
                            flush=True,
                        )

    except KeyboardInterrupt:
        pass

    ros_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()
    print("[Isaac Sim] Closed.")


if __name__ == '__main__':
    main()
