#!/usr/bin/env python3
"""
script_cup_manipulator_pendulam_tendon_sea_isaac_sim.py
──────────────────────────────────────────────────────────
Series Elastic Actuator (SEA) cable simulation in Isaac Sim.

NO PYDRAKE DEPENDENCY at runtime (uses Isaac Sim ArticulationView for M, h).
Cable visualization uses a headless Drake plant under the hood (via cable.py)
but the simulation and control loop are pure NumPy + Isaac Sim PhysX.

Architecture
~~~~~~~~~~~~
    Trajectory → IK → CT Controller → SEA Actuator → Isaac Sim PhysX
                                          ▲ state ──────────────┘

    Joint 1: rigid direct drive  (τ₁ passes straight through)
    Joint 2: motor → cable spring → pulley (k_s, b_c, ω_m modelled by SEA)

SEA Physics (joint 2 only)
──────────────────────────
    l_m_des  = r_p·q₂ + τ₂_des / (k_s·r_p)   steady-state inversion
    l̇_m     = ω_m · (l_m_des − l_m)           first-order motor servo
    δ        = l_m − r_p·q₂                    spring extension [m]
    F_cable  = k_s·δ + b_c·(l̇_m − r_p·q̇₂)   spring–damper force [N]
    τ₂       = r_p · F_cable                   applied joint torque [Nm]

Usage
~~~~~
    # Default SEA with rect trajectory
    python script_cup_manipulator_pendulam_tendon_sea_isaac_sim.py

    # Soft spring (high lag)
    python script_cup_manipulator_pendulam_tendon_sea_isaac_sim.py --spring-stiffness 30

    # Stiff spring (near-rigid)
    python script_cup_manipulator_pendulam_tendon_sea_isaac_sim.py --spring-stiffness 5000

    # WebRTC streaming (for remote)
    python script_cup_manipulator_pendulam_tendon_sea_isaac_sim.py --render websocket

    # Headless (benchmarking)
    python script_cup_manipulator_pendulam_tendon_sea_isaac_sim.py --render headless
"""

# ============================================================================
# PRE-PARSE render flag BEFORE any Isaac Sim import
# ============================================================================
import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_RENDER_CHOICES = ("native", "websocket", "headless")
_render_mode = "native"
for _i, _arg in enumerate(sys.argv):
    if _arg == "--render" and _i + 1 < len(sys.argv):
        _render_mode = sys.argv[_i + 1]
        if _render_mode not in _RENDER_CHOICES:
            print(f"[ERROR] --render must be one of {_RENDER_CHOICES}, got '{_render_mode}'")
            sys.exit(1)

# ============================================================================
# QUIET STARTUP
# ============================================================================
from project_utils.log_isaacsim import IsaacSimLogger

_log = IsaacSimLogger.from_argv()
_log.suppress()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": _render_mode == "headless",
    "width":    1280,
    "height":   720,
    "hide_ui":  True,
})

if _render_mode == "websocket":
    import subprocess
    from isaacsim.core.utils.extensions import enable_extension

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
    enable_extension("omni.kit.livestream.webrtc")

    _connect_ip = _tailscale_ip if _tailscale_ip else "localhost"
    _log.print("\n" + "=" * 60)
    _log.print("  WebRTC streaming enabled (omni.kit.livestream.webrtc)")
    _log.print(f"  Port          : 49100")
    if _tailscale_ip:
        _log.print(f"  Tailscale IP  : {_tailscale_ip}")
    _log.print(f"  Mac client    : connect to  {_connect_ip} : 49100")
    _log.print("=" * 60 + "\n")

# ============================================================================
# IMPORTS (safe after SimulationApp)
# ============================================================================
import argparse
import math
import numpy as np
import signal
import time as _time
from termcolor import colored

import matplotlib
try:
    matplotlib.use('Agg')
except Exception:
    pass
import matplotlib.pyplot as plt

import omni.usd
from pxr import UsdGeom, Gf

from omni.isaac.core import World

from robots.cup_manipulator_tendon_isaac import (
    CupManipulatorTendonIsaac,
    create_cable_manipulator_config,
    solve_2r_ik,
    forward_kinematics_2r,
    analytical_jacobian_2r,
)
from controller.computed_torque_isaacsim import (
    ComputedTorqueController,
    ik_to_joint_space_references,
)
from controller.trajectory import (
    RectTrajectory,
    CircleTrajectory,
    LineTrajectory,
    PreambleTrajectorySource,
    build_move_to_start,
)
from actuators.sea_isaacsim import SEACableActuatorNP
from actuators.motor_dynamics import MotorMode
from actuators.motor import get_motor, MOTOR_CHOICES
from project_utils.viz_cables_isaacsim import CableVisualizerIsaac

# Spring zigzag generator (from cable module — used for spring USD prims)
from cable import spring_zigzag_points

_log.restore()

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

parser = argparse.ArgumentParser(
    description='SEA cable manipulator — Isaac Sim (no Drake)',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument('--mode', type=str, default='sea',
                    choices=['sea', 'scene-viz'],
                    help='Simulation mode (default: sea)')
parser.add_argument('--render', type=str, default=_render_mode,
                    choices=['native', 'websocket', 'headless'],
                    help='Render mode')
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--duration', type=float, default=10.0,
                    help='Lap duration [s]')
parser.add_argument('--dt', type=float, default=1.0 / 100.0,
                    help='Physics timestep [s]')

# Tilt
parser.add_argument('--tilt-roll', type=float, default=0.0)
parser.add_argument('--tilt-pitch', type=float, default=0.0)

# Joint properties
parser.add_argument('--joint-damping', type=float, nargs=2, default=[0.05, 0.05],
                    metavar=('D1', 'D2'))
parser.add_argument('--joint-stiffness', type=float, nargs=2, default=[0.0, 0.0],
                    metavar=('K1', 'K2'))

# CT gains
parser.add_argument('--ct-kp', type=float, default=100.0,
                    help='CT position gain Kp [s⁻²]')
parser.add_argument('--ct-kd', type=float, default=40.0,
                    help='CT velocity gain Kd [s⁻¹]')
parser.add_argument('--ct-tau-max', type=float, default=None,
                    help='Torque saturation [Nm].  Default: motor peak_torque_joint.')

# Motor model
_mot = parser.add_argument_group("motor model  (elbow / joint 2)")
_mot.add_argument('--motor', choices=MOTOR_CHOICES, default='AK60_6_KV80_Config',
                  help='CubeMars motor model for the elbow joint.')

# SEA parameters
_sea = parser.add_argument_group("SEA cable model (joint 2)")
_sea.add_argument('--sea-mode', choices=['torque', 'position'], default='torque',
                  help="Motor dynamics mode: 'torque' = 2nd-order rotor, "
                       "'position' = 1st-order servo.")
_sea.add_argument('--spring-stiffness', type=float, default=3000, metavar='K_S',
                  help='Cable spring stiffness k_s [N/m]. Lower → more lag.')
_sea.add_argument('--cable-damping', type=float, default=2.0, metavar='B_C',
                  help='Cable dashpot damping b_c [N·s/m]')
_DEFAULT_MOTOR_BW = 100.0  # rad/s  (conservative closed-loop estimate)
_sea.add_argument('--motor-bandwidth', type=float, default=None, metavar='W_M',
                  help='Motor position servo bandwidth ω_m [rad/s].  '
                       f'Default: {_DEFAULT_MOTOR_BW} rad/s (position mode only).')
_sea.add_argument('--motor-substeps', type=int, default=None, metavar='N',
                  help='Motor integrator sub-steps per physics step.  '
                       'Default: auto-computed for numerical stability.')

# Trajectory
parser.add_argument('--traj-shape', type=str, default='rect',
                    choices=['rect', 'circle', 'line'])
parser.add_argument('--traj-n', type=int, default=60)
parser.add_argument('--traj-x-range', type=float, nargs=2, default=[0.49, 0.51],
                    metavar=('XMIN', 'XMAX'))
parser.add_argument('--traj-y-range', type=float, nargs=2, default=[-0.08, 0.08],
                    metavar=('YMIN', 'YMAX'))
parser.add_argument('--traj-cx', type=float, default=0.4)
parser.add_argument('--traj-cy', type=float, default=0.0)
parser.add_argument('--traj-radius', type=float, default=0.1)
parser.add_argument('--traj-v-max', type=float, default=0.9)
parser.add_argument('--traj-v-corner', type=float, default=0.05)
parser.add_argument('--traj-corner-blend', type=float, default=0.35)

# Move to start
parser.add_argument('--move-duration', type=float, default=3.0)

# Home
parser.add_argument('--home-ee', type=float, nargs=2, default=None, metavar=('X', 'Y'))
parser.add_argument('--home-joints', type=float, nargs=2, default=None,
                    metavar=('Q1_DEG', 'Q2_DEG'))

args = parser.parse_args()

# ─── Motor-derived defaults ───────────────────────────────────────────────────
_motor = get_motor(args.motor)
_motor_mode = MotorMode(args.sea_mode)
if args.motor_bandwidth is None:
    args.motor_bandwidth = _DEFAULT_MOTOR_BW
if args.ct_tau_max is None:
    args.ct_tau_max = _motor.peak_torque_joint

_mode_label = ("torque (2nd-order rotor)" if _motor_mode == MotorMode.TORQUE
               else "position (1st-order servo)")
print(colored(
    f"\n  Motor: {args.motor}  —  SEA mode: {_mode_label}"
    f"\n    gear ratio      = {_motor.gear_ratio}"
    f"\n    peak torque     = {_motor.peak_torque_joint} Nm  (τ_max)"
    f"\n    continuous τ    = {_motor.continuous_torque_joint} Nm"
    f"\n    max joint vel   = {_motor.max_velocity_joint:.2f} rad/s"
    f"  ({_motor.max_velocity_joint * 60 / (2 * np.pi):.1f} rpm)"
    f"\n    rotor inertia   = {_motor.rotor_inertia_joint:.5f} kg·m²  (reflected)"
    f"\n    → ω_m = {args.motor_bandwidth:.2f} rad/s"
    f"   τ_max = {args.ct_tau_max:.1f} Nm",
    "yellow",
))

# ============================================================================
# PATHS & CONFIG
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
URDF_PATH = str(PROJECT_ROOT / "model_using_onshape_to_robot"
                / "manipulator_cable" / "manipulator_cable_obj.urdf")

MANIP_CONFIG = create_cable_manipulator_config(
    urdf_path=URDF_PATH,
    joint_angles={
        'link1_base':  np.deg2rad(10.0),
        'link2_link1': np.deg2rad(-10.0),
    },
    damping=tuple(args.joint_damping),
    stiffness=tuple(args.joint_stiffness),
    tilt_roll_deg=args.tilt_roll,
    tilt_pitch_deg=args.tilt_pitch,
)


# ============================================================================
# SPRING VISUALIZER (USD zigzag prims)
# ============================================================================

_SPRING_ROOT = "/World/Springs"
_SPRING_RADIUS = 0.0008      # cylinder radius for spring coils [m]
_SPRING_N_COILS = 6
_SPRING_AMPLITUDE = 0.004    # helix radius [m]
_SPRING_COLOR_RGB = (0.9, 0.6, 0.0)   # orange/gold


class SpringVisualizerIsaac:
    """Visualize SEA springs as USD zigzag cylinder prims in Isaac Sim.

    Pre-allocates N cylinder prims per spring (green + red cable).
    Update transforms each frame based on spring extension δ.

    Usage::

        spring_viz = SpringVisualizerIsaac(stage, cable_viz)
        spring_viz.create_prims(q1_init, q2_init)   # BEFORE world.reset()
        # In sim loop:
        spring_viz.update(q1, q2, delta)
    """

    def __init__(self, stage, cable_viz: CableVisualizerIsaac,
                 k_s: float = 200.0, r_p: float = 0.048, tau_max: float = 50.0):
        self._stage = stage
        self._cable_viz = cable_viz
        self._n_coils = _SPRING_N_COILS
        self._amplitude = _SPRING_AMPLITUDE
        self._prim_count = 0
        # Normalisation: delta_max = tau_max / (k_s · r_p) is the cable
        # extension at full torque.  We map [0, delta_max] → visual
        # spring fraction [_REST_FRAC, _MAX_FRAC] so the stretch is
        # clearly visible for any k_s.
        self._delta_max = tau_max / max(k_s * r_p, 1e-9)

    def create_prims(self, q1: float, q2: float):
        """Pre-allocate spring USD cylinder prims. Call BEFORE world.reset()."""
        drake_cable = self._cable_viz.drake_cable
        if drake_cable is None:
            print("[SpringViz] Warning: drake_cable not initialized, skipping.")
            return

        # Get cable world points to find last-segment endpoints
        cable_data = list(drake_cable.get_cable_world_points())
        self._prim_count = 0

        for ri, (route, pts) in enumerate(cable_data):
            if len(pts) < 2:
                continue
            # Last segment: from second-to-last to last point
            p0 = pts[-2]   # big pulley exit
            p1 = pts[-1]   # endpoint on link2

            # Place spring at 50% of the last segment
            seg_dir = p1 - p0
            seg_len = np.linalg.norm(seg_dir)
            spring_frac = 0.30  # fraction of segment
            sf2 = spring_frac / 2.0
            mid = 0.5
            p_spring_start = p0 + (mid + sf2) * seg_dir / max(seg_len, 1e-9) * seg_len
            p_spring_end   = p0 + (mid - sf2) * seg_dir / max(seg_len, 1e-9) * seg_len

            # Generate zigzag points
            zz = spring_zigzag_points(
                p_spring_start, p_spring_end,
                n_coils=self._n_coils, amplitude=self._amplitude,
            )

            label = "green" if ri == 0 else "red"
            base_path = f"{_SPRING_ROOT}/{label}"

            # Create cylinder prims for each zigzag segment
            for j in range(len(zz) - 1):
                prim_path = f"{base_path}/coil{j:03d}"
                self._create_cylinder(prim_path, zz[j], zz[j + 1])

            self._prim_count += len(zz) - 1

        print(colored(
            f"  [SpringViz] Pre-allocated {self._prim_count} spring coil prims",
            "cyan",
        ))

    def update(self, q1: float, q2: float, spring_extension: float = 0.0):
        """Update spring visualization based on current joint angles and δ."""
        drake_cable = self._cable_viz.drake_cable
        if drake_cable is None:
            return

        cable_data = list(drake_cable.get_cable_world_points())
        for ri, (route, pts) in enumerate(cable_data):
            if len(pts) < 2:
                continue
            p0 = pts[-2]
            p1 = pts[-1]
            seg_dir = p1 - p0
            seg_len = np.linalg.norm(seg_dir)
            if seg_len < 1e-9:
                continue

            # Dynamic spring length based on SEA extension δ
            # Green (ri=0): taut when δ > 0 (F_raw > 0)
            # Red   (ri=1): taut when δ < 0 (F_raw < 0)
            _REST_FRAC = 0.15   # visual fraction at δ = 0
            _MAX_FRAC  = 0.65   # visual fraction at δ = δ_max
            route_ext = max(spring_extension, 0.0) if ri == 0 else max(-spring_extension, 0.0)
            norm = min(route_ext / max(self._delta_max, 1e-9), 1.0)
            spring_frac = np.clip(
                _REST_FRAC + norm * (_MAX_FRAC - _REST_FRAC),
                0.05, 0.70,
            )
            sf2 = spring_frac / 2.0
            mid = 0.5

            t0 = mid - sf2
            t1 = mid + sf2
            p_spring_start = p0 + t1 * seg_dir  # closer to p0 (pulley)
            p_spring_end   = p0 + t0 * seg_dir  # closer to p1 (endpoint)

            zz = spring_zigzag_points(
                p_spring_start, p_spring_end,
                n_coils=self._n_coils, amplitude=self._amplitude,
            )

            label = "green" if ri == 0 else "red"
            base_path = f"{_SPRING_ROOT}/{label}"

            for j in range(len(zz) - 1):
                prim_path = f"{base_path}/coil{j:03d}"
                self._update_cylinder(prim_path, zz[j], zz[j + 1])

    def _create_cylinder(self, path: str, p0: np.ndarray, p1: np.ndarray):
        """Create a USD cylinder prim between two 3D points."""
        diff = p1 - p0
        length = float(np.linalg.norm(diff))
        if length < 1e-9:
            length = 1e-6
        mid = (p0 + p1) * 0.5

        z_hat = diff / max(length, 1e-9)
        tmp = np.array([0., 1., 0.]) if abs(z_hat[0]) > 0.9 else np.array([1., 0., 0.])
        x_hat = np.cross(tmp, z_hat)
        norm = np.linalg.norm(x_hat)
        if norm > 1e-9:
            x_hat /= norm
        y_hat = np.cross(z_hat, x_hat)

        mat = Gf.Matrix4d(
            float(x_hat[0]), float(x_hat[1]), float(x_hat[2]), 0.0,
            float(y_hat[0]), float(y_hat[1]), float(y_hat[2]), 0.0,
            float(z_hat[0]), float(z_hat[1]), float(z_hat[2]), 0.0,
            float(mid[0]),   float(mid[1]),   float(mid[2]),   1.0,
        )

        cyl = UsdGeom.Cylinder.Define(self._stage, path)
        cyl.GetRadiusAttr().Set(_SPRING_RADIUS)
        cyl.GetHeightAttr().Set(length)
        cyl.GetDisplayColorAttr().Set([
            Gf.Vec3f(*[float(c) for c in _SPRING_COLOR_RGB])
        ])
        xf = UsdGeom.Xformable(cyl.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(mat)

    def _update_cylinder(self, path: str, p0: np.ndarray, p1: np.ndarray):
        """Update an existing USD cylinder prim."""
        prim = self._stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return

        diff = p1 - p0
        length = float(np.linalg.norm(diff))
        if length < 1e-9:
            UsdGeom.Cylinder(prim).GetHeightAttr().Set(0.0)
            return
        mid = (p0 + p1) * 0.5

        z_hat = diff / length
        tmp = np.array([0., 1., 0.]) if abs(z_hat[0]) > 0.9 else np.array([1., 0., 0.])
        x_hat = np.cross(tmp, z_hat)
        norm = np.linalg.norm(x_hat)
        if norm > 1e-9:
            x_hat /= norm
        y_hat = np.cross(z_hat, x_hat)

        mat = Gf.Matrix4d(
            float(x_hat[0]), float(x_hat[1]), float(x_hat[2]), 0.0,
            float(y_hat[0]), float(y_hat[1]), float(y_hat[2]), 0.0,
            float(z_hat[0]), float(z_hat[1]), float(z_hat[2]), 0.0,
            float(mid[0]),   float(mid[1]),   float(mid[2]),   1.0,
        )

        cyl = UsdGeom.Cylinder(prim)
        cyl.GetHeightAttr().Set(length)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(mat)


# ============================================================================
# TRAJECTORY BUILDER
# ============================================================================

def build_trajectory(args):
    """Create trajectory object from CLI args."""
    if args.traj_shape == 'rect':
        return RectTrajectory(
            x_range=tuple(args.traj_x_range),
            y_range=tuple(args.traj_y_range),
            N=args.traj_n, lap_duration=args.duration,
            v_max=args.traj_v_max, v_corner=args.traj_v_corner,
            corner_blend=args.traj_corner_blend,
        )
    elif args.traj_shape == 'circle':
        return CircleTrajectory(
            cx=args.traj_cx, cy=args.traj_cy, radius=args.traj_radius,
            N=args.traj_n, lap_duration=args.duration,
        )
    else:
        return LineTrajectory(
            cx=args.traj_cx, cy=args.traj_cy, radius=args.traj_radius,
            N=args.traj_n, lap_duration=args.duration,
        )


# ============================================================================
# MAIN SIMULATION — SEA
# ============================================================================

def run_sea(args):
    """Full SEA computed-torque simulation loop in Isaac Sim."""

    print("\n" + "=" * 80)
    print(colored(
        f"SEA CABLE — Isaac Sim  [{_mode_label}]",
        "cyan", attrs=["bold"],
    ))
    print(colored(
        f"  k_s = {args.spring_stiffness} N/m   "
        f"b_c = {args.cable_damping} N·s/m   "
        f"ω_m = {args.motor_bandwidth} rad/s"
        f"   motor={args.motor}  mode={args.sea_mode}",
        "cyan",
    ))
    print("=" * 80)

    # ── 1. Create robot wrapper ─────────────────────────────────────────────
    manip = CupManipulatorTendonIsaac(MANIP_CONFIG)
    manip.prepare_usd()

    # ── 2. Create World ─────────────────────────────────────────────────────
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=args.dt,
        rendering_dt=args.dt,
    )
    world.scene.add_default_ground_plane()

    manip.load_urdf(world)
    orientation = np.deg2rad([
        MANIP_CONFIG.tilt_roll_deg,
        MANIP_CONFIG.tilt_pitch_deg,
        0.0,
    ])
    manip.weld_base_to_world(position=np.zeros(3), orientation=orientation)
    manip.add_end_effector_frame()
    manip.set_joint_properties()
    manip.add_joint_actuators()

    # ── 3. Cable visualization — pre-allocate BEFORE world.reset() ──────────
    _stage = omni.usd.get_context().get_stage()
    _drake_urdf = str(PROJECT_ROOT / "model_using_onshape_to_robot"
                      / "manipulator_cable" / "manipulator_cable_obj.urdf")
    cable_viz = CableVisualizerIsaac(_stage, _drake_urdf)
    cable_viz.create_prims(
        q1=MANIP_CONFIG.joint_configs['link1_base'].position,
        q2=MANIP_CONFIG.joint_configs['link2_link1'].position,
    )
    _CABLE_UPDATE_INTERVAL = 5

    # ── 4. Spring visualization — pre-allocate BEFORE world.reset() ─────────
    spring_viz = SpringVisualizerIsaac(
        _stage, cable_viz,
        k_s=args.spring_stiffness, r_p=manip.r_p, tau_max=args.ct_tau_max,
    )
    spring_viz.create_prims(
        q1=MANIP_CONFIG.joint_configs['link1_base'].position,
        q2=MANIP_CONFIG.joint_configs['link2_link1'].position,
    )
    _SPRING_UPDATE_INTERVAL = 5

    # ── 5. First reset → Articulation ───────────────────────────────────────
    world.reset()
    manip.initialize_state()

    # ── 6. Initialize dynamics view (ArticulationView for M, C, g) ──────────
    manip.initialize_dynamics_view(world)

    L1, L2 = manip._get_link_lengths()
    r_p = manip.r_p
    print(colored(f"  Link lengths: L1={L1*1e3:.1f} mm  L2={L2*1e3:.1f} mm", "cyan"))
    print(colored(f"  Pulley radius: r_p={r_p*1e3:.1f} mm", "cyan"))

    # ── 7. Build trajectory ─────────────────────────────────────────────────
    main_traj = build_trajectory(args)
    print(colored(
        f"  ✓ Trajectory: {args.traj_shape}  N={args.traj_n}  "
        f"lap={args.duration:.1f} s",
        "green",
    ))

    # ── 8. Compute initial pose via IK ──────────────────────────────────────
    seed = np.array([np.deg2rad(5.0), np.deg2rad(15.0)])
    p_first = main_traj.eval_position(0.0)

    q_init = None
    if args.home_joints is not None:
        q_init = np.deg2rad(np.array(args.home_joints, dtype=float))
    elif args.home_ee is not None:
        q_init, ok = solve_2r_ik(L1, L2, np.array(args.home_ee), seed)
        if not ok:
            q_init = None
    if q_init is None:
        q_init, ok = solve_2r_ik(L1, L2, p_first, seed)
        if not ok:
            q_init = np.array([np.deg2rad(5.0), np.deg2rad(15.0)])

    # ── 9. Build move-to-start preamble ─────────────────────────────────────
    q_pre = q_init + np.array([np.deg2rad(-5.0), np.deg2rad(5.0)])
    p_start = forward_kinematics_2r(L1, L2, q_pre[0], q_pre[1])
    p_end = p_first
    v_end = main_traj.eval_velocity(0.0)

    move_spline = build_move_to_start(p_start, p_end, v_end, args.move_duration)
    traj_source = PreambleTrajectorySource(move_spline, main_traj)
    move_duration = traj_source.move_duration

    ee_init = forward_kinematics_2r(L1, L2, q_init[0], q_init[1])
    print(colored(
        f"  ✓ Move-to-start: {move_duration:.1f} s  "
        f"q_init=[{np.rad2deg(q_init[0]):.1f}°, {np.rad2deg(q_init[1]):.1f}°]  "
        f"EE_init=({ee_init[0]*1e3:.1f}, {ee_init[1]*1e3:.1f}) mm",
        "green",
    ))

    # ── 10. Set initial state ───────────────────────────────────────────────
    manip.set_positions_user_order(q_init)
    manip.set_velocities_user_order(np.zeros(2))

    # ── 11. Create CT controller ────────────────────────────────────────────
    ct = ComputedTorqueController(
        Kp=args.ct_kp, Kd=args.ct_kd,
        tau_max=args.ct_tau_max, pulley_radius=r_p,
    )
    wn = ct.omega_n
    zeta = ct.zeta

    # ── 12. Create SEA actuator ─────────────────────────────────────────────
    sea = SEACableActuatorNP(
        r_p=r_p,
        k_s=args.spring_stiffness,
        b_c=args.cable_damping,
        tau_max=args.ct_tau_max,
        dt=args.dt,
        motor_mode=_motor_mode,
        motor_cfg=_motor,
        omega_m=args.motor_bandwidth,
        motor_substeps=args.motor_substeps,
    )
    sea.initialize(q_init[1])
    print(colored(
        f"  ✓ SEA: k_s={args.spring_stiffness} N/m  b_c={args.cable_damping} N·s/m  "
        f"motor_substeps={sea.motor_substeps}  "
        f"(dt_motor={args.dt / sea.motor_substeps:.4f} s)",
        "green",
    ))

    print(colored(
        f"\n▶  SEA CABLE — Isaac Sim  ({_mode_label})"
        f"\n   CT:  Kp={args.ct_kp}  Kd={args.ct_kd}"
        f"   →  ωn={wn:.1f} rad/s  ζ={zeta:.2f}"
        f"\n   SEA: k_s={args.spring_stiffness} N/m  "
        f"b_c={args.cable_damping} N·s/m  "
        f"ω_m={args.motor_bandwidth} rad/s"
        f"\n   Motor: {args.motor}  mode={args.sea_mode}"
        f"\n   tau_max={args.ct_tau_max} Nm   dt={args.dt*1e3:.1f} ms"
        f"\n   Press Ctrl-C to stop and show plots.",
        "cyan",
    ))

    # ── 13. Data logging arrays ─────────────────────────────────────────────
    max_steps = int((args.duration * 5 + move_duration) / args.dt) + 1000
    log_t         = np.zeros(max_steps)
    log_q         = np.zeros((max_steps, 2))
    log_q_dot     = np.zeros((max_steps, 2))
    log_q_des     = np.zeros((max_steps, 2))
    log_tau_des   = np.zeros((max_steps, 2))     # CT desired torques
    log_tau_sea   = np.zeros((max_steps, 2))     # actual applied (after SEA)
    log_tens      = np.zeros((max_steps, 2))     # T_green, T_red
    log_ee_ref    = np.zeros((max_steps, 2))
    log_ee_vel_ref = np.zeros((max_steps, 2))
    log_ee_acc_ref = np.zeros((max_steps, 2))
    # SEA-specific
    log_l_m       = np.zeros(max_steps)
    log_l_m_des   = np.zeros(max_steps)
    log_delta     = np.zeros(max_steps)
    log_F_cable   = np.zeros(max_steps)
    log_tau_motor = np.zeros(max_steps)           # motor-side torque [Nm]

    # ── 14. Simulation loop ─────────────────────────────────────────────────
    step = 0
    t = 0.0
    last_q_des = q_init.copy()
    lap_prev = 0
    move_reported = move_duration <= 0.0

    stop_requested = False

    def _sigint_handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while not stop_requested and step < max_steps:
            # Current state
            q = manip.get_positions_user_order()
            q_dot = manip.get_velocities_user_order()

            # EE references from trajectory
            ee_pos_ref = traj_source.eval_position(t)
            ee_vel_ref = traj_source.eval_velocity(t)
            ee_acc_ref = traj_source.eval_acceleration(t)

            # IK + Jacobian → joint-space references
            q_des, q_dot_ref, q_ddot_ref, ik_ok = ik_to_joint_space_references(
                ee_pos_ref, ee_vel_ref, ee_acc_ref,
                L1, L2, last_q_des, solve_2r_ik,
            )
            if ik_ok:
                last_q_des = q_des.copy()
            else:
                q_des = last_q_des.copy()
                q_dot_ref = np.zeros(2)
                q_ddot_ref = np.zeros(2)

            # Dynamics queries
            M = manip.get_mass_matrix()
            h = manip.get_bias_forces()

            # Computed torque → desired torques
            ct_out = ct.compute(q, q_dot, q_des, q_dot_ref, q_ddot_ref, M, h)
            tau_desired = ct_out.tau_clip  # clipped CT output → SEA input

            # Warn if CT torque was clipped (torque saturation)
            if np.any(np.abs(ct_out.tau_raw) > ct.tau_max):
                _raw_max = np.max(np.abs(ct_out.tau_raw))
                print(colored(
                    f"  ⚠ Torque saturation at t={t:.3f}s: "
                    f"|τ_raw|_max={_raw_max:.2f} Nm > τ_limit={ct.tau_max:.1f} Nm",
                    "yellow",
                ))
            elif np.any(np.abs(ct_out.tau_raw) > 0.8 * ct.tau_max):
                _raw_max = np.max(np.abs(ct_out.tau_raw))
                print(colored(
                    f"  ⚠ Torque near limit at t={t:.3f}s: "
                    f"|τ_raw|_max={_raw_max:.2f} Nm  (80% of {ct.tau_max:.1f} Nm)",
                    "yellow",
                ))

            # SEA actuator: spring-mediated torque for joint 2
            tau_applied, sea_diag = sea.step(tau_desired, q, q_dot)

            # Apply torques to plant
            manip.set_joint_torques(tau_applied)

            # Log
            if step < max_steps:
                log_t[step]         = t
                log_q[step]         = q
                log_q_dot[step]     = q_dot
                log_q_des[step]     = q_des
                log_tau_des[step]   = tau_desired
                log_tau_sea[step]   = tau_applied
                log_tens[step]      = [sea_diag.T_green, sea_diag.T_red]
                log_ee_ref[step]    = ee_pos_ref
                log_ee_vel_ref[step] = ee_vel_ref
                log_ee_acc_ref[step] = ee_acc_ref
                log_l_m[step]       = sea_diag.l_m
                log_l_m_des[step]   = sea_diag.l_m_des
                log_delta[step]     = sea_diag.delta
                log_F_cable[step]   = sea_diag.F_cable
                log_tau_motor[step] = sea_diag.tau_motor

            # Update cable visualization (every N steps)
            if step % _CABLE_UPDATE_INTERVAL == 0:
                cable_viz.update(q[0], q[1])

            # Update spring visualization (every N steps)
            if step % _SPRING_UPDATE_INTERVAL == 0:
                spring_viz.update(q[0], q[1], spring_extension=sea_diag.delta)

            # Step physics
            world.step(render=(_render_mode != "headless"))

            t += args.dt
            step += 1

            # Progress reporting
            if not move_reported and t >= move_duration:
                move_reported = True
                print(colored(
                    f"  ✓ Move-to-start complete at t={t:.2f} s — tracking begins.",
                    "green",
                ))
            lap_now = int(max(0.0, t - move_duration) / args.duration)
            if lap_now > lap_prev:
                lap_prev = lap_now
                print(colored(f"  Lap {lap_now} complete  (t={t:.1f} s)", "cyan"))

    except Exception as e:
        print(colored(f"\n  ✗ Simulation error at t={t:.3f} s: {e}", "red"))
        import traceback
        traceback.print_exc()

    # Restore default signal handler
    signal.signal(signal.SIGINT, signal.default_int_handler)

    # Trim logs
    n = step
    log_t         = log_t[:n]
    log_q         = log_q[:n]
    log_q_dot     = log_q_dot[:n]
    log_q_des     = log_q_des[:n]
    log_tau_des   = log_tau_des[:n]
    log_tau_sea   = log_tau_sea[:n]
    log_tens      = log_tens[:n]
    log_ee_ref    = log_ee_ref[:n]
    log_ee_vel_ref = log_ee_vel_ref[:n]
    log_ee_acc_ref = log_ee_acc_ref[:n]
    log_l_m       = log_l_m[:n]
    log_l_m_des   = log_l_m_des[:n]
    log_delta     = log_delta[:n]
    log_F_cable   = log_F_cable[:n]
    log_tau_motor = log_tau_motor[:n]

    laps_done = int(max(0.0, t - move_duration) / args.duration)
    print(colored(
        f"\n  Simulation stopped at t={t:.2f} s  "
        f"({laps_done} full laps, {n} steps).",
        "yellow",
    ))

    # ── Plot ────────────────────────────────────────────────────────────────
    plot_sea_results(
        log_t, log_q, log_q_dot, log_q_des,
        log_tau_des, log_tau_sea, log_tens,
        log_ee_ref, log_ee_vel_ref, log_ee_acc_ref,
        log_l_m, log_l_m_des, log_delta, log_F_cable,
        log_tau_motor,
        main_traj, L1, L2, r_p, ct, args,
    )


# ============================================================================
# SCENE VIZ
# ============================================================================

def run_scene_viz(args):
    """Load robot, visualise, no control."""
    print("\n" + "=" * 80)
    print(colored("SCENE VISUALIZATION — Isaac Sim (SEA)", "cyan", attrs=["bold"]))
    print("=" * 80)

    manip = CupManipulatorTendonIsaac(MANIP_CONFIG)
    manip.prepare_usd()

    world = World(stage_units_in_meters=1.0, physics_dt=args.dt, rendering_dt=args.dt)
    world.scene.add_default_ground_plane()
    manip.load_urdf(world)
    orientation = np.deg2rad([MANIP_CONFIG.tilt_roll_deg, MANIP_CONFIG.tilt_pitch_deg, 0.0])
    manip.weld_base_to_world(position=np.zeros(3), orientation=orientation)
    manip.add_end_effector_frame()
    manip.set_joint_properties()
    manip.add_joint_actuators()

    _stage = omni.usd.get_context().get_stage()
    _drake_urdf = str(PROJECT_ROOT / "model_using_onshape_to_robot"
                      / "manipulator_cable" / "manipulator_cable_obj.urdf")
    cable_viz = CableVisualizerIsaac(_stage, _drake_urdf)
    cable_viz.create_prims(
        q1=MANIP_CONFIG.joint_configs['link1_base'].position,
        q2=MANIP_CONFIG.joint_configs['link2_link1'].position,
    )

    # Spring visualization
    spring_viz = SpringVisualizerIsaac(
        _stage, cable_viz,
        k_s=args.joint_stiffness[0], r_p=manip.r_p, tau_max=50.0,
    )
    spring_viz.create_prims(
        q1=MANIP_CONFIG.joint_configs['link1_base'].position,
        q2=MANIP_CONFIG.joint_configs['link2_link1'].position,
    )

    world.reset()
    manip.initialize_state()
    manip.set_initial_positions()

    print(colored("\n▶  Scene visualisation — press Ctrl-C to exit.", "cyan"))

    stop_requested = False

    def _sigint_handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while not stop_requested and simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        pass

    signal.signal(signal.SIGINT, signal.default_int_handler)
    print(colored("  Done.", "yellow"))


# ============================================================================
# PLOTTING — two figures (standard 3×3 + SEA diagnostics 4×2)
# ============================================================================

def plot_sea_results(
    t, q, q_dot, q_des, tau_des, tau_sea, tens,
    ee_ref, ee_vel_ref, ee_acc_ref,
    l_m, l_m_des, delta, F_cable,
    tau_motor,
    main_traj, L1, L2, r_p, ct, args,
):
    """Generate two figures: standard 3×3 + SEA-specific 4×2."""

    # EE actual via FK
    ee_x = np.array([forward_kinematics_2r(L1, L2, q[k, 0], q[k, 1])[0] for k in range(len(t))])
    ee_y = np.array([forward_kinematics_2r(L1, L2, q[k, 0], q[k, 1])[1] for k in range(len(t))])

    # EE velocity actual via J·q̇
    ee_vx = np.zeros(len(t))
    ee_vy = np.zeros(len(t))
    for k in range(len(t)):
        J = analytical_jacobian_2r(L1, L2, q[k, 0], q[k, 1])
        v = J @ q_dot[k]
        ee_vx[k], ee_vy[k] = v[0], v[1]

    # EE acceleration via finite diff
    ee_ax = np.gradient(ee_vx, t) if len(t) > 1 else np.zeros(len(t))
    ee_ay = np.gradient(ee_vy, t) if len(t) > 1 else np.zeros(len(t))

    # Joint acceleration via finite diff
    q1_ddot = np.gradient(q_dot[:, 0], t) if len(t) > 1 else np.zeros(len(t))
    q2_ddot = np.gradient(q_dot[:, 1], t) if len(t) > 1 else np.zeros(len(t))

    # Joint velocity/acceleration reference via J⁻¹
    q_dot_ref = np.zeros((len(t), 2))
    q_ddot_ref = np.zeros((len(t), 2))
    for k in range(len(t)):
        J = analytical_jacobian_2r(L1, L2, q_des[k, 0], q_des[k, 1])
        Ji = np.linalg.pinv(J)
        q_dot_ref[k] = Ji @ ee_vel_ref[k]
        q_ddot_ref[k] = Ji @ ee_acc_ref[k]

    wn = ct.omega_n
    zeta = ct.zeta

    def _pct_ylim(*arrays, pct=99.0, margin=0.15):
        all_vals = np.concatenate([a.ravel() for a in arrays])
        lo = np.percentile(all_vals, 100 - pct)
        hi = np.percentile(all_vals, pct)
        span = max(hi - lo, 1e-9)
        return lo - margin * span, hi + margin * span

    plots_dir = PROJECT_ROOT / 'plots'
    plots_dir.mkdir(exist_ok=True)
    stamp = _time.strftime('%Y%m%d_%H%M%S')

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 1: Standard 3×3 (same as CT script)
    # ═══════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(3, 3, figsize=(18, 11))
    fig1.suptitle(
        f'SEA Cable (Isaac Sim) — {args.traj_shape}   '
        f'k_s={args.spring_stiffness}  b_c={args.cable_damping}  '
        f'ω_m={args.motor_bandwidth}   '
        f'Kp={args.ct_kp}  Kd={args.ct_kd}  '
        f'ωn={wn:.1f}  ζ={zeta:.2f}',
        fontsize=11, fontweight='bold',
    )

    # Row 0: End-Effector
    ax = axes1[0, 0]
    ax.plot(t, ee_x, 'b-', lw=1.8, label='x actual')
    ax.plot(t, ee_y, 'r-', lw=1.8, label='y actual')
    ax.plot(t, ee_ref[:, 0], 'b--', lw=1.5, label='x ref')
    ax.plot(t, ee_ref[:, 1], 'r--', lw=1.5, label='y ref')
    ax.set_title('EE Position'); ax.set_ylabel('[m]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    ax = axes1[0, 1]
    ax.plot(t, ee_vx, 'b-', lw=1.8, label='ẋ actual')
    ax.plot(t, ee_vy, 'r-', lw=1.8, label='ẏ actual')
    ax.plot(t, ee_vel_ref[:, 0], 'b--', lw=1.5, label='ẋ ref')
    ax.plot(t, ee_vel_ref[:, 1], 'r--', lw=1.5, label='ẏ ref')
    ax.set_ylim(*_pct_ylim(ee_vx, ee_vy, ee_vel_ref[:, 0], ee_vel_ref[:, 1]))
    ax.set_title('EE Velocity'); ax.set_ylabel('[m/s]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    ax = axes1[0, 2]
    ax.plot(t, ee_ax, 'b-', lw=1.8, label='ẍ actual')
    ax.plot(t, ee_ay, 'r-', lw=1.8, label='ÿ actual')
    ax.plot(t, ee_acc_ref[:, 0], 'b--', lw=1.5, label='ẍ ref')
    ax.plot(t, ee_acc_ref[:, 1], 'r--', lw=1.5, label='ÿ ref')
    ax.set_ylim(*_pct_ylim(ee_ax, ee_ay, ee_acc_ref[:, 0], ee_acc_ref[:, 1]))
    ax.set_title('EE Acceleration'); ax.set_ylabel('[m/s²]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    # Row 1: Joints
    ax = axes1[1, 0]
    ax.plot(t, np.rad2deg(q[:, 0]), 'b-', lw=1.8, label='q1 act')
    ax.plot(t, np.rad2deg(q[:, 1]), 'r-', lw=1.8, label='q2 act')
    ax.plot(t, np.rad2deg(q_des[:, 0]), 'b--', lw=1.5, label='q1 des')
    ax.plot(t, np.rad2deg(q_des[:, 1]), 'r--', lw=1.5, label='q2 des')
    ax.set_title('Joint Position'); ax.set_ylabel('[deg]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    ax = axes1[1, 1]
    ax.plot(t, np.rad2deg(q_dot[:, 0]), 'b-', lw=1.8, label='q̇1 act')
    ax.plot(t, np.rad2deg(q_dot[:, 1]), 'r-', lw=1.8, label='q̇2 act')
    ax.plot(t, np.rad2deg(q_dot_ref[:, 0]), 'b--', lw=1.5, label='q̇1 ref')
    ax.plot(t, np.rad2deg(q_dot_ref[:, 1]), 'r--', lw=1.5, label='q̇2 ref')
    ax.set_ylim(*_pct_ylim(np.rad2deg(q_dot[:, 0]), np.rad2deg(q_dot[:, 1]),
                           np.rad2deg(q_dot_ref[:, 0]), np.rad2deg(q_dot_ref[:, 1])))
    ax.set_title('Joint Velocity'); ax.set_ylabel('[deg/s]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    ax = axes1[1, 2]
    ax.plot(t, np.rad2deg(q1_ddot), 'b-', lw=1.8, label='q̈1 act')
    ax.plot(t, np.rad2deg(q2_ddot), 'r-', lw=1.8, label='q̈2 act')
    ax.plot(t, np.rad2deg(q_ddot_ref[:, 0]), 'b--', lw=1.5, label='q̈1 ref')
    ax.plot(t, np.rad2deg(q_ddot_ref[:, 1]), 'r--', lw=1.5, label='q̈2 ref')
    ax.set_ylim(*_pct_ylim(np.rad2deg(q1_ddot), np.rad2deg(q2_ddot),
                           np.rad2deg(q_ddot_ref[:, 0]), np.rad2deg(q_ddot_ref[:, 1])))
    ax.set_title('Joint Acceleration'); ax.set_ylabel('[deg/s²]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    # Row 2: Torques | Tensions | EE XY
    ax = axes1[2, 0]
    ax.plot(t, tau_des[:, 0], 'b-', lw=1.8, label='τ1 desired (CT)')
    ax.plot(t, tau_des[:, 1], 'r-', lw=1.8, label='τ2 desired (CT)')
    ax.plot(t, tau_sea[:, 0], 'b--', lw=1.5, label='τ1 applied')
    ax.plot(t, tau_sea[:, 1], 'r--', lw=1.5, label='τ2 applied (SEA)')
    ax.axhline(args.ct_tau_max, color='k', ls=':', lw=0.8, label=f'±{args.ct_tau_max} Nm')
    ax.axhline(-args.ct_tau_max, color='k', ls=':', lw=0.8)
    ax.axhline(0, color='k', lw=0.5)
    _tau_peak = max(np.abs(tau_des).max(), np.abs(tau_sea).max(), args.ct_tau_max) * 1.15
    ax.set_ylim(-_tau_peak, _tau_peak)
    ax.set_title('Torque: CT desired vs SEA applied')
    ax.set_ylabel('[Nm]'); ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    ax = axes1[2, 1]
    ax.plot(t, tens[:, 0], 'g-', lw=1.2, label='T_green')
    ax.plot(t, tens[:, 1], 'r-', lw=1.2, label='T_red')
    ax.plot(t, F_cable, 'k--', lw=0.8, label='F_cable (net)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title('Cable Tensions'); ax.set_ylabel('[N]'); ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    ax = axes1[2, 2]
    ax.plot(main_traj.ee_x_tgt, main_traj.ee_y_tgt, 'k--', lw=1.0, label='Reference')
    ax.plot(ee_x, ee_y, 'b-', lw=1.3, label='Actual')
    ax.plot(ee_x[0], ee_y[0], 'go', ms=8, label='Start')
    ax.set_aspect('equal')
    ax.set_title(f'EE Path ({args.traj_shape})')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    fig1.tight_layout()
    fname1 = plots_dir / f'sea_isaac_{args.traj_shape}_{stamp}.png'
    fig1.savefig(str(fname1), dpi=150, bbox_inches='tight')
    print(colored(f"\n  📊 Figure 1 saved: {fname1}", "green"))

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 2: SEA-specific diagnostics (5×2)
    # ═══════════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(5, 2, figsize=(14, 17))
    fig2.suptitle(
        f'SEA Diagnostics — k_s={args.spring_stiffness} N/m   '
        f'b_c={args.cable_damping} N·s/m   '
        f'ω_m={args.motor_bandwidth} rad/s',
        fontsize=12, fontweight='bold',
    )

    # Row 0: EE position X and Y
    ax = axes2[0, 0]
    ax.plot(t, ee_x * 1e3, 'b-', lw=1.5, label='x actual')
    ax.plot(t, ee_ref[:, 0] * 1e3, 'b--', lw=1.2, label='x ref')
    ax.set_title('EE X Position'); ax.set_ylabel('[mm]')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    ax = axes2[0, 1]
    ax.plot(t, ee_y * 1e3, 'r-', lw=1.5, label='y actual')
    ax.plot(t, ee_ref[:, 1] * 1e3, 'r--', lw=1.2, label='y ref')
    ax.set_title('EE Y Position'); ax.set_ylabel('[mm]')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    # Row 1: Torque desired-vs-applied | EE XY path
    ax = axes2[1, 0]
    ax.plot(t, tau_des[:, 1], 'r-', lw=1.5, label='τ2 desired (CT)')
    ax.plot(t, tau_sea[:, 1], 'r--', lw=1.5, label='τ2 applied (SEA)')
    ax.plot(t, tau_des[:, 0], 'b-', lw=1.0, alpha=0.5, label='τ1 (rigid)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title('Torque: Desired vs SEA-Applied')
    ax.set_ylabel('[Nm]'); ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    ax = axes2[1, 1]
    ax.plot(main_traj.ee_x_tgt * 1e3, main_traj.ee_y_tgt * 1e3, 'k--', lw=1.0, label='Reference')
    ax.plot(ee_x * 1e3, ee_y * 1e3, 'b-', lw=1.3, label='Actual (SEA)')
    ax.plot(ee_x[0] * 1e3, ee_y[0] * 1e3, 'go', ms=8, label='Start')
    ax.set_aspect('equal')
    ax.set_title('EE XY Path')
    ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    # Row 2: Motor cable vs joint side | Spring extension + cable force
    ax = axes2[2, 0]
    joint_side = r_p * q[:, 1]
    ax.plot(t, l_m * 1e3, 'b-', lw=1.5, label='l_m (motor)')
    ax.plot(t, l_m_des * 1e3, 'b--', lw=1.2, label='l_m_des')
    ax.plot(t, joint_side * 1e3, 'r-', lw=1.2, label='r_p·q₂ (joint)')
    ax.set_title('Motor Cable vs Joint Side')
    ax.set_ylabel('[mm]'); ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    ax = axes2[2, 1]
    ax2_twin = ax.twinx()
    ln1 = ax.plot(t, delta * 1e3, 'b-', lw=1.5, label='δ (spring ext)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title('Spring Extension & Cable Force')
    ax.set_ylabel('δ [mm]', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ln2 = ax2_twin.plot(t, F_cable, 'r-', lw=1.2, alpha=0.7, label='F_cable')
    ax2_twin.set_ylabel('F [N]', color='r')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=7); ax.grid(True, alpha=0.4)

    # Row 3: Motor-side torque
    _peak_motor = _motor.peak_torque_joint / _motor.gear_ratio
    ax = axes2[3, 0]
    ax.plot(t, tau_motor, 'm-', lw=1.5, label='τ_motor (elbow)')
    ax.axhline( _peak_motor, color='k', ls='--', lw=1.0, label=f'±peak = {_peak_motor:.2f} Nm')
    ax.axhline(-_peak_motor, color='k', ls='--', lw=1.0)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f'Motor-Side Torque  (N={_motor.gear_ratio})')
    ax.set_ylabel('[Nm]'); ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    ax = axes2[3, 1]
    ax.plot(t, tau_des[:, 1], 'r-', lw=1.2, label='τ₂ desired (joint)')
    ax.plot(t, tau_motor * _motor.gear_ratio, 'm--', lw=1.2, label='τ_motor × N (reflected)')
    ax.axhline( args.ct_tau_max, color='k', ls='--', lw=1.0, label=f'±τ_max = {args.ct_tau_max:.1f} Nm')
    ax.axhline(-args.ct_tau_max, color='k', ls='--', lw=1.0)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title('Joint-Side: Desired vs Motor-Reflected')
    ax.set_ylabel('[Nm]'); ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    # Row 4: Torque tracking error | EE tracking error
    ax = axes2[4, 0]
    tau_err = tau_des[:, 1] - tau_sea[:, 1]
    ax.plot(t, tau_err, 'r-', lw=1.2)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title('τ₂ Tracking Error (desired − applied)')
    ax.set_ylabel('[Nm]'); ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.4)

    ax = axes2[4, 1]
    ee_err = np.sqrt((ee_x - ee_ref[:, 0])**2 + (ee_y - ee_ref[:, 1])**2) * 1e3
    ax.plot(t, ee_err, 'b-', lw=1.2)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title('EE Tracking Error (Euclidean)')
    ax.set_ylabel('[mm]'); ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.4)

    fig2.tight_layout()
    fname2 = plots_dir / f'sea_isaac_diag_{args.traj_shape}_{stamp}.png'
    fig2.savefig(str(fname2), dpi=150, bbox_inches='tight')
    print(colored(f"  📊 Figure 2 saved: {fname2}", "green"))

    # Tracking metrics
    err = np.sqrt((ee_x - ee_ref[:, 0])**2 + (ee_y - ee_ref[:, 1])**2)
    print(colored(f"  Final EE: [{ee_x[-1]:.4f}, {ee_y[-1]:.4f}] m", "green"))
    print(colored(f"  Ref at t_end: [{ee_ref[-1, 0]:.4f}, {ee_ref[-1, 1]:.4f}] m", "green"))
    print(colored(f"  Final tracking error: {err[-1]*1e3:.2f} mm", "green"))
    print(colored(f"  Mean tracking RMS: {np.sqrt(np.mean(err**2))*1e3:.2f} mm", "green"))
    print(colored(f"  Mean |δ|: {np.mean(np.abs(delta))*1e3:.3f} mm", "green"))
    print(colored(f"  Max  |δ|: {np.max(np.abs(delta))*1e3:.3f} mm", "green"))

    # Open saved images
    try:
        import subprocess as _sp
        _sp.Popen(["eog", str(fname1)], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        _sp.Popen(["eog", str(fname2)], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    except Exception:
        pass


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    if args.mode == 'scene-viz':
        run_scene_viz(args)
    elif args.mode == 'sea':
        run_sea(args)
    else:
        print(colored(f"Unknown mode: {args.mode}", "red"))
        sys.exit(1)

    _log.close(simulation_app)


if __name__ == "__main__":
    main()
