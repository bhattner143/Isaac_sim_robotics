#!/usr/bin/env python3
"""
script_cup_manipulator_pendulam_tendon_with_exo_isaac_sim.py
────────────────────────────────────────────────────────────
SEA cable simulation with exosuit co-contraction — Isaac Sim port.

Isaac Sim counterpart of
``script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py``.
Extends the plain SEA drive cable with two exosuit cable actuators
(Method B — centred elbow pulley).  The exo can be activated at a
configurable time to inject co-contraction stiffness at the elbow.

Signal graph
~~~~~~~~~~~~
::

    Trajectory → IK → CT Controller → SEA drive ──┐
                                                  ├── τ₁, τ₂_total → PhysX
                       SEA exo (co-contraction) ──┘

    τ₁       = τ₁_CT                              (shoulder, rigid)
    τ₂_total = r_p·F_cable + τ_exo                (elbow: drive + exo)

See the accompanying LaTeX note
``notes_all/notes_cup_manipulator_tendon/isaac-sim/
Exo_Cable_IsaacSim_Implementation.tex`` for the full derivation.

Usage
~~~~~
::

    # Default (exo OFF — passive, transparent)
    python script_cup_manipulator_pendulam_tendon_with_exo_isaac_sim.py

    # Exo ON, activate at t=4 s, Δθ=0.5 rad (co-contraction)
    python script_cup_manipulator_pendulam_tendon_with_exo_isaac_sim.py \
        --exo-activate --exo-activate-time 4 --exo-delta-theta 0.5

    # Higher exo stiffness
    python script_cup_manipulator_pendulam_tendon_with_exo_isaac_sim.py \
        --exo-activate --exo-ks 500 --exo-delta-theta 0.15

    # WebRTC streaming
    python script_cup_manipulator_pendulam_tendon_with_exo_isaac_sim.py --render websocket
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

from robots.cup_manipulator_tendon_with_exo_isaac import (
    CupManipulatorTendonWithExoIsaac,
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
from actuators.sea_exo_isaacsim import SEAExoActuatorNP
from actuators.motor_dynamics import MotorMode
from actuators.motor import get_motor, MOTOR_CHOICES
from project_utils.viz_cables_isaacsim import (
    CableVisualizerIsaac,
    ExoCableVisualizerIsaac,
    ExoSpringVisualizerIsaac,
)

_log.restore()

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

parser = argparse.ArgumentParser(
    description='SEA + Exo co-contraction cable manipulator — Isaac Sim',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--mode', type=str, default='sea-exo',
                    choices=['sea-exo', 'scene-viz'],
                    help='Simulation mode (default: sea-exo)')
parser.add_argument('--render', type=str, default=_render_mode,
                    choices=['native', 'websocket', 'headless'],
                    help='Render mode')
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--duration', type=float, default=10.0,
                    help='Lap duration [s]')
parser.add_argument('--num-laps', type=int, default=3, metavar='N',
                    help='Number of laps before auto-stopping (0 = infinite).')
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

# Drive motor
_mot = parser.add_argument_group("drive motor (elbow / joint 2)")
_mot.add_argument('--motor', choices=MOTOR_CHOICES, default='AK60_6_KV80_Config',
                  help='CubeMars motor for the drive cable.')

# Drive SEA parameters
_sea = parser.add_argument_group("SEA drive cable (joint 2)")
_sea.add_argument('--sea-mode', choices=['torque', 'position'], default='torque',
                  help="Motor dynamics mode for the drive cable.")
_sea.add_argument('--spring-stiffness', type=float, default=10, metavar='K_S',
                  help='Drive cable spring stiffness k_s [N/m].')
_sea.add_argument('--cable-damping', type=float, default=2.0, metavar='B_C',
                  help='Drive cable dashpot damping b_c [N·s/m].')
_DEFAULT_MOTOR_BW = 100.0
_sea.add_argument('--motor-bandwidth', type=float, default=None, metavar='W_M',
                  help='Motor position servo bandwidth ω_m [rad/s] (position mode).')
_sea.add_argument('--motor-substeps', type=int, default=None, metavar='N',
                  help='Motor integrator sub-steps. Default: auto.')

# Exosuit parameters (mirror the PyDrake CLI)
_exo = parser.add_argument_group("exosuit co-contraction (Method B)")
_exo.add_argument('--exo-motor', choices=MOTOR_CHOICES, default='AK60_6_KV80_Config',
                  help='CubeMars motor for both exo cables.')
_exo.add_argument('--exo-ks', type=float, default=500.0, metavar='K_EXO',
                  help='Exo cable spring stiffness [N/m].')
_exo.add_argument('--exo-bc', type=float, default=2.0, metavar='B_EXO',
                  help='Exo cable dashpot damping [N·s/m].')
_exo.add_argument('--exo-r', type=float, default=0.04775, metavar='R_EXO',
                  help='Exo elbow pulley radius [m].')
_exo.add_argument('--exo-delta-theta', type=float, default=0.1, metavar='DTHETA',
                  help='Co-contraction offset Δθ [rad] when activated.')
_exo.add_argument('--exo-activate-time', type=float, default=5.0, metavar='T_ACT',
                  help='Time [s] at which exo activates.')
_exo.add_argument('--no-exo-activate', action='store_true', default=True,
                  help='Keep exo deactivated for the entire simulation (default).')
_exo.add_argument('--exo-activate', dest='no_exo_activate', action='store_false',
                  help='Enable exo activation at --exo-activate-time.')
_exo.add_argument('--exo-reactive', action='store_true', default=False,
                  help='Error-triggered exo activation (hysteresis + hold-time).')
_exo.add_argument('--exo-e-on', type=float, default=5.0, metavar='DEG',
                  help='Reactive mode: elbow error threshold to activate [deg].')
_exo.add_argument('--exo-e-off', type=float, default=2.0, metavar='DEG',
                  help='Reactive mode: elbow error threshold to deactivate [deg].')
_exo.add_argument('--exo-t-hold', type=float, default=0.5, metavar='T_HOLD',
                  help='Reactive mode: seconds below e_off before deactivation.')

# Disturbance (collision simulation)
_dist = parser.add_argument_group("disturbance (collision)")
_dist.add_argument('--disturbance', action='store_true', default=False,
                   help='Inject a disturbance to simulate collision.')
_dist.add_argument('--disturbance-time', type=float, default=7.0, metavar='T_DIST',
                   help='Time [s] at which the disturbance is applied.')
_dist.add_argument('--disturbance-mode',
                   choices=['vel', 'pos', 'torque', 'sine'], default='vel',
                   help='Disturbance type.')
_dist.add_argument('--disturbance-dqdot', type=float, default=60.0, metavar='DQDOT_DEGS')
_dist.add_argument('--disturbance-dq', type=float, default=15.0, metavar='DQ_DEG')
_dist.add_argument('--disturbance-tau', type=float, default=2.0, metavar='TAU_EXT')
_dist.add_argument('--disturbance-dur', type=float, default=1.5, metavar='T_DUR')
_dist.add_argument('--disturbance-freq', type=float, default=3.0, metavar='F_HZ')
_dist.add_argument('--disturbance-cycles', type=float, default=1, metavar='N_CYCLES')

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

# Move-to-start preamble
parser.add_argument('--move-duration', type=float, default=3.0)
parser.add_argument('--home-ee', type=float, nargs=2, default=None, metavar=('X', 'Y'))
parser.add_argument('--home-joints', type=float, nargs=2, default=None,
                    metavar=('Q1_DEG', 'Q2_DEG'))

# I/O
parser.add_argument('--no-show', action='store_true',
                    help='Do not open the saved PNGs (for sweeps / CI runs).')
parser.add_argument('--log-npz', type=str, default=None,
                    help='Optional path to dump simulation logs as .npz (for comparison).')

args = parser.parse_args()

# --disturbance-cycles overrides --disturbance-dur for sine mode
if args.disturbance_cycles is not None:
    args.disturbance_dur = args.disturbance_cycles / args.disturbance_freq

# ── Motor-derived defaults ────────────────────────────────────────────────────
_drive_motor = get_motor(args.motor)
_exo_motor   = get_motor(args.exo_motor)
_motor_mode  = MotorMode(args.sea_mode)
if args.motor_bandwidth is None:
    args.motor_bandwidth = _DEFAULT_MOTOR_BW
if args.ct_tau_max is None:
    args.ct_tau_max = _drive_motor.peak_torque_joint

_mode_label = ("torque (2nd-order rotor)" if _motor_mode == MotorMode.TORQUE
               else "position (1st-order servo)")
print(colored(
    f"\n  Drive motor: {args.motor}  —  SEA mode: {_mode_label}"
    f"\n    gear ratio   = {_drive_motor.gear_ratio}"
    f"\n    peak torque  = {_drive_motor.peak_torque_joint} Nm"
    f"\n  Exo motor: {args.exo_motor}"
    f"\n    gear ratio   = {_exo_motor.gear_ratio}"
    f"\n    peak torque  = {_exo_motor.peak_torque_joint} Nm"
    f"\n  Exo: k_exo={args.exo_ks} N/m  r_exo={args.exo_r:.5f} m"
    f"   Δθ={args.exo_delta_theta:.3f} rad"
    f"\n  k_eff = 2·k_exo·r² = {2*args.exo_ks*args.exo_r**2:.4f} Nm/rad",
    "yellow",
))

# ============================================================================
# PATHS & CONFIG
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
URDF_PATH = str(PROJECT_ROOT / "model_using_onshape_to_robot"
                / "manipulator_cable_exo_springs_elbow_follow"
                / "manipulator_cable_exo_springs_elbow_follow_obj.urdf")

MANIP_CONFIG = create_cable_manipulator_config(
    urdf_path=URDF_PATH,
    joint_angles={
        'link1_base':  np.deg2rad(5.0),
        'link2_link1': np.deg2rad(15.0),
    },
    damping=tuple(args.joint_damping),
    stiffness=tuple(args.joint_stiffness),
    tilt_roll_deg=args.tilt_roll,
    tilt_pitch_deg=args.tilt_pitch,
)


# ============================================================================
# EXO COMMAND SOURCE (timed & reactive) — pure NumPy ports of the PyDrake
# LeafSystems ``_ExoCommandSource`` and ``_ExoReactiveSource``.
# ============================================================================

class _ExoTimedSource:
    """Timed activation.  Returns (activated, Δθ) based on sim time."""
    def __init__(self, t_activate: float, delta_theta: float,
                 never_activate: bool = False):
        self._t_act = float(t_activate)
        self._dth   = float(delta_theta)
        self._never = bool(never_activate)

    def eval(self, t: float, *_, **__) -> tuple[bool, float]:
        if self._never or t < self._t_act:
            return False, 0.0
        return True, self._dth


class _ExoReactiveSource:
    """Error-triggered activation with hysteresis + hold-time (matches the
    Drake ``_ExoReactiveSource`` LeafSystem)."""
    def __init__(self, delta_theta: float, e_on: float, e_off: float,
                 t_hold: float, dt: float):
        self._dth    = float(delta_theta)
        self._e_on   = float(e_on)
        self._e_off  = float(e_off)
        self._t_hold = float(t_hold)
        self._dt     = float(dt)
        self._is_active = False
        self._t_below   = 0.0

    def eval(self, t: float, q2: float, q2_des: float) -> tuple[bool, float]:
        err = abs(float(q2) - float(q2_des))
        if not self._is_active:
            if err >= self._e_on:
                self._is_active = True
                self._t_below   = 0.0
        else:
            if err < self._e_off:
                self._t_below += self._dt
                if self._t_below >= self._t_hold:
                    self._is_active = False
                    self._t_below   = 0.0
            else:
                self._t_below = 0.0
        if self._is_active:
            return True, self._dth
        return False, 0.0


# ============================================================================
# TRAJECTORY BUILDER
# ============================================================================

def build_trajectory(args, L1: float = None, L2: float = None):
    """Create trajectory object from CLI args.

    ``L1``/``L2`` are optional link lengths used by circle/line trajectories
    to clamp waypoints to the reachable annulus (mirrors PyDrake's
    ``build_circle_trajectory``).  Without this clamp, a circle whose
    radius pushes past ``L1+L2`` causes silent IK failures → frozen
    ``q_des`` → exo spring anchored on stale reference → high-frequency
    oscillation visible in q2/δ/τ_exo.
    """
    if args.traj_shape == 'rect':
        return RectTrajectory(
            x_range=tuple(args.traj_x_range),
            y_range=tuple(args.traj_y_range),
            N=args.traj_n, lap_duration=args.duration,
            v_max=args.traj_v_max, v_corner=args.traj_v_corner,
            corner_blend=args.traj_corner_blend,
        )
    if args.traj_shape == 'circle':
        return CircleTrajectory(
            cx=args.traj_cx, cy=args.traj_cy, radius=args.traj_radius,
            N=args.traj_n, lap_duration=args.duration,
            L1=L1, L2=L2,
        )
    return LineTrajectory(
        cx=args.traj_cx, cy=args.traj_cy, radius=args.traj_radius,
        N=args.traj_n, lap_duration=args.duration,
        L1=L1, L2=L2,
    )


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_sea_exo(args):
    """SEA + Exo computed-torque simulation in Isaac Sim (pure NumPy)."""

    print("\n" + "=" * 80)
    print(colored(
        f"SEA + EXO CABLE — Isaac Sim  [{_mode_label}]",
        "cyan", attrs=["bold"],
    ))
    print(colored(
        f"  Drive: k_s={args.spring_stiffness} N/m  b_c={args.cable_damping} N·s/m  "
        f"motor={args.motor}",
        "cyan",
    ))
    print(colored(
        f"  Exo:   k_exo={args.exo_ks} N/m  r_exo={args.exo_r:.5f} m  "
        f"Δθ={args.exo_delta_theta:.3f} rad  "
        f"activate@{args.exo_activate_time:.1f}s  "
        f"enabled={not args.no_exo_activate}  reactive={args.exo_reactive}",
        "cyan",
    ))
    print("=" * 80)

    # ── 1. Create robot wrapper (Isaac URDF → USD → Articulation) ───────────
    manip = CupManipulatorTendonWithExoIsaac(MANIP_CONFIG)
    manip.prepare_usd()

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

    # ── 2. Cable visualisation — PRE-ALLOCATE before world.reset() ─────────
    _stage = omni.usd.get_context().get_stage()
    cable_viz = CableVisualizerIsaac(_stage, URDF_PATH)
    cable_viz.create_prims(
        q1=MANIP_CONFIG.joint_configs['link1_base'].position,
        q2=MANIP_CONFIG.joint_configs['link2_link1'].position,
    )
    exo_viz = ExoCableVisualizerIsaac(_stage, URDF_PATH, springs_enabled=True)
    exo_viz.create_prims(
        q1=MANIP_CONFIG.joint_configs['link1_base'].position,
        q2=MANIP_CONFIG.joint_configs['link2_link1'].position,
    )
    # Helix-spring visualiser — drawn on top of exo cable, animates with δ.
    # Uses the exo motor's peak torque so δ_max maps cleanly to _MAX_FRAC.
    exo_spring_viz = ExoSpringVisualizerIsaac(
        _stage, exo_viz,
        k_exo=args.exo_ks, r_exo=args.exo_r,
        tau_max=_exo_motor.peak_torque_joint,
    )
    exo_spring_viz.create_prims()
    _CABLE_UPDATE_INTERVAL = 5

    # ── 3. First reset → Articulation ───────────────────────────────────────
    world.reset()
    manip.initialize_state()

    # ── 4. Dynamics view for M, C, g ───────────────────────────────────────
    manip.initialize_dynamics_view(world)

    L1, L2 = manip._get_link_lengths()
    r_p   = manip.r_p
    r_exo = manip.r_exo
    print(colored(f"  Link lengths: L1={L1*1e3:.1f} mm  L2={L2*1e3:.1f} mm", "cyan"))
    print(colored(f"  Pulley radii: r_p={r_p*1e3:.1f} mm  r_exo={r_exo*1e3:.1f} mm", "cyan"))

    # ── 5. Trajectory ───────────────────────────────────────────────────────
    main_traj = build_trajectory(args, L1=L1, L2=L2)
    print(colored(
        f"  ✓ Trajectory: {args.traj_shape}  N={args.traj_n}  "
        f"lap={args.duration:.1f} s",
        "green",
    ))

    # ── 6. Initial pose via IK ──────────────────────────────────────────────
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

    # ── 7. Move-to-start preamble ──────────────────────────────────────────
    q_pre = q_init + np.array([np.deg2rad(-5.0), np.deg2rad(5.0)])
    p_start = forward_kinematics_2r(L1, L2, q_pre[0], q_pre[1])
    p_end   = p_first
    v_end   = main_traj.eval_velocity(0.0)

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

    # ── 8. Initial state ───────────────────────────────────────────────────
    manip.set_positions_user_order(q_init)
    manip.set_velocities_user_order(np.zeros(2))

    # ── 9. CT controller + SEA drive + SEA exo ─────────────────────────────
    ct = ComputedTorqueController(
        Kp=args.ct_kp, Kd=args.ct_kd,
        tau_max=args.ct_tau_max, pulley_radius=r_p,
    )
    wn = ct.omega_n
    zeta = ct.zeta

    sea = SEACableActuatorNP(
        r_p=r_p,
        k_s=args.spring_stiffness,
        b_c=args.cable_damping,
        tau_max=args.ct_tau_max,
        dt=args.dt,
        motor_mode=_motor_mode,
        motor_cfg=_drive_motor,
        omega_m=args.motor_bandwidth,
        motor_substeps=args.motor_substeps,
    )
    sea.initialize(q_init[1])
    print(colored(
        f"  ✓ SEA drive: k_s={args.spring_stiffness} N/m  "
        f"b_c={args.cable_damping} N·s/m  substeps={sea.motor_substeps}",
        "green",
    ))

    exo = SEAExoActuatorNP(
        k_exo=args.exo_ks, b_exo=args.exo_bc, r_exo=args.exo_r,
        tau_max=_exo_motor.peak_torque_joint, dt=args.dt,
        motor_cfg=_exo_motor,
    )
    exo.initialize(q_init[1])
    print(colored(
        f"  ✓ SEA exo:   k_exo={args.exo_ks} N/m  "
        f"r_exo={args.exo_r:.5f} m  k_eff={exo.k_eff:.4f} Nm/rad",
        "green",
    ))

    # Exo command source (timed or reactive)
    if args.exo_reactive:
        exo_cmd = _ExoReactiveSource(
            delta_theta=args.exo_delta_theta,
            e_on=np.deg2rad(args.exo_e_on),
            e_off=np.deg2rad(args.exo_e_off),
            t_hold=args.exo_t_hold,
            dt=args.dt,
        )
    else:
        exo_cmd = _ExoTimedSource(
            t_activate=args.exo_activate_time,
            delta_theta=args.exo_delta_theta,
            never_activate=args.no_exo_activate,
        )

    # External torque disturbance (torque/sine modes only).  For vel/pos we
    # inject state impulses directly in the run loop.
    _ext_on = args.disturbance and args.disturbance_mode in ("torque", "sine")
    _t_ext0 = float(args.disturbance_time)
    _t_ext1 = _t_ext0 + float(args.disturbance_dur)
    _omega_ext = 2.0 * np.pi * float(args.disturbance_freq)

    def _ext_torque(t: float) -> float:
        if not _ext_on or t < _t_ext0 or t >= _t_ext1:
            return 0.0
        if args.disturbance_mode == "torque":
            return float(args.disturbance_tau)
        # sine
        return float(args.disturbance_tau) * float(np.sin(_omega_ext * (t - _t_ext0)))

    print(colored(
        f"\n▶  SEA + Exo Cable — Isaac Sim  ({_mode_label})"
        f"\n   CT:  Kp={args.ct_kp}  Kd={args.ct_kd}  →  ωn={wn:.1f} rad/s  ζ={zeta:.2f}"
        f"\n   Drive SEA: k_s={args.spring_stiffness} N/m  b_c={args.cable_damping} N·s/m  "
        f"ω_m={args.motor_bandwidth} rad/s"
        f"\n   Exo SEA:   k_exo={args.exo_ks} N/m  Δθ={args.exo_delta_theta:.3f} rad  "
        f"k_eff={2*args.exo_ks*args.exo_r**2:.4f} Nm/rad"
        f"\n   τ_max={args.ct_tau_max} Nm   dt={args.dt*1e3:.1f} ms"
        f"\n   Press Ctrl-C to stop and show plots.",
        "cyan",
    ))
    if _ext_on:
        _ext_label = (
            f"τ_ext = {args.disturbance_tau:+.2f} Nm (const)"
            if args.disturbance_mode == "torque"
            else f"τ_ext = {args.disturbance_tau:.2f}·sin(2π·{args.disturbance_freq:.1f}Hz·t) Nm"
        )
        print(colored(
            f"  💥 External-torque disturbance armed: {_ext_label}  "
            f"window=[{_t_ext0:.2f}, {_t_ext1:.2f}] s",
            "red",
        ))
    elif args.disturbance:
        _non_torque_label = (
            f"Δq̇₂ = +{args.disturbance_dqdot:.1f} °/s [vel impulse]"
            if args.disturbance_mode == "vel"
            else f"Δq₂ = +{args.disturbance_dq:.1f}° [pos jump]"
        )
        print(colored(
            f"  💥 Disturbance armed: {_non_torque_label}  at t={args.disturbance_time:.2f} s",
            "red",
        ))

    # ── 10. Log buffers ─────────────────────────────────────────────────────
    max_steps = int((args.duration * max(args.num_laps, 10) + move_duration) / args.dt) + 1000
    log_t         = np.zeros(max_steps)
    log_q         = np.zeros((max_steps, 2))
    log_q_dot     = np.zeros((max_steps, 2))
    log_q_des     = np.zeros((max_steps, 2))
    log_tau_des   = np.zeros((max_steps, 2))
    log_tau_app   = np.zeros((max_steps, 2))  # τ₁, τ₂_total
    log_tens      = np.zeros((max_steps, 2))  # T_green, T_red
    log_ee_ref    = np.zeros((max_steps, 2))
    log_ee_vel_ref = np.zeros((max_steps, 2))
    log_ee_acc_ref = np.zeros((max_steps, 2))
    # SEA drive diagnostics
    log_l_m       = np.zeros(max_steps)
    log_l_m_des   = np.zeros(max_steps)
    log_delta     = np.zeros(max_steps)
    log_F_cable   = np.zeros(max_steps)
    log_tau_motor = np.zeros(max_steps)
    # SEA exo diagnostics (10-field order matches Drake diagnostics port)
    log_exo_diag  = np.zeros((max_steps, 10))
    log_tau_exo   = np.zeros(max_steps)
    log_tau_ext   = np.zeros(max_steps)
    log_activated = np.zeros(max_steps)

    # ── 11. Main loop ──────────────────────────────────────────────────────
    step = 0
    t    = 0.0
    last_q_des = q_init.copy()
    lap_prev   = 0
    move_reported = move_duration <= 0.0
    exo_reported  = args.no_exo_activate or args.exo_reactive
    dist_applied  = not args.disturbance or args.disturbance_mode in ("torque", "sine")
    exo_was_active = False
    ik_fail_count = 0
    ik_fail_first_t = None

    stop_requested = False

    def _sigint_handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while not stop_requested and step < max_steps:
            # Current state
            q     = manip.get_positions_user_order()
            q_dot = manip.get_velocities_user_order()

            # EE trajectory references
            ee_pos_ref = traj_source.eval_position(t)
            ee_vel_ref = traj_source.eval_velocity(t)
            ee_acc_ref = traj_source.eval_acceleration(t)

            # IK + Jacobian → joint references
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
                ik_fail_count += 1
                if ik_fail_first_t is None:
                    ik_fail_first_t = t

            # Dynamics (M, h) from ArticulationView
            M = manip.get_mass_matrix()
            h = manip.get_bias_forces()

            # Computed-torque
            ct_out = ct.compute(q, q_dot, q_des, q_dot_ref, q_ddot_ref, M, h)
            tau_desired = ct_out.tau_clip

            # Saturation warning (only in verbose mode — otherwise very noisy)
            if args.verbose and np.any(np.abs(ct_out.tau_raw) > ct.tau_max):
                _raw_max = np.max(np.abs(ct_out.tau_raw))
                print(colored(
                    f"  ⚠ Torque saturation at t={t:.3f}s: "
                    f"|τ_raw|_max={_raw_max:.2f} Nm > {ct.tau_max:.1f}",
                    "yellow",
                ))

            # Drive SEA: τ₁ pass-through, τ₂ via spring
            tau_drive, sea_diag = sea.step(tau_desired, q, q_dot)

            # Exo command source
            if args.exo_reactive:
                activated, dtheta = exo_cmd.eval(t, q[1], q_des[1])
            else:
                activated, dtheta = exo_cmd.eval(t)

            # Exo SEA step (gives scalar torque on joint 2)
            tau_exo_val, exo_diag = exo.step(
                activated=activated, delta_theta=dtheta,
                q=q, q_dot=q_dot, q_des=q_des,
            )

            # External disturbance torque (torque/sine modes)
            tau_ext_val = _ext_torque(t)

            # Combined actuation:  τ₁_total = τ₁_drive; τ₂_total = τ₂_drive + τ_exo + τ_ext
            tau_total = np.array([
                tau_drive[0],
                tau_drive[1] + tau_exo_val + tau_ext_val,
            ])

            # Apply to plant
            manip.set_joint_torques(tau_total)

            # ── Logging ───────────────────────────────────────────────────
            if step < max_steps:
                log_t[step]          = t
                log_q[step]          = q
                log_q_dot[step]      = q_dot
                log_q_des[step]      = q_des
                log_tau_des[step]    = tau_desired
                log_tau_app[step]    = tau_total
                log_tens[step]       = [sea_diag.T_green, sea_diag.T_red]
                log_ee_ref[step]     = ee_pos_ref
                log_ee_vel_ref[step] = ee_vel_ref
                log_ee_acc_ref[step] = ee_acc_ref
                log_l_m[step]        = sea_diag.l_m
                log_l_m_des[step]    = sea_diag.l_m_des
                log_delta[step]      = sea_diag.delta
                log_F_cable[step]    = sea_diag.F_cable
                log_tau_motor[step]  = sea_diag.tau_motor
                log_exo_diag[step]   = exo_diag.as_array()
                log_tau_exo[step]    = tau_exo_val
                log_tau_ext[step]    = tau_ext_val
                log_activated[step]  = 1.0 if activated else 0.0

            # Cable viz (every N steps)
            if step % _CABLE_UPDATE_INTERVAL == 0:
                cable_viz.update(q[0], q[1])
                exo_viz.update(q[0], q[1])
                exo_spring_viz.update(
                    delta_R=exo_diag.delta_R,
                    delta_L=exo_diag.delta_L,
                )

            # Physics step
            world.step(render=(_render_mode != "headless"))

            # Reactive mode: log activation transitions
            if args.exo_reactive and activated != exo_was_active:
                if activated:
                    print(colored(
                        f"  ⚡ Exo ACTIVATED (reactive) at t={t:.2f} s  "
                        f"(Δθ={args.exo_delta_theta:.3f} rad, "
                        f"k_eff={exo.k_eff:.4f} Nm/rad)", "magenta"))
                else:
                    print(colored(
                        f"  💤 Exo DEACTIVATED (reactive) at t={t:.2f} s", "magenta"))
                exo_was_active = activated

            # Disturbance injection (vel/pos impulses)
            if not dist_applied and t >= args.disturbance_time:
                dist_applied = True
                if args.disturbance_mode == "vel":
                    dqdot = np.deg2rad(args.disturbance_dqdot)
                    v_before = manip.get_velocities_user_order().copy()
                    v_after  = v_before.copy()
                    v_after[1] += dqdot
                    manip.set_velocities_user_order(v_after)
                    print(colored(
                        f"  💥 DISTURBANCE at t={t:.2f} s  "
                        f"q̇₂: {np.rad2deg(v_before[1]):.1f} → "
                        f"{np.rad2deg(v_after[1]):.1f} °/s "
                        f"(Δq̇₂ = +{args.disturbance_dqdot:.1f}°/s) [vel impulse]",
                        "red"))
                elif args.disturbance_mode == "pos":
                    dq = np.deg2rad(args.disturbance_dq)
                    q_before = manip.get_positions_user_order().copy()
                    q_after  = q_before.copy()
                    q_after[1] += dq
                    manip.set_positions_user_order(q_after)
                    print(colored(
                        f"  💥 DISTURBANCE at t={t:.2f} s  "
                        f"q₂: {np.rad2deg(q_before[1]):.1f}° → "
                        f"{np.rad2deg(q_after[1]):.1f}° "
                        f"(Δq₂ = +{args.disturbance_dq:.1f}°) [pos jump]",
                        "red"))

            t    += args.dt
            step += 1

            # Progress reporting
            if not move_reported and t >= move_duration:
                move_reported = True
                print(colored(
                    f"  ✓ Move-to-start complete at t={t:.2f} s — tracking begins.",
                    "green"))
            if (not exo_reported) and t >= args.exo_activate_time:
                exo_reported = True
                print(colored(
                    f"  ⚡ Exo ACTIVATED at t={t:.2f} s  "
                    f"(Δθ={args.exo_delta_theta:.3f} rad, "
                    f"k_eff={exo.k_eff:.4f} Nm/rad)",
                    "magenta"))
            lap_now = int(max(0.0, t - move_duration) / args.duration)
            if lap_now > lap_prev:
                lap_prev = lap_now
                print(colored(f"  Lap {lap_now} complete  (t={t:.1f} s)", "cyan"))
                if args.num_laps > 0 and lap_now >= args.num_laps:
                    print(colored(
                        f"\n  ✓ {args.num_laps} lap(s) done — auto-stopping.",
                        "yellow"))
                    break

    except Exception as e:
        print(colored(f"\n  ✗ Simulation error at t={t:.3f} s: {e}", "red"))
        import traceback
        traceback.print_exc()

    signal.signal(signal.SIGINT, signal.default_int_handler)

    # Trim logs
    n = step
    log_t          = log_t[:n]
    log_q          = log_q[:n]
    log_q_dot      = log_q_dot[:n]
    log_q_des      = log_q_des[:n]
    log_tau_des    = log_tau_des[:n]
    log_tau_app    = log_tau_app[:n]
    log_tens       = log_tens[:n]
    log_ee_ref     = log_ee_ref[:n]
    log_ee_vel_ref = log_ee_vel_ref[:n]
    log_ee_acc_ref = log_ee_acc_ref[:n]
    log_l_m        = log_l_m[:n]
    log_l_m_des    = log_l_m_des[:n]
    log_delta      = log_delta[:n]
    log_F_cable    = log_F_cable[:n]
    log_tau_motor  = log_tau_motor[:n]
    log_exo_diag   = log_exo_diag[:n]
    log_tau_exo    = log_tau_exo[:n]
    log_tau_ext    = log_tau_ext[:n]
    log_activated  = log_activated[:n]

    laps_done = int(max(0.0, t - move_duration) / args.duration)
    print(colored(
        f"\n  Simulation stopped at t={t:.2f} s  "
        f"({laps_done} full laps, {n} steps).",
        "yellow"))
    if ik_fail_count > 0:
        print(colored(
            f"  ⚠ IK failed on {ik_fail_count}/{n} steps "
            f"({100.0 * ik_fail_count / max(n, 1):.1f} %)  "
            f"first at t={ik_fail_first_t:.2f}s. "
            f"→ q_des was frozen; this causes q2/δ/τ_exo oscillation. "
            f"Reduce --traj-radius or move --traj-cx closer to base "
            f"so every waypoint lies inside r∈[|L1-L2|, L1+L2] = "
            f"[{abs(L1-L2):.3f}, {L1+L2:.3f}] m.",
            "red"))

    data = dict(
        t=log_t, q=log_q, q_dot=log_q_dot, q_des=log_q_des,
        tau_des=log_tau_des, tau_app=log_tau_app,
        tens=log_tens, ee_ref=log_ee_ref,
        ee_vel_ref=log_ee_vel_ref, ee_acc_ref=log_ee_acc_ref,
        l_m=log_l_m, l_m_des=log_l_m_des, delta=log_delta,
        F_cable=log_F_cable, tau_motor=log_tau_motor,
        exo_diag=log_exo_diag, tau_exo=log_tau_exo,
        tau_ext=log_tau_ext, activated=log_activated,
        L1=L1, L2=L2, r_p=r_p, r_exo=r_exo,
        t_dist=(float(args.disturbance_time) if args.disturbance else None),
    )

    # Optional log dump for pydrake-vs-Isaac comparison
    if args.log_npz:
        out_path = Path(args.log_npz).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(out_path),
            **{k: v for k, v in data.items() if isinstance(v, np.ndarray)},
            t_dist=np.array([data['t_dist'] if data['t_dist'] is not None else -1.0]),
            L1=np.array([data['L1']]),
            L2=np.array([data['L2']]),
            r_p=np.array([data['r_p']]),
            r_exo=np.array([data['r_exo']]),
            ee_x_tgt=main_traj.ee_x_tgt, ee_y_tgt=main_traj.ee_y_tgt,
            args_json=np.array([repr(vars(args))]),
        )
        print(colored(f"  💾 Logs saved: {out_path}", "green"))

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_results(data, main_traj, ct, args)


# ============================================================================
# SCENE VIZ MODE
# ============================================================================

def run_scene_viz(args):
    """Load robot + cables (drive + exo), no control."""
    print("\n" + "=" * 80)
    print(colored("SCENE VIZ — Isaac Sim (SEA + Exo)", "cyan", attrs=["bold"]))
    print("=" * 80)

    manip = CupManipulatorTendonWithExoIsaac(MANIP_CONFIG)
    manip.prepare_usd()

    world = World(stage_units_in_meters=1.0,
                  physics_dt=args.dt, rendering_dt=args.dt)
    world.scene.add_default_ground_plane()
    manip.load_urdf(world)
    orientation = np.deg2rad([
        MANIP_CONFIG.tilt_roll_deg, MANIP_CONFIG.tilt_pitch_deg, 0.0,
    ])
    manip.weld_base_to_world(position=np.zeros(3), orientation=orientation)
    manip.add_end_effector_frame()
    manip.set_joint_properties()
    manip.add_joint_actuators()

    _stage = omni.usd.get_context().get_stage()
    cable_viz = CableVisualizerIsaac(_stage, URDF_PATH)
    cable_viz.create_prims(
        q1=MANIP_CONFIG.joint_configs['link1_base'].position,
        q2=MANIP_CONFIG.joint_configs['link2_link1'].position,
    )
    exo_viz = ExoCableVisualizerIsaac(_stage, URDF_PATH, springs_enabled=True)
    exo_viz.create_prims(
        q1=MANIP_CONFIG.joint_configs['link1_base'].position,
        q2=MANIP_CONFIG.joint_configs['link2_link1'].position,
    )
    # Scene viz: show helices at their rest pretension (δ = 0).
    exo_spring_viz = ExoSpringVisualizerIsaac(
        _stage, exo_viz,
        k_exo=args.exo_ks, r_exo=args.exo_r,
        tau_max=_exo_motor.peak_torque_joint,
    )
    exo_spring_viz.create_prims()

    world.reset()
    manip.initialize_state()
    manip.set_initial_positions()

    print(colored("\n▶  Scene viz — press Ctrl-C to exit.", "cyan"))

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
# PLOTTING — two figures mirroring the PyDrake exo script layout
# ============================================================================

def plot_results(data, main_traj, ct, args):
    """Generate Figure 1 (Manipulator & Drive SEA, 5×2) and Figure 2
    (Exosuit Co-Contraction, 4×2) — structural mirror of the PyDrake
    ``plot_results`` in ``script_..._with_exo_pydrake.py``.
    """
    t = data['t']
    if len(t) == 0:
        print(colored("  (no data to plot)", "yellow"))
        return

    q      = data['q']
    q_dot  = data['q_dot']
    q_des  = data['q_des']
    tau_des = data['tau_des']
    tau_app = data['tau_app']
    delta   = data['delta']
    F_cable = data['F_cable']
    T_green = data['tens'][:, 0]
    T_red   = data['tens'][:, 1]
    tau_motor = data['tau_motor']
    ee_ref  = data['ee_ref']
    exo_diag = data['exo_diag']
    tau_exo = data['tau_exo']
    L1, L2, r_p, r_exo = data['L1'], data['L2'], data['r_p'], data['r_exo']
    t_act = args.exo_activate_time
    t_dist = data.get('t_dist', None)

    # ── Derived signals ────────────────────────────────────────────────────
    q2     = q[:, 1]
    q2_dot = q_dot[:, 1]
    q2_des = q_des[:, 1]

    # Drive cable kinematics (joint-side)
    l_drive     = r_p * q2
    l_dot_drive = r_p * q2_dot
    # Exo cable kinematics
    l_exo_R     = r_exo * q2
    l_dot_exo_R = r_exo * q2_dot
    l_exo_L     = r_exo * (-q2)
    l_dot_exo_L = r_exo * (-q2_dot)

    exo_dR = exo_diag[:, 0]
    exo_dL = exo_diag[:, 1]
    exo_FR = exo_diag[:, 2]
    exo_FL = exo_diag[:, 3]
    exo_mR = exo_diag[:, 4]
    exo_mL = exo_diag[:, 5]

    # Actual EE via FK
    ee_x = np.array([forward_kinematics_2r(L1, L2, q[k, 0], q[k, 1])[0]
                     for k in range(len(t))])
    ee_y = np.array([forward_kinematics_2r(L1, L2, q[k, 0], q[k, 1])[1]
                     for k in range(len(t))])
    ee_err = np.sqrt((ee_x - ee_ref[:, 0])**2 + (ee_y - ee_ref[:, 1])**2)
    rms_ee = float(np.sqrt(np.mean(ee_err**2))) if ee_err.size else 0.0

    # Exo state label (matches PyDrake banner)
    if args.exo_reactive:
        exo_label = "EXO: REACTIVE"
        exo_tag   = "exo_reactive"
    elif not args.no_exo_activate:
        exo_label = f"EXO: ON  (t_act={args.exo_activate_time:.1f} s)"
        exo_tag   = "exo_on"
    else:
        exo_label = "EXO: OFF  (deactivated)"
        exo_tag   = "exo_off"

    suptitle = (
        f"SEA + Exo Co-Contraction (Isaac Sim)  —  "
        f"k_s={args.spring_stiffness}  k_exo={args.exo_ks}  "
        f"Δθ={args.exo_delta_theta:.3f}  "
        f"k_eff={2*args.exo_ks*args.exo_r**2:.4f} Nm/rad  |  {exo_label}"
    )

    def _style(ax, ylabel="", title="", xlabel=""):
        ax.axvline(t_act, color="m", ls=":", lw=1, alpha=0.6)
        if t_dist is not None and t_dist >= 0:
            ax.axvline(t_dist, color="r", ls="-", lw=1.5, alpha=0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.4)

    plots_dir = PROJECT_ROOT / 'plots'
    plots_dir.mkdir(exist_ok=True)
    stamp = _time.strftime('%Y%m%d_%H%M%S')

    # ═══════════════════════════════════════════════════════════════════════
    # Figure 1 — Manipulator & Drive SEA (5×2)
    # ═══════════════════════════════════════════════════════════════════════
    fig1, ax1 = plt.subplots(5, 2, figsize=(15, 18),
                             num="Fig 1: Manipulator & Drive SEA (Isaac)")
    fig1.suptitle(suptitle + "\n(Manipulator & Drive SEA)", fontsize=11)

    # Row 0: EE X / Y
    ax1[0][0].plot(t, ee_ref[:, 0], "r--", lw=1, label="ref X")
    ax1[0][0].plot(t, ee_x,         "b-", lw=1.2, label="actual X")
    _style(ax1[0][0], "[m]", "EE Position X")

    ax1[0][1].plot(t, ee_ref[:, 1], "r--", lw=1, label="ref Y")
    ax1[0][1].plot(t, ee_y,         "b-", lw=1.2, label="actual Y")
    _style(ax1[0][1], "[m]", "EE Position Y")

    # Row 1: q1 / q2
    ax1[1][0].plot(t, np.rad2deg(q[:, 0]),     "b-", lw=1.2, label="q₁ actual")
    ax1[1][0].plot(t, np.rad2deg(q_des[:, 0]), "r--", lw=1, label="q₁ desired")
    _style(ax1[1][0], "[deg]", "Joint 1 (Shoulder) — rigid direct drive")

    ax1[1][1].plot(t, np.rad2deg(q2),     "b-", lw=1.2, label="q₂ actual")
    ax1[1][1].plot(t, np.rad2deg(q2_des), "r--", lw=1, label="q₂ desired")
    _style(ax1[1][1], "[deg]", "Joint 2 (Elbow) — drive SEA + exo")

    # Row 2: torques
    ax1[2][0].plot(t, tau_app[:, 0], "g-", lw=1.2, label="τ₁ (CT direct)")
    ax1[2][0].plot(t, tau_app[:, 1], "b-", lw=1.2, label="τ₂ total")
    ax1[2][0].plot(t, tau_exo,       "m-", lw=1,   label="τ_exo")
    ax1[2][0].axhline(0, color="k", lw=0.5)
    _style(ax1[2][0], "[Nm]", "Joint Torques — τ₂ = r_p·F_cable + τ_exo")

    ax1[2][1].plot(t, tau_des[:, 1], "r--", lw=1, label="τ₂_des (CT)")
    ax1[2][1].plot(t, tau_app[:, 1], "b-", lw=1.2, label="τ₂_applied")
    ax1[2][1].plot(t, r_p * F_cable, "g-", lw=0.9, alpha=0.7,
                   label="r_p·F_cable (drive)")
    ax1[2][1].axhline(0, color="k", lw=0.5)
    _style(ax1[2][1], "[Nm]", "τ₂ Desired vs Applied")

    # Row 3: δ + cable force / tensions
    ax_3L = ax1[3][0]
    ax_3L.plot(t, delta * 1e3, "b-", lw=1.2, label="δ drive [mm]")
    ax_3L.axhline(0, color="k", lw=0.5)
    ax_3R = ax_3L.twinx()
    ax_3R.plot(t, F_cable, "r-", lw=1, alpha=0.7, label="F_cable [N]")
    ax_3R.set_ylabel("F [N]", color="r")
    ax_3R.tick_params(axis="y", labelcolor="r")
    ax_3R.legend(loc="upper right", fontsize=7)
    _style(ax_3L, "δ [mm]", "Drive SEA: δ = l_m − r_p·q₂,  F = k_s·δ + b_c·δ̇")

    ax1[3][1].plot(t, T_green, "g-", lw=1.2, label="T_green (retract)")
    ax1[3][1].plot(t, T_red,   "r-", lw=1.2, label="T_red (extend)")
    ax1[3][1].axhline(0, color="k", lw=0.5)
    _style(ax1[3][1], "[N]", "Cable Tensions")

    # Row 4: cable length+velocity / motor torque + EE error
    ax_4L = ax1[4][0]
    ax_4L.plot(t, l_drive * 1e3, "b-", lw=1.2, label="l = r_p·q₂ [mm]")
    ax_4Lr = ax_4L.twinx()
    ax_4Lr.plot(t, l_dot_drive * 1e3, "c-", lw=0.9, alpha=0.7,
                label="l̇ [mm/s]")
    ax_4Lr.set_ylabel("l̇ [mm/s]", color="c")
    ax_4Lr.tick_params(axis="y", labelcolor="c")
    ax_4Lr.legend(loc="upper right", fontsize=7)
    _style(ax_4L, "l [mm]", "Drive Cable — l = r_p·q₂", "Time [s]")

    ax_4R = ax1[4][1]
    ax_4R.plot(t, tau_motor, "g-", lw=1.2, label="τ_motor [Nm]")
    ax_4R.axhline(0, color="k", lw=0.5)
    ax_4R2 = ax_4R.twinx()
    ax_4R2.plot(t, ee_err * 1e3, "b-", lw=0.9, alpha=0.6,
                label="‖e_EE‖ [mm]")
    ax_4R2.axhline(rms_ee * 1e3, color="r", ls="--", lw=0.8, alpha=0.6,
                   label=f"RMS = {rms_ee*1e3:.2f} mm")
    ax_4R2.set_ylabel("EE err [mm]", color="b")
    ax_4R2.tick_params(axis="y", labelcolor="b")
    ax_4R2.legend(loc="upper right", fontsize=7)
    _style(ax_4R, "τ [Nm]", "Motor Torque & EE Error", "Time [s]")

    fig1.tight_layout(rect=[0, 0, 1, 0.95])

    # ═══════════════════════════════════════════════════════════════════════
    # Figure 2 — Exosuit Co-Contraction (4×2)
    # ═══════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(4, 2, figsize=(15, 14),
                             num="Fig 2: Exosuit Co-Contraction (Isaac)")
    fig2.suptitle(suptitle + "\n(Exosuit Co-Contraction Detail)", fontsize=11)

    # Row 0: exo cable lengths / velocities
    ax2[0][0].plot(t, l_exo_R * 1e3, "tab:orange", lw=1.2,
                   label="l_R = r_exo·q₂ [mm]")
    ax2[0][0].plot(t, l_exo_L * 1e3, "tab:purple", lw=1.2,
                   label="l_L = −r_exo·q₂ [mm]")
    ax2[0][0].axhline(0, color="k", lw=0.5)
    _style(ax2[0][0], "[mm]", "Exo Cable Lengths")

    ax2[0][1].plot(t, l_dot_exo_R * 1e3, "tab:orange", lw=1.2, label="l̇_R")
    ax2[0][1].plot(t, l_dot_exo_L * 1e3, "tab:purple", lw=1.2, label="l̇_L")
    ax2[0][1].axhline(0, color="k", lw=0.5)
    _style(ax2[0][1], "[mm/s]", "Exo Cable Velocities")

    # Row 1: exo δ + F / exo motor pos
    ax_e1L = ax2[1][0]
    ax_e1L.plot(t, exo_dR * 1e3, "tab:orange", lw=1.2, label="δ_R [mm]")
    ax_e1L.plot(t, exo_dL * 1e3, "tab:purple", lw=1.2, label="δ_L [mm]")
    ax_e1L.axhline(0, color="k", lw=0.5)
    ax_e1R = ax_e1L.twinx()
    ax_e1R.plot(t, exo_FR, "tab:orange", ls="--", lw=0.9, alpha=0.6,
                label="F_R [N]")
    ax_e1R.plot(t, exo_FL, "tab:purple", ls="--", lw=0.9, alpha=0.6,
                label="F_L [N]")
    ax_e1R.set_ylabel("F [N]")
    ax_e1R.legend(loc="lower right", fontsize=7)
    _style(ax_e1L, "δ [mm]",
           "Exo Spring  —  δ = l_m − r_exo·q₂,  F = k_exo·δ + b_exo·δ̇")
    ax_e1L.legend(loc="upper left", fontsize=7)

    ax2[1][1].plot(t, np.rad2deg(exo_mR), "tab:orange", lw=1.2,
                   label="θ_mR/N")
    ax2[1][1].plot(t, np.rad2deg(exo_mL), "tab:purple", lw=1.2,
                   label="θ_mL/N")
    ax2[1][1].plot(t, np.rad2deg(q2), "b-", lw=0.9, alpha=0.5,
                   label="q₂ actual")
    ax2[1][1].plot(t, np.rad2deg(q2_des), "r--", lw=0.8, alpha=0.5,
                   label="q₂ desired")
    _style(ax2[1][1], "[deg]", "Exo Motor Positions (joint-referred)")

    # Row 2: τ_exo / EE XY
    ax_e2L = ax2[2][0]
    ax_e2L.plot(t, tau_exo, "m-", lw=1.2, label="τ_exo [Nm]")
    ax_e2L.axhline(0, color="k", lw=0.5)
    ax_e2R = ax_e2L.twinx()
    ax_e2R.plot(t, exo_FR - exo_FL, "k-", lw=0.8, alpha=0.5,
                label="F_R − F_L [N]")
    ax_e2R.set_ylabel("ΔF [N]")
    ax_e2R.legend(loc="lower right", fontsize=7)
    _style(ax_e2L, "[Nm]", "Exo Torque  —  τ_exo = r_exo·(F_R − F_L)", "Time [s]")
    ax_e2L.legend(loc="upper left", fontsize=7)

    ax2[2][1].plot(ee_ref[:, 0], ee_ref[:, 1], "r--", lw=1, label="reference")
    ax2[2][1].plot(ee_x, ee_y, "b-", lw=1.2, label="actual")
    ax2[2][1].set_aspect("equal", adjustable="datalim")
    ax2[2][1].set_ylabel("[m]")
    ax2[2][1].set_xlabel("[m]")
    ax2[2][1].set_title("EE XY Path", fontsize=9)
    ax2[2][1].legend(fontsize=7)
    ax2[2][1].grid(True, alpha=0.4)

    # Row 3: τ₂ breakdown / EE tracking error
    ax2[3][0].plot(t, r_p * F_cable, "g-", lw=1.2, label="r_p·F_cable (drive)")
    ax2[3][0].plot(t, tau_exo,       "m-", lw=1.2, label="τ_exo")
    ax2[3][0].plot(t, tau_app[:, 1], "b--", lw=1, label="τ₂_total")
    ax2[3][0].plot(t, tau_des[:, 1], "r:", lw=1, alpha=0.6, label="τ₂_des")
    ax2[3][0].axhline(0, color="k", lw=0.5)
    _style(ax2[3][0], "[Nm]", "τ₂ Breakdown", "Time [s]")

    ax2[3][1].plot(t, ee_err * 1e3, "b-", lw=1.2, label="‖e_EE‖ [mm]")
    ax2[3][1].axhline(rms_ee * 1e3, color="r", ls="--", lw=1,
                      label=f"RMS = {rms_ee*1e3:.2f} mm")
    _style(ax2[3][1], "[mm]", "EE Tracking Error", "Time [s]")

    fig2.tight_layout(rect=[0, 0, 1, 0.94])

    base = (f"sea_exo_isaac_{args.traj_shape}_"
            f"kexo{int(args.exo_ks)}_dth{args.exo_delta_theta:.2f}_"
            f"{exo_tag}_{stamp}")
    f1 = plots_dir / f"{base}_manip.png"
    f2 = plots_dir / f"{base}_exo.png"
    fig1.savefig(str(f1), dpi=150, bbox_inches='tight')
    fig2.savefig(str(f2), dpi=150, bbox_inches='tight')
    print(colored(f"\n  📊 Fig 1 saved: {f1}", "green"))
    print(colored(f"  📊 Fig 2 saved: {f2}", "green"))

    print(colored(f"  Mean tracking RMS: {rms_ee*1e3:.2f} mm", "green"))
    print(colored(f"  Mean |δ| (drive): {np.mean(np.abs(delta))*1e3:.3f} mm", "green"))

    if not args.no_show:
        try:
            import subprocess as _sp
            _sp.Popen(["eog", str(f1)], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
            _sp.Popen(["eog", str(f2)], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        except Exception:
            pass


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    if args.mode == 'scene-viz':
        run_scene_viz(args)
    elif args.mode == 'sea-exo':
        run_sea_exo(args)
    else:
        print(colored(f"Unknown mode: {args.mode}", "red"))
        sys.exit(1)

    _log.close(simulation_app)


if __name__ == "__main__":
    main()
