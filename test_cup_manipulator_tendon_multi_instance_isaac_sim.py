"""
test_cup_manipulator_tendon_multi_instance_isaac_sim.py
──────────────────────────────────────────────────────────
N parallel instances of the cup manipulator tendon running independent
trajectories (rect, circle, line) simultaneously in Isaac Sim.

Each robot is placed on a square grid and runs a CT (computed-torque)
controller with SEA (Series Elastic Actuator) cable actuation on joint 2.
Joint 1 is rigid direct-drive; joint 2 passes through a spring-damper
cable model (k_s, b_c, ω_m) before reaching the plant.

Usage::

    conda activate env_isaacsim

    # 4 robots in a 2×2 grid, local window
    python test_cup_manipulator_tendon_multi_instance_isaac_sim.py

    # Soft spring (high lag)
    python test_cup_manipulator_tendon_multi_instance_isaac_sim.py \\
        --spring-stiffness 30

    # 9 robots, headless (benchmarking / CI)
    python test_cup_manipulator_tendon_multi_instance_isaac_sim.py \\
        --num-envs 9 --render headless

    # Stream to Mac via WebRTC / Tailscale
    python test_cup_manipulator_tendon_multi_instance_isaac_sim.py \\
        --num-envs 4 --render websocket

    # Custom duration and gains
    python test_cup_manipulator_tendon_multi_instance_isaac_sim.py \\
        --num-envs 6 --duration 30.0 --ct-kp 600 --ct-kd 35
"""

import os
import sys
import math
import argparse
import time

import numpy as np
from pathlib import Path

# ── Project root on sys.path ────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Pre-parse --render and --num-envs (cannot use argparse before SimulationApp)
_RENDER_CHOICES = ("native", "websocket", "headless")
_render_mode = "native"
_num_envs_pre = 4
for _i, _arg in enumerate(sys.argv):
    if _arg == "--render" and _i + 1 < len(sys.argv):
        _render_mode = sys.argv[_i + 1]
        if _render_mode not in _RENDER_CHOICES:
            print(f"[ERROR] --render must be one of {_RENDER_CHOICES}, got '{_render_mode}'")
            sys.exit(1)
    if _arg == "--num-envs" and _i + 1 < len(sys.argv):
        try:
            _num_envs_pre = int(sys.argv[_i + 1])
        except ValueError:
            pass

# # ── Quiet startup: suppress Isaac Sim extension / GPU / warning noise ────────
# from project_utils.log_isaacsim import IsaacSimLogger
# _log = IsaacSimLogger.from_argv()   # no-op when --verbose is in sys.argv
# _log.suppress()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── SimulationApp — MUST be first Isaac Sim import ───────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": _render_mode != "native",
    "width": 1280,
    "height": 720,
    "hide_ui": True,
})

# ── Isaac Sim / omni imports (safe after SimulationApp) ──────────────────────
from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView

# ── WebRTC streaming (--render websocket) ───────────────────────────────────
if _render_mode == "websocket":
    import subprocess
    from omni.kit.livestream.webrtc import set_setting, enable_extension
    try:
        _ip = subprocess.check_output(
            ["tailscale", "ip", "-4"], text=True
        ).strip()
    except Exception:
        _ip = "127.0.0.1"
    set_setting("/app/livestream/publicEndpointAddress", _ip)
    enable_extension("omni.kit.livestream.webrtc")
    print(f"  WebRTC (Tailscale) : connect to  {_ip} : 49100")

# ── Project imports ──────────────────────────────────────────────────────────
from termcolor import colored

from robots.cup_manipulator_tendon_isaac import (
    CupManipulatorTendonIsaac,
    create_cable_manipulator_config,
    solve_2r_ik,
    forward_kinematics_2r,
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
from actuators.sea_isaacsim import SEACableActuatorNP, SEADiagnostics

import matplotlib
matplotlib.use('Agg')  # non-interactive — works headless & within Isaac Sim
import matplotlib.pyplot as plt

# ── Argument parser ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Multi-instance cup manipulator tendon — Isaac Sim demo",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--num-envs", type=int, default=_num_envs_pre, metavar="N",
                    help="Number of parallel robot instances (default: 4)")
parser.add_argument("--spacing", type=float, default=1.5, metavar="M",
                    help="Grid spacing [m] between robots (default: 1.5)")
parser.add_argument("--render", choices=_RENDER_CHOICES, default=_render_mode,
                    help="Render mode: native | websocket | headless")
parser.add_argument("--verbose", action="store_true", default=False,
                    help="Show full Isaac Sim startup log")
parser.add_argument("--duration", type=float, default=200.0,
                    help="Simulation duration [s] (default: 20.0)")
parser.add_argument("--dt", type=float, default=1.0 / 100.0,
                    help="Physics timestep [s] (default: 0.01)")
parser.add_argument("--move-duration", type=float, default=3.0,
                    help="Move-to-start preamble duration [s] (default: 3.0)")
parser.add_argument("--ct-kp", type=float, default=800.0,
                    help="CT position gain Kp [s^-2] (default: 800)")
parser.add_argument("--ct-kd", type=float, default=57.0,
                    help="CT velocity gain Kd [s^-1] (default: 57, critically damped for Kp=800)")
parser.add_argument("--ct-tau-max", type=float, default=50.0,
                    help="Torque saturation [Nm] (default: 50)")
# SEA parameters
parser.add_argument("--spring-stiffness", type=float, default=5000.0, metavar="K_S",
                    help="SEA cable spring stiffness k_s [N/m] (default: 5000)")
parser.add_argument("--cable-damping", type=float, default=5.0, metavar="B_C",
                    help="SEA cable dashpot damping b_c [N·s/m] (default: 5)")
parser.add_argument("--motor-bandwidth", type=float, default=500.0, metavar="W_M",
                    help="SEA motor servo bandwidth ω_m [rad/s] (default: 500)")
parser.add_argument("--no-sea", action="store_true", default=False,
                    help="Bypass SEA — apply CT torques directly (for diagnostics)")
args = parser.parse_args()

N = args.num_envs
GRID_COLS = math.ceil(math.sqrt(N))

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
URDF_PATH = str(
    PROJECT_ROOT
    / "model_using_onshape_to_robot"
    / "manipulator_cable"
    / "manipulator_cable_obj.urdf"
)

# ── Shared configuration (same robot model for all instances) ────────────────
_BASE_CONFIG = create_cable_manipulator_config(
    urdf_path=URDF_PATH,
    joint_angles={
        "link1_base":   math.radians(10.0),
        "link2_link1":  math.radians(-10.0),
    },
    damping=(0.05, 0.05),
    stiffness=(0.5, 0.5),
)

# ── Per-environment trajectory definitions (cycled for N > len) ──────────────
# Each entry: type + shape-specific parameters.
_TRAJ_SPECS = [
    # 0 — Rectangle (wide, slow corners)
    {"type": "rect",   "lap": 8.0,
     "x_range": (0.36, 0.49), "y_range": (-0.10, 0.10)},
    # 1 — Circle (medium)
    {"type": "circle", "lap": 8.0, "cx": 0.42, "cy": 0.00, "radius": 0.09},
    # 2 — Horizontal line (back-and-forth)
    {"type": "line",   "lap": 5.0, "cx": 0.44, "cy": 0.00, "radius": 0.08},
    # 3 — Rectangle (taller, faster)
    {"type": "rect",   "lap": 6.0,
     "x_range": (0.40, 0.50), "y_range": (-0.12, 0.12)},
    # 4 — Small circle (inner workspace)
    {"type": "circle", "lap": 6.0, "cx": 0.40, "cy": 0.04, "radius": 0.05},
    # 5 — Diagonal line
    {"type": "line",   "lap": 4.0, "cx": 0.42, "cy": 0.03, "radius": 0.07},
    # 6 — Large circle
    {"type": "circle", "lap": 12.0, "cx": 0.43, "cy": 0.00, "radius": 0.10},
    # 7 — Compact rectangle
    {"type": "rect",   "lap": 10.0,
     "x_range": (0.41, 0.49), "y_range": (-0.06, 0.06)},
    # 8 — Offset circle
    {"type": "circle", "lap": 9.0, "cx": 0.45, "cy": -0.04, "radius": 0.08},
]


def _make_trajectory(spec: dict):
    """Build a trajectory object from a spec dict."""
    traj_type = spec["type"]
    lap = spec["lap"]
    if traj_type == "rect":
        return RectTrajectory(
            x_range=tuple(spec["x_range"]),
            y_range=tuple(spec["y_range"]),
            lap_duration=lap,
            N=40,
            v_max=0.10,
            v_corner=0.03,
            corner_blend=0.35,
        )
    if traj_type == "circle":
        return CircleTrajectory(
            cx=spec["cx"], cy=spec["cy"],
            radius=spec["radius"],
            lap_duration=lap,
            N=60,
        )
    if traj_type == "line":
        return LineTrajectory(
            cx=spec["cx"], cy=spec["cy"],
            radius=spec["radius"],
            lap_duration=lap,
            N=30,
        )
    raise ValueError(f"Unknown trajectory type: {traj_type!r}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — URDF → USD conversion  (once, before World creation)
# ═══════════════════════════════════════════════════════════════════════════════
print(colored(
    f"\n[multi-instance] Preparing {N} robot environment(s) …",
    "yellow", attrs=["bold"],
))

# Convert URDF → USD using robot 0; share the resulting file with all others.
_proto = CupManipulatorTendonIsaac(_BASE_CONFIG, enable_visualization=False)
_proto.prepare_usd()                          # URDF → USD on disk
_SHARED_USD = _proto._usd_path
print(colored(f"  USD cache : {_SHARED_USD}", "cyan"))

# Build N robot wrappers — each gets a unique stage prim path.
manips: list[CupManipulatorTendonIsaac] = []
for i in range(N):
    m = CupManipulatorTendonIsaac(_BASE_CONFIG, enable_visualization=False)
    m.prim_path  = f"/World/robot_{i}"
    m._usd_path  = _SHARED_USD   # skip re-conversion
    manips.append(m)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Create World and load all robots
# ═══════════════════════════════════════════════════════════════════════════════
world = World(
    stage_units_in_meters=1.0,
    physics_dt=args.dt,
    rendering_dt=args.dt,
)
world.scene.add_default_ground_plane()

# Robot base orientation: upright (no tilt — matches single-robot CT script)
_ORIENT = np.deg2rad([0.0, 0.0, 0.0])

for i, m in enumerate(manips):
    row, col = divmod(i, GRID_COLS)
    pos = np.array([col * args.spacing, row * args.spacing, 0.0])
    m.load_urdf(world)
    m.weld_base_to_world(position=pos, orientation=_ORIENT)
    m.set_joint_properties()
    m.add_joint_actuators()   # stiffness=0, maxForce=1e4

print(colored(
    f"\n  Grid : {GRID_COLS} × {math.ceil(N / GRID_COLS)}, "
    f"spacing = {args.spacing} m",
    "cyan",
))

# ── First physics reset + articulation initialization ───────────────────────
world.reset()
for m in manips:
    m.initialize_state()

# ── Dynamics views (batched: add all views → single reset → finalize all) ───
for m in manips:
    m.initialize_dynamics_view(world, reset=False)
world.reset()
for m in manips:
    m.initialize_state()          # re-create Articulation handles after reset
    m.finalize_dynamics_view(world)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Link lengths, trajectories, controllers, preambles
# ═══════════════════════════════════════════════════════════════════════════════
L1, L2 = manips[0]._get_link_lengths()
print(colored(
    f"\n  Link lengths : L1 = {L1 * 1e3:.1f} mm   L2 = {L2 * 1e3:.1f} mm",
    "cyan",
))

# Trajectories — cycled if N > len(_TRAJ_SPECS)
main_trajs = [_make_trajectory(_TRAJ_SPECS[i % len(_TRAJ_SPECS)]) for i in range(N)]

print(colored("\n  Trajectory assignments:", "cyan"))
for i in range(N):
    spec = _TRAJ_SPECS[i % len(_TRAJ_SPECS)]
    print(colored(f"    robot_{i}: {spec['type']}  (lap={spec['lap']} s)", "cyan"))

# CT controllers — same gains for all, independent state
r_p = manips[0].r_p
ct_controllers = [
    ComputedTorqueController(Kp=args.ct_kp, Kd=args.ct_kd, tau_max=args.ct_tau_max,
                             pulley_radius=r_p)
    for _ in range(N)
]

# SEA actuators — one per robot, independent motor state
sea_actuators = []
for i in range(N):
    sea = SEACableActuatorNP(
        r_p=r_p,
        k_s=args.spring_stiffness,
        b_c=args.cable_damping,
        omega_m=args.motor_bandwidth,
        tau_max=args.ct_tau_max,
        dt=args.dt,
    )
    q_cur = manips[i].get_positions_user_order()
    sea.initialize(q_cur[1])
    sea_actuators.append(sea)

print(colored(
    f"  SEA: k_s={args.spring_stiffness} N/m  "
    f"b_c={args.cable_damping} N·s/m  "
    f"ω_m={args.motor_bandwidth} rad/s",
    "cyan",
))

# Move-to-start preambles: cubic Hermite from current pose to first waypoint
q_seeds: list[np.ndarray] = []
traj_sources: list[PreambleTrajectorySource] = []

for i, (m, main_traj) in enumerate(zip(manips, main_trajs)):
    q_cur = m.get_positions_user_order()           # [q1, q2] rad
    ee_cur = forward_kinematics_2r(L1, L2, *q_cur) # current EE [x, y]

    first_ee  = main_traj.eval_position(0.0)
    first_vel = main_traj.eval_velocity(0.0)

    preamble = build_move_to_start(
        p_start=ee_cur,
        p_end=first_ee,
        v_end=first_vel,
        duration=args.move_duration,
    )
    traj_sources.append(PreambleTrajectorySource(preamble, main_traj))

    # IK seed: solve at first waypoint so the first loop step starts warm
    q_wp, ok = solve_2r_ik(L1, L2, first_ee, q_cur)
    q_seeds.append(q_wp if ok else q_cur.copy())

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Main simulation loop
# ═══════════════════════════════════════════════════════════════════════════════
max_steps = int(args.duration / args.dt)
t = 0.0

# ── Per-robot EE logging arrays ──────────────────────────────────────────────
log_ee_actual = [np.zeros((max_steps, 2)) for _ in range(N)]  # actual EE [x,y]
log_ee_ref    = [np.zeros((max_steps, 2)) for _ in range(N)]  # reference EE [x,y]
log_t         = np.zeros(max_steps)
# SEA-specific logging (per robot)
log_tau_des   = [np.zeros((max_steps, 2)) for _ in range(N)]  # CT desired
log_tau_sea   = [np.zeros((max_steps, 2)) for _ in range(N)]  # SEA applied
log_delta     = [np.zeros(max_steps) for _ in range(N)]        # spring extension

print(colored(
    f"\n[multi-instance] Running {N} robot(s) for "
    f"{args.duration:.0f} s  ({max_steps} steps) …",
    "yellow", attrs=["bold"],
))
print(colored("  Press Ctrl+C to stop early and show plots.\n", "cyan"))

# ── SIGINT handler: set flag so plotting code always runs after Ctrl+C ───────
import signal
_stop_requested = False
_orig_sigint = signal.getsignal(signal.SIGINT)

def _sigint_handler(sig, frame):
    global _stop_requested
    _stop_requested = True
    print(colored("\n[multi-instance] Ctrl+C — finishing step and plotting…", "yellow"))

signal.signal(signal.SIGINT, _sigint_handler)

_t_wall_start = time.time()
step = 0

while step < max_steps and not _stop_requested:
        # ── Per-robot CT control ─────────────────────────────────────────────
        for i in range(N):
            m    = manips[i]
            ctrl = ct_controllers[i]
            src  = traj_sources[i]

            # 1. Read state
            q     = m.get_positions_user_order()
            q_dot = m.get_velocities_user_order()

            # 2. EE reference from preamble / main trajectory
            ee_ref = src.eval_position(t)
            ee_vel = src.eval_velocity(t)
            ee_acc = src.eval_acceleration(t)

            # 3. IK + Jacobian → joint-space references
            q_des, q_dot_ref, q_ddot_ref, ik_ok = ik_to_joint_space_references(
                ee_ref, ee_vel, ee_acc,
                L1, L2,
                q_seeds[i],
                solve_2r_ik,
            )
            if ik_ok:
                q_seeds[i] = q_des.copy()   # warm-start for next step

            # 4. Dynamics (PhysX GPU tensors)
            M = m.get_mass_matrix()     # (2, 2)
            h = m.get_bias_forces()     # (2,)

            # 5. Computed-torque
            ct_out = ctrl.compute(q, q_dot, q_des, q_dot_ref, q_ddot_ref, M, h)

            # 6. SEA actuator: spring-mediated torque for joint 2
            tau_desired = ct_out.tau_raw
            if args.no_sea:
                tau_applied = ct_out.tau_clip
                sea_diag = SEADiagnostics(
                    l_m=0, l_m_des=0, delta=0, F_cable=0,
                    tau1_des=float(tau_desired[0]),
                    tau2_des=float(tau_desired[1]),
                    T_green=0, T_red=0,
                    tau_sea=float(ct_out.tau_clip[1]),
                )
            else:
                tau_applied, sea_diag = sea_actuators[i].step(tau_desired, q, q_dot)

            # 7. Apply torques
            m.set_joint_torques(tau_applied)

            # ── Diagnostic (first 3 steps, robot 0 only) ────────────────
            if step < 3 and i == 0:
                ee_fk = forward_kinematics_2r(L1, L2, q[0], q[1])
                ee_fk_des = forward_kinematics_2r(L1, L2, q_des[0], q_des[1])
                print(f"  [diag] step={step} t={t:.3f}")
                print(f"    q      = [{np.rad2deg(q[0]):+7.2f}, {np.rad2deg(q[1]):+7.2f}] deg")
                print(f"    q_des  = [{np.rad2deg(q_des[0]):+7.2f}, {np.rad2deg(q_des[1]):+7.2f}] deg")
                print(f"    q_err  = [{np.rad2deg(q_des[0]-q[0]):+7.3f}, {np.rad2deg(q_des[1]-q[1]):+7.3f}] deg")
                print(f"    ee_ref = [{ee_ref[0]*1e3:7.1f}, {ee_ref[1]*1e3:7.1f}] mm")
                print(f"    ee_fk  = [{ee_fk[0]*1e3:7.1f}, {ee_fk[1]*1e3:7.1f}] mm")
                print(f"    FK@des = [{ee_fk_des[0]*1e3:7.1f}, {ee_fk_des[1]*1e3:7.1f}] mm")
                print(f"    M diag = [{M[0,0]:.4f}, {M[1,1]:.4f}]")
                print(f"    h      = [{h[0]:+.4f}, {h[1]:+.4f}]")
                print(f"    tau_des= [{tau_desired[0]:+.3f}, {tau_desired[1]:+.3f}] Nm")
                print(f"    tau_app= [{tau_applied[0]:+.3f}, {tau_applied[1]:+.3f}] Nm")
                print(f"    ik_ok  = {ik_ok}  sea={'OFF' if args.no_sea else 'ON'}")

            # 8. Log EE positions + SEA data
            if step < max_steps:
                ee_actual = forward_kinematics_2r(L1, L2, q[0], q[1])
                log_ee_actual[i][step] = ee_actual
                log_ee_ref[i][step]    = ee_ref
                log_tau_des[i][step]   = tau_desired
                log_tau_sea[i][step]   = tau_applied
                log_delta[i][step]     = sea_diag.delta

        if step < max_steps:
            log_t[step] = t

        # ── Physics step ─────────────────────────────────────────────────────
        # render=True for native window and websocket stream; False only for headless
        world.step(render=(_render_mode != "headless"))

        t    += args.dt
        step += 1

        # ── Progress print every 500 steps (~5 s at 100 Hz) ─────────────────
        if step % 500 == 0:
            elapsed = time.time() - _t_wall_start
            rtf     = (step * args.dt) / elapsed
            print(
                f"  step={step:5d}  t={t:.1f} s  "
                f"RTF={rtf:.2f}×  ({N} envs)",
                flush=True,
            )

# Restore original SIGINT handler
signal.signal(signal.SIGINT, _orig_sigint)

# Trim logs to actual step count
n_logged = min(step, max_steps)
log_t = log_t[:n_logged]
for i in range(N):
    log_ee_actual[i] = log_ee_actual[i][:n_logged]
    log_ee_ref[i]    = log_ee_ref[i][:n_logged]
    log_tau_des[i]   = log_tau_des[i][:n_logged]
    log_tau_sea[i]   = log_tau_sea[i][:n_logged]
    log_delta[i]     = log_delta[i][:n_logged]

# ═══════════════════════════════════════════════════════════════════════════════
# Wrap-up
# ═══════════════════════════════════════════════════════════════════════════════
elapsed_total = time.time() - _t_wall_start
rtf_final     = (step * args.dt) / elapsed_total if elapsed_total > 0 else 0.0

print(colored(
    f"\n[multi-instance] Done — {N} envs × {step} steps "
    f"= {step * args.dt:.1f} s sim  |  {elapsed_total:.1f} s wall  "
    f"|  RTF {rtf_final:.2f}×",
    "green", attrs=["bold"],
))

# ═══════════════════════════════════════════════════════════════════════════════
# Plot — desired vs actual EE trajectory for every robot
# ═══════════════════════════════════════════════════════════════════════════════
if n_logged > 1:
    _COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
               'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'Multi-Instance SEA CT  —  {N} robots × {n_logged} steps  '
        f'(Kp={args.ct_kp}, Kd={args.ct_kd}, '
        f'k_s={args.spring_stiffness}, ω_m={args.motor_bandwidth})',
        fontsize=13, fontweight='bold',
    )

    # ── Top-left: XY path ────────────────────────────────────────────────────
    ax = axes[0, 0]
    for i in range(N):
        c = _COLORS[i % len(_COLORS)]
        spec = _TRAJ_SPECS[i % len(_TRAJ_SPECS)]
        lbl = f'robot_{i} ({spec["type"]})'
        ax.plot(log_ee_ref[i][:, 0] * 1e3, log_ee_ref[i][:, 1] * 1e3,
                '--', color=c, lw=1.0, alpha=0.5)
        ax.plot(log_ee_actual[i][:, 0] * 1e3, log_ee_actual[i][:, 1] * 1e3,
                '-', color=c, lw=1.5, label=lbl)
        ax.plot(log_ee_actual[i][0, 0] * 1e3, log_ee_actual[i][0, 1] * 1e3,
                'o', color=c, ms=6)
    ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')
    ax.set_title('EE Path  (solid=actual, dashed=ref)')
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.4)

    # ── Top-right: tracking error vs time ────────────────────────────────────
    ax = axes[0, 1]
    for i in range(N):
        c = _COLORS[i % len(_COLORS)]
        spec = _TRAJ_SPECS[i % len(_TRAJ_SPECS)]
        err = np.linalg.norm(log_ee_actual[i] - log_ee_ref[i], axis=1) * 1e3
        ax.plot(log_t, err, '-', color=c, lw=1.2,
                label=f'robot_{i} ({spec["type"]})')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('EE error [mm]')
    ax.set_title('Tracking Error')
    ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.4)

    # ── Bottom-left: τ₂ desired vs SEA-applied (per robot) ──────────────────
    ax = axes[1, 0]
    for i in range(N):
        c = _COLORS[i % len(_COLORS)]
        ax.plot(log_t, log_tau_des[i][:, 1], '-', color=c, lw=1.0, alpha=0.5)
        ax.plot(log_t, log_tau_sea[i][:, 1], '--', color=c, lw=1.2,
                label=f'robot_{i}')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('τ₂ [Nm]')
    ax.set_title('Joint-2 Torque  (solid=CT desired, dashed=SEA applied)')
    ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.4)

    # ── Bottom-right: spring extension δ (per robot) ────────────────────────
    ax = axes[1, 1]
    for i in range(N):
        c = _COLORS[i % len(_COLORS)]
        ax.plot(log_t, log_delta[i] * 1e3, '-', color=c, lw=1.2,
                label=f'robot_{i}')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('δ [mm]')
    ax.set_title('SEA Spring Extension δ')
    ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.4)

    fig.tight_layout()

    # Save (Agg backend — no display needed)
    _plots_dir = Path(_PROJECT_ROOT) / 'plots'
    _plots_dir.mkdir(exist_ok=True)
    _stamp = time.strftime('%Y%m%d_%H%M%S')
    _fname = _plots_dir / f'multi_instance_sea_{N}env_{_stamp}.png'
    fig.savefig(str(_fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(colored(f"\n  📊 Overview figure saved: {_fname}", "green"))

    # ── Per-robot figure: XY path + tracking error (one row per robot) ──────
    _n_rows = N if N > 1 else 1
    fig2, axes2 = plt.subplots(_n_rows, 2, figsize=(13, 4 * _n_rows),
                               squeeze=False)
    fig2.suptitle(
        f'Per-Robot EE Path & Tracking Error  —  {N} robot(s)  '
        f'(Kp={args.ct_kp}, Kd={args.ct_kd}, '
        f'k_s={args.spring_stiffness}, ω_m={args.motor_bandwidth}'
        f'{", NO-SEA" if args.no_sea else ""})',
        fontsize=13, fontweight='bold',
    )

    _preamble_steps = int(args.move_duration / args.dt)

    for i in range(N):
        c    = _COLORS[i % len(_COLORS)]
        spec = _TRAJ_SPECS[i % len(_TRAJ_SPECS)]

        # ── Left column: XY path ─────────────────────────────────────────
        ax_xy = axes2[i, 0]
        ax_xy.plot(
            log_ee_ref[i][:, 0] * 1e3, log_ee_ref[i][:, 1] * 1e3,
            '--', color='gray', lw=1.0, alpha=0.7, label='ref',
        )
        ax_xy.plot(
            log_ee_actual[i][:, 0] * 1e3, log_ee_actual[i][:, 1] * 1e3,
            '-', color=c, lw=1.5, label='actual',
        )
        ax_xy.plot(
            log_ee_actual[i][0, 0] * 1e3, log_ee_actual[i][0, 1] * 1e3,
            'o', color=c, ms=6,
        )
        ax_xy.set_xlabel('X [mm]')
        ax_xy.set_ylabel('Y [mm]')
        ax_xy.set_title(f'robot_{i} ({spec["type"]}) — EE Path')
        ax_xy.set_aspect('equal')
        ax_xy.legend(fontsize=8, loc='best')
        ax_xy.grid(True, alpha=0.4)

        # ── Right column: tracking error vs time ──────────────────────────
        ax_err = axes2[i, 1]
        err    = np.linalg.norm(log_ee_actual[i] - log_ee_ref[i], axis=1) * 1e3
        ax_err.plot(log_t, err, '-', color=c, lw=1.2)
        ax_err.axvline(
            args.move_duration, color='k', lw=0.8, ls='--',
            label='preamble end',
        )
        if n_logged > _preamble_steps:
            _mean_err = err[_preamble_steps:].mean()
            _max_err  = err[_preamble_steps:].max()
            ax_err.axhline(
                _mean_err, color=c, lw=0.9, ls=':',
                label=f'mean={_mean_err:.1f} mm  max={_max_err:.1f} mm',
            )
        ax_err.set_xlabel('Time [s]')
        ax_err.set_ylabel('EE error [mm]')
        ax_err.set_title(f'robot_{i} ({spec["type"]}) — Tracking Error')
        ax_err.legend(fontsize=8, loc='best')
        ax_err.grid(True, alpha=0.4)

    fig2.tight_layout()
    _fname2 = _plots_dir / f'multi_instance_perrobot_{N}env_{_stamp}.png'
    fig2.savefig(str(_fname2), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(colored(f"  📊 Per-robot figure saved: {_fname2}", "green"))
else:
    print(colored("  ⚠ No steps logged — skipping plot.", "yellow"))

simulation_app.close()

# ── Open the saved PNG with the system image viewer ──────────────────────────
if n_logged > 1:
    import subprocess
    for _fig_path in (_fname, _fname2):
        try:
            subprocess.Popen(['eog', str(_fig_path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(colored(f"  🖼  Opened: {_fig_path}", "green"))
        except Exception:
            pass  # eog not available — PNG is on disk
