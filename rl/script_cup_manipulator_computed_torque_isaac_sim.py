#!/usr/bin/env python3
"""
rl/script_cup_manipulator_computed_torque_isaac_sim.py
------------------------------------------------------
Pure Isaac Sim computed-torque controller for the cable manipulator.

NO PYDRAKE DEPENDENCY — uses Isaac Sim's ArticulationView for dynamics
(mass matrix, Coriolis, gravity) and pure NumPy for the controller.

This is the GPU-ready foundation for RL training with computed-torque
as the low-level controller.

Architecture
~~~~~~~~~~~~
    Trajectory → IK → CT Controller → Isaac Sim PhysX Articulation
                          ▲ state ─────────────────────┘

Usage
~~~~~
    # Default: computed-torque tracking a rectangle
    python rl/script_cup_manipulator_computed_torque_isaac_sim.py

    # Custom gains and trajectory
    python rl/script_cup_manipulator_computed_torque_isaac_sim.py \\
        --ct-kp 10000 --ct-kd 400 --duration 15 \\
        --traj-shape circle --traj-cx 0.4 --traj-cy 0.0 --traj-radius 0.08

    # Headless (no rendering — for benchmarking or RL)
    python rl/script_cup_manipulator_computed_torque_isaac_sim.py --render headless

Modes
~~~~~
    computed-torque  (default)  Full CT controller with EE trajectory tracking
    scene-viz                   Load robot, no control — just visualise
"""

# ============================================================================
# PRE-PARSE render flag BEFORE any Isaac Sim import
# ============================================================================
import os
import sys

# Suppress verbose Isaac Sim startup
os.environ.setdefault("CARB_LOG_LEVEL", "error")

import argparse

# Quick pre-parse for --render flag (needed before SimulationApp)
_render_mode = "native"
for _i, _arg in enumerate(sys.argv):
    if _arg == "--render" and _i + 1 < len(sys.argv):
        _render_mode = sys.argv[_i + 1]

# ============================================================================
# ISAAC SIM — must be first import
# ============================================================================
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": _render_mode != "native",
    "width": 1280,
    "height": 720,
    "hide_ui": True,
})

# ============================================================================
# IMPORTS (safe after SimulationApp)
# ============================================================================
import numpy as np
import signal
import time as _time
from pathlib import Path
from termcolor import colored

import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    pass
import matplotlib.pyplot as plt

# Isaac Sim
from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView

# Project imports (no Drake)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robots.cup_manipulator_tendon_isaac import (
    CupManipulatorTendonIsaac,
    create_cable_manipulator_config,
)
from rl.computed_torque_controller import (
    ComputedTorqueController,
    ik_to_joint_space_references,
)
from rl.trajectory import (
    RectTrajectory,
    CircleTrajectory,
    LineTrajectory,
    PreambleTrajectorySource,
    build_move_to_start,
)

# Analytical IK from the Isaac wrapper module
from robots.cup_manipulator_tendon_isaac import CupManipulatorTendonIsaac

# Cable (tendon) visualization — uses DrakeCablePlant from cable.py
from rl.cable_viz import CableVisualizerIsaac

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

parser = argparse.ArgumentParser(
    description='Cup manipulator computed-torque — Isaac Sim (no Drake)',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument('--mode', type=str, default='computed-torque',
                    choices=['computed-torque', 'scene-viz'],
                    help='Simulation mode (default: computed-torque)')
parser.add_argument('--render', type=str, default='native',
                    choices=['native', 'headless'],
                    help='Render mode (default: native)')
parser.add_argument('--duration', type=float, default=10.0,
                    help='Lap duration [s] (default: 10.0)')
parser.add_argument('--dt', type=float, default=1.0 / 100.0,
                    help='Physics timestep [s] (default: 0.002)')

# Tilt
parser.add_argument('--tilt-roll', type=float, default=0.0,
                    help='Base roll tilt [deg] (default: 0)')
parser.add_argument('--tilt-pitch', type=float, default=0.0,
                    help='Base pitch tilt [deg] (default: 0)')

# Joint properties
parser.add_argument('--joint-damping', type=float, nargs=2, default=[0.05, 0.05],
                    metavar=('D1', 'D2'),
                    help='Joint damping [Nm·s/rad] (default: 0.05 0.05)')
parser.add_argument('--joint-stiffness', type=float, nargs=2, default=[0.5, 0.5],
                    metavar=('K1', 'K2'),
                    help='Passive joint stiffness [Nm/rad] (default: 0.5 0.5)')

# CT gains
parser.add_argument('--ct-kp', type=float, default=800.0,
                    help='CT position gain Kp [s⁻²] (default: 800)')
parser.add_argument('--ct-kd', type=float, default=40.0,
                    help='CT velocity gain Kd [s⁻¹] (default: 40)')
parser.add_argument('--ct-tau-max', type=float, default=50.0,
                    help='Torque saturation [Nm] (default: 50.0)')

# Trajectory
parser.add_argument('--traj-shape', type=str, default='rect',
                    choices=['rect', 'circle', 'line'],
                    help='Trajectory shape (default: rect)')
parser.add_argument('--traj-n', type=int, default=60,
                    help='Waypoints (default: 60)')
parser.add_argument('--traj-x-range', type=float, nargs=2, default=[0.49, 0.51],
                    metavar=('XMIN', 'XMAX'))
parser.add_argument('--traj-y-range', type=float, nargs=2, default=[-0.08, 0.08],
                    metavar=('YMIN', 'YMAX'))
parser.add_argument('--traj-cx', type=float, default=0.4)
parser.add_argument('--traj-cy', type=float, default=0.0)
parser.add_argument('--traj-radius', type=float, default=0.1)
parser.add_argument('--traj-v-max', type=float, default=0.08)
parser.add_argument('--traj-v-corner', type=float, default=0.02)
parser.add_argument('--traj-corner-blend', type=float, default=0.35)

# Move to start
parser.add_argument('--move-duration', type=float, default=3.0,
                    help='Move-to-start duration [s] (default: 3.0)')

# Home
parser.add_argument('--home-ee', type=float, nargs=2, default=None,
                    metavar=('X', 'Y'),
                    help='Home EE position [m] (overrides auto)')
parser.add_argument('--home-joints', type=float, nargs=2, default=None,
                    metavar=('Q1_DEG', 'Q2_DEG'),
                    help='Home joint angles [deg] (overrides auto)')

args = parser.parse_args()

# ============================================================================
# URDF PATH
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = str(PROJECT_ROOT / "model_using_onshape_to_robot" / "manipulator_cable" / "manipulator_cable_obj.urdf")

# ============================================================================
# CONFIGURATION
# ============================================================================
MANIP_CONFIG = create_cable_manipulator_config(
    urdf_path=URDF_PATH,
    joint_angles={
        'link1_base': np.deg2rad(10.0),
        'link2_link1': np.deg2rad(-10.0),
    },
    damping=tuple(args.joint_damping),
    stiffness=tuple(args.joint_stiffness),
    tilt_roll_deg=args.tilt_roll,
    tilt_pitch_deg=args.tilt_pitch,
)


# ============================================================================
# BUILD TRAJECTORY (pure NumPy)
# ============================================================================

def build_trajectory(args):
    """Create the trajectory object from CLI args."""
    if args.traj_shape == 'rect':
        return RectTrajectory(
            x_range=tuple(args.traj_x_range),
            y_range=tuple(args.traj_y_range),
            N=args.traj_n,
            lap_duration=args.duration,
            v_max=args.traj_v_max,
            v_corner=args.traj_v_corner,
            corner_blend=args.traj_corner_blend,
        )
    elif args.traj_shape == 'circle':
        return CircleTrajectory(
            cx=args.traj_cx, cy=args.traj_cy, radius=args.traj_radius,
            N=args.traj_n, lap_duration=args.duration,
        )
    else:  # line
        return LineTrajectory(
            cx=args.traj_cx, cy=args.traj_cy, radius=args.traj_radius,
            N=args.traj_n, lap_duration=args.duration,
        )


# ============================================================================
# SCENE SETUP
# ============================================================================

def build_scene(manip: CupManipulatorTendonIsaac, world: World):
    """Load robot USD into the Isaac Sim stage and configure it."""
    manip.prepare_usd()
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


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_computed_torque(args):
    """Full computed-torque simulation loop in Isaac Sim."""

    print("\n" + "=" * 80)
    print(colored(
        "COMPUTED TORQUE — Isaac Sim (no Drake)",
        "cyan", attrs=["bold"],
    ))
    print("=" * 80)

    # ── 1. Create robot wrapper ─────────────────────────────────────────────
    manip = CupManipulatorTendonIsaac(MANIP_CONFIG)

    # ── 2. Scene setup (before World) ───────────────────────────────────────
    # prepare_usd() must run before World() is created
    manip.prepare_usd()

    # ── 3. Create World ─────────────────────────────────────────────────────
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=args.dt,
        rendering_dt=args.dt,
    )
    world.scene.add_default_ground_plane()

    # Load robot into stage (after World)
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

    # ── 3b. Cable visualization — pre-allocate prims BEFORE world.reset() ───
    import omni.usd
    _stage = omni.usd.get_context().get_stage()
    _drake_urdf = str(PROJECT_ROOT / "model_using_onshape_to_robot" / "manipulator_cable" / "manipulator_cable_obj.urdf")
    cable_viz = CableVisualizerIsaac(_stage, _drake_urdf)
    cable_viz.create_prims(
        q1=MANIP_CONFIG.joint_configs['link1_base'].position,
        q2=MANIP_CONFIG.joint_configs['link2_link1'].position,
    )
    _CABLE_UPDATE_INTERVAL = 5  # update cables every N physics steps

    # ── 4. First reset → create Articulation ────────────────────────────────
    world.reset()
    manip.initialize_state()

    # ── 5. Initialize dynamics view (ArticulationView for M, C, g) ──────────
    manip.initialize_dynamics_view(world)

    L1, L2 = manip._get_link_lengths()
    r_p = manip.r_p
    print(colored(f"  Link lengths: L1={L1*1e3:.1f} mm  L2={L2*1e3:.1f} mm", "cyan"))
    print(colored(f"  Pulley radius: r_p={r_p*1e3:.1f} mm", "cyan"))

    # ── 6. Build trajectory ─────────────────────────────────────────────────
    main_traj = build_trajectory(args)
    print(colored(
        f"  ✓ Trajectory: {args.traj_shape}  N={args.traj_n}  "
        f"lap={args.duration:.1f} s",
        "green",
    ))

    # ── 7. Compute initial pose via IK ──────────────────────────────────────
    from robots.cup_manipulator_tendon_isaac import solve_2r_ik

    seed = np.array([np.deg2rad(5.0), np.deg2rad(15.0)])
    p_first = main_traj.eval_position(0.0)

    q_init = None
    if args.home_joints is not None:
        q_init = np.deg2rad(np.array(args.home_joints, dtype=float))
        print(colored(f"  Home (--home-joints): q=[{args.home_joints[0]:.1f}°, {args.home_joints[1]:.1f}°]", "cyan"))
    elif args.home_ee is not None:
        q_init, ok = solve_2r_ik(L1, L2, np.array(args.home_ee), seed)
        if not ok:
            q_init = None
            print(colored(f"  ⚠  IK for --home-ee failed — auto-resolving.", "yellow"))
    if q_init is None:
        q_init, ok = solve_2r_ik(L1, L2, p_first, seed)
        if not ok:
            q_init = np.array([np.deg2rad(5.0), np.deg2rad(15.0)])

    # ── 8. Build move-to-start preamble ─────────────────────────────────────
    from robots.cup_manipulator_tendon_isaac import forward_kinematics_2r

    q_pre = q_init + np.array([np.deg2rad(-5.0), np.deg2rad(5.0)])
    p_start = forward_kinematics_2r(L1, L2, q_pre[0], q_pre[1])
    p_end = p_first
    v_end = main_traj.eval_velocity(0.0)

    move_spline = build_move_to_start(p_start, p_end, v_end, args.move_duration)
    traj_source = PreambleTrajectorySource(move_spline, main_traj)
    move_duration = traj_source.move_duration

    print(colored(
        f"  ✓ Move-to-start: {move_duration:.1f} s  "
        f"q_init=[{np.rad2deg(q_init[0]):.1f}°, {np.rad2deg(q_init[1]):.1f}°]  "
        f"EE_init=({forward_kinematics_2r(L1, L2, q_init[0], q_init[1])[0]*1e3:.1f}, "
        f"{forward_kinematics_2r(L1, L2, q_init[0], q_init[1])[1]*1e3:.1f}) mm",
        "green",
    ))

    # ── 9. Set initial state ────────────────────────────────────────────────
    manip.set_positions_user_order(q_init)
    manip.set_velocities_user_order(np.zeros(2))

    # ── 10. Create controller ───────────────────────────────────────────────
    ct = ComputedTorqueController(
        Kp=args.ct_kp,
        Kd=args.ct_kd,
        tau_max=args.ct_tau_max,
        pulley_radius=r_p,
    )
    wn = ct.omega_n
    zeta = ct.zeta

    print(colored(
        f"\n▶  COMPUTED-TORQUE — Isaac Sim"
        f"\n   Gains: Kp={args.ct_kp}  Kd={args.ct_kd}"
        f"   →  ωn={wn:.1f} rad/s  ζ={zeta:.2f}"
        f"\n   tau_max={args.ct_tau_max} Nm   dt={args.dt*1e3:.1f} ms"
        f"\n   Press Ctrl-C to stop and show plots.",
        "cyan",
    ))

    # ── 11. Data logging arrays ─────────────────────────────────────────────
    max_steps = int((args.duration * 5 + move_duration) / args.dt) + 1000
    log_t = np.zeros(max_steps)
    log_q = np.zeros((max_steps, 2))
    log_q_dot = np.zeros((max_steps, 2))
    log_q_des = np.zeros((max_steps, 2))
    log_tau_raw = np.zeros((max_steps, 2))
    log_tau_clip = np.zeros((max_steps, 2))
    log_tens = np.zeros((max_steps, 2))
    log_ee_ref = np.zeros((max_steps, 2))
    log_ee_vel_ref = np.zeros((max_steps, 2))
    log_ee_acc_ref = np.zeros((max_steps, 2))

    # ── 12. Simulation loop ─────────────────────────────────────────────────
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

            # Computed torque
            ct_out = ct.compute(q, q_dot, q_des, q_dot_ref, q_ddot_ref, M, h)

            # Apply torques
            manip.set_joint_torques(ct_out.tau_clip)

            # Log
            if step < max_steps:
                log_t[step] = t
                log_q[step] = q
                log_q_dot[step] = q_dot
                log_q_des[step] = ct_out.q_des
                log_tau_raw[step] = ct_out.tau_raw
                log_tau_clip[step] = ct_out.tau_clip
                log_tens[step] = [ct_out.T_green, ct_out.T_red]
                log_ee_ref[step] = ee_pos_ref
                log_ee_vel_ref[step] = ee_vel_ref
                log_ee_acc_ref[step] = ee_acc_ref

            # Update cable visualization (every N steps for performance)
            if step % _CABLE_UPDATE_INTERVAL == 0:
                cable_viz.update(q[0], q[1])

            # Step physics
            world.step(render=(_render_mode == "native"))

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
    n_logged = step
    log_t = log_t[:n_logged]
    log_q = log_q[:n_logged]
    log_q_dot = log_q_dot[:n_logged]
    log_q_des = log_q_des[:n_logged]
    log_tau_raw = log_tau_raw[:n_logged]
    log_tau_clip = log_tau_clip[:n_logged]
    log_tens = log_tens[:n_logged]
    log_ee_ref = log_ee_ref[:n_logged]
    log_ee_vel_ref = log_ee_vel_ref[:n_logged]
    log_ee_acc_ref = log_ee_acc_ref[:n_logged]

    laps_done = int(max(0.0, t - move_duration) / args.duration)
    print(colored(
        f"\n  Simulation stopped at t={t:.2f} s  ({laps_done} full laps, {n_logged} steps).",
        "yellow",
    ))

    # ── 13. Plot ────────────────────────────────────────────────────────────
    plot_results(
        log_t, log_q, log_q_dot, log_q_des,
        log_tau_raw, log_tau_clip, log_tens,
        log_ee_ref, log_ee_vel_ref, log_ee_acc_ref,
        main_traj, L1, L2, r_p, ct, args,
    )


def run_scene_viz(args):
    """Load robot, visualise, no control."""
    print("\n" + "=" * 80)
    print(colored("SCENE VISUALIZATION — Isaac Sim", "cyan", attrs=["bold"]))
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

    # Cable visualization — pre-allocate prims BEFORE world.reset()
    import omni.usd
    _stage = omni.usd.get_context().get_stage()
    _drake_urdf = str(PROJECT_ROOT / "model_using_onshape_to_robot" / "manipulator_cable" / "manipulator_cable_obj.urdf")
    cable_viz = CableVisualizerIsaac(_stage, _drake_urdf)
    cable_viz.create_prims(
        q1=MANIP_CONFIG.joint_configs['link1_base'].position,
        q2=MANIP_CONFIG.joint_configs['link2_link1'].position,
    )

    world.reset()
    manip.initialize_state()
    manip.set_initial_positions()

    print(colored("\n▶  Scene visualisation — press Ctrl-C to exit.", "cyan"))
    try:
        while simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        pass
    print(colored("  Done.", "yellow"))


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(
    t, q, q_dot, q_des, tau_raw, tau_clip, tens,
    ee_ref, ee_vel_ref, ee_acc_ref,
    main_traj, L1, L2, r_p, ct, args,
):
    """3×3 plot matching the PyDrake ComputedTorqueSimulation.plot()."""
    from robots.cup_manipulator_tendon_isaac import forward_kinematics_2r, analytical_jacobian_2r

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

    # EE acceleration actual via finite diff of velocity
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

    # ── 3×3 figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(18, 11))
    fig.suptitle(
        f'Computed Torque (Isaac Sim) — {args.traj_shape}   '
        f'Kp={args.ct_kp}  Kd={args.ct_kd}  '
        f'ωn={wn:.1f} rad/s  ζ={zeta:.2f}',
        fontsize=12, fontweight='bold',
    )

    # Row 0: End-Effector
    ax = axes[0, 0]
    ax.plot(t, ee_x, 'b-', lw=1.8, label='x actual')
    ax.plot(t, ee_y, 'r-', lw=1.8, label='y actual')
    ax.plot(t, ee_ref[:, 0], 'b--', lw=1.5, label='x ref')
    ax.plot(t, ee_ref[:, 1], 'r--', lw=1.5, label='y ref')
    ax.set_title('EE Position'); ax.set_ylabel('[m]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    ax = axes[0, 1]
    ax.plot(t, ee_vx, 'b-', lw=1.8, label='ẋ actual')
    ax.plot(t, ee_vy, 'r-', lw=1.8, label='ẏ actual')
    ax.plot(t, ee_vel_ref[:, 0], 'b--', lw=1.5, label='ẋ ref')
    ax.plot(t, ee_vel_ref[:, 1], 'r--', lw=1.5, label='ẏ ref')
    ax.set_ylim(*_pct_ylim(ee_vx, ee_vy, ee_vel_ref[:, 0], ee_vel_ref[:, 1]))
    ax.set_title('EE Velocity'); ax.set_ylabel('[m/s]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    ax = axes[0, 2]
    ax.plot(t, ee_ax, 'b-', lw=1.8, label='ẍ actual')
    ax.plot(t, ee_ay, 'r-', lw=1.8, label='ÿ actual')
    ax.plot(t, ee_acc_ref[:, 0], 'b--', lw=1.5, label='ẍ ref')
    ax.plot(t, ee_acc_ref[:, 1], 'r--', lw=1.5, label='ÿ ref')
    ax.set_ylim(*_pct_ylim(ee_ax, ee_ay, ee_acc_ref[:, 0], ee_acc_ref[:, 1]))
    ax.set_title('EE Acceleration'); ax.set_ylabel('[m/s²]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    # Row 1: Joints
    ax = axes[1, 0]
    ax.plot(t, np.rad2deg(q[:, 0]), 'b-', lw=1.8, label='q1 act')
    ax.plot(t, np.rad2deg(q[:, 1]), 'r-', lw=1.8, label='q2 act')
    ax.plot(t, np.rad2deg(q_des[:, 0]), 'b--', lw=1.5, label='q1 des')
    ax.plot(t, np.rad2deg(q_des[:, 1]), 'r--', lw=1.5, label='q2 des')
    ax.set_title('Joint Position'); ax.set_ylabel('[deg]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    ax = axes[1, 1]
    ax.plot(t, np.rad2deg(q_dot[:, 0]), 'b-', lw=1.8, label='q̇1 act')
    ax.plot(t, np.rad2deg(q_dot[:, 1]), 'r-', lw=1.8, label='q̇2 act')
    ax.plot(t, np.rad2deg(q_dot_ref[:, 0]), 'b--', lw=1.5, label='q̇1 ref')
    ax.plot(t, np.rad2deg(q_dot_ref[:, 1]), 'r--', lw=1.5, label='q̇2 ref')
    ax.set_ylim(*_pct_ylim(np.rad2deg(q_dot[:, 0]), np.rad2deg(q_dot[:, 1]),
                           np.rad2deg(q_dot_ref[:, 0]), np.rad2deg(q_dot_ref[:, 1])))
    ax.set_title('Joint Velocity'); ax.set_ylabel('[deg/s]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    ax = axes[1, 2]
    ax.plot(t, np.rad2deg(q1_ddot), 'b-', lw=1.8, label='q̈1 act')
    ax.plot(t, np.rad2deg(q2_ddot), 'r-', lw=1.8, label='q̈2 act')
    ax.plot(t, np.rad2deg(q_ddot_ref[:, 0]), 'b--', lw=1.5, label='q̈1 ref')
    ax.plot(t, np.rad2deg(q_ddot_ref[:, 1]), 'r--', lw=1.5, label='q̈2 ref')
    ax.set_ylim(*_pct_ylim(np.rad2deg(q1_ddot), np.rad2deg(q2_ddot),
                           np.rad2deg(q_ddot_ref[:, 0]), np.rad2deg(q_ddot_ref[:, 1])))
    ax.set_title('Joint Acceleration'); ax.set_ylabel('[deg/s²]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    # Row 2: Torques | Tensions | EE XY
    ax = axes[2, 0]
    ax.plot(t, tau_raw[:, 0], 'b-', lw=1.8, label='τ1 required')
    ax.plot(t, tau_raw[:, 1], 'r-', lw=1.8, label='τ2 required')
    ax.plot(t, tau_clip[:, 0], 'b--', lw=1.5, label='τ1 applied')
    ax.plot(t, tau_clip[:, 1], 'r--', lw=1.5, label='τ2 applied')
    ax.axhline(args.ct_tau_max, color='k', ls=':', lw=0.8, label=f'±{args.ct_tau_max} Nm')
    ax.axhline(-args.ct_tau_max, color='k', ls=':', lw=0.8)
    ax.axhline(0, color='k', lw=0.5)
    _tau_peak = max(np.abs(tau_raw).max(), args.ct_tau_max) * 1.15
    ax.set_ylim(-_tau_peak, _tau_peak)
    ax.set_title('Torque: required vs applied')
    ax.set_ylabel('[Nm]'); ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    ax = axes[2, 1]
    ax.plot(t, tens[:, 0], 'g-', lw=1.2, label='T_green')
    ax.plot(t, tens[:, 1], 'r-', lw=1.2, label='T_red')
    ax.plot(t, tau_raw[:, 1] / r_p, 'k--', lw=0.8,
            label=f'F_net=τ2/r_p  (r_p={r_p*1e3:.1f} mm)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title('Cable Tensions'); ax.set_ylabel('[N]'); ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    ax = axes[2, 2]
    ax.plot(main_traj.ee_x_tgt, main_traj.ee_y_tgt, 'k--', lw=1.0, label='Reference')
    ax.plot(ee_x, ee_y, 'b-', lw=1.3, label='Actual')
    ax.plot(ee_x[0], ee_y[0], 'go', ms=8, label='Start')
    ax.set_aspect('equal')
    ax.set_title(f'EE Path ({args.traj_shape})')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    fig.tight_layout()

    # Save
    stamp = _time.strftime('%Y%m%d_%H%M%S')
    plots_dir = PROJECT_ROOT / 'plots'
    plots_dir.mkdir(exist_ok=True)
    fname = plots_dir / f'ct_isaac_{args.traj_shape}_{stamp}.png'
    fig.savefig(str(fname), dpi=150, bbox_inches='tight')
    print(colored(f"\n  📊 Figure saved: {fname}", "green"))

    # Print tracking metrics
    err = np.sqrt((ee_x - ee_ref[:, 0])**2 + (ee_y - ee_ref[:, 1])**2)
    print(colored(f"  Final EE: [{ee_x[-1]:.4f}, {ee_y[-1]:.4f}] m", "green"))
    print(colored(f"  Ref at t_end: [{ee_ref[-1, 0]:.4f}, {ee_ref[-1, 1]:.4f}] m", "green"))
    print(colored(f"  Final tracking error: {err[-1]*1e3:.2f} mm", "green"))
    print(colored(f"  Mean tracking RMS: {np.sqrt(np.mean(err**2))*1e3:.2f} mm", "green"))

    try:
        plt.show(block=True)
    except Exception as e:
        print(colored(f"  ⚠ plt.show() failed ({e}) — see saved PNG.", "yellow"))


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    if args.mode == 'scene-viz':
        run_scene_viz(args)
    elif args.mode == 'computed-torque':
        run_computed_torque(args)
    else:
        print(colored(f"Unknown mode: {args.mode}", "red"))
        sys.exit(1)

    simulation_app.close()


if __name__ == "__main__":
    main()
