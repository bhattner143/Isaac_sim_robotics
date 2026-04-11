#!/usr/bin/env python3
"""
script_cup_manipulator_sea_cable_pydrake.py
============================================
Series Elastic Actuator (SEA) cable model for joint 2 of the cup manipulator.

REAL-WORLD MOTIVATION
─────────────────────
  Joint 1 (shoulder): motor shaft → gearbox → rigid link  →  no compliance
  Joint 2 (elbow):    motor drum → cable → SPRING → big pulley → link
                                             ↑
                       This spring (belt elasticity or explicit compliance
                       element) is "in series" between the motor and the
                       joint.  Weaker spring → larger deflection needed to
                       transmit the same torque → motor must travel more →
                       torque lags the CT command.

SEA PHYSICS (joint-2 only)
──────────────────────────
  Motor cable displacement (state):  l_m  [m]
  Spring extension:  δ = l_m − r_p·q₂
  Motor velocity:    l̇_m = ω_m·(l_m_des − l_m)     first-order position servo
  Cable force:       F = k_s·δ + b_c·(l̇_m − r_p·q̇₂)
                     F = max(F, 0)                    cable can ONLY pull
  Joint-2 torque:    τ₂ = r_p · F

CONTROL (same CT outer-loop as standard computed-torque)
─────────────────────────────────────────────────────────
  IK:        q_des = solve_2r_ik(p_ref)
  PD+FF:     a_des = q̈_ref + Kp·(q_des−q) + Kd·(q̇_ref−q̇)
  Inv-dyn:   τ_des = CalcInverseDynamics(ctx, a_des)   [M·a + C·v + g]
  Joint 1:   τ₁ = τ₁_des                              (rigid direct drive)
  Joint 2:   l_m_des = r_p·q₂ + τ₂_des/(k_s·r_p)     (desired motor pos)

EFFECT OF PARAMETERS
─────────────────────
  k_s  high → stiff tendon → small δ_des → fast torque → low lag
  k_s  low  → compliant    → large δ_des → slow torque → high lag
  ω_m  high → fast motor → tracks l_m_des quickly → less lag
  ω_m  low  → slow motor → can't keep up → more lag at high speed

USAGE
─────
  # Default spring, default CT gains, rect trajectory
  python script_cup_manipulator_sea_cable_pydrake.py

  # Very soft spring (high lag)
  python script_cup_manipulator_sea_cable_pydrake.py --spring-stiffness 30

  # Stiff spring (near-rigid)
  python script_cup_manipulator_sea_cable_pydrake.py --spring-stiffness 5000

  # Overlay spring vs rigid on the same plot
  python script_cup_manipulator_sea_cable_pydrake.py --compare

  # Sweep over multiple stiffness values
  python script_cup_manipulator_sea_cable_pydrake.py \\
      --compare --spring-stiffness 50 --compare-rigid-ks 5000
"""

import argparse
import os
import signal
import sys
from pathlib import Path

import numpy as np

import matplotlib
import platform as _platform
if _platform.system() == 'Darwin':
    try:
        matplotlib.use('MacOSX')
    except Exception:
        matplotlib.use('TkAgg')
else:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass   # leave whatever default was set; savefig still works

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from termcolor import colored

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Simulator,
    LogVectorOutput,
    LeafSystem,
    MeshcatVisualizer,
    StartMeshcat,
    PiecewisePolynomial,
    SceneGraph,
    SpatialInertia,
    UnitInertia,
    Parser,
)
from pydrake.multibody.tree import RevoluteSpring

sys.path.insert(0, str(Path(__file__).parent))
from robots.cup_manipulator_tendon import (
    CupManipulatorTendon,
    create_cable_manipulator_config,
)
from controller.controller import ComputedTorqueController
from actuators.sea import SEACableActuator
from actuators.motor_dynamics import MotorMode
from project_utils.viz_cables import draw_cables
from actuators.motor import get_motor, MOTOR_CHOICES

# ─── Constants ────────────────────────────────────────────────────────────────
_DT   = 0.002   # plant & controller timestep [s]
_URDF = "model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf"
_M_PATCH = SpatialInertia(
    mass=0.3, p_PScm_E=np.zeros(3), G_SP_E=UnitInertia(1e-2, 1e-2, 1e-2),
)


# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Series Elastic Actuator (SEA) cable simulation — "
                "joint 1 rigid motor, joint 2 motor→spring→cable→pulley",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

_mot = parser.add_argument_group("motor model  (elbow / joint 2)")
_mot.add_argument("--motor", choices=MOTOR_CHOICES, default="AK60_6_KV80_Config",
                  help="CubeMars motor model for the elbow joint.")

_sea = parser.add_argument_group("SEA cable model  (joint 2)")
_sea.add_argument("--sea-mode",  choices=["torque", "position"], default="torque",
                  help="Motor dynamics mode: 'torque' = 2nd-order rotor dynamics "
                       "(MIT torque mode, default), 'position' = 1st-order position servo.")
_sea.add_argument("--spring-stiffness", type=float, default=30000 ,
                  metavar="K_S",
                  help="Cable spring stiffness k_s [N/m].  Lower → more lag.")
_sea.add_argument("--cable-damping",    type=float, default=2.0,
                  metavar="B_C",
                  help="Cable dashpot damping b_c [N·s/m]")
# Motor open-loop bandwidth from datasheet: ω_b = 1/τ_m ≈ 400 rad/s (AK60-6).
# However, this is the *open-loop mechanical* bandwidth, not the closed-loop
# position-servo bandwidth.  Real closed-loop bandwidth is 2–5× lower depending
# on the motor's internal PID tuning.  We default to a conservative 100 rad/s
# since CubeMars does not publish a closed-loop position bandwidth spec.
_DEFAULT_MOTOR_BW = 100.0  # rad/s  (conservative closed-loop estimate)
_sea.add_argument("--motor-bandwidth",  type=float, default=None,
                  metavar="W_M",
                  help="Motor position servo bandwidth ω_m [rad/s].  "
                       f"Default: {_DEFAULT_MOTOR_BW} rad/s (conservative; "
                       "open-loop 1/τ_m ≈ 400 rad/s).")
_sea.add_argument("--compare",          action="store_true",
                  help="Run spring sim AND a near-rigid sim, overlay plots.")
_sea.add_argument("--compare-rigid-ks", type=float, default=5000.0,
                  metavar="K_S_RIGID",
                  help="Spring stiffness for the 'rigid' comparison run.")

_ct = parser.add_argument_group("computed-torque gains")
_ct.add_argument("--ct-kp",      type=float, default=100.0,
                 help="CT position gain Kp [1/s²]")
_ct.add_argument("--ct-kd",      type=float, default=40.0,
                 help="CT velocity gain Kd [1/s]")
_ct.add_argument("--ct-tau-max", type=float, default=None,
                 help="Torque saturation [Nm].  Default: motor peak_torque_joint.")

_sim = parser.add_argument_group("simulation")
_sim.add_argument("--duration",       type=float, default=10.0,
                  help="Lap duration [s]")
_sim.add_argument("--move-duration",  type=float, default=3.0,
                  help="Move-to-start preamble duration [s].  0 to disable.")
_sim.add_argument("--no-meshcat",     action="store_true",
                  help="Disable Meshcat 3-D visualisation")

_rob = parser.add_argument_group("robot mount")
_rob.add_argument("--tilt-roll",       type=float, default=0.0)
_rob.add_argument("--tilt-pitch",      type=float, default=0.0)
_rob.add_argument("--joint-damping",   type=float, nargs=2, default=[0.05, 0.05],
                  metavar=("D1", "D2"))
_rob.add_argument("--joint-stiffness", type=float, nargs=2, default=[0.0, 0.0],
                  metavar=("K1", "K2"),
                  help="Passive joint spring (RevoluteSpring) — independent of cable SEA.")

_traj = parser.add_argument_group("trajectory (rect)")
_traj.add_argument("--traj-x-range",      type=float, nargs=2, default=[0.49, 0.51],
                   metavar=("X_MIN", "X_MAX"))
_traj.add_argument("--traj-y-range",      type=float, nargs=2, default=[-0.08, 0.08],
                   metavar=("Y_MIN", "Y_MAX"))
_traj.add_argument("--traj-n",            type=int,   default=60)
_traj.add_argument("--traj-v-max",        type=float, default=0.9)
_traj.add_argument("--traj-v-corner",     type=float, default=0.05)
_traj.add_argument("--traj-corner-blend", type=float, default=0.35)

args = parser.parse_args()

# ─── Motor-derived defaults ───────────────────────────────────────────────────
_motor = get_motor(args.motor)
_motor_mode = MotorMode(args.sea_mode)
if args.motor_bandwidth is None:
    args.motor_bandwidth = _DEFAULT_MOTOR_BW  # see comment at --motor-bandwidth definition
if args.ct_tau_max is None:
    args.ct_tau_max = _motor.peak_torque_joint          # saturation from datasheet

_mode_label = "torque (2nd-order rotor)" if _motor_mode == MotorMode.TORQUE else "position (1st-order servo)"
print(colored(
    f"\n  Motor: {args.motor}  —  SEA mode: {_mode_label}"
    f"\n    gear ratio      = {_motor.gear_ratio}"
    f"\n    peak torque     = {_motor.peak_torque_joint} Nm  (τ_max)"
    f"\n    continuous τ    = {_motor.continuous_torque_joint} Nm"
    f"\n    max joint vel   = {_motor.max_velocity_joint:.2f} rad/s"
    f"  ({_motor.max_velocity_joint * 60 / (2 * np.pi):.1f} rpm)"
    f"\n    viscous damping = {_motor.viscous_damping_joint} Nm·s/rad"
    f"\n    rotor inertia   = {_motor.rotor_inertia_joint:.5f} kg·m²  (reflected)"
    f"\n    → ω_m = {args.motor_bandwidth:.2f} rad/s"
    f"   τ_max = {args.ct_tau_max:.1f} Nm",
    "yellow",
))


# ════════════════════════════════════════════════════════════════════════════
# Trajectory builders
# ════════════════════════════════════════════════════════════════════════════

def _build_rect_trajectory(manip, plant, args):
    """C² rectangular EE trajectory with speed-profiled corners.

    Returns (traj_ref, traj_vel, traj_acc, ee_x_tgt, ee_y_tgt).
    """
    L1, L2 = manip.ik.get_link_lengths(plant)
    r_max  = L1 + L2
    r_min  = abs(L1 - L2)

    x_min, x_max = args.traj_x_range
    y_min, y_max = args.traj_y_range
    N   = args.traj_n
    W   = x_max - x_min
    H   = y_max - y_min
    P   = 2.0 * (W + H)
    _cs = np.array([0.0, W, W + H, 2.0 * W + H])

    def _s_to_xy(s):
        s = s % P
        if   s <= W:            return x_min + s,            y_min
        elif s <= W + H:        return x_max,                 y_min + (s - W)
        elif s <= 2.0 * W + H:  return x_max - (s - W - H),  y_max
        else:                   return x_min,                 y_max - (s - 2.0 * W - H)

    def _corner_dist(s):
        s = s % P
        d = np.abs(s - _cs)
        return float(np.minimum(d, P - d).min())

    _v_max    = args.traj_v_max
    _v_corner = args.traj_v_corner
    _d_blend  = args.traj_corner_blend * min(W, H)

    def _speed(s):
        t = np.clip(_corner_dist(s) / max(_d_blend, 1e-9), 0.0, 1.0)
        return _v_corner + (_v_max - _v_corner) * t * t * (3.0 - 2.0 * t)

    _s_vals = np.linspace(0.0, P, N + 1, endpoint=True)
    _speeds = np.array([_speed(s) for s in _s_vals])
    _ds     = P / N
    _t_raw  = np.zeros(N + 1)
    for i in range(N):
        _t_raw[i + 1] = _t_raw[i] + _ds / (0.5 * (_speeds[i] + _speeds[i + 1]))
    t_wp = _t_raw * (args.duration / _t_raw[-1])
    _xy  = np.array([_s_to_xy(s) for s in _s_vals])
    ee_x = _xy[:N, 0]
    ee_y = _xy[:N, 1]

    # Clamp to reachable workspace
    _r = np.hypot(ee_x, ee_y)
    _far = _r > r_max
    if _far.any():
        ee_x[_far] *= r_max * 0.97 / _r[_far]
        ee_y[_far] *= r_max * 0.97 / _r[_far]
    _close = _r < r_min + 0.01
    if _close.any():
        _r_c = np.maximum(_r[_close], 1e-6)
        ee_x[_close] *= (r_min + 0.01) / _r_c
        ee_y[_close] *= (r_min + 0.01) / _r_c

    wp = np.column_stack([np.append(ee_x, ee_x[0]),
                          np.append(ee_y, ee_y[0])]).T
    traj     = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(t_wp, wp)
    traj_vel = traj.derivative(1)
    traj_acc = traj.derivative(2)
    return traj, traj_vel, traj_acc, ee_x, ee_y


def _build_move_to_start(manip, plant, traj, traj_vel, move_duration):
    """Cubic-Hermite approach from near the first waypoint (zero initial velocity)."""
    L1, L2 = manip.ik.get_link_lengths(plant)
    p_end  = traj.value(0.0).ravel()
    seed   = np.array([np.deg2rad(5.0), np.deg2rad(15.0)])
    q_end, ok = manip.ik._solve_2r_core(L1, L2, p_end, seed)
    if not ok:
        q_end = seed.copy()

    # Pre-home: small q-space offset  for a meaningful approach
    q_pre   = q_end + np.array([np.deg2rad(-5.0), np.deg2rad(5.0)])
    tmp_ctx = plant.CreateDefaultContext()
    manip.set_positions_user_order(plant, tmp_ctx, q_pre)
    p_start = manip.get_end_effector_position(plant, tmp_ctx)[:2]
    v_end   = traj_vel.value(0.0).ravel()

    t_br  = np.array([0.0, move_duration])
    smp   = np.column_stack([p_start, p_end])
    smp_d = np.column_stack([np.zeros(2), v_end])
    move_traj     = PiecewisePolynomial.CubicHermite(t_br, smp, smp_d)
    move_traj_vel = move_traj.derivative(1)
    move_traj_acc = move_traj.derivative(2)
    return move_traj, move_traj_vel, move_traj_acc, q_end


# ════════════════════════════════════════════════════════════════════════════
# Single simulation run
# ════════════════════════════════════════════════════════════════════════════

def run_simulation(
    ks: float,
    b_c: float,
    omega_m: float,
    label: str,
    meshcat,
    motor_mode: MotorMode = MotorMode.TORQUE,
) -> dict:
    """Build, run, and collect logs for a single SEA configuration."""
    _mode_str = "torque (2nd-order)" if motor_mode == MotorMode.TORQUE else "position (1st-order)"
    print(colored(f"\n{'=' * 60}", "cyan"))
    print(colored(f"  SEA Simulation: {label}  [{_mode_str}]", "cyan"))
    print(colored(f"  Motor: {args.motor}  (τ_peak={_motor.peak_torque_joint} Nm, "
                  f"τ_cont={_motor.continuous_torque_joint} Nm)", "cyan"))
    print(colored(f"  k_s={ks} N/m   b_c={b_c} N·s/m   ω_m={omega_m} rad/s", "cyan"))
    print(colored(f"{'=' * 60}", "cyan"))

    # ── 1. Config ────────────────────────────────────────────────────────────
    manip_config = create_cable_manipulator_config(
        urdf_path=_URDF,
        joint_angles={
            CupManipulatorTendon.JT1_NAME: np.deg2rad(5.0),
            CupManipulatorTendon.JT2_NAME: np.deg2rad(15.0),
        },
        damping=tuple(args.joint_damping),
        stiffness=tuple(args.joint_stiffness),
        tilt_roll_deg=args.tilt_roll,
        tilt_pitch_deg=args.tilt_pitch,
    )

    # ── 2. Plant ─────────────────────────────────────────────────────────────
    builder     = DiagramBuilder()
    plant       = MultibodyPlant(time_step=_DT)
    scene_graph = builder.AddSystem(SceneGraph())
    plant.RegisterAsSourceForSceneGraph(scene_graph)

    manipulator = CupManipulatorTendon(manip_config, enable_visualization=True)
    parser_urdf = Parser(plant)
    manipulator.load_urdf_to_plant(plant, parser_urdf)
    orientation = np.deg2rad([
        manip_config.tilt_roll_deg, manip_config.tilt_pitch_deg, 0.0,
    ])
    manipulator.weld_base_to_world(plant, position=np.zeros(3), orientation=orientation)
    manipulator.add_joint_actuators(plant)
    manipulator.set_joint_properties(plant)

    # Optional passive RevoluteSpring (independent of the cable SEA spring)
    for jt_name in [CupManipulatorTendon.JT1_NAME, CupManipulatorTendon.JT2_NAME]:
        cfg = manip_config.joint_configs.get(jt_name)
        if cfg and cfg.stiffness and cfg.stiffness > 0.0:
            jt = manipulator.get_joint_by_name(plant, jt_name)
            plant.AddForceElement(
                RevoluteSpring(jt, nominal_angle=0.0, stiffness=cfg.stiffness),
            )

    manipulator.add_end_effector_frame(plant)
    plant.Finalize()
    builder.AddSystem(plant)

    # Geometry wiring
    builder.Connect(
        plant.get_geometry_pose_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()),
    )
    builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port(),
    )
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    # ── 3. Trajectory ────────────────────────────────────────────────────────
    traj, traj_vel, traj_acc, ee_x_tgt, ee_y_tgt = \
        _build_rect_trajectory(manipulator, plant, args)

    if args.move_duration > 0.0:
        move_traj, move_vel, move_acc, q_init = \
            _build_move_to_start(manipulator, plant, traj, traj_vel, args.move_duration)
    else:
        move_traj = move_vel = move_acc = None
        L1, L2 = manipulator.ik.get_link_lengths(plant)
        p0 = traj.value(0.0).ravel()
        q_init, _ = manipulator.ik._solve_2r_core(
            L1, L2, p0, np.array([np.deg2rad(5.0), np.deg2rad(15.0)]),
        )

    # ── 4. Controller + Actuator (two-block architecture) ────────────────────
    ct = builder.AddSystem(
        ComputedTorqueController(
            plant, manipulator,
            Kp=args.ct_kp, Kd=args.ct_kd, tau_max=args.ct_tau_max,
        ),
    )
    ct.set_name("CT_ctrl")

    sea = builder.AddSystem(
        SEACableActuator(
            plant, manipulator,
            k_s=ks, b_c=b_c,
            tau_max=args.ct_tau_max, dt=_DT,
            motor_mode=motor_mode,
            motor_cfg=_motor,
            omega_m=omega_m,
        ),
    )
    sea.set_name(f"SEA_ks{ks:.0f}_{motor_mode.value}")

    # Preamble-aware looping trajectory source
    class _PreambleSrc(LeafSystem):
        def __init__(s, mv, md, main_traj, period):
            super().__init__()
            s._mv     = mv
            s._md     = float(md)
            s._main   = main_traj
            s._period = float(period)
            s.DeclareVectorOutputPort("out", main_traj.rows(), s._calc)

        def _calc(s, ctx, out):
            t = ctx.get_time()
            if s._mv is not None and t < s._md:
                out.SetFromVector(s._mv.value(t).ravel())
            else:
                tw = max(0.0, t - s._md) % s._period
                out.SetFromVector(s._main.value(tw).ravel())

    ee_src  = builder.AddSystem(_PreambleSrc(move_traj, args.move_duration, traj,     args.duration))
    vel_src = builder.AddSystem(_PreambleSrc(move_vel,  args.move_duration, traj_vel, args.duration))
    acc_src = builder.AddSystem(_PreambleSrc(move_acc,  args.move_duration, traj_acc, args.duration))
    ee_src.set_name("EE_ref")
    vel_src.set_name("Vel_ref")
    acc_src.set_name("Acc_ref")

    # ── 5. Wire  (Trajectory → CT → SEA → Plant) ─────────────────────────────
    builder.Connect(ee_src.get_output_port(),           ct.GetInputPort("desired_ee_pos"))
    builder.Connect(vel_src.get_output_port(),          ct.GetInputPort("ee_vel_ref"))
    builder.Connect(acc_src.get_output_port(),          ct.GetInputPort("ee_acc_ref"))
    builder.Connect(plant.get_state_output_port(),      ct.GetInputPort("plant_state"))
    builder.Connect(ct.GetOutputPort("actuation"),      sea.GetInputPort("tau_desired"))
    builder.Connect(plant.get_state_output_port(),      sea.GetInputPort("plant_state"))
    builder.Connect(sea.GetOutputPort("actuation"),     plant.get_actuation_input_port())

    # ── 6. Loggers ───────────────────────────────────────────────────────────
    log_state = LogVectorOutput(plant.get_state_output_port(),          builder)
    log_act   = LogVectorOutput(sea.GetOutputPort("actuation"),         builder)
    log_diag  = LogVectorOutput(sea.GetOutputPort("diagnostics"),       builder)
    log_qdes  = LogVectorOutput(ct.GetOutputPort("joint_positions"),    builder)
    log_ref   = LogVectorOutput(ee_src.get_output_port(),               builder)
    log_vel   = LogVectorOutput(vel_src.get_output_port(),              builder)
    log_acc   = LogVectorOutput(acc_src.get_output_port(),              builder)

    diagram   = builder.Build()
    simulator = Simulator(diagram)
    sim_ctx   = simulator.get_mutable_context()

    # ── 7. Initialize ────────────────────────────────────────────────────────
    plant_ctx = plant.GetMyMutableContextFromRoot(sim_ctx)

    # Patch zero-mass bodies (Onshape URDF without material → mass=0 → SAP NaN)
    patched = []
    for idx in plant.GetBodyIndices(manipulator.model_instance):
        body = plant.get_body(idx)
        if body.default_mass() < 1e-6:
            body.SetSpatialInertiaInBodyFrame(plant_ctx, _M_PATCH)
            patched.append(body.name())
    if patched:
        print(colored(f"  ✓ Patched zero-mass bodies: {patched}", "yellow"))

    manipulator.set_positions_user_order(plant, plant_ctx, q_init)
    plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))

    # Set initial motor cable position = r_p · q₂_init  (spring at rest: δ=0)
    sea_ctx = sea.GetMyMutableContextFromRoot(sim_ctx)
    sea.initialize_spring_at_rest(sea_ctx, q_init[1])
    l_m_init = manipulator.PULLEY_RADIUS * q_init[1]

    ee0 = manipulator.get_end_effector_position(plant, plant_ctx)
    print(colored(
        f"  ✓ Init: q=[{np.rad2deg(q_init[0]):.1f}°, {np.rad2deg(q_init[1]):.1f}°]  "
        f"EE=({ee0[0]*1e3:.1f}, {ee0[1]*1e3:.1f}) mm  "
        f"l_m_init={l_m_init*1e3:.2f} mm",
        "green",
    ))

    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    # ── 8. Cable visualisation (if available) ─────────────────────────────
    try:
        manipulator.init_cable_rig(_URDF, springs_enabled=True)
        _rig = manipulator.rig
    except Exception:
        _rig = None

    def _viz_tick():
        if meshcat is None or _rig is None:
            return
        _ctx = simulator.get_mutable_context()
        _pc  = plant.GetMyMutableContextFromRoot(_ctx)
        # Read spring extension δ from SEA diagnostics
        _sea_ctx = sea.GetMyMutableContextFromRoot(_ctx)
        _diag = sea.GetOutputPort("diagnostics").Eval(_sea_ctx)
        _delta = _diag[2]  # spring extension δ [m]
        manipulator.compute_tangents(plant, _pc)
        draw_cables(meshcat, plant, _pc, manipulator, _rig,
                    spring_extension=_delta)

    # ── 9. Run ───────────────────────────────────────────────────────────────
    wn   = np.sqrt(args.ct_kp)
    zeta = args.ct_kd / (2.0 * wn) if wn > 0 else 0.0
    _move_info = (
        f"  move-to-start: {args.move_duration:.1f} s  then  "
        if args.move_duration > 0.0 else ""
    )
    print(colored(
        f"\n▶  SEA Cable — {label}   Motor: {args.motor}"
        f"\n   k_s = {ks} N/m   b_c = {b_c} N·s/m   ω_m = {omega_m} rad/s"
        f"\n   CT:  Kp={args.ct_kp}   Kd={args.ct_kd}   ωn={wn:.1f} rad/s   ζ={zeta:.2f}"
        f"\n   {_move_info}Looping — lap={args.duration:.1f} s  (runs until Ctrl-C)"
        f"\n   Press Ctrl-C to stop and show plots.",
        "cyan",
    ))

    _chunk         = 0.1
    _lap_prev      = 0
    _move_reported = args.move_duration <= 0.0
    try:
        while True:
            t_now = sim_ctx.get_time()
            if not _move_reported and t_now >= args.move_duration:
                _move_reported = True
                print(colored(
                    f"  ✓ Move-to-start complete at t={t_now:.2f} s — trajectory tracking begins.",
                    "green",
                ))
            _lap_now = int(max(0.0, t_now - args.move_duration) / args.duration)
            if _lap_now > _lap_prev:
                _lap_prev = _lap_now
                print(colored(f"  Lap {_lap_now} complete  (t={t_now:.1f} s)", "cyan"))
            simulator.AdvanceTo(t_now + _chunk)
            _viz_tick()
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        _elapsed_tracking = max(0.0, sim_ctx.get_time() - args.move_duration)
        _laps = int(_elapsed_tracking / args.duration)
        print(colored(
            f"\n  Simulation stopped at t={sim_ctx.get_time():.2f} s  ({_laps} full laps)."
            f"\n  Generating plots...",
            "yellow",
        ))
        signal.signal(signal.SIGINT, signal.default_int_handler)

    # ── 10. Collect logs ─────────────────────────────────────────────────────
    def _get(log):
        obj = log.FindLog(sim_ctx)
        return obj.sample_times(), obj.data()

    t_log, state_data = _get(log_state)
    _,     act_data   = _get(log_act)
    _,     diag_data  = _get(log_diag)
    _,     qdes_data  = _get(log_qdes)
    _,     ref_data   = _get(log_ref)
    _,     vel_data   = _get(log_vel)
    _,     acc_data   = _get(log_acc)

    # Truncate to shortest common sample count (CT, SEA, and plant are
    # separate LeafSystems whose loggers may record slightly different counts).
    N = min(len(t_log), state_data.shape[1], act_data.shape[1],
            diag_data.shape[1], qdes_data.shape[1],
            ref_data.shape[1], vel_data.shape[1], acc_data.shape[1])
    t_log      = t_log[:N]
    state_data = state_data[:, :N]
    act_data   = act_data[:, :N]
    diag_data  = diag_data[:, :N]
    qdes_data  = qdes_data[:, :N]
    ref_data   = ref_data[:, :N]
    vel_data   = vel_data[:, :N]
    acc_data   = acc_data[:, :N]

    # FK for actual EE position
    nq       = plant.num_positions()
    ee_x_act = np.zeros(len(t_log))
    ee_y_act = np.zeros(len(t_log))
    tmp_ctx  = plant.CreateDefaultContext()
    for k in range(len(t_log)):
        plant.SetPositionsAndVelocities(tmp_ctx, state_data[:, k])
        p = manipulator.get_end_effector_position(plant, tmp_ctx)
        ee_x_act[k] = p[0]
        ee_y_act[k] = p[1]

    L1, L2 = manipulator.ik.get_link_lengths(plant)

    return dict(
        t=t_log, state=state_data, actuation=act_data,
        diagnostics=diag_data, q_des=qdes_data,
        ref=ref_data, vel_ref=vel_data, acc_ref=acc_data,
        ee_x=ee_x_act, ee_y=ee_y_act,
        ee_x_tgt=ee_x_tgt, ee_y_tgt=ee_y_tgt,
        k_s=ks, omega_m=omega_m, b_c=b_c, label=label,
        r_p=manipulator.PULLEY_RADIUS, nq=nq,
        L1=L1, L2=L2,
    )


# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════

def plot_sea_results(data_spring: dict, data_rigid: dict = None):
    """Generate two figures:

    Figure 1  (3×3) — mirroring ComputedTorqueSimulation.plot():
        Row 0: EE position / velocity / acceleration  (actual vs ref)
        Row 1: Joint position / velocity / acceleration
        Row 2: Torques  |  Cable tensions  |  EE XY path

    Figure 2  (5×2) — SEA-specific diagnostics:
        Row 0: EE position X and Y
        Row 1: Torque desired-vs-applied  |  EE XY path
        Row 2: Motor cable vs joint side  |  Spring extension δ
        Row 3: Cable tension F_cable      |  Green/Red tensions
        Row 4: Torque tracking error      |  EE tracking error

    All y-axes are clipped to the 99th percentile to avoid initial
    transients dominating the view.
    """
    datasets = [("Spring", data_spring, "tab:blue")]
    if data_rigid is not None:
        datasets.append(("Rigid", data_rigid, "tab:orange"))

    d_s   = data_spring
    t     = d_s["t"]
    nq    = d_s["nq"]
    r_p   = d_s["r_p"]
    L1    = d_s["L1"]
    L2    = d_s["L2"]
    state = d_s["state"]
    q_des = d_s["q_des"]
    ref   = d_s["ref"]
    vel   = d_s["vel_ref"]
    acc   = d_s["acc_ref"]
    act   = d_s["actuation"]
    diag  = d_s["diagnostics"]

    tau1_des = diag[4]; tau2_des = diag[5]
    tau1_act = act[0];  tau2_act = act[1]
    T_green  = diag[6]; T_red = diag[7]

    # ── Derived signals ──────────────────────────────────────────────────
    q1_act     = state[0]; q2_act     = state[1]
    q1_dot_act = state[nq]; q2_dot_act = state[nq + 1]

    s1_act  = np.sin(q1_act);              c1_act  = np.cos(q1_act)
    s12_act = np.sin(q1_act + q2_act);     c12_act = np.cos(q1_act + q2_act)

    ee_vx_act = (-L1*s1_act - L2*s12_act)*q1_dot_act + (-L2*s12_act)*q2_dot_act
    ee_vy_act = ( L1*c1_act + L2*c12_act)*q1_dot_act + ( L2*c12_act)*q2_dot_act
    ee_ax_act = np.gradient(ee_vx_act, t)
    ee_ay_act = np.gradient(ee_vy_act, t)

    q1_ddot_act = np.gradient(q1_dot_act, t)
    q2_ddot_act = np.gradient(q2_dot_act, t)

    # Joint velocity / acceleration reference via J^{-1} at desired joints
    q1d = q_des[0]; q2d = q_des[1]
    s1  = np.sin(q1d);   c1  = np.cos(q1d)
    s12 = np.sin(q1d + q2d); c12 = np.cos(q1d + q2d)
    J_all = np.stack([
        np.stack([-L1*s1 - L2*s12, -L2*s12], axis=1),
        np.stack([ L1*c1 + L2*c12,  L2*c12], axis=1),
    ], axis=1)   # (T, 2, 2)
    q_dot_ref  = np.array([np.linalg.pinv(J_all[k]) @ vel[:, k]
                           for k in range(len(t))])   # (T, 2)
    q_ddot_ref = np.array([np.linalg.pinv(J_all[k]) @ acc[:, k]
                           for k in range(len(t))])   # (T, 2)

    wn   = np.sqrt(args.ct_kp)
    zeta = args.ct_kd / (2.0 * wn) if wn > 0 else 0.0
    ks_label = f"k_s = {d_s['k_s']:.0f} N/m"
    rid_info = f"  vs  k_s = {data_rigid['k_s']:.0f} N/m (rigid)" if data_rigid else ""

    def _pct_ylim(*arrays, pct=99.0, margin=0.15):
        all_vals = np.concatenate([a.ravel() for a in arrays])
        lo = np.percentile(all_vals, 100 - pct)
        hi = np.percentile(all_vals, pct)
        span = max(hi - lo, 1e-9)
        return lo - margin * span, hi + margin * span

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 1:  3 × 3 — Position | Velocity | Acceleration
    # ════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(3, 3, figsize=(18, 11))
    fig1.suptitle(
        f"SEA Computed Torque — {ks_label}{rid_info}   "
        f"Motor: {args.motor}   "
        f"Kp={args.ct_kp}  Kd={args.ct_kd}  "
        f"ωn={wn:.1f} rad/s  ζ={zeta:.2f}",
        fontsize=12, fontweight="bold",
    )

    # ── Row 0: End-Effector ───────────────────────────────────────────
    ax = axes1[0, 0]
    ax.plot(t, d_s["ee_x"], 'b-', lw=1.8, label='x actual')
    ax.plot(t, d_s["ee_y"], 'r-', lw=1.8, label='y actual')
    ax.plot(t, ref[0],      'b--', lw=1.5, label='x ref')
    ax.plot(t, ref[1],      'r--', lw=1.5, label='y ref')
    ax.set_title('EE Position'); ax.set_ylabel('[m]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    ax = axes1[0, 1]
    ax.plot(t, ee_vx_act, 'b-',  lw=1.8, label='ẋ actual')
    ax.plot(t, ee_vy_act, 'r-',  lw=1.8, label='ẏ actual')
    ax.plot(t, vel[0],    'b--', lw=1.5, label='ẋ ref')
    ax.plot(t, vel[1],    'r--', lw=1.5, label='ẏ ref')
    ax.set_ylim(*_pct_ylim(ee_vx_act, ee_vy_act, vel[0], vel[1]))
    ax.set_title('EE Velocity'); ax.set_ylabel('[m/s]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    ax = axes1[0, 2]
    ax.plot(t, ee_ax_act, 'b-',  lw=1.8, label='ẍ actual')
    ax.plot(t, ee_ay_act, 'r-',  lw=1.8, label='ÿ actual')
    ax.plot(t, acc[0],    'b--', lw=1.5, label='ẍ ref')
    ax.plot(t, acc[1],    'r--', lw=1.5, label='ÿ ref')
    ax.set_ylim(*_pct_ylim(ee_ax_act, ee_ay_act, acc[0], acc[1]))
    ax.set_title('EE Acceleration'); ax.set_ylabel('[m/s²]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    # ── Row 1: Joints ─────────────────────────────────────────────────
    ax = axes1[1, 0]
    ax.plot(t, np.rad2deg(state[0]),  'b-',  lw=1.8, label='q1 act')
    ax.plot(t, np.rad2deg(state[1]),  'r-',  lw=1.8, label='q2 act')
    ax.plot(t, np.rad2deg(q_des[0]),  'b--', lw=1.5, label='q1 des')
    ax.plot(t, np.rad2deg(q_des[1]),  'r--', lw=1.5, label='q2 des')
    ax.set_title('Joint Position'); ax.set_ylabel('[deg]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    ax = axes1[1, 1]
    ax.plot(t, np.rad2deg(q1_dot_act),      'b-',  lw=1.8, label='q̇1 act')
    ax.plot(t, np.rad2deg(q2_dot_act),      'r-',  lw=1.8, label='q̇2 act')
    ax.plot(t, np.rad2deg(q_dot_ref[:, 0]), 'b--', lw=1.5, label='q̇1 ref')
    ax.plot(t, np.rad2deg(q_dot_ref[:, 1]), 'r--', lw=1.5, label='q̇2 ref')
    ax.set_ylim(*_pct_ylim(np.rad2deg(q1_dot_act), np.rad2deg(q2_dot_act),
                           np.rad2deg(q_dot_ref[:, 0]), np.rad2deg(q_dot_ref[:, 1])))
    ax.set_title('Joint Velocity'); ax.set_ylabel('[deg/s]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    ax = axes1[1, 2]
    ax.plot(t, np.rad2deg(q1_ddot_act),      'b-',  lw=1.8, label='q̈1 act')
    ax.plot(t, np.rad2deg(q2_ddot_act),      'r-',  lw=1.8, label='q̈2 act')
    ax.plot(t, np.rad2deg(q_ddot_ref[:, 0]), 'b--', lw=1.5, label='q̈1 ref')
    ax.plot(t, np.rad2deg(q_ddot_ref[:, 1]), 'r--', lw=1.5, label='q̈2 ref')
    ax.set_ylim(*_pct_ylim(np.rad2deg(q1_ddot_act), np.rad2deg(q2_ddot_act),
                           np.rad2deg(q_ddot_ref[:, 0]), np.rad2deg(q_ddot_ref[:, 1])))
    ax.set_title('Joint Acceleration'); ax.set_ylabel('[deg/s²]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    # ── Row 2: Torques | Tensions | EE XY path ───────────────────────
    ax = axes1[2, 0]
    ax.plot(t, tau1_des, 'b-',  lw=1.8, label='τ1 required')
    ax.plot(t, tau2_des, 'r-',  lw=1.8, label='τ2 required')
    ax.plot(t, tau1_act, 'b--', lw=1.5, label='τ1 applied')
    ax.plot(t, tau2_act, 'r--', lw=1.5, label='τ2 applied')
    ax.axhline( args.ct_tau_max, color='k', ls=':', lw=0.8, label=f'±{args.ct_tau_max} Nm')
    ax.axhline(-args.ct_tau_max, color='k', ls=':', lw=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylim(*_pct_ylim(tau1_des, tau2_des, tau1_act, tau2_act))
    ax.set_title('Torque: required (solid) vs applied (dashed)')
    ax.set_ylabel('[Nm]'); ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    ax = axes1[2, 1]
    ax.plot(t, T_green, 'g-',  lw=1.2, label='T_green')
    ax.plot(t, T_red,   'r-',  lw=1.2, label='T_red')
    ax.plot(t, tau2_des / r_p, 'k--', lw=0.8,
            label=f'F_net=τ2/r_p  (r_p={r_p*1e3:.1f} mm)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylim(*_pct_ylim(T_green, T_red))
    ax.set_title('Cable Tensions'); ax.set_ylabel('[N]'); ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    ax = axes1[2, 2]
    ax.plot(d_s["ee_x_tgt"], d_s["ee_y_tgt"], 'k--', lw=1.0, label='Reference')
    for lbl_, d_, col_ in datasets:
        ls_ = '-' if lbl_ == 'Spring' else '--'
        lw_ = 1.3 if lbl_ == 'Spring' else 1.0
        ax.plot(d_["ee_x"], d_["ee_y"], color=col_, ls=ls_, lw=lw_, label=lbl_)
    ax.plot(d_s["ee_x"][0], d_s["ee_y"][0], 'go', ms=8, label='Start')
    ax.set_aspect('equal')
    ax.set_title('EE Path'); ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    fig1.tight_layout()

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 2:  5 × 2 — SEA-specific diagnostics
    # ════════════════════════════════════════════════════════════════════
    fig2 = plt.figure(figsize=(16, 17))
    gs   = GridSpec(5, 2, figure=fig2, hspace=0.55, wspace=0.35)
    axes2 = [[fig2.add_subplot(gs[r, c]) for c in range(2)] for r in range(5)]

    fig2.suptitle(
        "Series Elastic Actuator — Cable compliance effect on joint-2 tracking\n"
        f"{ks_label}{rid_info}   Motor: {args.motor}",
        fontsize=12, fontweight="bold",
    )

    # ── Row 0: EE tracking X and Y ──────────────────────────────────────
    for lbl, d, col in datasets:
        ls = "-" if lbl == "Spring" else "--"
        lw = 1.8 if lbl == "Spring" else 1.4
        axes2[0][0].plot(d["t"], d["ee_x"] * 1e3, color=col, ls=ls, lw=lw,
                        label=f"{lbl} actual")
        axes2[0][1].plot(d["t"], d["ee_y"] * 1e3, color=col, ls=ls, lw=lw,
                        label=f"{lbl} actual")

    axes2[0][0].plot(d_s["t"], ref[0] * 1e3, "k:", lw=1.2, label="reference")
    axes2[0][1].plot(d_s["t"], ref[1] * 1e3, "k:", lw=1.2, label="reference")
    axes2[0][0].set_title("EE Position — X"); axes2[0][0].set_ylabel("[mm]")
    axes2[0][1].set_title("EE Position — Y"); axes2[0][1].set_ylabel("[mm]")
    for ax in axes2[0]:
        ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # ── Row 1: Torque comparison | EE XY path ────────────────────────────
    axes2[1][0].plot(t, tau2_des, "r-",  lw=1.8, label="τ₂ desired (CT)")
    axes2[1][0].plot(t, tau2_act, "b-",  lw=1.8, label="τ₂ actual (spring)")
    if data_rigid is not None:
        axes2[1][0].plot(data_rigid["t"], data_rigid["actuation"][1],
                        "b--", lw=1.2, label="τ₂ actual (rigid)")
    axes2[1][0].plot(t, tau1_des, "g-",  lw=1.0, alpha=0.5, label="τ₁ desired")
    axes2[1][0].plot(t, tau1_act, "g--", lw=0.8, alpha=0.5, label="τ₁ actual")
    axes2[1][0].axhline(0, color="k", lw=0.5)
    axes2[1][0].set_ylim(*_pct_ylim(tau1_des, tau2_des, tau1_act, tau2_act))
    axes2[1][0].set_title("Torque: CT desired vs applied (spring lag visible on τ₂)")
    axes2[1][0].set_ylabel("[Nm]")
    axes2[1][0].legend(fontsize=7, ncol=2); axes2[1][0].grid(True, alpha=0.4)

    ax_xy = axes2[1][1]
    ax_xy.plot(d_s["ee_x_tgt"], d_s["ee_y_tgt"], "k--", lw=1.0, label="target")
    for lbl_, d_, col_ in datasets:
        ls_ = "-" if lbl_ == "Spring" else "--"
        lw_ = 1.8 if lbl_ == "Spring" else 1.2
        ax_xy.plot(d_["ee_x"], d_["ee_y"], color=col_, ls=ls_, lw=lw_, label=lbl_)
    ax_xy.plot(d_s["ee_x"][0], d_s["ee_y"][0], "go", ms=8)
    ax_xy.set_aspect("equal")
    ax_xy.set_title("EE XY Path"); ax_xy.set_xlabel("X [m]"); ax_xy.set_ylabel("Y [m]")
    ax_xy.legend(fontsize=7); ax_xy.grid(True, alpha=0.4)

    # ── Row 2: Motor cable vs joint side | Spring extension δ ───────────
    l_m     = diag[0]
    l_m_des = diag[1]
    delta   = diag[2]
    F_cable = diag[3]
    q2_rp   = l_m - delta

    axes2[2][0].plot(t, l_m     * 1e3, "b-",  lw=1.5, label="l_m (motor cable) [mm]")
    axes2[2][0].plot(t, l_m_des * 1e3, "b--", lw=1.2, label="l_m_des [mm]")
    axes2[2][0].plot(t, q2_rp   * 1e3, "r-",  lw=1.5, label="r_p·q₂ (joint side) [mm]")
    axes2[2][0].set_ylim(*_pct_ylim(l_m * 1e3, l_m_des * 1e3, q2_rp * 1e3))
    axes2[2][0].set_title(
        "Motor cable l_m vs joint side r_p·q₂\n"
        "(gap = spring extension δ)"
    )
    axes2[2][0].set_ylabel("[mm]")
    axes2[2][0].legend(fontsize=7); axes2[2][0].grid(True, alpha=0.4)

    ax_d = axes2[2][1]
    ax_d.plot(t, delta * 1e3, color="purple", lw=1.5, label="δ = spring extension [mm]")
    ax_d.axhline(0, color="k", lw=0.5)
    ax_d.set_ylim(*_pct_ylim(delta * 1e3))
    ax_d.set_title(
        f"Spring extension δ\n"
        f"k_s = {d_s['k_s']:.0f} N/m"
    )
    ax_d.set_ylabel("δ [mm]")
    ax_d.legend(fontsize=7); ax_d.grid(True, alpha=0.4)

    # ── Row 3: Cable tension (separate) | Cable tension breakdown ────────
    ax_ft = axes2[3][0]
    ax_ft.plot(t, F_cable, color="orange", lw=1.5, label="F_cable [N]")
    ax_ft.axhline(0, color="k", lw=0.5)
    ax_ft.set_ylim(*_pct_ylim(F_cable))
    ax_ft.set_title(
        f"Cable tension  F = k_s·δ + b_c·δ̇\n"
        f"k_s = {d_s['k_s']:.0f} N/m  (cable can only pull)"
    )
    ax_ft.set_ylabel("F [N]")
    ax_ft.legend(fontsize=7); ax_ft.grid(True, alpha=0.4)

    ax_tens = axes2[3][1]
    ax_tens.plot(t, T_green, "g-", lw=1.2, label="T_green (retract)")
    ax_tens.plot(t, T_red,   "r-", lw=1.2, label="T_red (extend)")
    ax_tens.axhline(0, color="k", lw=0.5)
    ax_tens.set_ylim(*_pct_ylim(T_green, T_red))
    ax_tens.set_title("Green / Red cable tensions")
    ax_tens.set_ylabel("[N]")
    ax_tens.legend(fontsize=7); ax_tens.grid(True, alpha=0.4)

    # ── Row 4: Torque tracking error | EE tracking error ─────────────────
    tau_err = tau2_des - tau2_act
    axes2[4][0].plot(t, tau2_des, "r-",  lw=1.2, label="τ₂ desired")
    axes2[4][0].plot(t, tau2_act, "b-",  lw=1.2, label="τ₂ actual")
    axes2[4][0].fill_between(
        t, tau2_des, tau2_act,
        where=(np.abs(tau_err) > 0.01),
        alpha=0.25, color="red", label="|error|",
    )
    axes2[4][0].axhline(0, color="k", lw=0.5)
    axes2[4][0].set_ylim(*_pct_ylim(tau2_des, tau2_act))
    rms_err = np.sqrt(np.mean(tau_err ** 2))
    axes2[4][0].set_title(
        f"τ₂ tracking error (shaded = lag)   RMS = {rms_err:.3f} Nm"
    )
    axes2[4][0].set_ylabel("[Nm]"); axes2[4][0].set_xlabel("Time [s]")
    axes2[4][0].legend(fontsize=7); axes2[4][0].grid(True, alpha=0.4)

    for lbl_, d_, col_ in datasets:
        t_d = d_["t"]
        ee_err = np.hypot(
            d_["ee_x"] - np.interp(t_d, d_["t"], d_["ref"][0]),
            d_["ee_y"] - np.interp(t_d, d_["t"], d_["ref"][1]),
        ) * 1e3
        ls_ = "-" if lbl_ == "Spring" else "--"
        axes2[4][1].plot(t_d, ee_err, color=col_, ls=ls_, lw=1.4,
                        label=f"{lbl_} k_s={d_['k_s']:.0f}")
    axes2[4][1].set_ylim(*_pct_ylim(ee_err))
    axes2[4][1].set_title("EE tracking error |p_act − p_ref|")
    axes2[4][1].set_ylabel("[mm]"); axes2[4][1].set_xlabel("Time [s]")
    axes2[4][1].legend(fontsize=7); axes2[4][1].grid(True, alpha=0.4)

    fig2.tight_layout(rect=[0, 0, 1, 0.95])

    # ── Save both figures ────────────────────────────────────────────────
    import time as _time
    _stamp    = _time.strftime("%Y%m%d_%H%M%S")
    _plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(_plot_dir, exist_ok=True)
    _f1 = os.path.join(
        _plot_dir, f"sea_ct_3x3_ks{int(d_s['k_s'])}_{_stamp}.png",
    )
    _f2 = os.path.join(
        _plot_dir, f"sea_cable_ks{int(d_s['k_s'])}_{_stamp}.png",
    )
    fig1.savefig(_f1, dpi=150, bbox_inches="tight")
    fig2.savefig(_f2, dpi=150, bbox_inches="tight")
    print(colored(f"\n  📊 Figures saved:\n     {_f1}\n     {_f2}", "green"))
    try:
        plt.show(block=True)
    except Exception as _e:
        print(colored(f"  ⚠ plt.show() failed ({_e}) — open the saved PNG files above.", "yellow"))


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════

def main():
    meshcat = None if args.no_meshcat else StartMeshcat()
    if meshcat is not None:
        print(colored(f"  Meshcat: {meshcat.web_url()}", "green"))

    data_spring = run_simulation(
        ks=args.spring_stiffness,
        b_c=args.cable_damping,
        omega_m=args.motor_bandwidth,
        label=f"Spring  k_s={args.spring_stiffness} N/m",
        meshcat=meshcat,
        motor_mode=_motor_mode,
    )

    data_rigid = None
    if args.compare:
        data_rigid = run_simulation(
            ks=args.compare_rigid_ks,
            b_c=args.cable_damping,
            omega_m=args.motor_bandwidth * 5,   # faster motor for "rigid"
            label=f"Rigid  k_s={args.compare_rigid_ks} N/m",
            meshcat=meshcat,
            motor_mode=_motor_mode,
        )

    plot_sea_results(data_spring, data_rigid)


if __name__ == "__main__":
    main()
