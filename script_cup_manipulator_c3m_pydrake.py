#!/usr/bin/env python3
"""
script_cup_manipulator_c3m_pydrake.py
======================================
C3M (Control Contraction Metrics) tracking controller for the cup manipulator
with Series Elastic Actuator (SEA) cable on joint 2.

Replaces the Computed Torque controller with a trained C3M neural controller
that accounts for the spring dynamics and underactuation.

Architecture:
    Trajectory → C3M Controller → SEA Actuator → Plant

The C3M controller produces [τ₁, τ₂_des] that anticipate the motor-spring
dynamics, unlike CT which assumes instant torque authority.

USAGE
─────
  # First train the C3M controller:
  cd contraction-theory/C3M
  python main.py --task CUPMANIP_SEA --lambda 1.5 \\
      --log checkpoints/cupmanip_sea/lambda_1.5 --epochs 15

  # Then run this script:
  python script_cup_manipulator_c3m_pydrake.py

  # Compare C3M vs CT:
  python script_cup_manipulator_c3m_pydrake.py --compare-ct

  # Custom spring stiffness:
  python script_cup_manipulator_c3m_pydrake.py --spring-stiffness 100

  # Custom checkpoint:
  python script_cup_manipulator_c3m_pydrake.py \\
      --checkpoint contraction-theory/C3M/checkpoints/cupmanip_sea/lambda_1.5/controller_best.pth.tar
"""

import signal
import sys
import argparse
import platform as _platform
from pathlib import Path

import numpy as np

if _platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('macosx')
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('TkAgg')
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
from controller.c3m_controller import C3MController
from controller.controller import ComputedTorqueController
from actuators.sea import SEACableActuator
from actuators.motor_dynamics import MotorMode
from actuators.motor import get_motor, MOTOR_CHOICES

# ─── Constants ────────────────────────────────────────────────────────────────
_DT   = 0.01
_URDF = "model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf"
_M_PATCH = SpatialInertia(
    mass=0.3, p_PScm_E=np.zeros(3), G_SP_E=UnitInertia(1e-2, 1e-2, 1e-2),
)
_DEFAULT_CHECKPOINT = (
    "contraction-theory/C3M/checkpoints/cupmanip_sea/lambda_1.5/controller_best.pth.tar"
)


# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="C3M tracking controller for cup manipulator + SEA cable",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

_c3m = parser.add_argument_group("C3M controller")
_c3m.add_argument("--checkpoint", type=str, default=_DEFAULT_CHECKPOINT,
                  help="Path to trained C3M controller_best.pth.tar")
_c3m.add_argument("--compare-ct", action="store_true",
                  help="Run C3M and CT side by side, overlay plots.")

_mot = parser.add_argument_group("motor model")
_mot.add_argument("--motor", choices=MOTOR_CHOICES, default="AK60_6_KV80_Config")

_sea = parser.add_argument_group("SEA cable model")
_sea.add_argument("--spring-stiffness", type=float, default=300.0, metavar="K_S")
_sea.add_argument("--cable-damping", type=float, default=2.0, metavar="B_C")
_sea.add_argument("--sea-mode", choices=["torque", "position"], default="torque")

_ct = parser.add_argument_group("CT gains (for --compare-ct)")
_ct.add_argument("--ct-kp", type=float, default=100.0)
_ct.add_argument("--ct-kd", type=float, default=40.0)
_ct.add_argument("--ct-tau-max", type=float, default=None)

_sim = parser.add_argument_group("simulation")
_sim.add_argument("--duration", type=float, default=10.0)
_sim.add_argument("--move-duration", type=float, default=3.0)
_sim.add_argument("--no-meshcat", action="store_true")

_rob = parser.add_argument_group("robot mount")
_rob.add_argument("--tilt-roll", type=float, default=0.0)
_rob.add_argument("--tilt-pitch", type=float, default=0.0)
_rob.add_argument("--joint-damping", type=float, nargs=2, default=[0.05, 0.05])
_rob.add_argument("--joint-stiffness", type=float, nargs=2, default=[0.0, 0.0])

_traj = parser.add_argument_group("trajectory (rect)")
_traj.add_argument("--traj-x-range", type=float, nargs=2, default=[0.49, 0.51])
_traj.add_argument("--traj-y-range", type=float, nargs=2, default=[-0.08, 0.08])
_traj.add_argument("--traj-n", type=int, default=60)
_traj.add_argument("--traj-v-max", type=float, default=0.9)
_traj.add_argument("--traj-v-corner", type=float, default=0.05)
_traj.add_argument("--traj-corner-blend", type=float, default=0.35)

args = parser.parse_args()

_motor = get_motor(args.motor)
_motor_mode = MotorMode(args.sea_mode)
if args.ct_tau_max is None:
    args.ct_tau_max = _motor.peak_torque_joint

print(colored(
    f"\n  Motor: {args.motor}  —  SEA mode: {args.sea_mode}"
    f"\n    gear ratio      = {_motor.gear_ratio}"
    f"\n    peak torque     = {_motor.peak_torque_joint} Nm"
    f"\n    k_s = {args.spring_stiffness} N/m   b_c = {args.cable_damping} N·s/m"
    f"\n    C3M checkpoint  = {args.checkpoint}",
    "yellow",
))


# ════════════════════════════════════════════════════════════════════════════
# Trajectory builders (same as SEA script)
# ════════════════════════════════════════════════════════════════════════════

def _build_rect_trajectory(manip, plant, args):
    """C² rectangular EE trajectory with speed-profiled corners."""
    L1, L2 = manip.ik.get_link_lengths(plant)
    r_max = L1 + L2
    r_min = abs(L1 - L2)

    x_min, x_max = args.traj_x_range
    y_min, y_max = args.traj_y_range
    N = args.traj_n
    W = x_max - x_min
    H = y_max - y_min
    P = 2.0 * (W + H)
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

    _v_max = args.traj_v_max
    _v_corner = args.traj_v_corner
    _d_blend = args.traj_corner_blend * min(W, H)

    def _speed(s):
        t = np.clip(_corner_dist(s) / max(_d_blend, 1e-9), 0.0, 1.0)
        return _v_corner + (_v_max - _v_corner) * t * t * (3.0 - 2.0 * t)

    _s_vals = np.linspace(0.0, P, N + 1, endpoint=True)
    _speeds = np.array([_speed(s) for s in _s_vals])
    _ds = P / N
    _t_raw = np.zeros(N + 1)
    for i in range(N):
        _t_raw[i + 1] = _t_raw[i] + _ds / (0.5 * (_speeds[i] + _speeds[i + 1]))
    t_wp = _t_raw * (args.duration / _t_raw[-1])
    _xy = np.array([_s_to_xy(s) for s in _s_vals])
    ee_x = _xy[:N, 0]
    ee_y = _xy[:N, 1]

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
    traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(t_wp, wp)
    traj_vel = traj.derivative(1)
    traj_acc = traj.derivative(2)
    return traj, traj_vel, traj_acc, ee_x, ee_y


def _build_move_to_start(manip, plant, traj, traj_vel, move_duration):
    """Cubic-Hermite approach from near the first waypoint."""
    L1, L2 = manip.ik.get_link_lengths(plant)
    p_end = traj.value(0.0).ravel()
    seed = np.array([np.deg2rad(5.0), np.deg2rad(15.0)])
    q_end, ok = manip.ik._solve_2r_core(L1, L2, p_end, seed)
    if not ok:
        q_end = seed.copy()

    q_pre = q_end + np.array([np.deg2rad(-5.0), np.deg2rad(5.0)])
    tmp_ctx = plant.CreateDefaultContext()
    manip.set_positions_user_order(plant, tmp_ctx, q_pre)
    p_start = manip.get_end_effector_position(plant, tmp_ctx)[:2]
    v_end = traj_vel.value(0.0).ravel()

    t_br = np.array([0.0, move_duration])
    smp = np.column_stack([p_start, p_end])
    smp_d = np.column_stack([np.zeros(2), v_end])
    move_traj = PiecewisePolynomial.CubicHermite(t_br, smp, smp_d)
    move_traj_vel = move_traj.derivative(1)
    move_traj_acc = move_traj.derivative(2)
    return move_traj, move_traj_vel, move_traj_acc, q_end


# ════════════════════════════════════════════════════════════════════════════
# Reference state generator (LeafSystem)
# ════════════════════════════════════════════════════════════════════════════

class ReferenceStateSource(LeafSystem):
    """Generates the 6D reference state and 2D feedforward control for C3M.

    The reference state is:
        x_ref = [q1_ref, q2_ref, q̇1_ref, q̇2_ref, θ_m_ref, θ̇_m_ref]

    where θ_m_ref = N * q2_ref (spring at rest) and the joint-space
    references come from IK on the EE trajectory.

    The feedforward control u_ref is the inverse-dynamics torque at the
    reference trajectory (gravity + Coriolis compensation).
    """

    def __init__(self, plant, manipulator, gear_ratio,
                 move_traj, move_vel, move_acc,
                 main_traj, main_vel, main_acc,
                 move_duration, lap_duration):
        super().__init__()
        self._plant = plant
        self._manip = manipulator
        self._N = float(gear_ratio)
        self._move_traj = move_traj
        self._move_vel = move_vel
        self._move_acc = move_acc
        self._main_traj = main_traj
        self._main_vel = main_vel
        self._main_acc = main_acc
        self._move_dur = float(move_duration)
        self._lap_dur = float(lap_duration)

        self._L1, self._L2 = manipulator.ik.get_link_lengths(plant)
        self._last_q = np.array([np.deg2rad(5.0), np.deg2rad(15.0)])

        # Plant context for inverse dynamics
        self._plant_ctx = plant.CreateDefaultContext()
        from pydrake.multibody.tree import MultibodyForces
        self._forces = MultibodyForces(plant)
        j1 = manipulator.get_joint_by_name(plant, manipulator.JT1_NAME)
        j2 = manipulator.get_joint_by_name(plant, manipulator.JT2_NAME)
        self._v_idx = [j1.velocity_start(), j2.velocity_start()]

        self.DeclareVectorOutputPort("x_ref", 6, self._calc_x_ref)
        self.DeclareVectorOutputPort("u_ref", 2, self._calc_u_ref)
        self.DeclareVectorOutputPort("ee_pos_ref", 2, self._calc_ee_pos)
        self.DeclareVectorOutputPort("ee_vel_ref", 2, self._calc_ee_vel)
        self.DeclareVectorOutputPort("ee_acc_ref", 2, self._calc_ee_acc)

        self._t_cache = -np.inf
        self._xref_cache = np.zeros(6)
        self._uref_cache = np.zeros(2)
        self._ee_cache = np.zeros(2)
        self._ee_vel_cache = np.zeros(2)
        self._ee_acc_cache = np.zeros(2)

    def _eval_trajectory(self, t):
        """Get EE position, velocity, acceleration at time t."""
        if self._move_traj is not None and t < self._move_dur:
            return (self._move_traj.value(t).ravel(),
                    self._move_vel.value(t).ravel(),
                    self._move_acc.value(t).ravel())
        else:
            tw = max(0.0, t - self._move_dur) % self._lap_dur
            return (self._main_traj.value(tw).ravel(),
                    self._main_vel.value(tw).ravel(),
                    self._main_acc.value(tw).ravel())

    def _solve(self, context):
        t = context.get_time()
        if t == self._t_cache:
            return

        ee_pos, ee_vel, ee_acc = self._eval_trajectory(t)

        # IK: EE position → joint angles
        q_des, ok = self._manip.ik._solve_2r_core(
            self._L1, self._L2, ee_pos, self._last_q,
        )
        if ok:
            self._last_q = q_des.copy()
        else:
            q_des = self._last_q.copy()

        # Jacobian for velocity/acceleration mapping
        q1d, q2d = q_des
        s1 = np.sin(q1d);  c1 = np.cos(q1d)
        s12 = np.sin(q1d + q2d); c12 = np.cos(q1d + q2d)
        J = np.array([
            [-self._L1 * s1 - self._L2 * s12, -self._L2 * s12],
            [ self._L1 * c1 + self._L2 * c12,  self._L2 * c12],
        ])
        J_inv = np.linalg.pinv(J)
        q_dot_ref = J_inv @ ee_vel
        q1d_dot, q2d_dot = q_dot_ref
        q12d_dot = q1d_dot + q2d_dot
        Jdot = np.array([
            [-self._L1 * c1 * q1d_dot - self._L2 * c12 * q12d_dot,
             -self._L2 * c12 * q12d_dot],
            [-self._L1 * s1 * q1d_dot - self._L2 * s12 * q12d_dot,
             -self._L2 * s12 * q12d_dot],
        ])
        q_ddot_ref = J_inv @ (ee_acc - Jdot @ q_dot_ref)

        # Reference state: spring at rest → θ_m = N * q2
        theta_m_ref = self._N * q_des[1]
        theta_m_dot_ref = self._N * q_dot_ref[1]

        self._xref_cache = np.array([
            q_des[0], q_des[1],
            q_dot_ref[0], q_dot_ref[1],
            theta_m_ref, theta_m_dot_ref,
        ])

        # Feedforward: inverse dynamics at reference (gravity + Coriolis)
        nv = self._plant.num_velocities()
        nq = self._plant.num_positions()
        state_ref = np.zeros(nq + nv)
        self._manip.set_positions_user_order(self._plant, self._plant_ctx, q_des)
        v_drake = np.zeros(nv)
        v_drake[self._v_idx[0]] = q_dot_ref[0]
        v_drake[self._v_idx[1]] = q_dot_ref[1]
        self._plant.SetVelocities(self._plant_ctx, v_drake)

        a_drake = np.zeros(nv)
        a_drake[self._v_idx[0]] = q_ddot_ref[0]
        a_drake[self._v_idx[1]] = q_ddot_ref[1]

        self._forces.SetZero()
        tau_full = self._plant.CalcInverseDynamics(
            self._plant_ctx, a_drake, self._forces,
        )
        self._uref_cache = np.array([
            tau_full[self._v_idx[0]],
            tau_full[self._v_idx[1]],
        ])

        self._ee_cache = ee_pos
        self._ee_vel_cache = ee_vel
        self._ee_acc_cache = ee_acc
        self._t_cache = t

    def _calc_x_ref(self, context, output):
        self._solve(context)
        output.SetFromVector(self._xref_cache)

    def _calc_u_ref(self, context, output):
        self._solve(context)
        output.SetFromVector(self._uref_cache)

    def _calc_ee_pos(self, context, output):
        self._solve(context)
        output.SetFromVector(self._ee_cache)

    def _calc_ee_vel(self, context, output):
        self._solve(context)
        output.SetFromVector(self._ee_vel_cache)

    def _calc_ee_acc(self, context, output):
        self._solve(context)
        output.SetFromVector(self._ee_acc_cache)


# ════════════════════════════════════════════════════════════════════════════
# Single simulation run
# ════════════════════════════════════════════════════════════════════════════

def run_simulation(
    ks: float,
    b_c: float,
    label: str,
    meshcat,
    motor_mode: MotorMode,
    controller_type: str = "c3m",   # "c3m" or "ct"
) -> dict:
    """Build, run, and collect logs for a single simulation."""
    print(colored(f"\n{'=' * 60}", "cyan"))
    print(colored(f"  Simulation: {label}  [{controller_type.upper()}]", "cyan"))
    print(colored(f"  Motor: {args.motor}   k_s={ks} N/m   b_c={b_c} N·s/m", "cyan"))
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
    builder = DiagramBuilder()
    plant = MultibodyPlant(time_step=_DT)
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

    # ── 4. SEA actuator ─────────────────────────────────────────────────────
    sea = builder.AddSystem(
        SEACableActuator(
            plant, manipulator,
            k_s=ks, b_c=b_c,
            tau_max=args.ct_tau_max, dt=_DT,
            motor_mode=motor_mode,
            motor_cfg=_motor,
        ),
    )
    sea.set_name(f"SEA_ks{ks:.0f}_{motor_mode.value}")

    # ── 5. Controller + Reference source ─────────────────────────────────────
    ref_src = builder.AddSystem(ReferenceStateSource(
        plant, manipulator, _motor.gear_ratio,
        move_traj, move_vel, move_acc,
        traj, traj_vel, traj_acc,
        args.move_duration, args.duration,
    ))
    ref_src.set_name("Ref_src")

    if controller_type == "c3m":
        ctrl = builder.AddSystem(C3MController(
            plant, manipulator,
            checkpoint_path=args.checkpoint,
            tau_max=args.ct_tau_max,
        ))
        ctrl.set_name("C3M_ctrl")

        # Wire: plant_state → C3M
        # Note: uses base C3MController (motor state approximated from
        # x_ref) to avoid algebraic loop with SEA diagnostics port.
        builder.Connect(plant.get_state_output_port(), ctrl.GetInputPort("plant_state"))
        builder.Connect(ref_src.GetOutputPort("x_ref"), ctrl.GetInputPort("x_ref"))
        builder.Connect(ref_src.GetOutputPort("u_ref"), ctrl.GetInputPort("u_ref"))
        # Wire: C3M → SEA → Plant
        builder.Connect(ctrl.GetOutputPort("actuation"), sea.GetInputPort("tau_desired"))
    else:
        # CT controller (for comparison)
        ctrl = builder.AddSystem(ComputedTorqueController(
            plant, manipulator,
            Kp=args.ct_kp, Kd=args.ct_kd, tau_max=args.ct_tau_max,
        ))
        ctrl.set_name("CT_ctrl")
        builder.Connect(ref_src.GetOutputPort("ee_pos_ref"), ctrl.GetInputPort("desired_ee_pos"))
        builder.Connect(ref_src.GetOutputPort("ee_vel_ref"), ctrl.GetInputPort("ee_vel_ref"))
        builder.Connect(ref_src.GetOutputPort("ee_acc_ref"), ctrl.GetInputPort("ee_acc_ref"))
        builder.Connect(plant.get_state_output_port(), ctrl.GetInputPort("plant_state"))
        builder.Connect(ctrl.GetOutputPort("actuation"), sea.GetInputPort("tau_desired"))

    builder.Connect(plant.get_state_output_port(), sea.GetInputPort("plant_state"))
    builder.Connect(sea.GetOutputPort("actuation"), plant.get_actuation_input_port())

    # ── 6. Loggers ───────────────────────────────────────────────────────────
    log_state = LogVectorOutput(plant.get_state_output_port(), builder)
    log_act = LogVectorOutput(sea.GetOutputPort("actuation"), builder)
    log_diag = LogVectorOutput(sea.GetOutputPort("diagnostics"), builder)
    log_ref = LogVectorOutput(ref_src.GetOutputPort("ee_pos_ref"), builder)
    log_xref = LogVectorOutput(ref_src.GetOutputPort("x_ref"), builder)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    sim_ctx = simulator.get_mutable_context()

    # ── 7. Initialize ────────────────────────────────────────────────────────
    plant_ctx = plant.GetMyMutableContextFromRoot(sim_ctx)

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

    sea_ctx = sea.GetMyMutableContextFromRoot(sim_ctx)
    sea.initialize_spring_at_rest(sea_ctx, q_init[1])

    ee0 = manipulator.get_end_effector_position(plant, plant_ctx)
    print(colored(
        f"  ✓ Init: q=[{np.rad2deg(q_init[0]):.1f}°, {np.rad2deg(q_init[1]):.1f}°]  "
        f"EE=({ee0[0]*1e3:.1f}, {ee0[1]*1e3:.1f}) mm",
        "green",
    ))

    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    # ── 8. Run ───────────────────────────────────────────────────────────────
    print(colored(
        f"\n▶  {label}   [{controller_type.upper()}]"
        f"\n   k_s = {ks} N/m   b_c = {b_c} N·s/m"
        f"\n   Looping — lap={args.duration:.1f} s  (Ctrl-C to stop)",
        "cyan",
    ))

    _chunk = 0.1
    _lap_prev = 0
    _move_reported = args.move_duration <= 0.0
    try:
        while True:
            t_now = sim_ctx.get_time()
            if not _move_reported and t_now >= args.move_duration:
                _move_reported = True
                print(colored(
                    f"  ✓ Move-to-start complete at t={t_now:.2f} s",
                    "green",
                ))
            _lap_now = int(max(0.0, t_now - args.move_duration) / args.duration)
            if _lap_now > _lap_prev:
                _lap_prev = _lap_now
                print(colored(f"  Lap {_lap_now} complete  (t={t_now:.1f} s)", "cyan"))
            simulator.AdvanceTo(t_now + _chunk)
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print(colored(
            f"\n  Stopped at t={sim_ctx.get_time():.2f} s.  Generating plots...",
            "yellow",
        ))
        signal.signal(signal.SIGINT, signal.default_int_handler)

    # ── 9. Collect logs ──────────────────────────────────────────────────────
    def _get(log):
        obj = log.FindLog(sim_ctx)
        return obj.sample_times(), obj.data()

    t_log, state_data = _get(log_state)
    _, act_data = _get(log_act)
    _, diag_data = _get(log_diag)
    _, ref_data = _get(log_ref)
    _, xref_data = _get(log_xref)

    N = min(len(t_log), state_data.shape[1], act_data.shape[1],
            diag_data.shape[1], ref_data.shape[1], xref_data.shape[1])
    t_log = t_log[:N]
    state_data = state_data[:, :N]
    act_data = act_data[:, :N]
    diag_data = diag_data[:, :N]
    ref_data = ref_data[:, :N]
    xref_data = xref_data[:, :N]

    # FK for actual EE position
    nq = plant.num_positions()
    ee_x_act = np.zeros(N)
    ee_y_act = np.zeros(N)
    tmp_ctx = plant.CreateDefaultContext()
    for k in range(N):
        plant.SetPositionsAndVelocities(tmp_ctx, state_data[:, k])
        p = manipulator.get_end_effector_position(plant, tmp_ctx)
        ee_x_act[k] = p[0]
        ee_y_act[k] = p[1]

    L1, L2 = manipulator.ik.get_link_lengths(plant)

    return dict(
        t=t_log, state=state_data, actuation=act_data,
        diagnostics=diag_data, ref=ref_data, x_ref=xref_data,
        ee_x=ee_x_act, ee_y=ee_y_act,
        ee_x_tgt=ee_x_tgt, ee_y_tgt=ee_y_tgt,
        k_s=ks, b_c=b_c, label=label,
        r_p=manipulator.PULLEY_RADIUS, nq=nq,
        L1=L1, L2=L2, controller_type=controller_type,
    )


# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════

def plot_results(data_c3m: dict, data_ct: dict = None):
    """Plot EE tracking and torques.  If data_ct is given, overlay both."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    datasets = [("C3M", data_c3m, "tab:blue")]
    if data_ct is not None:
        datasets.append(("CT", data_ct, "tab:orange"))

    # ── 1. XY tracking ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(data_c3m['ee_x_tgt'], data_c3m['ee_y_tgt'], 'k--', lw=1.5, label='Target')
    for name, d, color in datasets:
        ax1.plot(d['ee_x'], d['ee_y'], color=color, alpha=0.7, label=f'{name} actual')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('End-Effector XY Tracking')
    ax1.legend(fontsize=9)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # ── 2. EE X tracking error ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for name, d, color in datasets:
        ee_ref_x = d['ref'][0, :]
        ax2.plot(d['t'], (d['ee_x'] - ee_ref_x) * 1e3, color=color, alpha=0.7, label=name)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('EE X error [mm]')
    ax2.set_title('X Tracking Error')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── 3. EE Y tracking error ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for name, d, color in datasets:
        ee_ref_y = d['ref'][1, :]
        ax3.plot(d['t'], (d['ee_y'] - ee_ref_y) * 1e3, color=color, alpha=0.7, label=name)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('EE Y error [mm]')
    ax3.set_title('Y Tracking Error')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── 4. Applied torques ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for name, d, color in datasets:
        ax4.plot(d['t'], d['actuation'][0, :], color=color, alpha=0.5, label=f'{name} τ₁')
        ax4.plot(d['t'], d['actuation'][1, :], color=color, alpha=0.8, ls='--', label=f'{name} τ₂')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Torque [Nm]')
    ax4.set_title('Applied Torques (after SEA)')
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)

    # ── 5. Spring extension ──────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    for name, d, color in datasets:
        ax5.plot(d['t'], d['diagnostics'][2, :] * 1e3, color=color, alpha=0.7, label=name)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('δ [mm]')
    ax5.set_title('Spring Extension δ')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # ── 6. RMS error summary ─────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    # Compute RMS after move-to-start
    t_start = args.move_duration
    labels = []
    rms_vals = []
    colors_bar = []
    for name, d, color in datasets:
        mask = d['t'] >= t_start
        err_x = d['ee_x'][mask] - d['ref'][0, mask]
        err_y = d['ee_y'][mask] - d['ref'][1, mask]
        rms = np.sqrt(np.mean(err_x**2 + err_y**2)) * 1e3
        labels.append(name)
        rms_vals.append(rms)
        colors_bar.append(color)
    ax6.bar(labels, rms_vals, color=colors_bar, alpha=0.7)
    ax6.set_ylabel('RMS EE error [mm]')
    ax6.set_title('Tracking RMS (after move-to-start)')
    for i, v in enumerate(rms_vals):
        ax6.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f"Cup Manipulator + SEA — k_s={data_c3m['k_s']} N/m",
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    meshcat = None if args.no_meshcat else StartMeshcat()

    if not args.no_meshcat and meshcat is not None:
        print(colored(f"  Meshcat: {meshcat.web_url()}", "magenta"))

    data_c3m = run_simulation(
        ks=args.spring_stiffness,
        b_c=args.cable_damping,
        label=f"C3M  k_s={args.spring_stiffness}",
        meshcat=meshcat,
        motor_mode=_motor_mode,
        controller_type="c3m",
    )

    data_ct = None
    if args.compare_ct:
        data_ct = run_simulation(
            ks=args.spring_stiffness,
            b_c=args.cable_damping,
            label=f"CT   k_s={args.spring_stiffness}",
            meshcat=None,  # Don't duplicate meshcat
            motor_mode=_motor_mode,
            controller_type="ct",
        )

    plot_results(data_c3m, data_ct)
