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
from pydrake.multibody.tree import MultibodyForces, RevoluteSpring

sys.path.insert(0, str(Path(__file__).parent))
from robots.cup_manipulator_tendon import (
    CupManipulatorTendon,
    create_cable_manipulator_config,
)
from project_utils.viz_cables import draw_cables

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

_sea = parser.add_argument_group("SEA cable model  (joint 2)")
_sea.add_argument("--spring-stiffness", type=float, default=200.0,
                  metavar="K_S",
                  help="Cable spring stiffness k_s [N/m].  Lower → more lag.")
_sea.add_argument("--cable-damping",    type=float, default=2.0,
                  metavar="B_C",
                  help="Cable dashpot damping b_c [N·s/m]")
_sea.add_argument("--motor-bandwidth",  type=float, default=30.0,
                  metavar="W_M",
                  help="Motor position servo bandwidth ω_m [rad/s].")
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
_ct.add_argument("--ct-tau-max", type=float, default=10.0,
                 help="Torque saturation [Nm]")

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


# ════════════════════════════════════════════════════════════════════════════
# SEACableController — LeafSystem
# ════════════════════════════════════════════════════════════════════════════

class SEACableController(LeafSystem):
    r"""Computed-torque outer-loop + first-order series-elastic cable for joint 2.

    Topology
    ────────
    Joint 1 (shoulder): CT inverse dynamics → τ₁ applied directly (rigid).
    Joint 2 (elbow):
        CT inverse dynamics → τ₂_des
            l_m_des  = r_p·q₂ + τ₂_des / (k_s·r_p)      ← steady-state inversion
            dl_m/dt  = ω_m · (l_m_des − l_m)              ← first-order motor servo
            δ        = l_m − r_p·q₂                        ← spring extension  [m]
            F_cable  = max(k_s·δ + b_c·(l̇_m − r_p·q̇₂), 0) ← cable force (pull-only)
            τ₂       = r_p · F_cable                       ← applied joint torque

    Discrete state
    ───────────────
        l_m  [m]  — motor-side cable displacement (wound on drum)

    Input ports
    ────────────
        desired_ee_pos  [2]   — reference EE position  [m]
        ee_vel_ref      [2]   — reference EE velocity  [m/s]
        ee_acc_ref      [2]   — reference EE acceleration [m/s²]
        plant_state     [n]   — from plant.get_state_output_port()

    Output ports
    ─────────────
        actuation     [2]   — [τ₁, τ₂] → plant.get_actuation_input_port()
        diagnostics   [8]   — [l_m, l_m_des, δ, F_cable, τ₁_des, τ₂_des, T_green, T_red]
        joint_positions [2] — [q₁_des, q₂_des]  from IK
    """

    def __init__(
        self,
        plant:       MultibodyPlant,
        manipulator: CupManipulatorTendon,
        k_s:         float = 200.0,
        b_c:         float = 2.0,
        omega_m:     float = 30.0,
        Kp:          float = 10000.0,
        Kd:          float = 400.0,
        tau_max:     float = 10.0,
        dt:          float = _DT,
    ):
        super().__init__()
        self._plant    = plant
        self._manip    = manipulator
        self._k_s      = float(k_s)
        self._b_c      = float(b_c)
        self._omega_m  = float(omega_m)
        self._Kp       = float(Kp)
        self._Kd       = float(Kd)
        self._tau_max  = float(tau_max)
        self._dt       = float(dt)
        self._r_p      = manipulator.PULLEY_RADIUS

        # Link lengths (constant URDF geometry)
        self._L1, self._L2 = manipulator.ik.get_link_lengths(plant)

        # Velocity-vector indices for [q1, q2] in Drake's nv-vector
        j1 = manipulator.get_joint_by_name(plant, CupManipulatorTendon.JT1_NAME)
        j2 = manipulator.get_joint_by_name(plant, CupManipulatorTendon.JT2_NAME)
        self._v_idx = [j1.velocity_start(), j2.velocity_start()]
        self._nv    = plant.num_velocities()

        # Internal plant context — for CalcInverseDynamics only, never integrated
        self._plant_ctx = plant.CreateDefaultContext()
        self._forces    = MultibodyForces(plant)

        # IK warm-start seed
        self._last_q_des = np.zeros(2)

        # Per-timestep cache (keyed on context time)
        self._t_cache = -np.inf
        self._cache   = None  # (tau_des, l_m_des, q, q_dot, q_des)

        # ── Discrete state: l_m ──────────────────────────────────────────────
        self._l_m_idx = self.DeclareDiscreteState(1)
        self.DeclarePeriodicDiscreteUpdateEvent(dt, 0.0, self._update_motor)

        # ── Ports ────────────────────────────────────────────────────────────
        nstate = plant.num_multibody_states()
        self._ee_port  = self.DeclareVectorInputPort("desired_ee_pos", 2)
        self._vel_port = self.DeclareVectorInputPort("ee_vel_ref",     2)
        self._acc_port = self.DeclareVectorInputPort("ee_acc_ref",     2)
        self._st_port  = self.DeclareVectorInputPort("plant_state",    nstate)

        self.DeclareVectorOutputPort("actuation",       2, self._calc_actuation)
        self.DeclareVectorOutputPort("diagnostics",     8, self._calc_diagnostics)
        self.DeclareVectorOutputPort("joint_positions", 2, self._calc_joint_positions)

    # ── Per-timestep CT + motor-target solve (cached) ────────────────────────

    def _solve(self, context):
        """IK → feedforward CT → motor target.  Cached per timestep."""
        t = context.get_time()
        if t == self._t_cache and self._cache is not None:
            return self._cache

        state  = self._st_port.Eval(context)
        ee_des = self._ee_port.Eval(context)
        ee_vel = self._vel_port.Eval(context)
        ee_acc = self._acc_port.Eval(context)

        # Sync internal plant context
        self._plant.SetPositionsAndVelocities(self._plant_ctx, state)
        q     = self._manip.get_positions_user_order(self._plant, self._plant_ctx)
        q_dot = self._manip.get_velocities_user_order(self._plant, self._plant_ctx)

        # Analytical 2R IK (warm-started from last solution)
        seed = self._last_q_des if np.any(self._last_q_des != 0) else q
        q_des, ok = self._manip.ik._solve_2r_core(self._L1, self._L2, ee_des, seed)
        if ok:
            self._last_q_des = q_des.copy()
        else:
            q_des = self._last_q_des.copy()

        # Feedforward via analytical 2R Jacobian at q_des
        c1  = np.cos(q_des[0]);           s1  = np.sin(q_des[0])
        c12 = np.cos(q_des[0]+q_des[1]);  s12 = np.sin(q_des[0]+q_des[1])
        J = np.array([
            [-self._L1*s1 - self._L2*s12, -self._L2*s12],
            [ self._L1*c1 + self._L2*c12,  self._L2*c12],
        ])
        J_inv      = np.linalg.pinv(J)
        q_dot_ref  = J_inv @ ee_vel
        q_ddot_ref = J_inv @ ee_acc

        # PD + feedforward desired acceleration
        a_des_user = (q_ddot_ref
                      + self._Kp * (q_des - q)
                      + self._Kd * (q_dot_ref - q_dot))

        # Map to Drake nv-vector order
        vdot_des = np.zeros(self._nv)
        vdot_des[self._v_idx[0]] = a_des_user[0]
        vdot_des[self._v_idx[1]] = a_des_user[1]

        # Computed torque: τ = M·a + C·v + g
        self._forces.SetZero()
        tau_full = self._plant.CalcInverseDynamics(
            self._plant_ctx, vdot_des, self._forces,
        )
        tau_des = np.array([tau_full[self._v_idx[0]], tau_full[self._v_idx[1]]])

        # Motor target cable position  (steady-state spring inversion)
        #   τ₂ = k_s · r_p · δ  →  δ_ss = τ₂_des / (k_s · r_p)
        #   l_m_des = r_p · q₂ + δ_ss
        l_m_des = self._r_p * q[1] + tau_des[1] / (self._k_s * self._r_p)

        self._t_cache = t
        self._cache   = (tau_des, l_m_des, q, q_dot, q_des)
        return self._cache

    def _spring_force(self, l_m, l_m_des, q, q_dot):
        """Compute cable force F, spring extension δ, and motor velocity l̇_m.

        Returns (F_cable, delta, l_m_dot, T_green, T_red).
        """
        delta     = l_m - self._r_p * q[1]
        l_m_dot   = self._omega_m * (l_m_des - l_m)   # motor velocity
        delta_dot = l_m_dot - self._r_p * q_dot[1]
        F_raw = self._k_s * delta + self._b_c * delta_dot
        # Decompose into two cable tensions (cables can only pull)
        T_green = float(max(F_raw,  0.0))   # retracting side
        T_red   = float(max(-F_raw, 0.0))   # extending side
        F_cable = T_green - T_red            # net (= F_raw clamped if both > 0)
        # Cable can only pull — net force is non-negative
        F_cable = float(max(F_raw, 0.0))
        return F_cable, delta, l_m_dot, T_green, T_red

    # ── Discrete update: first-order motor position servo ─────────────────

    def _update_motor(self, context, discrete_state):
        """Euler-step motor cable: l_m ← l_m + dt·ω_m·(l_m_des − l_m)."""
        l_m = context.get_discrete_state(self._l_m_idx).value()[0]
        _, l_m_des, _, _, _ = self._solve(context)
        l_m_new = l_m + self._dt * self._omega_m * (l_m_des - l_m)
        discrete_state.get_mutable_vector(self._l_m_idx).SetFromVector(
            np.array([l_m_new]),
        )

    # ── Output port callbacks ─────────────────────────────────────────────

    def _calc_actuation(self, context, output):
        tau_des, l_m_des, q, q_dot, _ = self._solve(context)
        l_m = context.get_discrete_state(self._l_m_idx).value()[0]
        F_cable, _, _, _, _ = self._spring_force(l_m, l_m_des, q, q_dot)
        tau_out = np.array([
            tau_des[0],                  # J1: CT direct drive (rigid)
            self._r_p * F_cable,         # J2: cable spring
        ])
        output.SetFromVector(np.clip(tau_out, -self._tau_max, self._tau_max))

    def _calc_diagnostics(self, context, output):
        tau_des, l_m_des, q, q_dot, _ = self._solve(context)
        l_m = context.get_discrete_state(self._l_m_idx).value()[0]
        F_cable, delta, _, T_green, T_red = self._spring_force(l_m, l_m_des, q, q_dot)
        output.SetFromVector(np.array([
            l_m,          # [0]  motor cable displacement       [m]
            l_m_des,      # [1]  desired motor cable position   [m]
            delta,        # [2]  spring extension δ             [m]
            F_cable,      # [3]  cable tension (net)            [N]
            tau_des[0],   # [4]  CT desired τ₁                  [Nm]
            tau_des[1],   # [5]  CT desired τ₂                  [Nm]
            T_green,      # [6]  retracting cable tension       [N]
            T_red,        # [7]  extending cable tension        [N]
        ]))

    def _calc_joint_positions(self, context, output):
        _, _, _, _, q_des = self._solve(context)
        output.SetFromVector(q_des)


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
) -> dict:
    """Build, run, and collect logs for a single SEA configuration."""
    print(colored(f"\n{'=' * 60}", "cyan"))
    print(colored(f"  SEA Simulation: {label}", "cyan"))
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

    # ── 4. Controller ────────────────────────────────────────────────────────
    ctrl = builder.AddSystem(
        SEACableController(
            plant, manipulator,
            k_s=ks, b_c=b_c, omega_m=omega_m,
            Kp=args.ct_kp, Kd=args.ct_kd, tau_max=args.ct_tau_max,
        ),
    )
    ctrl.set_name(f"SEA_ctrl_ks{ks:.0f}")

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

    # ── 5. Wire ──────────────────────────────────────────────────────────────
    builder.Connect(ee_src.get_output_port(),           ctrl.GetInputPort("desired_ee_pos"))
    builder.Connect(vel_src.get_output_port(),          ctrl.GetInputPort("ee_vel_ref"))
    builder.Connect(acc_src.get_output_port(),          ctrl.GetInputPort("ee_acc_ref"))
    builder.Connect(plant.get_state_output_port(),      ctrl.GetInputPort("plant_state"))
    builder.Connect(ctrl.GetOutputPort("actuation"),    plant.get_actuation_input_port())

    # ── 6. Loggers ───────────────────────────────────────────────────────────
    log_state = LogVectorOutput(plant.get_state_output_port(),          builder)
    log_act   = LogVectorOutput(ctrl.GetOutputPort("actuation"),        builder)
    log_diag  = LogVectorOutput(ctrl.GetOutputPort("diagnostics"),      builder)
    log_qdes  = LogVectorOutput(ctrl.GetOutputPort("joint_positions"),  builder)
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
    ctrl_ctx = ctrl.GetMyMutableContextFromRoot(sim_ctx)
    l_m_init = manipulator.PULLEY_RADIUS * q_init[1]
    ctrl_ctx.get_mutable_discrete_state(ctrl._l_m_idx).SetFromVector(
        np.array([l_m_init]),
    )

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
        manipulator.compute_tangents(plant, _pc)
        draw_cables(meshcat, plant, _pc, manipulator, _rig)

    # ── 9. Run ───────────────────────────────────────────────────────────────
    wn   = np.sqrt(args.ct_kp)
    zeta = args.ct_kd / (2.0 * wn) if wn > 0 else 0.0
    _move_info = (
        f"  move-to-start: {args.move_duration:.1f} s  then  "
        if args.move_duration > 0.0 else ""
    )
    print(colored(
        f"\n▶  SEA Cable — {label}"
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

    Figure 2  (4×2) — SEA-specific diagnostics:
        Row 0: EE position X and Y
        Row 1: Torque desired-vs-applied  |  EE XY path
        Row 2: Motor cable vs joint side  |  Spring extension + cable force
        Row 3: Torque tracking error  |  EE tracking error
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
    _tau_peak = max(np.abs(np.concatenate([tau1_des, tau2_des])).max(), args.ct_tau_max) * 1.15
    ax.set_ylim(-_tau_peak, _tau_peak)
    ax.set_title('Torque: required (solid) vs applied (dashed)')
    ax.set_ylabel('[Nm]'); ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    ax = axes1[2, 1]
    ax.plot(t, T_green, 'g-',  lw=1.2, label='T_green')
    ax.plot(t, T_red,   'r-',  lw=1.2, label='T_red')
    ax.plot(t, tau2_des / r_p, 'k--', lw=0.8,
            label=f'F_net=τ2/r_p  (r_p={r_p*1e3:.1f} mm)')
    ax.axhline(0, color='k', lw=0.5)
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
    # FIGURE 2:  4 × 2 — SEA-specific diagnostics
    # ════════════════════════════════════════════════════════════════════
    fig2 = plt.figure(figsize=(16, 14))
    gs   = GridSpec(4, 2, figure=fig2, hspace=0.48, wspace=0.35)
    axes2 = [[fig2.add_subplot(gs[r, c]) for c in range(2)] for r in range(4)]

    fig2.suptitle(
        "Series Elastic Actuator — Cable compliance effect on joint-2 tracking\n"
        f"{ks_label}{rid_info}",
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

    # ── Row 2: Motor cable vs joint side | Spring extension + cable force ──
    l_m     = diag[0]
    l_m_des = diag[1]
    delta   = diag[2]
    F_cable = diag[3]
    q2_rp   = l_m - delta

    axes2[2][0].plot(t, l_m     * 1e3, "b-",  lw=1.5, label="l_m (motor cable) [mm]")
    axes2[2][0].plot(t, l_m_des * 1e3, "b--", lw=1.2, label="l_m_des [mm]")
    axes2[2][0].plot(t, q2_rp   * 1e3, "r-",  lw=1.5, label="r_p·q₂ (joint side) [mm]")
    axes2[2][0].set_title(
        "Motor cable l_m vs joint side r_p·q₂\n"
        "(gap = spring extension δ)"
    )
    axes2[2][0].set_ylabel("[mm]")
    axes2[2][0].legend(fontsize=7); axes2[2][0].grid(True, alpha=0.4)

    ax_d = axes2[2][1]
    ax_d.plot(t, delta * 1e3, color="purple", lw=1.5, label="δ = spring extension [mm]")
    ax_d_twin = ax_d.twinx()
    ax_d_twin.plot(t, F_cable, color="orange", lw=1.5, label="F_cable [N]")
    ax_d.axhline(0, color="k", lw=0.5)
    ax_d.set_title(
        f"Spring extension δ  &  cable tension\n"
        f"k_s = {d_s['k_s']:.0f} N/m  →  F = k_s·δ + b_c·δ̇  (cable can only pull)"
    )
    ax_d.set_ylabel("δ [mm]", color="purple")
    ax_d_twin.set_ylabel("F [N]", color="orange")
    lines_d, labs_d = ax_d.get_legend_handles_labels()
    lines_t, labs_t = ax_d_twin.get_legend_handles_labels()
    ax_d.legend(lines_d + lines_t, labs_d + labs_t, fontsize=7, loc="upper right")
    ax_d.grid(True, alpha=0.4)

    # ── Row 3: Torque tracking error | EE tracking error ─────────────────
    tau_err = tau2_des - tau2_act
    axes2[3][0].plot(t, tau2_des, "r-",  lw=1.2, label="τ₂ desired")
    axes2[3][0].plot(t, tau2_act, "b-",  lw=1.2, label="τ₂ actual")
    axes2[3][0].fill_between(
        t, tau2_des, tau2_act,
        where=(np.abs(tau_err) > 0.01),
        alpha=0.25, color="red", label="|error|",
    )
    axes2[3][0].axhline(0, color="k", lw=0.5)
    rms_err = np.sqrt(np.mean(tau_err ** 2))
    axes2[3][0].set_title(
        f"τ₂ tracking error (shaded = lag)   RMS = {rms_err:.3f} Nm"
    )
    axes2[3][0].set_ylabel("[Nm]"); axes2[3][0].set_xlabel("Time [s]")
    axes2[3][0].legend(fontsize=7); axes2[3][0].grid(True, alpha=0.4)

    for lbl_, d_, col_ in datasets:
        t_d = d_["t"]
        ee_err = np.hypot(
            d_["ee_x"] - np.interp(t_d, d_["t"], d_["ref"][0]),
            d_["ee_y"] - np.interp(t_d, d_["t"], d_["ref"][1]),
        ) * 1e3
        ls_ = "-" if lbl_ == "Spring" else "--"
        axes2[3][1].plot(t_d, ee_err, color=col_, ls=ls_, lw=1.4,
                        label=f"{lbl_} k_s={d_['k_s']:.0f}")
    axes2[3][1].set_title("EE tracking error |p_act − p_ref|")
    axes2[3][1].set_ylabel("[mm]"); axes2[3][1].set_xlabel("Time [s]")
    axes2[3][1].legend(fontsize=7); axes2[3][1].grid(True, alpha=0.4)

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
    )

    data_rigid = None
    if args.compare:
        data_rigid = run_simulation(
            ks=args.compare_rigid_ks,
            b_c=args.cable_damping,
            omega_m=args.motor_bandwidth * 5,   # faster motor for "rigid"
            label=f"Rigid  k_s={args.compare_rigid_ks} N/m",
            meshcat=meshcat,
        )

    plot_sea_results(data_spring, data_rigid)


if __name__ == "__main__":
    main()
