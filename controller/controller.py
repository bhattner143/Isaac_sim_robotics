"""
controller/controller.py
========================
Drake LeafSystem controllers for the cable-driven manipulator.

Classes
-------
ComputedTorqueController
    Computed-torque (inverse-dynamics) controller for CupManipulatorTendon.
SEACableController
    Monolithic CT + first-order series-elastic cable model for joint 2.
SEACableActuator
    Standalone SEA actuator model (sits between any controller and the plant).
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from pydrake.all import (
    MultibodyPlant,
    LeafSystem,
    BasicVector,
)
from pydrake.multibody.tree import MultibodyForces

if TYPE_CHECKING:
    from robots.cup_manipulator_tendon import CupManipulatorTendon


# ════════════════════════════════════════════════════════════════════════════
# ComputedTorqueController
# ════════════════════════════════════════════════════════════════════════════

class ComputedTorqueController(LeafSystem):
    """Computed-torque (inverse-dynamics) controller for CupManipulatorTendon.

    Feedback-linearizes the 2-DOF planar manipulator by cancelling nonlinear
    dynamics (Coriolis + gravity) and commanding a desired joint-space
    acceleration obtained from a PD law in joint space:

        a_des = Kp · (q_des − q) − Kd · q̇         [rad/s²]
        τ     = M(q) · a_des + h(q, q̇)             [Nm]

    where  h(q, q̇)  is the bias term (Coriolis + gravity) computed via
    Drake's CalcInverseDynamics with vdot=0:

        h(q, q̇)  =  CalcInverseDynamics(ctx, vdot=0, forces_with_gravity)

    Because the system is feedback-linearized, the closed-loop error dynamics
    are decoupled linear 2nd-order systems:

        ë + Kd · ė + Kp · e = 0   →   poles at  −ζωn ± ωn√(ζ²−1)

    Gains are dimensionless in acceleration space:
        Kp  [s⁻²]  →  ωn = √Kp        (e.g. Kp=400 → ωn=20 rad/s)
        Kd  [s⁻¹]  →  ζ  = Kd/(2√Kp)  (e.g. Kd=40  → ζ=1, critically damped)

    Joint-2 torque is passed through cable tension decomposition
    (identical to the IK-diagram PD controller) so the two non-negative
    cable tensions are physically meaningful:

        F_net   = τ2 / r_p                   [N]
        T_green = max( F_net, 0)              retracting side
        T_red   = max(−F_net, 0)             extending side
        τ2_cmd  = (T_green − T_red) · r_p    [Nm]   (= τ2, but logged separately)

    Input ports
    -----------
    ``desired_ee_pos``  [2]   target (x, y) in world frame [m]
    ``plant_state``     [n]   from ``plant.get_state_output_port()``

    Output ports
    ------------
    ``actuation``       [2]   joint torques  [Nm]  → plant actuation input
    ``joint_positions`` [2]   IK solution  [q1_des, q2_des]  [rad]
    ``torques_raw``     [2]   pre-clip joint torques from inverse dynamics  [Nm]
    ``cable_tensions``  [2]   [T_green, T_red]  [N]
    """

    def __init__(
        self,
        plant: MultibodyPlant,
        manipulator: "CupManipulatorTendon",
        Kp: float   = 400.0,   # position gain  [s⁻²]  → ωn = 20 rad/s
        Kd: float   = 40.0,    # velocity gain  [s⁻¹]  → ζ = 1 (critically damped)
        tau_max: float = 10.0, # torque saturation  [Nm]
    ) -> None:
        super().__init__()
        self._plant    = plant
        self._manip    = manipulator
        self._ik       = manipulator.ik
        self._r_p      = manipulator.PULLEY_RADIUS
        self._Kp       = float(Kp)
        self._Kd       = float(Kd)
        self._tau_max  = float(tau_max)

        # Private plant context for kinematics / inverse-dynamics queries.
        # Never used for integration — only for FK and CalcInverseDynamics.
        self._plant_ctx = plant.CreateDefaultContext()

        # Pre-allocated MultibodyForces: reused every timestep to avoid GC churn.
        self._forces = MultibodyForces(plant)

        # Pre-cache link lengths (constant URDF geometry).
        self._L1, self._L2 = manipulator.ik.get_link_lengths(plant)

        # Persistent IK seed — keeps IK on the same elbow branch across steps.
        self._last_q_des = np.zeros(2)

        # Velocity-vector indices for user-order joints [q1, q2] in Drake's nv-vector.
        j1 = manipulator.get_joint_by_name(plant, manipulator.JT1_NAME)
        j2 = manipulator.get_joint_by_name(plant, manipulator.JT2_NAME)
        self._v_idx = [j1.velocity_start(), j2.velocity_start()]

        # Per-timestep cache: keyed on context time, avoids 3× redundant IDyn solve.
        self._t_cache        = -np.inf
        self._q_des_cache    = np.zeros(2)
        self._q_cache        = np.zeros(2)
        self._q_dot_cache    = np.zeros(2)
        self._tau_clip_cache = np.zeros(2)  # clipped actuation output
        self._tau_raw_cache  = np.zeros(2)  # pre-clip inverse-dynamics torques
        self._tens_cache     = np.zeros(2)  # [T_green, T_red]

        nstate = plant.num_multibody_states()
        self._ee_port    = self.DeclareVectorInputPort("desired_ee_pos", 2)
        self._vel_port   = self.DeclareVectorInputPort("ee_vel_ref",     2)  # EE velocity ref  [m/s]
        self._acc_port   = self.DeclareVectorInputPort("ee_acc_ref",     2)  # EE accel ref     [m/s²]
        self._state_port = self.DeclareVectorInputPort("plant_state", nstate)

        self.DeclareVectorOutputPort("actuation",       2, self._calc_actuation)
        self.DeclareVectorOutputPort("joint_positions", 2, self._calc_joint_positions)
        self.DeclareVectorOutputPort("torques_raw",     2, self._calc_torques_raw)
        self.DeclareVectorOutputPort("cable_tensions",  2, self._calc_cable_tensions)

    # ── Per-timestep solve (cached) ───────────────────────────────────────────

    def _solve(self, context):
        """IK + inverse-dynamics torque, cached per simulation timestep.

        Returns
        -------
        q_des    : ndarray (2,) — desired joints [q1, q2]  [rad]
        q        : ndarray (2,) — current joints
        q_dot    : ndarray (2,) — current joint velocities
        tau_clip : ndarray (2,) — clipped actuation torques  [Nm]
        tau_raw  : ndarray (2,) — pre-clip inverse-dynamics torques  [Nm]
        tens     : ndarray (2,) — [T_green, T_red]  [N]
        """
        t = context.get_time()
        if t == self._t_cache:
            return (self._q_des_cache, self._q_cache, self._q_dot_cache,
                    self._tau_clip_cache, self._tau_raw_cache, self._tens_cache)

        # ── Read ports ──────────────────────────────────────────────────────
        state  = self._state_port.Eval(context)
        ee_des = self._ee_port.Eval(context)

        # Sync internal plant context
        self._plant.SetPositionsAndVelocities(self._plant_ctx, state)
        q     = self._manip.get_positions_user_order(self._plant, self._plant_ctx)
        q_dot = self._manip.get_velocities_user_order(self._plant, self._plant_ctx)

        # ── Analytical 2R IK (warm-started from last solution) ──────────────
        seed   = self._last_q_des if np.any(self._last_q_des != 0.0) else q
        best_q, ok = self._ik._solve_2r_core(self._L1, self._L2, ee_des, seed)
        if ok:
            self._last_q_des = best_q.copy()
            q_des = best_q
        else:
            q_des = self._last_q_des.copy()  # hold last known good

        # ── Feedforward: reference joint velocity & acceleration ────────────
        ee_vel_ref = self._vel_port.Eval(context)   # ẋ_ref  [m/s]
        ee_acc_ref = self._acc_port.Eval(context)   # ẍ_ref  [m/s²]
        q1d, q2d   = q_des
        s1  = np.sin(q1d);          c1  = np.cos(q1d)
        s12 = np.sin(q1d + q2d);    c12 = np.cos(q1d + q2d)
        J = np.array([
            [-self._L1 * s1 - self._L2 * s12, -self._L2 * s12],
            [ self._L1 * c1 + self._L2 * c12,  self._L2 * c12],
        ])
        J_inv      = np.linalg.pinv(J)          # well-conditioned away from singularities
        q_dot_ref  = J_inv @ ee_vel_ref          # [rad/s]
        q_ddot_ref = J_inv @ ee_acc_ref          # [rad/s²]

        # ── Full CT law with feedforward ─────────────────────────────────────
        a_des_user = q_ddot_ref + self._Kp * (q_des - q) + self._Kd * (q_dot_ref - q_dot)

        # Map to Drake velocity-vector order
        nv = self._plant.num_velocities()
        a_des_drake = np.zeros(nv)
        a_des_drake[self._v_idx[0]] = a_des_user[0]
        a_des_drake[self._v_idx[1]] = a_des_user[1]

        # ── Computed torque: τ = M(q)·a_des + h(q, q̇) ────────────────────
        self._forces.SetZero()
        tau_full = self._plant.CalcInverseDynamics(
            self._plant_ctx, a_des_drake, self._forces
        )

        tau1 = float(tau_full[self._v_idx[0]])
        tau2 = float(tau_full[self._v_idx[1]])

        # ── Joint-2 cable tension decomposition ─────────────────────────────
        F_net   = tau2 / self._r_p
        T_green = max(F_net,  0.0)   # retracting side tightens
        T_red   = max(-F_net, 0.0)   # extending side tightens on reversal
        tau2_cmd = (T_green - T_red) * self._r_p   # = tau2 (identity; logged separately)

        tau_raw  = np.array([tau1, tau2])
        tau_clip = np.clip(np.array([tau1, tau2_cmd]), -self._tau_max, self._tau_max)
        tens     = np.array([T_green, T_red])

        # Cache all results for this timestep
        self._t_cache        = t
        self._q_des_cache    = q_des
        self._q_cache        = q
        self._q_dot_cache    = q_dot
        self._tau_clip_cache = tau_clip
        self._tau_raw_cache  = tau_raw
        self._tens_cache     = tens

        return q_des, q, q_dot, tau_clip, tau_raw, tens

    # ── Output port callbacks ─────────────────────────────────────────────────

    def _calc_actuation(self, context, output):
        _, _, _, tau_clip, _, _ = self._solve(context)
        output.SetFromVector(tau_clip)

    def _calc_joint_positions(self, context, output):
        q_des, _, _, _, _, _ = self._solve(context)
        output.SetFromVector(q_des)

    def _calc_torques_raw(self, context, output):
        _, _, _, _, tau_raw, _ = self._solve(context)
        output.SetFromVector(tau_raw)

    def _calc_cable_tensions(self, context, output):
        _, _, _, _, _, tens = self._solve(context)
        output.SetFromVector(tens)


# ════════════════════════════════════════════════════════════════════════════
# SEACableController  —  monolithic CT + SEA (backward compat)
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
        dt:          float = 0.002,
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
        j1 = manipulator.get_joint_by_name(plant, manipulator.JT1_NAME)
        j2 = manipulator.get_joint_by_name(plant, manipulator.JT2_NAME)
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

        Antagonistic cable model: two cables wrap the joint-2 pulley in
        opposite directions.  Only one cable is taut at any instant:
            F_raw > 0  →  green cable pulls  (T_green = F_raw, T_red = 0)
            F_raw < 0  →  red   cable pulls  (T_green = 0, T_red = |F_raw|)
        The net signed cable force F_cable = T_green − T_red = F_raw is
        transmitted to Joint 2 as τ₂ = r_p · F_cable.  This allows the
        cable drive to push AND pull the joint (via antagonistic routing).

        Returns (F_cable, delta, l_m_dot, T_green, T_red).
        """
        delta     = l_m - self._r_p * q[1]
        l_m_dot   = self._omega_m * (l_m_des - l_m)   # motor velocity
        delta_dot = l_m_dot - self._r_p * q_dot[1]
        F_raw = self._k_s * delta + self._b_c * delta_dot

        # Unilateral cable model: cables can only PULL, never push.
        # Which cable is taut depends on the sign of δ (spring extension).
        if delta > 0.0:
            # Green cable taut, red cable slack
            T_green = float(max(F_raw, 0.0))
            T_red   = 0.0
        elif delta < 0.0:
            # Red cable taut, green cable slack
            T_green = 0.0
            T_red   = float(max(-F_raw, 0.0))
        else:
            # Both cables at rest
            T_green = 0.0
            T_red   = 0.0

        F_cable = T_green - T_red   # unilateral: only one side active
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
