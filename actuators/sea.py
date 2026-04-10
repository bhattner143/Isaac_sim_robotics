"""
actuators/sea.py

Series Elastic Actuator (SEA) model for the cable-driven joint.

SEACableActuator is a pure actuator model -- it contains no control law.
It sits between any torque-output controller and the plant, modelling
the physical cable compliance on joint 2.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from pydrake.all import (
    MultibodyPlant,
    LeafSystem,
    BasicVector,
)

if TYPE_CHECKING:
    from robots.cup_manipulator_tendon import CupManipulatorTendon


# ════════════════════════════════════════════════════════════════════════════
# SEACableActuator  —  standalone actuator (composable with any controller)
# ════════════════════════════════════════════════════════════════════════════

class SEACableActuator(LeafSystem):
    """Series Elastic Actuator model for the cable-driven joint 2.

    This is a **pure actuator model** — it contains no control law.  It sits
    between any torque-output controller and the plant, modelling the physical
    cable compliance on joint 2.  Joint 1 torque passes through unchanged.

    Physical topology (joint 2 only)
    ─────────────────────────────────
        Motor drum → cable → Big Pulley → SPRING → Link 2 anchor

    State variable
    ──────────────
        l_m  [m]  — motor-side cable displacement (wound on drum)

    SEA equations
    ─────────────
        δ       = l_m − r_p · q₂                        spring extension  [m]
        l̇_m    = ω_m · (l_m_des − l_m)                  motor position servo
        l_m_des = r_p · q₂ + τ₂_des / (k_s · r_p)       steady-state inversion
        F_raw   = k_s · δ + b_c · (l̇_m − r_p · q̇₂)     spring–damper force
        T_green = max(F_raw, 0)                          retracting cable
        T_red   = max(−F_raw, 0)                         extending cable
        τ₂_out  = r_p · (T_green − T_red) = r_p · F_raw

    Diagram wiring
    ──────────────
    ::

        ComputedTorqueController ──→ SEACableActuator ──→ Plant
             (or any controller)     "tau_desired"        "actuation"
                                     "plant_state"
                                      ↓
                                   "diagnostics"

    Input ports
    ───────────
        ``tau_desired``   [2]   desired joint torques [τ₁, τ₂]  [Nm]
        ``plant_state``   [n]   from plant.get_state_output_port()

    Output ports
    ────────────
        ``actuation``     [2]   actual torques [τ₁, r_p·F_cable]  [Nm]
        ``diagnostics``   [8]   [l_m, l_m_des, δ, F_cable,
                                 τ₁_des, τ₂_des, T_green, T_red]
    """

    def __init__(
        self,
        plant:       MultibodyPlant,
        manipulator: "CupManipulatorTendon",
        k_s:         float = 200.0,
        b_c:         float = 2.0,
        omega_m:     float = 30.0,
        tau_max:     float = 10.0,
        dt:          float = 0.002,
    ):
        """
        Parameters
        ----------
        plant       : Finalized MultibodyPlant.
        manipulator : CupManipulatorTendon instance (provides PULLEY_RADIUS).
        k_s         : Cable spring stiffness  [N/m].
        b_c         : Cable dashpot damping    [N·s/m].
        omega_m     : Motor position servo bandwidth  [rad/s].
        tau_max     : Output torque saturation  [Nm].
        dt          : Discrete update period  [s].
        """
        super().__init__()
        self._plant   = plant
        self._manip   = manipulator
        self._k_s     = float(k_s)
        self._b_c     = float(b_c)
        self._omega_m = float(omega_m)
        self._tau_max = float(tau_max)
        self._dt      = float(dt)
        self._r_p     = manipulator.PULLEY_RADIUS

        # Velocity-vector indices for [q1, q2] in Drake's nv-vector
        j1 = manipulator.get_joint_by_name(plant, manipulator.JT1_NAME)
        j2 = manipulator.get_joint_by_name(plant, manipulator.JT2_NAME)
        self._v_idx = [j1.velocity_start(), j2.velocity_start()]

        # ── Discrete state: l_m ──────────────────────────────────────────────
        self._l_m_idx = self.DeclareDiscreteState(1)
        self.DeclarePeriodicDiscreteUpdateEvent(dt, 0.0, self._update_motor)

        # ── Ports ────────────────────────────────────────────────────────────
        nstate = plant.num_multibody_states()
        self._tau_port   = self.DeclareVectorInputPort("tau_desired",  2)
        self._state_port = self.DeclareVectorInputPort("plant_state",  nstate)

        self.DeclareVectorOutputPort("actuation",   2, self._calc_actuation)
        self.DeclareVectorOutputPort("diagnostics", 8, self._calc_diagnostics)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _read_joint_state(self, context):
        """Extract [q1, q2] and [q1_dot, q2_dot] from plant_state port."""
        state = self._state_port.Eval(context)
        nq = self._plant.num_positions()
        q_all  = state[:nq]
        v_all  = state[nq:]
        q     = np.array([q_all[self._v_idx[0]], q_all[self._v_idx[1]]])
        q_dot = np.array([v_all[self._v_idx[0]], v_all[self._v_idx[1]]])
        return q, q_dot

    def _spring_force(self, l_m, l_m_des, q2, q2_dot):
        """Compute cable force, spring extension, and motor velocity.

        Antagonistic cable model:
            F_raw > 0  →  green cable taut  (T_green = F_raw, T_red = 0)
            F_raw < 0  →  red cable taut    (T_green = 0, T_red = |F_raw|)
            F_cable = T_green − T_red = F_raw   (bidirectional via antagonism)

        Returns (F_cable, delta, l_m_dot, T_green, T_red).
        """
        delta     = l_m - self._r_p * q2
        l_m_dot   = self._omega_m * (l_m_des - l_m)
        delta_dot = l_m_dot - self._r_p * q2_dot
        F_raw     = self._k_s * delta + self._b_c * delta_dot
        T_green   = float(max(F_raw,  0.0))
        T_red     = float(max(-F_raw, 0.0))
        F_cable   = T_green - T_red
        return F_cable, delta, l_m_dot, T_green, T_red

    def _compute_l_m_des(self, tau2_des, q2):
        """Steady-state spring inversion: l_m_des = r_p·q₂ + τ₂/(k_s·r_p)."""
        return self._r_p * q2 + tau2_des / (self._k_s * self._r_p)

    # ── Discrete update: first-order motor position servo ─────────────────────

    def _update_motor(self, context, discrete_state):
        """Euler-step: l_m ← l_m + dt · ω_m · (l_m_des − l_m)."""
        l_m     = context.get_discrete_state(self._l_m_idx).value()[0]
        tau_des = self._tau_port.Eval(context)
        q, _    = self._read_joint_state(context)
        l_m_des = self._compute_l_m_des(tau_des[1], q[1])
        l_m_new = l_m + self._dt * self._omega_m * (l_m_des - l_m)
        discrete_state.get_mutable_vector(self._l_m_idx).SetFromVector(
            np.array([l_m_new]),
        )

    # ── Output port callbacks ─────────────────────────────────────────────────

    def _calc_actuation(self, context, output):
        tau_des = self._tau_port.Eval(context)
        l_m     = context.get_discrete_state(self._l_m_idx).value()[0]
        q, q_dot = self._read_joint_state(context)
        l_m_des  = self._compute_l_m_des(tau_des[1], q[1])
        F_cable, _, _, _, _ = self._spring_force(l_m, l_m_des, q[1], q_dot[1])
        tau_out = np.array([
            tau_des[0],              # J1: pass-through (rigid)
            self._r_p * F_cable,     # J2: SEA cable spring
        ])
        output.SetFromVector(np.clip(tau_out, -self._tau_max, self._tau_max))

    def _calc_diagnostics(self, context, output):
        tau_des = self._tau_port.Eval(context)
        l_m     = context.get_discrete_state(self._l_m_idx).value()[0]
        q, q_dot = self._read_joint_state(context)
        l_m_des  = self._compute_l_m_des(tau_des[1], q[1])
        F_cable, delta, _, T_green, T_red = self._spring_force(
            l_m, l_m_des, q[1], q_dot[1],
        )
        output.SetFromVector(np.array([
            l_m,          # [0]  motor cable displacement       [m]
            l_m_des,      # [1]  desired motor cable position   [m]
            delta,        # [2]  spring extension δ             [m]
            F_cable,      # [3]  net cable force                [N]
            tau_des[0],   # [4]  desired τ₁ (pass-through)     [Nm]
            tau_des[1],   # [5]  desired τ₂ (before spring)    [Nm]
            T_green,      # [6]  retracting cable tension       [N]
            T_red,        # [7]  extending cable tension        [N]
        ]))

    def initialize_spring_at_rest(self, context, q2_init: float) -> None:
        """Set l_m = r_p · q₂ so the spring starts with δ = 0 (no pre-load).

        Call this on the SEACableActuator's own context from the diagram,
        before calling simulator.Initialize().

        Parameters
        ----------
        context  : The SEACableActuator's mutable context from the diagram.
        q2_init  : Initial joint-2 angle [rad] (user-order).
        """
        l_m_init = self._r_p * q2_init
        context.get_mutable_discrete_state(self._l_m_idx).SetFromVector(
            np.array([l_m_init]),
        )
