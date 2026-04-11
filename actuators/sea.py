"""
actuators/sea.py

Series Elastic Actuator (SEA) model for the cable-driven joint.

SEACableActuator is a pure actuator model -- it contains no control law.
It sits between any torque-output controller and the plant, modelling
the physical cable compliance on joint 2.

Motor dynamics are delegated to a pluggable ``MotorDynamics`` object (see
``actuators.motor_dynamics``).  Two modes are supported:

  - **torque** (default) — 2nd-order rotor dynamics driven by torque command.
    Uses CubeMars MIT torque mode.  Parameters: ``J_m``, ``b_m`` from motor
    datasheet.
  - **position** — 1st-order position servo with bandwidth ``ω_m``.  Legacy
    mode for motors running a factory position controller.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from pydrake.all import (
    MultibodyPlant,
    LeafSystem,
    BasicVector,
)

from actuators.motor_dynamics import (
    MotorDynamics,
    MotorMode,
    PositionServoMotor,
    TorqueMotor,
    create_motor_dynamics,
)

if TYPE_CHECKING:
    from actuators.motor import MotorModelConfig
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
    ::

        Motor drum → cable → Big Pulley → SPRING → Link 2 anchor

    Motor dynamics modes
    ────────────────────
    The motor model is selected via the ``motor_mode`` parameter (or by
    passing a pre-built ``MotorDynamics`` instance):

    **Torque mode** (default, ``MotorMode.TORQUE``)

        2nd-order rotor dynamics — matches CubeMars MIT torque mode::

            J_m · θ̈_m = τ_m − b_m · θ̇_m − τ_s / N

        where ``τ_m = τ₂_des / N`` and
        ``τ_s = k_s · (θ_m/N − q₂) + b_c · (θ̇_m/N − q̇₂)``.

        State: ``[θ_m, θ̇_m]`` (motor-side angle and velocity).

    **Position mode** (``MotorMode.POSITION``)

        1st-order position servo with bandwidth ``ω_m``::

            l̇_m = ω_m · (l_m_des − l_m)

        State: ``[l_m]`` (cable displacement).

    Unilateral cable model
    ──────────────────────
    ::

        Cables can only PULL (tension ≥ 0), never push:
          δ > 0  →  green taut:  T_green = max(F_raw, 0),  T_red = 0
          δ < 0  →  red taut:    T_green = 0,  T_red = max(−F_raw, 0)
          δ = 0  →  both slack:  T_green = T_red = 0

        τ₂_out  = r_p · (T_green − T_red)

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
        ``diagnostics``   [9]   [motor_pos, motor_aux, δ, F_cable,
                                 τ₁_des, τ₂_des, T_green, T_red, τ_motor]

        The first two diagnostic slots are motor-mode dependent:

        +-----------+-----------------------------+----------------------------+
        | Mode      | slot [0]                    | slot [1]                   |
        +===========+=============================+============================+
        | torque    | θ_m / N  (joint-side pos)   | θ̇_m / N  (joint-side vel) |
        +-----------+-----------------------------+----------------------------+
        | position  | l_m      (cable displ.)     | l_m_des  (target displ.)   |
        +-----------+-----------------------------+----------------------------+
    """

    def __init__(
        self,
        plant:          MultibodyPlant,
        manipulator:    "CupManipulatorTendon",
        k_s:            float = 200.0,
        b_c:            float = 2.0,
        tau_max:        float = 10.0,
        dt:             float = 0.002,
        motor_mode:     MotorMode = MotorMode.TORQUE,
        motor_cfg:      "MotorModelConfig | None" = None,
        omega_m:        float | None = None,
        motor_dynamics: MotorDynamics | None = None,
    ):
        """
        Parameters
        ----------
        plant          Finalized MultibodyPlant.
        manipulator    CupManipulatorTendon instance (provides PULLEY_RADIUS).
        k_s            Cable spring stiffness  [N/m].
        b_c            Cable dashpot damping    [N·s/m].
        tau_max        Output torque saturation  [Nm].
        dt             Discrete update period  [s].
        motor_mode     Which motor dynamics to use (default: TORQUE).
        motor_cfg      Motor datasheet config (required for torque mode).
        omega_m        Motor bandwidth [rad/s] (position mode only).
                       Defaults to ``motor_cfg.max_velocity_joint``.
        motor_dynamics Pre-built MotorDynamics instance.  When provided,
                       ``motor_mode``, ``motor_cfg``, and ``omega_m`` are
                       ignored.
        """
        super().__init__()
        self._plant   = plant
        self._manip   = manipulator
        self._k_s     = float(k_s)
        self._b_c     = float(b_c)
        self._tau_max = float(tau_max)
        self._dt      = float(dt)
        self._r_p     = manipulator.PULLEY_RADIUS

        # ── Motor dynamics ───────────────────────────────────────────────────
        if motor_dynamics is not None:
            self._motor = motor_dynamics
        else:
            self._motor = create_motor_dynamics(
                mode=motor_mode,
                motor_cfg=motor_cfg,
                k_s=k_s, b_c=b_c, r_p=self._r_p, dt=dt,
                omega_m=omega_m,
            )
        self._motor_mode = (
            motor_mode if motor_dynamics is None
            else (MotorMode.TORQUE if isinstance(motor_dynamics, TorqueMotor)
                  else MotorMode.POSITION)
        )

        # Velocity-vector indices for [q1, q2] in Drake's nv-vector
        j1 = manipulator.get_joint_by_name(plant, manipulator.JT1_NAME)
        j2 = manipulator.get_joint_by_name(plant, manipulator.JT2_NAME)
        self._v_idx = [j1.velocity_start(), j2.velocity_start()]

        # ── Discrete state: motor internal state ─────────────────────────────
        self._motor_state_idx = self.DeclareDiscreteState(self._motor.num_states)
        self.DeclarePeriodicDiscreteUpdateEvent(dt, 0.0, self._update_motor)

        # ── Ports ────────────────────────────────────────────────────────────
        nstate = plant.num_multibody_states()
        self._tau_port   = self.DeclareVectorInputPort("tau_desired",  2)
        self._state_port = self.DeclareVectorInputPort("plant_state",  nstate)

        self.DeclareVectorOutputPort("actuation",   2, self._calc_actuation)
        self.DeclareVectorOutputPort("diagnostics", 9, self._calc_diagnostics)

    @property
    def motor_mode(self) -> MotorMode:
        """Active motor dynamics mode."""
        return self._motor_mode

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

    # ── Discrete update: delegates to motor dynamics ──────────────────────────

    def _update_motor(self, context, discrete_state):
        """Advance motor state by one timestep via the motor dynamics model."""
        motor_state = context.get_discrete_state(self._motor_state_idx).value().copy()
        tau_des     = self._tau_port.Eval(context)
        q, q_dot    = self._read_joint_state(context)
        new_state   = self._motor.step(motor_state, tau_des[1], q[1], q_dot[1])
        discrete_state.get_mutable_vector(self._motor_state_idx).SetFromVector(new_state)

    # ── Output port callbacks ─────────────────────────────────────────────────

    def _calc_actuation(self, context, output):
        tau_des     = self._tau_port.Eval(context)
        motor_state = context.get_discrete_state(self._motor_state_idx).value()
        q, q_dot    = self._read_joint_state(context)

        F_cable, _, _, _, _, _ = self._motor.compute_spring_force(
            motor_state, tau_des[1], q[1], q_dot[1],
        )
        tau_out = np.array([
            tau_des[0],              # J1: pass-through (rigid)
            self._r_p * F_cable,     # J2: SEA cable spring
        ])
        output.SetFromVector(np.clip(tau_out, -self._tau_max, self._tau_max))

    def _calc_diagnostics(self, context, output):
        tau_des     = self._tau_port.Eval(context)
        motor_state = context.get_discrete_state(self._motor_state_idx).value()
        q, q_dot    = self._read_joint_state(context)

        F_cable, delta, T_green, T_red, s0, s1 = self._motor.compute_spring_force(
            motor_state, tau_des[1], q[1], q_dot[1],
        )
        output.SetFromVector(np.array([
            s0,           # [0]  motor pos (l_m or θ_m/N)      [m or rad]
            s1,           # [1]  motor aux (l_m_des or θ̇_m/N)
            delta,        # [2]  spring extension δ             [m]
            F_cable,      # [3]  net cable force                [N]
            tau_des[0],   # [4]  desired τ₁ (pass-through)     [Nm]
            tau_des[1],   # [5]  desired τ₂ (before spring)    [Nm]
            T_green,      # [6]  retracting cable tension       [N]
            T_red,        # [7]  extending cable tension        [N]
            self._motor.compute_motor_torque(motor_state, tau_des[1]),
                          # [8]  motor-side electromagnetic τ   [Nm]
        ]))

    def initialize_spring_at_rest(self, context, q2_init: float) -> None:
        """Set motor state so the spring starts with δ = 0 (no pre-load).

        Call this on the SEACableActuator's own context from the diagram,
        before calling simulator.Initialize().

        Parameters
        ----------
        context  : The SEACableActuator's mutable context from the diagram.
        q2_init  : Initial joint-2 angle [rad] (user-order).
        """
        init_state = self._motor.initial_state(q2_init)
        context.get_mutable_discrete_state(self._motor_state_idx).SetFromVector(init_state)
