"""
actuators/sea_isaacsim.py
─────────────────────────
Pure-NumPy Series Elastic Actuator (SEA) model for Isaac Sim.

This is the Drake-free counterpart of ``actuators/sea.py``.
It contains identical physics but uses a simple ``step()`` call instead of
Drake's LeafSystem / discrete-state machinery.

Motor dynamics are delegated to a pluggable ``MotorDynamics`` object (see
``actuators.motor_dynamics``).  Two modes are supported:

  - **torque** (default) — 2nd-order rotor dynamics driven by torque command.
    Uses CubeMars MIT torque mode.  Parameters: ``J_m``, ``b_m`` from motor
    datasheet.
  - **position** — 1st-order position servo with bandwidth ``ω_m``.  Legacy
    mode for motors running a factory position controller.

Physical topology (joint 2 only)
─────────────────────────────────
    Motor drum → cable → SPRING → Big Pulley → Link 2 anchor

Unilateral cable model
──────────────────────
    Cables can only PULL (tension ≥ 0), never push:
      δ > 0 → green taut:  T_green = max(F_raw, 0),  T_red = 0
      δ < 0 → red taut:    T_green = 0,  T_red = max(−F_raw, 0)
      δ = 0 → both slack:  T_green = T_red = 0

    τ₂_out  = r_p · (T_green − T_red)

Usage::

    from actuators.sea_isaacsim import SEACableActuatorNP
    from actuators.motor_dynamics import MotorMode
    from actuators.motor import get_motor

    motor = get_motor("AK60_6_KV80_Config")
    sea = SEACableActuatorNP(r_p=0.04775, k_s=300.0, b_c=2.0,
                             tau_max=9.0, dt=0.01,
                             motor_mode=MotorMode.TORQUE, motor_cfg=motor)
    sea.initialize(q2_init=np.deg2rad(15.0))

    # In sim loop:
    tau_out, diag = sea.step(tau_desired=np.array([τ1, τ2]), q=q, q_dot=q_dot)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from dataclasses import dataclass

from actuators.motor_dynamics import (
    MotorDynamics,
    MotorMode,
    TorqueMotor,
    create_motor_dynamics,
)

if TYPE_CHECKING:
    from actuators.motor import MotorModelConfig


@dataclass
class SEADiagnostics:
    """Diagnostic snapshot from one SEA step.

    The first two fields are motor-mode dependent:

    +-----------+---------------------------+---------------------------+
    | Mode      | motor_pos                 | motor_aux                 |
    +===========+===========================+===========================+
    | torque    | θ_m / N  (joint-side pos) | θ̇_m / N (joint-side vel) |
    +-----------+---------------------------+---------------------------+
    | position  | l_m  (cable displacement) | l_m_des  (target displ.)  |
    +-----------+---------------------------+---------------------------+
    """
    motor_pos: float   # l_m (position) or θ_m/N (torque)       [m or rad]
    motor_aux: float   # l_m_des (position) or θ̇_m/N (torque)  [m or rad/s]
    delta:     float   # spring extension δ                      [m]
    F_cable:   float   # net cable force                         [N]
    tau1_des:  float   # desired τ₁ (pass-through)              [Nm]
    tau2_des:  float   # desired τ₂ (before spring)             [Nm]
    T_green:   float   # retracting cable tension                [N]
    T_red:     float   # extending cable tension                 [N]
    tau_sea:   float   # actual τ₂ applied via spring           [Nm]
    tau_motor: float   # motor-side electromagnetic torque       [Nm]

    # Backward-compatible aliases (used by older scripts / multi-instance)
    @property
    def l_m(self) -> float:
        return self.motor_pos

    @property
    def l_m_des(self) -> float:
        return self.motor_aux


class SEACableActuatorNP:
    """Pure-NumPy SEA cable actuator for Isaac Sim (no Drake dependency).

    Joint 1 torque passes through unchanged (rigid direct drive).
    Joint 2 torque is mediated by a cable spring-damper.

    Motor dynamics are delegated to a :class:`MotorDynamics` object,
    matching the architecture of the Drake-based ``SEACableActuator``.

    Parameters
    ----------
    r_p              : Pulley pitch radius [m].
    k_s              : Cable spring stiffness [N/m].
    b_c              : Cable dashpot damping [N·s/m].
    tau_max          : Output torque saturation [Nm].
    dt               : Physics timestep [s].
    motor_mode       : Which motor dynamics to use (default: TORQUE).
    motor_cfg        : Motor datasheet config (required for torque mode).
    omega_m          : Motor bandwidth [rad/s] (position mode only).
    motor_dynamics   : Pre-built MotorDynamics instance.  When provided,
                       ``motor_mode``, ``motor_cfg``, and ``omega_m`` are
                       ignored.
    motor_substeps   : Number of sub-integration steps per physics step for
                       the motor dynamics.  ``None`` (default) auto-computes a
                       safe value from the motor-spring natural frequency so
                       that the semi-implicit Euler integrator stays stable.
    """

    def __init__(
        self,
        r_p:              float = 60 * 0.005 / (2.0 * np.pi),
        k_s:              float = 200.0,
        b_c:              float = 2.0,
        tau_max:          float = 50.0,
        dt:               float = 0.01,
        motor_mode:       MotorMode = MotorMode.TORQUE,
        motor_cfg:        "MotorModelConfig | None" = None,
        omega_m:          float | None = None,
        motor_dynamics:   MotorDynamics | None = None,
        motor_substeps:   int | None = None,
    ):
        self.r_p     = float(r_p)
        self.k_s     = float(k_s)
        self.b_c     = float(b_c)
        self.tau_max = float(tau_max)
        self.dt      = float(dt)

        # ── Resolve motor mode (needed before substep calculation) ───────────
        if motor_dynamics is not None:
            resolved_mode = (
                MotorMode.TORQUE if isinstance(motor_dynamics, TorqueMotor)
                else MotorMode.POSITION
            )
        else:
            resolved_mode = motor_mode
            if motor_cfg is None and omega_m is not None:
                resolved_mode = MotorMode.POSITION
        self._motor_mode = resolved_mode

        # ── Auto-compute motor substeps for numerical stability ──────────────
        if motor_substeps is None:
            motor_substeps = self._auto_substeps(
                resolved_mode, motor_cfg, r_p, k_s, dt,
            )
        self._motor_substeps = int(max(1, motor_substeps))
        dt_motor = dt / self._motor_substeps

        # ── Motor dynamics ───────────────────────────────────────────────────
        if motor_dynamics is not None:
            self._motor = motor_dynamics
        else:
            self._motor = create_motor_dynamics(
                mode=resolved_mode,
                motor_cfg=motor_cfg,
                k_s=k_s, b_c=b_c, r_p=self.r_p, dt=dt_motor,
                omega_m=omega_m,
            )

        # Motor state vector (1 for position mode, 2 for torque mode)
        self._motor_state = np.zeros(self._motor.num_states)

    @staticmethod
    def _auto_substeps(
        mode: MotorMode,
        motor_cfg: "MotorModelConfig | None",
        r_p: float,
        k_s: float,
        dt: float,
    ) -> int:
        """Compute minimum substeps so ω_n × dt_sub < 1 (stable Euler)."""
        if mode != MotorMode.TORQUE or motor_cfg is None:
            return 1
        N   = motor_cfg.gear_ratio
        J_m = motor_cfg.rotor_inertia_motor
        # Motor-side resonance: ω_n = sqrt(r_p² · k_s / (N² · J_m))
        omega_n = np.sqrt(r_p ** 2 * k_s / (N ** 2 * J_m))
        # Need ω_n * dt_sub < 1.0 for safe semi-implicit Euler
        return int(np.ceil(omega_n * dt / 1.0))

    @property
    def motor_mode(self) -> MotorMode:
        """Active motor dynamics mode."""
        return self._motor_mode

    @property
    def motor_substeps(self) -> int:
        """Number of sub-integration steps per physics step."""
        return self._motor_substeps

    def initialize(self, q2_init: float) -> None:
        """Set motor state so the spring starts at rest (δ = 0)."""
        self._motor_state = self._motor.initial_state(q2_init)

    def step(
        self,
        tau_desired: np.ndarray,   # (2,) desired [τ₁, τ₂] from CT
        q:           np.ndarray,   # (2,) current [q₁, q₂]
        q_dot:       np.ndarray,   # (2,) current [q̇₁, q̇₂]
    ) -> tuple[np.ndarray, SEADiagnostics]:
        """Advance one timestep.

        Returns
        -------
        tau_out : (2,) actual torques applied to plant [Nm]
        diag    : SEADiagnostics snapshot
        """
        tau1_des = float(tau_desired[0])
        tau2_des = float(tau_desired[1])
        q2       = float(q[1])
        q2_dot   = float(q_dot[1])

        # Compute spring force BEFORE stepping motor (consistent with Drake)
        F_cable, delta, T_green, T_red, s0, s1 = self._motor.compute_spring_force(
            self._motor_state, tau2_des, q2, q2_dot,
        )

        # Output torques
        tau_sea = self.r_p * F_cable
        tau_out = np.array([tau1_des, tau_sea])
        tau_out = np.clip(tau_out, -self.tau_max, self.tau_max)

        # Step motor state (sub-stepped for numerical stability)
        for _ in range(self._motor_substeps):
            self._motor_state = self._motor.step(
                self._motor_state, tau2_des, q2, q2_dot,
            )

        diag = SEADiagnostics(
            motor_pos=s0, motor_aux=s1, delta=delta,
            F_cable=F_cable,
            tau1_des=tau1_des, tau2_des=tau2_des,
            T_green=T_green, T_red=T_red,
            tau_sea=float(tau_out[1]),
            tau_motor=self._motor.compute_motor_torque(self._motor_state, tau2_des),
        )
        return tau_out, diag
