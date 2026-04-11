"""
actuators/motor_dynamics.py

Motor dynamics models for SEA cable actuation.

Two modes are provided:

**Position-servo** (1st-order):
    Models the motor as a closed-loop position servo with bandwidth ``ω_m``.
    Input is desired cable position ``l_m_des``; output is actual cable
    displacement ``l_m`` after first-order lag.

      l̇_m = ω_m · (l_m_des − l_m)

    Suitable when the motor runs a factory position controller (CAN position
    mode).  Uses one discrete state.

**Torque** (2nd-order):
    Models the motor rotor as an inertia ``J_m`` with viscous damping ``b_m``,
    driven by commanded torque ``τ_m`` and loaded by the spring reaction.
    Input is desired joint torque ``τ_des``; the motor equation is:

      J_m · θ̈_m = τ_m − b_m · θ̇_m − τ_s / N

    where ``τ_s = k_s · δ + b_c · δ̇`` (spring + damper) and
    ``δ = θ_m / N − q₂`` (spring deflection at the joint side).

    Suitable when the motor runs in MIT torque/impedance mode (the real-world
    default for quasi-direct-drive actuators like CubeMars AK60-6).
    Uses two discrete states: ``θ_m``, ``θ̇_m``.

Both models expose the same interface so that ``SEACableActuator`` can
swap them transparently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from actuators.motor import MotorModelConfig


# ── Motor mode enum ────────────────────────────────────────────────────────────

class MotorMode(Enum):
    """Motor dynamics mode for the SEA cable actuator."""
    TORQUE   = "torque"       # 2nd-order rotor dynamics (recommended)
    POSITION = "position"     # 1st-order position servo (legacy)


# ── Abstract base ──────────────────────────────────────────────────────────────

class MotorDynamics(ABC):
    """Abstract motor dynamics model for a cable-driven SEA joint.

    Concrete subclasses implement ``step()`` and ``spring_force()`` with
    the same signature so that the ``SEACableActuator`` does not care
    which model is running.

    State layout
    ────────────
    Each model stores its discrete state as a 1-D NumPy array.  The length
    differs by model; use ``num_states`` to query it.
    """

    def __init__(
        self,
        k_s:   float,
        b_c:   float,
        r_p:   float,
        N:     float,
        dt:    float,
    ):
        self._k_s = float(k_s)
        self._b_c = float(b_c)
        self._r_p = float(r_p)
        self._N   = float(N)
        self._dt  = float(dt)

    @property
    @abstractmethod
    def num_states(self) -> int:
        """Number of discrete state scalars."""

    @abstractmethod
    def initial_state(self, q2_init: float) -> np.ndarray:
        """State vector that puts the spring at rest (δ = 0)."""

    @abstractmethod
    def step(
        self,
        state:    np.ndarray,
        tau2_des: float,
        q2:       float,
        q2_dot:   float,
    ) -> np.ndarray:
        """Advance motor state by one timestep.

        Parameters
        ----------
        state     Current discrete state vector.
        tau2_des  Desired joint-2 torque from the CT controller  [Nm].
        q2        Current joint-2 angle  [rad].
        q2_dot    Current joint-2 velocity  [rad/s].

        Returns
        -------
        new_state  Updated state vector (same length as *state*).
        """

    @abstractmethod
    def compute_spring_force(
        self,
        state:    np.ndarray,
        tau2_des: float,
        q2:       float,
        q2_dot:   float,
    ) -> tuple[float, float, float, float, float, float]:
        """Compute cable spring force and diagnostics (no state mutation).

        Returns
        -------
        (F_cable, delta, T_green, T_red, l_m_or_theta_m, l_m_des_or_theta_m_dot)
            F_cable    Net cable force [N] (positive = tension pulling joint)
            delta      Spring extension δ [m] (+ green taut, − red taut)
            T_green    Retracting cable tension [N]  (>= 0)
            T_red      Extending  cable tension [N]  (>= 0)
            state_0    Primary motor state (l_m or θ_m/N ≡ motor pos at joint side) [m or rad]
            state_1    Secondary motor state (l_m_des or θ̇_m/N)
        """

    # ── shared helper: unilateral cable model ────────────────────────────────

    def _cable_tensions(self, F_raw: float, delta: float):
        """Split raw spring force into antagonistic cable tensions."""
        if delta > 0.0:
            T_green = float(max(F_raw, 0.0))
            T_red   = 0.0
        elif delta < 0.0:
            T_green = 0.0
            T_red   = float(max(-F_raw, 0.0))
        else:
            T_green = 0.0
            T_red   = 0.0
        F_cable = T_green - T_red
        return F_cable, T_green, T_red

    def compute_motor_torque(
        self,
        state:    np.ndarray,
        tau2_des: float,
    ) -> float:
        """Motor-side electromagnetic torque command [Nm].

        Default returns NaN (not applicable for position-servo mode).
        Overridden by ``TorqueMotor`` to return ``τ₂_des / N``.
        """
        return float('nan')


# ── Position-servo motor (1st-order) ──────────────────────────────────────────

class PositionServoMotor(MotorDynamics):
    """First-order position servo: l̇_m = ω_m · (l_m_des − l_m).

    This models the motor as a closed-loop position controller with
    bandwidth ``ω_m``.  The spring inversion computes the desired
    cable displacement from the commanded torque.

    State: [l_m]  (1 scalar — cable displacement in metres).
    """

    def __init__(
        self,
        k_s:     float,
        b_c:     float,
        r_p:     float,
        N:       float,
        dt:      float,
        omega_m: float,
    ):
        super().__init__(k_s, b_c, r_p, N, dt)
        self._omega_m = float(omega_m)

    @property
    def num_states(self) -> int:
        return 1

    def initial_state(self, q2_init: float) -> np.ndarray:
        return np.array([self._r_p * q2_init])

    def _compute_l_m_des(self, tau2_des: float, q2: float) -> float:
        return self._r_p * q2 + tau2_des / (self._k_s * self._r_p)

    def step(self, state, tau2_des, q2, q2_dot):
        l_m     = state[0]
        l_m_des = self._compute_l_m_des(tau2_des, q2)
        l_m_new = l_m + self._dt * self._omega_m * (l_m_des - l_m)
        return np.array([l_m_new])

    def compute_spring_force(self, state, tau2_des, q2, q2_dot):
        l_m     = state[0]
        l_m_des = self._compute_l_m_des(tau2_des, q2)
        delta   = l_m - self._r_p * q2
        l_m_dot = self._omega_m * (l_m_des - l_m)
        delta_dot = l_m_dot - self._r_p * q2_dot
        F_raw   = self._k_s * delta + self._b_c * delta_dot
        F_cable, T_green, T_red = self._cable_tensions(F_raw, delta)
        return F_cable, delta, T_green, T_red, l_m, l_m_des


# ── Torque-mode motor (2nd-order) ─────────────────────────────────────────────

class TorqueMotor(MotorDynamics):
    """Second-order rotor dynamics: J_m · θ̈_m = τ_m − b_m · θ̇_m − r_p·F/N.

    Models the rotor as an inertia loaded by the cable spring reaction through
    the gearbox.  The controller's desired joint torque ``τ₂_des`` is converted
    to a motor-side torque command: ``τ_m = τ₂_des / N``.

    The cable wraps on a drum of radius ``r_p`` at the gearbox output, so::

        l_motor = r_p · θ_m / N          cable displacement from motor  [m]
        l_joint = r_p · q₂               cable displacement from joint  [m]
        δ       = l_motor − l_joint       linear spring extension       [m]
                = r_p · (θ_m/N − q₂)

    Cable force (linear, in Newtons)::

        F = k_s · δ + b_c · δ̇

    Motor equation (motor side)::

        J_m · θ̈_m = τ_m − b_m · θ̇_m − r_p · F / N

    The effective torsional spring stiffness seen by the motor rotor is
    ``r_p² · k_s / N²``, giving a motor-side resonance frequency::

        ω_n = sqrt(r_p² · k_s / (N² · J_m))

    For AK60-6 with k_s = 300 N/m, r_p = 47.75 mm, N = 6:
    ω_n ≈ 24 rad/s (4 Hz).

    State: [θ_m, θ̇_m]  (2 scalars — motor-side angle and velocity).
    """

    def __init__(
        self,
        k_s:   float,
        b_c:   float,
        r_p:   float,
        N:     float,
        dt:    float,
        J_m:   float,
        b_m:   float,
    ):
        super().__init__(k_s, b_c, r_p, N, dt)
        self._J_m = float(J_m)
        self._b_m = float(b_m)

    @property
    def num_states(self) -> int:
        return 2

    def initial_state(self, q2_init: float) -> np.ndarray:
        theta_m_init = self._N * q2_init   # spring at rest: θ_m/N = q₂
        return np.array([theta_m_init, 0.0])

    def _cable_force_raw(self, theta_m, theta_m_dot, q2, q2_dot):
        """Compute raw cable force and linear spring extension.

        The spring operates in the linear domain::

            δ      = r_p · (θ_m/N − q₂)                    [m]
            δ̇     = r_p · (θ̇_m/N − q̇₂)                   [m/s]
            F_raw  = k_s · δ + b_c · δ̇                      [N]

        Returns (F_raw, delta_lin).
        """
        delta_rad     = theta_m / self._N - q2
        delta_dot_rad = theta_m_dot / self._N - q2_dot
        delta_lin     = self._r_p * delta_rad
        delta_dot_lin = self._r_p * delta_dot_rad
        F_raw         = self._k_s * delta_lin + self._b_c * delta_dot_lin
        return F_raw, delta_lin

    def step(self, state, tau2_des, q2, q2_dot):
        theta_m     = state[0]
        theta_m_dot = state[1]

        # Motor-side torque command
        tau_m = tau2_des / self._N

        # Cable force → motor load: r_p · F / N
        F_raw, _ = self._cable_force_raw(theta_m, theta_m_dot, q2, q2_dot)
        tau_spring_motor = self._r_p * F_raw / self._N

        # Rotor acceleration
        theta_m_ddot = (tau_m - self._b_m * theta_m_dot - tau_spring_motor) / self._J_m

        # Semi-implicit Euler (velocity first, then position)
        theta_m_dot_new = theta_m_dot + self._dt * theta_m_ddot
        theta_m_new     = theta_m     + self._dt * theta_m_dot_new

        return np.array([theta_m_new, theta_m_dot_new])

    def compute_spring_force(self, state, tau2_des, q2, q2_dot):
        theta_m     = state[0]
        theta_m_dot = state[1]

        F_raw, delta_lin = self._cable_force_raw(theta_m, theta_m_dot, q2, q2_dot)

        F_cable, T_green, T_red = self._cable_tensions(F_raw, delta_lin)

        # Diagnostic outputs: motor pos at joint side, motor vel at joint side
        motor_pos_joint = theta_m / self._N      # equivalent of l_m  [rad]
        motor_vel_joint = theta_m_dot / self._N   # equivalent of l_m_dot  [rad/s]
        return F_cable, delta_lin, T_green, T_red, motor_pos_joint, motor_vel_joint

    def compute_motor_torque(self, state, tau2_des):
        """Motor-side electromagnetic torque command: τ_m = τ₂_des / N  [Nm]."""
        return tau2_des / self._N


# ── Factory ────────────────────────────────────────────────────────────────────

def create_motor_dynamics(
    mode:       MotorMode,
    motor_cfg:  MotorModelConfig | None,
    k_s:        float,
    b_c:        float,
    r_p:        float,
    dt:         float,
    omega_m:    float | None = None,
) -> MotorDynamics:
    """Create a motor dynamics model from a motor configuration.

    Parameters
    ----------
    mode       ``MotorMode.TORQUE`` or ``MotorMode.POSITION``.
    motor_cfg  Motor datasheet config (required for torque mode; gear_ratio,
               rotor_inertia_motor, viscous_damping_joint are read from it).
               For position mode, only gear_ratio is needed (or can pass None
               and provide omega_m directly).
    k_s        Cable spring stiffness [N/m].
    b_c        Cable dashpot damping [N·s/m].
    r_p        Pulley radius [m].
    dt         Discrete timestep [s].
    omega_m    Motor bandwidth [rad/s] (position mode).  If None, derived
               from motor_cfg.position_servo_bandwidth (= 1 / τ_m).
    """
    N = motor_cfg.gear_ratio if motor_cfg is not None else 1.0

    if mode == MotorMode.POSITION:
        if omega_m is None:
            if motor_cfg is None:
                raise ValueError("omega_m required when motor_cfg is None")
            omega_m = motor_cfg.position_servo_bandwidth
        return PositionServoMotor(
            k_s=k_s, b_c=b_c, r_p=r_p, N=N, dt=dt,
            omega_m=omega_m,
        )

    elif mode == MotorMode.TORQUE:
        if motor_cfg is None:
            raise ValueError("motor_cfg required for torque mode")
        J_m = motor_cfg.rotor_inertia_motor
        # Motor-side viscous damping: b_m = b_v_joint / N²
        b_m = motor_cfg.viscous_damping_joint / (N ** 2)
        return TorqueMotor(
            k_s=k_s, b_c=b_c, r_p=r_p, N=N, dt=dt,
            J_m=J_m, b_m=b_m,
        )

    else:
        raise ValueError(f"Unknown motor mode: {mode}")
