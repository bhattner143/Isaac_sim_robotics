"""
actuators/sea_exo_isaacsim.py
─────────────────────────────
Pure-NumPy Exosuit Series Elastic Actuator — Isaac Sim counterpart of
``actuators/sea_exo.py`` (PyDrake LeafSystem).

Two CubeMars motors drive antagonistic exo cables through springs on the
forearm side of the elbow pulley (Method B — centred elbow pulley).

**Deactivated** (transparent): both motors track the elbow encoder so the
spring extensions stay at zero → no added stiffness, no torque.

**Activated** (co-contraction): a symmetric angular offset Δθ is added to
both motor commands → both springs extend by δ = r_exo·Δθ.  A joint
deflection Δq from the reference produces restoring torque::

    τ_exo ≈ −2·k_exo·r_exo²·Δq       (k_eff = 2·k_exo·r_exo²)

Motor-side dynamics use two independent ``TorqueMotor`` rotor integrators
(same CubeMars MIT torque mode as the drive SEA), with sub-stepped
semi-implicit Euler for numerical stability.

Usage::

    from actuators.sea_exo_isaacsim import SEAExoActuatorNP
    from actuators.motor import get_motor

    motor = get_motor("AK60_6_KV80_Config")
    exo = SEAExoActuatorNP(
        k_exo=200.0, b_exo=2.0, r_exo=0.04775,
        tau_max=motor.peak_torque_joint, dt=0.01, motor_cfg=motor,
    )
    exo.initialize(q2_init=np.deg2rad(15.0))

    # In sim loop:
    tau_exo, diag = exo.step(
        activated=True, delta_theta=0.5,
        q=q, q_dot=q_dot, q_des=q_des,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from actuators.motor_dynamics import TorqueMotor

if TYPE_CHECKING:
    from actuators.motor import MotorModelConfig


@dataclass
class SEAExoDiagnostics:
    """Diagnostic snapshot from one exo step.  Field order matches the
    PyDrake ``SEAExoActuator.diagnostics`` port for logging parity."""
    delta_R:     float   # [0]  right spring extension δ_R  [m]
    delta_L:     float   # [1]  left  spring extension δ_L  [m]
    F_R:         float   # [2]  right cable force            [N]
    F_L:         float   # [3]  left  cable force            [N]
    motor_pos_R: float   # [4]  right motor joint-side pos   [rad]
    motor_pos_L: float   # [5]  left  motor joint-side pos   [rad]
    motor_vel_R: float   # [6]  right motor joint-side vel   [rad/s]
    motor_vel_L: float   # [7]  left  motor joint-side vel   [rad/s]
    tau_exo:     float   # [8]  net exo torque               [Nm]
    activated:   float   # [9]  1.0 if active, 0.0 otherwise

    def as_array(self) -> np.ndarray:
        """Return the 10-element vector in the same order as Drake."""
        return np.array([
            self.delta_R, self.delta_L, self.F_R, self.F_L,
            self.motor_pos_R, self.motor_pos_L,
            self.motor_vel_R, self.motor_vel_L,
            self.tau_exo, self.activated,
        ])


class SEAExoActuatorNP:
    """Pure-NumPy exo co-contraction actuator for Isaac Sim (no Drake).

    Two antagonistic cables, each with its own CubeMars motor and spring,
    wrap around a centred elbow pulley.  Motor dynamics match the Drake
    ``SEAExoActuator`` bit-for-bit (same TorqueMotor, same PD tracking
    gains, same sub-step count).

    Internal state
    --------------
    ``[θ_mR, θ̇_mR, θ_mL, θ̇_mL, q2_anchor, was_active]``.

    ``q2_anchor`` is captured at the OFF→ON activation edge so the
    pre-tension is centred about the current joint pose (no transient
    kick at activation).  When ``q_des`` is supplied to :meth:`step`, the
    motor tracks the reference trajectory instead of the frozen anchor
    so the exo provides stiffness ABOUT q_des rather than a static pose.
    """

    # PD gains for motor encoder tracking (joint-side units) — identical
    # to the Drake implementation in ``actuators/sea_exo.py``.
    KP_TRACK = 200.0    # [Nm/rad]
    KD_TRACK = 2.0      # [Nm·s/rad]

    # Sub-stepping: run rotor integrator at dt/N_SUBSTEPS so that the
    # motor-side PD natural frequency stays well within the Nyquist limit.
    _N_SUBSTEPS = 10

    def __init__(
        self,
        k_exo:     float = 200.0,
        b_exo:     float = 2.0,
        r_exo:     float = 0.04775,
        tau_max:   float = 9.0,
        dt:        float = 0.01,
        motor_cfg: "MotorModelConfig | None" = None,
    ):
        """
        Parameters
        ----------
        k_exo      Exo cable spring stiffness [N/m].
        b_exo      Exo cable dashpot damping [N·s/m].
        r_exo      Exo elbow pulley radius [m] (Method B: centred).
        tau_max    Output torque saturation per motor [Nm].
        dt         Physics timestep [s] (outer loop).
        motor_cfg  CubeMars motor config shared by both exo motors.
        """
        if motor_cfg is None:
            raise ValueError("motor_cfg required for exo motors")

        self._k_exo   = float(k_exo)
        self._b_exo   = float(b_exo)
        self._r_exo   = float(r_exo)
        self._tau_max = float(tau_max)
        self._dt      = float(dt)
        self._N       = motor_cfg.gear_ratio

        # Two independent TorqueMotor instances (right + left cables).
        J_m = motor_cfg.rotor_inertia_motor
        b_m = motor_cfg.viscous_damping_joint / (self._N ** 2)
        dt_sub = dt / self._N_SUBSTEPS
        _kw = dict(k_s=k_exo, b_c=b_exo, r_p=r_exo, N=self._N, dt=dt_sub,
                   J_m=J_m, b_m=b_m)
        self._motor_R = TorqueMotor(**_kw)
        self._motor_L = TorqueMotor(**_kw)

        # Internal state: [θ_mR, θ̇_mR, θ_mL, θ̇_mL, q2_anchor, was_active]
        self._state = np.zeros(6)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def k_eff(self) -> float:
        """Theoretical effective stiffness when fully activated [Nm/rad]."""
        return 2.0 * self._k_exo * self._r_exo ** 2

    @property
    def r_exo(self) -> float:
        return self._r_exo

    @property
    def k_exo(self) -> float:
        return self._k_exo

    # ── Initialisation ────────────────────────────────────────────────────

    def initialize(self, q2_init: float) -> None:
        """Set motor states so both exo springs start with δ = 0.

        Right motor: θ_mR =  N · q₂     (spring at rest)
        Left  motor: θ_mL = −N · q₂     (spring at rest)
        """
        self._state = np.array([
             self._N * q2_init, 0.0,   # right motor [θ_mR, θ̇_mR]
            -self._N * q2_init, 0.0,   # left  motor [θ_mL, θ̇_mL]
            float(q2_init),            # q2_anchor — re-captured at OFF→ON edge
            0.0,                       # was_active flag
        ])

    # ── Step ──────────────────────────────────────────────────────────────

    def step(
        self,
        activated:   bool,
        delta_theta: float,
        q:           np.ndarray,           # (2,)
        q_dot:       np.ndarray,           # (2,)
        q_des:       np.ndarray | None = None,   # (2,) optional reference
    ) -> tuple[float, SEAExoDiagnostics]:
        """Advance one outer timestep; return (tau_exo, diagnostics)."""
        q2     = float(q[1])
        q2_dot = float(q_dot[1])

        state_R   = self._state[0:2].copy()
        state_L   = self._state[2:4].copy()
        q2_anchor = float(self._state[4])
        was_active = self._state[5] > 0.5

        # Capture q2 at the OFF→ON transition → pre-tension about this anchor.
        if activated and not was_active:
            q2_anchor = q2

        # Reference anchor: track q_des when active (if provided) else anchor.
        # When deactivated, track actual joint so δ stays at zero.
        if activated:
            q2_ref = float(q_des[1]) if q_des is not None else q2_anchor
        else:
            q2_ref = q2

        # Desired motor-side angle (joint-side units)
        dtheta = float(delta_theta) if activated else 0.0
        q_des_R =  q2_ref + dtheta
        q_des_L = -q2_ref + dtheta

        # Sub-stepped PD + rotor integration
        for _ in range(self._N_SUBSTEPS):
            theta_mR, theta_dot_mR = state_R
            theta_mL, theta_dot_mL = state_L

            tau2_R = (self.KP_TRACK * (q_des_R - theta_mR / self._N)
                      - self.KD_TRACK * (theta_dot_mR / self._N))
            tau2_L = (self.KP_TRACK * (q_des_L - theta_mL / self._N)
                      - self.KD_TRACK * (theta_dot_mL / self._N))

            tau2_R = np.clip(tau2_R, -self._tau_max, self._tau_max)
            tau2_L = np.clip(tau2_L, -self._tau_max, self._tau_max)

            # Left motor sees virtual joint (−q₂, −q̇₂).
            state_R = self._motor_R.step(state_R, tau2_R,  q2,  q2_dot)
            state_L = self._motor_L.step(state_L, tau2_L, -q2, -q2_dot)

        # ── Spring forces (unilateral: tension ≥ 0) ──────────────────────
        theta_mR, theta_dot_mR = state_R
        theta_mL, theta_dot_mL = state_L

        delta_R     = self._r_exo * (theta_mR     / self._N - q2)
        delta_dot_R = self._r_exo * (theta_dot_mR / self._N - q2_dot)
        F_R = max(self._k_exo * delta_R + self._b_exo * delta_dot_R, 0.0)

        delta_L     = self._r_exo * (theta_mL     / self._N + q2)
        delta_dot_L = self._r_exo * (theta_dot_mL / self._N + q2_dot)
        F_L = max(self._k_exo * delta_L + self._b_exo * delta_dot_L, 0.0)

        # τ_exo = r_exo · (F_R − F_L) — restoring in co-contraction.
        tau_exo_raw = self._r_exo * (F_R - F_L)
        if activated:
            tau_exo = float(np.clip(tau_exo_raw, -self._tau_max, self._tau_max))
        else:
            tau_exo = 0.0   # transparent when deactivated

        # ── Persist state ────────────────────────────────────────────────
        self._state = np.concatenate([
            state_R, state_L,
            [q2_anchor, 1.0 if activated else 0.0],
        ])

        diag = SEAExoDiagnostics(
            delta_R=delta_R, delta_L=delta_L,
            F_R=F_R, F_L=F_L,
            motor_pos_R=theta_mR / self._N,
            motor_pos_L=theta_mL / self._N,
            motor_vel_R=theta_dot_mR / self._N,
            motor_vel_L=theta_dot_mL / self._N,
            tau_exo=tau_exo,
            activated=1.0 if activated else 0.0,
        )
        return tau_exo, diag
