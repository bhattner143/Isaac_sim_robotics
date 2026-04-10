"""
actuators/sea_isaacsim.py
─────────────────────────
Pure-NumPy Series Elastic Actuator (SEA) model for Isaac Sim.

This is the Drake-free counterpart of ``actuators/sea.py``.
It contains identical physics but uses a simple ``step()`` call instead of
Drake's LeafSystem / discrete-state machinery.

Physical topology (joint 2 only)
─────────────────────────────────
    Motor drum → cable → SPRING → Big Pulley → Link 2 anchor

State variable
──────────────
    l_m  [m]  — motor-side cable displacement (wound on drum)

SEA equations
─────────────
    δ       = l_m − r_p · q₂                        spring extension  [m]
    l̇_m    = ω_m · (l_m_des − l_m)                  motor position servo
    l_m_des = r_p · q₂ + τ₂_des / (k_s · r_p)       steady-state inversion
    F_raw   = k_s · δ + b_c · (l̇_m − r_p · q̇₂)     spring–damper force
    T_green = max( F_raw, 0)                          retracting cable
    T_red   = max(−F_raw, 0)                          extending cable
    τ₂_out  = r_p · (T_green − T_red) = r_p · F_raw

Usage::

    from actuators.sea_isaacsim import SEACableActuatorNP

    sea = SEACableActuatorNP(r_p=0.04775, k_s=200.0, b_c=2.0,
                             omega_m=30.0, tau_max=50.0, dt=0.01)
    sea.initialize(q2_init=np.deg2rad(15.0))

    # In sim loop:
    tau_out, diag = sea.step(tau_desired=np.array([τ1, τ2]), q=q, q_dot=q_dot)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SEADiagnostics:
    """Diagnostic snapshot from one SEA step."""
    l_m:       float   # motor cable displacement       [m]
    l_m_des:   float   # desired motor cable position   [m]
    delta:     float   # spring extension δ             [m]
    F_cable:   float   # net cable force                [N]
    tau1_des:  float   # desired τ₁ (pass-through)     [Nm]
    tau2_des:  float   # desired τ₂ (before spring)    [Nm]
    T_green:   float   # retracting cable tension       [N]
    T_red:     float   # extending cable tension        [N]
    tau_sea:   float   # actual τ₂ applied via spring  [Nm]


class SEACableActuatorNP:
    """Pure-NumPy SEA cable actuator for Isaac Sim (no Drake dependency).

    Joint 1 torque passes through unchanged (rigid direct drive).
    Joint 2 torque is mediated by a cable spring-damper.

    Parameters
    ----------
    r_p     : Pulley pitch radius [m].
    k_s     : Cable spring stiffness [N/m].
    b_c     : Cable dashpot damping [N·s/m].
    omega_m : Motor position servo bandwidth [rad/s].
    tau_max : Output torque saturation [Nm].
    dt      : Integration timestep [s].
    """

    def __init__(
        self,
        r_p:     float = 60 * 0.005 / (2.0 * np.pi),
        k_s:     float = 200.0,
        b_c:     float = 2.0,
        omega_m: float = 30.0,
        tau_max: float = 50.0,
        dt:      float = 0.01,
    ):
        self.r_p     = float(r_p)
        self.k_s     = float(k_s)
        self.b_c     = float(b_c)
        self.omega_m = float(omega_m)
        self.tau_max = float(tau_max)
        self.dt      = float(dt)
        self.l_m     = 0.0   # motor cable displacement [m]

    def initialize(self, q2_init: float) -> None:
        """Set l_m = r_p·q₂ so the spring starts at rest (δ = 0)."""
        self.l_m = self.r_p * q2_init

    def _compute_l_m_des(self, tau2_des: float, q2: float) -> float:
        """Steady-state spring inversion: l_m_des = r_p·q₂ + τ₂/(k_s·r_p)."""
        return self.r_p * q2 + tau2_des / (self.k_s * self.r_p)

    def _spring_force(self, l_m: float, l_m_des: float,
                      q2: float, q2_dot: float):
        """Compute cable force, spring extension, and motor velocity.

        Returns (F_cable, delta, l_m_dot, T_green, T_red).
        """
        delta     = l_m - self.r_p * q2
        l_m_dot   = self.omega_m * (l_m_des - l_m)
        delta_dot = l_m_dot - self.r_p * q2_dot
        F_raw     = self.k_s * delta + self.b_c * delta_dot
        T_green   = float(max(F_raw,  0.0))
        T_red     = float(max(-F_raw, 0.0))
        F_cable   = T_green - T_red   # bidirectional via antagonism
        return F_cable, delta, l_m_dot, T_green, T_red

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

        # Motor target via steady-state inversion
        l_m_des = self._compute_l_m_des(tau2_des, q2)

        # Snapshot motor state BEFORE update (for consistent force calc)
        l_m_now = self.l_m

        # Spring force at current state
        F_cable, delta, l_m_dot, T_green, T_red = self._spring_force(
            l_m_now, l_m_des, q2, q2_dot,
        )

        # Output torques
        tau_sea = self.r_p * F_cable
        tau_out = np.array([tau1_des, tau_sea])
        tau_out = np.clip(tau_out, -self.tau_max, self.tau_max)

        # Euler-step motor servo: l_m ← l_m + dt·ω_m·(l_m_des − l_m)
        self.l_m += self.dt * self.omega_m * (l_m_des - l_m_now)

        diag = SEADiagnostics(
            l_m=l_m_now, l_m_des=l_m_des, delta=delta,
            F_cable=F_cable,
            tau1_des=tau1_des, tau2_des=tau2_des,
            T_green=T_green, T_red=T_red,
            tau_sea=float(tau_out[1]),
        )
        return tau_out, diag
