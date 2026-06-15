"""Simulated CubeMars MIT-mode motor for cable-driven MHP (no hardware CAN).

MIT impedance law (CubeMars V3, joint-side quantities)::

    τ_m = Kp · (p_des − p) + Kd · (v_des − v) + τ_ff        [Nm]

Two integration modes:

**Algebraic** (default)
    τ_out = τ_m immediately — infinite bandwidth servo.

**Dynamic** (optional)
    J_m · θ̈_m = τ_m − b_m · θ̇_m
    τ_out = τ_m applied to the rotor; cable tension F = τ_m / r_spool.

Rigid cable (no series spring)::
    T_cable = τ_m / r_spool   [N]   (tension ≥ 0 when winding pulls)
    τ_joint = r_joint · T_cable  via the cable wrench matrix W(q).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DummyMITMotorConfig:
    """Parameters for one simulated MIT motor."""
    name: str = "motor"
    r_spool: float = 0.02          # spool pitch radius [m]  (40 mm drum → 20 mm)
    kp: float = 15.0               # MIT position gain [Nm/rad]
    kd: float = 0.5                # MIT velocity gain [Nm·s/rad]
    tau_max: float = 10.0          # torque saturation [Nm]
    J_m: float = 0.01              # rotor inertia (joint side) [kg·m²]
    b_m: float = 0.05              # viscous damping [Nm·s/rad]
    use_dynamics: bool = False     # False → algebraic MIT
    tension_noise_std: float = 0.0 # simulated load-cell noise [N]


@dataclass
class DummyMITMotorState:
    theta_m: float = 0.0
    theta_m_dot: float = 0.0
    tau_m: float = 0.0
    T_cable: float = 0.0
    T_meas: float = 0.0


@dataclass
class DummyMITMotor:
    """Single MIT-mode motor with optional 2nd-order rotor dynamics."""

    cfg: DummyMITMotorConfig = field(default_factory=DummyMITMotorConfig)
    state: DummyMITMotorState = field(default_factory=DummyMITMotorState)

    def reset(self, theta_init: float = 0.0) -> None:
        self.state = DummyMITMotorState(theta_m=theta_init, theta_m_dot=0.0)

    def mit_torque(
        self,
        p: float,
        v: float,
        p_des: float,
        v_des: float,
        tau_ff: float,
        kp: float | None = None,
        kd: float | None = None,
    ) -> float:
        """Evaluate the MIT impedance command (unsaturated)."""
        kp = self.cfg.kp if kp is None else kp
        kd = self.cfg.kd if kd is None else kd
        return kp * (p_des - p) + kd * (v_des - v) + tau_ff

    def step(
        self,
        dt: float,
        p: float,
        v: float,
        p_des: float,
        v_des: float,
        tau_ff: float,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, float, float]:
        """Advance one timestep.

        Parameters
        ----------
        p, v       : measured shaft position / velocity [rad, rad/s]
        p_des, v_des, tau_ff : MIT command

        Returns
        -------
        tau_out    : joint-side torque applied to the plant [Nm]
        T_cable    : cable tension [N]  (= |τ_out|/r_spool for rigid pull)
        T_meas     : simulated load-cell reading [N]
        """
        tau_m = self.mit_torque(p, v, p_des, v_des, tau_ff)
        tau_m = float(np.clip(tau_m, -self.cfg.tau_max, self.cfg.tau_max))

        if self.cfg.use_dynamics:
            theta = self.state.theta_m
            theta_dot = self.state.theta_m_dot
            theta_ddot = (tau_m - self.cfg.b_m * theta_dot) / self.cfg.J_m
            theta_dot_new = theta_dot + dt * theta_ddot
            theta_new = theta + dt * theta_dot_new
            self.state.theta_m = theta_new
            self.state.theta_m_dot = theta_dot_new
            tau_out = tau_m
        else:
            self.state.theta_m = p
            self.state.theta_m_dot = v
            tau_out = tau_m

        self.state.tau_m = tau_out
        r = self.cfg.r_spool
        T_cable = tau_out / r if abs(r) > 1e-12 else 0.0
        self.state.T_cable = T_cable

        noise = 0.0
        if self.cfg.tension_noise_std > 0.0 and rng is not None:
            noise = float(rng.normal(0.0, self.cfg.tension_noise_std))
        self.state.T_meas = T_cable + noise

        return tau_out, T_cable, self.state.T_meas
