"""
rl/computed_torque_controller.py
---------------------------------
Computed-torque (inverse-dynamics) controller — pure NumPy, no Drake.

Feedback-linearizes a 2-DOF planar manipulator by cancelling nonlinear
dynamics (Coriolis + gravity) and commanding a desired joint-space
acceleration obtained from a PD law in joint space:

    a_des = q̈_ref + Kp·(q_des − q) + Kd·(q̇_ref − q̇)   [rad/s²]
    τ     = M(q)·a_des + h(q, q̇)                         [Nm]

where h(q, q̇) = C(q,q̇)·q̇ + g(q)  is the bias term.

Joint-2 torque is decomposed into cable tensions for physical logging:
    F_net   = τ₂ / r_p           [N]
    T_green = max( F_net, 0)      retracting side
    T_red   = max(−F_net, 0)      extending side

This module is engine-agnostic — it depends only on:
    - mass_matrix M(q)       ∈ ℝ^{2×2}
    - bias_forces h(q,q̇)    ∈ ℝ^{2}
which can come from Isaac Sim ArticulationView, Drake, or analytical formulas.

Usage::

    from rl.computed_torque_controller import ComputedTorqueController

    ct = ComputedTorqueController(Kp=400.0, Kd=40.0, tau_max=10.0, pulley_radius=0.04775)
    tau, info = ct.compute(q, q_dot, q_des, q_dot_ref, q_ddot_ref, M, h)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CTControllerOutput:
    """Output bundle from the computed-torque controller."""
    tau_clip: np.ndarray     # (2,) clipped actuation torques [Nm]
    tau_raw: np.ndarray      # (2,) pre-clip inverse-dynamics torques [Nm]
    T_green: float           # cable tension green side [N]
    T_red: float             # cable tension red side [N]
    q_des: np.ndarray        # (2,) desired joint positions [rad]
    a_des: np.ndarray        # (2,) desired joint acceleration [rad/s²]


class ComputedTorqueController:
    """Computed-torque (inverse-dynamics) controller for 2-DOF manipulator.

    Parameters
    ----------
    Kp : float
        Position gain [s⁻²]. ωn = √Kp  (e.g. Kp=400 → ωn=20 rad/s).
    Kd : float
        Velocity gain [s⁻¹]. ζ = Kd / (2√Kp)  (e.g. Kd=40 → ζ=1, critically damped).
    tau_max : float
        Torque saturation limit [Nm].
    pulley_radius : float
        Belt/pulley pitch radius [m] for cable tension decomposition.
    """

    def __init__(
        self,
        Kp: float = 400.0,
        Kd: float = 40.0,
        tau_max: float = 10.0,
        pulley_radius: float = 60 * 0.005 / (2.0 * np.pi),
    ):
        self.Kp = float(Kp)
        self.Kd = float(Kd)
        self.tau_max = float(tau_max)
        self.r_p = float(pulley_radius)

    def compute(
        self,
        q: np.ndarray,            # (2,) current joint positions [rad]
        q_dot: np.ndarray,        # (2,) current joint velocities [rad/s]
        q_des: np.ndarray,        # (2,) desired joint positions [rad]
        q_dot_ref: np.ndarray,    # (2,) desired joint velocities [rad/s]
        q_ddot_ref: np.ndarray,   # (2,) desired joint accelerations [rad/s²]
        M: np.ndarray,            # (2,2) mass matrix
        h: np.ndarray,            # (2,) bias forces: C(q,q̇)·q̇ + g(q)
    ) -> CTControllerOutput:
        """Compute torques using the full CT law with feedforward.

        a_des = q̈_ref + Kp·(q_des − q) + Kd·(q̇_ref − q̇)
        τ = M(q)·a_des + h(q, q̇)
        """
        # Full CT law with feedforward
        a_des = q_ddot_ref + self.Kp * (q_des - q) + self.Kd * (q_dot_ref - q_dot)

        # Computed torque: τ = M(q)·a_des + h(q, q̇)
        tau_raw = M @ a_des + h

        tau1 = float(tau_raw[0])
        tau2 = float(tau_raw[1])

        # Cable tension decomposition for joint 2
        F_net = tau2 / self.r_p
        T_green = max(F_net, 0.0)
        T_red = max(-F_net, 0.0)
        tau2_cmd = (T_green - T_red) * self.r_p

        tau_clip = np.clip(np.array([tau1, tau2_cmd]), -self.tau_max, self.tau_max)

        return CTControllerOutput(
            tau_clip=tau_clip,
            tau_raw=tau_raw,
            T_green=T_green,
            T_red=T_red,
            q_des=q_des,
            a_des=a_des,
        )

    @property
    def omega_n(self) -> float:
        """Natural frequency [rad/s]."""
        return np.sqrt(self.Kp)

    @property
    def zeta(self) -> float:
        """Damping ratio."""
        wn = self.omega_n
        return self.Kd / (2.0 * wn) if wn > 0 else 0.0


def ik_to_joint_space_references(
    ee_pos_ref: np.ndarray,     # (2,)  target EE [x, y] [m]
    ee_vel_ref: np.ndarray,     # (2,)  target EE velocity [m/s]
    ee_acc_ref: np.ndarray,     # (2,)  target EE acceleration [m/s²]
    L1: float,
    L2: float,
    q_seed: np.ndarray,
    solve_ik_fn,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Convert EE-space references to joint-space using IK + analytical Jacobian.

    Returns
    -------
    q_des      : (2,) desired joint positions [rad]
    q_dot_ref  : (2,) desired joint velocities [rad/s]
    q_ddot_ref : (2,) desired joint accelerations [rad/s²]
    ok         : bool — IK success
    """
    q_des, ok = solve_ik_fn(L1, L2, ee_pos_ref, q_seed)
    if not ok or q_des is None:
        q_des = q_seed.copy()

    # Analytical Jacobian at q_des
    q1d, q2d = q_des
    s1 = np.sin(q1d)
    c1 = np.cos(q1d)
    s12 = np.sin(q1d + q2d)
    c12 = np.cos(q1d + q2d)
    J = np.array([
        [-L1 * s1 - L2 * s12, -L2 * s12],
        [L1 * c1 + L2 * c12, L2 * c12],
    ])

    # J⁻¹ maps EE space → joint space
    J_inv = np.linalg.pinv(J)
    q_dot_ref = J_inv @ ee_vel_ref
    q_ddot_ref = J_inv @ ee_acc_ref  # J̇·q̇ bias dropped (O(q̇²), small)

    return q_des, q_dot_ref, q_ddot_ref, ok
