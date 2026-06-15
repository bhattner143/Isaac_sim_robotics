"""Cable actuation map for the MHP manipulator (hardware topology).

Physical actuation (real robot)::

    Joint 1  (jt_upper_base / shoulder)  — DIRECT DRIVE motor
        τ₁ applied straight to the joint (no cable).

    Joint 2  (jt_lower_upper / elbow)  — ONE motor, TWO antagonistic cables
        Lower cable (+Y spool groove) and upper cable (−Y spool groove) are
        wound in opposite directions on the same drum.  Only one side is taut
        at any instant; the other is slack.

        F_net   = τ₂ / r_p                         [N]
        T_lower = max( F_net, 0)   (+Y retracts)  [N]
        T_upper = max(-F_net, 0)   (−Y retracts)  [N]
        τ₂      = r_p · (T_lower − T_upper)       [Nm]

Controller cycle::

    CT → τ_req  →  shoulder MIT (direct)  +  elbow tension split → elbow MIT
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Elbow cable pitch radius at the joint drum / roller [m]
R_ELBOW_PULLEY_MM = 39.4   # elbow roller OD 78.8 mm


@dataclass
class CableWrenchConfig:
    """MHP cable / pulley geometry."""
    r_elbow: float = R_ELBOW_PULLEY_MM * 1e-3   # elbow antagonistic pair [m]


def decompose_antagonistic_tensions(
    F_net: float,
) -> tuple[float, float]:
    """Split signed net cable force into non-negative antagonistic tensions.

    Returns
    -------
    T_lower : tension in lower (+Y) cable  [N]  (≥ 0)
    T_upper : tension in upper (−Y) cable  [N]  (≥ 0)

    Exactly one is non-zero unless F_net = 0.
    """
    T_lower = float(max(F_net, 0.0))
    T_upper = float(max(-F_net, 0.0))
    return T_lower, T_upper


def distribute_mhp_actuation(
    tau_req: np.ndarray,
    cfg: CableWrenchConfig | None = None,
) -> dict:
    """Map CT joint torques to shoulder direct + elbow antagonistic cables.

    Parameters
    ----------
    tau_req : (2,)  [τ₁, τ₂] from computed torque  [Nm]

    Returns
    -------
    dict with keys:
        tau_shoulder_ff, tau_elbow_ff  — MIT feed-forward [Nm]
        F_net, T_lower, T_upper        — elbow cable forces [N]
        tau_elbow_cmd                  — τ₂ after unilateral split [Nm]
        residual                       — [0, τ₂_req − τ_elbow_cmd]
        W_eff                          — (2, 2) effective map for logging
    """
    cfg = CableWrenchConfig() if cfg is None else cfg
    tau_req = np.asarray(tau_req, dtype=float).ravel()
    r_p = cfg.r_elbow

    # Shoulder: direct drive — MIT feed-forward equals requested torque.
    tau_shoulder_ff = float(tau_req[0])

    # Elbow: one motor, antagonistic lower/upper cables.
    F_net = float(tau_req[1] / r_p) if abs(r_p) > 1e-12 else 0.0
    T_lower, T_upper = decompose_antagonistic_tensions(F_net)
    tau_elbow_cmd = r_p * (T_lower - T_upper)
    tau_elbow_ff = float(tau_req[1])

    residual = np.array([0.0, tau_req[1] - tau_elbow_cmd])

    # Effective wrench for diagnostics: [τ₁, τ₂]ᵀ ≈ W_eff @ [τ_m1, F_net]ᵀ
    W_eff = np.array([[1.0, 0.0],
                      [0.0, r_p]], dtype=float)

    return {
        "tau_shoulder_ff": tau_shoulder_ff,
        "tau_elbow_ff": tau_elbow_ff,
        "F_net": F_net,
        "T_lower": T_lower,
        "T_upper": T_upper,
        "tau_elbow_cmd": tau_elbow_cmd,
        "residual": residual,
        "W_eff": W_eff,
    }


# Backward-compatible aliases
def build_wrench_matrix_mhp(
    q1: float,
    q2: float,
    cfg: CableWrenchConfig | None = None,
) -> np.ndarray:
    """Effective W for MHP: shoulder direct + elbow cable scalar r_p."""
    del q1, q2  # planar pitch radii constant away from singularities
    cfg = CableWrenchConfig() if cfg is None else cfg
    return np.array([[1.0, 0.0],
                     [0.0, cfg.r_elbow]], dtype=float)


def solve_tension_distribution(
    W: np.ndarray,
    tau_req: np.ndarray,
    *,
    t_min: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Elbow antagonistic split; shoulder handled outside W."""
    del W, t_min
    dist = distribute_mhp_actuation(tau_req)
    T = np.array([dist["T_lower"], dist["T_upper"]])
    return T, dist["residual"]


def tensions_to_mit_feedforward(
    tau_shoulder: float,
    tau_elbow: float,
) -> np.ndarray:
    """MIT τ_ff for [shoulder direct, elbow cable motor]."""
    return np.array([tau_shoulder, tau_elbow], dtype=float)


def elbow_torque_from_cable_command(F_cmd: float, r_p: float) -> float:
    """Convert commanded net cable force to elbow MIT feed-forward torque.

    τ_ff₂ = r_p · F_cmd   [Nm]
    """
    return float(r_p * F_cmd)
