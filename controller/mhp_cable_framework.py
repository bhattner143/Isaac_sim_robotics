"""Drake cable-actuation layer for MHP Plant A.

Hardware topology::

    Shoulder (q1) — direct-drive MIT motor  →  τ₁
    Elbow    (q2) — ONE MIT motor, antagonistic lower (+Y) / upper (−Y) cables
                    →  T_lower ⊥ T_upper (one slack),  τ₂ = r_p·(T_lower − T_upper)

No series springs — rigid tendons only.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pydrake.all import LeafSystem, MultibodyPlant

from actuators.dummy_mit_motor import DummyMITMotor, DummyMITMotorConfig
from controller.cable_wrench_mhp import (
    CableWrenchConfig,
    distribute_mhp_actuation,
    elbow_torque_from_cable_command,
    tensions_to_mit_feedforward,
)

if TYPE_CHECKING:
    from robots.mhp_manipulator import MHPManipulator


class MHPCableFramework(LeafSystem):
    """Shoulder direct + elbow antagonistic-cable actuation for Plant A.

    Input ports
    -----------
    ``tau_req``     [2]  required joint torques from CT  [Nm]
    ``plant_state`` [n]  from ``plant.get_state_output_port()``

    Output ports
    ------------
    ``actuation``     [2]  torques applied to Plant A  [Nm]
    ``tensions``      [2]  [T_lower (+Y), T_upper (−Y)]  [N]  (one is always 0)
    ``tensions_meas`` [2]  simulated load-cell on each cable  [N]
    ``tau_ff``        [2]  MIT feed-forward [shoulder, elbow motor]
    ``cable_cmd``     [3]  [F_net, F_cmd, tau_elbow_ff]
    ``wrench_flat``   [4]  effective W_eff row-major
    ``diagnostics``   [8]  [τ₁_req, τ₂_req, τ₁_out, τ₂_out, T_lo, T_up, ‖res‖, t]
    """

    def __init__(
        self,
        plant: MultibodyPlant,
        manipulator: "MHPManipulator",
        *,
        wrench_cfg: CableWrenchConfig | None = None,
        mit_kp: tuple[float, float] = (30.0, 15.0),
        mit_kd: tuple[float, float] = (1.5, 0.5),
        tension_kp: float = 0.0,
        elbow_ff_from_cable: bool = True,
        tau_max: float = 10.0,
        dt: float = 0.002,
        use_motor_dynamics: bool = False,
        tension_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self._plant = plant
        self._manip = manipulator
        self._wrench_cfg = wrench_cfg or CableWrenchConfig()
        self._tau_max = float(tau_max)
        self._dt = float(dt)
        self._tension_kp = float(tension_kp)
        self._elbow_ff_from_cable = bool(elbow_ff_from_cable)
        self._rng = np.random.default_rng(0)
        r_p = self._wrench_cfg.r_elbow

        # Motor 0: shoulder direct drive.
        # Motor 1: elbow — single drum, antagonistic lower/upper cables.
        self._shoulder_motor = DummyMITMotor(DummyMITMotorConfig(
            name="shoulder_direct",
            r_spool=1.0,   # not used for direct drive
            kp=mit_kp[0],
            kd=mit_kd[0],
            tau_max=tau_max,
            use_dynamics=use_motor_dynamics,
            tension_noise_std=0.0,
        ))
        self._elbow_motor = DummyMITMotor(DummyMITMotorConfig(
            name="elbow_cable",
            r_spool=r_p,
            kp=mit_kp[1],
            kd=mit_kd[1],
            tau_max=tau_max,
            use_dynamics=use_motor_dynamics,
            tension_noise_std=tension_noise_std,
        ))

        self._plant_ctx = plant.CreateDefaultContext()

        nstate = plant.num_multibody_states()
        self._tau_port = self.DeclareVectorInputPort("tau_req", 2)
        self._state_port = self.DeclareVectorInputPort("plant_state", nstate)

        self.DeclareVectorOutputPort("actuation", 2, self._calc_actuation)
        self.DeclareVectorOutputPort("tensions", 2, self._calc_tensions)
        self.DeclareVectorOutputPort("tensions_meas", 2, self._calc_tensions_meas)
        self.DeclareVectorOutputPort("tau_ff", 2, self._calc_tau_ff)
        self.DeclareVectorOutputPort("cable_cmd", 3, self._calc_cable_cmd)
        self.DeclareVectorOutputPort("wrench_flat", 4, self._calc_wrench_flat)
        self.DeclareVectorOutputPort("diagnostics", 8, self._calc_diagnostics)

        self._cache_t = -np.inf
        self._cache: dict = {}

    def _read_q(self, context) -> tuple[np.ndarray, np.ndarray]:
        state = self._state_port.Eval(context)
        self._plant.SetPositionsAndVelocities(self._plant_ctx, state)
        q = self._manip.get_positions_user_order(self._plant, self._plant_ctx)
        qd = self._manip.get_velocities_user_order(self._plant, self._plant_ctx)
        return q, qd

    def _solve(self, context) -> dict:
        t = context.get_time()
        if t == self._cache_t:
            return self._cache

        tau_req = self._tau_port.Eval(context)
        q, qd = self._read_q(context)

        dist = distribute_mhp_actuation(tau_req, self._wrench_cfg)
        T_lower = dist["T_lower"]
        T_upper = dist["T_upper"]
        F_net = dist["F_net"]

        # Optional inner tension loop on elbow (step 8): correct F_net from sensors.
        F_cmd = F_net
        if self._tension_kp > 0.0:
            # Measure taut side only; slack side reads ~0.
            T_lo_m = self._last_T_meas_lower if hasattr(self, "_last_T_meas_lower") else T_lower
            T_up_m = self._last_T_meas_upper if hasattr(self, "_last_T_meas_upper") else T_upper
            F_meas = T_lo_m - T_up_m
            F_cmd = F_net + self._tension_kp * (F_net - F_meas)

        T_lower_cmd, T_upper_cmd = (
            max(F_cmd, 0.0),
            max(-F_cmd, 0.0),
        )

        if self._elbow_ff_from_cable:
            tau_elbow_ff = elbow_torque_from_cable_command(
                F_cmd, self._wrench_cfg.r_elbow,
            )
        else:
            tau_elbow_ff = dist["tau_elbow_ff"]

        tau_ff = tensions_to_mit_feedforward(
            dist["tau_shoulder_ff"],
            tau_elbow_ff,
        )

        q_des = q.copy()
        qd_des = qd.copy()

        # Shoulder: direct MIT.
        tau_shoulder, _, _ = self._shoulder_motor.step(
            self._dt,
            p=q[0], v=qd[0],
            p_des=q_des[0], v_des=qd_des[0],
            tau_ff=tau_ff[0],
            rng=self._rng,
        )

        # Elbow: single motor driving antagonistic cable pair.
        tau_elbow, _, _ = self._elbow_motor.step(
            self._dt,
            p=q[1], v=qd[1],
            p_des=q_des[1], v_des=qd_des[1],
            tau_ff=tau_ff[1],
            rng=self._rng,
        )

        tau_out = np.clip(
            np.array([tau_shoulder, tau_elbow]),
            -self._tau_max, self._tau_max,
        )

        # Commanded tensions from decomposition (one side slack).
        T_cable = np.array([T_lower_cmd, T_upper_cmd])

        # Simulated load cells: noise on taut strand only.
        noise = 0.0
        if self._elbow_motor.cfg.tension_noise_std > 0.0:
            noise = float(self._rng.normal(0.0, self._elbow_motor.cfg.tension_noise_std))
        T_meas = np.array([
            T_lower_cmd + (noise if T_lower_cmd > 0.0 else 0.0),
            T_upper_cmd + (noise if T_upper_cmd > 0.0 else 0.0),
        ])
        self._last_T_meas_lower = T_meas[0]
        self._last_T_meas_upper = T_meas[1]

        result = {
            "tau_req": tau_req,
            "q": q,
            "W": dist["W_eff"],
            "F_net": F_net,
            "F_cmd": F_cmd,
            "tau_elbow_ff": tau_elbow_ff,
            "T_lower": T_lower_cmd,
            "T_upper": T_upper_cmd,
            "T_cable": T_cable,
            "T_meas": T_meas,
            "tau_ff": tau_ff,
            "tau_out": tau_out,
            "residual": dist["residual"],
            "res_norm": float(np.linalg.norm(dist["residual"])),
        }
        self._cache_t = t
        self._cache = result
        return result

    def _calc_actuation(self, context, output):
        output.SetFromVector(self._solve(context)["tau_out"])

    def _calc_tensions(self, context, output):
        output.SetFromVector(self._solve(context)["T_cable"])

    def _calc_tensions_meas(self, context, output):
        output.SetFromVector(self._solve(context)["T_meas"])

    def _calc_tau_ff(self, context, output):
        output.SetFromVector(self._solve(context)["tau_ff"])

    def _calc_cable_cmd(self, context, output):
        r = self._solve(context)
        output.SetFromVector(np.array([
            r["F_net"], r["F_cmd"], r["tau_elbow_ff"],
        ]))

    def _calc_wrench_flat(self, context, output):
        output.SetFromVector(self._solve(context)["W"].ravel())

    def _calc_diagnostics(self, context, output):
        r = self._solve(context)
        output.SetFromVector(np.array([
            r["tau_req"][0], r["tau_req"][1],
            r["tau_out"][0], r["tau_out"][1],
            r["T_lower"], r["T_upper"],
            r["res_norm"],
            context.get_time(),
        ]))
