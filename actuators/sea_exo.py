"""
actuators/sea_exo.py

Exosuit Series Elastic Actuator — co-contraction stiffness control.

Two CubeMars motors drive antagonistic exo cables through springs on the
forearm side of the elbow pulley.  The system has two modes:

**Deactivated** (transparent / zero-resistance):
    Both exo motors track the elbow encoder continuously so that the spring
    extensions stay at zero.  No additional stiffness is added to the joint.
    The tracking law (Method B, centred elbow pulley):

        θ_m_R / N =  q₂        θ_m_L / N = -q₂

    Spring extensions:
        δ_R = r_exo · (θ_m_R/N − q₂)  = 0
        δ_L = r_exo · (θ_m_L/N + q₂)  = 0

**Activated** (co-contraction stiffness):
    A symmetric angular offset Δθ is added to both motor commands:

        θ_m_R / N =  q₂ + Δθ      θ_m_L / N = -q₂ + Δθ

    Both springs extend by  δ = r_exo · Δθ  (independent of q₂).  A joint
    deflection Δq from the current position produces restoring torque:

        τ_exo ≈ −2 · k_exo · r_exo² · Δq

    giving effective stiffness  k_eff = 2 · k_exo · r_exo².

Sign convention (virtual work):
    δ_R = r_exo · (θ_mR/N − q₂)      →  ∂δ_R/∂q₂ = −r_exo
    δ_L = r_exo · (θ_mL/N + q₂)      →  ∂δ_L/∂q₂ = +r_exo

    Right cable applies torque  τ_R = +r_exo · F_R   (pulls in +q₂ direction)
    Left  cable applies torque  τ_L = −r_exo · F_L   (pulls in −q₂ direction)

    Net:  τ_exo = r_exo · (F_R − F_L)

Physics:
    Each motor is modelled as a 2nd-order rotor (TorqueMotor from
    motor_dynamics.py) — same CubeMars MIT torque mode as the drive SEA.
    Each cable has a unilateral spring (tension ≥ 0).

Diagram wiring:
    [activate_cmd]  →  SEAExoActuator  →  exo_torque  [scalar]
    [plant_state]   →                  →  diagnostics  [vector]
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
    TorqueMotor,
    create_motor_dynamics,
)

if TYPE_CHECKING:
    from actuators.motor import MotorModelConfig


class SEAExoActuator(LeafSystem):
    """Exosuit co-contraction actuator for elbow joint stiffness control.

    Two antagonistic cables, each with its own CubeMars motor and spring,
    wrap around the centred elbow pulley (Method B).  The actuator can be
    activated or deactivated at runtime via the ``activate_cmd`` input port.

    Modes
    -----
    **Deactivated** (``activate_cmd[0] == 0``):
        Motors track encoder: θ_mR/N → q₂,  θ_mL/N → −q₂.
        Springs stay at rest → zero additional torque.

    **Activated** (``activate_cmd[0] == 1``):
        Motors add symmetric offset Δθ = ``activate_cmd[1]``:
        θ_mR/N → q₂ + Δθ,  θ_mL/N → −q₂ + Δθ.
        Effective co-contraction stiffness: k_eff = 2·k_exo·r_exo².

    Input ports
    -----------
    ``activate_cmd``  [2]   [on/off (0 or 1), Δθ (rad)]
    ``plant_state``   [n]   from plant.get_state_output_port()

    Output ports
    ------------
    ``exo_torque``    [1]   additional elbow torque from exo cables [Nm]
    ``diagnostics``   [10]  [δ_R, δ_L, F_R, F_L, θ_mR/N, θ_mL/N,
                             θ̇_mR/N, θ̇_mL/N, τ_exo, activated]
    """

    # Default PD gains for motor encoder tracking (joint-side units).
    # Tuned for CubeMars AK60-6 (J_m ≈ 3.32e-5, N = 6).
    KP_TRACK = 200.0    # [Nm/rad]
    KD_TRACK = 2.0      # [Nm·s/rad]

    # Sub-stepping: run the rotor integrator at dt/N_SUBSTEPS so that the
    # motor-side PD natural frequency stays well within the Nyquist limit.
    # Without this, ω_pd·dt ≈ 4 for AK60-6 at dt=0.01 → inaccurate tracking.
    _N_SUBSTEPS = 10

    def __init__(
        self,
        plant:       MultibodyPlant,
        manipulator,
        k_exo:       float = 200.0,
        b_exo:       float = 2.0,
        r_exo:       float = 0.04775,
        tau_max:     float = 9.0,
        dt:          float = 0.002,
        motor_cfg:   "MotorModelConfig | None" = None,
    ):
        """
        Parameters
        ----------
        plant        Finalized MultibodyPlant.
        manipulator  Robot wrapper instance (provides joint name constants).
        k_exo        Exo cable spring stiffness [N/m].
        b_exo        Exo cable dashpot damping [N·s/m].
        r_exo        Exo elbow pulley radius [m] (Method B: centred on axis).
        tau_max      Output torque saturation per motor [Nm].
        dt           Discrete update period [s].
        motor_cfg    CubeMars motor config for both exo motors (required).
        """
        super().__init__()
        self._plant   = plant
        self._manip   = manipulator
        self._k_exo   = float(k_exo)
        self._b_exo   = float(b_exo)
        self._r_exo   = float(r_exo)
        self._tau_max = float(tau_max)
        self._dt      = float(dt)

        if motor_cfg is None:
            raise ValueError("motor_cfg required for exo motors")
        self._N = motor_cfg.gear_ratio

        # Two independent TorqueMotor instances (right and left cables).
        # Motors use sub-stepped dt for numerical stability.
        J_m = motor_cfg.rotor_inertia_motor
        b_m = motor_cfg.viscous_damping_joint / (self._N ** 2)
        dt_sub = dt / self._N_SUBSTEPS
        _kw = dict(k_s=k_exo, b_c=b_exo, r_p=r_exo, N=self._N, dt=dt_sub,
                   J_m=J_m, b_m=b_m)
        self._motor_R = TorqueMotor(**_kw)
        self._motor_L = TorqueMotor(**_kw)

        # Joint-2 velocity index in Drake's state vector.
        j2 = manipulator.get_joint_by_name(plant, manipulator.JT2_NAME)
        self._q2_v_idx = j2.velocity_start()

        # Discrete state: [θ_mR, θ̇_mR, θ_mL, θ̇_mL,
        #                  q2_anchor, was_active] = 6 scalars.
        # q2_anchor is captured the first time the exo activates so that the
        # pre-tension is centred about the current joint pose (net τ_exo=0
        # at activation, pure restoring torque about q2_anchor thereafter).
        # was_active is a 1/0 flag used to detect the OFF→ON edge.
        self._state_idx = self.DeclareDiscreteState(6)
        self.DeclarePeriodicDiscreteUpdateEvent(dt, 0.0, self._update_motors)

        # Input / output ports.
        nstate = plant.num_multibody_states()
        self._cmd_port   = self.DeclareVectorInputPort("activate_cmd", 2)
        self._state_port = self.DeclareVectorInputPort("plant_state",  nstate)
        # Optional: reference joint trajectory so exo provides stiffness
        # ABOUT q_des (not a static pose).  If left unconnected, the motor
        # falls back to the captured anchor (legacy behaviour).
        self._qdes_port  = self.DeclareVectorInputPort("q_des", 2)

        self.DeclareVectorOutputPort("exo_torque",   1,  self._calc_exo_torque)
        self.DeclareVectorOutputPort("diagnostics",  10, self._calc_diagnostics)

    @property
    def k_eff(self) -> float:
        """Theoretical effective stiffness when fully activated [Nm/rad]."""
        return 2.0 * self._k_exo * self._r_exo ** 2

    # ── Joint state extraction ────────────────────────────────────────────────

    def _read_q2(self, context):
        """Extract q₂ and q̇₂ from the plant_state port."""
        state = self._state_port.Eval(context)
        nq = self._plant.num_positions()
        q2     = float(state[self._q2_v_idx])
        q2_dot = float(state[nq + self._q2_v_idx])
        return q2, q2_dot

    # ── Discrete update ───────────────────────────────────────────────────────

    def _update_motors(self, context, discrete_state):
        """Advance both exo motor rotor states by one timestep.

        Each motor receives a PD tracking torque (joint-side [Nm]) that the
        TorqueMotor.step() converts to motor-side and integrates with the
        rotor + cable-spring dynamics.

        The rotor integration is sub-stepped (_N_SUBSTEPS times per external
        dt) so that the PD natural frequency stays within the Nyquist limit
        of the integrator.

        Right motor tracks q₂ (+ Δθ when active).
        Left  motor sees virtual joint  q_eff = −q₂, tracking −q₂ (+ Δθ).
        """
        all_state = context.get_discrete_state(self._state_idx).value().copy()
        state_R, state_L = all_state[:2], all_state[2:4]
        q2_anchor        = float(all_state[4])
        was_active       = float(all_state[5]) > 0.5
        cmd = self._cmd_port.Eval(context)
        q2, q2_dot = self._read_q2(context)

        activated   = cmd[0] > 0.5
        delta_theta = cmd[1] if activated else 0.0

        # Capture q2 at the OFF→ON transition.  Motors pre-tension about this
        # anchor so that net τ_exo ≈ 0 at activation (no transient kick).
        if activated and not was_active:
            q2_anchor = q2

        # ── Reference-tracking anchor ────────────────────────────────────────
        # DEACTIVATED (transparent): motor tracks the actual joint q2 so that
        #   spring extension δ = r·(θ_m/N − q2) = 0 everywhere → τ_exo = 0.
        # ACTIVATED (co-contraction): motor tracks the REFERENCE q2_des (if
        #   connected) with a symmetric offset Δθ.  This makes both cables
        #   pretensioned about the intended trajectory so under clean
        #   tracking (q2≈q2_des) τ_exo ≈ 0, but under disturbance deflection
        #   τ_exo = −2·k_exo·r_exo²·(q2 − q2_des) pulls back to the reference.
        #   Falls back to captured anchor when q_des port is unconnected.
        if activated:
            if self._qdes_port.HasValue(context):
                q_des_vec = self._qdes_port.Eval(context)
                q2_ref = float(q_des_vec[1])
            else:
                q2_ref = q2_anchor
        else:
            # Transparent: track actual joint so spring stays at rest.
            q2_ref = q2

        # ── Desired motor-side angle (joint-side units) ──────────────────────
        #   right motor: θ_mR/N →  q2_ref + Δθ
        #   left  motor: θ_mL/N → −q2_ref + Δθ
        q_des_R =  q2_ref + delta_theta
        q_des_L = -q2_ref + delta_theta

        for _ in range(self._N_SUBSTEPS):
            # ── PD tracking torque (joint side) → fed to TorqueMotor ─────────
            theta_mR, theta_dot_mR = state_R
            theta_mL, theta_dot_mL = state_L

            # Motor is a position servo on fixed target → velocity ref = 0.
            tau2_R = (self.KP_TRACK * (q_des_R - theta_mR / self._N)
                      - self.KD_TRACK * (theta_dot_mR / self._N))
            tau2_L = (self.KP_TRACK * (q_des_L - theta_mL / self._N)
                      - self.KD_TRACK * (theta_dot_mL / self._N))

            tau2_R = np.clip(tau2_R, -self._tau_max, self._tau_max)
            tau2_L = np.clip(tau2_L, -self._tau_max, self._tau_max)

            # Step each rotor.  Left motor sees virtual joint (−q₂, −q̇₂).
            state_R = self._motor_R.step(state_R, tau2_R,  q2,  q2_dot)
            state_L = self._motor_L.step(state_L, tau2_L, -q2, -q2_dot)

        discrete_state.get_mutable_vector(self._state_idx).SetFromVector(
            np.concatenate([state_R, state_L,
                            [q2_anchor, 1.0 if activated else 0.0]])
        )

    # ── Spring force computation ──────────────────────────────────────────────

    def _compute_exo_forces(self, context):
        """Compute exo cable forces and net torque.

        Spring extensions:
            δ_R = r_exo · (θ_mR/N − q₂)     right cable (upper groove)
            δ_L = r_exo · (θ_mL/N + q₂)     left cable  (lower groove)

        Net exo torque (virtual-work sign, see module docstring):
            τ_exo = r_exo · (F_R − F_L)
        """
        all_state = context.get_discrete_state(self._state_idx).value()
        theta_mR, theta_dot_mR = all_state[0], all_state[1]
        theta_mL, theta_dot_mL = all_state[2], all_state[3]
        q2, q2_dot = self._read_q2(context)

        # Right cable spring (wraps so that +θ_mR → +q₂ direction).
        delta_R     = self._r_exo * (theta_mR / self._N - q2)
        delta_dot_R = self._r_exo * (theta_dot_mR / self._N - q2_dot)
        F_R = max(self._k_exo * delta_R + self._b_exo * delta_dot_R, 0.0)

        # Left cable spring (wraps opposite: +θ_mL → −q₂ direction).
        delta_L     = self._r_exo * (theta_mL / self._N + q2)
        delta_dot_L = self._r_exo * (theta_dot_mL / self._N + q2_dot)
        F_L = max(self._k_exo * delta_L + self._b_exo * delta_dot_L, 0.0)

        # τ_exo = r_exo · (F_R − F_L) — restoring in co-contraction.
        tau_exo = self._r_exo * (F_R - F_L)

        cmd = self._cmd_port.Eval(context)
        activated = float(cmd[0] > 0.5)

        return (delta_R, delta_L, F_R, F_L,
                theta_mR / self._N, theta_mL / self._N,
                theta_dot_mR / self._N, theta_dot_mL / self._N,
                tau_exo, activated)

    # ── Output port callbacks ─────────────────────────────────────────────────

    def _calc_exo_torque(self, context, output):
        *_, tau_exo, activated = self._compute_exo_forces(context)
        # When deactivated, output exactly zero — fully transparent.
        if activated < 0.5:
            output.SetFromVector(np.array([0.0]))
        else:
            output.SetFromVector(np.array([
                np.clip(tau_exo, -self._tau_max, self._tau_max)
            ]))

    def _calc_diagnostics(self, context, output):
        (delta_R, delta_L, F_R, F_L,
         motor_pos_R, motor_pos_L,
         motor_vel_R, motor_vel_L,
         tau_exo, activated) = self._compute_exo_forces(context)
        output.SetFromVector(np.array([
            delta_R,       # [0]  right spring extension δ_R  [m]
            delta_L,       # [1]  left spring extension δ_L   [m]
            F_R,           # [2]  right cable force            [N]
            F_L,           # [3]  left cable force             [N]
            motor_pos_R,   # [4]  right motor joint-side pos   [rad]
            motor_pos_L,   # [5]  left motor joint-side pos    [rad]
            motor_vel_R,   # [6]  right motor joint-side vel   [rad/s]
            motor_vel_L,   # [7]  left motor joint-side vel    [rad/s]
            tau_exo,       # [8]  net exo torque               [Nm]
            activated,     # [9]  1.0 if activated, 0.0 if not
        ]))

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize_at_rest(self, context, q2_init: float) -> None:
        """Set motor states so both exo springs start with δ = 0.

        Right motor: θ_mR = N · q₂        (spring at rest)
        Left  motor: θ_mL = N · (−q₂)     (spring at rest)
        """
        init_state = np.array([
            self._N * q2_init,  0.0,     # right motor [θ_mR, θ̇_mR]
           -self._N * q2_init,  0.0,     # left motor  [θ_mL, θ̇_mL]
            float(q2_init),              # q2_anchor (will be re-captured
                                         # at the OFF→ON activation edge)
            0.0,                         # was_active flag
        ])
        context.get_mutable_discrete_state(self._state_idx).SetFromVector(init_state)
