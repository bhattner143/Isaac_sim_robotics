"""
controller/ik_system.py

IK-based PD torque controller (LeafSystem) for CupManipulatorTendon.

CupManipulatorIKSystem tracks Cartesian EE targets using analytical
2-R IK and a PD control law with cable tension decomposition.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from pydrake.all import (
    MultibodyPlant,
    LeafSystem,
    BasicVector,
)

if TYPE_CHECKING:
    from robots.cup_manipulator_tendon import CupManipulatorTendon


class CupManipulatorIKSystem(LeafSystem):
    """Drake LeafSystem: analytical IK → PD torque controller for CupManipulatorTendon.

    Wires into a DiagramBuilder alongside MultibodyPlant so the robot
    tracks Cartesian EE targets in closed-loop simulation.

    The two joints have **physically different** actuation mechanisms and
    therefore use **different gain units**:

    Joint 1  (link1_base) — **direct drive**
        The motor shaft couples to the joint directly (possibly through a
        gearbox, but no cable).  The commanded quantity is joint torque τ1.
        Gains:  ``Kp1``  [Nm/rad],  ``Kd1``  [Nm·s/rad]

    Joint 2  (link2_link1) — **single cable wrapped around motor pulley**
        A single cable is wrapped around the motor pulley (r_p ≈ 47.75 mm).
        Both free ends of the cable attach to opposite sides of link2.
        When the motor rotates by Δq2, one end ("green") retracts by r_p·Δq2
        while the other ("red") extends by the same amount — total cable length
        is conserved.  The tension difference drives the joint:

            τ2 = (T_green − T_red) · r_p     [τ = J_c^T · T,  J_c = r_p]

        Cables can only pull (T ≥ 0), so the two tensions decompose from the
        signed PD command F_net = Kp_cable·Δl − Kd_cable·ḷ  [N]:

            T_green = max(F_net,  0)   [N]   retracting side tightens
            T_red   = max(−F_net, 0)  [N]   extending side tightens on reversal

        One cable half is always slack at any instant (no pre-tension model).
        Gains:  ``Kp_cable``  [N/m],  ``Kd_cable``  [N·s/m]

    Effective joint-space equivalents (for reference):
        Kp_j2 = Kp_cable · r_p²
        Kd_j2 = Kd_cable · r_p²

    Input ports
    -----------
    ``desired_ee_pos``  [2]   target (x, y) in world frame [m]
    ``plant_state``     [n]   from ``plant.get_state_output_port()``

    Output ports
    ------------
    ``actuation``       [2]   joint torques → ``plant.get_actuation_input_port()``
                               τ1 = Kp1·Δq1 − Kd1·q1̇
                               τ2 = (T_green − T_red)·r_p  [= F_net·r_p]
    ``joint_positions`` [2]   IK solution (q1, q2) for logging/viz [rad]
    ``cable_lengths``   [2]   cable displacements from q2=0 [m]
                               delta_green =  r_p · q2_des  (retracting side)
                               delta_red   = −delta_green    (extending side)
    ``cable_tensions``  [2]   individual half-cable tensions [N]
                               T_green = max(F_net, 0)    (tightens when τ2 > 0)
                               T_red   = max(−F_net, 0)   (tightens when τ2 < 0)
                               one side always slack (no pre-tension model)

    Parameters
    ----------
    plant        : Finalized MultibodyPlant that the system will control.
    manipulator  : CupManipulatorTendon instance (provides IK + PULLEY_RADIUS).
    Kp1          : Joint 1 direct-drive position gain  [Nm/rad].
    Kd1          : Joint 1 direct-drive damping gain   [Nm·s/rad].
    Kp_cable     : Joint 2 cable stiffness             [N/m].
    Kd_cable     : Joint 2 cable damping               [N·s/m].
    tau_max      : Saturation limit on joint torque    [Nm].
                   Equivalent cable-force limit: F_max = tau_max / r_p.

    Typical diagram wiring
    ----------------------
    ::

        builder = DiagramBuilder()
        plant, sg = AddMultibodyPlantSceneGraph(builder, time_step=0.002)
        # ... load URDF, finalize ...
        ik_sys = builder.AddSystem(CupManipulatorIKSystem(plant, manipulator))
        ref    = builder.AddSystem(ConstantVectorSource(np.array([0.4, 0.3])))
        builder.Connect(ref.get_output_port(),           ik_sys.GetInputPort("desired_ee_pos"))
        builder.Connect(plant.get_state_output_port(),   ik_sys.GetInputPort("plant_state"))
        builder.Connect(ik_sys.GetOutputPort("actuation"), plant.get_actuation_input_port())
    """

    def __init__(
        self,
        plant: MultibodyPlant,
        manipulator: "CupManipulatorTendon",
        Kp1: float      = 80.0,    # Joint 1 direct-drive stiffness   [Nm/rad]
        Kd1: float      = 16.0,    # Joint 1 direct-drive damping     [Nm·s/rad]
        Kp_cable: float = 260.0,   # Joint 2 cable spring constant    [N/m]
                                   #   → Kp_j2 = Kp_cable·r_p² ≈ 0.59 Nm/rad
                                   #   → ωn ≈ 15 rad/s for Izz_link2 ≈ 0.0026 kg·m²
        Kd_cable: float = 35.0,    # Joint 2 cable damping constant   [N·s/m]
                                   #   → Kd_j2 = Kd_cable·r_p² ≈ 0.08 Nm·s/rad  (ζ≈1)
        tau_max: float  = 10.0,    # Joint torque saturation limit    [Nm]
                                   #   ↔ cable force limit: F_max = tau_max/r_p ≈ 209 N
    ) -> None:
        super().__init__()
        self._plant       = plant
        self._manip       = manipulator
        self._ik          = manipulator.ik
        self._r_p         = manipulator.PULLEY_RADIUS
        self._Kp1         = float(Kp1)
        self._Kd1         = float(Kd1)
        self._Kp_cable    = float(Kp_cable)
        self._Kd_cable    = float(Kd_cable)
        self._tau_max     = float(tau_max)
        # Private plant context for FK / Jacobian calls — never used for
        # integration, only for kinematics queries inside output callbacks.
        self._plant_ctx = plant.CreateDefaultContext()

        # Pre-cache L1, L2: constant URDF geometry, never changes at runtime.
        # Avoids calling get_link_lengths() (which creates a context) on every
        # output port evaluation.
        self._L1, self._L2 = manipulator.ik.get_link_lengths(plant)

        # Persistent IK seed: stores last successful q_des so the IK always
        # warm-starts from the previous solution rather than from the current
        # joint angle.  This prevents elbow-solution flipping when the robot
        # moves through configurations where both elbow signs are equidistant
        # from the instantaneous q.
        self._last_q_des: np.ndarray = np.zeros(2)

        # Per-timestep IK cache: all 3 output port callbacks share one solve.
        # Key = context.get_time(); avoids 3× redundant IK + context creation.
        self._t_cache:     float      = -np.inf
        self._q_des_cache: np.ndarray = np.zeros(2)
        self._q_cache:     np.ndarray = np.zeros(2)
        self._q_dot_cache: np.ndarray = np.zeros(2)

        nstate = plant.num_multibody_states()   # nq + nv

        # ── Input ports ───────────────────────────────────────────────────
        self._ee_port    = self.DeclareVectorInputPort("desired_ee_pos", 2)
        self._state_port = self.DeclareVectorInputPort("plant_state", nstate)

        # ── Output ports ──────────────────────────────────────────────────
        self.DeclareVectorOutputPort("actuation",       2, self._calc_torques)
        self.DeclareVectorOutputPort("joint_positions", 2, self._calc_joint_positions)
        self.DeclareVectorOutputPort("cable_lengths",   2, self._calc_cable_lengths)
        self.DeclareVectorOutputPort("cable_tensions",  2, self._calc_cable_tensions)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _solve_ik(self, context) -> tuple:
        """Read ports, sync internal plant context, run analytical IK.

        Per-timestep caching
        ~~~~~~~~~~~~~~~~~~~~
        All three output port callbacks (_calc_torques, _calc_joint_positions,
        _calc_cable_lengths) call this method.  Without caching the IK would
        be solved 3× per timestep.  We use context.get_time() as a key: if the
        time is the same as the last call, return the pre-computed result.

        Seed strategy
        ~~~~~~~~~~~~~
        Use ``self._last_q_des`` (previous step's IK solution) rather than the
        current joint angle ``q``.  This keeps the IK on the same elbow branch:

        * Near target: q ≈ q_des ≈ last_q_des — both seeds give same result.
        * Far away: last_q_des tracks the desired branch even while the plant
          is still moving, preventing inter-step elbow-solution flipping.

        L1 / L2 shortcut
        ~~~~~~~~~~~~~~~~
        Link lengths are URDF constants cached at construction in self._L1 /
        self._L2.  We call _solve_2r_core directly to skip get_link_lengths()
        (which would create a new Drake context every call).

        Returns
        -------
        q_des : ndarray (2,)  desired joint angles in user order [q1, q2]
        q     : ndarray (2,)  current joint angles in user order
        q_dot : ndarray (2,)  current joint velocities in user order
        """
        t = context.get_time()
        if t == self._t_cache:
            return self._q_des_cache, self._q_cache, self._q_dot_cache

        state  = self._state_port.Eval(context)
        ee_des = self._ee_port.Eval(context)

        # Sync internal context so FK / Jacobian calls see current state
        self._plant.SetPositionsAndVelocities(self._plant_ctx, state)

        # Positions and velocities in user-defined order [q1, q2]
        q     = self._manip.get_positions_user_order(self._plant, self._plant_ctx)
        q_dot = self._manip.get_velocities_user_order(self._plant, self._plant_ctx)

        # Use persistent seed to keep IK on the same elbow branch.
        # Fall back to current q on the very first call (seed is all zeros).
        seed = self._last_q_des if np.any(self._last_q_des != 0.0) else q

        # Direct 2R solve with pre-cached L1, L2 — no context creation needed.
        best_q, ok = self._ik._solve_2r_core(self._L1, self._L2, ee_des, seed)
        if ok:
            # Update seed only on success to avoid drifting on out-of-reach targets.
            self._last_q_des = best_q.copy()
            q_des = best_q
        else:
            # Hold last known good target — do not jump to current q.
            q_des = self._last_q_des.copy()

        # Store in per-timestep cache
        self._t_cache     = t
        self._q_des_cache = q_des
        self._q_cache     = q
        self._q_dot_cache = q_dot

        return q_des, q, q_dot

    # ── Output port callbacks ─────────────────────────────────────────────

    def _calc_torques(self, context, output):
        """Motor commands for the two physically distinct actuation paths.

        Joint 1 (link1_base) — direct drive
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        The motor torque couples straight to the joint (no cable):

            τ1 = Kp1·(q1_des − q1) − Kd1·q1̇     [Nm]

        Joint 2 (link2_link1) — single cable wrapped around motor pulley
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        A single cable is wrapped around the motor pulley.  Both free ends
        attach to opposite sides of link2.  Motor rotation by Δq2 causes one
        end to retract and the other to extend by r_p·Δq2.

        PD command in cable space:
            Δl    = r_p · (q2_des − q2)              [m]   cable displacement error
            ḷ     = r_p · q2̇                        [m/s] cable speed
            F_net = Kp_cable·Δl − Kd_cable·ḷ         [N]   signed net tension

        Decompose into individual cable tensions (cables can only pull, T ≥ 0):
            T_green = max(F_net,  0)   [N]   retracting side tightens
            T_red   = max(−F_net, 0)  [N]   extending side tightens on reversal

        Net joint torque via τ = J_c^T · T  (cable Jacobian J_c = r_p):
            τ2 = (T_green − T_red) · r_p = F_net · r_p     [Nm]

        Both torques are clamped to ±tau_max [Nm].
        Equivalent cable-force clamp: F_max = tau_max / r_p.
        """
        q_des, q, q_dot = self._solve_ik(context)

        # ── Joint 1: direct drive ────────────────────────────────────────────
        tau1 = self._Kp1 * (q_des[0] - q[0]) - self._Kd1 * q_dot[0]

        # ── Joint 2: single cable, tension decomposition ─────────────────────
        delta_l = self._r_p * (q_des[1] - q[1])     # cable displacement error [m]
        l_dot   = self._r_p * q_dot[1]               # cable speed              [m/s]
        F_net   = self._Kp_cable * delta_l - self._Kd_cable * l_dot   # [N]
        # Positivity constraint: cables can only pull, so decompose into two
        # non-negative tensions.  One side is always slack (no pre-tension).
        T_green = max(F_net, 0.0)          # retracting side
        T_red   = max(-F_net, 0.0)         # extending side (reversal)
        tau2    = (T_green - T_red) * self._r_p      # = F_net · r_p   [Nm]

        tau = np.clip(np.array([tau1, tau2]), -self._tau_max, self._tau_max)
        output.SetFromVector(tau)

    def _calc_joint_positions(self, context, output):
        """IK solution q_des in user order [q1, q2] [rad]."""
        q_des, _, _ = self._solve_ik(context)
        output.SetFromVector(q_des)

    def _calc_cable_lengths(self, context, output):
        """Cable displacements of each half of the single drive cable [m].

        When q2_des changes by δ, the retracting (green) side shortens by
        r_p·δ and the extending (red) side lengthens by the same amount,
        keeping total cable length constant.

        delta_green =  r_p · q2_des   (positive q2 → green retracts)
        delta_red   = −delta_green     (red extends by equal amount)
        """
        q_des, _, _ = self._solve_ik(context)
        delta_G =  self._r_p * q_des[1]
        delta_R = -delta_G
        output.SetFromVector(np.array([delta_G, delta_R]))

    def _calc_cable_tensions(self, context, output):
        """Individual tensions in each half of the single drive cable [N].

        T_green and T_red are the tensions in the retracting and extending
        cable halves respectively.  Both are ≥ 0 (cables can only pull).
        One side is always slack in the no-pre-tension model.

        F_net   = Kp_cable·Δl − Kd_cable·ḷ   (signed PD command)
        T_green = max(F_net,  0)   [N]
        T_red   = max(−F_net, 0)   [N]

        Net joint torque: τ2 = (T_green − T_red) · r_p  [Nm]
        """
        q_des, q, q_dot = self._solve_ik(context)
        delta_l = self._r_p * (q_des[1] - q[1])
        l_dot   = self._r_p * q_dot[1]
        F_net   = self._Kp_cable * delta_l - self._Kd_cable * l_dot
        T_green = max(F_net, 0.0)
        T_red   = max(-F_net, 0.0)
        output.SetFromVector(np.array([T_green, T_red]))

