"""
robots/cup_manipulator_tendon.py
---------------------------------
Backward-compatible re-export.

The full CupManipulator implementation (formerly CupManipulatorTendon) lives in
test_drive_pulley.py alongside the cable classes (PulleyBase, CableRig, etc.)
so that the module is self-contained with no circular import.

All existing code that imports from this module continues to work unchanged::

    from robots.cup_manipulator_tendon import CupManipulatorTendon
    from robots.cup_manipulator_tendon import create_cable_manipulator_config

Dependency direction (no cycle):
    cup_manipulator_tendon  ->  test_drive_pulley  ->  robots.cup_manipulator
"""

import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pydrake.geometry import Rgba, Cylinder
from pydrake.all import (
    MultibodyPlant,
    Parser,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
    FixedOffsetFrame,
    RevoluteJoint,
    PrismaticJoint,
    JacobianWrtVariable,
)
from robots.cup_manipulator import RobotBase
from configs.robot.robot_types import ManipulatorConfig, JointConfig, Pose
from actuators.motor import get_motor, MotorModelConfig

from cable import (
    PulleyBase,
    BigPulley,
    CableRig,
    _parse_urdf_part_origins,
)

from termcolor import colored

class CupManipulatorTendon(RobotBase):
    """Cable-driven (tendon) 2-DOF manipulator for Drake.

    Wraps manipulator_cable.urdf which uses a belt/pulley transmission.
    Joint names:
        JT1_NAME = "link1_base"   (q1)
        JT2_NAME = "link2_link1"  (q2)
    """

    JT1_NAME  = "link1_base"
    JT2_NAME  = "link2_link1"
    ACT1_NAME = f"tau_{JT1_NAME}"
    ACT2_NAME = f"tau_{JT2_NAME}"

    BASE_LINK_NAME = "base_mate"
    LINK2_NAME     = "link2_tendon"

    EE_XYZ_LINK2  = np.array([0.19, 0.0, 0.0515])
    EE_RPY_LINK2  = np.array([0.0, 0.0, 0.0])
    EE_FRAME_NAME = "tendon_ee"
    EE_OFFSET     = EE_XYZ_LINK2

    # Pitch radius of the HTD 5M 60T belt/pulley transmission on joint 2.
    # Derived from BigPulley.belt_teeth and BigPulley.belt_pitch_m so any
    # change to the pulley spec propagates here automatically.
    PULLEY_RADIUS: float = BigPulley.pitch_radius   # N·p / 2π ≈ 47.746 mm [m]

    def __init__(self, config: ManipulatorConfig, enable_visualization: bool = True):
        super().__init__(config)
        self.joint_names: List[str]    = [self.JT1_NAME, self.JT2_NAME]
        self.actuator_names: List[str] = []
        self.enable_visualization      = enable_visualization
        self.rig                       = None  # CableRig — set via init_cable_rig()
        self.ik                        = CupManipulatorIK(self)
        # Instantiate motor model from config name, or None if not specified.
        # Motor parameters (damping, effort limits, inertia) take priority over
        # the raw JointConfig values when a motor is configured.
        _motor_name = getattr(config, 'motor_name', None)
        self.motor: Optional[MotorModelConfig] = (
            get_motor(_motor_name) if _motor_name else None
        )
        if self.motor is not None:
            print(colored(f"✓ Motor model loaded: {type(self.motor).__name__}  "
                          f"τ_peak={self.motor.peak_torque_joint} Nm  "
                          f"b={self.motor.viscous_damping_joint} Nm·s/rad  "
                          f"N={self.motor.gear_ratio}", 'magenta'))

    # ── URDF loading ────────────────────────────────────────────────────────

    def load_urdf_to_plant(self, plant: MultibodyPlant, parser: Parser) -> int:
        model_instance = super().load_urdf_to_plant(plant, parser)
        self.JT1_NAME    = "link1_base"
        self.JT2_NAME    = "link2_link1"
        self.ACT1_NAME   = f"tau_{self.JT1_NAME}"
        self.ACT2_NAME   = f"tau_{self.JT2_NAME}"
        self.joint_names = [self.JT1_NAME, self.JT2_NAME]
        print(colored(
            f"✓ CupManipulator: joints confirmed: [{self.JT1_NAME}, {self.JT2_NAME}]",
            'green'
        ))
        return model_instance
    
    # ── Weld base ───────────────────────────────────────────────────────────

    def weld_base_to_world(
        self,
        plant: MultibodyPlant,
        position:    np.ndarray = np.array([0.0, 0.0, 0.0]),
        orientation: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ):
        if plant.is_finalized():
            raise RuntimeError("Cannot weld base after plant is finalized")
        base_body = plant.GetBodyByName(self.BASE_LINK_NAME, self.model_instance)
        X_WB      = RigidTransform(RollPitchYaw(orientation), position)
        plant.WeldFrames(plant.world_frame(), base_body.body_frame(), X_WB)
        print(colored(
            f"✓ Welded '{self.BASE_LINK_NAME}' to world at pos={position}, rpy={orientation}",
            'green'
        ))

    # ── End-effector frame ──────────────────────────────────────────────────

    def add_end_effector_frame(self, plant: MultibodyPlant):
        if plant.is_finalized():
            raise RuntimeError("Cannot add EE frame after plant is finalized")
        link2_body = plant.GetBodyByName(self.LINK2_NAME, self.model_instance)
        X_L2_EE    = RigidTransform(RollPitchYaw(self.EE_RPY_LINK2), self.EE_XYZ_LINK2)
        try:
            return plant.GetFrameByName(self.EE_FRAME_NAME, self.model_instance)
        except Exception:
            pass
        return plant.AddFrame(
            FixedOffsetFrame(
                self.EE_FRAME_NAME,
                link2_body.body_frame(),
                X_L2_EE,
                self.model_instance,
            )
        )

    def get_end_effector_frame(self, plant: MultibodyPlant):
        return plant.GetFrameByName(self.EE_FRAME_NAME, self.model_instance)

    # ── Joint actuators ─────────────────────────────────────────────────────

    def set_joint_properties(self, plant: MultibodyPlant):
        """Set joint damping. Motor viscous_damping_joint overrides JointConfig.damping."""
        print(colored(f"\nSetting joint properties for '{self.name}':", 'yellow'))
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_mutable_joint(joint_idx)
            joint_name = joint.name()
            if joint.num_velocities() > 0 and joint_name in self.config.joint_configs:
                if self.motor is not None:
                    damping = self.motor.viscous_damping_joint
                    source  = f"motor:{type(self.motor).__name__}"
                else:
                    damping = self.config.joint_configs[joint_name].damping
                    source  = "JointConfig"
                if hasattr(joint, 'set_default_damping_vector') and damping > 0:
                    joint.set_default_damping_vector([damping])
                print(colored(f"  ✓ {joint_name}: damping={damping:.4f} Nm·s/rad  [{source}]", 'cyan'))
        print(colored(f"✓ Joint properties configured", 'green'))

    def add_joint_actuators(self, plant: MultibodyPlant):
        if plant.is_finalized():
            raise RuntimeError("Cannot add actuators after plant is finalized")
        jt1 = self.get_joint_by_name(plant, self.JT1_NAME)
        jt2 = self.get_joint_by_name(plant, self.JT2_NAME)
        if self.motor is not None:
            effort_limit  = self.motor.peak_torque_joint
            rotor_inertia = self.motor.rotor_inertia_motor
            gear_ratio    = self.motor.gear_ratio
        else:
            effort_limit  = np.inf
            rotor_inertia = 0.0
            gear_ratio    = 1.0
        act1 = plant.AddJointActuator(self.ACT1_NAME, jt1, effort_limit)
        act2 = plant.AddJointActuator(self.ACT2_NAME, jt2, effort_limit)
        if self.motor is not None:
            act1.set_default_rotor_inertia(rotor_inertia)
            act1.set_default_gear_ratio(gear_ratio)
            act2.set_default_rotor_inertia(rotor_inertia)
            act2.set_default_gear_ratio(gear_ratio)
        self.actuator_names = [self.ACT1_NAME, self.ACT2_NAME]
        print(colored(f"✓ Added actuators: {self.ACT1_NAME}, {self.ACT2_NAME}", 'green'))
        if self.motor is not None:
            print(colored(f"  τ_limit={effort_limit} Nm  I_rotor={rotor_inertia} kg·m²  "
                          f"N={gear_ratio}  [motor:{type(self.motor).__name__}]", 'cyan'))

    # ── EE kinematics ───────────────────────────────────────────────────────

    def get_end_effector_position(self, plant: MultibodyPlant, context) -> np.ndarray:
        ee_frame = self.get_end_effector_frame(plant)
        X_WE     = plant.CalcRelativeTransform(context, plant.world_frame(), ee_frame)
        return X_WE.translation()

    def CalcPosition(self, plant: MultibodyPlant, context) -> np.ndarray:
        return self.get_end_effector_position(plant, context)

    # ── State helpers ───────────────────────────────────────────────────────

    def get_state_from_plant(self, plant: MultibodyPlant, context) -> np.ndarray:
        return plant.GetPositionsAndVelocities(context, self.model_instance)

    def set_state_in_plant(self, plant: MultibodyPlant, context, user_state: np.ndarray):
        q1, q2, q1_dot, q2_dot = user_state
        self.set_jt([self.JT1_NAME, self.JT2_NAME], plant, context, [q1, q2])
        self.set_jt_velocity([self.JT1_NAME, self.JT2_NAME], plant, context, [q1_dot, q2_dot])

    def get_positions_user_order(self, plant: MultibodyPlant, context) -> np.ndarray:
        return np.array(self.get_jt([self.JT1_NAME, self.JT2_NAME], plant, context))

    def set_positions_user_order(self, plant: MultibodyPlant, context, user_positions):
        if isinstance(user_positions, dict):
            for joint_name, angle in user_positions.items():
                self.set_jt([joint_name], plant, context, [angle])
        else:
            q1, q2 = user_positions
            self.set_jt([self.JT1_NAME, self.JT2_NAME], plant, context, [q1, q2])

    def get_velocities_user_order(self, plant: MultibodyPlant, context) -> np.ndarray:
        return self.get_jt_velocity([self.JT1_NAME, self.JT2_NAME], plant, context)

    def set_velocities_user_order(self, plant: MultibodyPlant, context, user_velocities):
        if isinstance(user_velocities, dict):
            for joint_name, velocity in user_velocities.items():
                self.set_jt_velocity([joint_name], plant, context, [velocity])
        else:
            q1_dot, q2_dot = user_velocities
            self.set_jt_velocity([self.JT1_NAME, self.JT2_NAME], plant, context, [q1_dot, q2_dot])

    def get_joint_positions(self, plant: MultibodyPlant, context) -> dict:
        positions = {}
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0:
                if isinstance(joint, RevoluteJoint):
                    positions[joint.name()] = joint.get_angle(context)
                elif isinstance(joint, PrismaticJoint):
                    positions[joint.name()] = joint.get_translation(context)
        return positions

    def get_joint_velocities(self, plant: MultibodyPlant, context) -> dict:
        velocities = {}
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() > 0:
                if isinstance(joint, RevoluteJoint):
                    velocities[joint.name()] = joint.get_angular_rate(context)
                elif isinstance(joint, PrismaticJoint):
                    velocities[joint.name()] = joint.get_translation_rate(context)
        return velocities

    # ── Joint helpers ───────────────────────────────────────────────────────

    def get_joint_by_name(self, plant: MultibodyPlant, joint_name: str):
        return plant.GetJointByName(joint_name, self.model_instance)

    def get_jt(self, joint_name, plant: MultibodyPlant, context):
        if isinstance(joint_name, list):
            return np.array([
                self.get_joint_by_name(plant, n).get_angle(context) for n in joint_name
            ])
        return self.get_joint_by_name(plant, joint_name).get_angle(context)

    def set_jt(self, joint_name, plant: MultibodyPlant, context, angle):
        if isinstance(joint_name, list):
            angles = np.atleast_1d(angle)
            for name, ang in zip(joint_name, angles):
                self.get_joint_by_name(plant, name).set_angle(context, float(ang))
        else:
            self.get_joint_by_name(plant, joint_name).set_angle(context, float(angle))

    def get_jt_velocity(self, joint_name, plant: MultibodyPlant, context):
        if isinstance(joint_name, list):
            return np.array([
                self.get_joint_by_name(plant, n).get_angular_rate(context) for n in joint_name
            ])
        return self.get_joint_by_name(plant, joint_name).get_angular_rate(context)

    def set_jt_velocity(self, joint_name, plant: MultibodyPlant, context, velocity):
        if isinstance(joint_name, list):
            velocities = np.atleast_1d(velocity)
            for name, vel in zip(joint_name, velocities):
                self.get_joint_by_name(plant, name).set_angular_rate(context, float(vel))
        else:
            self.get_joint_by_name(plant, joint_name).set_angular_rate(context, float(velocity))

    # ── Inverse kinematics (delegates to self.ik) ──────────────────────────

    def compute_ik_analytical(self, plant, target_xy, q_seed, **kwargs):
        """Closed-form 2-R IK. Delegates to self.ik.analytical()."""
        return self.ik.analytical(plant, target_xy, q_seed, **kwargs)

    def jacobian_joint_space_to_ee_velocity(self, plant, context) -> np.ndarray:
        """World-XY geometric Jacobian J ∈ ℝ^{2×2}. Delegates to self.ik.jacobian()."""
        return self.ik.jacobian(plant, context)

    def jacobian_hybrid(self, plant, context) -> np.ndarray:
        """Hybrid Jacobian J_h ∈ ℝ^{2×2} (actuation space). Delegates to self.ik.jacobian_hybrid()."""
        return self.ik.jacobian_hybrid(plant, context)

    def compute_velocity_ik(self, plant, context, ee_velocity_xy, damping=1e-4) -> np.ndarray:
        """Joint-space velocity IK. Delegates to self.ik.velocity()."""
        return self.ik.velocity(plant, context, ee_velocity_xy, damping)

    def compute_velocity_ik_hybrid(self, plant, context, ee_velocity_xy, damping=1e-4) -> np.ndarray:
        """Actuation-space velocity IK. Delegates to self.ik.hybrid()."""
        return self.ik.hybrid(plant, context, ee_velocity_xy, damping)

    # ── Cable rig ───────────────────────────────────────────────────────────

    def init_cable_rig(self, urdf_path: str = None, assets_dir: str = None,
                       springs_enabled: bool = True) -> None:
        """Initialize the cable rig.  Call after the plant is built."""
        if urdf_path is None:
            urdf_path = self.config.urdf_path
        if assets_dir is None:
            assets_dir = str(Path(urdf_path).parent / "assets")
        PulleyBase._urdf_origins = _parse_urdf_part_origins(urdf_path)
        PulleyBase.assets_dir    = assets_dir
        self.rig = CableRig(springs_enabled=springs_enabled)

    def compute_tangents(self, plant, plant_context) -> None:
        """Recompute all cable tangent contacts at the current joint configuration."""
        if self.rig is None:
            raise RuntimeError("init_cable_rig() must be called before compute_tangents()")
        self.rig.compute_tangents(plant, plant_context, self)

    def length_cable_route(
        self,
        route,
        plant: MultibodyPlant,
        context,
        start_label: str = "",
    ) -> float:
        """World-frame path length along a cable route from a named waypoint.

        Sums the Euclidean distances between consecutive world-frame waypoints
        on *route*, starting from the first segment whose label begins with
        *start_label*.  If *start_label* is empty or not found the full route
        is used.

        Args:
            route:       CableRoute (e.g. self.rig.cable_green).
            plant:       Finalized MultibodyPlant.
            context:     Plant context at the desired configuration.
            start_label: Label prefix of the first waypoint to include
                         (e.g. ``"Drive exit B_R"``).  Defaults to ``""``
                         which matches every label and returns the full length.

        Returns:
            Cable length in metres.
        """
        pts    = route.world_points(plant, context, self)   # (N, 3)
        labels = [cfg.label for cfg, _ in route.segments]
        try:
            start_idx = next(i for i, l in enumerate(labels) if l.startswith(start_label))
        except StopIteration:
            start_idx = 0
        seg_pts = pts[start_idx:]
        return float(np.sum(np.linalg.norm(np.diff(seg_pts, axis=0), axis=1)))

class CupManipulatorIK:
    """Inverse kinematics and Jacobian solver for the CupManipulatorTendon.

    Holds a back-reference to the parent manipulator for geometry queries
    (EE frame, joint handles, link lengths) so that all Drake plant calls
    are routed through the manipulator's helpers.

    Typical usage::

        manipulator = CupManipulatorTendon(config)
        # ... build and finalise plant ...
        q, ok  = manipulator.ik.analytical(plant, target_xy, q_seed)
        q_dot  = manipulator.ik.velocity(plant, ctx, ee_vel)
        u_dot  = manipulator.ik.hybrid(plant, ctx, ee_vel)   # actuation space
        J      = manipulator.ik.jacobian(plant, ctx)
        J_h    = manipulator.ik.jacobian_hybrid(plant, ctx)

    Public attributes:
        manip: back-reference to the parent CupManipulatorTendon.
    """

    def __init__(self, manip: "CupManipulatorTendon") -> None:
        self.manip = manip

    # ── Link-length geometry ─────────────────────────────────────────────

    def get_link_lengths(self, plant: MultibodyPlant) -> tuple:
        """Return (L1, L2) in metres extracted from FK at q=0.

        L1 = world-XY distance from joint-1 origin to joint-2 origin.
        L2 = world-XY distance from joint-2 origin to EE frame origin.
        Both are constant (URDF geometry, not joint-angle dependent).
        """
        m   = self.manip
        ctx = plant.CreateDefaultContext()
        world = plant.world_frame()
        plant.SetPositions(ctx, m.model_instance, np.zeros(2))

        j2  = m.get_joint_by_name(plant, m.JT2_NAME)
        Xj2 = plant.CalcRelativeTransform(ctx, world, j2.frame_on_child())
        L1  = np.linalg.norm(Xj2.translation()[:2])

        ee_frame = m.get_end_effector_frame(plant)
        ee_pos0  = plant.CalcPointsPositions(ctx, ee_frame, np.zeros((3, 1)), world).ravel()
        L2 = np.linalg.norm(ee_pos0[:2] - Xj2.translation()[:2])
        return L1, L2

    def _solve_2r_core(
        self,
        L1: float,
        L2: float,
        target_xy: np.ndarray,
        q_seed: np.ndarray,
        q2_limit_rad: Optional[float] = None,
    ) -> tuple:
        """Pure-math closed-form 2-R IK in the XY plane.

        Tries both elbow configurations (sign = ±1) and returns the solution
        closest to *q_seed* that satisfies the optional q2 joint limit.

        Returns:
            (best_q, success) where best_q is ndarray([q1, q2]) or None.
        """
        tx, ty = target_xy
        r2 = tx**2 + ty**2
        D  = (r2 - L1**2 - L2**2) / (2.0 * L1 * L2)

        if abs(D) > 1.0 + 1e-4:
            return None, False

        D = np.clip(D, -1.0, 1.0)
        best_q, best_dist = None, np.inf
        for sign in (1, -1):
            q2 = np.arctan2(sign * np.sqrt(max(0.0, 1.0 - D**2)), D)
            if q2_limit_rad is not None and abs(q2) > q2_limit_rad:
                continue
            k1   = L1 + L2 * np.cos(q2)
            k2   = L2 * np.sin(q2)
            q1   = np.arctan2(ty, tx) - np.arctan2(k2, k1)
            dist = (q1 - q_seed[0])**2 + (q2 - q_seed[1])**2
            if dist < best_dist:
                best_dist = dist
                best_q    = np.array([q1, q2])

        return best_q, best_q is not None

    # ── Jacobians ────────────────────────────────────────────────────────

    def jacobian(self, plant: MultibodyPlant, context) -> np.ndarray:
        """World-XY geometric Jacobian  J ∈ ℝ^{2×2}.

        Maps joint velocities to EE translational velocity:

            ṗ = J(q) · q̇,    q̇ = [q1_dot, q2_dot]ᵀ

        Returns:
            J: ndarray shape (2, 2)
        """
        m      = self.manip
        world  = plant.world_frame()
        ee_frm = m.get_end_effector_frame(plant)
        j1     = m.get_joint_by_name(plant, m.JT1_NAME)
        j2     = m.get_joint_by_name(plant, m.JT2_NAME)
        J_full = plant.CalcJacobianTranslationalVelocity(
            context, JacobianWrtVariable.kV,
            ee_frm, np.zeros(3), world, world,
        )
        return J_full[:2, [j1.velocity_start(), j2.velocity_start()]]


    def jacobian_analytical(self, plant: MultibodyPlant, context) -> np.ndarray:
        """Closed-form planar 2-R Jacobian  J ∈ ℝ^{2×2}  in user order [q1, q2].

        Derived from the forward kinematics used by _solve_2r_core::

            px = L1·cos(q1) + L2·cos(q1+q2)
            py = L1·sin(q1) + L2·sin(q1+q2)

        Differentiating:

            J = [[ ∂px/∂q1,  ∂px/∂q2 ],   =  [[ -L1·s1 - L2·s12,  -L2·s12 ],
                 [ ∂py/∂q1,  ∂py/∂q2 ]]       [  L1·c1 + L2·c12,   L2·c12 ]]

        where s1=sin(q1), c1=cos(q1), s12=sin(q1+q2), c12=cos(q1+q2).

        Returns:
            J: ndarray shape (2, 2)
        """
        L1, L2 = self.get_link_lengths(plant)
        q  = self.manip.get_positions_user_order(plant, context)   # [q1, q2]
        q1, q2 = q[0], q[1]
        s1  = np.sin(q1);        c1  = np.cos(q1)
        s12 = np.sin(q1 + q2);   c12 = np.cos(q1 + q2)
        return np.array([
            [-L1 * s1 - L2 * s12,  -L2 * s12],
            [ L1 * c1 + L2 * c12,   L2 * c12],
        ])

    def _fd_jacobian(self, plant: MultibodyPlant, context, h: float = 1e-6) -> np.ndarray:
        """Central-difference Jacobian  J ∈ ℝ^{2×2} (reference for verify).

        Each column k: J[:, k] = (EE(q + h·eₖ) − EE(q − h·eₖ)) / (2h)

        Does NOT modify *context*.  Allocates two temporary plant contexts.
        """
        m  = self.manip
        q  = m.get_positions_user_order(plant, context)
        J  = np.zeros((2, 2))
        for k in range(2):
            qp, qm = q.copy(), q.copy()
            qp[k] += h;  qm[k] -= h
            ctx_p = plant.CreateDefaultContext()
            ctx_m = plant.CreateDefaultContext()
            m.set_positions_user_order(plant, ctx_p, qp)
            m.set_positions_user_order(plant, ctx_m, qm)
            ee_p = m.get_end_effector_position(plant, ctx_p)[:2]
            ee_m = m.get_end_effector_position(plant, ctx_m)[:2]
            J[:, k] = (ee_p - ee_m) / (2.0 * h)
        return J
    
    def jacobian_hybrid(self, plant: MultibodyPlant, context) -> np.ndarray:
        """Hybrid Jacobian  J_h ∈ ℝ^{2×2} in actuation space.

        Maps actuation rates u̇ = [q1_dot, l_G_dot]ᵀ to EE velocity::

            ṗ = J_h · u̇ = J · A⁻¹ · u̇

        where A = diag(1, r_p) maps q̇ to u̇  (u1 = q1, u2 = l_G = r_p·q2):
            A⁻¹ = diag(1, 1/r_p)   maps u̇ to q̇
            J_h  = J · A⁻¹

        Antagonistic constraint:  l_R_dot = −l_G_dot.

        Returns:
            J_h: ndarray shape (2, 2)
        """
        J = self.jacobian(plant, context)
        return J @ np.diag([1.0, 1.0 / self.manip.PULLEY_RADIUS])

    def verify(
        self,
        plant: MultibodyPlant,
        context,
        tol: float = 1e-6,
        label: str = "",
    ) -> bool:
        """Cross-check three Jacobian formulations at the current configuration.

        Three-way comparison
        --------------------
        1. **Drake geometric**  — ``CalcJacobianTranslationalVelocity`` (kV)
        2. **Analytical 2-R**   — closed-form ∂FK/∂q  (self.jacobian_analytical)
        3. **Finite-difference** — central differences with h=1e-6 rad

        Cable Jacobian check (scalar)
        -----------------------------
        The net joint-2 torque satisfies  τ₂ = J_c^T · F_net  with
        J_c = r_p (single cable scalar Jacobian).  Verified via virtual work:
        δW = F_net · δl_G = F_net · (r_p · δq₂) = τ₂ · δq₂  ✓ (always exact).
        r_p is printed for reference.

        Parameters
        ----------
        plant   : Finalized MultibodyPlant (context must belong to it).
        context : Plant context at the configuration to test.
        tol     : Absolute tolerance in metres per radian.
        label   : Optional tag printed in the summary line.

        Returns
        -------
        bool — True when all matrix checks pass within *tol*.
        """
        q = self.manip.get_positions_user_order(plant, context)
        tag = f" [{label}]" if label else ""
        print(colored(
            f"\n── Jacobian verify{tag}  q=[{np.rad2deg(q[0]):.1f}°, {np.rad2deg(q[1]):.1f}°] ──",
            "cyan",
        ))

        J_drake  = self.jacobian(plant, context)
        J_analyt = self.jacobian_analytical(plant, context)
        J_fd     = self._fd_jacobian(plant, context)

        err_da = np.abs(J_drake  - J_analyt)   # Drake vs analytical
        err_df = np.abs(J_drake  - J_fd)        # Drake vs FD
        err_af = np.abs(J_analyt - J_fd)        # analytical vs FD

        def _row(name_a, name_b, err):
            ok  = err.max() < tol
            sym = colored("✓", "green") if ok else colored("✗", "red")
            return f"  {sym}  {name_a} vs {name_b:12s}  max_err={err.max():.2e} m/rad"

        print(_row("Drake    ", "Analytical ", err_da))
        print(_row("Drake    ", "Fin-diff   ", err_df))
        print(_row("Analytical", "Fin-diff  ", err_af))

        # Cable Jacobian: Jc = r_p (scalar, exact by construction)
        r_p = self.manip.PULLEY_RADIUS
        print(colored(
            f"  ✓  Cable Jacobian  Jc = r_p = {r_p*1e3:.4f} mm  "
            f"(τ₂ = Jc^T · F_net by virtual work, exact)",
            "green",
        ))

        all_ok = (err_da.max() < tol) and (err_df.max() < tol) and (err_af.max() < tol)
        status = colored("ALL PASS", "green", attrs=["bold"]) if all_ok else colored("FAIL", "red", attrs=["bold"])
        print(f"  → {status}  (tol={tol:.0e} m/rad)\n")

        if not all_ok:
            print("  Drake:\n", J_drake)
            print("  Analytical:\n", J_analyt)
            print("  Fin-diff:\n", J_fd)

        return all_ok

    # ── IK solvers ───────────────────────────────────────────────────────

    def analytical(
        self,
        plant: MultibodyPlant,
        target_xy,
        q_seed,
        pos_tol: float = 1e-3,
        verbose: bool = False,
        q2_limit_rad: Optional[float] = None,
        ee_frame_name: Optional[str] = None,   # unused, kept for API compat
        target_z: Optional[float] = None,       # unused, kept for API compat
        **kwargs,
    ) -> tuple:
        """Closed-form 2-R IK in the horizontal XY plane.

        Orchestrates get_link_lengths → _solve_2r_core → FK verify.

        Returns:
            (q, success) where q = [q1, q2] in radians.
        """
        m         = self.manip
        target_xy = np.asarray(target_xy, dtype=float).reshape(2,)
        q_seed    = np.asarray(q_seed,    dtype=float).reshape(2,)
        L1, L2    = self.get_link_lengths(plant)

        if verbose:
            print(f"  2R IK: L1={L1*1e3:.2f} mm  L2={L2*1e3:.2f} mm")
            print(f"  Target XY: ({target_xy[0]*1e3:.2f}, {target_xy[1]*1e3:.2f}) mm")

        best_q, ok = self._solve_2r_core(L1, L2, target_xy, q_seed, q2_limit_rad)

        if not ok:
            if verbose:
                D = (target_xy[0]**2 + target_xy[1]**2 - L1**2 - L2**2) / (2 * L1 * L2)
                msg = "Target out of reach" if abs(D) > 1.0 + 1e-4 else "No solution satisfying q2 limit"
                print(f"  {msg}")
            return q_seed, False

        ctx = plant.CreateDefaultContext()
        m.set_positions_user_order(plant, ctx, best_q)
        ee_check  = m.get_end_effector_position(plant, ctx)
        final_err = np.linalg.norm(target_xy - ee_check[:2])
        success   = final_err < pos_tol * 5

        if verbose:
            print(f"  Analytical IK: q1={np.rad2deg(best_q[0]):.2f}°"
                  f"  q2={np.rad2deg(best_q[1]):.2f}°"
                  f"  err={final_err*1e3:.3f} mm  ok={success}")
        return best_q, success

    def velocity(
        self,
        plant: MultibodyPlant,
        context,
        ee_velocity_xy: np.ndarray,
        damping: float = 1e-4,
    ) -> np.ndarray:
        """Velocity IK in joint space: q_dot = J⁺ · ẋ.

        Uses the damped pseudo-inverse of the geometric Jacobian::

            q̇ = Jᵀ (J Jᵀ + λI)⁻¹ ẋ

        Returns:
            q_dot: ndarray (2,) — [q1_dot, q2_dot].
        """
        ee_vel = np.asarray(ee_velocity_xy, dtype=float).reshape(2,)
        J      = self.jacobian(plant, context)
        return J.T @ np.linalg.solve(J @ J.T + damping * np.eye(2), ee_vel)

    def hybrid(
        self,
        plant: MultibodyPlant,
        context,
        ee_velocity_xy: np.ndarray,
        damping: float = 1e-4,
    ) -> np.ndarray:
        """Velocity IK in actuation space: u_dot = J_h⁺ · ẋ.

        Solves::

            u̇ = J_hᵀ (J_h J_hᵀ + λI)⁻¹ ẋ

        Returns:
            u_dot: ndarray (2,) — [q1_dot, l_G_dot].

        Notes:
            - Recover joint velocity: q2_dot = u_dot[1] / PULLEY_RADIUS
            - Antagonistic:           l_R_dot = −u_dot[1]
        """
        ee_vel = np.asarray(ee_velocity_xy, dtype=float).reshape(2,)
        J_h    = self.jacobian_hybrid(plant, context)
        return J_h.T @ np.linalg.solve(J_h @ J_h.T + damping * np.eye(2), ee_vel)






def __getattr__(name):
    # Backward-compatible lazy re-exports: controllers + actuators + IK system.
    # Each class now lives in its dedicated module but can still be imported
    # via  from robots.cup_manipulator_tendon import <ClassName>.
    _CONTROLLER_MAP = {
        "ComputedTorqueController": "controller.controller",
        "SEACableController":       "controller.controller",
        "CupManipulatorIKSystem":   "controller.ik_system",
    }
    _ACTUATOR_MAP = {
        "SEACableActuator": "actuators.sea",
    }
    if name in _CONTROLLER_MAP:
        import importlib
        mod = importlib.import_module(_CONTROLLER_MAP[name])
        return getattr(mod, name)
    if name in _ACTUATOR_MAP:
        import importlib
        mod = importlib.import_module(_ACTUATOR_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_cable_manipulator_config(
    urdf_path: str = "model_using_onshape_to_robot/manipulator_cable/manipulator_cable_obj.urdf",
    joint_angles: Optional[dict] = None,
    damping:       tuple = (0.1, 0.1),
    stiffness:     tuple = (0.0, 0.0),
    friction:      tuple = (0.0, 0.0),
    tilt_roll_deg:  float = 0.0,
    tilt_pitch_deg: float = 0.0,
    motor: Optional[str] = None,
) -> ManipulatorConfig:
    """Factory for the cable (tendon) manipulator configuration.

    Args:
        motor: Registered motor name, e.g. ``"AK80_8_KV60_Config"``.  When set,
               ``CupManipulatorTendon`` will read viscous damping, torque limits,
               and rotor inertia from the motor model instead of *damping*.
               Pass ``None`` (default) to keep the raw *damping* tuple.
    """
    urdf_dir    = str(Path(urdf_path).parent)
    joint_names = ["link1_base", "link2_link1"]
    if joint_angles is None:
        joint_angles = {n: 0.0 for n in joint_names}
    joint_configs = {}
    for i, name in enumerate(joint_names):
        joint_configs[name] = JointConfig(
            position=joint_angles.get(name, 0.0),
            damping=damping[i],
            stiffness=stiffness[i],
            friction=friction[i],
        )
    return ManipulatorConfig(
        name="manipulator_cable",
        urdf_path=urdf_path,
        joint_configs=joint_configs,
        base_pose=Pose(),
        tilt_roll_deg=tilt_roll_deg,
        tilt_pitch_deg=tilt_pitch_deg,
        motor_name=motor,
        package_map={"assets": urdf_dir + "/assets/"},
    )

