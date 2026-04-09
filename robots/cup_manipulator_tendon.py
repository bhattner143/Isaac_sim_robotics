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
    LeafSystem,
    BasicVector,
)
from pydrake.multibody.tree import MultibodyForces
from robots.cup_manipulator import RobotBase
from configs.robot.robot_types import ManipulatorConfig, JointConfig, Pose
from robots.motor import get_motor, MotorModelConfig

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






    

# ============================================================================
# COMPUTED-TORQUE CONTROLLER
# ============================================================================

class ComputedTorqueController(LeafSystem):
    """Computed-torque (inverse-dynamics) controller for CupManipulatorTendon.

    Feedback-linearizes the 2-DOF planar manipulator by cancelling nonlinear
    dynamics (Coriolis + gravity) and commanding a desired joint-space
    acceleration obtained from a PD law in joint space:

        a_des = Kp · (q_des − q) − Kd · q̇         [rad/s²]
        τ     = M(q) · a_des + h(q, q̇)             [Nm]

    where  h(q, q̇)  is the bias term (Coriolis + gravity) computed via
    Drake's CalcInverseDynamics with vdot=0:

        h(q, q̇)  =  CalcInverseDynamics(ctx, vdot=0, forces_with_gravity)

    Because the system is feedback-linearized, the closed-loop error dynamics
    are decoupled linear 2nd-order systems:

        ë + Kd · ė + Kp · e = 0   →   poles at  −ζωn ± ωn√(ζ²−1)

    Gains are dimensionless in acceleration space:
        Kp  [s⁻²]  →  ωn = √Kp        (e.g. Kp=400 → ωn=20 rad/s)
        Kd  [s⁻¹]  →  ζ  = Kd/(2√Kp)  (e.g. Kd=40  → ζ=1, critically damped)

    Joint-2 torque is passed through cable tension decomposition
    (identical to the IK-diagram PD controller) so the two non-negative
    cable tensions are physically meaningful:

        F_net   = τ2 / r_p                   [N]
        T_green = max( F_net, 0)              retracting side
        T_red   = max(−F_net, 0)             extending side
        τ2_cmd  = (T_green − T_red) · r_p    [Nm]   (= τ2, but logged separately)

    Input ports
    -----------
    ``desired_ee_pos``  [2]   target (x, y) in world frame [m]
    ``plant_state``     [n]   from ``plant.get_state_output_port()``

    Output ports
    ------------
    ``actuation``       [2]   joint torques  [Nm]  → plant actuation input
    ``joint_positions`` [2]   IK solution  [q1_des, q2_des]  [rad]
    ``torques_raw``     [2]   pre-clip joint torques from inverse dynamics  [Nm]
    ``cable_tensions``  [2]   [T_green, T_red]  [N]
    """

    def __init__(
        self,
        plant: MultibodyPlant,
        manipulator: "CupManipulatorTendon",
        Kp: float   = 400.0,   # position gain  [s⁻²]  → ωn = 20 rad/s
        Kd: float   = 40.0,    # velocity gain  [s⁻¹]  → ζ = 1 (critically damped)
        tau_max: float = 10.0, # torque saturation  [Nm]
    ) -> None:
        super().__init__()
        self._plant    = plant
        self._manip    = manipulator
        self._ik       = manipulator.ik
        self._r_p      = manipulator.PULLEY_RADIUS
        self._Kp       = float(Kp)
        self._Kd       = float(Kd)
        self._tau_max  = float(tau_max)

        # Private plant context for kinematics / inverse-dynamics queries.
        # Never used for integration — only for FK and CalcInverseDynamics.
        self._plant_ctx = plant.CreateDefaultContext()

        # Pre-allocated MultibodyForces: reused every timestep to avoid GC churn.
        self._forces = MultibodyForces(plant)

        # Pre-cache link lengths (constant URDF geometry).
        self._L1, self._L2 = manipulator.ik.get_link_lengths(plant)

        # Persistent IK seed — keeps IK on the same elbow branch across steps.
        self._last_q_des = np.zeros(2)

        # Velocity-vector indices for user-order joints [q1, q2] in Drake's nv-vector.
        j1 = manipulator.get_joint_by_name(plant, manipulator.JT1_NAME)
        j2 = manipulator.get_joint_by_name(plant, manipulator.JT2_NAME)
        self._v_idx = [j1.velocity_start(), j2.velocity_start()]

        # Per-timestep cache: keyed on context time, avoids 3× redundant IDyn solve.
        self._t_cache        = -np.inf
        self._q_des_cache    = np.zeros(2)
        self._q_cache        = np.zeros(2)
        self._q_dot_cache    = np.zeros(2)
        self._tau_clip_cache = np.zeros(2)  # clipped actuation output
        self._tau_raw_cache  = np.zeros(2)  # pre-clip inverse-dynamics torques
        self._tens_cache     = np.zeros(2)  # [T_green, T_red]

        nstate = plant.num_multibody_states()
        self._ee_port    = self.DeclareVectorInputPort("desired_ee_pos", 2)
        self._vel_port   = self.DeclareVectorInputPort("ee_vel_ref",     2)  # EE velocity ref  [m/s]
        self._acc_port   = self.DeclareVectorInputPort("ee_acc_ref",     2)  # EE accel ref     [m/s²]
        self._state_port = self.DeclareVectorInputPort("plant_state", nstate)

        self.DeclareVectorOutputPort("actuation",       2, self._calc_actuation)
        self.DeclareVectorOutputPort("joint_positions", 2, self._calc_joint_positions)
        self.DeclareVectorOutputPort("torques_raw",     2, self._calc_torques_raw)
        self.DeclareVectorOutputPort("cable_tensions",  2, self._calc_cable_tensions)

    # ── Per-timestep solve (cached) ───────────────────────────────────────────

    def _solve(self, context):
        """IK + inverse-dynamics torque, cached per simulation timestep.

        Returns
        -------
        q_des    : ndarray (2,) — desired joints [q1, q2]  [rad]
        q        : ndarray (2,) — current joints
        q_dot    : ndarray (2,) — current joint velocities
        tau_clip : ndarray (2,) — clipped actuation torques  [Nm]
        tau_raw  : ndarray (2,) — pre-clip inverse-dynamics torques  [Nm]
        tens     : ndarray (2,) — [T_green, T_red]  [N]
        """
        t = context.get_time()
        if t == self._t_cache:
            return (self._q_des_cache, self._q_cache, self._q_dot_cache,
                    self._tau_clip_cache, self._tau_raw_cache, self._tens_cache)

        # ── Read ports ──────────────────────────────────────────────────────
        state  = self._state_port.Eval(context)
        ee_des = self._ee_port.Eval(context)

        # Sync internal plant context
        self._plant.SetPositionsAndVelocities(self._plant_ctx, state)
        q     = self._manip.get_positions_user_order(self._plant, self._plant_ctx)
        q_dot = self._manip.get_velocities_user_order(self._plant, self._plant_ctx)

        # ── Analytical 2R IK (warm-started from last solution) ──────────────
        seed   = self._last_q_des if np.any(self._last_q_des != 0.0) else q
        best_q, ok = self._ik._solve_2r_core(self._L1, self._L2, ee_des, seed)
        if ok:
            self._last_q_des = best_q.copy()
            q_des = best_q
        else:
            q_des = self._last_q_des.copy()  # hold last known good

        # ── Feedforward: reference joint velocity & acceleration ────────────
        # Invert the analytical 2R Jacobian at q_des to map EE ref vel/acc
        # to joint space.  Closed-form J saves a plant-context round-trip.
        #
        #   J(q) = [[ -L1 s1 - L2 s12,  -L2 s12 ],
        #           [  L1 c1 + L2 c12,   L2 c12  ]]
        #
        # q̇_ref  = J⁻¹ · ẋ_ref
        # q̈_ref  ≈ J⁻¹ · ẍ_ref   (J̇·q̇ bias term dropped — O(q̇²), small)
        ee_vel_ref = self._vel_port.Eval(context)   # ẋ_ref  [m/s]
        ee_acc_ref = self._acc_port.Eval(context)   # ẍ_ref  [m/s²]
        q1d, q2d   = q_des
        s1  = np.sin(q1d);          c1  = np.cos(q1d)
        s12 = np.sin(q1d + q2d);    c12 = np.cos(q1d + q2d)
        J = np.array([
            [-self._L1 * s1 - self._L2 * s12, -self._L2 * s12],
            [ self._L1 * c1 + self._L2 * c12,  self._L2 * c12],
        ])
        J_inv      = np.linalg.pinv(J)          # well-conditioned away from singularities
        q_dot_ref  = J_inv @ ee_vel_ref          # [rad/s]
        q_ddot_ref = J_inv @ ee_acc_ref          # [rad/s²]

        # ── Full CT law with feedforward ─────────────────────────────────────
        # a_des = q̈_ref  +  Kp·(q_des − q)  +  Kd·(q̇_ref − q̇)
        # Feedforward drives M(q)·q̈_ref ≠ 0, giving significant torque.
        # Feedback terms reject disturbances & correct trajectory errors.
        a_des_user = q_ddot_ref + self._Kp * (q_des - q) + self._Kd * (q_dot_ref - q_dot)

        # Map to Drake velocity-vector order
        nv = self._plant.num_velocities()
        a_des_drake = np.zeros(nv)
        a_des_drake[self._v_idx[0]] = a_des_user[0]
        a_des_drake[self._v_idx[1]] = a_des_user[1]

        # ── Computed torque: τ = M(q)·a_des + h(q, q̇) ────────────────────
        # Drake's CalcInverseDynamics signature:
        #     τ = M(q)·vdot + C(q,v)·v + g(q) − external_forces
        # The gravity term g(q) is ALWAYS included internally; external_forces
        # is for truly external loads (e.g. contact).  Passing the result of
        # CalcForceElementsContribution as external_forces would subtract those
        # forces a second time, double-counting gravity.  Pass zero instead.
        self._forces.SetZero()
        tau_full = self._plant.CalcInverseDynamics(
            self._plant_ctx, a_des_drake, self._forces
        )

        tau1 = float(tau_full[self._v_idx[0]])
        tau2 = float(tau_full[self._v_idx[1]])

        # ── Joint-2 cable tension decomposition ─────────────────────────────
        # τ2 = F_net · r_p  →  F_net = τ2 / r_p
        # Cables can only pull (T ≥ 0); decompose into two non-negative tensions.
        F_net   = tau2 / self._r_p
        T_green = max(F_net,  0.0)   # retracting side tightens
        T_red   = max(-F_net, 0.0)   # extending side tightens on reversal
        tau2_cmd = (T_green - T_red) * self._r_p   # = tau2 (identity; logged separately)

        tau_raw  = np.array([tau1, tau2])
        tau_clip = np.clip(np.array([tau1, tau2_cmd]), -self._tau_max, self._tau_max)
        tens     = np.array([T_green, T_red])

        # Cache all results for this timestep
        self._t_cache        = t
        self._q_des_cache    = q_des
        self._q_cache        = q
        self._q_dot_cache    = q_dot
        self._tau_clip_cache = tau_clip
        self._tau_raw_cache  = tau_raw
        self._tens_cache     = tens

        return q_des, q, q_dot, tau_clip, tau_raw, tens

    # ── Output port callbacks ─────────────────────────────────────────────────

    def _calc_actuation(self, context, output):
        _, _, _, tau_clip, _, _ = self._solve(context)
        output.SetFromVector(tau_clip)

    def _calc_joint_positions(self, context, output):
        q_des, _, _, _, _, _ = self._solve(context)
        output.SetFromVector(q_des)

    def _calc_torques_raw(self, context, output):
        _, _, _, _, tau_raw, _ = self._solve(context)
        output.SetFromVector(tau_raw)

    def _calc_cable_tensions(self, context, output):
        _, _, _, _, _, tens = self._solve(context)
        output.SetFromVector(tens)


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

