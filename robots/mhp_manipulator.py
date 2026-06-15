"""Drake wrapper for the real MHP (manipulator hybrid planar) URDF."""
from __future__ import annotations

from typing import List

import numpy as np
from termcolor import colored

from pydrake.all import (
    MultibodyPlant,
    Parser,
    RigidTransform,
    RollPitchYaw,
    FixedOffsetFrame,
    RevoluteJoint,
    PrismaticJoint,
)

from configs.robot.robot_types import ManipulatorConfig
from robots.cup_manipulator import RobotBase
from robots.cup_manipulator_tendon import CupManipulatorIK


class MHPManipulator(RobotBase):
    """2-DOF MHP cable-driven arm for Drake CT simulation.

    Joint names (URDF):
        jt_upper_base  — shoulder (q1)
        jt_lower_upper — elbow    (q2)
    """

    JT1_NAME = "jt_upper_base"
    JT2_NAME = "jt_lower_upper"
    BASE_LINK_NAME = "base_link_aka_shoulder_transmission"
    LINK2_NAME = "lower_arm"
    EE_FRAME_NAME = "mhp_ee"
    EE_XYZ_LINK2 = np.array([0.20, 0.0, 0.0])
    EE_RPY_LINK2 = np.array([0.0, 0.0, 0.0])
    EE_OFFSET = EE_XYZ_LINK2
    # Nominal radius for CT cable-tension logging (MHP uses dual tendons).
    PULLEY_RADIUS = 0.01

    def __init__(self, config: ManipulatorConfig, enable_visualization: bool = True):
        super().__init__(config)
        self.joint_names: List[str] = [self.JT1_NAME, self.JT2_NAME]
        self.actuator_names: List[str] = []
        self.enable_visualization = enable_visualization
        self.ik = CupManipulatorIK(self)

    def load_urdf_to_plant(self, plant: MultibodyPlant, parser: Parser) -> int:
        model_instance = super().load_urdf_to_plant(plant, parser)
        self.JT1_NAME = "jt_upper_base"
        self.JT2_NAME = "jt_lower_upper"
        self.ACT1_NAME = f"tau_{self.JT1_NAME}"
        self.ACT2_NAME = f"tau_{self.JT2_NAME}"
        self.joint_names = [self.JT1_NAME, self.JT2_NAME]
        print(colored(
            f"✓ MHPManipulator: joints [{self.JT1_NAME}, {self.JT2_NAME}]",
            "green",
        ))
        return model_instance

    def weld_base_to_world(
        self,
        plant: MultibodyPlant,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
    ) -> None:
        if plant.is_finalized():
            raise RuntimeError("Cannot weld base after plant is finalized")
        position = np.zeros(3) if position is None else np.asarray(position, float)
        orientation = np.zeros(3) if orientation is None else np.asarray(orientation, float)
        base_body = plant.GetBodyByName(self.BASE_LINK_NAME, self.model_instance)
        plant.WeldFrames(
            plant.world_frame(),
            base_body.body_frame(),
            RigidTransform(RollPitchYaw(orientation), position),
        )

    def add_end_effector_frame(self, plant: MultibodyPlant):
        if plant.is_finalized():
            raise RuntimeError("Cannot add EE frame after plant is finalized")
        link2_body = plant.GetBodyByName(self.LINK2_NAME, self.model_instance)
        X_L2_EE = RigidTransform(RollPitchYaw(self.EE_RPY_LINK2), self.EE_XYZ_LINK2)
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

    def add_joint_actuators(self, plant: MultibodyPlant) -> None:
        if plant.is_finalized():
            raise RuntimeError("Cannot add actuators after plant is finalized")
        jt1 = self.get_joint_by_name(plant, self.JT1_NAME)
        jt2 = self.get_joint_by_name(plant, self.JT2_NAME)
        effort_limit = 10.0
        act1 = plant.AddJointActuator(self.ACT1_NAME, jt1, effort_limit)
        act2 = plant.AddJointActuator(self.ACT2_NAME, jt2, effort_limit)
        self.actuator_names = [self.ACT1_NAME, self.ACT2_NAME]
        print(colored(f"✓ MHP actuators: {act1.name()}, {act2.name()}", "green"))

    def set_joint_properties(self, plant: MultibodyPlant) -> None:
        for joint_idx in plant.GetJointIndices(self.model_instance):
            joint = plant.get_joint(joint_idx)
            if joint.num_velocities() == 0:
                continue
            cfg = self.config.joint_configs.get(joint.name())
            if cfg is None:
                continue
            if isinstance(joint, RevoluteJoint) and cfg.damping:
                joint.set_default_damping(cfg.damping)

    def get_end_effector_position(self, plant: MultibodyPlant, context) -> np.ndarray:
        ee_frame = self.get_end_effector_frame(plant)
        return plant.CalcRelativeTransform(
            context, plant.world_frame(), ee_frame
        ).translation()

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
            for name, ang in zip(joint_name, np.atleast_1d(angle)):
                self.get_joint_by_name(plant, name).set_angle(context, float(ang))
        else:
            self.get_joint_by_name(plant, joint_name).set_angle(context, float(angle))

    def get_jt_velocity(self, joint_name, plant: MultibodyPlant, context):
        if isinstance(joint_name, list):
            return np.array([
                self.get_joint_by_name(plant, n).get_angular_rate(context)
                for n in joint_name
            ])
        return self.get_joint_by_name(plant, joint_name).get_angular_rate(context)

    def set_jt_velocity(self, joint_name, plant: MultibodyPlant, context, velocity):
        if isinstance(joint_name, list):
            for name, vel in zip(joint_name, np.atleast_1d(velocity)):
                self.get_joint_by_name(plant, name).set_angular_rate(context, float(vel))
        else:
            self.get_joint_by_name(plant, joint_name).set_angular_rate(
                context, float(velocity)
            )

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
        q1_dot, q2_dot = user_velocities
        self.set_jt_velocity(
            [self.JT1_NAME, self.JT2_NAME], plant, context, [q1_dot, q2_dot]
        )
