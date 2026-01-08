# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": False}, experience=f'{os.environ["EXP_PATH"]}/isaacsim.exp.base.xr.openxr.kit'
)

# Handle start/stop teleop commands from XR device
import carb
from omni.kit.xr.core import XR_INPUT_DEVICE_HAND_TRACKING_POSE_NAMES, XRCore, XRPoseValidityFlags

TELEOP_COMMAND_EVENT_TYPE = "teleop_command"

tracking_enabled = False


def on_message(event: carb.events.IEvent):
    """Processes the received message using key word."""
    message_in = event.payload["message"]

    global tracking_enabled
    if "start" in message_in:
        tracking_enabled = True
    elif "stop" in message_in:
        tracking_enabled = False
    elif "reset" in message_in:
        carb.log_info("Reset recieved")
    else:
        carb.log_warn(f"Unexpected message recieved {message_in}")


message_bus = XRCore.get_singleton().get_message_bus()
incoming_message_event = carb.events.type_from_string(TELEOP_COMMAND_EVENT_TYPE)
subscription = message_bus.create_subscription_to_pop_by_type(incoming_message_event, on_message)


import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils.prims import create_prim, set_prim_visibility
from omni.isaac.core.prims import XFormPrim
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

my_world = World(stage_units_in_meters=1.0)

# Add Light Source
stage = omni.usd.get_context().get_stage()
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(300)

hidden_prim = create_prim("/Hidden/Prototypes", "Scope")
base_cube_path = "/Hidden/Prototypes/BaseCube"
VisualCuboid(
    prim_path=base_cube_path,
    size=0.01,
    color=np.array([255, 0, 0]),
)
set_prim_visibility(hidden_prim, False)

instancer_path = "/World/CubeInstancer"
point_instancer = UsdGeom.PointInstancer.Define(my_world.stage, instancer_path)
point_instancer.CreatePrototypesRel().SetTargets([Sdf.Path(base_cube_path)])

hand_joint_count = len(XR_INPUT_DEVICE_HAND_TRACKING_POSE_NAMES)
joint_count = hand_joint_count * 2 + 1

# Initially hide all cubes until hands are tracked
point_instancer.CreateProtoIndicesAttr().Set([1 for _ in range(joint_count)])

positions = [Gf.Vec3f(0.0, 0.0, 0.0) for i in range(joint_count)]
point_instancer.CreatePositionsAttr().Set(positions)

orientations = [Gf.Quath(1.0, 0.0, 0.0, 0.0) for _ in range(joint_count)]
point_instancer.CreateOrientationsAttr().Set(orientations)

instancer_prim = XFormPrim(prim_path=instancer_path)
my_world.scene.add(instancer_prim)

my_world.reset()
reset_needed = False

positions_attr = point_instancer.GetPositionsAttr()
orientations_attr = point_instancer.GetOrientationsAttr()
proto_idx_attr = point_instancer.GetProtoIndicesAttr()

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False

        current_positions = positions_attr.Get()
        current_orientations = orientations_attr.Get()
        proto_indices = proto_idx_attr.Get()

        left_hand_device = XRCore.get_singleton().get_input_device("/user/hand/left")
        right_hand_device = XRCore.get_singleton().get_input_device("/user/hand/right")

        left_joints = dict()
        right_joints = dict()
        if left_hand_device:
            left_joints = left_hand_device.get_all_virtual_world_poses()
        if right_hand_device:
            right_joints = right_hand_device.get_all_virtual_world_poses()

        proto_indices = [1 for _ in range(joint_count)]
        proto_idx_attr.Set(proto_indices)

        if tracking_enabled:
            # Update hand joint indicators
            def update_joint_visual(joint, idx):
                joint_transform = joint.pose_matrix
                if joint.validity_flags & XRPoseValidityFlags.POSITION_VALID:
                    world_position = joint_transform.ExtractTranslation()
                    current_positions[idx] = Gf.Vec3f(world_position[0], world_position[1], world_position[2])

                if joint.validity_flags & XRPoseValidityFlags.ORIENTATION_VALID:
                    world_orientation = joint_transform.ExtractRotationQuat()
                    current_orientations[idx] = Gf.Quath(
                        world_orientation.GetReal(),
                        world_orientation.GetImaginary()[0],
                        world_orientation.GetImaginary()[1],
                        world_orientation.GetImaginary()[2],
                    )

                proto_indices[idx] = 0

            # Update left hand joints
            for joint_idx, joint_name in enumerate(XR_INPUT_DEVICE_HAND_TRACKING_POSE_NAMES):
                if joint_name in left_joints:
                    update_joint_visual(left_joints[joint_name], joint_idx)

            # Update right hand joints
            for joint_idx, joint_name in enumerate(XR_INPUT_DEVICE_HAND_TRACKING_POSE_NAMES):
                if joint_name in right_joints:
                    update_joint_visual(right_joints[joint_name], joint_idx + hand_joint_count)

            # Get head device
            head_device = XRCore.get_singleton().get_input_device("/user/head")

            if head_device:
                hmd = head_device.get_virtual_world_pose("")
                position = hmd.ExtractTranslation()
                quat = hmd.ExtractRotationQuat()

                # Local to world matrix
                mtx = Gf.Matrix4d()
                mtx.SetRotate(quat)
                mtx.SetTranslateOnly(position)

                # Local transform
                local_transform = Gf.Matrix4d()
                local_transform.SetTranslate(Gf.Vec3d(0.0, 0.0, -0.5))

                # World transform
                world_transform = local_transform * mtx

                world_position = world_transform.ExtractTranslation()
                world_orientation = world_transform.ExtractRotation().GetQuat()

                # Update head indicator
                current_positions[2 * hand_joint_count] = Gf.Vec3f(
                    world_position[0], world_position[1], world_position[2]
                )
                current_orientations[2 * hand_joint_count] = Gf.Quath(
                    world_orientation.GetReal(),
                    world_orientation.GetImaginary()[0],
                    world_orientation.GetImaginary()[1],
                    world_orientation.GetImaginary()[2],
                )
                proto_indices[2 * hand_joint_count] = 0
            else:
                # Hide head indicator
                proto_indices[2 * hand_joint_count] = 1

        positions_attr.Set(current_positions)
        orientations_attr.Set(current_orientations)
        proto_idx_attr.Set(proto_indices)

simulation_app.close()
