# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import argparse
import asyncio
import os
import time

import numpy as np
from isaacsim import SimulationApp

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--test-steps", type=int, default=30)
args, _ = parser.parse_known_args()

# Launch Isaac Sim in headless zero-delay mode
simulation_app = SimulationApp(
    {"headless": True}, experience=f'{os.environ["EXP_PATH"]}/isaacsim.exp.base.zero_delay.kit'
)

# Import post-launch modules
import carb
import isaacsim.core.utils.numpy.rotations as rot_utils
import omni.graph.core as og
import omni.usd
import usdrt.Sdf
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.physics import simulate_async
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import rclpy
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage

rclpy.init()

# ROS state
tf_msg = None
img_msg = None


def tf_callback(msg):
    global tf_msg
    tf_msg = msg


def img_callback(msg):
    global img_msg
    img_msg = msg


# Create ROS2 node + subscribers
node = rclpy.create_node("sync_test_node")
tf_sub = node.create_subscription(TFMessage, "/tf_test", tf_callback, 10)
img_sub = node.create_subscription(Image, "rgb", img_callback, 10)

# Create world and add cube
world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
CUBE_PRIM_PATH = "/Cube"
cube = world.scene.add(DynamicCuboid(prim_path=CUBE_PRIM_PATH, scale=[0.5, 0.5, 0.5]))
world.scene.add_default_ground_plane()

# Add camera
CAMERA_PRIM_PATH = "/Camera"
CAMERA_EYE = [0, -2, 1]
CAMERA_TARGET = [0, 0, 0.1]

stage = omni.usd.get_context().get_stage()
stage.DefinePrim(CAMERA_PRIM_PATH, "Camera")
set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET, camera_prim_path=CAMERA_PRIM_PATH)

world.reset()

# Create Action Graph
og.Controller.edit(
    {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("TfPublisher", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ("CameraPublisher", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("TfPublisher.inputs:targetPrims", [usdrt.Sdf.Path(CUBE_PRIM_PATH)]),
            ("TfPublisher.inputs:topicName", "/tf_test"),
            ("CameraPublisher.inputs:type", "rgb"),
            ("CameraPublisher.inputs:topicName", "rgb"),
            ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(CAMERA_PRIM_PATH)]),
            ("CreateRenderProduct.inputs:width", 1920),
            ("CreateRenderProduct.inputs:height", 1080),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "TfPublisher.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", "TfPublisher.inputs:timeStamp"),
            ("CreateRenderProduct.outputs:execOut", "CameraPublisher.inputs:execIn"),
            ("CreateRenderProduct.outputs:renderProductPath", "CameraPublisher.inputs:renderProductPath"),
        ],
    },
)

simulation_app.update()

# Warmup
for _ in range(20):
    world.step(render=True)

# Run sync test
deltas = []
for step in range(args.test_steps):
    cube.set_world_pose(position=np.random.uniform(-1, 1, 3))
    world.step(render=True)

    # Block to make sure both messages are received
    MAX_WAIT_STEPS = 20
    wait_count = 0
    while (tf_msg is None or img_msg is None) and wait_count < MAX_WAIT_STEPS:
        rclpy.spin_once(node, timeout_sec=0.05)
        wait_count += 1

    if tf_msg and img_msg:
        tf_stamp = tf_msg.transforms[0].header.stamp
        img_stamp = img_msg.header.stamp
        tf_ns = tf_stamp.sec * 1e9 + tf_stamp.nanosec
        img_ns = img_stamp.sec * 1e9 + img_stamp.nanosec
        delta = abs(tf_ns - img_ns)
        deltas.append(delta)
        tf_msg = None
        img_msg = None

    # Reset messages
    tf_msg = None
    img_msg = None

# Print results
if deltas:
    avg_delay = np.mean(deltas) / 1e6
    max_delay = np.max(deltas) / 1e6
    print("\nTF / CAMERA TIMESTAMP SYNC TEST")
    print(f"Steps:         {len(deltas)}")
    print(f"Average delay: {avg_delay:.3f} ms")
    print(f"Max delay:     {max_delay:.3f} ms")
    print("Status:        ", "PASS ✅" if avg_delay < 0.5 else "FAIL ❌")
else:
    print("\n[ERROR] No messages received.")

# Cleanup
node.destroy_node()
rclpy.shutdown()
simulation_app.close()
