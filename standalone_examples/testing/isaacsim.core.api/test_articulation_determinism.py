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

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import asyncio

import carb
import numpy as np
import omni.kit.test
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import open_stage, update_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path

assets_root_path = get_assets_root_path()


my_world = World(stage_units_in_meters=1.0)
my_world.reset()


def test_franka_slow_convergence():
    open_stage(get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd")
    robot_prim_path = "/panda"

    # Start Simulation and wait
    my_world = World(stage_units_in_meters=1.0)
    my_world.reset()

    robot = Robot(robot_prim_path)
    robot.initialize()
    robot.get_articulation_controller().set_gains(1e4 * np.ones(9), 1e3 * np.ones(9))
    robot.set_solver_position_iteration_count(64)
    robot.set_solver_velocity_iteration_count(64)
    robot.post_reset()

    my_world.step(render=True)

    timeout = 200

    action = ArticulationAction(
        joint_positions=np.array(
            [
                -0.40236897393760085,
                -0.44815597748391767,
                -0.16028112816211953,
                -2.4554393933564986,
                -0.34608791253975374,
                2.9291361940824485,
                0.4814803907662416,
                None,
                None,
            ]
        )
    )

    robot.get_articulation_controller().apply_action(action)

    for i in range(timeout):
        my_world.step()
        diff = robot.get_joint_positions() - action.joint_positions
        if np.linalg.norm(diff) < 0.01:
            return i

    return timeout


frames_to_converge = np.empty(5)
for i in range(5):
    num_frames = test_franka_slow_convergence()
    frames_to_converge[i] = num_frames

# Takes the same number of frames to converge every time
print(f"Over 5 trials, the Franka converged to target in {frames_to_converge} frames.")
if np.unique(frames_to_converge).shape[0] != 1:
    print(f"Non-deterministic test converged in varying number of frames: {frames_to_converge}")
    raise Exception

# On the develop branch, this test always takes 26 frames to converge
if frames_to_converge[0] != 26:
    print("Didn't converge in the right number of frames")
    raise Exception

simulation_app.close()
