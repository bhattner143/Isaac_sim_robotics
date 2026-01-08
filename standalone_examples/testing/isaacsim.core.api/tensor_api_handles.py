# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

simulation_app = SimulationApp({"headless": True})
import omni.physx as _physx
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()
assets_root_path = get_assets_root_path()
asset_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka")
articulated_system_1 = my_world.scene.add(Robot(prim_path="/World/Franka", name="my_franka_1"))


def step_callback_1(step_size):
    b = articulated_system_1.get_joint_velocities()


physx_subs = _physx.get_physx_interface().subscribe_physics_step_events(step_callback_1)
my_world.reset()
for j in range(10):
    for i in range(5):
        my_world.step(render=False)
simulation_app.close()
