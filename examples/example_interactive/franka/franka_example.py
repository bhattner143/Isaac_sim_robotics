# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import isaacsim.core.utils.stage as stage_utils
import numpy as np
import omni
from isaacsim.core.prims import SingleArticulation
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.robot.policy.examples.robots.franka import FrankaOpenDrawerPolicy
from isaacsim.storage.native import get_assets_root_path


class FrankaExample(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 400.0
        self._world_settings["rendering_dt"] = 1.0 / 60.0

    def setup_scene(self) -> None:

        self.get_world().scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.01,
        )

        cabinet_prim_path = "/World/cabinet"
        cabinet_usd_path = get_assets_root_path() + "/Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"

        cabinet_name = "cabinet"
        cabinet_position = np.array([0.8, 0.0, 0.4])
        cabinet_orientation = np.array([0.0, 0.0, 0.0, 1.0])

        stage_utils.add_reference_to_stage(cabinet_usd_path, cabinet_prim_path)

        self.cabinet = SingleArticulation(
            prim_path=cabinet_prim_path, name=cabinet_name, position=cabinet_position, orientation=cabinet_orientation
        )

        self.franka = FrankaOpenDrawerPolicy(
            prim_path="/World/franka", name="franka", position=np.array([0, -0, 0]), cabinet=self.cabinet
        )

        timeline = omni.timeline.get_timeline_interface()
        self._event_timer_callback = timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP), self._timeline_timer_callback_fn
        )

    async def setup_post_load(self) -> None:
        self._physics_ready = False
        self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)

        await self.get_world().play_async()

    async def setup_post_reset(self) -> None:
        self._physics_ready = False
        self.franka.previous_action = np.zeros(9)
        await self.get_world().play_async()

    def on_physics_step(self, step_size) -> None:
        if self._physics_ready:
            self.franka.forward(step_size)
        else:
            self._physics_ready = True
            self.franka.initialize()
            self.franka.post_reset()
            self.franka.robot.set_joints_default_state(self.franka.default_pos)

    def _timeline_timer_callback_fn(self, event) -> None:
        if self.franka:
            self._physics_ready = False

    def world_cleanup(self):
        world = self.get_world()
        self._event_timer_callback = None
        if world.physics_callback_exists("physics_step"):
            world.remove_physics_callback("physics_step")
