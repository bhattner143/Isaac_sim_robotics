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

import numpy as np
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.robot.manipulators.examples.universal_robots.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.manipulators.examples.universal_robots.tasks import BinFilling as BinFillingTask


class BinFilling(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._controller = None
        self._articulation_controller = None
        self._added_screws = False

    def setup_scene(self):
        world = self.get_world()
        world.add_task(BinFillingTask(name="bin_filling"))
        return

    async def setup_post_load(self):
        self._ur10_task = self._world.get_task(name="bin_filling")
        self._task_params = self._ur10_task.get_params()
        my_ur10 = self._world.scene.get_object(self._task_params["robot_name"]["value"])
        self._controller = PickPlaceController(
            name="pick_place_controller", gripper=my_ur10.gripper, robot_articulation=my_ur10
        )
        self._articulation_controller = my_ur10.get_articulation_controller()
        return

    def _on_fill_bin_physics_step(self, step_size):
        observations = self._world.get_observations()
        actions = self._controller.forward(
            picking_position=observations[self._task_params["bin_name"]["value"]]["position"],
            placing_position=observations[self._task_params["bin_name"]["value"]]["target_position"],
            current_joint_positions=observations[self._task_params["robot_name"]["value"]]["joint_positions"],
            end_effector_offset=np.array([0, -0.098, 0.03]),
            end_effector_orientation=euler_angles_to_quat(np.array([np.pi, 0, np.pi / 2.0])),
        )
        if not self._added_screws and self._controller.get_current_event() == 6 and not self._controller.is_paused():
            self._controller.pause()
            self._ur10_task.add_screws(screws_number=20)
            self._added_screws = True
        if self._controller.is_done():
            self._world.pause()
        self._articulation_controller.apply_action(actions)
        return

    async def on_fill_bin_event_async(self):
        world = self.get_world()
        world.add_physics_callback("sim_step", self._on_fill_bin_physics_step)
        await world.play_async()
        return

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        self._controller.reset()
        self._added_screws = False
        return

    def world_cleanup(self):
        self._controller = None
        self._added_screws = False
        return
