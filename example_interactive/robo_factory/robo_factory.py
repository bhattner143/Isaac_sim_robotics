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
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.robot.manipulators.examples.franka.controllers.stacking_controller import StackingController
from isaacsim.robot.manipulators.examples.franka.tasks import Stacking


class RoboFactory(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._tasks = []
        self._controllers = []
        self._articulation_controllers = []
        self._robots = []
        self._num_of_tasks = 4
        return

    def setup_scene(self):
        world = self.get_world()
        for i in range(self._num_of_tasks):
            task = Stacking(name="task" + str(i), offset=np.array([0, (i * 2) - 3, 0]))
            world.add_task(task)
        return

    async def setup_post_load(self):
        for i in range(self._num_of_tasks):
            self._tasks.append(self._world.get_task(name="task" + str(i)))
        for i in range(self._num_of_tasks):
            self._robots.append(self._world.scene.get_object(self._tasks[i].get_params()["robot_name"]["value"]))
            self._controllers.append(
                StackingController(
                    name="stacking_controller",
                    gripper=self._robots[i].gripper,
                    robot_articulation=self._robots[i],
                    picking_order_cube_names=self._tasks[i].get_cube_names(),
                    robot_observation_name=self._robots[i].name,
                )
            )
        for i in range(self._num_of_tasks):
            self._articulation_controllers.append(self._robots[i].get_articulation_controller())
        return

    def _on_start_factory_physics_step(self, step_size):
        observations = self._world.get_observations()
        for i in range(self._num_of_tasks):
            actions = self._controllers[i].forward(observations=observations, end_effector_offset=np.array([0, 0, 0]))
            self._articulation_controllers[i].apply_action(actions)
        return

    async def _on_start_stacking_event_async(self):
        world = self.get_world()
        world.add_physics_callback("sim_step", self._on_start_factory_physics_step)
        await world.play_async()
        return

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
            for i in range(len(self._controllers)):
                self._controllers[i].reset()
        return

    def world_cleanup(self):
        self._tasks = []
        self._controllers = []
        self._articulation_controllers = []
        self._robots = []
        return
