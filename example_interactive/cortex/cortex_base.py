# SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import gc
from abc import abstractmethod

from isaacsim.core.api import World
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.tasks.base_task import BaseTask
from isaacsim.core.utils.stage import create_new_stage_async, update_stage_async
from isaacsim.cortex.framework.cortex_world import CortexWorld
from isaacsim.examples.interactive import base_sample


class CortexBase(base_sample.BaseSample):
    async def load_world_async(self):
        """
        Function called when clicking load buttton.
        The difference between this class and Base Sample is that we initialize a CortexWorld specialization.
        """
        if CortexWorld.instance() is None:
            await create_new_stage_async()
            self._world = CortexWorld(**self._world_settings)
            await self._world.initialize_simulation_context_async()
            self.setup_scene()
        else:
            self._world = CortexWorld.instance()
        self._current_tasks = self._world.get_current_tasks()
        await self._world.reset_async()
        await self._world.pause_async()
        await self.setup_post_load()
        if len(self._current_tasks) > 0:
            self._world.add_physics_callback("tasks_step", self._world.step_async)
        return
