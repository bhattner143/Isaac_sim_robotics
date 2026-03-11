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

simulation_app = SimulationApp()

import omni.kit.app
from isaacsim.core.api import World

world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")
world.scene.add_default_ground_plane()
world.reset()

frame_idx = 0
while simulation_app.is_running():
    if world.is_playing():
        world.step(render=True)
    else:
        simulation_app.update()
        # we should exit this loop before we hit frame 200 unless we are stuck on an exit screen
        assert frame_idx < 200
    # try exiting, it should exit unless a save file dialog shows up.
    if frame_idx == 100:
        omni.kit.app.get_app().post_quit()
    frame_idx += 1

simulation_app.close()
