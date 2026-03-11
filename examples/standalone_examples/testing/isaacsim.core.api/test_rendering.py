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

import torch
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import XFormPrim

my_world = World(stage_units_in_meters=1.0, device="cuda:0", backend="torch")
cube_2 = my_world.scene.add(
    DynamicCuboid(
        prim_path="/new_cube_2",
        name="cube_1",
        position=torch.tensor([0, 0, 1.0]),
        scale=torch.tensor([0.6, 0.5, 0.2]),
        size=1.0,
        color=torch.tensor([255, 0, 0]),
    )
)
xfrom_cube = XFormPrim("/new_cube_2")
my_world.scene.add_default_ground_plane()
my_world.reset()
for i in range(500):
    my_world.step(render=False)
my_world.render()
if not (xfrom_cube.get_world_poses()[0][:, -1].item() < 10e-02):
    raise (ValueError(f"PhysX status is not updated in the rendering call"))

simulation_app.close()
