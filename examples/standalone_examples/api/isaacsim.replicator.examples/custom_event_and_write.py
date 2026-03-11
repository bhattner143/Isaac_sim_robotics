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

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import os

import omni.replicator.core as rep
import omni.usd

omni.usd.get_context().new_stage()
distance_light = rep.create.light(rotation=(315, 0, 0), intensity=4000, light_type="distant")

large_cube = rep.create.cube(scale=1.25, position=(1, 1, 0))
small_cube = rep.create.cube(scale=0.75, position=(-1, -1, 0))
large_cube_prim = large_cube.get_output_prims()["prims"][0]
small_cube_prim = small_cube.get_output_prims()["prims"][0]

rp = rep.create.render_product("/OmniverseKit_Persp", (512, 512))
writer = rep.WriterRegistry.get("BasicWriter")
out_dir = os.path.join(os.getcwd(), "_out_custom_event")
print(f"Writing data to {out_dir}")
writer.initialize(output_dir=out_dir, rgb=True)
writer.attach(rp)

with rep.trigger.on_custom_event(event_name="randomize_large_cube"):
    with large_cube:
        rep.randomizer.rotation()

with rep.trigger.on_custom_event(event_name="randomize_small_cube"):
    with small_cube:
        rep.randomizer.rotation()


def run_example():
    print(f"Randomizing small cube")
    rep.utils.send_og_event(event_name="randomize_small_cube")
    print("Capturing frame")
    rep.orchestrator.step(rt_subframes=8)

    print("Moving small cube")
    small_cube_prim.GetAttribute("xformOp:translate").Set((-2, -2, 0))
    print("Capturing frame")
    rep.orchestrator.step(rt_subframes=8)

    print(f"Randomizing large cube")
    rep.utils.send_og_event(event_name="randomize_large_cube")
    print("Capturing frame")
    rep.orchestrator.step(rt_subframes=8)

    print("Moving large cube")
    large_cube_prim.GetAttribute("xformOp:translate").Set((2, 2, 0))
    print("Capturing frame")
    rep.orchestrator.step(rt_subframes=8)

    # Wait until all the data is saved to disk
    rep.orchestrator.wait_until_complete()


run_example()

simulation_app.close()
