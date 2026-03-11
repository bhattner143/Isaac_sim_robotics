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

import os
import random

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import omni.replicator.core as rep
import omni.usd
from isaacsim.core.utils.semantics import add_labels
from pxr import UsdGeom


# Custom randomizer function using USD API
def randomize_location(prim):
    if not prim.GetAttribute("xformOp:translate"):
        UsdGeom.Xformable(prim).AddTranslateOp()
    translate = prim.GetAttribute("xformOp:translate")
    translate.Set((random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)))


def run_example():
    # Create a new stage and disable capture on play
    omni.usd.get_context().new_stage()
    rep.orchestrator.set_capture_on_play(False)
    random.seed(42)
    rep.set_global_seed(42)

    # Setup stage
    stage = omni.usd.get_context().get_stage()
    cube = stage.DefinePrim("/World/Cube", "Cube")
    add_labels(cube, labels=["MyCube"], instance_name="class")

    # Create a replicator randomizer with custom event trigger
    with rep.trigger.on_custom_event(event_name="randomize_dome_light_color"):
        rep.create.light(light_type="Dome", color=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

    # Create a render product using the viewport perspective camera
    rp = rep.create.render_product("/OmniverseKit_Persp", (512, 512))

    # Write data using the basic writer with the rgb and bounding box annotators
    writer = rep.writers.get("BasicWriter")
    out_dir = os.path.join(os.getcwd(), "_out_basic_writer_rand")
    print(f"Output directory: {out_dir}")
    writer.initialize(output_dir=out_dir, rgb=True, semantic_segmentation=True, colorize_semantic_segmentation=True)
    writer.attach(rp)

    # Trigger a data capture request (data will be written to disk by the writer)
    for i in range(3):
        print(f"Step {i}")
        # Trigger the custom event randomizer every other step
        if i % 2 == 1:
            rep.utils.send_og_event(event_name="randomize_dome_light_color")

        # Run the custom USD API location randomizer on the prims
        randomize_location(cube)

        # Since the replicator randomizer is set to trigger on custom events, step will only trigger the writer
        rep.orchestrator.step(rt_subframes=32)

    # Destroy the render product to release resources by detaching it from the writer first
    writer.detach()
    rp.destroy()

    # Wait for the data to be written to disk
    rep.orchestrator.wait_until_complete()


# Run the example
run_example()

simulation_app.close()
