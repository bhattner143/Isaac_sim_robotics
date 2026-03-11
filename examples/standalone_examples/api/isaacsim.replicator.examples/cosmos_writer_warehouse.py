# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import carb
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdGeom

STAGE_URL = "/Isaac/Samples/Replicator/Stage/full_warehouse_worker_and_anim_cameras.usd"
CARTER_NAV_ASSET_URL = "/Isaac/Samples/Replicator/OmniGraph/nova_carter_nav_only.usd"
CARTER_NAV_PATH = "/NavWorld/CarterNav"
CARTER_CHASSIS_PATH = f"{CARTER_NAV_PATH}/chassis_link"
CARTER_NAV_TARGET_PATH = f"{CARTER_NAV_PATH}/targetXform"
CARTER_CAMERA_PATH = f"{CARTER_NAV_PATH}/chassis_link/sensors/front_hawk/left/camera_left"
CARTER_NAV_POSITION = (-6, 4, 0)
CARTER_NAV_TARGET_POSITION = (3, 3, 0)


def advance_timeline_by_duration(duration: float, max_updates: int = 1000):
    timeline = omni.timeline.get_timeline_interface()
    current_time = timeline.get_current_time()
    target_time = current_time + duration

    if timeline.get_end_time() < target_time:
        timeline.set_end_time(1000000)

    if not timeline.is_playing():
        timeline.play()

    print(f"Advancing timeline from {current_time:.4f}s to {target_time:.4f}s")
    step_count = 0
    while current_time < target_time:
        if step_count >= max_updates:
            print(f"Max updates reached: {step_count}, finishing timeline advance.")
            break

        prev_time = current_time
        simulation_app.update()
        current_time = timeline.get_current_time()
        step_count += 1

        if step_count % 10 == 0:
            print(f"\tStep {step_count}, {current_time:.4f}s/{target_time:.4f}s")

        if current_time <= prev_time:
            print(f"Warning: Timeline did not advance at update {step_count} (time: {current_time:.4f}s).")
    print(f"Finished advancing timeline to {timeline.get_end_time():.4f}s in {step_count} steps")


def run_sdg_pipeline(camera_path, num_frames, capture_interval, use_instance_id=True, segmentation_mapping=None):
    rp = rep.create.render_product(camera_path, (1280, 720))
    cosmos_writer = rep.WriterRegistry.get("CosmosWriter")
    backend = rep.backends.get("DiskBackend")
    out_dir = os.path.join(os.getcwd(), f"_out_cosmos")
    print(f"output_directory: {out_dir}")
    backend.initialize(output_dir=out_dir)
    cosmos_writer.initialize(
        backend=backend, use_instance_id=use_instance_id, segmentation_mapping=segmentation_mapping
    )
    cosmos_writer.attach(rp)

    # Make sure the timeline is playing
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    print(f"Starting SDG pipeline. Capturing {num_frames} frames, every {capture_interval} simulation step(s).")
    frames_captured_count = 0
    simulation_step_index = 0
    while frames_captured_count < num_frames:
        print(f"Simulation step {simulation_step_index}")
        if simulation_step_index % capture_interval == 0:
            print(f"\t Capturing frame {frames_captured_count + 1}/{num_frames}")
            rep.orchestrator.step(pause_timeline=False)
            frames_captured_count += 1
        else:
            simulation_app.update()
        simulation_step_index += 1

    print("Waiting to finish processing and writing the data")
    rep.orchestrator.wait_until_complete()
    print(f"Finished SDG pipeline. Captured {frames_captured_count} frames")
    cosmos_writer.detach()
    rp.destroy()
    timeline.pause()


def run_example(num_frames, capture_interval=1, start_delay=None, use_instance_id=True, segmentation_mapping=None):
    assets_root_path = get_assets_root_path()
    stage_path = assets_root_path + STAGE_URL
    print(f"Opening stage: '{stage_path}'")
    omni.usd.get_context().open_stage(stage_path)
    stage = omni.usd.get_context().get_stage()

    # Enable script nodes
    carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)

    # Disable capture on play on the new stage, data is captured manually using the step function
    rep.orchestrator.set_capture_on_play(False)

    # Load carter nova asset with its navigation graph
    carter_url_path = assets_root_path + CARTER_NAV_ASSET_URL
    print(f"Loading carter nova asset: '{carter_url_path}' at prim path: '{CARTER_NAV_PATH}'")
    carter_nav_prim = add_reference_to_stage(usd_path=carter_url_path, prim_path=CARTER_NAV_PATH)
    if not carter_nav_prim.GetAttribute("xformOp:translate"):
        UsdGeom.Xformable(carter_nav_prim).AddTranslateOp()
    carter_nav_prim.GetAttribute("xformOp:translate").Set(CARTER_NAV_POSITION)

    # Set the navigation target position
    carter_navigation_target_prim = stage.GetPrimAtPath(CARTER_NAV_TARGET_PATH)
    if not carter_navigation_target_prim.IsValid():
        print(f"Carter navigation target prim not found at path: {CARTER_NAV_TARGET_PATH}, exiting")
        return
    if not carter_navigation_target_prim.GetAttribute("xformOp:translate"):
        UsdGeom.Xformable(carter_navigation_target_prim).AddTranslateOp()
    carter_navigation_target_prim.GetAttribute("xformOp:translate").Set(CARTER_NAV_TARGET_POSITION)

    # Use the carter nova front hawk camera for capturing data
    camera_prim = stage.GetPrimAtPath(CARTER_CAMERA_PATH)
    if not camera_prim.IsValid():
        print(f"Camera prim not found at path: {CARTER_CAMERA_PATH}, exiting")
        return

    # Advance the timeline with the start delay if provided
    if start_delay is not None and start_delay > 0:
        advance_timeline_by_duration(start_delay)

    # Run the SDG pipeline
    run_sdg_pipeline(camera_prim.GetPath(), num_frames, capture_interval, use_instance_id, segmentation_mapping)


NUM_FRAMES = 4
CAPTURE_INTERVAL = 2
START_DELAY = 0.1
run_example(num_frames=NUM_FRAMES, capture_interval=4, start_delay=START_DELAY, use_instance_id=True)
