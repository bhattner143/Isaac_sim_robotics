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

import asyncio
import os

import omni.kit.app
import omni.usd
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.storage.native import get_assets_root_path

# Make sure the grasping extension is loaded and enabled
ext_manager = omni.kit.app.get_app().get_extension_manager()
if not ext_manager.is_extension_enabled("isaacsim.replicator.grasping"):
    ext_manager.set_extension_enabled_immediate("isaacsim.replicator.grasping", True)
from isaacsim.replicator.grasping.grasping_manager import GraspingManager


def run_example(
    stage_path,
    config_path=None,
    sampler_config=None,
    physics_scene_path=None,
    output_dir=None,
    gripper_path=None,
    object_prim_path=None,
):
    assets_root_path = get_assets_root_path()
    print(f"Assets root path: {assets_root_path}")
    stage_url = assets_root_path + stage_path
    print(f"Opening stage: {stage_url}")
    omni.usd.get_context().open_stage(stage_url)
    stage = omni.usd.get_context().get_stage()

    grasping_manager = GraspingManager()

    if config_path is not None:
        load_status = grasping_manager.load_config(config_path)
        print(f"Config load status: {load_status}")

    # Make sure the object to grasp is set (either from the config file or from the argument)
    if not grasping_manager.get_object_prim_path() and object_prim_path:
        grasping_manager.object_path = object_prim_path

    if not grasping_manager.get_object_prim_path():
        print("Warning: Object to grasp is not set (missing in config and argument). Aborting.")
        return

    # Make sure the gripper is set (either from the config file or from the argument)
    if not grasping_manager.gripper_path and gripper_path:
        grasping_manager.gripper_path = gripper_path

    if not grasping_manager.gripper_path:
        print("Warning: Gripper path is not set (missing in config and argument). Aborting.")
        return

    # If there are already grasp poses in the configuration, don't generate new ones
    if grasping_manager.grasp_locations:
        print(
            f"Found {len(grasping_manager.grasp_locations)} grasp poses in the configuration file. No new poses will be generated."
        )
    else:
        print("No grasp poses found in configuration, generating new ones...")

        # Determine Sampler Configuration
        if not (grasping_manager.sampler_config and grasping_manager.sampler_config.get("sampler_type")):
            if sampler_config:
                grasping_manager.sampler_config = sampler_config.copy()
            else:
                print(
                    "Warning: Sampler configuration is missing or invalid (not in config file and not provided as argument). Aborting pose generation."
                )
                return

        # Generate the grasp poses
        success_generation = grasping_manager.generate_grasp_poses()
        if not success_generation or not grasping_manager.grasp_locations:
            print("Failed to generate grasp poses or no poses were generated.")
            return
        print(f"Generated {len(grasping_manager.grasp_locations)} new grasp poses.")

    # Store the initial gripper pose to be able to restore it after the evaluation
    grasping_manager.store_initial_gripper_pose()

    print("Evaluating grasp poses...")
    poses_to_evaluate = grasping_manager.get_grasp_poses(in_world_frame=True)
    if not poses_to_evaluate:
        print("No poses available to evaluate..")
        return

    # Determine Output Path
    if not output_dir:
        print("Warning: Output path is not defined data will not be saved.")

    # Set the output path and overwrite flag
    grasping_manager.set_results_output_dir(output_dir)
    grasping_manager.set_overwrite_results_output(True)

    # Determine Physics Scene Path
    physics_scene_path_for_eval = None
    if physics_scene_path and stage.GetPrimAtPath(physics_scene_path):
        physics_scene_path_for_eval = physics_scene_path
    print(f"Physics scene path for evaluation: {physics_scene_path_for_eval}")

    grasping_workflow_task = asyncio.ensure_future(
        grasping_manager.evaluate_grasp_poses(
            grasp_poses=poses_to_evaluate,
            render=True,
            physics_scene_path=physics_scene_path_for_eval,
            simulate_using_timeline=False,
        )
    )

    while not grasping_workflow_task.done():
        simulation_app.update()

    print("Grasping workflow example finished.")
    grasping_manager.clear()


stage_path = "/Isaac/Samples/Replicator/Stage/sdg_grasping_xarm.usd"
ext_path = get_extension_path_from_name("isaacsim.replicator.grasping")
config_path = os.path.join(ext_path, "data/gripper_configs/xarm_antipodal_soup_can.yaml")
output_dir = os.path.join(os.getcwd(), "xarm_antipodal")

run_example(stage_path=stage_path, config_path=config_path, output_dir=output_dir)


simulation_app.close()
