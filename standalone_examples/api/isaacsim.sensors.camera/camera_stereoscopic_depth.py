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

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.test})

import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
import omni
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCone, VisualCuboid
from isaacsim.sensors.camera import SingleViewDepthSensor
from isaacsim.storage.native.nucleus import get_assets_root_path
from PIL import Image, ImageDraw

# Create a world
world = World(stage_units_in_meters=1.0)

# Add two cubes and a cone
cube_1 = world.scene.add(
    VisualCuboid(
        prim_path="/cube_1",
        name="cube_1",
        position=np.array([0.25, 0.25, 0.25]),
        scale=np.array([0.5, 0.5, 0.5]),
        size=1.0,
        color=np.array([255, 0, 0]),
    )
)
cube_2 = world.scene.add(
    VisualCuboid(
        prim_path="/cube_2",
        name="cube_2",
        position=np.array([-1.0, -1.0, 0.25]),
        scale=np.array([1.0, 1.0, 1.0]),
        size=1.0,
        color=np.array([0, 0, 255]),
    )
)
cone = world.scene.add(
    VisualCone(
        prim_path="/cone",
        name="cone",
        position=np.array([-0.1, -0.3, 0.2]),
        scale=np.array([1.0, 1.0, 1.0]),
        color=np.array([0, 255, 0]),
    )
)
# Add a stereoscopic camera
camera = SingleViewDepthSensor(
    prim_path="/World/camera",
    name="depth_camera",
    position=np.array([3.0, 0.0, 0.6]),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 180]), degrees=True),
    frequency=20,
    resolution=(1920, 1080),
)

# Initialize the black grid scene for the background
assets_root_path = get_assets_root_path()
path_to = omni.kit.commands.execute(
    "CreateReferenceCommand",
    usd_context=omni.usd.get_context(),
    path_to="/World/black_grid",
    asset_path=assets_root_path + "/Isaac/Environments/Grid/gridroom_black.usd",
    instanceable=False,
)

# Reset the world state
world.reset()

# Initialize the camera, applying the appropriate schemas to the render product to enable depth sensing
camera.initialize(attach_rgb_annotator=False)

# Now that the camera is initialized, we can configure its parameters
# First, camera lens parameters
camera.set_focal_length(1.814756)
camera.set_focus_distance(400.0)
# Next, depth sensor parameters
camera.set_baseline_mm(55)
camera.set_focal_length_pixel(891.0)
camera.set_sensor_size_pixel(1280.0)
camera.set_max_disparity_pixel(110.0)
camera.set_confidence_threshold(0.99)
camera.set_noise_mean(0.5)
camera.set_noise_sigma(1.0)
camera.set_noise_downscale_factor_pixel(1.0)
camera.set_min_distance(0.5)
camera.set_max_distance(9999.9)

# Attach the DepthSensorDistance annotator and DistanceToImagePlane annotator to the camera
camera.attach_annotator("DepthSensorDistance")
camera.attach_annotator("distance_to_image_plane")

# Run for 10 frames in test mode
i = 0
while simulation_app.is_running() and (not args.test or i < 10):
    world.step(render=True)
    i += 1

# Retrieve the rendered frames
latest_frame = camera.get_current_frame()
img_stereoscopic_depth = Image.fromarray(latest_frame["DepthSensorDistance"]).convert("RGB")
img_distance_to_image_plane = Image.fromarray(latest_frame["distance_to_image_plane"]).convert("RGB")

# Save the rendered frames
img_stereoscopic_depth.save("depth_sensor_distance.png")
img_distance_to_image_plane.save("distance_to_image_plane.png")

simulation_app.close()
