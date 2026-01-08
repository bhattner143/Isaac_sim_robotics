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

"""Generate augmented synthetic from a writer
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import argparse
import os
import time

import carb.settings
import numpy as np
import omni.replicator.core as rep
import warp as wp
from isaacsim.core.utils.stage import open_stage
from isaacsim.storage.native import get_assets_root_path

parser = argparse.ArgumentParser()
parser.add_argument("--num_frames", type=int, default=25, help="The number of frames to capture")
parser.add_argument("--use_warp", action="store_true", help="Use warp augmentations instead of numpy")
args, unknown = parser.parse_known_args()

NUM_FRAMES = args.num_frames
USE_WARP = args.use_warp
ENV_URL = "/Isaac/Environments/Grid/default_environment.usd"

# Enable scripts
carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)


# Gaussian noise augmentation on rgba data in numpy (CPU) and warp (GPU)
def gaussian_noise_rgb_np(data_in, sigma: float, seed: int):
    np.random.seed(seed)
    data_in[:, :, 0] = data_in[:, :, 0] + np.random.randn(*data_in.shape[:-1]) * sigma
    data_in[:, :, 1] = data_in[:, :, 1] + np.random.randn(*data_in.shape[:-1]) * sigma
    data_in[:, :, 2] = data_in[:, :, 2] + np.random.randn(*data_in.shape[:-1]) * sigma
    return data_in


@wp.kernel
def gaussian_noise_rgb_wp(
    data_in: wp.array3d(dtype=wp.uint8), data_out: wp.array3d(dtype=wp.uint8), sigma: float, seed: int
):
    # Get thread coordinates and image dimensions to calculate unique pixel ID for random generation
    i, j = wp.tid()
    dim_i = data_in.shape[0]
    dim_j = data_in.shape[1]
    pixel_id = i * dim_i + j

    # Use pixel_id as offset to create unique seeds for each pixel and channel (ensure independent noise patterns across R,G,B channels)
    state_r = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 0))
    state_g = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 1))
    state_b = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 2))

    # Apply noise to each channel independently using unique seeds
    data_out[i, j, 0] = wp.uint8(wp.int32(data_in[i, j, 0]) + wp.int32(sigma * wp.randn(state_r)))
    data_out[i, j, 1] = wp.uint8(wp.int32(data_in[i, j, 1]) + wp.int32(sigma * wp.randn(state_g)))
    data_out[i, j, 2] = wp.uint8(wp.int32(data_in[i, j, 2]) + wp.int32(sigma * wp.randn(state_b)))
    data_out[i, j, 3] = data_in[i, j, 3]


# Gaussian noise augmentation on depth data in numpy (CPU) and warp (GPU)
def gaussian_noise_depth_np(data_in, sigma: float, seed: int):
    np.random.seed(seed)
    return data_in + np.random.randn(*data_in.shape) * sigma


rep.AnnotatorRegistry.register_augmentation(
    "gn_depth_np", rep.annotators.Augmentation.from_function(gaussian_noise_depth_np, sigma=0.1, seed=None)
)


@wp.kernel
def gaussian_noise_depth_wp(
    data_in: wp.array2d(dtype=wp.float32), data_out: wp.array2d(dtype=wp.float32), sigma: float, seed: int
):
    i, j = wp.tid()
    # Unique ID for random seed per pixel
    scalar_pixel_id = i * data_in.shape[1] + j
    state = wp.rand_init(seed, scalar_pixel_id)
    data_out[i, j] = data_in[i, j] + sigma * wp.randn(state)


rep.AnnotatorRegistry.register_augmentation(
    "gn_depth_wp", rep.annotators.Augmentation.from_function(gaussian_noise_depth_wp, sigma=0.1, seed=None)
)

# Setup the environment
assets_root_path = get_assets_root_path()
open_stage(assets_root_path + ENV_URL)

# Disable capture on play and async rendering
carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
carb.settings.get_settings().set("/omni/replicator/asyncRendering", False)
carb.settings.get_settings().set("/app/asyncRendering", False)

# Create a red cube and a render product from a camera looking at the cube from the top
red_mat = rep.create.material_omnipbr(diffuse=(1, 0, 0))
red_cube = rep.create.cube(position=(0, 0, 0.71), material=red_mat)
cam = rep.create.camera(position=(0, 0, 5), look_at=(0, 0, 0))
rp = rep.create.render_product(cam, (512, 512))

# Update the app a couple of times to fully load texture/materials
for _ in range(5):
    simulation_app.update()

# Access default annotators from replicator
rgb_to_hsv_augm = rep.annotators.Augmentation.from_function(rep.augmentations_default.aug_rgb_to_hsv)
hsv_to_rgb_augm = rep.annotators.Augmentation.from_function(rep.augmentations_default.aug_hsv_to_rgb)

# Access the custom annotators as functions or from the registry
gn_rgb_augm = None
gn_depth_augm = None
if USE_WARP:
    gn_rgb_augm = rep.annotators.Augmentation.from_function(gaussian_noise_rgb_wp, sigma=6.0, seed=None)
    gn_depth_augm = rep.AnnotatorRegistry.get_augmentation("gn_depth_wp")
else:
    gn_rgb_augm = rep.annotators.Augmentation.from_function(gaussian_noise_rgb_np, sigma=6.0, seed=None)
    gn_depth_augm = rep.AnnotatorRegistry.get_augmentation("gn_depth_np")

# Create a writer and apply the augmentations to its corresponding annotators
out_dir = os.path.join(os.getcwd(), "_out_augm_writer")
print(f"Writing data to: {out_dir}")
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=out_dir, rgb=True, distance_to_camera=True)

augmented_rgb_annot = rep.annotators.get("rgb").augment_compose(
    [rgb_to_hsv_augm, gn_rgb_augm, hsv_to_rgb_augm], name="rgb"
)
writer.add_annotator(augmented_rgb_annot)
writer.augment_annotator("distance_to_camera", gn_depth_augm)

# Attach render product to writer
writer.attach([rp])

# Generate a replicator graph randomizing the cube's rotation every frame
with rep.trigger.on_frame():
    with red_cube:
        rep.randomizer.rotation()

# Evaluate the graph
rep.orchestrator.preview()

# Measure the duration of capturing the data
start_time = time.time()

# The `step()` function will trigger the randomization graph, feed annotators with new data, and trigger the writers
for i in range(NUM_FRAMES):
    rep.orchestrator.step(rt_subframes=32)

print(
    f"The duration for capturing {NUM_FRAMES} frames using '{'warp' if USE_WARP else 'numpy'}' was: {time.time() - start_time:.4f} seconds, with an average of {(time.time() - start_time) / NUM_FRAMES:.4f} seconds per frame."
)

simulation_app.close()
