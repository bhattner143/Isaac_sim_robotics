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
import sys

from isaacsim import SimulationApp

# The most basic usage for creating a simulation app
kit = SimulationApp()

ADDITIONAL_EXTENSIONS_PEOPLE = [
    "omni.isaac.core",
    "omni.anim.people",
    "omni.anim.navigation.bundle",
    "omni.anim.timeline",
    "omni.anim.graph.bundle",
    "omni.anim.graph.core",
    "omni.anim.graph.ui",
    "omni.anim.retarget.bundle",
    "omni.anim.retarget.core",
    "omni.anim.retarget.ui",
    "omni.kit.scripting",
]

import carb
import omni
from isaacsim.core.utils.extensions import enable_extension

for e in ADDITIONAL_EXTENSIONS_PEOPLE:
    enable_extension(e)
    kit.update()

enable_extension("isaacsim.ros2.bridge")
kit.update()

# Locate Isaac Sim assets folder to load sample
from isaacsim.storage.native import get_assets_root_path, is_file

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    kit.close()
    sys.exit()
usd_path = assets_root_path + "/Isaac/Samples/NvBlox/nvblox_sample_scene.usd"

omni.usd.get_context().open_stage(usd_path)

for i in range(100):
    kit.update()

omni.timeline.get_timeline_interface().play()

for i in range(100):
    kit.update()

kit.close()  # Cleanup application
