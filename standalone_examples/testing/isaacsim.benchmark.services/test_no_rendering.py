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

simulation_app = SimulationApp({"headless": True})

import sys

import carb
import omni.kit.test
from isaacsim.core.api import PhysicsContext
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom

enable_extension("isaacsim.benchmark.services")
from isaacsim.benchmark.services import BaseIsaacBenchmark

# ----------------------------------------------------------------------
# Create benchmark
benchmark = BaseIsaacBenchmark(
    benchmark_name="benchmark_physx_lidar",
)


assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()


asset_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
robot = add_reference_to_stage(usd_path=asset_path, prim_path="/World/panda")
robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
robot.GetVariantSet("Mesh").SetVariantSelection("Quality")

benchmark.set_phase("benchmark")

timeline = omni.timeline.get_timeline_interface()
timeline.play()

physics_context = PhysicsContext(physics_dt=1.0 / 60.0)
time = 0
for _ in range(0, 1000):
    physics_context._step(time)
    time += physics_context.get_physics_dt()

benchmark.store_measurements()
min_physics_time = 0
for measurement in benchmark._test_phases[0].measurements:
    if "Min Physics Frametime" in measurement.name:
        min_physics_time = measurement.value

benchmark.stop()
timeline.stop()
assert min_physics_time > 0
simulation_app.close()
