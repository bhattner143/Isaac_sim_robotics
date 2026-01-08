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


import carb
import omni.graph.core as og
import omni.usd
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.storage.native import get_assets_root_path
from pxr import Sdf, UsdLux


class KayaGamepad(BaseSample):
    def __init__(self) -> None:
        super().__init__()

    def setup_scene(self):
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        # add kaya robot rigged with gamepad controller
        kaya_ogn_usd = assets_root_path + "/Isaac/Robots/NVIDIA/Kaya/kaya_ogn_gamepad.usd"
        stage = omni.usd.get_context().get_stage()
        graph_prim = stage.DefinePrim("/kaya", "Xform")
        graph_prim.GetReferences().AddReference(kaya_ogn_usd)

        # add ground plane and light
        GroundPlane("/World/ground_plane", visible=True)
        dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
        dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(450.0)

    def world_cleanup(self):
        pass
