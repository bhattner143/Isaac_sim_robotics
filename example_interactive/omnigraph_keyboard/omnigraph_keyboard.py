# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import numpy as np
import omni.ext
import omni.graph.core as og
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.examples.interactive.base_sample import BaseSample


class OmnigraphKeyboard(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._gamepad_gains = (40.0, 40.0, 2.0)
        self._gamepad_deadzone = 0.15

    def setup_scene(self):
        world = self.get_world()
        world.scene.add(
            VisualCuboid(
                prim_path="/Cube",  # The prim path of the cube in the USD stage
                name="cube",  # The unique name used to retrieve the object from the scene later on
                position=np.array([0, 0, 1.0]),  # Using the current stage units which is cms by default.
                size=1.0,  # most arguments accept mainly numpy arrays.
                color=np.array([0, 1.0, 1.0]),  # RGB channels, going from 0-1
            )
        )
        world.scene.add_default_ground_plane()
        set_camera_view(eye=np.array([5, 5, 3]), target=np.array([0, 0, 0]))

        # setup graph
        keys = og.Controller.Keys
        og.Controller.edit(
            {"graph_path": "/controller_graph", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("A", "omni.graph.nodes.ReadKeyboardState"),
                    ("D", "omni.graph.nodes.ReadKeyboardState"),
                    ("ToDouble1", "omni.graph.nodes.ToDouble"),
                    ("ToDouble2", "omni.graph.nodes.ToDouble"),
                    ("Negate", "omni.graph.nodes.Multiply"),
                    ("DeltaAdd", "omni.graph.nodes.Add"),
                    ("SizeAdd", "omni.graph.nodes.Add"),
                    ("NegOne", "omni.graph.nodes.ConstantInt"),
                    ("ScaleDown", "omni.graph.nodes.Multiply"),
                    ("ScaleFactor", "omni.graph.nodes.ConstantDouble"),
                    ("CubeWrite", "omni.graph.nodes.WritePrimAttribute"),  # write prim property translate
                    ("CubeRead", "omni.graph.nodes.ReadPrimAttribute"),
                ],
                keys.SET_VALUES: [
                    ("A.inputs:key", "A"),
                    ("D.inputs:key", "D"),
                    ("OnTick.inputs:onlyPlayback", True),  # only tick when simulator is playing
                    ("NegOne.inputs:value", -1),
                    ("ScaleFactor.inputs:value", 0.1),
                    ("CubeWrite.inputs:name", "size"),
                    ("CubeWrite.inputs:primPath", "/Cube"),
                    ("CubeWrite.inputs:usePath", True),
                    ("CubeRead.inputs:name", "size"),
                    ("CubeRead.inputs:primPath", "/Cube"),
                    ("CubeRead.inputs:usePath", True),
                ],
                keys.CONNECT: [
                    ("OnTick.outputs:tick", "CubeWrite.inputs:execIn"),
                    ("A.outputs:isPressed", "ToDouble1.inputs:value"),
                    ("D.outputs:isPressed", "ToDouble2.inputs:value"),
                    ("ToDouble2.outputs:converted", "Negate.inputs:a"),
                    ("NegOne.inputs:value", "Negate.inputs:b"),
                    ("ToDouble1.outputs:converted", "DeltaAdd.inputs:a"),
                    ("Negate.outputs:product", "DeltaAdd.inputs:b"),
                    ("DeltaAdd.outputs:sum", "ScaleDown.inputs:a"),
                    ("CubeRead.outputs:value", "SizeAdd.inputs:b"),
                    ("SizeAdd.outputs:sum", "CubeWrite.inputs:value"),
                    ("ScaleFactor.inputs:value", "ScaleDown.inputs:b"),
                    ("ScaleDown.outputs:product", "SizeAdd.inputs:a"),
                ],
            },
        )
