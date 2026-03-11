# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import os
import weakref

import numpy as np
import omni.ext
import omni.ui as ui
import omni.usd
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from isaacsim.examples.interactive.getting_started.getting_started import GettingStarted
from isaacsim.gui.components.ui_utils import btn_builder


class GettingStartedExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "Part I: Basics"
        self.category = "Tutorials"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Getting Started",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/introduction/quickstart_isaacsim.html",
            "overview": " Select the tutorial in the Example Browser again to restart the tutorial.\n\n 'Reset' Button is disabled. to Restart, click on the thumbnail in the browser instead. \n\n This Example follows the 'Getting Started' tutorials from the documentation\n\n Press the 'Open in IDE' button to view the source code.",
            "sample": GettingStarted(),
        }

        ui_handle = GettingStartedUI(**ui_kwargs)

        # register the example with examples browser
        get_browser_instance().register_example(
            name=self.example_name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

        return

    def on_shutdown(self):
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)

        return


class GettingStartedUI(BaseSampleUITemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def build_extra_frames(self):
        extra_stacks = self.get_extra_frames_handle()
        self.task_ui_elements = {}
        with extra_stacks:
            with ui.CollapsableFrame(
                title="Getting Started with Isaac Sim",
                width=ui.Fraction(0.33),
                height=0,
                visible=True,
                collapsed=False,
                # style=get_style(),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                self.build_getting_started_ui()

    def build_getting_started_ui(self):
        with ui.VStack(spacing=5):
            dict = {
                "label": "Add Ground Plane",
                "type": "button",
                "text": "Add Ground Plane",
                "tooltip": "Add a ground plane to the scene",
                "on_clicked_fn": self._add_ground_plane,
            }
            self.task_ui_elements["Add Ground Plane"] = btn_builder(**dict)

            dict = {
                "label": "Add a Light Source",
                "type": "button",
                "text": "Add Light Source",
                "tooltip": "Add a light source to the scene",
                "on_clicked_fn": self._add_light_source,
            }

            self.task_ui_elements["Add Light Source"] = btn_builder(**dict)

            dict = {
                "label": "Add Visual Cube",
                "type": "button",
                "text": "Add Visual Cube",
                "tooltip": "Add a visual cube to scene. A visual cube has no physics properties.",
                "on_clicked_fn": self._add_visual_cube,
            }
            self.task_ui_elements["Add Visual Cube"] = btn_builder(**dict)

            dict = {
                "label": "Add Physics Cube",
                "type": "button",
                "text": "Add Physics Cube",
                "tooltip": "Add a cube with physics and collision properties",
                "on_clicked_fn": self._add_physics_cube,
            }
            self.task_ui_elements["Add Physics Cube"] = btn_builder(**dict)

            dict = {
                "label": "Add Physics Properties",
                "type": "button",
                "text": "Add Physics Properties",
                "tooltip": "Append physics properties to existing objects",
                "on_clicked_fn": self._add_physics_properties,
            }
            self.task_ui_elements["Add Physics Properties"] = btn_builder(**dict)
            self.task_ui_elements["Add Physics Properties"].enabled = False

            dict = {
                "label": "Add Collision Properties",
                "type": "button",
                "text": "Add Collision Properties",
                "tooltip": "Append collision properties to existing objects",
                "on_clicked_fn": self._add_collision_properties,
            }
            self.task_ui_elements["Add Collision Properties"] = btn_builder(**dict)
            self.task_ui_elements["Add Collision Properties"].enabled = False

    def _add_visual_cube(self):
        from isaacsim.core.api.objects import VisualCuboid

        VisualCuboid(
            prim_path="/visual_cube",
            name="visual_cube",
            position=np.array([0, 0.5, 1.0]),
            size=0.3,
            color=np.array([255, 255, 0]),
        )

        VisualCuboid(
            prim_path="/visual_cube_static",
            name="visual_cube_static",
            position=np.array([0.5, 0, 0.5]),
            size=0.3,
            color=np.array([0, 255, 0]),
        )

        self.task_ui_elements["Add Visual Cube"].enabled = False
        # enable the add physics properties button
        self.task_ui_elements["Add Physics Properties"].enabled = True

    def _add_physics_cube(self):
        from isaacsim.core.api.objects import DynamicCuboid

        DynamicCuboid(
            prim_path="/dynamic_cube",
            name="dynamic_cube",
            position=np.array([0, -0.5, 1.5]),
            size=0.3,
            color=np.array([0, 255, 255]),
        )

        self.task_ui_elements["Add Physics Cube"].enabled = False

    def _add_ground_plane(self):
        from isaacsim.core.api.objects.ground_plane import GroundPlane

        GroundPlane(prim_path="/World/GroundPlane", z_position=0)
        self.task_ui_elements["Add Ground Plane"].enabled = False

    def _add_light_source(self):

        import omni.usd
        from pxr import Sdf, UsdLux

        stage = omni.usd.get_context().get_stage()
        distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
        distantLight.CreateIntensityAttr(300)
        self.task_ui_elements["Add Light Source"].enabled = False

    def _add_physics_properties(self):
        # add physics properties to existing object
        from isaacsim.core.prims import RigidPrim

        RigidPrim("/visual_cube")
        self.task_ui_elements["Add Collision Properties"].enabled = True
        self.task_ui_elements["Add Physics Properties"].enabled = False

    def _add_collision_properties(self):
        from isaacsim.core.prims import GeometryPrim

        prim = GeometryPrim("/visual_cube")
        prim.apply_collision_apis()

        self.task_ui_elements["Add Collision Properties"].enabled = False
