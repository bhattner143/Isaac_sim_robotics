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

import omni.ext
import omni.ui as ui
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from isaacsim.examples.interactive.robo_factory import RoboFactory
from isaacsim.gui.components.ui_utils import btn_builder


class RoboFactoryExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "RoboFactory"
        self.category = "Multi-Robot"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "RoboFactory",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/core_api_tutorials/tutorial_core_adding_multiple_robots.html",
            "overview": "This Example shows how to run multiple tasks in the same scene.\n\nPress 'LOAD' to load the scene, \npress 'START STACKING' to start stacking.\n\nPress the 'Open in IDE' button to view the source code.",
            "sample": RoboFactory(),
        }

        ui_handle = RoboFactoryUI(**ui_kwargs)

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


class RoboFactoryUI(BaseSampleUITemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_extra_frames(self):
        extra_stacks = self.get_extra_frames_handle()
        self.task_ui_elements = {}

        with extra_stacks:
            with ui.CollapsableFrame(
                title="Task Control",
                width=ui.Fraction(0.33),
                height=0,
                visible=True,
                collapsed=False,
                # style=get_style(),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                self.build_task_controls_ui()

    def _on_start_stacking_button_event(self):
        asyncio.ensure_future(self.sample._on_start_stacking_event_async())
        self.task_ui_elements["Start Stacking"].enabled = False
        return

    def post_reset_button_event(self):
        self.task_ui_elements["Start Stacking"].enabled = True
        return

    def post_load_button_event(self):
        self.task_ui_elements["Start Stacking"].enabled = True
        return

    def post_clear_button_event(self):
        self.task_ui_elements["Start Stacking"].enabled = False
        return

    def build_task_controls_ui(self):
        with ui.VStack(spacing=5):

            dict = {
                "label": "Start Stacking",
                "type": "button",
                "text": "Start Stacking",
                "tooltip": "Start Stacking",
                "on_clicked_fn": self._on_start_stacking_button_event,
            }

            self.task_ui_elements["Start Stacking"] = btn_builder(**dict)
            self.task_ui_elements["Start Stacking"].enabled = False
