# SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from abc import abstractmethod

import omni.kit.app
import omni.ui as ui
from isaacsim.core.api import World
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.gui.components.ui_utils import btn_builder, get_style, setup_ui_headers


class BaseSampleUITemplate:
    def __init__(self, *args, **kwargs):
        self._ext_id = kwargs.get("ext_id")
        self._file_path = kwargs.get("file_path", "")
        self._title = kwargs.get("title", "Isaac Sim Example")
        self._doc_link = kwargs.get("doc_link", "")
        self._overview = kwargs.get("overview", "")
        self._sample = kwargs.get("sample", BaseSample())

        self._buttons = dict()
        self.extra_stacks = None

    @property
    def sample(self):
        return self._sample

    @sample.setter
    def sample(self, sample):
        self._sample = sample

    def get_world(self):
        return World.instance()

    def build_window(self):
        # separating out building the window and building the UI, so that example browser can build_ui but not the window
        # self._window = omni.ui.Window(
        #     self.example_name, width=350, height=0, visible=True, dockPreference=ui.DockPreference.LEFT_BOTTOM
        # )
        # with self._window.frame:
        #     self.build_ui()
        # return self._window
        pass

    def build_ui(self):
        # separating out building default frame and extra frames, so examples can override the extra frames function

        self.build_default_frame()
        self.build_extra_frames()
        return

    def build_default_frame(self):
        self._main_stack = ui.VStack(spacing=5, height=0)
        with self._main_stack:
            setup_ui_headers(
                self._ext_id, self._file_path, self._title, self._doc_link, self._overview, info_collapsed=False
            )
            self._controls_frame = ui.CollapsableFrame(
                title="World Controls",
                width=ui.Fraction(1),
                height=0,
                collapsed=False,
                style=get_style(),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            )
            self.extra_stacks = ui.VStack(margin=5, spacing=5, height=0)

        with self._controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                dict = {
                    "label": "Load World",
                    "type": "button",
                    "text": "Load",
                    "tooltip": "Load World and Task",
                    "on_clicked_fn": self._on_load_world,
                }
                self._buttons["Load World"] = btn_builder(**dict)
                self._buttons["Load World"].enabled = True
                dict = {
                    "label": "Reset",
                    "type": "button",
                    "text": "Reset",
                    "tooltip": "Reset robot and environment",
                    "on_clicked_fn": self._on_reset,
                }
                self._buttons["Reset"] = btn_builder(**dict)
                self._buttons["Reset"].enabled = False

        return

    def get_extra_frames_handle(self):
        return self.extra_stacks

    @abstractmethod
    def build_extra_frames(self):
        return

    def _on_load_world(self):
        async def _on_load_world_async():
            await self._sample.load_world_async()
            await omni.kit.app.get_app().next_update_async()
            self._sample._world.add_stage_callback("stage_event_1", self.on_stage_event)
            self._enable_all_buttons(True)
            self._buttons["Load World"].enabled = False
            self.post_load_button_event()
            self._sample._world.add_timeline_callback("stop_reset_event", self._reset_on_stop_event)

        asyncio.ensure_future(_on_load_world_async())
        return

    def _on_reset(self):
        async def _on_reset_async():
            await self._sample.reset_async()
            await omni.kit.app.get_app().next_update_async()
            self.post_reset_button_event()

        asyncio.ensure_future(_on_reset_async())
        return

    @abstractmethod
    def post_reset_button_event(self):
        return

    @abstractmethod
    def post_load_button_event(self):
        return

    @abstractmethod
    def post_clear_button_event(self):
        return

    def _enable_all_buttons(self, flag):
        for btn_name, btn in self._buttons.items():
            if isinstance(btn, omni.ui._ui.Button):
                btn.enabled = flag
        return

    def on_shutdown(self):

        self.extra_stacks = None
        self._buttons = {}
        self._sample = None
        return

    def on_stage_event(self, event):
        if event.type == int(omni.usd.StageEventType.CLOSED):
            if World.instance() is not None:
                self._sample._world_cleanup()
                self._sample._world.clear_instance()
                if hasattr(self, "_buttons"):
                    if self._buttons is not None:
                        self._enable_all_buttons(False)
                        self._buttons["Load World"].enabled = True
        return

    def _reset_on_stop_event(self, e):
        if e.type == int(omni.timeline.TimelineEventType.STOP):
            world = World.instance()
            if world is not None:
                world.clear_physics_callbacks()

            self._buttons["Load World"].enabled = False
            self._buttons["Reset"].enabled = True
            self.post_clear_button_event()

        return
