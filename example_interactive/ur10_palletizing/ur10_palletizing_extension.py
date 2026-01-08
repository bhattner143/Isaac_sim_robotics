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
from isaacsim.cortex.framework.cortex_world import CortexWorld
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from isaacsim.examples.interactive.ur10_palletizing.ur10_palletizing import BinStacking
from isaacsim.gui.components.ui_utils import btn_builder, cb_builder, get_style, str_builder


class BinStackingExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):

        self.example_name = "UR10 Palletizing"
        self.category = "Cortex"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "UR10 Palletizing",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/cortex_tutorials/tutorial_cortex_5_ur10_bin_stacking.html",
            "overview": "This Example shows how to do Palletizing using UR10 robot and Cortex behaviors in Isaac Sim.\n\nPress the 'Open in IDE' button to view the source code.",
        }

        ui_handle = BinStackingUI(**ui_kwargs)
        ui_handle.sample = BinStacking(ui_handle.on_diagnostics)

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


class BinStackingUI(BaseSampleUITemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision_stack = ""

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

            with ui.CollapsableFrame(
                title="Diagnostic",
                width=ui.Fraction(0.33),
                height=0,
                visible=True,
                collapsed=False,
                # style=get_style(),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):

                self.build_diagnostic_ui()

    def on_diagnostics(self, diagnostic, decision_stack):
        if self.decision_stack != decision_stack:
            self.decision_stack = decision_stack
            if decision_stack:
                decision_stack = "\n".join(
                    [
                        "{0}{1}".format("  " * (i + 1) if i > 0 else "", element)
                        for i, element in enumerate(decision_stack.replace("]", "").split("["))
                    ]
                )
            self.state_model.set_value(decision_stack)
        if diagnostic.bin_name:
            self.selected_bin.set_value(str(diagnostic.bin_name))
            self.bin_base.set_value(str(diagnostic.bin_base.prim_path))
            self.grasp_reached.set_value((diagnostic.grasp_reached))
            self.is_attached.set_value((diagnostic.attached))
            self.needs_flip.set_value((diagnostic.needs_flip))
        else:
            self.selected_bin.set_value(str("No Bin Selected"))
            self.bin_base.set_value("")
            self.grasp_reached.set_value(False)
            self.is_attached.set_value(False)
            self.needs_flip.set_value(False)

    def get_world(self):
        return CortexWorld.instance()

    def _on_start_button_event(self):
        asyncio.ensure_future(self.sample.on_event_async())
        self.task_ui_elements["Start Palletizing"].enabled = False
        return

    def post_reset_button_event(self):
        self.task_ui_elements["Start Palletizing"].enabled = True
        return

    def post_load_button_event(self):
        self.task_ui_elements["Start Palletizing"].enabled = True
        return

    def post_clear_button_event(self):
        self.task_ui_elements["Start Palletizing"].enabled = False
        return

    def build_task_controls_ui(self):
        with ui.VStack(spacing=5):

            dict = {
                "label": "Start Palletizing",
                "type": "button",
                "text": "Start Palletizing",
                "tooltip": "Start Palletizing",
                "on_clicked_fn": self._on_start_button_event,
            }

            self.task_ui_elements["Start Palletizing"] = btn_builder(**dict)
            self.task_ui_elements["Start Palletizing"].enabled = False

    def build_diagnostic_ui(self):
        with ui.VStack(spacing=5):
            ui.Label("Decision Stack", height=20)
            self.state_model = ui.SimpleStringModel()
            ui.StringField(self.state_model, multiline=True, height=120)
            self.selected_bin = str_builder("Selected Bin", "<No Bin Selected>", read_only=True)
            self.bin_base = str_builder("Bin Base", "", read_only=True)
            self.grasp_reached = cb_builder("Grasp Reached", False)
            self.is_attached = cb_builder("Is Attached", False)
            self.needs_flip = cb_builder("Needs Flip", False)
