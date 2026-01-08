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

import os

import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from isaacsim.examples.interactive.omnigraph_keyboard import OmnigraphKeyboard


class OmnigraphKeyboardExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):

        self.example_name = "Omnigraph Keyboard"
        self.category = "Input Devices"

        overview = "This Example shows how to change the size of a cube using the keyboard through omnigraph progrmaming in Isaac Sim."
        overview += "\n\tKeybord Input:"
        overview += "\n\t\ta: Grow"
        overview += "\n\t\td: Shrink"
        overview += "\n\nPress the 'Open in IDE' button to view the source code."
        overview += "\nOpen Visual Scripting Window to see Omnigraph"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Omnigraph Keyboard Example",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/introduction/examples.html",
            "overview": overview,
            "sample": OmnigraphKeyboard(),
        }

        ui_handle = BaseSampleUITemplate(**ui_kwargs)

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
