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

import omni.kit.app
import omni.usd
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.examples.interactive.base_sample import BaseSample


class GettingStarted(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        return

    @property
    def name(self):
        return "Getting Started"

    def setup_scene(self):
        pass

    async def setup_post_load(self):
        set_camera_view(eye=[5.0, 2.0, 2.5], target=[0.00, 0.00, 0.00], camera_prim_path="/OmniverseKit_Persp")

        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):

        return
