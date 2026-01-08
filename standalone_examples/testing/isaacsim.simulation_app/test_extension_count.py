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

from isaacsim import SimulationApp

simulation_app = SimulationApp()

simulation_app.update()
simulation_app.update()

from typing import Tuple

import omni
import omni.ext
import omni.kit.app

app = omni.kit.app.get_app()
ext_manager = app.get_extension_manager()
ext_summaries = ext_manager.get_extensions()


def get_bundled_exts() -> dict[str, Tuple]:
    local_exts: dict[str, Tuple] = {}

    for ext_summary in ext_summaries:
        ext_name = ext_summary["name"]
        ext_enabled = bool(ext_summary["enabled"])
        if not ext_enabled:
            continue

        local_exts[ext_name] = (ext_name, ext_enabled)
    return local_exts


bundled_exts = get_bundled_exts()
print(f"Enabled extensions count: {len(bundled_exts)}")

# Cleanup application
simulation_app.close()
