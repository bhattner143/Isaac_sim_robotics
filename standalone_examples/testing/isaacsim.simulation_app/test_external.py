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
import sys

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp()

import omni
from isaacsim.core.utils.extensions import disable_extension, enable_extension

simulation_app.update()

enable_extension("semantics.schema.editor")
simulation_app.update()
disable_extension("semantics.schema.editor")
simulation_app.update()
enable_extension("omni.cuopt.examples")
simulation_app.update()
disable_extension("omni.cuopt.examples")
simulation_app.update()
enable_extension("omni.anim.people")
simulation_app.update()
disable_extension("omni.anim.people")
simulation_app.update()
# Cleanup application
simulation_app.close()
