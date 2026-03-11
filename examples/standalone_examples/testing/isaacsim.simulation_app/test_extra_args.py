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

from isaacsim import SimulationApp

# The most basic usage for creating a simulation app
kit = SimulationApp({"extra_args": ["--/app/extra/arg=1", "--/app/some/other/arg=2"]})

import carb

kit.update()

server_check = carb.settings.get_settings().get_as_string("/persistent/isaac/asset_root/default")

if server_check != "omniverse://ov-test-this-is-working":
    raise ValueError(f"isaac nucleus default setting not omniverse://ov-test-this-is-working, instead: {server_check}")

arg_1 = carb.settings.get_settings().get_as_int("/app/extra/arg")
arg_2 = carb.settings.get_settings().get_as_int("/app/some/other/arg")

if arg_1 != 1:
    raise ValueError(f"/app/extra/arg was not 1 and was {arg_1} instead")

if arg_2 != 2:
    raise ValueError(f"/app/some/other/arg was not 2 and was {arg_2} instead")

kit.close()  # Cleanup application
