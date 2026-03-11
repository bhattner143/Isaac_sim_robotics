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

import time

from isaacsim import SimulationApp

# Example ROS bridge sample showing rospy and rosclock interaction
kit = SimulationApp()
import omni
from isaacsim.core.utils.extensions import enable_extension

# enable ROS bridge extension
enable_extension("isaacsim.ros2.bridge")
kit.update()
kit.close()
