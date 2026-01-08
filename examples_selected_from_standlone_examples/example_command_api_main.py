# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

simulation_app = SimulationApp({"headless": False})

import time

import numpy as np
import omni
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.cortex.framework.cortex_world import CortexWorld
from isaacsim.cortex.framework.df import DfNetwork, DfState, DfStateMachineDecider, DfStateSequence
from isaacsim.cortex.framework.dfb import DfBasicContext
from isaacsim.cortex.framework.robot import add_franka_to_stage


class NullspaceShiftState(DfState):
    def __init__(self):
        super().__init__()
        self.config_mean = np.array([0.00, -1.3, 0.00, -2.87, 0.00, 2.00, 0.75])
        self.target_p = np.array([0.7, 0.0, 0.5])
        self.construction_time = time.time()

    def enter(self):
        # Change the posture configuration while maintaining a consistent target.
        posture_config = self.config_mean + np.random.randn(7)
        self.context.robot.arm.send_end_effector(target_position=self.target_p, posture_config=posture_config)

        self.entry_time = time.time()

        # Close the gripper if open and open the gripper if closed. It closes more quickly than it
        # opens.
        gripper = self.context.robot.gripper
        if gripper.get_width() > 0.05:
            gripper.close(speed=0.5)
        else:
            gripper.open(speed=0.1)

        print("[%f] <enter> sampling posture config" % (self.entry_time - self.construction_time))

    def step(self):
        if time.time() - self.entry_time < 2.0:
            return self
        return None


def main():
    world = CortexWorld()
    robot = world.add_robot(add_franka_to_stage(name="franka", prim_path="/World/franka"))
    world.scene.add_default_ground_plane()

    decider_network = DfNetwork(
        DfStateMachineDecider(DfStateSequence([NullspaceShiftState()], loop=True)), context=DfBasicContext(robot)
    )
    world.add_decider_network(decider_network)

    world.run(simulation_app)
    simulation_app.close()


if __name__ == "__main__":
    main()
