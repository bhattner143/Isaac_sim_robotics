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

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka

my_world = World(stage_units_in_meters=1.0)


# TODO: this should be converted to a test, for now this is not working, we need to verify if force control works.
class FrankaTask(BaseTask):
    def __init__(self):
        BaseTask.__init__(self, name="dummy_task", offset=None)
        self._my_franka = None
        self._pd_gains = None

    def set_up_scene(self, scene):
        BaseTask.set_up_scene(self, scene)
        scene.add_default_ground_plane()
        self._my_franka = scene.add(
            Franka(
                prim_path="/World/Franka",
                name="my_franka",
                gripper_dof_names=["panda_finger_joint1", "panda_finger_joint2"],
                end_effector_prim_name="panda_rightfinger",
                gripper_open_position=np.array([0.4, 0.4]) / 0.01,
                gripper_closed_position=np.array([0.0, 0.0]),
            )
        )
        return

    def get_observations(self):
        joints_state = self.scene.get_object("my_franka").get_joints_state()
        return {
            "franka": {
                "joint_positions": np.array(joints_state.positions),
                "joint_velcoities": np.array(joints_state.velocities),
            }
        }

    def post_reset(self):
        self._pd_gains = self._my_franka.get_articulation_controller().get_gains()
        self._my_franka.get_articulation_controller().switch_control_mode("effort")
        return


class PDController(BaseController):
    def __init__(self, name, kp, kd):
        super().__init__(name)
        self._kp = kp
        self._kd = kd
        return

    def forward(self, observations):
        position_error = observations["franka"]["target_joint_positions"] - observations["franka"]["joint_positions"]
        velocity_error = -observations["franka"]["joint_velcoities"]
        joint_efforts = self._kp * position_error + self._kd * velocity_error
        return ArticulationAction(joint_efforts=joint_efforts / 100.0)


my_task = FrankaTask()
my_world.add_task(my_task)
my_world.reset()
my_franka = my_world.scene.get_object("my_franka")
my_controller = PDController(name="generic_pd_controller", kp=my_task._pd_gains[0], kd=my_task._pd_gains[1])
articulation_controller = my_franka.get_articulation_controller()

reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            my_controller.reset()
            reset_needed = False
        observations = my_world.get_observations()
        target_joint_positions = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
        observations["franka"]["target_joint_positions"] = target_joint_positions
        actions = my_controller.forward(observations)
        articulation_controller.apply_action(actions)

simulation_app.close()
