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

from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
from isaacsim.core.api.objects import FixedCuboid, VisualCuboid
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.robot.manipulators.examples.franka import Franka


class PathPlanningTask(BaseTask):
    def __init__(
        self,
        name: str,
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:

        BaseTask.__init__(self, name=name, offset=offset)
        self._robot = None
        self._target_name = target_name
        self._target = None
        self._target_prim_path = target_prim_path
        self._target_position = target_position
        self._target_orientation = target_orientation
        self._target_visual_material = None
        self._obstacle_walls = OrderedDict()
        if self._target_position is None:
            self._target_position = np.array([0.65, 0.3, 0.4]) / get_stage_units()
        return

    def set_up_scene(self, scene: Scene) -> None:
        """[summary]

        Args:
            scene (Scene): [description]
        """
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        if self._target_orientation is None:
            self._target_orientation = euler_angles_to_quat(np.array([-np.pi, 0, np.pi]))
        if self._target_prim_path is None:
            self._target_prim_path = find_unique_string_name(
                initial_name="/World/TargetCube", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
        if self._target_name is None:
            self._target_name = find_unique_string_name(
                initial_name="target", is_unique_fn=lambda x: not self.scene.object_exists(x)
            )
        self.set_params(
            target_prim_path=self._target_prim_path,
            target_position=self._target_position,
            target_orientation=self._target_orientation,
            target_name=self._target_name,
        )
        self._robot = self.set_robot()
        scene.add(self._robot)
        self._task_objects[self._robot.name] = self._robot
        self._move_task_objects_to_their_frame()
        return

    def set_params(
        self,
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
    ) -> None:
        """[summary]

        Args:
            target_prim_path (Optional[str], optional): [description]. Defaults to None.
            target_name (Optional[str], optional): [description]. Defaults to None.
            target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            target_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        """
        if target_prim_path is not None:
            if self._target is not None:
                del self._task_objects[self._target.name]
            if is_prim_path_valid(target_prim_path):
                self._target = self.scene.add(
                    SingleXFormPrim(
                        prim_path=target_prim_path,
                        position=target_position,
                        orientation=target_orientation,
                        name=target_name,
                    )
                )
            else:
                self._target = self.scene.add(
                    VisualCuboid(
                        name=target_name,
                        prim_path=target_prim_path,
                        position=target_position,
                        orientation=target_orientation,
                        color=np.array([1, 0, 0]),
                        size=1.0,
                        scale=np.array([0.03, 0.03, 0.03]) / get_stage_units(),
                    )
                )
            self._task_objects[self._target.name] = self._target
            self._target_visual_material = self._target.get_applied_visual_material()
            if self._target_visual_material is not None:
                if hasattr(self._target_visual_material, "set_color"):
                    self._target_visual_material.set_color(np.array([1, 0, 0]))
        else:
            self._target.set_local_pose(position=target_position, orientation=target_orientation)
        return

    def get_params(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        params_representation = dict()
        params_representation["target_prim_path"] = {"value": self._target.prim_path, "modifiable": True}
        params_representation["target_name"] = {"value": self._target.name, "modifiable": True}
        position, orientation = self._target.get_local_pose()
        params_representation["target_position"] = {"value": position, "modifiable": True}
        params_representation["target_orientation"] = {"value": orientation, "modifiable": True}
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation

    def get_task_objects(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        return self._task_objects

    def get_observations(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        joints_state = self._robot.get_joints_state()
        target_position, target_orientation = self._target.get_local_pose()
        return {
            self._robot.name: {
                "joint_positions": np.array(joints_state.positions),
                "joint_velocities": np.array(joints_state.velocities),
            },
            self._target.name: {"position": np.array(target_position), "orientation": np.array(target_orientation)},
        }

    def target_reached(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        end_effector_position, _ = self._robot.end_effector.get_world_pose()
        target_position, _ = self._target.get_world_pose()
        if np.mean(np.abs(np.array(end_effector_position) - np.array(target_position))) < (0.035 / get_stage_units()):
            return True
        else:
            return False

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """[summary]

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """
        if self._target_visual_material is not None:
            if hasattr(self._target_visual_material, "set_color"):
                if self.target_reached():
                    self._target_visual_material.set_color(color=np.array([0, 1.0, 0]))
                else:
                    self._target_visual_material.set_color(color=np.array([1.0, 0, 0]))

        return

    def add_obstacle(self, position: np.ndarray = None, orientation=None):
        """[summary]

        Args:
            position (np.ndarray, optional): [description]. Defaults to np.array([0.1, 0.1, 1.0]).
        """
        # TODO: move to task frame if there is one
        cube_prim_path = find_unique_string_name(
            initial_name="/World/WallObstacle", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        cube_name = find_unique_string_name(initial_name="wall", is_unique_fn=lambda x: not self.scene.object_exists(x))
        if position is None:
            position = np.array([0.6, 0.1, 0.2]) / get_stage_units()
        if orientation is None:
            orientation = euler_angles_to_quat(np.array([0, 0, np.pi / 3]))
        cube = self.scene.add(
            VisualCuboid(
                name=cube_name,
                position=position + self._offset,
                orientation=orientation,
                prim_path=cube_prim_path,
                size=1.0,
                scale=np.array([0.1, 0.5, 0.6]) / get_stage_units(),
                color=np.array([0, 0, 1.0]),
            )
        )
        self._obstacle_walls[cube.name] = cube
        return cube

    def remove_obstacle(self, name: Optional[str] = None) -> None:
        """[summary]

        Args:
            name (Optional[str], optional): [description]. Defaults to None.
        """
        if name is not None:
            self.scene.remove_object(name)
            del self._obstacle_walls[name]
        else:
            obstacle_to_delete = list(self._obstacle_walls.keys())[-1]
            self.scene.remove_object(obstacle_to_delete)
            del self._obstacle_walls[obstacle_to_delete]
        return

    def get_obstacles(self) -> List:
        return list(self._obstacle_walls.values())

    def get_obstacle_to_delete(self) -> None:
        """[summary]

        Returns:
            [type]: [description]
        """
        obstacle_to_delete = list(self._obstacle_walls.keys())[-1]
        return self.scene.get_object(obstacle_to_delete)

    def obstacles_exist(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        if len(self._obstacle_walls) > 0:
            return True
        else:
            return False

    def cleanup(self) -> None:
        """[summary]"""
        obstacles_to_delete = list(self._obstacle_walls.keys())
        for obstacle_to_delete in obstacles_to_delete:
            self.scene.remove_object(obstacle_to_delete)
            del self._obstacle_walls[obstacle_to_delete]
        return

    def get_custom_gains(self) -> Tuple[np.array, np.array]:
        return None, None


class FrankaPathPlanningTask(PathPlanningTask):
    def __init__(
        self,
        name: str,
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        franka_prim_path: Optional[str] = None,
        franka_robot_name: Optional[str] = None,
    ) -> None:
        PathPlanningTask.__init__(
            self,
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            target_position=target_position,
            target_orientation=target_orientation,
            offset=offset,
        )
        self._franka_prim_path = franka_prim_path
        self._franka_robot_name = franka_robot_name
        self._franka = None
        return

    def set_robot(self) -> Franka:
        """[summary]

        Returns:
            Franka: [description]
        """
        if self._franka_prim_path is None:
            self._franka_prim_path = find_unique_string_name(
                initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
        if self._franka_robot_name is None:
            self._franka_robot_name = find_unique_string_name(
                initial_name="my_franka", is_unique_fn=lambda x: not self.scene.object_exists(x)
            )
        self._franka = Franka(prim_path=self._franka_prim_path, name=self._franka_robot_name)
        return self._franka

    def get_custom_gains(self) -> Tuple[np.array, np.array]:
        return (1e15 * np.ones(9), 1e13 * np.ones(9))
