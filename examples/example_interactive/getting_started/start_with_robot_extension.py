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

import numpy as np
import omni.ext
import omni.timeline
import omni.ui as ui
import omni.usd
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from isaacsim.examples.interactive.getting_started.start_with_robot import GettingStartedRobot
from isaacsim.gui.components.ui_utils import btn_builder


class GettingStartedRobotExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "Part II: Robot"
        self.category = "Tutorials"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Getting Started with a Robot",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/introduction/quickstart_isaacsim_robot.html",
            "overview": "This Example follows the 'Getting Started' tutorials from the documentation\n\n 'Reset' Button is disabled. to Restart, click on the thumbnail in the browser instead.\n\n Press the 'Open in IDE' button to view the source code.",
            "sample": GettingStartedRobot(),
        }

        self.ui_handle = GettingStartedRobotUI(**ui_kwargs)

        # register the example with examples browser
        get_browser_instance().register_example(
            name=self.example_name,
            execute_entrypoint=self.ui_handle.build_window,
            ui_hook=self.ui_handle.build_ui,
            category=self.category,
        )

        return

    def on_shutdown(self):
        self.ui_handle.cleanup()
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)

        return


class GettingStartedRobotUI(BaseSampleUITemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def build_ui(self):
        """
        overwriting the build_ui function to add timeline callbacks that only registeres when the tutorial is clicked on and UI is built
        """
        self.arm_handle = None
        self.car_handle = None
        self._timeline = omni.timeline.get_timeline_interface()
        self._event_timer_callback = self._timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._timeline_timer_callback_fn
        )
        super().build_ui()

    def build_extra_frames(self):
        extra_stacks = self.get_extra_frames_handle()
        self.task_ui_elements = {}
        with extra_stacks:
            with ui.CollapsableFrame(
                title="Getting Started with a Robot",
                width=ui.Fraction(0.33),
                height=0,
                visible=True,
                collapsed=False,
                # style=get_style(),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                self.build_getting_started_ui()

    def build_getting_started_ui(self):
        with ui.VStack(spacing=5):

            dict = {
                "label": "Add A Manipulator",
                "type": "button",
                "text": "Add Arm",
                "tooltip": "Add a manipulator (Franka Panda) to scene.",
                "on_clicked_fn": self._add_arm,
            }
            self.task_ui_elements["Add Arm"] = btn_builder(**dict)
            # self.task_ui_elements["Add Arm"].enabled= False

            dict = {
                "label": "Add A Mobile Robot",
                "type": "button",
                "text": "Add Vehicle",
                "tooltip": "Add a mobile robot to scene",
                "on_clicked_fn": self._add_vehicle,
            }
            self.task_ui_elements["Add Vehicle"] = btn_builder(**dict)
            # self.task_ui_elements["Add Vehicle"].enabled = False

            dict = {
                "label": "Move Arm",
                "type": "button",
                "text": "Move Arm",
                "tooltip": "Move the manipulator",
                "on_clicked_fn": self._move_arm,
            }
            self.task_ui_elements["Move Arm"] = btn_builder(**dict)
            self.task_ui_elements["Move Arm"].enabled = False

            dict = {
                "label": "Move Vehicle",
                "type": "button",
                "text": "Move Vehicle",
                "tooltip": "Move the mobile robot",
                "on_clicked_fn": self._move_vehicle,
            }
            self.task_ui_elements["Move Vehicle"] = btn_builder(**dict)
            self.task_ui_elements["Move Vehicle"].enabled = False

            dict = {
                "label": "Print State",
                "type": "button",
                "text": "Print Joint State",
                "tooltip": "Print the state of the robot",
                "on_clicked_fn": self._print_state,
            }
            self.task_ui_elements["Print Joint State"] = btn_builder(**dict)
            self.task_ui_elements["Print Joint State"].enabled = False

    def _add_arm(self):

        import carb
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path

        if self._timeline.is_playing():
            print("Timeline is playing. Stop the timeline to add robot to avoid collision")
            self._timeline.stop()
        else:

            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            usd_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
            prim_path = "/World/Arm"

            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

            self.arm_handle = Articulation(prim_paths_expr=prim_path, name="Arm")
            self.arm_handle.set_world_poses(positions=np.array([[0, -1.0, 0]]))

            self.sample.arm_handle = self.arm_handle

            self.task_ui_elements["Move Arm"].text = "PRESS PLAY"
            self.task_ui_elements["Add Arm"].enabled = False

    def _add_vehicle(self):
        import carb
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path

        if self._timeline.is_playing():
            print("Timeline is playing. Stop the timeline to add robot to avoid collision")
            self._timeline.stop()
        else:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            usd_path = assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
            prim_path = "/World/Car"

            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

            # add vehicle to World
            self.car_handle = Articulation(prim_paths_expr=prim_path, name="Car")

            self.sample.car_handle = self.car_handle

            self.task_ui_elements["Move Vehicle"].text = "PRESS PLAY"
            self.task_ui_elements["Add Vehicle"].enabled = False

    def _move_arm(self):
        if self.task_ui_elements["Move Arm"].text.upper() == "MOVE ARM":
            # move the arm
            self.arm_handle.set_joint_positions([[-1.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5, 0.04, 0.04]])

            # toggle btn
            self.task_ui_elements["Move Arm"].text = "RESET ARM"

        elif self.task_ui_elements["Move Arm"].text.upper() == "RESET ARM":
            # stop the arm
            self.arm_handle.set_joint_positions([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            # toggle btn
            self.task_ui_elements["Move Arm"].text = "MOVE ARM"
        else:
            pass

    def _move_vehicle(self):

        if self.task_ui_elements["Move Vehicle"].text.upper() == "MOVE VEHICLE":
            # move the vehicle
            self.car_handle.set_joint_velocities([[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])
            # toggle btn
            self.task_ui_elements["Move Vehicle"].text = "STOP VEHICLE"

        elif self.task_ui_elements["Move Vehicle"].text.upper() == "STOP VEHICLE":
            # stop the vehicle
            self.car_handle.set_joint_velocities([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

            # toggle btn
            self.task_ui_elements["Move Vehicle"].text = "MOVE VEHICLE"
        else:
            pass

    def _print_state(self):
        self.sample.print_state = not self.sample.print_state  # toggle print state
        if self.sample.print_state:
            self.task_ui_elements["Print Joint State"].text = "STOP PRINTING"
        else:
            self.task_ui_elements["Print Joint State"].text = "PRINT JOINT STATE"

    def _timeline_timer_callback_fn(self, event):
        if event.type == int(omni.timeline.TimelineEventType.STOP):  # reset buttons when pressed STOP
            if self.car_handle is not None:
                self.task_ui_elements["Move Vehicle"].enabled = False
                self.task_ui_elements["Move Vehicle"].text = "PRESS PLAY"
            if self.arm_handle is not None:
                self.task_ui_elements["Move Arm"].enabled = False
                self.task_ui_elements["Move Arm"].text = "PRESS PLAY"
            self.sample.print_state = False
            self.task_ui_elements["Print Joint State"].enabled = False
            self.task_ui_elements["Print Joint State"].text = "PRESS PLAY"

        elif event.type == int(omni.timeline.TimelineEventType.PLAY):  # enable buttons when pressed PLAY
            if self.car_handle is not None:
                self.task_ui_elements["Move Vehicle"].enabled = True
                self.task_ui_elements["Move Vehicle"].text = "MOVE VEHICLE"
            if self.arm_handle is not None:
                self.task_ui_elements["Move Arm"].enabled = True
                self.task_ui_elements["Move Arm"].text = "MOVE ARM"
            if self.car_handle or self.arm_handle:
                self.task_ui_elements["Print Joint State"].enabled = True
                if self.sample.print_state:
                    self.task_ui_elements["Print Joint State"].text = "STOP PRINTING"
                else:
                    self.task_ui_elements["Print Joint State"].text = "PRINT JOINT STATE"

    def cleanup(self):
        self._event_timer_callback = None
        return
