# SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import os
import weakref

import carb

# Import extension python module we are testing with absolute import path, as if we are external user (other extension)
import isaacsim.robot.surface_gripper._surface_gripper as surface_gripper
import numpy as np
import omni
import omni.ext
import omni.kit.app
import omni.kit.commands
import omni.kit.usd
import omni.physics.tensors as physics
import omni.physx as _physx
import omni.ui as ui
import usd.schema.isaac.robot_schema as robot_schema
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.gui.components.menu import make_menu_item_description
from isaacsim.gui.components.ui_utils import (
    add_separator,
    btn_builder,
    combo_floatfield_slider_builder,
    get_style,
    setup_ui_headers,
    state_btn_builder,
)
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items
from omni.kit.window.property.templates import LABEL_HEIGHT, LABEL_WIDTH
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

EXTENSION_NAME = "Surface Gripper"


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        """Initialize extension and UI elements"""

        self._ext_id = ext_id

        # Loads interfaces
        self._timeline = omni.timeline.get_timeline_interface()
        self._usd_context = omni.usd.get_context()
        self._stage_event_sub = None
        self._window = None
        self._models = {}

        # register the example with examples browser
        get_browser_instance().register_example(
            name=EXTENSION_NAME, execute_entrypoint=self.build_window, ui_hook=self.build_ui, category="Manipulation"
        )

        self.surface_gripper = None
        self._stage_id = -1

    def build_window(self):
        pass

    def build_ui(self):
        self._usd_context = omni.usd.get_context()
        if self._usd_context is not None:
            self._stage_event_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
                event_name=omni.kit.app.GLOBAL_EVENT_UPDATE,
                on_event=self._on_update_ui,
                observer_name="isaacsim.examples.interactive.surface_gripper.Extension._on_update_ui",
            )

        with ui.VStack(spacing=5, height=0):
            title = "Surface Gripper Example"
            doc_link = "https://docs.isaacsim.omniverse.nvidia.com/latest/robot_simulation/ext_isaacsim_robot_surface_gripper.html"

            overview = "This Example shows how to simulate a suction-cup gripper in Isaac Sim. \n"
            overview += "It simulates suction by creating a Joint between two bodies when the parent and child bodies are close at the gripper's point of contact.\n"
            overview += "The gripper is defined as a USD prim and its behavior is managed by the SurfaceGripperManager. You can interact with it by calling the manager to Open/Close, and get its status. \n"
            overview += "Additional information can be directly extracted from the USD prim.\n"
            overview += "\n\nPress the 'Open in IDE' button to view the source code."

            setup_ui_headers(self._ext_id, __file__, title, doc_link, overview, info_collapsed=False)

            frame = ui.CollapsableFrame(
                title="Command Panel",
                height=0,
                collapsed=False,
                style=get_style(),
                style_type_name_override="CollapsableFrame",
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            )
            with frame:
                with ui.VStack(style=get_style(), spacing=5):
                    with ui.Frame(height=LABEL_HEIGHT):
                        args = {
                            "label": "Load Scene",
                            "type": "button",
                            "text": "Load",
                            "tooltip": "Load a gripper into the Scene",
                            "on_clicked_fn": self._on_create_scenario_button_clicked,
                        }
                        self._models["create_button"] = btn_builder(**args)
                    with ui.Frame(height=LABEL_HEIGHT):
                        args = {
                            "label": "Gripper State",
                            "type": "button",
                            "a_text": "Close",
                            "b_text": "Open",
                            "tooltip": "Open and Close the Gripper",
                            "on_clicked_fn": self._on_toggle_gripper_button_clicked,
                        }
                        self._models["toggle_button"] = state_btn_builder(**args)
                    with ui.HStack():
                        ui.Label("Gripped Objects", width=LABEL_WIDTH, height=LABEL_HEIGHT * 5)
                        self._models["gripped_objects"] = ui.SimpleStringModel()
                        ui.StringField(self._models["gripped_objects"])
                    add_separator()

    def on_shutdown(self):
        self._physx_subs = None
        self._stage_event_sub = None
        self._window = None
        get_browser_instance().deregister_example(name=EXTENSION_NAME, category="Manipulation")

    def _on_update_ui(self, widget):
        self._models["create_button"].enabled = self._timeline.is_playing()
        self._models["toggle_button"].enabled = self._timeline.is_playing()
        # If the scene has been reloaded, reset UI to create Scenario
        if self._usd_context.get_stage_id() != self._stage_id:
            self._models["create_button"].enabled = True
            # self._models["create_button"].text = "Create Scenario"
            self._models["create_button"].set_tooltip(
                "Creates a new scenario with a gantry containing a surface gripper and some cubes to pick up."
            )
            self._models["create_button"].set_clicked_fn(self._on_create_scenario_button_clicked)
            self._stage_id = -1

    def _toggle_gripper_button_ui(self):
        # Checks if the surface gripper has been created
        status = self.gripper_interface.get_gripper_status(self.gripper_prim_path)
        if status == "Open":
            self._models["toggle_button"].text = "OPEN"
        else:
            self._models["toggle_button"].text = "CLOSED"

    def _on_simulation_step(self, step):
        # Checks if the simulation is playing, and if the stage has been loaded
        if self._timeline.is_playing() and self._stage_id != -1:
            self._toggle_gripper_button_ui()
            surface_gripper = self._stage.GetPrimAtPath(self.gripper_prim_path)
            gripped_objs_rel = surface_gripper.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name)
            gripped_objects = [f"{a}" for a in gripped_objs_rel.GetTargets()]
            self._models["gripped_objects"].set_value("\n".join(gripped_objects))

    def _on_reset_scenario_button_clicked(self):
        pass

    async def _create_scenario(self, task):
        done, pending = await asyncio.wait({task})
        if task in done:
            # Repurpose button to reset Scene
            self._models["create_button"].text = "Reset Scene"
            self._models["create_button"].set_tooltip("Resets scenario")

            # Get Handle for stage and stage ID to check if stage was reloaded
            self._stage = self._usd_context.get_stage()
            self._stage_id = self._usd_context.get_stage_id()
            self._timeline.stop()
            self._models["create_button"].set_clicked_fn(self._on_reset_scenario_button_clicked)

            self.gripper_prim_path = "/World/SurfaceGripper"
            self.gripper_interface = surface_gripper.acquire_surface_gripper_interface()

            # Create the Surface Gripper Prim
            # This prim could be already defined in the stage,
            # but creating it in code instead to demonstrate how to do it.
            # Once it is created it can be saved and this doesn't need to be redone
            robot_schema.CreateSurfaceGripper(self._stage, self.gripper_prim_path)
            gripper_prim = self._stage.GetPrimAtPath(self.gripper_prim_path)
            attachment_points_rel = gripper_prim.GetRelationship(robot_schema.Relations.ATTACHMENT_POINTS.name)

            # Select the joints to the gripper
            # The joints should be D6 joints defined in the usd file.
            # All joint attributes can be defined as desired, except for:
            # Joint Should be enabled
            # Joint Type should be D6
            # All Joint Parents should be the same Rigid body
            # Exclude from Articulation must be checked
            # No Break force/Torque should be set
            # Joint drives can be used to derive the desired joint bounce/stretch behavior
            # Enable/Disable the joint DoFs and limits as desired.

            gripper_joints = [
                p.GetPath() for p in self._stage.GetPrimAtPath("/World/Surface_Gripper_Joints").GetChildren()
            ]
            attachment_points_rel.SetTargets(gripper_joints)

            # Define the distance the joint can grasp, and at what distance from the origin of the joints it will settle
            gripper_prim.GetAttribute(robot_schema.Attributes.MAX_GRIP_DISTANCE.name).Set(0.011)
            # Define the Override Break limits
            gripper_prim.GetAttribute(robot_schema.Attributes.COAXIAL_FORCE_LIMIT.name).Set(0.005)
            gripper_prim.GetAttribute(robot_schema.Attributes.SHEAR_FORCE_LIMIT.name).Set(5)

            # How long the gripper will try to close if it is open
            gripper_prim.GetAttribute(robot_schema.Attributes.RETRY_INTERVAL.name).Set(1.0)

            # Select the gripper on the stage, and set the camera view to look at the machine
            selection = omni.usd.get_context().get_selection()
            selection.set_selected_prim_paths([self.gripper_prim_path], False)

            self.gripper_start_pose = physics.Transform([0, 0, 1.301], [0, 0, 0, 1])
            set_camera_view(
                eye=[2.00, 2.00, 2.00], target=self.gripper_start_pose.p, camera_prim_path="/OmniverseKit_Persp"
            )

            self._physx_subs = _physx.get_physx_interface().subscribe_physics_step_events(self._on_simulation_step)
            self._timeline.play()

    def _on_create_scenario_button_clicked(self):
        # wait for new stage before creating scenario
        # Load the gantry USD scene
        async def load_gantry_scene():
            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_path = ext_manager.get_extension_path(self._ext_id)
            usd_path = os.path.join(ext_path, "data", "SurfaceGripper_gantry.usda")
            await omni.usd.get_context().new_stage_async()
            stage = omni.usd.get_context().get_stage()
            # root_layer = stage.GetRootLayer()
            # root_layer.subLayerPaths.append(usd_path)
            stage.DefinePrim("/World", "Xform").GetReferences().AddReference(usd_path)
            stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

        task = asyncio.ensure_future(load_gantry_scene())
        asyncio.ensure_future(self._create_scenario(task))

    def _on_toggle_gripper_button_clicked(self, val=False):
        if self._timeline.is_playing():
            status = self.gripper_interface.get_gripper_status(self.gripper_prim_path)
            if status == "Open":
                self.gripper_interface.close_gripper(self.gripper_prim_path)
            else:
                self.gripper_interface.open_gripper(self.gripper_prim_path)
