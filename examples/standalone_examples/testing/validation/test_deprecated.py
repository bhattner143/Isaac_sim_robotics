# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
kit = SimulationApp()


import argparse
import asyncio
import builtins
import datetime
import glob
import itertools
import json
import math
import os
import pprint
import random
import shutil
import signal
import struct
import sys
import time
import unittest
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain, cycle
from pathlib import Path
from pprint import pprint
from typing import List, Optional

# carb related imports
import carb
import carb.events
import carb.settings
import cv2
import matplotlib.pyplot as plt
import numpy as np

# object_based_sdg_utils related imports
import omni
import omni.appwindow
import omni.client
import omni.graph.core as og
import omni.isaac.core.tasks as tasks
import omni.isaac.core.utils.deformable_mesh_utils as deformableMeshUtils
import omni.isaac.core.utils.extensions as extensions_utils
import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.cortex.math_util as math_util
import omni.isaac.cortex.sample_behaviors.ur10.bin_stacking_behavior as behavior
import omni.isaac.IsaacSensorSchema as IsaacSensorSchema
import omni.isaac.manipulators.controllers as manipulators_controllers
import omni.isaac.motion_generation as mg
import omni.kit
import omni.kit.app
import omni.kit.asset_converter
import omni.kit.commands
import omni.kit.viewport.utility
import omni.physx
import omni.physx as _physx
import omni.replicator.core as rep
import omni.replicator.isaac as dr
import omni.syntheticdata._syntheticdata as sd
import omni.timeline
import omni.usd
import omni.usd.commands
import scipy
import scipy.io as sio
import torch
import torchvision
import usdrt
import usdrt.Sdf
import warp as wp
import yaml
from omni.isaac.cloner import Cloner, GridCloner
from omni.isaac.core import PhysicsContext, SimulationContext, World
from omni.isaac.core.articulations import Articulation, ArticulationView
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.materials.deformable_material import DeformableMaterial
from omni.isaac.core.materials.omni_glass import OmniGlass
from omni.isaac.core.materials.omni_pbr import OmniPBR
from omni.isaac.core.materials.particle_material import ParticleMaterial
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.objects import (
    DynamicCapsule,
    DynamicCone,
    DynamicCuboid,
    DynamicCylinder,
    DynamicSphere,
    FixedCuboid,
    VisualCapsule,
    VisualCuboid,
    VisualSphere,
    cuboid,
)
from omni.isaac.core.objects.sphere import VisualSphere
from omni.isaac.core.prims import GeometryPrimView, RigidPrim, RigidPrimView, XFormPrim
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.prims.geometry_prim_view import GeometryPrimView
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.prims.rigid_prim_view import RigidPrimView
from omni.isaac.core.prims.soft.cloth_prim import ClothPrim
from omni.isaac.core.prims.soft.cloth_prim_view import ClothPrimView
from omni.isaac.core.prims.soft.deformable_prim import DeformablePrim
from omni.isaac.core.prims.soft.deformable_prim_view import DeformablePrimView
from omni.isaac.core.prims.soft.particle_system import ParticleSystem
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.prims.xform_prim_view import XFormPrimView
from omni.isaac.core.robots import Robot
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils import extensions, prims, rotations, stage, viewports
from omni.isaac.core.utils.bounds import compute_combined_aabb, compute_obb, create_bbox_cache, get_obb_corners
from omni.isaac.core.utils.collisions import ray_cast
from omni.isaac.core.utils.extensions import disable_extension, enable_extension, get_extension_path_from_name
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
from omni.isaac.core.utils.prims import (
    add_reference_to_stage,
    create_prim,
    define_prim,
    delete_prim,
    get_prim_at_path,
    get_prim_attribute_value,
    is_prim_path_valid,
)
from omni.isaac.core.utils.random import get_random_world_pose_in_view
from omni.isaac.core.utils.render_product import (
    add_aov,
    create_hydra_texture,
    get_camera_prim_path,
    get_resolution,
    set_camera_prim_path,
    set_resolution,
)
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.core.utils.semantics import add_update_semantics, remove_all_semantics
from omni.isaac.core.utils.stage import (
    add_reference_to_stage,
    create_new_stage,
    get_current_stage,
    get_stage_units,
    is_stage_loading,
    open_stage,
    open_stage_async,
    save_stage,
    set_stage_up_axis,
    update_stage,
    update_stage_async,
)
from omni.isaac.core.utils.torch.rotations import euler_angles_to_quats
from omni.isaac.core.utils.transformations import get_world_pose_from_relative
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.viewports import add_aov_to_viewport, set_camera_view
from omni.isaac.core.world import World
from omni.isaac.cortex.cortex_object import CortexObject
from omni.isaac.cortex.cortex_rigid_prim import CortexRigidPrim
from omni.isaac.cortex.cortex_utils import get_assets_root_path_or_die, load_behavior_module
from omni.isaac.cortex.cortex_world import Behavior, CortexWorld, LogicalStateMonitor
from omni.isaac.cortex.df import DfNetwork, DfState, DfStateMachineDecider, DfStateSequence
from omni.isaac.cortex.dfb import DfBasicContext, DfDiagnosticsMonitor, DfRobotApiContext
from omni.isaac.cortex.robot import CortexUr10, add_franka_to_stage
from omni.isaac.cortex.sample_behaviors.franka.simple import simple_decider_network, simple_state_machine
from omni.isaac.cortex.tools import SteadyRate
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.franka import Franka, KinematicsSolver
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.franka.controllers.stacking_controller import StackingController
from omni.isaac.franka.tasks import FollowTarget, PickPlace, Stacking
from omni.isaac.kit import SimulationApp
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper, SurfaceGripper
from omni.isaac.motion_generation.articulation_kinematics_solver import ArticulationKinematicsSolver
from omni.isaac.motion_generation.articulation_motion_policy import ArticulationMotionPolicy
from omni.isaac.motion_generation.lula import RmpFlow
from omni.isaac.motion_generation.lula.kinematics import LulaKinematicsSolver
from omni.isaac.nucleus import get_assets_root_path, is_file, recursive_list_folder

# from omni.isaac.ocs2.end_effector_pose_tracking_mpc import EndEffectorPoseTrackingMpc
from omni.isaac.quadruped.robots import AnymalFlatTerrainPolicy, SpotFlatTerrainPolicy
from omni.isaac.sensor import Camera, CameraView, ContactSensor, IMUSensor, LidarRtx, RotatingLidarPhysX
from omni.isaac.sensor.effort_sensor import EffortSensor
from omni.isaac.universal_robots import KinematicsSolver
from omni.isaac.universal_robots.controllers import StackingController
from omni.isaac.universal_robots.controllers.pick_place_controller import PickPlaceController
from omni.isaac.universal_robots.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.universal_robots.tasks import BinFilling, FollowTarget, PickPlace, Stacking
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.wheeled_robots.controllers.holonomic_controller import HolonomicController
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.robots.holonomic_robot_usd_setup import HolonomicRobotUsdSetup
from omni.kit.scripting import ApplyScriptingAPICommand
from omni.kit.viewport.utility import get_active_viewport
from omni.physx import get_physx_interface, get_physx_scene_query_interface
from omni.physx.scripts import deformableUtils, physicsUtils
from omni.replicator.core import AnnotatorRegistry, Writer, random_colours
from omni.replicator.isaac.scripts.writers import PoseWriter, YCBVideoWriter
from omni.syntheticdata import sensors
from omni.syntheticdata.tests.utils import add_semantics
from PIL import Image, ImageDraw
from pxr import Gf, OmniScriptingSchema, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
from torch.utils.data import DataLoader

print("All imports from standalone examples successful")

for i in range(100):
    kit.update()
kit.close()  # Cleanup application
