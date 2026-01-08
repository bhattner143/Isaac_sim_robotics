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


"""Convert ShapeNetCore V2 to USD without materials.
By only converting the ShapeNet geometry, we can more quickly load assets into scenes for the purpose of creating
large datasets or for online training of Deep Learning models.
"""


import argparse
import os

from isaacsim import SimulationApp

if "SHAPENET_LOCAL_DIR" not in os.environ:
    import carb

    carb.log_error("SHAPENET_LOCAL_DIR not defined:")
    carb.log_error(
        "Please specify the SHAPENET_LOCAL_DIR environment variable to the location of your local shapenet database, exiting"
    )
    exit()

kit = SimulationApp()

from isaacsim.core.utils.extensions import enable_extension

enable_extension("omni.kit.asset_converter")

from shapenet_utils import shapenet_convert

parser = argparse.ArgumentParser("Convert ShapeNet assets to USD")
parser.add_argument(
    "--categories", type=str, nargs="+", default=None, help="List of ShapeNet categories to convert (space seperated)."
)
parser.add_argument(
    "--max_models", type=int, default=50, help="If specified, convert up to `max_models` per category, default is 50"
)
parser.add_argument(
    "--load_materials", action="store_true", help="If specified, materials will be loaded from shapenet meshes"
)
args, unknown_args = parser.parse_known_args()

# Ensure Omniverse Kit is launched via SimulationApp before shapenet_convert() is called
shapenet_convert(args.categories, args.max_models, args.load_materials)
# cleanup
kit.close()
