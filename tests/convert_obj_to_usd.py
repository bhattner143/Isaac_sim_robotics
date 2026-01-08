"""
Convert OBJ file to USD format using Isaac Sim's asset converter.
This script converts part_dips_coarse_rot.obj to part_dips_coarse_rot.usd
"""

from isaacsim import SimulationApp

# Launch Isaac Sim (headless mode for conversion)
simulation_app = SimulationApp({
    "headless": True,
})

import omni.kit.asset_converter
from pathlib import Path
import asyncio

async def convert_obj_to_usd():
    """Convert OBJ to USD"""
    
    # File paths
    obj_file = str(Path("plate_dips/part_dips_coarse_rot.obj").absolute())
    usd_file = str(Path("plate_dips/part_dips_coarse_rot.usd").absolute())
    
    print(f"Converting: {obj_file}")
    print(f"To: {usd_file}")
    
    # Get converter instance
    converter = omni.kit.asset_converter.get_instance()
    
    # Create converter context with settings
    context = omni.kit.asset_converter.AssetConverterContext()
    context.ignore_materials = False  # Keep materials from MTL file
    context.ignore_animations = True
    context.ignore_cameras = True
    context.ignore_lights = True
    context.single_mesh = False
    context.smooth_normals = True
    context.export_preview_surface = True
    context.use_meter_as_world_unit = True
    context.create_world_as_default_root_prim = False
    
    # Create conversion task
    task = converter.create_converter_task(obj_file, usd_file, None, context)
    
    # Wait for conversion to complete
    success = await task.wait_until_finished()
    
    if not success:
        print(f"❌ Conversion failed!")
        detail = task.get_status()
        print(f"Details: {detail}")
    else:
        print(f"✅ Conversion successful!")
        print(f"Created: {usd_file}")
    
    return success

# Run the conversion
loop = asyncio.get_event_loop()
success = loop.run_until_complete(convert_obj_to_usd())

# Close Isaac Sim
simulation_app.close()

if success:
    print("\n✅ OBJ to USD conversion complete!")
else:
    print("\n❌ Conversion failed!")
