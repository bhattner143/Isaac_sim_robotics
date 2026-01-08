"""
Convert STEP file to formats compatible with Isaac Sim

Conversion paths:
1. STEP → STL → USD (best for Isaac Sim)
2. STEP → OBJ → USD
3. STEP → direct USD (if pythonocc available)

Recommended: Use external tools like FreeCAD, Blender, or CAD Assistant
"""

import sys
from pathlib import Path


def convert_step_to_stl_freecad(step_file, output_stl):
    """
    Convert STEP to STL using FreeCAD (if installed)
    
    Install FreeCAD: sudo apt install freecad
    Or download from: https://www.freecad.org/
    """
    try:
        import FreeCAD
        import Import
        import Mesh
        
        print(f"Converting {step_file} → {output_stl} using FreeCAD...")
        
        # Import STEP
        doc = FreeCAD.newDocument("StepConversion")
        Import.insert(str(step_file), doc.Name)
        
        # Export to STL
        objects = [obj for obj in doc.Objects]
        Mesh.export(objects, str(output_stl))
        
        FreeCAD.closeDocument(doc.Name)
        
        print(f"✓ Converted to STL: {output_stl}")
        return True
        
    except ImportError:
        print("ERROR: FreeCAD not installed")
        print("Install with: sudo apt install freecad")
        print("Or: conda install -c conda-forge freecad")
        return False
    except Exception as e:
        print(f"ERROR: Conversion failed: {e}")
        return False


def convert_step_to_obj_pythonocc(step_file, output_obj):
    """
    Convert STEP to OBJ using pythonocc-core
    
    Install: conda install -c conda-forge pythonocc-core
    """
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Extend.DataExchange import write_obj_file
        
        print(f"Converting {step_file} → {output_obj} using pythonocc...")
        
        # Read STEP
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(str(step_file))
        
        if status != IFSelect_RetDone:
            print("ERROR: Failed to read STEP file")
            return False
        
        step_reader.TransferRoots()
        shape = step_reader.OneShape()
        
        # Write OBJ
        write_obj_file(shape, str(output_obj))
        
        print(f"✓ Converted to OBJ: {output_obj}")
        return True
        
    except ImportError:
        print("ERROR: pythonocc-core not installed")
        print("Install with: conda install -c conda-forge pythonocc-core")
        return False
    except Exception as e:
        print(f"ERROR: Conversion failed: {e}")
        return False


def import_to_isaac_sim(file_path, isaac_prim_path="/World/ImportedModel"):
    """
    Import a file into Isaac Sim (must be STL, OBJ, USD, etc.)
    This creates a standalone script that imports the file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return
    
    # Determine file type
    ext = file_path.suffix.lower()
    
    if ext not in ['.stl', '.obj', '.usd', '.usda', '.usdc', '.fbx', '.gltf', '.glb']:
        print(f"ERROR: Unsupported format: {ext}")
        print("Supported: .stl, .obj, .usd, .usda, .usdc, .fbx, .gltf, .glb")
        return
    
    # Create Isaac Sim import script
    script_name = f"import_{file_path.stem}_to_isaac.py"
    
    script_content = f'''"""
Import {file_path.name} into Isaac Sim
Auto-generated script
"""

from isaacsim import SimulationApp

# Launch Isaac Sim
simulation_app = SimulationApp({{
    "headless": False,
    "width": 1280,
    "height": 720,
}})

from omni.isaac.core import World
from pxr import UsdGeom, Gf
import omni.usd
import omni.kit.commands

def main():
    # Create world
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    print("=" * 70)
    print("Importing {file_path.name} into Isaac Sim")
    print("=" * 70)
    
    # Import the file
    stage = omni.usd.get_context().get_stage()
    
    # Create parent xform
    parent_prim = UsdGeom.Xform.Define(stage, "{isaac_prim_path}")
    
    # Import using asset converter
    print("Loading file: {file_path.absolute()}")
    
    success, error = omni.kit.commands.execute(
        "CreateReferenceCommand",
        usd_context=omni.usd.get_context(),
        path_to="{isaac_prim_path}/mesh",
        asset_path=str("{file_path.absolute()}"),
        instanceable=False
    )
    
    if success:
        print("✓ File imported successfully")
        print(f"✓ Prim path: {isaac_prim_path}")
    else:
        print(f"ERROR: Import failed - {{error}}")
        return
    
    # Reset and run
    world.reset()
    
    print("\\nPress Ctrl+C to exit...")
    
    # Keep running
    while simulation_app.is_running():
        world.step(render=True)
    
    simulation_app.close()

if __name__ == "__main__":
    main()
'''
    
    with open(script_name, 'w') as f:
        f.write(script_content)
    
    print(f"\n✓ Created Isaac Sim import script: {script_name}")
    print(f"\nTo run:")
    print(f"  conda activate env_isaacsim")
    print(f"  python {script_name}")


def show_conversion_guide():
    """Show step-by-step guide for converting STEP to Isaac Sim compatible format"""
    
    print("=" * 70)
    print("STEP → Isaac Sim Conversion Guide")
    print("=" * 70)
    
    print("\n📋 METHOD 1: Using FreeCAD (Recommended)")
    print("-" * 70)
    print("1. Install FreeCAD:")
    print("   sudo apt install freecad")
    print("\n2. Open FreeCAD")
    print("3. File → Open → Select your .step file")
    print("4. File → Export → Choose format:")
    print("   - STL (best for simple geometry)")
    print("   - OBJ (with textures)")
    print("   - GLTF (for animations)")
    print("5. Import into Isaac Sim using the script generated above")
    
    print("\n📋 METHOD 2: Using Blender")
    print("-" * 70)
    print("1. Install Blender: https://www.blender.org/")
    print("2. Install CAD Sketcher addon for STEP import")
    print("3. Import STEP, export as OBJ or GLTF")
    print("4. Import into Isaac Sim")
    
    print("\n📋 METHOD 3: Using CAD Assistant (Free)")
    print("-" * 70)
    print("1. Download: https://www.opencascade.com/products/cad-assistant/")
    print("2. Open STEP file")
    print("3. Export as STL or OBJ")
    print("4. Import into Isaac Sim")
    
    print("\n📋 METHOD 4: Using Python (this script)")
    print("-" * 70)
    print("Install dependencies:")
    print("  conda install -c conda-forge freecad")
    print("  OR")
    print("  conda install -c conda-forge pythonocc-core")
    print("\nThen run:")
    print("  python convert_step_to_isaac.py yourfile.step --convert")
    
    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 2:
        show_conversion_guide()
        print("\nUsage:")
        print("  python convert_step_to_isaac.py <file.step> --convert    # Convert to STL/OBJ")
        print("  python convert_step_to_isaac.py <file.stl>  --import     # Create Isaac import script")
        return
    
    input_file = Path(sys.argv[1])
    
    if not input_file.exists():
        print(f"ERROR: File not found: {input_file}")
        return
    
    if "--convert" in sys.argv or "-c" in sys.argv:
        # Try conversion
        output_stl = input_file.with_suffix('.stl')
        output_obj = input_file.with_suffix('.obj')
        
        print("Attempting conversions...\n")
        
        # Try FreeCAD first
        if convert_step_to_stl_freecad(input_file, output_stl):
            print(f"\nNext step: Import STL to Isaac Sim")
            import_to_isaac_sim(output_stl)
        # Try pythonocc
        elif convert_step_to_obj_pythonocc(input_file, output_obj):
            print(f"\nNext step: Import OBJ to Isaac Sim")
            import_to_isaac_sim(output_obj)
        else:
            print("\n" + "=" * 70)
            print("Automatic conversion failed - use manual methods:")
            print("=" * 70)
            show_conversion_guide()
    
    elif "--import" in sys.argv or "-i" in sys.argv:
        # Create import script for already-converted file
        import_to_isaac_sim(input_file)
    
    else:
        # Default: show guide
        print(f"File: {input_file}")
        print(f"Extension: {input_file.suffix}\n")
        show_conversion_guide()


if __name__ == "__main__":
    main()
