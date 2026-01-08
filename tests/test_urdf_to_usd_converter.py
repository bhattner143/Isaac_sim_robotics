"""
URDF to USD Converter

Simple script to convert URDF files to USD format using Isaac Sim.
Extracts the URDF import logic from the Franka example.
"""

from isaacsim import SimulationApp

# Launch Isaac Sim
simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
})

# Import Isaac Sim modules after SimulationApp is created
from isaacsim.asset.importer.urdf import _urdf
import omni.kit.commands
import omni.usd
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

# Input URDF file
URDF_PATH = str(Path("manipulators/cart_pendulum_2dof.urdf").absolute())

# Output USD file (optional - if you want to save to a specific file)
OUTPUT_USD_PATH = str(Path("manipulators/cart_pendulum_2dof.usd").absolute())

# Import configuration
IMPORT_CONFIG = {
    "convex_decomp": False,      # Disable convex decomposition
    "fix_base": False,           # Don't fix base (allow cart to move)
    "make_default_prim": True,   # Make robot the default prim
    "self_collision": False,     # Disable self-collision
    "distance_scale": 1.0,       # Distance scale
    "density": 0.0,              # Use default density
}


# ============================================================================
# URDF TO USD CONVERSION
# ============================================================================

def convert_urdf_to_usd(urdf_path, output_usd_path=None, import_config=None):
    """
    Convert URDF file to USD format.
    
    Args:
        urdf_path: Path to input URDF file
        output_usd_path: Optional path to save USD file
        import_config: Import configuration dictionary
    
    Returns:
        prim_path: Path to the imported robot prim in the stage
    """
    print(f"\n{'='*70}")
    print("URDF to USD Conversion")
    print(f"{'='*70}")
    print(f"Input URDF: {urdf_path}")
    if output_usd_path:
        print(f"Output USD: {output_usd_path}")
    
    # Acquire URDF interface
    urdf_interface = _urdf.acquire_urdf_interface()
    
    # Create import config
    config = _urdf.ImportConfig()
    if import_config:
        config.convex_decomp = import_config.get("convex_decomp", False)
        config.fix_base = import_config.get("fix_base", False)
        config.make_default_prim = import_config.get("make_default_prim", True)
        config.self_collision = import_config.get("self_collision", False)
        config.distance_scale = import_config.get("distance_scale", 1.0)
        config.density = import_config.get("density", 0.0)
    
    print("\nImport Configuration:")
    print(f"  Convex Decomposition: {config.convex_decomp}")
    print(f"  Fix Base: {config.fix_base}")
    print(f"  Make Default Prim: {config.make_default_prim}")
    print(f"  Self Collision: {config.self_collision}")
    print(f"  Distance Scale: {config.distance_scale}")
    print(f"  Density: {config.density}")
    
    # Method 1: Parse and import in current stage
    if output_usd_path is None:
        print("\n--- Method 1: Import to Current Stage ---")
        
        # Step 1: Parse URDF file
        print("Parsing URDF file...")
        result, robot_model = omni.kit.commands.execute(
            "URDFParseFile",
            urdf_path=urdf_path,
            import_config=config
        )
        
        if not result:
            print("ERROR: Failed to parse URDF file")
            return None
        
        print(f"✓ URDF parsed successfully")
        print(f"  Robot has {len(robot_model.joints)} joints")
        
        # Step 2: Import robot to stage
        print("Importing robot to stage...")
        result, prim_path = omni.kit.commands.execute(
            "URDFImportRobot",
            urdf_robot=robot_model,
            import_config=config,
        )
        
        if not result:
            print("ERROR: Failed to import robot")
            return None
        
        print(f"✓ Robot imported to stage at: {prim_path}")
        return prim_path
    
    # Method 2: Parse and import to new USD file
    else:
        print("\n--- Method 2: Import to USD File ---")
        
        print("Parsing and importing URDF to USD file...")
        result, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=config,
            dest_path=output_usd_path
        )
        
        if not result:
            print("ERROR: Failed to parse and import URDF")
            return None
        
        print(f"✓ URDF converted to USD file: {output_usd_path}")
        print(f"✓ Robot prim path: {prim_path}")
        
        # Optionally reference the USD in current stage
        stage = omni.usd.get_context().get_stage()
        prim_path_in_stage = omni.usd.get_stage_next_free_path(
            stage, f"/World{prim_path}", False
        )
        robot_prim = stage.OverridePrim(prim_path_in_stage)
        robot_prim.GetReferences().AddReference(output_usd_path)
        
        print(f"✓ Referenced in current stage at: {prim_path_in_stage}")
        return prim_path_in_stage


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution."""
    try:
        # Convert URDF to USD
        prim_path = convert_urdf_to_usd(
            urdf_path=URDF_PATH,
            output_usd_path=OUTPUT_USD_PATH,  # Set to None to skip file saving
            import_config=IMPORT_CONFIG
        )
        
        if prim_path:
            print(f"\n{'='*70}")
            print("Conversion Successful!")
            print(f"{'='*70}")
            print(f"Robot available at: {prim_path}")
            print("\nKeeping window open for inspection...")
            print("Press Ctrl+C or close window to exit")
            print(f"{'='*70}\n")
            
            # Keep window open
            while simulation_app.is_running():
                simulation_app.update()
        else:
            print("\n{'='*70}")
            print("Conversion Failed!")
            print(f"{'='*70}\n")
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        simulation_app.close()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
