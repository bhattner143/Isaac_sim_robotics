"""
Simple plate viewer in Isaac Sim - Most stable version
Just displays the plate model without complex conversions
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
import omni.usd
from pathlib import Path

def main():
    world = World()
    world.scene.add_default_ground_plane()
    
    print("=" * 70)
    print("Loading Plate Model")
    print("=" * 70)
    
    # Use OBJ file (simpler than FBX)
    model_file = Path(__file__).parent / "plate" / "plate.obj"
    
    if not model_file.exists():
        print(f"ERROR: File not found: {model_file}")
        simulation_app.close()
        return
    
    print(f"File: {model_file}")
    
    # Simple import using CreateReference
    import omni.kit.commands
    
    success = omni.kit.commands.execute(
        'CreateReferenceCommand',
        usd_context=omni.usd.get_context(),
        path_to='/World/Plate',
        asset_path=str(model_file.absolute())
    )
    
    if success:
        print("✓ Model loaded")
    else:
        print("⚠ Import may have failed")
    
    print("\n" + "=" * 70)
    print("Running - Press Ctrl+C to exit")
    print("=" * 70)
    
    world.reset()
    
    try:
        while simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        pass
    
    simulation_app.close()

if __name__ == "__main__":
    main()
