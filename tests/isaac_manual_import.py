"""
Display plate model in Isaac Sim using built-in viewer
This opens Isaac Sim and you manually import the file through the UI
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World

def main():
    world = World()
    world.scene.add_default_ground_plane()
    
    print("=" * 70)
    print("Isaac Sim Ready - Import Plate Model via UI")
    print("=" * 70)
    print("\nTo import your plate model:")
    print("1. File → Import")
    print("2. Navigate to: ~/Documents/isac_sim_pydrake/plate/")
    print("3. Select: plate.fbx or plate.obj")
    print("4. Click 'Import'")
    print("\n" + "=" * 70)
    
    world.reset()
    
    try:
        while simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        pass
    
    simulation_app.close()

if __name__ == "__main__":
    main()
