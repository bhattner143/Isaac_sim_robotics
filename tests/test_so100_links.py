"""
Test Script: Check SO100 Link Names and Test Jaw Movement

Purpose:
1. List all link names in the SO100 robot
2. Test gripper jaw movement
3. Identify the correct end-effector link
"""

from pathlib import Path
from time import sleep

import numpy as np
from isaacsim import SimulationApp

# Launch Isaac Sim
simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 1280,
        "height": 720,
    }
)

# -- Isaac Sim imports must happen after SimulationApp is created --
from omni.isaac.core import World
from isaacsim.core.experimental.prims import Articulation
from pxr import Gf, UsdGeom
import omni.timeline
import omni.usd
import omni.kit.commands

# Configuration
ROBOT_USD_FILE = str(Path("manipulators/so101_physics.usd").absolute())
ROBOT_PATH = "/World/SO100"
ROBOT_POSITION = (0.0, 0.0, 0.5)


def list_all_prims(stage, prim_path="/World/SO100"):
    """Recursively print all prims under prim_path."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Prim not found: {prim_path}")
        return

    def recurse(node, indent=0):
        prefix = "  " * indent
        print(f"{prefix}{node.GetName()} [{node.GetTypeName()}] - Path: {node.GetPath()}")
        for child in node.GetChildren():
            recurse(child, indent + 1)

    print("\n" + "=" * 70)
    print("USD Prim Hierarchy:")
    print("=" * 70)
    recurse(prim)


def fallback_link_scan(stage, robot_path):
    """Collect link-like prim paths by walking the USD hierarchy."""
    robot_prim = stage.GetPrimAtPath(robot_path)
    links = []

    def walk(prim):
        if prim.GetTypeName() in ("Xform", ""):
            links.append(str(prim.GetPath()))
        for child in prim.GetChildren():
            walk(child)

    walk(robot_prim)
    return links


def main():
    print("=" * 70)
    print("SO100 Link Names and Jaw Movement Test")
    print("=" * 70)

    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()

    try:
        simulation_app.update()  # ensure extensions finish loading

        print("\nInitializing world...")
        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        stage = omni.usd.get_context().get_stage()
        print("✓ World initialized")

        # Remove any existing prim at ROBOT_PATH to avoid duplicates
        existing = stage.GetPrimAtPath(ROBOT_PATH)
        if existing.IsValid():
            omni.kit.commands.execute(
                "DeletePrimsCommand", paths=[ROBOT_PATH], destructive=False
            )

        print(f"\nImporting SO100 from: {ROBOT_USD_FILE}")
        usd_context = omni.usd.get_context()
        omni.kit.commands.execute(
            "CreateReferenceCommand",
            usd_context=usd_context,
            path_to=ROBOT_PATH,
            asset_path=ROBOT_USD_FILE,
            instanceable=False,
        )
        print(f"✓ Robot imported at {ROBOT_PATH}")

        prim = stage.GetPrimAtPath(ROBOT_PATH)
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*ROBOT_POSITION))
        xformable.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        print("✓ Transform applied")

        print("\nResetting world...")
        world.reset()
        print("✓ World reset")

        print("\nCreating robot articulation...")
        robot = Articulation(ROBOT_PATH)
        robot.initialize()
        print("✓ Articulation initialized")

        num_dof = robot.num_dof
        dof_names = robot.dof_names
        print("\n" + "=" * 70)
        print(f"Robot has {num_dof} DOF:")
        print("=" * 70)
        for idx, name in enumerate(dof_names):
            print(f"  DOF {idx:2d}: {name}")

        print("\n" + "=" * 70)
        print("Available Link Names:")
        print("=" * 70)
        try:
            link_names = robot.get_link_names()
            for idx, link in enumerate(link_names):
                print(f"  Link {idx:2d}: {link}")
        except AttributeError:
            print("  get_link_names() unavailable; falling back to USD tree.")
            link_paths = fallback_link_scan(stage, ROBOT_PATH)
            for idx, link in enumerate(link_paths):
                print(f"  Link {idx:2d}: {link.split('/')[-1]} (path: {link})")

        list_all_prims(stage, ROBOT_PATH)

        print("\n" + "=" * 70)
        print("Testing Link Indices for Potential End-Effectors:")
        print("=" * 70)
        candidate_links = [
            "tcp_link",
            "gripperframe",
            "gripper",
            "wrist",
            "moving_jaw_so101_v1",
            "end_effector",
        ]
        for name in candidate_links:
            try:
                idx = robot.get_link_index(name)
                if idx is not None and idx >= 0:
                    print(f"  ✓ '{name}': index = {idx}")
                else:
                    print(f"  ✗ '{name}': not found")
            except AttributeError:
                print(f"  ✗ '{name}': get_link_index() unavailable in this build")

        world.step(render=False)

        initial_positions = robot.get_dof_positions()
        print("\n" + "=" * 70)
        print("Initial DOF Positions:")
        print("=" * 70)
        for name, pos in zip(dof_names, initial_positions):
            print(f"  {name:30s}: {pos:8.4f}")

        print("\n" + "=" * 70)
        print("Testing Jaw Movement (if gripper DOFs exist)...")
        print("=" * 70)

        timeline.play()
        simulation_app.update()

        if num_dof > 6:
            last_idx = num_dof - 1
            print(f"\nAttempting to move DOF {last_idx} ('{dof_names[last_idx]}')")

            target_positions = np.array(initial_positions)
            print("Opening jaw...")
            for _ in range(60):
                target_positions[last_idx] = 0.04
                robot.set_dof_positions(target_positions)
                world.step(render=True)
                sleep(0.016)

            print("Closing jaw...")
            for _ in range(60):
                target_positions[last_idx] = 0.0
                robot.set_dof_positions(target_positions)
                world.step(render=True)
                sleep(0.016)

            print("✓ Jaw movement test complete")
        else:
            print("No gripper DOFs detected (robot has only 6 DOFs)")

        print("\n" + "=" * 70)
        print("End-Effector Positions (world coordinates):")
        print("=" * 70)
        for name in ["gripperframe", "gripper", "wrist"]:
            link_path = f"{ROBOT_PATH}/so101/{name}"
            link_prim = stage.GetPrimAtPath(link_path)
            if link_prim.IsValid():
                xf = UsdGeom.Xformable(link_prim).ComputeLocalToWorldTransform(0)
                pos = xf.ExtractTranslation()
                print(f"  {name:20s}: ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f})")
            else:
                print(f"  {name:20s}: prim not found at {link_path}")

        print("\n" + "=" * 70)
        print("Test Complete! Keep window open for inspection.")
        print("Close the viewer window to exit.")
        print("=" * 70 + "\n")

        while simulation_app.is_running():
            world.step(render=True)

    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()