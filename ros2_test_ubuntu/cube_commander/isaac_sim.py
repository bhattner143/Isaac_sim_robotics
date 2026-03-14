"""
Isaac Sim Cube Listener — conda env_isaacsim
=============================================
Isaac Sim script that:
1. Creates a blue cube at the world origin
2. Reads target positions from stdin pipe (forwarded from ROS 2 listener)
3. Moves the cube to each received position in real-time

Controlled by PyDrake via ROS 2:
    Drake → ROS 2 /cube_target_pos → ros2_cube_listener_node.py → stdin → this script

Message format from stdin:
    CUBE_POS:x,y,z    e.g. CUBE_POS:0.0100,0.0000,0.5000

Usage:
    bash ros2_test_ubuntu/launch/run_cube_isaac.sh

DO NOT run this directly — use run_cube_isaac.sh which sets up the stdin pipe.
"""

# ============================================================================
# MUST BE FIRST — Isaac Sim requirement
# ============================================================================
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width":    1280,
    "height":   720,
    "window_title": "Drake → ROS 2 → Isaac Sim: Cube Commander",
})

# ============================================================================
# IMPORTS (after SimulationApp)
# ============================================================================
import sys
import threading
import queue
import numpy as np

import omni
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from pxr import UsdGeom


# ============================================================================
# STDIN READER: reads Drake positions from pipe in background thread
# ============================================================================
class StdinPositionReader:
    """
    Reads 'CUBE_POS:x,y,z' lines from stdin in a background thread.
    Thread-safe queue stores incoming positions for the main sim loop.
    """

    def __init__(self):
        self._queue  = queue.Queue()
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True
        )
        self._thread.start()
        print("[Isaac Sim] StdinPositionReader started — waiting for CUBE_POS messages.")

    def _read_loop(self):
        """Background thread: read positions from stdin pipe."""
        for line in sys.stdin:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if not line.startswith('CUBE_POS:'):
                print(f"[Isaac Sim] Unknown format: '{line}'", flush=True)
                continue

            try:
                coords = line.replace('CUBE_POS:', '').split(',')
                x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                self._queue.put(np.array([x, y, z]))
                print(
                    f"[Isaac Sim] Received → x={x*100:.1f}cm, "
                    f"y={y*100:.1f}cm, z={z*100:.1f}cm",
                    flush=True
                )
            except (ValueError, IndexError) as e:
                print(f"[Isaac Sim] Parse error on '{line}': {e}", flush=True)

    def get_latest_position(self):
        """
        Return the latest position from the queue, or None if empty.
        Drains queue to always get the most recent position.
        """
        latest = None
        while not self._queue.empty():
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break
        return latest


# ============================================================================
# ISAAC SIM SCENE SETUP
# ============================================================================
def setup_scene(world: World):
    """Create ground plane and the Drake-controlled cube."""
    # Set Z-up so Isaac Sim matches Drake / ROS2 convention:
    #   X = forward, Y = left, Z = up
    # Without this Isaac Sim defaults to Y-up (OpenGL/USD convention)
    # where Drake's X maps to Isaac Sim's Z, causing the observed
    # "moves in Z instead of X" behaviour.
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    world.scene.add_default_ground_plane()

    # Blue cube — will be moved by Drake commands
    # Use VisualCuboid (no physics/gravity) so set_world_pose() is respected
    # without the physics engine pulling it down every step.
    # z=0.1 = half of 0.2m cube scale → cube sits flush on the ground plane.
    cube = world.scene.add(
        VisualCuboid(
            prim_path = "/World/DrakeCube",
            name      = "drake_cube",
            position  = np.array([0.0, 0.0, 0.1]),   # z=0.1 → resting on ground
            scale     = np.array([0.2, 0.2, 0.2]),    # 20cm cube
            color     = np.array([0.2, 0.6, 1.0]),    # blue
        )
    )
    print("[Isaac Sim] Blue cube created at x=0.0, y=0.0, z=0.1 (on ground)")
    return cube


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("  Drake → ROS 2 → Isaac Sim: Cube Commander")
    print("=" * 60)
    print("  Cube starts at x=0.0m, moves +1cm per Drake command")
    print("  Waiting for CUBE_POS messages from ROS 2 listener...")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    # Start stdin reader (background thread)
    reader = StdinPositionReader()

    # Create world
    # physics_dt=1/500: 500 Hz physics substeps resolve contact/joint forces
    #   accurately at impedance control rates (200 Hz commands).
    # rendering_dt=1/60: GPU renders at 60 Hz independently of physics.
    #   Isaac Sim runs ceil(rendering_dt / physics_dt) = 8 physics steps per frame.
    world = World(
        stage_units_in_meters = 1.0,
        physics_dt            = 1 / 500,
        rendering_dt          = 1 / 60,
    )
    cube  = setup_scene(world)
    world.reset()

    GROUND_Z = 0.1  # half of 0.2m cube scale — keeps cube on ground plane
    current_position = np.array([0.0, 0.0, GROUND_Z])
    step_count       = 0

    # ── Simulation loop ───────────────────────────────────────────────────────
    while simulation_app.is_running():
        world.step(render=True)

        # Check for new position from Drake via ROS 2
        new_pos = reader.get_latest_position()
        if new_pos is not None:
            # Lock z to ground level regardless of what Drake sends
            candidate = np.array([new_pos[0], new_pos[1], GROUND_Z])
            if np.allclose(candidate, current_position):
                continue
            current_position = candidate
            step_count += 1

            # Move cube to new position
            cube.set_world_pose(
                position    = current_position,
                orientation = np.array([1.0, 0.0, 0.0, 0.0]),  # no rotation
            )

            print(
                f"[Isaac Sim] Moved cube → step #{step_count} | "
                f"x={current_position[0]*100:.1f}cm | "
                f"y={current_position[1]*100:.1f}cm | "
                f"z={current_position[2]*100:.1f}cm",
                flush=True
            )

    simulation_app.close()
    print("[Isaac Sim] Simulation closed.")


if __name__ == '__main__':
    main()
